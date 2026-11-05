from uuid import uuid4
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import Select


def test_normalize_native_parts_preserves_order_mentions_and_card_text():
    from nonebot.adapters.onebot.v11 import Message, MessageSegment

    from nonebot_plugin_ai_groupmate.structured_message import normalize_parts

    original = Message([
        MessageSegment.text("看\x00"), MessageSegment.at("123"),
        MessageSegment.image("https://example.test/a.png"), MessageSegment.text("这张"),
        MessageSegment.json('{"meta":{"detail":{"title":"分享标题"}}}'),
    ])
    event = SimpleNamespace(original_message=original)
    parts = normalize_parts(event, [MessageSegment.text("stripped")])
    assert [part["type"] for part in parts] == ["text", "mention", "image", "text", "card"]
    assert parts[0]["text"] == "看"
    assert parts[1]["target"] == "123"
    assert "分享标题" in parts[-1]["text"]
    assert "example.test" not in str(parts)


async def test_registered_files_bind_only_when_native_order_matches(tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import File, UniMessage
    from nonebot.adapters.onebot.v11 import Message, MessageSegment

    from nonebot_plugin_ai_groupmate.model import StructuredMessage
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound_safely
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileTaskError, FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.chat_file_inbox import ChatFileInbox, register_chat_file_resources

    session = f"structured-files-{uuid4()}"
    workspace = FileWorkspace(tmp_path, session_id=session, user_id="alice", bot_id="bot", is_private=False)
    native = Message([MessageSegment.text("文件"),
                      MessageSegment("file", {"file": "a.csv", "file_id": "native-a", "path": "", "file_size": 10}),
                      MessageSegment("file", {"file": "b.csv", "file_id": "native-b", "path": "", "file_size": 20})])
    files = list(UniMessage.of(native, adapter="OneBot V11").include(File))
    assert [file.id for file in files] == ["native-a", "native-b"]
    marker, ids = register_chat_file_resources(files, workspace, message_id="message", group_id=session)
    assert "a.csv" in marker
    assert len(ids) == 2
    assert all(ids) and ids[0] is not None
    assert ChatFileInbox(workspace).get(ids[0]).name == "a.csv"
    other = FileWorkspace(tmp_path, session_id=session, user_id="bob", bot_id="bot", is_private=False)
    with pytest.raises(FileTaskError):
        ChatFileInbox(other).get(ids[0])
    event = SimpleNamespace(original_message=native)
    async with get_session() as db:
        base = {"scope": "group", "session_id": session, "sender_id": "alice", "sender_name": "Alice",
                "reply_to_id": None, "platform": "qq", "event": event, "fallback_segments": [],
                "mention_names": {}, "file_ids": ids}
        assert await record_inbound_safely(db, platform_id="matched", file_source_ids=("native-a", "native-b"), **base)
        assert await record_inbound_safely(db, platform_id="mismatched", file_source_ids=("native-b", "native-a"), **base)
        await db.commit()
        rows = (await db.scalars(Select(StructuredMessage).where(StructuredMessage.session_id == session))).all()
        parts = {row.platform_id: row.parts for row in rows}
        assert [part.get("file_id") for part in parts["matched"] if part["type"] == "file"] == list(ids)
        assert all("file_id" not in part for part in parts["mismatched"] if part["type"] == "file")
        assert "native-a" not in str(parts)


async def test_structured_record_idempotent_and_scoped_in_real_database():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import StructuredMessage
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound

    group = f"structured-{uuid4()}"
    parts = [{"type": "text", "text": "你好"}, {"type": "mention", "target": "123"}, {"type": "image"}]
    async with get_session() as db:
        kwargs = {"session_id": group, "platform_id": "77", "sender_id": "u", "sender_name": "a\x00b",
                  "reply_to_id": "42", "parts": parts}
        assert await record_inbound(db, scope="group", **kwargs)
        assert not await record_inbound(db, scope="group", **kwargs)
        assert await record_inbound(db, scope="private", **kwargs)
        assert await record_inbound(db, scope="group", platform="another-platform", **kwargs)
        assert not await record_inbound(db, scope="group", **{**kwargs, "platform_id": "unknown"})
        await db.commit()
        rows = (await db.execute(Select(StructuredMessage).where(StructuredMessage.session_id == group))).scalars().all()
        assert len(rows) == 3
        assert {row.scope for row in rows} == {"group", "private"}
        assert all(row.parts == parts and row.reply_to_id == "42" and row.sender_name == "ab" for row in rows)


async def test_structured_failure_preserves_legacy_message(monkeypatch):
    from nonebot_plugin_orm import get_session
    from nonebot.adapters.onebot.v11 import MessageSegment

    from nonebot_plugin_ai_groupmate import structured_message as module
    from nonebot_plugin_ai_groupmate.model import ChatHistory, StructuredMessage

    group = f"structured-fallback-{uuid4()}"
    original = module.record_inbound
    async def fail_after_insert(db, **kwargs):
        await original(db, **kwargs)
        raise RuntimeError("simulated structured storage failure")
    monkeypatch.setattr(module, "record_inbound", fail_after_insert)
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="u", user_name="user", content_type="text", content="id: 77\n保留"))
        saved = await module.record_inbound_safely(
            db, event=SimpleNamespace(original_message=None), fallback_segments=[MessageSegment.text("保留")],
            mention_names={}, scope="group", session_id=group, platform_id="77", sender_id="u",
            sender_name="user", reply_to_id=None,
        )
        assert not saved
        await db.commit()
        assert await db.scalar(Select(ChatHistory.msg_id).where(ChatHistory.session_id == group)) is not None
        assert await db.scalar(Select(StructuredMessage.id).where(StructuredMessage.session_id == group)) is None


async def test_structured_image_failure_preserves_legacy_image(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import SceneType
    from nonebot_plugin_alconna.uniseg import Image

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import structured_message as module
    from nonebot_plugin_ai_groupmate.model import ChatHistory, StructuredMessage

    group = f"structured-image-fallback-{uuid4()}"
    monkeypatch.setattr(plugin.plugin_config, "message_refactor_pilot_groups", [group])
    session = SimpleNamespace(scene=SimpleNamespace(id=group, type=SceneType.GROUP), user=SimpleNamespace(id="u"))
    async def fetch(_event, _bot, _state, image):
        return image.id.encode()
    monkeypatch.setattr(plugin, "image_fetch", fetch)
    monkeypatch.setattr(plugin, "check_and_compress_image_bytes", lambda data, **kwargs: data)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    original = module.bind_image_resource
    async def fail_after_binding(db, **kwargs):
        await original(db, **kwargs)
        raise RuntimeError("simulated image association failure")
    monkeypatch.setattr(module, "bind_image_resource", fail_after_binding)
    async with get_session() as db:
        await module.record_inbound(db, scope="group", session_id=group, platform_id="message", sender_id="u",
                                    sender_name="user", reply_to_id=None, parts=[{"type": "image"}])
        await db.commit()
        await plugin.process_image_message(db, Image(id="image.jpg"), cast(Any, None), cast(Any, None), {}, cast(Any, session),
                                           "user", "id: message\n", image_index=0, image_count=1)
    async with get_session() as db:
        history = await db.scalar(Select(ChatHistory).where(ChatHistory.session_id == group))
        structured = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert history is not None
        assert history.media_id is not None
        assert structured is not None
        assert "media_id" not in structured.parts[0]


async def test_structured_context_failure_restores_database_session(monkeypatch):
    from sqlalchemy import text
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import structured_context as module

    async def broken_load(db, *_args, **_kwargs):
        await db.execute(text("SELECT * FROM intentionally_missing_structured_table"))
        return {}
    monkeypatch.setattr(module, "load_structured_context", broken_load)
    async with get_session() as db:
        result = await module.load_structured_context_safely(db, [], session_id="g", scope="group", platform="qq")
        assert result is None
        assert await db.scalar(Select(1)) == 1
        await db.commit()


def test_migration_adds_and_removes_table_without_touching_legacy():
    from importlib import import_module

    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = import_module("nonebot_plugin_ai_groupmate.migrations.e1c7a5b9d240_add_structured_messages")
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE legacy_history (id INTEGER PRIMARY KEY, content TEXT)"))
        connection.execute(text("INSERT INTO legacy_history VALUES (1, '保留')"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
            assert "nonebot_plugin_ai_groupmate_structuredmessage" in inspect(connection).get_table_names()
            migration.downgrade()
        assert connection.execute(text("SELECT content FROM legacy_history")).scalar_one() == "保留"
        assert "nonebot_plugin_ai_groupmate_structuredmessage" not in inspect(connection).get_table_names()


async def test_downloaded_images_bind_to_correct_ordered_parts(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import SceneType
    from nonebot_plugin_alconna.uniseg import Image

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage, StructuredMessage
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound, bind_image_resource

    group = f"structured-image-{uuid4()}"
    monkeypatch.setattr(plugin.plugin_config, "message_refactor_pilot_groups", [group])
    session = SimpleNamespace(scene=SimpleNamespace(id=group, type=SceneType.GROUP), user=SimpleNamespace(id="u"))
    async def fetch(_event, _bot, _state, image):
        return image.id.encode()
    monkeypatch.setattr(plugin, "image_fetch", fetch)
    monkeypatch.setattr(plugin, "check_and_compress_image_bytes", lambda data, **kwargs: data)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)

    async with get_session() as db:
        await record_inbound(db, scope="group", session_id=group, platform_id="message", sender_id="u",
                             sender_name="user", reply_to_id=None, parts=[
                                 {"type": "text", "text": "前"}, {"type": "image"},
                                 {"type": "text", "text": "中"}, {"type": "image"},
                             ])
        await db.commit()
        # Complete in reverse order, as background downloads can do.
        for index in (1, 0):
            await plugin.process_image_message(db, Image(id=f"image-{index}.jpg"), cast(Any, None), cast(Any, None), {}, cast(Any, session),
                                               "user", "id: message\n", image_index=index, image_count=2)
        row = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert row is not None
        first, second = int(row.parts[1]["media_id"]), int(row.parts[3]["media_id"])
        first_media = await db.get(MediaStorage, first)
        second_media = await db.get(MediaStorage, second)
        assert first_media is not None
        assert second_media is not None
        assert (tmp_path / first_media.file_path).read_bytes() == b"image-0.jpg"
        assert (tmp_path / second_media.file_path).read_bytes() == b"image-1.jpg"
        assert not await bind_image_resource(db, scope="group", session_id=group, platform_id="message",
                                             image_index=0, image_count=1, media_id=second)
