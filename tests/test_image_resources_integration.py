import json
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest


async def test_published_image_search_uses_scoped_publication(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent import web_image_search as search
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.reply_tools import record_image_reply
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"resource-search-{uuid4()}"
    root = tmp_path / "files"
    owner = FileWorkspace(root, session_id=group, user_id="alice", bot_id="bot", is_private=False)
    viewer = FileWorkspace(root, session_id=group, user_id="bob", bot_id="bot", is_private=False)
    other = FileWorkspace(root, session_id="other", user_id="bob", bot_id="bot", is_private=False)
    artifact = owner.save(png("red"), "draw.png")
    owner.publish_image(artifact.file_id, "sent-123")
    provider = AsyncMock(return_value=search.WebDetectionSearchResult({"visual_matches": [{"title": "candidate"}]}, False, 1, 900))
    monkeypatch.setattr(search, "perform_web_detection", provider)
    async with get_session() as db:
        record_image_reply(db, session_id=group, bot_name="bot", message_id="sent-123",
                           image_content=f"[图片文件] draw.png file_id={artifact.file_id}")
        await db.commit()
        tool = search.create_reverse_image_search_tool(db, group, None, None, reply_to_id="sent-123",
                 pic_dir=tmp_path, config=ScopedConfig(google_web_detection_enabled=True), workspace=viewer)
        response = json.loads(await tool.ainvoke({}))
        assert response["status"] == "succeeded"
        assert provider.await_count == 1
        assert provider.await_args is not None
        assert provider.await_args.args[2] == png("red")
        bad = search.create_reverse_image_search_tool(db, group, None, None, reply_to_id="sent-123",
              pic_dir=tmp_path, config=ScopedConfig(google_web_detection_enabled=True), workspace=other)
        result = json.loads(await bad.ainvoke({}))
        assert result["reason_code"] == "source_file_not_found"
        assert provider.await_count == 1


async def test_structured_only_bot_image_is_shared_by_drawing_and_search(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config

    from nonebot_plugin_ai_groupmate.agent import web_image_search as search, image_generation_tools as drawing
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    group = f"structured-images-{uuid4()}"
    root = tmp_path / "files"
    owner = FileWorkspace(root, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    viewer = FileWorkspace(root, session_id=group, user_id="bob", bot_id="bot", is_private=False, peer_bot_ids=())
    artifact = owner.save(png("red"), "draw.png")
    owner.publish_image(artifact.file_id, "sent")
    provider = AsyncMock(return_value=GeneratedImage(png("blue"), {}))
    search_provider = AsyncMock(return_value=search.WebDetectionSearchResult({"visual_matches": [{"title": "candidate"}]}, False, 1, 900))
    monkeypatch.setattr(drawing, "request_image", provider)
    monkeypatch.setattr(search, "perform_web_detection", search_provider)
    monkeypatch.setattr(drawing, "load_replied_message_histories", AsyncMock(side_effect=AssertionError("legacy lookup used")))
    monkeypatch.setattr(search, "_find_source_media", AsyncMock(side_effect=AssertionError("legacy lookup used")))
    async with get_session() as db:
        await record_inbound(db, platform="qq", scope="group", session_id=group, platform_id="sent",
                             sender_id="bot", sender_name="miyuki", sender_kind="bot", reply_to_id=None,
                             parts=[{"type": "text", "text": "图"}, {"type": "image", "file_id": artifact.file_id}])
        await db.commit()
        generate = drawing.create_image_generation_tool(db, group, None, config=image_config(message_refactor_pilot_groups=[group]),
                    workspace=viewer, pic_dir=tmp_path, current_message_id="question", replied_message_id="sent",
                    platform="qq", is_private=False)
        edited = json.loads(await generate.ainvoke({"prompt": "改成蓝色", "reference_message_ids": ["replied"]}))
        assert edited["reason_code"] == "image_generated"
        assert provider.await_args.args[2][0][0] == png("red")
        reverse = search.create_reverse_image_search_tool(db, group, None, None, reply_to_id="sent", pic_dir=tmp_path,
                    config=ScopedConfig(google_web_detection_enabled=True, message_refactor_pilot_groups=[group]),
                    workspace=viewer, platform="qq", is_private=False)
        found = json.loads(await reverse.ainvoke({}))
        assert found["status"] == "succeeded"
        assert found["data"]["source_platform_message_ids"] == ["sent"]
        assert search_provider.await_args.args[2] == png("red")


async def test_structured_image_lookup_rejects_other_scope_and_unbound_image(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as drawing
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"structured-scope-{uuid4()}"
    workspace = FileWorkspace(tmp_path, session_id=group, user_id="bob", bot_id="bot", is_private=False, peer_bot_ids=())
    provider = AsyncMock()
    monkeypatch.setattr(drawing, "request_image", provider)
    monkeypatch.setattr(drawing, "load_replied_message_histories", AsyncMock(side_effect=AssertionError("legacy lookup used")))
    async with get_session() as db:
        base = {"platform": "qq", "session_id": group, "sender_id": "alice", "sender_name": "Alice", "reply_to_id": None,
                "parts": [{"type": "image"}]}
        await record_inbound(db, scope="private", platform_id="private-id", **base)
        await record_inbound(db, scope="group", platform_id="unbound-id", **base)
        await db.commit()
        tool = drawing.create_image_generation_tool(db, group, None, config=image_config(message_refactor_pilot_groups=[group]),
                    workspace=workspace, pic_dir=tmp_path, current_message_id="current", replied_message_id=None,
                    platform="qq", is_private=False)
        for message_id in ("private-id", "unbound-id"):
            result = json.loads(await tool.ainvoke({"prompt": "edit", "reference_message_ids": [message_id]}))
            assert result["reason_code"] == "reference_not_found"
        provider.assert_not_awaited()


async def test_structured_user_images_keep_original_order_without_legacy_rows(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as drawing
    from nonebot_plugin_ai_groupmate.model import MediaStorage
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    group = f"structured-user-images-{uuid4()}"
    workspace = FileWorkspace(tmp_path / "files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "red.png").write_bytes(png("red"))
    (pic_dir / "blue.png").write_bytes(png("blue"))
    provider = AsyncMock(return_value=GeneratedImage(png("red"), {}))
    monkeypatch.setattr(drawing, "request_image", provider)
    monkeypatch.setattr(drawing, "load_replied_message_histories", AsyncMock(side_effect=AssertionError("legacy lookup used")))
    async with get_session() as db:
        red = MediaStorage(file_hash=f"red-{uuid4()}", file_path="red.png", description="")
        blue = MediaStorage(file_hash=f"blue-{uuid4()}", file_path="blue.png", description="")
        db.add_all([red, blue])
        await db.flush()
        await record_inbound(db, platform="qq", scope="group", session_id=group, platform_id="two-images",
                             sender_id="alice", sender_name="Alice", reply_to_id=None,
                             parts=[{"type": "image", "media_id": str(red.media_id)}, {"type": "text", "text": "然后"},
                                    {"type": "image", "media_id": str(blue.media_id)}])
        await db.commit()
        tool = drawing.create_image_generation_tool(db, group, None, config=image_config(message_refactor_pilot_groups=[group]),
                    workspace=workspace, pic_dir=pic_dir, current_message_id="two-images", replied_message_id=None,
                    platform="qq", is_private=False)
        result = json.loads(await tool.ainvoke({"prompt": "合起来", "reference_message_ids": ["current"]}))
        assert result["reason_code"] == "image_generated"
        assert [item[0] for item in provider.await_args.args[2]] == [png("red"), png("blue")]


async def test_pilot_reads_old_image_when_no_structured_record_exists(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as drawing
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    group = f"legacy-image-{uuid4()}"
    workspace = FileWorkspace(tmp_path / "files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "old.png").write_bytes(png("red"))
    provider = AsyncMock(return_value=GeneratedImage(png("blue"), {}))
    monkeypatch.setattr(drawing, "request_image", provider)
    async with get_session() as db:
        media = MediaStorage(file_hash=f"legacy-{uuid4()}", file_path="old.png", description="")
        db.add(media)
        await db.flush()
        db.add(ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="image",
                           content="id: old-id\n图片文件: old.png", media_id=media.media_id))
        await db.commit()
        tool = drawing.create_image_generation_tool(db, group, None, config=image_config(message_refactor_pilot_groups=[group]),
                    workspace=workspace, pic_dir=pic_dir, current_message_id="current", replied_message_id="old-id",
                    platform="qq", is_private=False)
        result = json.loads(await tool.ainvoke({"prompt": "edit", "reference_message_ids": ["replied"]}))
        assert result["reason_code"] == "image_generated"
        assert provider.await_args.args[2][0][0] == png("red")


async def test_structured_image_database_failure_does_not_fall_back_to_unscoped_history(monkeypatch, tmp_path):
    from sqlalchemy import event
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate.agent import web_image_search as search, image_generation_tools as drawing
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"structured-failure-{uuid4()}"
    workspace = FileWorkspace(tmp_path, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    monkeypatch.setattr(drawing, "load_replied_message_histories", AsyncMock(side_effect=AssertionError("legacy lookup used")))
    monkeypatch.setattr(search, "_find_source_media", AsyncMock(side_effect=AssertionError("legacy lookup used")))
    async with get_session() as db:
        engine = db.get_bind()
        def fail_structured(_conn, _cursor, statement, _parameters, _context, _executemany):
            if "structuredmessage" in statement.lower():
                raise RuntimeError("simulated database failure")
        event.listen(engine, "before_cursor_execute", fail_structured)
        try:
            generate = drawing.create_image_generation_tool(db, group, None, config=image_config(message_refactor_pilot_groups=[group]),
                        workspace=workspace, pic_dir=tmp_path, current_message_id="current", replied_message_id="sent",
                        platform="qq", is_private=False)
            result = json.loads(await generate.ainvoke({"prompt": "edit", "reference_message_ids": ["replied"]}))
            assert result["reason_code"] == "reference_unavailable"
            reverse = search.create_reverse_image_search_tool(db, group, None, None, reply_to_id="sent", pic_dir=tmp_path,
                        config=ScopedConfig(google_web_detection_enabled=True, message_refactor_pilot_groups=[group]),
                        workspace=workspace, platform="qq", is_private=False)
            result = json.loads(await reverse.ainvoke({}))
            assert result["reason_code"] == "source_image_unavailable"
        finally:
            event.remove(engine, "before_cursor_execute", fail_structured)


async def test_expired_reference_is_distinguished_from_unknown_and_cross_scope(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from test_image_generation import png, tool_fixture

    from nonebot_plugin_ai_groupmate.agent import file_workspace as storage
    from nonebot_plugin_ai_groupmate.agent.image_resources import ImageResources, ImageResourceError
    from nonebot_plugin_ai_groupmate.agent.web_image_search import _source_resource_error

    _, owner, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path)
    artifact = owner.save(png(), "draw.png")
    owner.publish_image(artifact.file_id, "sent-123")
    viewer = storage.FileWorkspace(tmp_path / "files", session_id="group", user_id="bob", bot_id="bot", is_private=False)
    other = storage.FileWorkspace(tmp_path / "files", session_id="other", user_id="bob", bot_id="bot", is_private=False)
    row = SimpleNamespace(content_type="bot", media_id=None,
                          content=f"id: sent-123\n[图片文件] draw.png file_id={artifact.file_id}")
    monkeypatch.setattr(storage.time, "time", lambda: artifact.created_at + storage.FILE_TTL_SECONDS + 1)
    with pytest.raises(ImageResourceError) as expired:
        ImageResources(tmp_path, viewer).read_history(row)
    assert expired.value.code == "expired"
    assert _source_resource_error(expired.value).reason_code == "source_image_expired"
    with pytest.raises(ImageResourceError) as inaccessible:
        ImageResources(tmp_path, other).read_history(row)
    assert inaccessible.value.code == "not_found"
    result = json.loads(await generate.ainvoke({"prompt": "edit", "reference_file_ids": [artifact.file_id]}))
    assert result["reason_code"] == "reference_expired"
    provider.assert_not_awaited()
    storage.cleanup_workspace(tmp_path / "files")
    with pytest.raises(ImageResourceError) as missing:
        ImageResources(tmp_path, viewer).read_history(row)
    assert missing.value.code == "not_found"


def test_vision_and_drawing_reject_escaped_media_paths(tmp_path):
    from test_history_metadata import _history
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent.history_format import get_history_image_data_uri
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError
    from nonebot_plugin_ai_groupmate.agent.image_generation_tools import _read_reference_path

    (tmp_path / "secret.png").write_bytes(png())
    pic_dir = tmp_path / "pictures"
    pic_dir.mkdir()
    row = _history(1, "image", "id: 1\n../secret.png")
    assert get_history_image_data_uri(row, pic_dir=pic_dir) is None
    with pytest.raises(ImageGenerationError, match="参考图原文件"):
        _read_reference_path(pic_dir, "../secret.png")
