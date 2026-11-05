from uuid import uuid4
from typing import Any, cast

from sqlalchemy import Select


async def test_structured_context_preserves_interleaved_images_and_sender_scope(tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage, ChatHistorySchema
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import load_structured_context

    group = f"structured-context-{uuid4()}"
    (tmp_path / "one.png").write_bytes(png("red"))
    (tmp_path / "two.png").write_bytes(png("blue"))
    async with get_session() as db:
        a = MediaStorage(file_hash=uuid4().hex * 2, file_path="one.png", description="first")
        b = MediaStorage(file_hash=uuid4().hex * 2, file_path="two.png", description="second")
        db.add_all([a, b])
        await db.flush()
        await record_inbound(db, platform="test", scope="group", session_id=group, platform_id="101",
            sender_id="alice", sender_name="Alice", reply_to_id="50", parts=[
                {"type": "text", "text": "看"}, {"type": "image", "media_id": str(a.media_id)},
                {"type": "mention", "target": "123", "name": "小明"}, {"type": "text", "text": "还有"},
                {"type": "image", "media_id": str(b.media_id)}, {"type": "text", "text": "结束"},
            ])
        db.add_all([
            ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="text", content="id: 101\n回复id: 50\nlegacy text"),
            ChatHistory(session_id=group, user_id="bob", user_name="Bob", content_type="text", content="id: 102\n另一条"),
            ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="image", content="id: 101\none.png", media_id=a.media_id),
            ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="image", content="id: 101\ntwo.png", media_id=b.media_id),
            ChatHistory(session_id=group, user_id="imposter", user_name="Eve", content_type="text", content="id: 101\n其他用户"),
        ])
        await db.commit()
        result = await db.execute(Select(ChatHistory).where(ChatHistory.session_id == group).order_by(ChatHistory.msg_id))
        history = [ChatHistorySchema.model_validate(row) for row in result.scalars().all()]
        snapshots = await load_structured_context(db, history, session_id=group, scope="group", platform="test")
        assert await load_structured_context(db, history, session_id=group, scope="private", platform="test") == {}
    messages = format_chat_history(history, pic_dir=tmp_path, bot_name="bot", structured_messages=snapshots)
    assert len(messages) == 3
    blocks = messages[0].content
    assert isinstance(blocks, list)
    blocks = cast(list[dict[str, Any]], blocks)
    assert [part["type"] for part in blocks] == ["text", "image_url", "text", "image_url", "text"]
    assert "看" in blocks[0]["text"]
    assert "@小明（user_id=123）" in blocks[2]["text"]
    assert "还有" in blocks[2]["text"]
    assert blocks[4]["text"] == "结束"
    assert "另一条" in messages[1].content
    assert "其他用户" in messages[2].content
    # The append-only thread may already have the text while an image arrives later.
    later = format_chat_history([history[2]], pic_dir=tmp_path, bot_name="bot", structured_messages=snapshots)
    assert "@123" not in str(later[0].content)
    assert "还有" not in str(later[0].content)


def test_structured_format_falls_back_for_unrecognized_messages(tmp_path):
    from test_history_metadata import _history

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    row = _history(1, "text", "id: old\n旧记录")
    result = format_chat_history([row], pic_dir=tmp_path, bot_name="bot", structured_messages={})
    assert "旧记录" in result[0].content
    snapshot = StructuredSnapshot(platform_id="old", sender_id=row.user_id, reply_to_id=None,
                                  parts=({"type": "file"},), media_paths={})
    result = format_chat_history([row], pic_dir=tmp_path, bot_name="bot", structured_messages={("old", row.user_id): snapshot})
    assert "旧记录" in result[0].content


async def test_structured_bot_text_keeps_assistant_role_and_native_mention(tmp_path):
    from nonebot_plugin_orm import get_session
    from langchain_core.messages import AIMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import load_structured_context

    group = f"structured-bot-{uuid4()}"
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="miyuki", user_name="miyuki", content_type="bot", content="id: sent\n旧文本"))
        await record_inbound(db, platform="qq", scope="group", session_id=group, platform_id="sent", sender_id="bot-id",
                             sender_name="miyuki", sender_kind="bot", bot_id="bot-id", reply_to_id=None,
                             parts=[{"type": "text", "text": "你好 "}, {"type": "mention", "target": "123"}])
        await db.commit()
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        history = [ChatHistorySchema.model_validate(row) for row in rows]
        snapshots = await load_structured_context(db, history, session_id=group, scope="group", platform="qq")
    messages = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki", structured_messages=snapshots)
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert str(messages[0].content).strip() == "你好 @123"


def test_structured_bot_image_without_workspace_does_not_leak_bytes(tmp_path):
    from test_history_metadata import _history

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    row = _history(1, "bot", "id: sent\n[图片文件] image.png file_id=" + "a" * 32)
    snapshot = StructuredSnapshot(platform_id="sent", sender_id="bot-id", reply_to_id=None,
                                  parts=({"type": "image", "file_id": "a" * 32},), media_paths={})
    new = format_chat_history([row], pic_dir=tmp_path, bot_name="miyuki", structured_messages={("sent", "miyuki"): snapshot})
    assert len(new) == 1
    assert "图片已过期/无法加载" in str(new[0].content)


def test_structured_bot_mixed_images_preserve_order_and_scoped_visibility(tmp_path):
    import base64
    from io import BytesIO

    from PIL import Image
    from test_history_metadata import _history
    from test_image_generation import png
    from langchain_core.messages import AIMessage, HumanMessage

    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    root = tmp_path / "files"
    owner = FileWorkspace(root, session_id="group-1", user_id="alice", bot_id="bot-a", is_private=False, peer_bot_ids=())
    red = owner.save(png("red"), "red.png")
    blue = owner.save(png("blue"), "blue.png")
    owner.publish_image(red.file_id, "sent")
    owner.publish_image(blue.file_id, "sent")
    viewer = FileWorkspace(root, session_id="group-1", user_id="bob", bot_id="bot-b", is_private=False, peer_bot_ids=())
    other = FileWorkspace(root, session_id="another-group", user_id="bob", bot_id="bot-b", is_private=False, peer_bot_ids=())
    history = [
        _history(1, "bot", "id: sent\n回复id: quoted\n旧文本"),
        _history(2, "bot", f"id: sent\n[图片文件] red.png file_id={red.file_id}"),
        _history(3, "bot", f"id: sent\n[图片文件] blue.png file_id={blue.file_id}"),
    ]
    snapshot = StructuredSnapshot(platform_id="sent", sender_id="bot-a", reply_to_id="quoted", parts=(
        {"type": "text", "text": "前"}, {"type": "image", "file_id": red.file_id},
        {"type": "text", "text": "中"}, {"type": "image", "file_id": blue.file_id},
        {"type": "text", "text": "后"},
    ), media_paths={})
    kwargs = {"pic_dir": tmp_path, "bot_name": "miyuki", "structured_messages": {("sent", "miyuki"): snapshot}}
    messages = format_chat_history(history, workspace=viewer, **kwargs)
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert "(回复了一条消息) 前[图片]中[图片]后" == messages[0].content
    assert isinstance(messages[1], HumanMessage)
    blocks = cast(list[dict[str, Any]], messages[1].content)
    urls = [part["image_url"]["url"] for part in blocks if part["type"] == "image_url"]
    assert len(urls) == 2
    pixels = [Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((0, 0)) for url in urls]
    assert pixels[0][0] > 240
    assert pixels[0][2] < 10
    assert pixels[1][2] > 240
    assert pixels[1][0] < 10
    inaccessible = format_chat_history(history, workspace=other, **kwargs)
    assert len(inaccessible) == 1
    assert str(inaccessible[0].content).count("图片已过期/无法加载") == 2
    no_images = format_chat_history(history, workspace=viewer, max_inline_images=0, **kwargs)
    assert len(no_images) == 1
    assert "前[图片]中[图片]后" in str(no_images[0].content)


def test_structured_bot_image_requires_matching_history_and_avoids_text_replay(tmp_path):
    from test_history_metadata import _history

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    image = _history(2, "bot", "id: sent\n[图片文件] image.png file_id=" + "a" * 32)
    snapshot = StructuredSnapshot(platform_id="sent", sender_id="bot-id", reply_to_id=None,
                                  parts=({"type": "text", "text": "旧配文"}, {"type": "image", "file_id": "a" * 32}), media_paths={})
    options = {"pic_dir": tmp_path, "bot_name": "miyuki", "structured_messages": {("sent", "miyuki"): snapshot}}
    messages = format_chat_history([image], **options)
    assert "旧配文" not in str(messages[0].content)
    mismatched = StructuredSnapshot(platform_id="sent", sender_id="bot-id", reply_to_id=None,
                                    parts=({"type": "image", "file_id": "b" * 32},), media_paths={})
    fallback = format_chat_history([image], pic_dir=tmp_path, bot_name="miyuki",
                                   structured_messages={("sent", "miyuki"): mismatched})
    legacy = format_chat_history([image], pic_dir=tmp_path, bot_name="miyuki")
    assert [message.content for message in fallback] == [message.content for message in legacy]


def test_structured_bot_meme_keeps_description_and_private_image_isolation(tmp_path):
    from test_history_metadata import _history
    from test_image_generation import png
    from langchain_core.messages import HumanMessage

    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    (tmp_path / "meme.png").write_bytes(png("red"))
    row = _history(1, "bot", "id: sent\n发送了图片，图片描述是: 高兴的猫\n图片文件: meme.png", media_id=72)
    snapshot = StructuredSnapshot(platform_id="sent", sender_id="bot", reply_to_id=None,
                                  parts=({"type": "image", "media_id": "72"},), media_paths={72: "meme.png"})
    result = format_chat_history([row], pic_dir=tmp_path, bot_name="miyuki", structured_messages={("sent", "miyuki"): snapshot})
    assert "高兴的猫" in str(result[0].content)
    assert isinstance(result[1], HumanMessage)
    assert any(part.get("type") == "image_url" for part in result[1].content if isinstance(part, dict))

    root = tmp_path / "private-files"
    owner = FileWorkspace(root, session_id="alice", user_id="alice", bot_id="bot", is_private=True, peer_bot_ids=())
    artifact = owner.save(png(), "private.png")
    owner.publish_image(artifact.file_id, "private-sent")
    foreign = FileWorkspace(root, session_id="alice", user_id="bob", bot_id="bot", is_private=True, peer_bot_ids=())
    private_row = _history(2, "bot", f"id: private-sent\n[图片文件] private.png file_id={artifact.file_id}", session_id="alice")
    private = StructuredSnapshot(platform_id="private-sent", sender_id="bot", reply_to_id=None,
                                 parts=({"type": "image", "file_id": artifact.file_id},), media_paths={})
    options = {"pic_dir": tmp_path, "bot_name": "miyuki", "structured_messages": {("private-sent", "miyuki"): private}}
    assert len(format_chat_history([private_row], workspace=owner, **options)) == 2
    inaccessible = format_chat_history([private_row], workspace=foreign, **options)
    assert len(inaccessible) == 1
    assert "图片已过期/无法加载" in str(inaccessible[0].content)


async def test_agent_uses_structured_message_context(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound

    group = f"structured-agent-{uuid4()}"
    monkeypatch.setattr(agent.plugin_config, "message_refactor_pilot_groups", [group])
    cfg = ScopedConfig(chat_model="deepseek-flash", chat_multimodal=True)
    monkeypatch.setattr(agent, "resolve_session_chat_config", lambda **kwargs: cfg)
    monkeypatch.setattr(agent, "plugin_data_dir", tmp_path)
    captured = {}
    class Graph:
        async def ainvoke(self, state, **kwargs):
            captured["messages"] = state["messages"]
            return {**state, "called_finish": 1, "llm_call_count": 1}
    monkeypatch.setattr(agent, "create_chat_graph", AsyncMock(return_value=(Graph(), None, "")))
    monkeypatch.setattr(agent, "record_token_usage", AsyncMock())
    monkeypatch.setattr(agent, "update_active_thread", AsyncMock())
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="bob", user_name="Bob", content_type="text", content="id: quoted\n旧引用内容"))
        await record_inbound(db, scope="group", session_id=group, platform_id="quoted", sender_id="bob",
                             sender_name="Bob", reply_to_id=None, parts=[{"type": "text", "text": "引用的真实内容"}])
        row = ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="text", content="id: abc\n旧兼容文字")
        db.add(row)
        await record_inbound(db, scope="group", session_id=group, platform_id="abc", sender_id="alice",
                             sender_name="Alice", reply_to_id=None, parts=[{"type": "text", "text": "真实内容"},
                             {"type": "mention", "target": "456"}])
        await db.commit()
        await db.refresh(row)
        history = [ChatHistorySchema.model_validate(row)]
        await agent.choice_response_strategy(db, group, None, history, "alice", "Alice", bot_id="bot", group_members=[], reply_to_id="quoted")
    assert "messages" in captured
    assert any("真实内容@456" in str(message.content) for message in captured["messages"])
    assert any("引用的真实内容" in str(message.content) for message in captured["messages"])
