import json
from uuid import uuid4
from types import SimpleNamespace

import pytest
from sqlalchemy import Select


@pytest.mark.asyncio
@pytest.mark.parametrize(("request_text", "incoming", "expected"), [
    ("总结我的群聊情况", "**用户交流群**\n聊了工具。\n**结论**：经常交流。", "用户交流群\n聊了工具。\n结论：经常交流。"),
    ("请用 Markdown 格式输出", "**标题**\n这是正文。", "**标题**\n这是正文。"),
    ("请原样发送", "**保留原文**", "**保留原文**"),
    ("总结对话", "他说过“**原样保留**”，另有 **总结**。", "他说过“**原样保留**”，另有 总结。"),
    ("解释一下", "### 标题\n```python\n    x = 2 ** 3\n```\n**答案**", "标题\n```python\n    x = 2 ** 3\n```\n答案"),
    ("不要使用 Markdown", "**标题**", "标题"),
    ("做一道数学题", "**答案** 2**3**4", "答案 2**3**4"),
    ("总结", "他说：'**原话**'，这是 **结论**。", "他说：'**原话**'，这是 结论。"),
    ("总结", "他说「**原话**」，这是 **结论**。", "他说「**原话**」，这是 结论。"),
    ("给我脚本", "# 注释\nx = 2**3**4", "# 注释\nx = 2**3**4"),
])
async def test_text_replies_render_unrequested_markdown_without_changing_requested_content(monkeypatch, request_text, incoming, expected):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "sent-md"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    async with get_session() as db:
        tool = create_reply_tool(db, f"format-{uuid4()}", bot_name="miyuki", parse_msg_meta=parse_message_metadata,
                                 group_members=[], request_text=request_text)
        result = json.loads(await tool.ainvoke({"content": incoming, "next_step": "end"}))
        assert result["status"] == "succeeded"
        assert str(sent[0]) == expected


@pytest.mark.asyncio
async def test_plain_model_response_fallback_uses_chat_formatting(monkeypatch):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage
    from langchain_core.messages import AIMessage, HumanMessage

    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "fallback-md"}])

    class Model:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="**用户交流群**\n大家在讨论工具。")

    monkeypatch.setattr(UniMessage, "send", capture)
    group = f"format-fallback-{uuid4()}"
    async with get_session() as db:
        tool = create_reply_tool(db, group, bot_name="miyuki", parse_msg_meta=parse_message_metadata,
                                 group_members=[], request_text="说说群里在聊什么")
        graph = build_chat_graph(Model(), [tool], "自然聊天")
        await graph.ainvoke({"messages": [HumanMessage(content="说说群里在聊什么")], "session_id": group, "request_id": None})
        assert str(sent[0]) == "用户交流群\n大家在讨论工具。"


@pytest.mark.asyncio
@pytest.mark.parametrize(("reply_to", "expected"), [(None, None), ("current", "-123"), ("-456", "-456"), ("-789", "-789"), ("alias-789", "alias-789")])
async def test_text_quote_reaches_transport_and_history(monkeypatch, tmp_path, reply_to, expected):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import Reply, UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "sent-1"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    group = f"quote-{uuid4()}"
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="user", user_name="user", content_type="text", content="id: -789\nalias_id: alias-789\n前面的那条消息"))
        await db.commit()
        tool = create_reply_tool(db, group, bot_name="miyuki", parse_msg_meta=parse_message_metadata, group_members=[], current_message_id="-123", replied_message_id="-456")
        result = json.loads(await tool.ainvoke({"content": "收到，这条我看见了", "next_step": "end", "reply_to": reply_to}))
        assert result["status"] == "succeeded"
        assert result["data"]["reply_to_message_id"] == expected
        assert [p.id for p in sent[0] if isinstance(p, Reply)] == ([] if expected is None else [expected])
        stored = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == group, ChatHistory.content_type == "bot"))).scalar_one()
        assert parse_message_metadata(stored.content)[1] == expected
        if expected is not None:
            from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

            formatted = format_chat_history([ChatHistorySchema.model_validate(stored)], pic_dir=tmp_path, bot_name="miyuki")
            assert "回复" in formatted[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["missing", "55", "current"])
async def test_unavailable_quote_does_not_send_or_match_a_reply_header(monkeypatch, target):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    async def forbidden(*args, **kwargs):
        pytest.fail("An unavailable quote must not silently become an ordinary send")

    monkeypatch.setattr(UniMessage, "send", forbidden)
    group = f"quote-scope-{uuid4()}"
    async with get_session() as db:
        db.add_all([
            ChatHistory(session_id=group + "-other", user_id="user", user_name="user", content_type="text", content="id: 55\n另一群的消息"),
            ChatHistory(session_id=group, user_id="user", user_name="user", content_type="text", content="id: 66\n回复id: 55\n只是引用过那个ID"),
        ])
        await db.commit()
        tool = create_reply_tool(db, group, bot_name="miyuki", parse_msg_meta=parse_message_metadata, group_members=[])
        result = json.loads(await tool.ainvoke({"content": "这条不能错引", "next_step": "end", "reply_to": target}))
        assert result["reason_code"] == "invalid_reply_target"
        assert result["delivery_state"] == "not_attempted"


@pytest.mark.asyncio
async def test_identical_text_can_reply_to_different_messages(monkeypatch):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": f"sent-{len(sent)}"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    async with get_session() as db:
        group = f"quote-duplicates-{uuid4()}"
        db.add_all([ChatHistory(session_id=group, user_id="user", user_name="user", content_type="text", content=f"id: {target}\nhello")
                    for target in ("11", "22")])
        await db.commit()
        tool = create_reply_tool(db, group, bot_name="miyuki", parse_msg_meta=parse_message_metadata, group_members=[])
        for target, expected in [("11", "succeeded"), ("22", "succeeded"), ("22", "skipped")]:
            result = json.loads(await tool.ainvoke({"content": "收到，这条我看见了", "next_step": "end", "reply_to": target}))
            assert result["status"] == expected
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_meme_quote_reaches_transport_and_keeps_caption(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import Reply, UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.meme_tools import create_send_meme_tool
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "sent-meme"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    (tmp_path / "meme.png").write_bytes(b"image-bytes")
    async with get_session() as db:
        media = MediaStorage(file_hash=uuid4().hex * 2, file_path="meme.png", description="配文‘来啦’，表示回应邀请")
        db.add(media)
        await db.commit()
        await db.refresh(media)
        group = f"quote-meme-{uuid4()}"
        tool = create_send_meme_tool(db, group, pic_dir=tmp_path, bot_name="miyuki", approved_meme_ids={media.media_id}, current_message_id="-123")
        result = json.loads(await tool.ainvoke({"pic_id": str(media.media_id), "reply_to": "current"}))
        assert result["status"] == "succeeded"
        assert [p.id for p in sent[0] if isinstance(p, Reply)] == ["-123"]
        stored = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == group))).scalar_one()
        assert parse_message_metadata(stored.content)[1] == "-123"
        assert "来啦" in stored.content
