import io
import json
import base64
from uuid import uuid4
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Select
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["meme", "web", "file"])
@pytest.mark.parametrize("quoted", [False, True])
async def test_image_and_text_are_one_native_message_and_keep_shared_history(monkeypatch, tmp_path, source, quoted):
    from PIL import Image
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata

    stream = io.BytesIO()
    Image.new("RGB", (9, 7), "red").save(stream, format="PNG")
    raw = stream.getvalue()
    sent = []

    async def capture(message, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "mixed-1"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    group = f"mixed-{uuid4()}"
    args: dict[str, Any]
    async with get_session() as db:
        if source == "meme":
            from nonebot_plugin_ai_groupmate.agent.meme_tools import create_send_meme_tool

            (tmp_path / "meme.png").write_bytes(raw)
            media = MediaStorage(file_hash=uuid4().hex * 2, file_path="meme.png", description="挥手，表示收到")
            db.add(media)
            await db.commit()
            await db.refresh(media)
            sender = create_send_meme_tool(db, group, pic_dir=tmp_path, bot_name="miyuki", approved_meme_ids={media.media_id}, current_message_id="-123")
            args = {"pic_id": str(media.media_id)}
        elif source == "web":
            from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

            search = SimpleNamespace(ainvoke=AsyncMock(return_value={"images": [{"url": "https://example.com/image.png", "description": "挥手"}]}))
            monkeypatch.setattr(module, "TavilySearch", lambda **kwargs: search)
            monkeypatch.setattr(module, "_download_preview", AsyncMock(return_value="data:image/png;base64," + base64.b64encode(raw).decode()))
            lookup, sender, preview = module.create_web_image_tools(db, group, None, tavily_api_key="test", bot_name="miyuki", supports_images=True, current_message_id="-123")
            result = json.loads(await lookup.ainvoke({"query": "挥手"}))
            image_id = result["data"]["images"][0]["image_id"]
            await preview.ainvoke({"image_ids": [image_id]})
            args = {"image_id": image_id}
        else:
            from nonebot_plugin_alconna import Target

            from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
            from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

            workspace = FileWorkspace(tmp_path, session_id=group, user_id="user", bot_id="bot", is_private=False)
            artifact = workspace.save(raw, "generated.png")
            tools = create_file_tools(db, group, None, root=tmp_path, user_id="user", bot_id="bot", is_private=False, bot_name="miyuki", send_target=Target(group), current_message_id="-123")
            sender = next(t for t in tools if t.name == "send_file")
            args = {"file_id": artifact.file_id}

        args.update(text="这句和图一起", reply_to="current" if quoted else None)
        result = json.loads(await sender.ainvoke(args))
        assert result["status"] == "succeeded"
        assert result["data"]["text"] == "这句和图一起"
        assert len(sent) == 1
        assert [part.type for part in sent[0]] == (["reply"] if quoted else []) + ["text", "image"]
        assert str(sent[0][1 if quoted else 0]) == "这句和图一起"
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == group).order_by(ChatHistory.msg_id))).scalars().all()
        assert len(rows) == 2
        assert {parse_message_metadata(row.content)[0] for row in rows} == {"mixed-1"}
        assert {parse_message_metadata(row.content)[1] for row in rows} == ({"-123"} if quoted else {None})
        assert parse_message_metadata(rows[0].content)[2] == "这句和图一起"


class ReplyModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return next(self.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["send_meme_image", "send_web_image", "send_file"])
@pytest.mark.parametrize("continuation", [False, True])
async def test_graph_recognizes_delivered_caption_without_sending_it_again(name, continuation):
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    tool_name: str = name

    @tool(tool_name)
    async def send_image(text: str, next_step: str = "end") -> str:
        """Capture one mixed image reply."""
        return tool_success("image_sent", "图文已发送", data={"text": text}, delivery_state="completed")

    replies = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Capture text."""
        replies.append(content)
        return tool_success("message_sent", "captured")

    responses = [AIMessage(content="", tool_calls=[{"name": name, "args": {"text": "这句和图一起", "next_step": "continue" if continuation else "end"}, "id": "mixed"}])]
    if continuation:
        responses.append(AIMessage(content="", tool_calls=[{"name": "reply_user", "args": {"content": "另一个问题的结果", "next_step": "end"}, "id": "rest"}]))
    model = ReplyModel(responses)
    graph = build_chat_graph(model, [send_image, reply_user], "system")
    result = await graph.ainvoke({"messages": [HumanMessage("图和文字放在同一条消息里")], "session_id": "group", "request_id": None})
    assert model.calls == (2 if continuation else 1)
    assert result["reply_count"] == (2 if continuation else 1)
    assert replies == (["另一个问题的结果"] if continuation else [])


@pytest.mark.asyncio
async def test_mixed_reply_does_not_drop_remaining_required_images():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str, text: str = "") -> str:
        """Capture images."""
        return tool_success("meme_sent", "sent", data={"text": text}, delivery_state="completed")

    model = ReplyModel([
        AIMessage(content="", tool_calls=[{"name": "send_meme_image", "args": {"pic_id": "1", "text": "第一张"}, "id": "first"}]),
        AIMessage(content="", tool_calls=[{"name": "send_meme_image", "args": {"pic_id": "2"}, "id": "second"}]),
    ])
    graph = build_chat_graph(model, [send_meme_image], "system", required_side_effect_tool="send_meme_image", required_side_effect_count=2)
    result = await graph.ainvoke({"messages": [HumanMessage("发两张，第一张带配文")], "session_id": "group", "request_id": None})
    assert model.calls == 2
    assert result["required_side_effect_success_count"] == 2


@pytest.mark.asyncio
async def test_file_caption_does_not_silently_send_non_image_content(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import Target, UniMessage

    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid image content must not partially deliver the caption or a file")

    monkeypatch.setattr(UniMessage, "send", forbidden)
    workspace = FileWorkspace(tmp_path, session_id="group", user_id="user", bot_id="bot", is_private=False)
    artifact = workspace.save(b"ordinary text", "notes.txt")
    async with get_session() as db:
        tools = create_file_tools(db, "group", None, root=tmp_path, user_id="user", bot_id="bot", is_private=False, bot_name="miyuki", send_target=Target("group"))
        sender = next(t for t in tools if t.name == "send_file")
        result = json.loads(await sender.ainvoke({"file_id": artifact.file_id, "text": "这是配文"}))
        assert result["status"] == "failed"
        assert result["delivery_state"] == "not_attempted"


@pytest.mark.asyncio
async def test_failed_mixed_delivery_does_not_finish_before_explaining_failure():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_failure, tool_success

    @tool("send_meme_image")
    async def send_meme_image(text: str) -> str:
        """Simulate a rejected delivery."""
        return tool_failure("not_sent", "尚未发送", data={"text": text}, delivery_state="not_attempted")

    replies = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Capture the explanation."""
        replies.append(content)
        return tool_success("message_sent", "sent")

    model = ReplyModel([
        AIMessage(content="", tool_calls=[{"name": "send_meme_image", "args": {"text": "这句和图一起"}, "id": "mixed"}]),
        AIMessage(content="", tool_calls=[{"name": "reply_user", "args": {"content": "图片没有发送成功", "next_step": "end"}, "id": "explain"}]),
    ])
    result = await build_chat_graph(model, [send_meme_image, reply_user], "system").ainvoke({"messages": [HumanMessage("发图文")], "session_id": "group", "request_id": None})
    assert model.calls == 2
    assert result["reply_count"] == 1
    assert replies == ["图片没有发送成功"]
