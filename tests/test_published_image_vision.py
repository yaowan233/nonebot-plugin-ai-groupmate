import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage


async def test_history_image_tool_reads_exact_scoped_images(tmp_path):
    from uuid import uuid4

    from nonebot_plugin_orm import get_session
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_image_tools import create_history_image_tool
    from nonebot_plugin_ai_groupmate.agent.reply_tools import record_image_reply
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    group = f"history-image-{uuid4()}"
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "upload.png").write_bytes(png("red"))
    owner = FileWorkspace(tmp_path / "files", session_id=group, user_id="alice", bot_id="bot", is_private=False)
    artifact = owner.save(png("blue"), "draw.png")
    owner.publish_image(artifact.file_id, "bot-image")
    viewer = FileWorkspace(tmp_path / "files", session_id=group, user_id="bob", bot_id="bot", is_private=False)
    async with get_session() as db:
        row = ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="image",
                          content="id: upload-id\nupload.png")
        db.add(row)
        record_image_reply(db, session_id=group, bot_name="miyuki", message_id="bot-image",
                           image_content=f"[图片文件] draw.png file_id={artifact.file_id}")
        await db.commit()
        await db.refresh(row)
        kwargs = dict(pic_dir=pics, workspace=viewer, config=ScopedConfig(), platform="qq", is_private=False)
        image_tool = create_history_image_tool(db, group, None, **kwargs)
        for target in ({"message_id": "upload-id"}, {"history_row_id": row.msg_id}, {"message_id": "bot-image"}):
            result = await image_tool.ainvoke(target)
            assert isinstance(result, list)
            assert result[1]["image_url"]["url"].startswith("data:image/")

        foreign = FileWorkspace(tmp_path / "files", session_id="another-group", user_id="bob", bot_id="bot", is_private=False)
        other_tool = create_history_image_tool(db, "another-group", None, **{**kwargs, "workspace": foreign})
        assert json.loads(await other_tool.ainvoke({"history_row_id": row.msg_id}))["status"] == "failed"
        assert json.loads(await other_tool.ainvoke({"message_id": "bot-image"}))["status"] == "failed"


def test_recent_image_reference_list_is_bounded_and_uses_transport_ids():
    from test_history_metadata import _history

    from nonebot_plugin_ai_groupmate.agent.history_format import format_recent_message_references
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    history = [_history(i, "bot", f"id: old-{i}\n[图片文件] x.png file_id={'a' * 32}") for i in range(1, 7)]
    history += [
        _history(7, "image", "id: old-6\n图片文件: bot-echo.png", media_id=1),
        _history(8, "image", "id: user-image\n图片文件: upload.png", media_id=2),
        _history(9, "text", "id: unrelated\n看起来像图片 file_id=" + "a" * 32),
        _history(10, "bot", "没有传输消息 ID\n[图片文件] x.png file_id=" + "b" * 32),
    ]
    history[6].user_id = history[5].user_id
    row = history[7]
    snapshot = StructuredSnapshot(platform_id="indexed-id", sender_id=row.user_id, reply_to_id=None,
                                  parts=({"type": "image", "media_id": "2"},), media_paths={2: "upload.png"},
                                  history_ids=(row.msg_id,))
    rendered = format_recent_message_references(history, session_id="group-1", bot_name="miyuki", message_limit=0,
                                               structured_messages={("indexed-id", row.user_id): snapshot})
    entries = json.loads(rendered.split("\n", 1)[1])
    assert [item["message_id"] for item in entries] == ["old-3", "old-4", "old-5", "old-6", "indexed-id"]
    assert entries[-2]["images"] == 1
    assert entries[-2]["sender"] == "miyuki"
    assert entries[-1]["sender"] == row.user_name
    assert "unrelated" not in rendered


async def test_unquoted_followup_can_reference_published_bot_image(monkeypatch, tmp_path):
    from uuid import uuid4
    from datetime import datetime
    from typing import cast
    from unittest.mock import AsyncMock

    from sqlalchemy import select
    from nonebot_plugin_orm import get_session
    from nonebot.adapters import Bot
    from test_image_generation import png, image_config

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.agent import image_generation_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema
    from nonebot_plugin_ai_groupmate.agent.reply_tools import record_image_reply
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    group = f"unquoted-edit-{uuid4()}"
    monkeypatch.setattr(agent, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(agent.plugin_config, "image_generation_enabled", True)
    owner = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot-a", is_private=False)
    artifact = owner.save(png("red"), "draw.png")
    owner.publish_image(artifact.file_id, "sent-image")
    captured = {}

    class Graph:
        async def ainvoke(self, state, **kwargs):
            captured["messages"] = state["messages"]
            return {**state, "called_finish": 1, "llm_call_count": 1}

    monkeypatch.setattr(agent, "create_chat_graph", AsyncMock(return_value=(Graph(), None, "")))
    monkeypatch.setattr(agent, "record_token_usage", AsyncMock())
    monkeypatch.setattr(agent, "update_active_thread", AsyncMock())
    monkeypatch.setattr(agent, "resolve_session_chat_config", lambda **kw: image_config())
    provider = AsyncMock(return_value=GeneratedImage(png("blue"), {}))
    monkeypatch.setattr(image_generation_tools, "request_image", provider)
    async with get_session() as db:
        record_image_reply(db, session_id=group, bot_name="miyuki", message_id="sent-image",
                           image_content=f"[图片文件] draw.png file_id={artifact.file_id}")
        await db.commit()
        rows = (await db.scalars(select(ChatHistory).where(ChatHistory.session_id == group))).all()
        history = [ChatHistorySchema.model_validate(row) for row in rows]
        history.append(ChatHistorySchema(msg_id=99999999, session_id=group, user_id="bob", user_name="Bob",
                                        content_type="text", content="id: new\n把刚才那张图改成蓝色", created_at=datetime.now()))
        bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))
        await agent.choice_response_strategy(db, group, None, history, "bob", "Bob", bot_id="bot-a",
                                             bot=cast(Bot, bot), group_members=[], reply_to_id=None)
        assert "sent-image" in str([m.content for m in captured["messages"]]), "unquoted history must expose the actual image message ID"
        viewer = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="bob", bot_id="bot-a", is_private=False)
        tool = image_generation_tools.create_image_generation_tool(db, group, None, config=image_config(), workspace=viewer,
               pic_dir=tmp_path / "pics", current_message_id="new", replied_message_id=None)
        result = json.loads(await tool.ainvoke({"prompt": "改成蓝色", "reference_message_ids": ["sent-image"]}))
        assert result["status"] == "succeeded"
        assert provider.await_args is not None
        assert provider.await_args.args[2][0][0] == png("red")


async def test_another_member_can_see_a_quoted_bot_file_image(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_format import is_image_history, format_chat_history, load_replied_message_histories

    root = tmp_path / "files"
    owner = FileWorkspace(root, session_id="published-vision", user_id="alice", bot_id="bot-a", is_private=False)
    artifact = owner.save(png(), "gpt-image.png")

    async def send(*args, **kwargs):
        return SimpleNamespace(msg_ids=[{"message_id": "published-vision-message"}])

    monkeypatch.setattr(UniMessage, "send", send)
    async with get_session() as session:
        tools = create_file_tools(session, "published-vision", None, root=root, user_id="alice", bot_id="bot-a", is_private=False, bot_name="miyuki", send_target=None)
        send_file = next(t for t in tools if t.name == "send_file")
        result = json.loads(await send_file.ainvoke({"file_id": artifact.file_id, "as_image": True}))
        assert result["delivery_state"] == "completed"
        await session.commit()
        quoted = await load_replied_message_histories(session, "published-vision", "published-vision-message")
    assert len(quoted) == 1
    assert is_image_history(quoted[0]), "Bot image artifacts must be recognized as images, not plain assistant text"
    viewer = FileWorkspace(root, session_id="published-vision", user_id="bob", bot_id="bot-b", is_private=False)
    messages = format_chat_history(quoted, pic_dir=tmp_path / "pics", bot_name="miyuki", workspace=viewer)
    assert isinstance(messages[0], AIMessage)
    blocks = [b for m in messages if isinstance(m.content, list) for b in m.content if isinstance(b, dict) and b.get("type") == "image_url"]
    assert len(blocks) == 1
    assert blocks[0]["image_url"]["url"].startswith("data:image/")


def test_publication_vision_keeps_scope_exact_message_and_expiry(monkeypatch, tmp_path):
    from test_history_metadata import _history
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent import file_workspace as storage
    from nonebot_plugin_ai_groupmate.agent.history_format import is_image_history, get_history_image_data_uri

    def workspace(group="group-1", user="alice", private=False):
        return storage.FileWorkspace(tmp_path, session_id=group, user_id=user, bot_id="bot", is_private=private)

    owner = workspace()
    artifact = owner.save(png(), "draw.png")
    row = _history(1, "bot", f"id: 101\n[图片文件] draw.png file_id={artifact.file_id}")
    resolve = lambda ws: get_history_image_data_uri(row, pic_dir=tmp_path, workspace=ws)
    assert resolve(owner) is None, "Unsent artifacts are not public images"
    owner.publish_image(artifact.file_id, "101")
    assert resolve(workspace(user="bob"))
    assert resolve(workspace(group="other")) is None
    assert resolve(workspace(private=True)) is None
    row.content = row.content.replace("id: 101", "id: 102")
    assert resolve(owner) is None
    row.content_type = "text"
    assert not is_image_history(row), "User-supplied file markers are not trusted bot publications"
    row.content_type = "bot"
    row.content = row.content.replace("id: 102", "id: 101")
    now = storage.time.time()
    monkeypatch.setattr(storage.time, "time", lambda: now + storage.FILE_TTL_SECONDS + 1)
    assert resolve(owner) is None


@pytest.mark.parametrize("multimodal", [True, False])
async def test_quote_attaches_published_pixels_to_actual_agent_prompt(monkeypatch, tmp_path, multimodal):
    from datetime import datetime
    from unittest.mock import AsyncMock

    from nonebot_plugin_orm import get_session
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.reply_tools import record_image_reply
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"agent-published-vision-{multimodal}"
    monkeypatch.setattr(agent, "plugin_data_dir", tmp_path)
    cfg = ScopedConfig(chat_model="gpt-6.1-terra", chat_multimodal=multimodal)
    monkeypatch.setattr(agent, "resolve_session_chat_config", lambda **kw: cfg)
    workspace = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot-a", is_private=False)
    artifact = workspace.save(png(), "draw.png")
    workspace.publish_image(artifact.file_id, "quote-me")
    captured = {}

    class Graph:
        async def ainvoke(self, state, **kwargs):
            captured["prompt"] = state["messages"][-1].content
            return {**state, "called_finish": 1, "llm_call_count": 1}

    monkeypatch.setattr(agent, "create_chat_graph", AsyncMock(return_value=(Graph(), None, "")))
    monkeypatch.setattr(agent, "record_token_usage", AsyncMock())
    monkeypatch.setattr(agent, "update_active_thread", AsyncMock())
    summary = AsyncMock(return_value="画面里是一只蓝色小猫。")
    monkeypatch.setattr(agent, "_summarize_image_content", summary)
    current = ChatHistorySchema(msg_id=999999, session_id=group, user_id="bob", user_name="Bob", content_type="text", content="id: new\n回复id: quote-me\n这张图是什么", created_at=datetime.now())
    async with get_session() as db:
        record_image_reply(db, session_id=group, bot_name="miyuki", message_id="quote-me", image_content=f"[图片文件] draw.png file_id={artifact.file_id}")
        # A later message quoting the target must not be mistaken for the target.
        db.add(ChatHistory(session_id=group, user_id="someone", user_name="Someone", content_type="text", content="id: other\n回复id: quote-me\nirrelevant quote"))
        await db.commit()
        await agent.choice_response_strategy(db, group, None, [current], "bob", "Bob", bot_id="bot-b", group_members=[], reply_to_id="quote-me")
    assert "prompt" in captured, "Agent setup must reach graph invocation"
    if multimodal:
        assert isinstance(captured["prompt"], list)
        images = [b for b in captured["prompt"] if isinstance(b, dict) and b.get("type") == "image_url"]
        assert len(images) == 1
        assert images[0]["image_url"]["url"].startswith("data:image/")
        summary.assert_not_awaited()
    else:
        assert "蓝色小猫" in captured["prompt"]
        assert len(summary.call_args.args[0]) == 1
        assert summary.call_args.args[0][0]["image_url"]["url"].startswith("data:image/")


@pytest.mark.parametrize("multimodal", [True, False])
@pytest.mark.parametrize("include_image", [True, False])
async def test_structured_only_quote_supplies_images_to_actual_agent(monkeypatch, tmp_path, multimodal, include_image):
    from types import SimpleNamespace
    from datetime import datetime
    from unittest.mock import AsyncMock

    from nonebot_plugin_orm import get_session
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"structured-only-vision-{multimodal}-{include_image}"
    monkeypatch.setattr(agent, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(agent.plugin_config, "message_refactor_pilot_groups", [group])
    monkeypatch.setattr(agent, "resolve_session_chat_config", lambda **kw: ScopedConfig(chat_multimodal=multimodal))
    owner = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot-a", is_private=False)
    artifact = owner.save(png(), "draw.png")
    owner.publish_image(artifact.file_id, "quote-me")
    captured = {}

    class Graph:
        async def ainvoke(self, state, **kwargs):
            captured["prompt"] = state["messages"][-1].content
            return {**state, "called_finish": 1, "llm_call_count": 1}

    monkeypatch.setattr(agent, "create_chat_graph", AsyncMock(return_value=(Graph(), None, "")))
    monkeypatch.setattr(agent, "record_token_usage", AsyncMock())
    monkeypatch.setattr(agent, "update_active_thread", AsyncMock())
    summary = AsyncMock(return_value="画面是一只蓝色小猫。")
    monkeypatch.setattr(agent, "_summarize_image_content", summary)
    monkeypatch.setattr(agent, "_load_replied_message_histories", AsyncMock(side_effect=AssertionError("legacy quote lookup used")))
    current = ChatHistorySchema(msg_id=999999, session_id=group, user_id="bob", user_name="Bob", content_type="text",
                                content="id: new\n回复id: quote-me\n这张图是什么", created_at=datetime.now())
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))
    async with get_session() as db:
        await record_inbound(db, platform="qq", scope="group", session_id=group, platform_id="quote-me",
                             sender_id="bot-a", sender_name="miyuki", sender_kind="bot", reply_to_id=None,
                             parts=[{"type": "text", "text": "这是一幅小画"}] +
                                   ([{"type": "image", "file_id": artifact.file_id}] if include_image else []))
        await db.commit()
        await agent.choice_response_strategy(db, group, None, [current], "bob", "Bob", bot_id="bot-b",
                                             bot=cast(Any, bot), group_members=[], reply_to_id="quote-me")
    assert "prompt" in captured
    assert "这是一幅小画" in str(captured["prompt"])
    if multimodal and include_image:
        images = [part for part in captured["prompt"] if isinstance(part, dict) and part.get("type") == "image_url"]
        assert len(images) == 1
        assert images[0]["image_url"]["url"].startswith("data:image/")
        summary.assert_not_awaited()
    elif include_image:
        assert "蓝色小猫" in captured["prompt"]
        assert summary.await_args is not None
        assert len(summary.await_args.args[0]) == 1
    else:
        assert isinstance(captured["prompt"], str)
        summary.assert_not_awaited()
