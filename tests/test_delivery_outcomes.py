import json
from uuid import uuid4
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock


async def test_background_delivery_opens_session_only_after_send_ack(monkeypatch):
    from contextlib import asynccontextmanager

    from nonebot_plugin_ai_groupmate.agent import delivery
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    events = []
    receipt = SimpleNamespace(msg_ids=[{"message_id": "sent"}])
    async def send(**kwargs):
        events.append("send")
        return receipt
    async def commit():
        events.append("commit")
    @asynccontextmanager
    async def session_factory():
        events.append("open_session")
        yield SimpleNamespace(add=lambda row: None, commit=commit)
    def record(message_id):
        return [ChatHistory(session_id="group", user_id="bot", user_name="bot", content_type="bot", content=message_id)]
    result = await delivery.send_and_finalize(None, "group", SimpleNamespace(send=send), send_options={},
                                              record=record, recovery_enabled=False, session_factory=session_factory)
    assert result.message_id == "sent"
    assert result.history_saved
    assert events == ["send", "open_session", "commit"]


async def test_background_ack_is_journaled_even_if_opening_database_fails(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager

    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import delivery
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    group = f"failed-session-{uuid4()}"
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "sent"}]))
    @asynccontextmanager
    async def failed_session():
        raise RuntimeError("database unavailable")
        yield
    def record(message_id):
        return [ChatHistory(session_id=group, user_id="bot", user_name="bot", content_type="bot", content=message_id)]
    result = await delivery.send_and_finalize(None, group, SimpleNamespace(send=sender), send_options={},
                                              record=record, session_factory=failed_session)
    assert result.message_id == "sent"
    assert not result.history_saved
    assert len(list((tmp_path / "delivery_recovery").glob("*.json"))) == 1
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    async with get_session() as db:
        assert await db.scalar(Select(ChatHistory.msg_id).where(ChatHistory.session_id == group)) is not None
    sender.assert_awaited_once()


async def test_outbound_text_records_mentions_and_recovers_structured_failure(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import structured_message as structured
    from nonebot_plugin_ai_groupmate.model import ChatHistory, StructuredMessage
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending

    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "sent-1"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    original = structured.record_inbound
    async def fail_after_insert(db, **kwargs):
        await original(db, **kwargs)
        raise RuntimeError("structured storage unavailable")
    monkeypatch.setattr(structured, "record_inbound", fail_after_insert)
    group = f"outbound-{uuid4()}"
    async with get_session() as db:
        tool = create_reply_tool(db, group, bot_name="miyuki", parse_msg_meta=parse_message_metadata,
                                 group_members=[SimpleNamespace(id="123", name="朋友")], current_message_id="incoming",
                                 outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"))
        result = json.loads(await tool.ainvoke({"content": "你好 @123！", "next_step": "end", "reply_to": "current"}))
        assert result["delivery_state"] == "completed"
        assert result["data"]["history_saved"] is True
        assert await db.scalar(Select(ChatHistory.msg_id).where(ChatHistory.session_id == group)) is not None
        assert await db.scalar(Select(StructuredMessage.id).where(StructuredMessage.session_id == group)) is None
    pending = list((tmp_path / "delivery_recovery").glob("*.json"))
    assert len(pending) == 1
    backup = pending[0].read_bytes()
    monkeypatch.setattr(structured, "record_inbound", original)
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    pending[0].write_bytes(backup)
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    async with get_session() as db:
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert snapshot is not None
        assert snapshot.sender_kind == "bot"
        assert snapshot.sender_id == "bot"
        assert snapshot.reply_to_id == "incoming"
        assert [part["type"] for part in snapshot.parts] == ["text", "mention", "text"]
        assert snapshot.parts[1]["target"] == "123"
        histories = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        assert len(histories) == 1
        assert snapshot.created_at == histories[0].created_at
    assert sender.await_count == 1


async def test_text_receipt_survives_history_commit_failure_without_resending(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending

    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"msg_id": "sent-1"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    group = f"delivery-{uuid4()}"
    async with get_session() as db:
        original_commit = db.commit
        calls = 0
        async def fail_record_commit():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("database unavailable after platform send")
            await original_commit()
        monkeypatch.setattr(db, "commit", fail_record_commit)
        tool = create_reply_tool(db, group, bot_name="miyuki",
                                 parse_msg_meta=parse_message_metadata, group_members=[])
        result = json.loads(await tool.ainvoke({"content": "已经发出", "next_step": "end"}))
        assert result["status"] == "succeeded"
        assert result["delivery_state"] == "completed"
        assert result["data"]["message_id"] == "sent-1"
        assert result["data"]["history_saved"] is False
        assert "请勿重复发送" in result["message"]
        second = json.loads(await tool.ainvoke({"content": "已经发出", "next_step": "end"}))
        assert second["reason_code"] == "duplicate_reply"
        assert sender.await_count == 1
    pending = list((tmp_path / "delivery_recovery").glob("*.json"))
    assert len(pending) == 1
    backup = pending[0].read_bytes()
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    # A crash after commit but before file deletion can replay the same receipt.
    pending[0].write_bytes(backup)
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    async with get_session() as db:
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        assert len(rows) == 1
        assert "已经发出" in rows[0].content
    assert sender.await_count == 1


async def test_text_failure_before_send_is_not_attempted(monkeypatch):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    sender = AsyncMock()
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        monkeypatch.setattr(db, "commit", AsyncMock(side_effect=RuntimeError("database unavailable")))
        tool = create_reply_tool(db, f"delivery-before-{uuid4()}", bot_name="miyuki",
                                 parse_msg_meta=parse_message_metadata, group_members=[])
        result = json.loads(await tool.ainvoke({"content": "测试", "next_step": "end"}))
        assert result["delivery_state"] == "not_attempted"
        sender.assert_not_awaited()


async def test_outbound_mixed_image_preserves_order_and_quote(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema, StructuredMessage
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import load_structured_context

    group = f"outbound-image-{uuid4()}"
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    root = tmp_path / "agent_files"
    artifact = FileWorkspace(root, session_id=group, user_id="user", bot_id="bot", is_private=False).save(png(), "p.png")
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "mixed-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        tools = create_file_tools(db, group, None, root=root, user_id="user", bot_id="bot", is_private=False,
                                  bot_name="miyuki", send_target=None, current_message_id="incoming",
                                  outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"))
        send_file = next(tool for tool in tools if tool.name == "send_file")
        result = json.loads(await send_file.ainvoke({"file_id": artifact.file_id, "text": "给你", "reply_to": "current"}))
        assert result["delivery_state"] == "completed"
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert snapshot is not None
        assert snapshot.reply_to_id == "incoming"
        assert snapshot.parts == [{"type": "text", "text": "给你"}, {"type": "image", "file_id": artifact.file_id}]
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group).order_by(ChatHistory.msg_id))).all()
        history = [ChatHistorySchema.model_validate(row) for row in rows]
        snapshots = await load_structured_context(db, history, session_id=group, scope="group", platform="qq")
        viewer = FileWorkspace(root, session_id=group, user_id="other", bot_id="bot-2", is_private=False, peer_bot_ids=())
        messages = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki", workspace=viewer, structured_messages=snapshots)
        assert "给你[图片]" in str(messages[0].content)
        assert any(part.get("type") == "image_url" for part in messages[1].content if isinstance(part, dict))
    assert sender.await_count == 1


async def test_scheduled_delivery_recovers_after_ack_without_resending(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import delivery, schedule_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory, StructuredMessage
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending

    group = f"scheduled-outbound-{uuid4()}"
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(schedule_tools, "get_runtime_config", lambda: ScopedConfig(message_refactor_pilot_groups=[group]))
    monkeypatch.setattr(schedule_tools, "get_bots", lambda: {})
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "scheduled-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    original = delivery.apply_history
    monkeypatch.setattr(delivery, "apply_history", AsyncMock(side_effect=RuntimeError("database unavailable")))
    await schedule_tools.send_scheduled_text(group, "定时内容", is_private=False, bot_id="bot", bot_name="miyuki")
    assert sender.await_count == 1
    assert len(list((tmp_path / "delivery_recovery").glob("*.json"))) == 1
    monkeypatch.setattr(delivery, "apply_history", original)
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    async with get_session() as db:
        history = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert len(history) == 1
        assert snapshot is not None
        assert snapshot.parts == [{"type": "text", "text": "定时内容"}]
    assert sender.await_count == 1


async def test_private_delivery_recovers_both_histories_without_resending(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import delivery, private_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory, StructuredMessage
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending

    group = f"private-origin-{uuid4()}"
    target = f"private-target-{uuid4()}"
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(private_tools, "get_runtime_config", lambda: ScopedConfig(message_refactor_pilot_private=True))
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "private-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    original = delivery.apply_history
    monkeypatch.setattr(delivery, "apply_history", AsyncMock(side_effect=RuntimeError("database unavailable")))
    async with get_session() as db:
        tool = private_tools.create_private_message_tool(db, group, None, cast(Any, object()), bot_id="bot", bot_name="miyuki",
                                                          group_members=[SimpleNamespace(id=target)], platform="qq")
        result = json.loads(await tool.ainvoke({"target_user_id": target, "content": "私聊内容"}))
        assert result["delivery_state"] == "completed"
        assert result["data"]["history_saved"] is False
        assert "请勿重复发送" in result["message"]
    monkeypatch.setattr(delivery, "apply_history", original)
    assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
    async with get_session() as db:
        histories = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id.in_((group, target))))).all()
        assert {row.session_id for row in histories} == {group, target}
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == target))
        assert snapshot is not None
        assert snapshot.scope == "private"
        assert snapshot.parts == [{"type": "text", "text": "私聊内容"}]
    assert sender.await_count == 1


async def test_relation_image_delivery_recovers_and_is_visible_to_group(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import delivery
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema, StructuredMessage
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending
    from nonebot_plugin_ai_groupmate.agent.profile_tools import create_relation_tool
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import load_structured_context

    group = f"relation-image-{uuid4()}"
    user = f"relation-user-{uuid4()}"
    root = tmp_path / "agent_files"
    owner = FileWorkspace(root, session_id=group, user_id=user, bot_id="bot", is_private=False, peer_bot_ids=())
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "relation-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    original = delivery.apply_history
    monkeypatch.setattr(delivery, "apply_history", AsyncMock(side_effect=RuntimeError("database unavailable")))
    async with get_session() as db:
        tool = create_relation_tool(db, group, None, user, "Alice", bot_name="miyuki", up_pic=png("red"), down_pic=png("blue"),
                                    outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"), workspace=owner)
        result = json.loads(await tool.ainvoke({"score_change": 10, "reason": "测试"}))
        assert result["status"] == "succeeded"
        assert sender.await_count == 1
    assert len(list((tmp_path / "delivery_recovery").glob("*.json"))) == 1
    monkeypatch.setattr(delivery, "apply_history", original)
    assert await recover_pending(tmp_path / "delivery_recovery", root) == 1
    async with get_session() as db:
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group).order_by(ChatHistory.msg_id))).all()
        history = [ChatHistorySchema.model_validate(row) for row in rows]
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert snapshot is not None
        assert [part["type"] for part in snapshot.parts] == ["image", "text"]
        snapshots = await load_structured_context(db, history, session_id=group, scope="group", platform="qq")
        viewer = FileWorkspace(root, session_id=group, user_id="bob", bot_id="bot-2", is_private=False, peer_bot_ids=())
        rendered = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki", workspace=viewer, structured_messages=snapshots)
        assert "[图片]好感度提升" in str(rendered[0].content)
        assert any(part.get("type") == "image_url" for part in rendered[1].content if isinstance(part, dict))
    assert sender.await_count == 1


async def test_relation_image_still_sends_when_artifact_storage_is_unavailable(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.profile_tools import create_relation_tool
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group, user = f"relation-fallback-{uuid4()}", f"relation-user-{uuid4()}"
    root = tmp_path / "agent_files"
    workspace = FileWorkspace(root, session_id=group, user_id=user, bot_id="bot", is_private=False, peer_bot_ids=())
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    def fail_save(*_args, **_kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(workspace, "save", fail_save)
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "relation-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        tool = create_relation_tool(db, group, None, user, "Alice", bot_name="miyuki", up_pic=png(), down_pic=png(),
                                    outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"), workspace=workspace)
        result = json.loads(await tool.ainvoke({"score_change": 10, "reason": "测试"}))
        assert result["status"] == "succeeded"
    assert sender.await_count == 1


async def test_web_image_publication_can_be_referenced_by_another_member(monkeypatch, tmp_path):
    import base64

    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import image_search_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema, StructuredMessage
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import load_structured_context

    group = f"web-image-publication-{uuid4()}"
    root = tmp_path / "agent_files"
    owner = FileWorkspace(root, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    viewer = FileWorkspace(root, session_id=group, user_id="bob", bot_id="bot-2", is_private=False, peer_bot_ids=())
    raw = png("red")
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    search = SimpleNamespace(ainvoke=AsyncMock(return_value={"images": [{"url": "https://example.com/cat.png", "description": "cat"}]}))
    monkeypatch.setattr(image_search_tools, "TavilySearch", lambda **_kwargs: search)
    monkeypatch.setattr(image_search_tools, "_download_preview", AsyncMock(return_value="data:image/png;base64," + base64.b64encode(raw).decode()))
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "web-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        lookup, send, preview = image_search_tools.create_web_image_tools(
            db, group, None, tavily_api_key="test", bot_name="miyuki", supports_images=True,
            current_message_id="incoming", outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"), workspace=owner,
        )
        result = json.loads(await lookup.ainvoke({"query": "cat"}))
        image_id = result["data"]["images"][0]["image_id"]
        await preview.ainvoke({"image_ids": [image_id]})
        result = json.loads(await send.ainvoke({"image_id": image_id, "text": "这张", "reply_to": "current"}))
        assert result["status"] == "succeeded"
        file_id = result["data"]["file_id"]
        assert viewer.read_published_image(file_id, "web-sent") == raw
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group).order_by(ChatHistory.msg_id))).all()
        assert any("搜索描述: cat" in row.content and "图片来源: https://example.com/cat.png" in row.content for row in rows)
        history = [ChatHistorySchema.model_validate(row) for row in rows]
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert snapshot is not None
        assert snapshot.reply_to_id == "incoming"
        assert snapshot.parts == [{"type": "text", "text": "这张"}, {"type": "image", "file_id": file_id}]
        snapshots = await load_structured_context(db, history, session_id=group, scope="group", platform="qq")
        rendered = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki", workspace=viewer, structured_messages=snapshots)
        assert "这张[图片]" in str(rendered[0].content)
        assert any(part.get("type") == "image_url" for part in rendered[1].content if isinstance(part, dict))
    assert sender.await_count == 1


async def test_web_image_sends_when_workspace_is_unavailable(monkeypatch, tmp_path):
    import base64

    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import image_search_tools
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"web-image-fallback-{uuid4()}"
    workspace = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    def fail_save(*_args, **_kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(workspace, "save", fail_save)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    search = SimpleNamespace(ainvoke=AsyncMock(return_value={"images": ["https://example.com/cat.png"]}))
    monkeypatch.setattr(image_search_tools, "TavilySearch", lambda **_kwargs: search)
    monkeypatch.setattr(image_search_tools, "_download_preview", AsyncMock(return_value="data:image/png;base64," + base64.b64encode(png()).decode()))
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "web-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        lookup, send, preview = image_search_tools.create_web_image_tools(
            db, group, None, tavily_api_key="test", bot_name="miyuki", supports_images=True,
            outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"), workspace=workspace,
        )
        result = json.loads(await lookup.ainvoke({"query": "cat"}))
        image_id = result["data"]["images"][0]["image_id"]
        await preview.ainvoke({"image_ids": [image_id]})
        result = json.loads(await send.ainvoke({"image_id": image_id}))
        assert result["status"] == "succeeded"
        assert result["data"]["file_id"] is None
    assert sender.await_count == 1


async def test_web_image_publication_failure_recovers_without_resending(monkeypatch, tmp_path):
    import base64

    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import image_search_tools
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileTaskError, FileWorkspace

    group = f"web-image-recovery-{uuid4()}"
    root = tmp_path / "agent_files"
    owner = FileWorkspace(root, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    viewer = FileWorkspace(root, session_id=group, user_id="bob", bot_id="bot-2", is_private=False, peer_bot_ids=())
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    search = SimpleNamespace(ainvoke=AsyncMock(return_value={"images": ["https://example.com/cat.png"]}))
    monkeypatch.setattr(image_search_tools, "TavilySearch", lambda **_kwargs: search)
    monkeypatch.setattr(image_search_tools, "_download_preview", AsyncMock(return_value="data:image/png;base64," + base64.b64encode(png()).decode()))
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "web-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    original_publish = FileWorkspace.publish_image
    def fail_publish(*_args, **_kwargs):
        raise FileTaskError("published_image_not_found", "storage unavailable")
    monkeypatch.setattr(FileWorkspace, "publish_image", fail_publish)
    async with get_session() as db:
        lookup, send, preview = image_search_tools.create_web_image_tools(
            db, group, None, tavily_api_key="test", bot_name="miyuki", supports_images=True,
            outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"), workspace=owner,
        )
        result = json.loads(await lookup.ainvoke({"query": "cat"}))
        image_id = result["data"]["images"][0]["image_id"]
        await preview.ainvoke({"image_ids": [image_id]})
        result = json.loads(await send.ainvoke({"image_id": image_id}))
        assert result["status"] == "succeeded"
        assert result["data"]["publication_saved"] is False
        file_id = result["data"]["file_id"]
    monkeypatch.setattr(FileWorkspace, "publish_image", original_publish)
    assert await recover_pending(tmp_path / "delivery_recovery", root) == 1
    assert viewer.read_published_image(file_id, "web-sent") == png()
    assert sender.await_count == 1


async def test_command_image_recovery_preserves_reply_and_publication(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import delivery
    from nonebot_plugin_ai_groupmate.model import ChatHistory, StructuredMessage
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.command_delivery import send_command_image

    group = f"command-image-{uuid4()}"
    root = tmp_path / "agent_files"
    owner = FileWorkspace(root, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    viewer = FileWorkspace(root, session_id=group, user_id="bob", bot_id="other", is_private=False, peer_bot_ids=())
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "command-sent"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    original = delivery.apply_history
    monkeypatch.setattr(delivery, "apply_history", AsyncMock(side_effect=RuntimeError("database unavailable")))
    async with get_session() as db:
        await send_command_image(db, png(), workspace=owner, identity=OutboundIdentity("qq", "group", "bot", "miyuki"),
                                 reply_to_id="request", name="词云.png")
    sender.assert_awaited_once()
    assert sender.await_args is not None
    assert sender.await_args.kwargs["reply_to"] is True
    assert len(list((tmp_path / "delivery_recovery").glob("*.json"))) == 1
    monkeypatch.setattr(delivery, "apply_history", original)
    assert await recover_pending(tmp_path / "delivery_recovery", root) == 1
    async with get_session() as db:
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert len(rows) == 1
        assert snapshot is not None
        assert snapshot.reply_to_id == "request"
        file_id = snapshot.parts[0]["file_id"]
        assert viewer.read_published_image(file_id, "command-sent") == png()
    assert sender.await_count == 1


async def test_wordcloud_delivery_switch_is_scoped(monkeypatch, tmp_path):
    from nonebot_plugin_uninfo import SceneType
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin.plugin_config, "message_refactor_pilot_groups", ["pilot"])
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin, "get_bots", lambda: {})
    monkeypatch.setattr(plugin, "get_message_id", lambda: "request")
    command_send = AsyncMock()
    monkeypatch.setattr(plugin, "send_command_image", command_send)
    legacy_send = AsyncMock()
    monkeypatch.setattr(UniMessage, "send", legacy_send)
    for group in ("pilot", "other"):
        session = SimpleNamespace(scene=SimpleNamespace(id=group, type=SceneType.GROUP),
                                  user=SimpleNamespace(id="alice"), self_id="bot")
        await plugin._send_wordcloud_image(object(), cast(Any, session), b"image")
    command_send.assert_awaited_once()
    legacy_send.assert_awaited_once()
    assert command_send.await_args is not None
    assert command_send.await_args.kwargs["reply_to_id"] == "request"
    assert legacy_send.await_args is not None
    assert legacy_send.await_args.kwargs["reply_to"] is True


def test_wordcloud_output_can_be_saved_as_an_image():
    from PIL import Image

    import nonebot_plugin_ai_groupmate as plugin

    image = plugin._build_wordcloud_image("聊天 聊天 图片 绘图")
    assert image.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(image) as opened:
        assert opened.size == (1000, 500)


async def test_published_image_failure_keeps_confirmed_send_and_history(monkeypatch, tmp_path):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from test_image_generation import png
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileTaskError, FileWorkspace

    group = f"delivery-image-{uuid4()}"
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    root = tmp_path / "agent_files"
    workspace = FileWorkspace(root, session_id=group, user_id="user", bot_id="bot", is_private=False)
    artifact = workspace.save(png(), "picture.png")
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "sent-image"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    def fail_publish(*args):
        raise FileTaskError("published_image_not_found", "test")
    original_publish = FileWorkspace.publish_image
    monkeypatch.setattr(FileWorkspace, "publish_image", fail_publish)
    async with get_session() as db:
        tools = create_file_tools(db, group, None, root=root, user_id="user", bot_id="bot",
                                  is_private=False, bot_name="miyuki", send_target=None,
                                  outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki"))
        send_file = next(tool for tool in tools if tool.name == "send_file")
        result = json.loads(await send_file.ainvoke({"file_id": artifact.file_id, "as_image": True}))
        assert result["delivery_state"] == "completed"
        assert result["status"] == "succeeded"
        assert result["data"]["publication_saved"] is False
        assert result["data"]["history_saved"] is True
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        assert len(rows) == 1
        assert "sent-image" in rows[0].content
        second = json.loads(await send_file.ainvoke({"file_id": artifact.file_id, "as_image": True}))
        assert second["reason_code"] == "duplicate_file"
        assert sender.await_count == 1
    monkeypatch.setattr(FileWorkspace, "publish_image", original_publish)
    assert await recover_pending(tmp_path / "delivery_recovery", root) == 1
    viewer = FileWorkspace(root, session_id=group, user_id="other", bot_id="bot", is_private=False)
    assert viewer.read_published_image(artifact.file_id, "sent-image") == png()
    async with get_session() as db:
        from nonebot_plugin_ai_groupmate.model import StructuredMessage
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        assert len(rows) == 1
        snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
        assert snapshot is not None
        assert snapshot.parts == [{"type": "image", "file_id": artifact.file_id}]
    assert sender.await_count == 1
