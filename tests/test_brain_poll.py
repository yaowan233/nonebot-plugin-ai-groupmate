import json
from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
async def cleanup_direct_tasks():
    yield
    from sqlalchemy import delete, select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask, BrainAction

    async with get_session() as db:
        ids = select(BrainTask.task_id).where(BrainTask.source_scene.like("cross-send-test-%"))
        await db.execute(delete(BrainAction).where(BrainAction.task_id.in_(ids)))
        await db.execute(delete(BrainTask).where(BrainTask.source_scene.like("cross-send-test-%")))
        await db.commit()


@pytest.mark.asyncio
async def test_send_to_linked_group_uses_explicit_content_and_confirms_delivery(cleanup_direct_tasks):
    from nonebot_plugin_ai_groupmate.brain.poll import poll
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool

    group_a, group_b, bot = f"cross-send-test-{uuid4().hex}", uuid4().hex, uuid4().hex
    spaces = {"friends": [group_a, group_b]}
    reachable = AsyncMock(return_value={group_a, group_b})
    tool = create_coordination_tool(config=SimpleNamespace(cross_group_spaces=spaces), platform="qq", bot_id=bot,
                                    scene_id=group_a, user_id="alice", message_id="source", request_id=None,
                                    request_text="你随便吧", reachable_groups=reachable)
    args = {"reason": "用户让我去交流群发一句问候", "operation": "send", "target_group_id": group_b, "message": "大家好呀"}
    result = json.loads(await tool.ainvoke(args))
    assert result["reason_code"] == "coordination_queued"
    assert json.loads(await tool.ainvoke(args))["data"]["created"] is False
    conflict = json.loads(await tool.ainvoke({**args, "message": "另一条消息"}))
    assert conflict["reason_code"] == "task_conflict"
    assert json.loads(await tool.ainvoke({"reason": "询问群友"}))["reason_code"] == "task_conflict"
    model = SimpleNamespace(ainvoke=AsyncMock())
    sent = []

    async def sender(platform, bot_id, scene, text, reply_to):
        sent.append((platform, bot_id, scene, text, reply_to))
        return f"receipt-{len(sent)}"

    kwargs = {"spaces": spaces, "connected": {bot: "qq"}, "model_factory": lambda: model, "sender": sender}
    assert await poll(**kwargs) == "completed"
    assert sent[0] == ("qq", bot, group_b, "大家好呀", None)
    assert sent[1][2] == group_a
    assert sent[1][4] == "source"
    assert "发送消息" in sent[1][3]
    assert await poll(**kwargs) == "idle"
    model.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_send_rejects_unlinked_or_unreachable_targets(cleanup_direct_tasks):
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool

    source, target, outsider = f"cross-send-test-{uuid4().hex}", uuid4().hex, uuid4().hex
    reachable = AsyncMock(return_value={source})
    tool = create_coordination_tool(config=SimpleNamespace(cross_group_spaces={"friends": [source, target]}),
        platform="qq", bot_id="bot", scene_id=source, user_id="alice", message_id="source", request_id=None,
        request_text="发个招呼", reachable_groups=reachable)
    args = {"reason": "打招呼", "operation": "send", "target_group_id": outsider, "message": "大家好"}
    assert json.loads(await tool.ainvoke(args))["reason_code"] == "group_not_allowed"
    assert json.loads(await tool.ainvoke({**args, "target_group_id": target}))["reason_code"] == "group_unavailable"
    assert json.loads(await tool.ainvoke({**args, "target_group_id": source}))["reason_code"] == "invalid_send"
    assert json.loads(await tool.ainvoke({**args, "target_group_id": target, "message": ""}))["reason_code"] == "invalid_send"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_call", [1, 2])
async def test_direct_send_with_unknown_delivery_does_not_repeat(failed_call, cleanup_direct_tasks):
    from nonebot_plugin_ai_groupmate.brain.poll import poll
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool

    source, target, bot = f"cross-send-test-{uuid4().hex}", uuid4().hex, uuid4().hex
    spaces = {"friends": [source, target]}
    tool = create_coordination_tool(config=SimpleNamespace(cross_group_spaces=spaces), platform="qq", bot_id=bot,
        scene_id=source, user_id="alice", message_id="source", request_id=None, request_text="发个消息",
        reachable_groups=AsyncMock(return_value={source, target}))
    result = json.loads(await tool.ainvoke({"reason": "打招呼", "operation": "send", "target_group_id": target, "message": "你好"}))
    assert result["ok"]
    sent = []

    async def sender(platform, bot_id, scene, text, reply_to):
        sent.append(scene)
        if len(sent) == failed_call:
            raise TimeoutError()
        return "receipt"

    kwargs = {"spaces": spaces, "connected": {bot: "qq"}, "model_factory": lambda: None, "sender": sender}
    assert await poll(**kwargs) == "delivery_unknown"
    assert await poll(**kwargs) == "idle"
    assert sent.count(target) == 1
    assert len(sent) == failed_call


@pytest.mark.asyncio
async def test_direct_send_recovers_confirmation_without_resending(monkeypatch, cleanup_direct_tasks):
    from datetime import timedelta

    from sqlalchemy import update
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain import worker
    from nonebot_plugin_ai_groupmate.model import BrainTask
    from nonebot_plugin_ai_groupmate.brain.poll import poll
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool

    source, target, bot = f"cross-send-test-{uuid4().hex}", uuid4().hex, uuid4().hex
    spaces = {"friends": [source, target]}
    tool = create_coordination_tool(config=SimpleNamespace(cross_group_spaces=spaces), platform="qq", bot_id=bot,
        scene_id=source, user_id="alice", message_id="source", request_id=None, request_text="发个消息",
        reachable_groups=AsyncMock(return_value={source, target}))
    result = json.loads(await tool.ainvoke({"reason": "打招呼", "operation": "send", "target_group_id": target, "message": "你好"}))
    task_id = result["data"]["task_id"]
    sender = AsyncMock(side_effect=["target-receipt", "source-receipt"])
    kwargs = {"spaces": spaces, "connected": {bot: "qq"}, "model_factory": lambda: None, "sender": sender}
    real_confirm = worker._confirm_direct_send
    monkeypatch.setattr(worker, "_confirm_direct_send", AsyncMock(return_value="unavailable"))
    assert await poll(**kwargs) == "unavailable"
    async with get_session() as db:
        await db.execute(update(BrainTask).where(BrainTask.task_id == task_id).values(
            lease_until=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)))
        await db.commit()
    monkeypatch.setattr(worker, "_confirm_direct_send", real_confirm)
    assert await poll(**kwargs) == "completed"
    assert [call.args[2] for call in sender.await_args_list] == [target, source]


@pytest.mark.asyncio
async def test_tool_and_poll_complete_cross_group_question():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.poll import poll
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool
    from nonebot_plugin_ai_groupmate.reply_guard import activate_request_id, deactivate_request_id
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    group_a, group_b, bot = uuid4().hex, uuid4().hex, uuid4().hex
    spaces = {"friends": [group_a, group_b]}
    config = SimpleNamespace(cross_group_spaces=spaces)
    await activate_request_id(group_a, "request")
    try:
        tool = create_coordination_tool(config=config, platform="qq", bot_id=bot, scene_id=group_a,
                                        user_id="alice", message_id="source", request_id="request",
                                        request_text=f"去 {group_b} 群问几点出发")
        result = json.loads(await tool.ainvoke({"reason": "询问出发时间"}))
        assert result["reason_code"] == "coordination_queued"
        repeated = json.loads(await tool.ainvoke({"reason": "询问出发时间"}))
        assert repeated["data"]["task_id"] == result["data"]["task_id"]
        assert not repeated["data"]["created"]
    finally:
        await deactivate_request_id(group_a, "request")

    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"kind": "ask", "target_scene": group_b, "text": "几点出发？"}),
    )))
    sent = []

    async def sender(platform, bot_id, scene, text, reply_to):
        sent.append((platform, bot_id, scene, text, reply_to))
        return f"receipt-{len(sent)}"

    kwargs = {"spaces": spaces, "connected": {bot: "qq"}, "model_factory": lambda: model, "sender": sender}
    assert await poll(**kwargs) == "waiting_input"
    async with get_session() as db:
        assert await record_message(db, platform="qq", bot_id=bot, scene_id=group_b, message_id="unrelated",
                                    user_id="bob", text="哈哈", reply_to="receipt-1",
                                    now=datetime.now(timezone.utc).replace(tzinfo=None))
        await db.commit()
    model.ainvoke.return_value.content = '{"result":"unrelated"}'
    assert await poll(**kwargs) == "unrelated"
    assert await poll(**kwargs) == "idle"
    async with get_session() as db:
        assert await record_message(db, platform="qq", bot_id=bot, scene_id=group_b, message_id="reply",
                                    user_id="bob", text="八点出发", reply_to="receipt-1",
                                    now=datetime.now(timezone.utc).replace(tzinfo=None))
        await db.commit()
    model.ainvoke.return_value.content = '{"result":"answer"}'
    assert await poll(**kwargs) == "answer"
    assert await poll(**kwargs) == "completed"
    assert sent[0] == ("qq", bot, group_b, "几点出发？", None)
    assert sent[1] == ("qq", bot, group_a, f"{group_b} 群回复：八点出发", "source")
    assert await poll(**kwargs) == "idle"


@pytest.mark.asyncio
async def test_poll_sends_timeout_notice_for_expired_queued_task():
    from datetime import timedelta

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask
    from nonebot_plugin_ai_groupmate.brain.poll import poll
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    group_a, group_b, bot = uuid4().hex, uuid4().hex, uuid4().hex
    async with get_session() as db:
        task, _ = await submit(db, TaskInput(bot, "qq", group_a, "source", "alice", "询问",
                                             now - timedelta(minutes=1)), now=now - timedelta(minutes=2))
        task_id = task.task_id
        await db.commit()
    sender = AsyncMock(return_value="notice")
    model = SimpleNamespace(ainvoke=AsyncMock())
    result = await poll(spaces={"friends": [group_a, group_b]}, connected={bot: "qq"},
                        model_factory=lambda: model, sender=sender)
    assert result == "expired"
    sender.assert_awaited_once()
    model.ainvoke.assert_not_awaited()
    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        assert task is not None
        assert task.status == "expired"


@pytest.mark.asyncio
async def test_unavailable_main_model_notifies_source_without_sending_to_target():
    from datetime import timedelta

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask
    from nonebot_plugin_ai_groupmate.brain.poll import poll
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    group_a, group_b, bot = uuid4().hex, uuid4().hex, uuid4().hex
    async with get_session() as db:
        task, _ = await submit(db, TaskInput(bot, "qq", group_a, "source", "alice", "询问", now + timedelta(minutes=30)), now=now)
        task_id = task.task_id
        await db.commit()

    def missing_model():
        raise RuntimeError("unavailable")

    sender = AsyncMock(return_value="notice")
    kwargs = {"spaces": {"friends": [group_a, group_b]}, "connected": {bot: "qq"},
              "model_factory": missing_model, "sender": sender}
    assert await poll(**kwargs) == "finalizing"
    sender.assert_not_awaited()
    assert await poll(**kwargs) == "failed"
    assert sender.await_args is not None
    assert sender.await_args.args[2] == group_a
    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        assert task is not None
        assert task.status == "failed"


@pytest.mark.asyncio
async def test_agent_exposes_coordination_only_for_addressed_group_requests(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    group, peer = uuid4().hex, uuid4().hex
    captured = []
    monkeypatch.setattr(agent.plugin_config, "cross_group_spaces", {"friends": [group, peer]})
    monkeypatch.setattr(agent.plugin_config, "agent_file_tools_enabled", False)
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context", "relevant_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "load_group_style_context", AsyncMock(return_value=""))

    def build(_model, _tools, _prompt, **kwargs):
        captured.extend(item.name for item in kwargs["base_tools"])
        return object()

    monkeypatch.setattr(agent, "build_chat_graph", build)
    bot = SimpleNamespace(self_id="bot", adapter=SimpleNamespace(get_name=lambda: "qq"))
    event = SimpleNamespace(message_id="source", is_tome=lambda: True, get_plaintext=lambda: "去另一个群问问")
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), group, "request", "alice", "Alice",
                                  history=[], bot_id="bot", bot=bot, event=event, group_members=[])  # pyright: ignore[reportArgumentType]
    assert "request_cross_group_coordination" in captured
    captured.clear()
    event.is_tome = lambda: False
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), group, "request", "alice", "Alice",
                                  history=[], bot_id="bot", bot=bot, event=event, group_members=[])  # pyright: ignore[reportArgumentType]
    assert "request_cross_group_coordination" not in captured


@pytest.mark.asyncio
async def test_plugin_scheduler_sends_and_records_both_sides(monkeypatch, tmp_path):
    from sqlalchemy import select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import BrainTask, ChatHistory
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool
    from nonebot_plugin_ai_groupmate.reply_guard import activate_request_id, deactivate_request_id
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    group_a, group_b, bot_id = uuid4().hex, uuid4().hex, uuid4().hex
    spaces = {"friends": [group_a, group_b]}
    monkeypatch.setattr(plugin.plugin_config, "cross_group_spaces", spaces)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    fake_bot = SimpleNamespace(self_id=bot_id, adapter=SimpleNamespace(get_name=lambda: "qq"))
    monkeypatch.setattr(plugin, "get_bots", lambda: {bot_id: fake_bot})
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"kind": "ask", "target_scene": group_b, "text": "几点出发？"}),
    )))
    monkeypatch.setattr(agent, "get_chat_model", lambda: model)
    sends = []

    async def send(message, target, bot, **_kwargs):
        sends.append((target.id, bot.self_id, str(message)))
        return SimpleNamespace(msg_ids=[{"message_id": f"receipt-{len(sends)}"}], bot=bot)

    monkeypatch.setattr(UniMessage, "send", send)
    await activate_request_id(group_a, "request")
    try:
        tool = create_coordination_tool(config=plugin.plugin_config, platform="qq", bot_id=bot_id, scene_id=group_a,
                                        user_id="alice", message_id="source", request_id="request",
                                        request_text=f"去 {group_b} 群问出发时间")
        task_id = json.loads(await tool.ainvoke({"reason": "问时间"}))["data"]["task_id"]
    finally:
        await deactivate_request_id(group_a, "request")
    await plugin._poll_cross_group_tasks()
    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        assert task is not None
        assert task.status == "waiting_input"
        assert await record_message(db, platform="qq", bot_id=bot_id, scene_id=group_b, message_id="reply",
                                    user_id="bob", text="八点", reply_to="receipt-1",
                                    now=datetime.now(timezone.utc).replace(tzinfo=None))
        await db.commit()
    model.ainvoke.return_value.content = '{"result":"answer"}'
    await plugin._poll_cross_group_tasks()
    await plugin._poll_cross_group_tasks()
    assert [(scene, bot) for scene, bot, _ in sends] == [(group_b, bot_id), (group_a, bot_id)]
    assert "八点" in sends[1][2]
    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        assert task is not None
        assert task.status == "completed"
        rows = list((await db.scalars(select(ChatHistory).where(ChatHistory.session_id.in_([group_a, group_b]), ChatHistory.content_type == "bot"))).all())
        assert len(rows) == 2
