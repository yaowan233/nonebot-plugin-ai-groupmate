from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_cross_group_round_trip_survives_reloaded_sessions():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask
    from nonebot_plugin_ai_groupmate.brain.worker import run_task, process_candidate
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit, pending_reply_candidates
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 12)
    group_a, group_b = uuid4().hex, uuid4().hex
    item = TaskInput("brain", "qq", group_a, "source-message", "alice", "去 B 群问郊游时间",
                     now + timedelta(hours=1), transport_bot_id="bot")
    async with get_session() as db:
        task, _ = await submit(db, item, now=now)
        task_id = task.task_id
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=f'{{"kind":"ask","target_scene":"{group_b}","text":"郊游几点出发？"}}',
    )))
    sent = []

    async def send(platform, bot, scene, text, reply_to):
        sent.append((platform, bot, scene, text, reply_to))
        return f"receipt-{len(sent)}"

    spaces = {"friends": [group_a, group_b]}
    assert await run_task(task_id, spaces=spaces, model=model, sender=send, clock=lambda: now) == "waiting_input"
    assert sent == [("qq", "bot", group_b, "郊游几点出发？", None)]
    async with get_session() as db:
        assert await record_message(db, platform="qq", bot_id="bot", scene_id=group_b, message_id="answer",
                                    user_id="bob", text="八点出发", reply_to="receipt-1", now=now)
        candidates = await pending_reply_candidates(db, task_id=task_id, now=now)
        assert [row.message_id for row in candidates] == ["answer"]
        await db.commit()
    model.ainvoke.return_value.content = '{"result":"answer"}'
    assert await process_candidate(task_id, spaces=spaces, model=model, clock=lambda: now) == "answer"
    assert await run_task(task_id, spaces=spaces, model=model, sender=send, clock=lambda: now) == "completed"
    assert sent[1] == ("qq", "bot", group_a, f"{group_b} 群回复：八点出发", "source-message")
    assert await run_task(task_id, spaces=spaces, model=model, sender=send, clock=lambda: now) == "unavailable"
    assert len(sent) == 2
    async with get_session() as db:
        stored = await db.get(BrainTask, task_id)
        assert stored is not None
        assert stored.status == "completed"


@pytest.mark.asyncio
async def test_unknown_send_is_blocked_and_not_retried():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.worker import run_task
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit

    now = datetime(2026, 11, 10, 12)
    group_a, group_b = uuid4().hex, uuid4().hex
    async with get_session() as db:
        task, _ = await submit(db, TaskInput("bot", "qq", group_a, "message", "alice", "问 B", now + timedelta(hours=1)), now=now)
        task_id = task.task_id
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=f'{{"kind":"ask","target_scene":"{group_b}","text":"几点？"}}',
    )))
    sender = AsyncMock(side_effect=TimeoutError())
    spaces = {"friends": [group_a, group_b]}
    assert await run_task(task_id, spaces=spaces, model=model, sender=sender, clock=lambda: now) == "delivery_unknown"
    assert await run_task(task_id, spaces=spaces, model=model, sender=sender, clock=lambda: now) == "unavailable"
    sender.assert_awaited_once()


@pytest.mark.asyncio
async def test_unrelated_candidate_is_not_reclassified_on_every_poll():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.worker import run_task, process_candidate
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 12)
    group_a, group_b = uuid4().hex, uuid4().hex
    spaces = {"friends": [group_a, group_b]}
    async with get_session() as db:
        task, _ = await submit(db, TaskInput("bot", "qq", group_a, "message", "alice", "问 B", now + timedelta(hours=1)), now=now)
        task_id = task.task_id
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=f'{{"kind":"ask","target_scene":"{group_b}","text":"几点？"}}',
    )))
    assert await run_task(task_id, spaces=spaces, model=model, sender=AsyncMock(return_value="question"), clock=lambda: now) == "waiting_input"
    async with get_session() as db:
        assert await record_message(db, platform="qq", bot_id="bot", scene_id=group_b, message_id="reply",
                                    user_id="bob", text="哈哈", reply_to="question", now=now)
        await db.commit()
    model.ainvoke.return_value.content = '{"result":"unrelated"}'
    assert await process_candidate(task_id, spaces=spaces, model=model, clock=lambda: now) == "unrelated"
    assert await process_candidate(task_id, spaces=spaces, model=model, clock=lambda: now) == "unavailable"
    assert model.ainvoke.await_count == 2
