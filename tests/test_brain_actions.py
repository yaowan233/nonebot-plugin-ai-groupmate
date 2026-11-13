from uuid import uuid4
from datetime import datetime, timedelta
from dataclasses import replace

import pytest


@pytest.mark.asyncio
async def test_replanning_and_unknown_delivery_do_not_send_twice():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainAction
    from nonebot_plugin_ai_groupmate.brain.actions import ActionSpec, confirm, prepare, claim_send, mark_unknown
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, cancel, finish, submit

    now = datetime(2026, 11, 10, 12)
    task_input = TaskInput("bot", "qq", uuid4().hex, "m1", "alice", "问 B 群", now + timedelta(hours=1))
    async with get_session() as db:
        task, _ = await submit(db, task_input, now=now)
        task_id = task.task_id
        assert await claim(db, task_id, "owner", now=now)
        spec = ActionSpec(task_id, "ask", "first", "qq", "B", "几点出发？")
        action, created = await prepare(db, spec, task_token="owner", now=now)
        assert created
        action_id = action.action_id
        same, created = await prepare(db, replace(spec, content="请问几点出发？"), task_token="owner", now=now)
        assert not created
        assert same.action_id == action_id
        assert same.content == "几点出发？"
        assert not await finish(db, task_id, "owner", now=now)
        assert await claim_send(db, action_id, task_token="owner", send_token="sender", now=now, lease_seconds=10)
        await db.commit()

    async with get_session() as db:
        assert await mark_unknown(db, now=now + timedelta(seconds=11)) == 1
        assert not await claim_send(db, action_id, task_token="owner", send_token="retry", now=now + timedelta(seconds=11))
        assert await cancel(db, task_id, brain_id="bot", platform="qq", source_scene=task_input.source_scene, source_user="alice", now=now)
        assert await confirm(db, action_id, send_token="sender", message_id="platform-receipt", now=now + timedelta(seconds=12))
        await db.commit()
    async with get_session() as db:
        action = await db.get(BrainAction, action_id)
        assert action is not None
        assert action.status == "confirmed"
        assert action.receipt_message_id == "platform-receipt"


@pytest.mark.asyncio
async def test_stale_task_owner_cannot_prepare_or_start_a_send():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.actions import ActionSpec, prepare, claim_send
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, cancel, submit

    now = datetime(2026, 11, 10, 12)
    item = TaskInput("bot", "qq", uuid4().hex, "m1", "alice", "问 B 群", now + timedelta(hours=1))
    async with get_session() as db:
        task, _ = await submit(db, item, now=now)
        assert await claim(db, task.task_id, "owner", now=now)
        spec = ActionSpec(task.task_id, "ask", "first", "qq", "B", "几点出发？")
        action, _ = await prepare(db, spec, task_token="owner", now=now)
        assert await cancel(db, task.task_id, brain_id="bot", platform="qq", source_scene=item.source_scene, source_user="alice", now=now)
        assert not await claim_send(db, action.action_id, task_token="owner", send_token="sender", now=now)
        with pytest.raises(ValueError, match="lease"):
            await prepare(db, spec, task_token="owner", now=now)
        await db.commit()
