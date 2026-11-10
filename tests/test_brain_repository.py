from uuid import uuid4
from datetime import datetime, timedelta
from importlib import import_module
from dataclasses import replace

import pytest


def test_brain_migration_creates_and_removes_tables(monkeypatch):
    from sqlalchemy import inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = import_module("nonebot_plugin_ai_groupmate.migrations.a8b9c0d1e2f3_add_brain_tasks")
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(conn)))
        migration.upgrade()
        tables = inspect(conn).get_table_names()
        assert "nonebot_plugin_ai_groupmate_braintask" in tables
        assert "nonebot_plugin_ai_groupmate_brainwait" in tables
        migration.downgrade()
        assert "nonebot_plugin_ai_groupmate_braintask" not in inspect(conn).get_table_names()
    engine.dispose()


@pytest.mark.asyncio
async def test_task_keeps_scope_across_restarts_and_only_accepts_exact_reply():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask, BrainWait
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, finish, submit, accept_reply, matching_waits, wait_for_reply

    now = datetime(2026, 11, 10, 12)
    item = TaskInput("bot", "qq", uuid4().hex, "message-1", "alice", "问 B 群几点出发", now + timedelta(hours=1))
    async with get_session() as db:
        task, created = await submit(db, item, now=now)
        assert created
        task_id = task.task_id
        duplicate, created = await submit(db, item, now=now)
        assert not created
        assert duplicate.task_id == task_id
        distinct, created = await submit(db, replace(item, source_scene=uuid4().hex), now=now)
        assert created
        assert distinct.task_id != task_id
        assert await claim(db, task_id, "worker", now=now)
        assert not await claim(db, task_id, "other", now=now)
        assert await wait_for_reply(db, task_id, "worker", platform="qq", target_scene="B", question_message="question",
                                    deadline=now + timedelta(minutes=30), now=now)
        await db.commit()

    async with get_session() as db:
        assert await matching_waits(db, platform="qq", target_scene="B", reply_to="question", now=now) == [task_id]
        assert await matching_waits(db, platform="qq", target_scene="C", reply_to="question", now=now) == []
        assert not await accept_reply(db, task_id, platform="qq", target_scene="C", reply_to="question",
                                      message_id="reply", user_id="bob", text="八点", now=now)
        assert not await accept_reply(db, task_id, platform="qq", target_scene="B", reply_to="other",
                                      message_id="reply", user_id="bob", text="八点", now=now)
        assert await accept_reply(db, task_id, platform="qq", target_scene="B", reply_to="question",
                                  message_id="reply", user_id="bob", text="八点", now=now)
        assert not await accept_reply(db, task_id, platform="qq", target_scene="B", reply_to="question",
                                      message_id="reply", user_id="bob", text="八点", now=now)
        await db.commit()
    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        waiting = await db.get(BrainWait, task_id)
        assert task is not None
        assert task.status == "queued"
        assert waiting is not None
        assert waiting.answer_text == "八点"
        assert waiting.status == "answered"
        assert await claim(db, task_id, "continuation", now=now)
        assert await finish(db, task_id, "continuation", now=now)
        await db.commit()


@pytest.mark.asyncio
async def test_timeout_and_reply_are_mutually_exclusive():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, submit, expire_wait, accept_reply, wait_for_reply

    now = datetime(2026, 11, 10, 12)
    deadline = now + timedelta(minutes=30)
    item = TaskInput("bot", "qq", uuid4().hex, "message-1", "alice", "问 B 群", now + timedelta(hours=1))
    async with get_session() as db:
        task, _ = await submit(db, item, now=now)
        task_id = task.task_id
        assert await claim(db, task_id, "worker", now=now)
        assert await wait_for_reply(db, task_id, "worker", platform="qq", target_scene="B", question_message="q", deadline=deadline, now=now)
        assert not await expire_wait(db, task_id, now=now)
        assert await expire_wait(db, task_id, now=deadline)
        assert not await accept_reply(db, task_id, platform="qq", target_scene="B", reply_to="q", message_id="r", user_id="bob", text="八点", now=deadline)
        assert not await expire_wait(db, task_id, now=deadline)
        await db.commit()


@pytest.mark.asyncio
async def test_expired_lease_and_cancellation_fence_old_worker():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, cancel, submit, wait_for_reply, recover_expired_claims

    now = datetime(2026, 11, 10, 12)
    item = TaskInput("bot", "qq", uuid4().hex, "message-1", "alice", "问 B 群", now + timedelta(hours=1))
    async with get_session() as db:
        task, _ = await submit(db, item, now=now)
        task_id = task.task_id
        assert await claim(db, task_id, "old", now=now, lease_seconds=10)
        assert await recover_expired_claims(db, now=now + timedelta(seconds=11)) == 1
        assert await claim(db, task_id, "new", now=now + timedelta(seconds=11))
        assert not await wait_for_reply(db, task_id, "old", platform="qq", target_scene="B", question_message="q",
                                        deadline=now + timedelta(minutes=1), now=now + timedelta(seconds=11))
        assert not await cancel(db, task_id, brain_id="bot", platform="qq", source_scene="other", source_user="alice", now=now)
        assert not await cancel(db, task_id, brain_id="bot", platform="qq", source_scene=item.source_scene, source_user="bob", now=now)
        assert await cancel(db, task_id, brain_id="bot", platform="qq", source_scene=item.source_scene, source_user="alice", now=now)
        assert not await wait_for_reply(db, task_id, "new", platform="qq", target_scene="B", question_message="q",
                                        deadline=now + timedelta(minutes=1), now=now + timedelta(seconds=11))
        await db.commit()
