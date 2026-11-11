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
    from nonebot_plugin_ai_groupmate.brain.actions import ActionSpec, confirm, prepare, claim_send
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
        action, _ = await prepare(db, ActionSpec(task_id, "ask", "message", "qq", "B", "来自群 A的跨群提问；引用这条消息回复会转告：几点出发？"), task_token="worker", now=now)
        assert await claim_send(db, action.action_id, task_token="worker", send_token="sender", now=now)
        assert await confirm(db, action.action_id, send_token="sender", message_id="question", now=now)
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
    from sqlalchemy import select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainAction
    from nonebot_plugin_ai_groupmate.brain.actions import confirm, claim_send
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, submit, expire_wait, accept_reply, finish_expired, wait_for_reply, claim_finalization

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
        action = await db.scalar(select(BrainAction).where(BrainAction.task_id == task_id))
        assert action is not None
        assert action.target_scene == item.source_scene
        assert await claim_finalization(db, task_id, "finalizer", now=deadline)
        assert not await finish_expired(db, task_id, "finalizer", now=deadline)
        assert await claim_send(db, action.action_id, task_token="finalizer", send_token="sender", now=deadline)
        assert await confirm(db, action.action_id, send_token="sender", message_id="timeout-notice", now=deadline)
        assert await finish_expired(db, task_id, "finalizer", now=deadline)
        await db.commit()


@pytest.mark.asyncio
async def test_reply_recorded_before_wait_registration_is_found_without_crossing_scope():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.actions import ActionSpec, confirm, prepare, claim_send
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, submit, wait_for_reply, pending_reply_candidates
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 12)
    item = TaskInput("shared-brain", "qq", uuid4().hex, "m1", "alice", "问 B 群", now + timedelta(hours=1), transport_bot_id="bot")
    async with get_session() as db:
        task, _ = await submit(db, item, now=now)
        task_id = task.task_id
        assert await claim(db, task_id, "worker", now=now)
        action, _ = await prepare(db, ActionSpec(task_id, "ask", "message", "qq", "B", "来自群 A的跨群提问；引用这条消息回复会转告：几点出发？"), task_token="worker", now=now)
        assert await claim_send(db, action.action_id, task_token="worker", send_token="sender", now=now)
        assert await confirm(db, action.action_id, send_token="sender", message_id="question", now=now)
        for scene, bot, message in [("B", "bot", "good"), ("C", "bot", "wrong-group"), ("B", "other-bot", "wrong-bot")]:
            assert await record_message(db, platform="qq", bot_id=bot, scene_id=scene, message_id=message,
                                        user_id="bob", text="八点", reply_to="question", now=now)
        await db.commit()
    async with get_session() as db:
        assert await wait_for_reply(db, task_id, "worker", platform="qq", target_scene="B", question_message="question",
                                    deadline=now + timedelta(minutes=30), now=now)
        await db.commit()
    async with get_session() as db:
        candidates = await pending_reply_candidates(db, task_id=task_id, now=now)
        assert [event.message_id for event in candidates] == ["good"]
        assert candidates[0].text == "八点"


@pytest.mark.asyncio
async def test_old_undisclosed_question_cannot_relay_a_reply():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.actions import ActionSpec, confirm, prepare, claim_send
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, claim, submit, accept_reply, wait_for_reply, pending_reply_candidates
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 12)
    source, target, bot = (uuid4().hex for _ in range(3))
    async with get_session() as db:
        task, _ = await submit(db, TaskInput(bot, "qq", source, "m", "alice", "问 B", now + timedelta(hours=1)), now=now)
        assert await claim(db, task.task_id, "worker", now=now)
        action, _ = await prepare(db, ActionSpec(task.task_id, "ask", "message", "qq", target, "几点出发？"), task_token="worker", now=now)
        assert await claim_send(db, action.action_id, task_token="worker", send_token="send", now=now)
        assert await confirm(db, action.action_id, send_token="send", message_id="old-question", now=now)
        assert await wait_for_reply(db, task.task_id, "worker", platform="qq", target_scene=target,
                                    question_message="old-question", deadline=now + timedelta(minutes=30), now=now)
        assert await record_message(db, platform="qq", bot_id=bot, scene_id=target, message_id="reply", user_id="bob",
                                    text="八点", reply_to="old-question", now=now)
        await db.flush()
        assert await pending_reply_candidates(db, task_id=task.task_id, now=now) == []
        assert not await accept_reply(db, task.task_id, platform="qq", target_scene=target, reply_to="old-question",
                                      message_id="reply", user_id="bob", text="八点", now=now)
        await db.rollback()


@pytest.mark.asyncio
async def test_task_submission_rolls_back_before_acknowledgement():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit

    now = datetime(2026, 11, 10, 12)
    item = TaskInput("bot", "qq", uuid4().hex, "source", "alice", "询问", now + timedelta(hours=1))
    async with get_session() as db:
        task, _ = await submit(db, item, now=now)
        task_id = task.task_id
        await db.rollback()
    async with get_session() as db:
        assert await db.get(BrainTask, task_id) is None


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
