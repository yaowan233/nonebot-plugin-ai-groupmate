import asyncio
from uuid import uuid4
from datetime import datetime, timedelta
from importlib import import_module
from dataclasses import replace

from sqlalchemy import text, inspect, create_engine
from alembic.migration import MigrationContext
from alembic.operations import Operations


def request():
    from nonebot_plugin_ai_groupmate.drawing_jobs import DrawingRequest

    return DrawingRequest(platform="qq", session_id=uuid4().hex, is_private=False, bot_id="bot",
                          bot_name="miyuki", user_id="alice", request_id="turn-1", reply_to_id="message-1",
                          fingerprint="a" * 64, prompt="画一只猫", reference_file_ids=("b" * 32,),
                          size="1024x1024", transparent_background=False, model="test-image-model",
                          quality="medium", endpoint_hash="d" * 64)


async def test_enqueue_is_idempotent_with_scoped_identity():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing

    item = request()
    async with get_session() as db:
        first = await enqueue_drawing(db, item)
        same = await enqueue_drawing(db, item)
        other = await enqueue_drawing(db, replace(item, user_id="bob"))
        await db.commit()
        assert first.job_id == same.job_id
        assert first.job_id != other.job_id
        assert first.status == "queued"
        stored = await db.get(DrawingJob, first.job_id)
        assert stored is not None
        assert stored.reference_file_ids == ["b" * 32]


async def test_crash_after_provider_or_delivery_is_never_retried():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import claim_delivery, enqueue_drawing, claim_generation, complete_generation, recover_expired_claims

    base = datetime(2026, 1, 1)
    async with get_session() as db:
        pending = await enqueue_drawing(db, request(), now=base)
        generation = await enqueue_drawing(db, request(), now=base)
        sending = await enqueue_drawing(db, request(), now=base)
        ready = await enqueue_drawing(db, request(), now=base)
        assert await claim_generation(db, generation.job_id, "one", now=base)
        assert not await claim_generation(db, generation.job_id, "another", now=base)
        assert await claim_generation(db, sending.job_id, "two", now=base)
        assert await complete_generation(db, sending.job_id, "two", "c" * 32, now=base) == "generated"
        assert await claim_generation(db, ready.job_id, "four", now=base)
        assert await complete_generation(db, ready.job_id, "four", "d" * 32, now=base) == "generated"
        assert await claim_delivery(db, sending.job_id, "three", now=base)
        await db.commit()
        counts = await recover_expired_claims(db, now=base + timedelta(minutes=4))
        await db.commit()
        assert counts == (1, 1)
        pending_row = await db.get(DrawingJob, pending.job_id, populate_existing=True)
        generation_row = await db.get(DrawingJob, generation.job_id, populate_existing=True)
        sending_row = await db.get(DrawingJob, sending.job_id, populate_existing=True)
        ready_row = await db.get(DrawingJob, ready.job_id, populate_existing=True)
        assert pending_row is not None
        assert generation_row is not None
        assert sending_row is not None
        assert ready_row is not None
        assert pending_row.status == "queued"
        assert generation_row.status == "interrupted_notice_pending"
        assert sending_row.status == "delivery_unknown"
        assert ready_row.status == "generated"
        assert not await claim_generation(db, generation.job_id, "retry", now=base)
        assert not await claim_delivery(db, sending.job_id, "retry", now=base)
        assert await claim_delivery(db, ready.job_id, "safe-resume", now=base + timedelta(minutes=4))


async def test_crash_during_failure_notice_never_resends_it():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing, fail_generation, claim_generation, claim_failure_notice, recover_expired_claims

    base = datetime(2026, 1, 1)
    async with get_session() as db:
        job = await enqueue_drawing(db, request(), now=base)
        assert await claim_generation(db, job.job_id, "generation", now=base)
        assert await fail_generation(db, job.job_id, "generation", "reference_unavailable", outcome_unknown=False, now=base)
        assert await claim_failure_notice(db, job.job_id, "notice", now=base)
        await db.commit()
        await recover_expired_claims(db, now=base + timedelta(minutes=4))
        await db.commit()
        stored = await db.get(DrawingJob, job.job_id, populate_existing=True)
        assert stored is not None
        assert stored.status == "failed_notice_unknown"
        assert stored.error_code == "reference_unavailable"
        assert not await claim_failure_notice(db, job.job_id, "repeat", now=base + timedelta(minutes=4))


async def test_cancelled_generation_does_not_queue_failure_notice():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import cancel_drawing, enqueue_drawing, fail_generation, claim_generation, recover_expired_claims

    base = datetime(2026, 1, 1)
    item = request()
    async with get_session() as db:
        first = await enqueue_drawing(db, item, now=base)
        second = await enqueue_drawing(db, replace(item, request_id="second"), now=base)
        for job in (first, second):
            assert await claim_generation(db, job.job_id, "generation", now=base)
            assert await cancel_drawing(db, job.job_id, platform=item.platform, session_id=item.session_id,
                                        is_private=item.is_private, user_id=item.user_id, now=base) == "cancel_requested"
        assert await fail_generation(db, first.job_id, "generation", "generation_outcome_unknown", now=base)
        await recover_expired_claims(db, now=base + timedelta(minutes=4))
        await db.commit()
        for job in (first, second):
            stored = await db.get(DrawingJob, job.job_id, populate_existing=True)
            assert stored is not None
            assert stored.status == "cancelled"
            assert stored.error_code == "generation_outcome_unknown"


async def test_owner_cancellation_prevents_delivery_of_inflight_result():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import cancel_drawing, claim_delivery, enqueue_drawing, claim_generation, complete_generation

    item = request()
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        assert await claim_generation(db, job.job_id, "owner")
        assert await cancel_drawing(db, job.job_id, platform="qq", session_id=item.session_id,
                                    is_private=False, user_id="bob") is None
        assert await cancel_drawing(db, job.job_id, platform="qq", session_id=item.session_id,
                                    is_private=False, user_id="alice") == "cancel_requested"
        assert await complete_generation(db, job.job_id, "owner", "c" * 32) == "cancelled"
        assert not await claim_delivery(db, job.job_id, "sender")
        await db.commit()
        stored = await db.get(DrawingJob, job.job_id, populate_existing=True)
        assert stored is not None
        assert stored.status == "cancelled"


async def test_delivery_claim_owner_and_terminal_completion():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import claim_delivery, enqueue_drawing, finish_delivery, claim_generation, complete_generation

    async with get_session() as db:
        job = await enqueue_drawing(db, request())
        assert await claim_generation(db, job.job_id, "generate")
        assert await complete_generation(db, job.job_id, "wrong", "c" * 32) is None
        assert await complete_generation(db, job.job_id, "generate", "c" * 32) == "generated"
        assert await claim_delivery(db, job.job_id, "deliver")
        assert not await claim_delivery(db, job.job_id, "second")
        assert not await finish_delivery(db, job.job_id, "wrong", message_id="sent")
        assert await finish_delivery(db, job.job_id, "deliver", message_id="sent")
        await db.commit()
        stored = await db.get(DrawingJob, job.job_id, populate_existing=True)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.result_message_id == "sent"


def test_drawing_migration_preserves_existing_data():
    migration = import_module("nonebot_plugin_ai_groupmate.migrations.a4d8f0c2b615_add_drawing_jobs")
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE old_messages (id INTEGER PRIMARY KEY, content TEXT)"))
        connection.execute(text("INSERT INTO old_messages VALUES (1, '保留')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert "nonebot_plugin_ai_groupmate_drawingjob" in inspect(connection).get_table_names()
            migration.downgrade()
        assert connection.execute(text("SELECT content FROM old_messages")).scalar_one() == "保留"


async def test_worker_claims_once_and_delivers_once():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_delivery, run_generation

    async with get_session() as db:
        job = await enqueue_drawing(db, request())
        await db.commit()
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def generate(task):
        calls.append(task.job_id)
        started.set()
        await finish.wait()
        return "c" * 32
    first = asyncio.create_task(run_generation(job.job_id, generate))
    await asyncio.wait_for(started.wait(), timeout=2)
    assert not await run_generation(job.job_id, generate)
    finish.set()
    assert await first
    delivered = []
    async def deliver(task):
        delivered.append((task.result_file_id, task.reply_to_id))
        return "sent-123"
    assert await run_delivery(job.job_id, deliver)
    assert not await run_delivery(job.job_id, deliver)
    assert calls == [job.job_id]
    assert delivered == [("c" * 32, "message-1")]
    async with get_session() as db:
        stored = await db.get(DrawingJob, job.job_id)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.result_message_id == "sent-123"


async def test_cancelled_worker_never_restarts_provider():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation

    async with get_session() as db:
        job = await enqueue_drawing(db, request())
        await db.commit()
    started = asyncio.Event()
    calls = []
    async def generate(task):
        calls.append(task.job_id)
        started.set()
        await asyncio.Event().wait()
        return "c" * 32
    task = asyncio.create_task(run_generation(job.job_id, generate))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert not await run_generation(job.job_id, generate)
    assert calls == [job.job_id]
    async with get_session() as db:
        stored = await db.get(DrawingJob, job.job_id)
        assert stored is not None
        assert stored.status == "interrupted_notice_pending"


async def test_delivery_exception_does_not_repeat_platform_send():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_delivery, run_generation

    async with get_session() as db:
        job = await enqueue_drawing(db, request())
        await db.commit()
    async def generate(task):
        return "c" * 32
    assert await run_generation(job.job_id, generate)
    calls = []
    async def uncertain_send(task):
        calls.append(task.job_id)
        raise TimeoutError("adapter outcome unknown")
    assert not await run_delivery(job.job_id, uncertain_send)
    assert not await run_delivery(job.job_id, uncertain_send)
    assert calls == [job.job_id]
    async with get_session() as db:
        stored = await db.get(DrawingJob, job.job_id)
        assert stored is not None
        assert stored.status == "delivery_unknown"
