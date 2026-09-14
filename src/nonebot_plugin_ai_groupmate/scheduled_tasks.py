"""Database is the task queue; only currently executing coroutines live in memory."""

import uuid
import asyncio
from typing import Any, cast
from datetime import datetime, timezone, timedelta

from nonebot import logger, get_bots
from sqlalchemy import CursorResult, or_, select, update
from nonebot_plugin_orm import get_session

from .model import ScheduledTask

MISFIRE_GRACE_SECONDS = 300
LEASE_SECONDS = 180
HEARTBEAT_SECONDS = 30
MAX_RUNNING_TASKS = 20
MAINTENANCE_BATCH_SIZE = 100

# Execution handles only: no pending task content, timing or state is cached here.
_running_tasks: dict[str, asyncio.Task[None]] = {}
_stopping = False


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def execute_scheduled_task(task: ScheduledTask) -> None:
    """Resolve executable code at run time, so restarting needs no saved closures."""
    if task.task_type == "message":
        from .agent.schedule_tools import send_scheduled_text

        await send_scheduled_text(
            task.session_id, task.content, is_private=task.is_private,
            bot_id=task.bot_id, bot_name=task.bot_name,
        )
    elif task.task_type == "agent":
        from .agent import _run_scheduled_agent_task

        await _run_scheduled_agent_task(
            task.session_id, task.content, is_private=task.is_private, bot_id=task.bot_id,
        )
    else:
        raise ValueError(f"Unknown scheduled task type: {task.task_type}")


async def _claim_task(job_id: str, token: str) -> ScheduledTask | None:
    now = utcnow()
    bot_ids = list(get_bots())
    if not bot_ids:
        return None
    async with get_session() as session:
        result = await session.execute(
            update(ScheduledTask).where(
                ScheduledTask.job_id == job_id,
                ScheduledTask.status == "scheduled",
                ScheduledTask.run_at <= now,
                ScheduledTask.run_at >= now - timedelta(seconds=MISFIRE_GRACE_SECONDS),
                or_(ScheduledTask.bot_id.in_(bot_ids), ScheduledTask.bot_id.is_(None)),
            ).values(
                status="running", claim_token=token, started_at=now, updated_at=now,
                lease_until=now + timedelta(seconds=LEASE_SECONDS),
            )
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            await session.rollback()
            return None
        task = await session.get(ScheduledTask, job_id)
        # Release the transaction before sending messages or waiting on a model.
        if task is not None:
            session.expunge(task)
        await session.commit()
        return task


async def _heartbeat(job_id: str, token: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        now = utcnow()
        async with get_session() as session:
            result = await session.execute(
                update(ScheduledTask).where(
                    ScheduledTask.job_id == job_id,
                    ScheduledTask.status == "running",
                    ScheduledTask.claim_token == token,
                ).values(lease_until=now + timedelta(seconds=LEASE_SECONDS), updated_at=now)
            )
            await session.commit()
            if cast(CursorResult[Any], result).rowcount != 1:
                raise RuntimeError("Scheduled task claim lost")


async def _finish_task(job_id: str, token: str, status: str, error: str | None) -> None:
    now = utcnow()
    async with get_session() as session:
        await session.execute(
            update(ScheduledTask).where(
                ScheduledTask.job_id == job_id,
                ScheduledTask.status == "running",
                ScheduledTask.claim_token == token,
            ).values(status=status, error=error, finished_at=now, updated_at=now, lease_until=None)
        )
        await session.commit()


async def run_scheduled_task(job_id: str) -> None:
    token = uuid.uuid4().hex
    task = await _claim_task(job_id, token)
    if task is None:
        return
    status, error = "completed", None
    execution = asyncio.create_task(execute_scheduled_task(task))
    heartbeat = asyncio.create_task(_heartbeat(job_id, token))
    try:
        done, _ = await asyncio.wait({execution, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
        if heartbeat in done:
            # Stop execution if ownership can no longer be maintained.
            await heartbeat
        await execution
    except asyncio.CancelledError:
        status, error = "interrupted", "执行被中断，结果可能已部分送达；为避免重复发送，不自动重试。"
        raise
    except Exception as exc:
        status, error = "failed", f"执行失败（{type(exc).__name__}），不自动重试。"
        logger.exception(f"[定时任务] 执行失败 job_id={job_id}")
    finally:
        execution.cancel()
        heartbeat.cancel()
        await asyncio.gather(execution, heartbeat, return_exceptions=True)
        await _finish_task(job_id, token, status, error)


def _execution_done(job_id: str, task: asyncio.Task[None]) -> None:
    _running_tasks.pop(job_id, None)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error(f"[定时任务] 数据库执行状态更新失败 job_id={job_id}, error_type={type(error).__name__}")


async def poll_scheduled_tasks() -> None:
    """Called periodically after startup; due rows survive restarts automatically."""
    if _stopping:
        return
    now = utcnow()
    missed_filter = (
        ScheduledTask.status == "scheduled",
        ScheduledTask.run_at < now - timedelta(seconds=MISFIRE_GRACE_SECONDS),
    )
    interrupted_filter = (
        ScheduledTask.status == "running", ScheduledTask.lease_until < now,
    )
    # Even an UPDATE matching zero rows takes SQLite's writer lock. Discover
    # maintenance candidates with reads, and release that transaction first.
    async with get_session() as session:
        missed_ids = list((await session.scalars(
            select(ScheduledTask.job_id).where(*missed_filter)
            .order_by(ScheduledTask.run_at, ScheduledTask.job_id).limit(MAINTENANCE_BATCH_SIZE)
        )).all())
        interrupted_ids = list((await session.scalars(
            select(ScheduledTask.job_id).where(*interrupted_filter)
            .order_by(ScheduledTask.lease_until, ScheduledTask.job_id).limit(MAINTENANCE_BATCH_SIZE)
        )).all())
        capacity = MAX_RUNNING_TASKS - len(_running_tasks)
        bot_ids = list(get_bots())
        job_ids = []
        if capacity > 0 and bot_ids:
            job_ids = list((await session.scalars(
                select(ScheduledTask.job_id).where(
                    ScheduledTask.status == "scheduled",
                    ScheduledTask.run_at <= now,
                    ScheduledTask.run_at >= now - timedelta(seconds=MISFIRE_GRACE_SECONDS),
                    or_(ScheduledTask.bot_id.in_(bot_ids), ScheduledTask.bot_id.is_(None)),
                    ScheduledTask.job_id.not_in(list(_running_tasks)),
                ).order_by(ScheduledTask.run_at, ScheduledTask.job_id).limit(capacity)
            )).all())
    if missed_ids or interrupted_ids:
        async with get_session() as session:
            # Recheck state/time: a task may have been edited or renewed since
            # discovery. Bound each maintenance write instead of locking all rows.
            if missed_ids:
                await session.execute(
                    update(ScheduledTask).where(
                        ScheduledTask.job_id.in_(missed_ids), *missed_filter,
                    ).values(status="missed", finished_at=now, updated_at=now, error="超过 5 分钟执行宽限期，未执行。")
                )
            if interrupted_ids:
                await session.execute(
                    update(ScheduledTask).where(
                        ScheduledTask.job_id.in_(interrupted_ids), *interrupted_filter,
                    ).values(
                        status="interrupted", finished_at=now, updated_at=now, lease_until=None,
                        error="执行进程中断或失去数据库连接，结果未知；为避免重复发送，不自动重试。",
                    )
                )
            await session.commit()
    # Shutdown may have begun while the database query was in flight.
    if _stopping:
        return
    for job_id in job_ids:
        task = asyncio.create_task(run_scheduled_task(job_id))
        _running_tasks[job_id] = task
        task.add_done_callback(lambda task, job_id=job_id: _execution_done(job_id, task))


async def stop_scheduled_tasks() -> None:
    global _stopping
    _stopping = True
    tasks = list(_running_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _running_tasks.clear()
