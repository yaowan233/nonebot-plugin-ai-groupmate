"""Poll durable drawing jobs without holding database connections during work."""

import asyncio
from datetime import timedelta
from pathlib import Path

from nonebot import logger, get_bots
from sqlalchemy import Select
from nonebot_plugin_orm import get_session

from .model import DrawingJob, OrderedDeliveryJob
from .config import ScopedConfig
from .drawing_jobs import utcnow, recover_expired_claims
from .drawing_runner import run_delivery, run_generation, run_failure_notice
from .drawing_service import deliver_job, generate_job, deliver_failure_notice
from .structured_message import adapter_name
from .ordered_delivery_jobs import recover_expired as recover_ordered_deliveries
from .ordered_delivery_service import run_ordered_delivery

_running: dict[str, tuple[str, asyncio.Task[bool]]] = {}
_delivery_groups: dict[str, tuple[str, str, bool]] = {}
_stopping = False
MAX_GENERATIONS = 1
MAX_DELIVERIES = 2
GROUP_DELIVERY_INTERVAL_SECONDS = 5


def start_drawing_jobs() -> None:
    global _stopping
    _stopping = False


def _finished(job_id: str, task: asyncio.Task[bool]) -> None:
    _running.pop(job_id, None)
    _delivery_groups.pop(job_id, None)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning(f"绘图任务后台执行失败 job_id={job_id} error_type={type(error).__name__}")


async def poll_drawing_jobs(root: Path, config: ScopedConfig) -> None:
    if _stopping:
        return
    async with get_session() as db:
        await recover_expired_claims(db)
        await recover_ordered_deliveries(db)
        await db.commit()
    bots = get_bots()
    if _stopping or not bots:
        return
    async with get_session() as db:
        columns = (DrawingJob.job_id, DrawingJob.status, DrawingJob.bot_id, DrawingJob.platform,
                   DrawingJob.session_id, DrawingJob.is_private)
        ready = (await db.execute(Select(*columns).where(
            DrawingJob.status.in_(("generated", "failed_notice_pending", "interrupted_notice_pending")),
            DrawingJob.batch_id.is_(None),
            DrawingJob.bot_id.in_(list(bots)),
        ).order_by(DrawingJob.created_at, DrawingJob.job_id).limit(100))).all()
        queued = (await db.execute(Select(*columns).where(
            DrawingJob.status == "queued", DrawingJob.bot_id.in_(list(bots)),
        ).order_by(DrawingJob.created_at, DrawingJob.job_id).limit(100))).all()
        batches = (await db.execute(Select(OrderedDeliveryJob.job_id, OrderedDeliveryJob.status,
            OrderedDeliveryJob.bot_id, OrderedDeliveryJob.platform, OrderedDeliveryJob.session_id,
            OrderedDeliveryJob.is_private).where(OrderedDeliveryJob.status.in_(("waiting", "failed_notice_pending")),
            OrderedDeliveryJob.bot_id.in_(list(bots))).order_by(OrderedDeliveryJob.created_at).limit(100))).all()
        batch_ids = [row[0] for row in batches]
        child_statuses: dict[str, list[str]] = {}
        if batch_ids:
            rows = (await db.execute(Select(DrawingJob.batch_id, DrawingJob.status).where(DrawingJob.batch_id.in_(batch_ids)))).all()
            for batch_id, status in rows:
                child_statuses.setdefault(batch_id, []).append(status)
        group_ids = {session_id for _, _, _, _, session_id, is_private in [*ready, *batches] if not is_private}
        recently_delivered = set()
        if group_ids:
            recent = (await db.execute(Select(DrawingJob.platform, DrawingJob.session_id).where(
                DrawingJob.session_id.in_(group_ids), DrawingJob.is_private.is_(False),
                DrawingJob.status.in_(("completed", "delivery_unknown", "failed", "interrupted",
                                       "failed_notice_unknown", "interrupted_notice_unknown")),
                DrawingJob.finished_at >= utcnow() - timedelta(seconds=GROUP_DELIVERY_INTERVAL_SECONDS),
            ))).all()
            recently_delivered = {(platform, session_id, False) for platform, session_id in recent}
            recent_batches = (await db.execute(Select(OrderedDeliveryJob.platform, OrderedDeliveryJob.session_id).where(
                OrderedDeliveryJob.session_id.in_(group_ids), OrderedDeliveryJob.is_private.is_(False),
                OrderedDeliveryJob.status.in_(("completed", "delivery_unknown", "failed")),
                OrderedDeliveryJob.finished_at >= utcnow() - timedelta(seconds=GROUP_DELIVERY_INTERVAL_SECONDS),
            ))).all()
            recently_delivered.update((platform, session_id, False) for platform, session_id in recent_batches)
    generations = sum(kind == "generation" for kind, _ in _running.values())
    deliveries = sum(kind == "delivery" for kind, _ in _running.values())
    cfg = config.model_copy(deep=True)
    for batch_id, status, bot_id, platform, session_id, is_private in batches:
        key = "batch:" + batch_id
        states = child_statuses.get(batch_id, [])
        ready_batch = status == "failed_notice_pending" or not states or all(value == "generated" for value in states) or any(
            value in {"cancelled", "failed_notice_pending", "interrupted_notice_pending", "failed", "interrupted"} for value in states)
        scope = (platform, session_id, is_private)
        if (_stopping or not ready_batch or key in _running or deliveries >= MAX_DELIVERIES or
                bot_id not in bots or adapter_name(bots[bot_id]) != platform or
                scope in _delivery_groups.values() or scope in recently_delivered):
            continue
        task = asyncio.create_task(run_ordered_delivery(batch_id, root=root))
        _running[key] = ("delivery", task)
        _delivery_groups[key] = scope
        deliveries += 1
        task.add_done_callback(lambda done, key=key: _finished(key, done))
    # Deliver completed work first, while independently limiting GPU requests.
    for job_id, status, bot_id, platform, session_id, is_private in [*ready, *queued]:
        if _stopping or job_id in _running or bot_id not in bots or adapter_name(bots[bot_id]) != platform:
            continue
        if status in {"generated", "failed_notice_pending", "interrupted_notice_pending"} and deliveries < MAX_DELIVERIES:
            scope = (platform, session_id, is_private)
            if scope in _delivery_groups.values() or scope in recently_delivered:
                continue
            task = (asyncio.create_task(run_delivery(job_id, lambda job: deliver_job(job, root=root)))
                    if status == "generated" else asyncio.create_task(run_failure_notice(job_id, deliver_failure_notice)))
            kind = "delivery"
            deliveries += 1
            _delivery_groups[job_id] = scope
        elif status == "queued" and cfg.image_generation_enabled and generations < MAX_GENERATIONS:
            task = asyncio.create_task(run_generation(job_id, lambda job: generate_job(job, root=root, config=cfg)))
            kind = "generation"
            generations += 1
        else:
            continue
        _running[job_id] = (kind, task)
        task.add_done_callback(lambda done, key=job_id: _finished(key, done))


async def stop_drawing_jobs() -> None:
    global _stopping
    _stopping = True
    tasks = [task for _, task in _running.values()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _running.clear()
    _delivery_groups.clear()
