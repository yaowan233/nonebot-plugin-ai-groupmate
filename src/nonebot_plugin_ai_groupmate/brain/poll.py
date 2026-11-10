"""Poll durable cross-group tasks without holding a DB session during I/O."""

import asyncio
from collections.abc import Callable, Awaitable

from sqlalchemy import or_, func, exists, select
from nonebot_plugin_orm import get_session

from ..model import BrainTask, BrainWait, BrainCandidateDecision, BrainObservationMessage
from .worker import Sender, Spaces, utcnow, run_task, run_finalization, process_candidate
from .actions import mark_unknown
from .repository import expire_wait, expire_queued, recover_expired_claims, pending_reply_candidates

_lock = asyncio.Lock()


class _LazyModel:
    def __init__(self, factory: Callable):
        self.factory = factory

    async def ainvoke(self, messages):
        return await self.factory().ainvoke(messages)


async def poll(*, spaces: Spaces, connected: dict[str, str], model_factory: Callable, sender: Sender,
               target_names: dict[str, str] | None = None,
               name_resolver: Callable[[str, str, tuple[str, ...]], Awaitable[dict[str, str]]] | None = None) -> str:
    """Advance at most one runnable task per call (single active process)."""
    if _lock.locked():
        return "busy"
    async with _lock:
        now = utcnow()
        async with get_session() as db:
            await mark_unknown(db, now=now)
            await recover_expired_claims(db, now=now)
            expired = list((await db.scalars(select(BrainWait.task_id).where(
                BrainWait.status == "open", BrainWait.deadline <= now,
            ).order_by(BrainWait.deadline).limit(20))).all())
            for task_id in expired:
                await expire_wait(db, task_id, now=now)
            timed_out = list((await db.scalars(select(BrainTask.task_id).where(
                BrainTask.status.in_(("queued", "running")), BrainTask.deadline <= now,
            ).order_by(BrainTask.deadline).limit(20))).all())
            for task_id in timed_out:
                await expire_queued(db, task_id, now=now)
            await db.commit()
        async with get_session() as db:
            decisions = select(func.count()).select_from(BrainCandidateDecision).where(
                BrainCandidateDecision.task_id == BrainTask.task_id,
            ).correlate(BrainTask).scalar_subquery()
            pending = exists(select(BrainObservationMessage.id).join(
                BrainWait, (BrainObservationMessage.platform == BrainWait.platform)
                & (BrainObservationMessage.scene_id == BrainWait.target_scene)
                & or_(BrainObservationMessage.reply_to == BrainWait.question_message,
                      (BrainObservationMessage.reply_to.is_(None)) & (BrainObservationMessage.addressed_other.is_(False))
                      & (BrainObservationMessage.created_at >= BrainWait.unquoted_from)
                      & (BrainObservationMessage.created_at <= BrainWait.unquoted_until)),
            ).where(
                BrainWait.task_id == BrainTask.task_id, BrainWait.status == "open", BrainWait.deadline > now,
                BrainObservationMessage.bot_id == BrainTask.bot_id,
                BrainObservationMessage.created_at >= BrainTask.created_at,
                BrainObservationMessage.created_at <= now,
                ~exists(select(BrainCandidateDecision.task_id).where(
                    BrainCandidateDecision.task_id == BrainTask.task_id,
                    BrainCandidateDecision.observation_id == BrainObservationMessage.id,
                )),
            ))
            tasks = list((await db.scalars(select(BrainTask).where(
                BrainTask.bot_id.in_(list(connected)),
                (BrainTask.status == "finalizing") | ((BrainTask.status == "queued") & (BrainTask.deadline > now))
                | ((BrainTask.status == "waiting_input") & (BrainTask.deadline > now) & (decisions < 5) & pending),
            ).order_by(BrainTask.updated_at, BrainTask.task_id).limit(100))).all())
            chosen = None
            for task in tasks:
                if connected.get(task.bot_id) != task.platform:
                    continue
                if task.status == "waiting_input" and not await pending_reply_candidates(db, task_id=task.task_id, now=now):
                    continue
                if task.status == "finalizing" and task.claim_token and task.lease_until and task.lease_until > now:
                    continue
                chosen = task.task_id, task.status
                break
        if chosen is None:
            return "idle"
        task_id, status = chosen
        if status == "waiting_input":
            return await process_candidate(task_id, spaces=spaces, model=_LazyModel(model_factory))
        if status == "finalizing":
            return await run_finalization(task_id, spaces=spaces, sender=sender)
        return await run_task(task_id, spaces=spaces, model=_LazyModel(model_factory), sender=sender, target_names=target_names,
                              name_resolver=name_resolver)
