"""Persist coordination state with explicit source and reply identities.

Callers own the transaction: commit before acknowledging input or doing I/O.
All timestamps are naive UTC, matching scheduled_tasks.
"""

from uuid import uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass

from sqlalchemy import exists, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..model import BrainTask, BrainWait, BrainAction, BrainObservationMessage


@dataclass(frozen=True)
class TaskInput:
    brain_id: str
    platform: str
    source_scene: str
    source_message: str
    source_user: str
    request: str
    deadline: datetime
    transport_bot_id: str | None = None


async def submit(db: AsyncSession, item: TaskInput, *, now: datetime) -> tuple[BrainTask, bool]:
    if not all((item.brain_id, item.platform, item.source_scene, item.source_message, item.source_user, item.request.strip())):
        raise ValueError("Task source, message identity and request are required")
    if item.deadline <= now:
        raise ValueError("Task deadline must be in the future")
    query = select(BrainTask).where(
        BrainTask.brain_id == item.brain_id, BrainTask.platform == item.platform,
        BrainTask.source_scene == item.source_scene, BrainTask.source_message == item.source_message,
    )
    existing = await db.scalar(query)
    if existing is not None:
        return existing, False
    task = BrainTask(task_id=uuid4().hex, brain_id=item.brain_id, bot_id=item.transport_bot_id or item.brain_id, platform=item.platform,
                     source_scene=item.source_scene, source_message=item.source_message, source_user=item.source_user,
                     request=item.request, status="queued", revision=0, deadline=item.deadline,
                     claim_token=None, lease_until=None, created_at=now, updated_at=now)
    try:
        async with db.begin_nested():
            db.add(task)
            await db.flush()
    except IntegrityError:
        existing = await db.scalar(query)
        if existing is None:
            raise
        return existing, False
    return task, True


async def claim(db: AsyncSession, task_id: str, token: str, *, now: datetime, lease_seconds: int = 180) -> bool:
    if not token or lease_seconds <= 0:
        raise ValueError("A claim token and positive lease duration are required")
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "queued", BrainTask.deadline > now,
    ).values(status="running", claim_token=token, lease_until=now + timedelta(seconds=lease_seconds),
             revision=BrainTask.revision + 1, updated_at=now))
    return bool(result.rowcount)


async def recover_expired_claims(db: AsyncSession, *, now: datetime) -> int:
    result = await db.execute(update(BrainTask).where(
        BrainTask.status == "running", BrainTask.lease_until <= now, BrainTask.deadline > now,
    ).values(status="queued", claim_token=None, lease_until=None, revision=BrainTask.revision + 1, updated_at=now))
    return int(result.rowcount or 0)


async def wait_for_reply(db: AsyncSession, task_id: str, token: str, *, platform: str, target_scene: str,
                         question_message: str, deadline: datetime, now: datetime) -> bool:
    if not all((platform, target_scene, question_message)) or deadline <= now:
        raise ValueError("A scoped question and future deadline are required")
    previous = await db.get(BrainWait, task_id)
    if previous is not None:
        raise ValueError("Task already has a wait; multiple-step waits are not yet supported")
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "running", BrainTask.claim_token == token,
        BrainTask.lease_until > now, BrainTask.deadline >= deadline,
    ).values(status="waiting_input", claim_token=None, lease_until=None,
             revision=BrainTask.revision + 1, updated_at=now))
    if not result.rowcount:
        return False
    db.add(BrainWait(task_id=task_id, platform=platform, target_scene=target_scene,
                     question_message=question_message, answer_message=None, answer_user=None,
                     answer_text=None, deadline=deadline, status="open"))
    await db.flush()
    return True


async def matching_waits(db: AsyncSession, *, platform: str, target_scene: str, reply_to: str, now: datetime) -> list[str]:
    """Find candidates only: matching an inquiry does not prove an answer is correct."""
    if not all((platform, target_scene, reply_to)):
        return []
    return list((await db.scalars(select(BrainWait.task_id).where(
        BrainWait.platform == platform, BrainWait.target_scene == target_scene,
        BrainWait.question_message == reply_to, BrainWait.status == "open", BrainWait.deadline > now,
    ))).all())


async def pending_reply_candidates(db: AsyncSession, *, task_id: str, now: datetime) -> list[BrainObservationMessage]:
    """Read scoped candidates, including messages stored before wait registration.

    A quoted message is evidence of association, not proof it answers the
    question. The coordinator must validate its meaning before accept_reply.
    """
    return list((await db.scalars(select(BrainObservationMessage).join(
        BrainWait, (BrainObservationMessage.platform == BrainWait.platform)
        & (BrainObservationMessage.scene_id == BrainWait.target_scene)
        & (BrainObservationMessage.reply_to == BrainWait.question_message),
    ).join(BrainTask, BrainTask.task_id == BrainWait.task_id).where(
        BrainTask.task_id == task_id, BrainTask.status == "waiting_input", BrainTask.deadline > now,
        BrainWait.status == "open", BrainWait.deadline > now,
        BrainObservationMessage.bot_id == BrainTask.bot_id,
        BrainObservationMessage.created_at >= BrainTask.created_at,
        BrainObservationMessage.created_at <= now,
    ).order_by(BrainObservationMessage.created_at, BrainObservationMessage.id).limit(30))).all())


async def finish(db: AsyncSession, task_id: str, token: str, *, now: datetime) -> bool:
    """Finish only after the current owner has no unconfirmed actions."""
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "running", BrainTask.claim_token == token,
        BrainTask.lease_until > now, BrainTask.deadline > now,
        ~exists(select(BrainAction.action_id).where(BrainAction.task_id == task_id, BrainAction.status != "confirmed")),
    ).values(status="completed", claim_token=None, lease_until=None,
             revision=BrainTask.revision + 1, updated_at=now))
    return bool(result.rowcount)


async def expire_wait(db: AsyncSession, task_id: str, *, now: datetime) -> bool:
    """Atomically reserve a timeout notice when a task's wait expires."""
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "waiting_input",
        BrainTask.task_id.in_(select(BrainWait.task_id).where(BrainWait.status == "open", BrainWait.deadline <= now)),
    ).values(status="finalizing", revision=BrainTask.revision + 1, updated_at=now))
    if result.rowcount:
        await db.execute(update(BrainWait).where(BrainWait.task_id == task_id, BrainWait.status == "open").values(status="expired"))
        task = await db.get(BrainTask, task_id)
        if task is None:
            raise RuntimeError("Expired task disappeared")
        db.add(BrainAction(action_id=uuid4().hex, task_id=task_id, logical_step="timeout", effect_slot="notice",
                           platform=task.platform, target_scene=task.source_scene, content="还没有收到对方群的答复。",
                           status="prepared", claim_token=None, lease_until=None, receipt_message_id=None,
                           created_at=now, updated_at=now))
        await db.flush()
    return bool(result.rowcount)


async def claim_finalization(db: AsyncSession, task_id: str, token: str, *, now: datetime, lease_seconds: int = 180) -> bool:
    if not token or lease_seconds <= 0:
        raise ValueError("A claim token and positive lease duration are required")
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "finalizing",
        (BrainTask.claim_token.is_(None)) | (BrainTask.lease_until <= now),
    ).values(claim_token=token, lease_until=now + timedelta(seconds=lease_seconds),
             revision=BrainTask.revision + 1, updated_at=now))
    return bool(result.rowcount)


async def finish_expired(db: AsyncSession, task_id: str, token: str, *, now: datetime) -> bool:
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "finalizing", BrainTask.claim_token == token,
        BrainTask.lease_until > now,
        exists(select(BrainAction.action_id).where(BrainAction.task_id == task_id,
               BrainAction.logical_step == "timeout", BrainAction.effect_slot == "notice", BrainAction.status == "confirmed")),
    ).values(status="expired", claim_token=None, lease_until=None,
             revision=BrainTask.revision + 1, updated_at=now))
    return bool(result.rowcount)


async def accept_reply(db: AsyncSession, task_id: str, *, platform: str, target_scene: str, reply_to: str,
                       message_id: str, user_id: str, text: str, now: datetime) -> bool:
    if not all((platform, target_scene, reply_to, message_id, user_id, text.strip())):
        return False
    waiting = await db.get(BrainWait, task_id)
    if waiting is None or (waiting.platform, waiting.target_scene, waiting.question_message) != (platform, target_scene, reply_to):
        return False
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.status == "waiting_input", BrainTask.deadline > now,
        BrainTask.task_id.in_(select(BrainWait.task_id).where(BrainWait.status == "open", BrainWait.deadline > now)),
    ).values(status="queued", revision=BrainTask.revision + 1, updated_at=now))
    if not result.rowcount:
        return False
    await db.execute(update(BrainWait).where(BrainWait.task_id == task_id, BrainWait.status == "open").values(
        status="answered", answer_message=message_id, answer_user=user_id, answer_text=text))
    return True


async def cancel(db: AsyncSession, task_id: str, *, brain_id: str, platform: str, source_scene: str, source_user: str, now: datetime) -> bool:
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == task_id, BrainTask.brain_id == brain_id, BrainTask.source_user == source_user,
        BrainTask.platform == platform, BrainTask.source_scene == source_scene,
        BrainTask.status.in_(("queued", "running", "waiting_input")),
    ).values(status="cancelled", claim_token=None, lease_until=None, revision=BrainTask.revision + 1, updated_at=now))
    if result.rowcount:
        await db.execute(update(BrainWait).where(BrainWait.task_id == task_id, BrainWait.status == "open").values(status="cancelled"))
    return bool(result.rowcount)
