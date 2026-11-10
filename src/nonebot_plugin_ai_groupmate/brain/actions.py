"""Durably record side effects; this module does not send platform messages.

The caller owns commit boundaries. Commit the sending state before issuing I/O.
An unknown delivery is never automatically retried.
"""

from uuid import uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass

from sqlalchemy import or_, exists, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..model import BrainTask, BrainAction


@dataclass(frozen=True)
class ActionSpec:
    task_id: str
    logical_step: str
    effect_slot: str
    platform: str
    target_scene: str
    content: str


async def prepare(db: AsyncSession, spec: ActionSpec, *, task_token: str, now: datetime) -> tuple[BrainAction, bool]:
    if not all((spec.task_id, spec.logical_step, spec.effect_slot, spec.platform, spec.target_scene, spec.content.strip(), task_token)):
        raise ValueError("Action identity, target and content are required")
    # Lock and fence the task owner before registering an externally visible action.
    result = await db.execute(update(BrainTask).where(
        BrainTask.task_id == spec.task_id, BrainTask.status == "running", BrainTask.claim_token == task_token,
        BrainTask.lease_until > now, BrainTask.deadline > now,
    ).values(revision=BrainTask.revision + 1, updated_at=now))
    if not result.rowcount:
        raise ValueError("Task lease is not active")
    query = select(BrainAction).where(
        BrainAction.task_id == spec.task_id, BrainAction.logical_step == spec.logical_step,
        BrainAction.effect_slot == spec.effect_slot, BrainAction.platform == spec.platform,
        BrainAction.target_scene == spec.target_scene,
    )
    existing = await db.scalar(query)
    if existing is not None:
        return existing, False
    action = BrainAction(action_id=uuid4().hex, task_id=spec.task_id, logical_step=spec.logical_step,
                         effect_slot=spec.effect_slot, platform=spec.platform, target_scene=spec.target_scene,
                         content=spec.content, status="prepared", claim_token=None, lease_until=None,
                         receipt_message_id=None, created_at=now, updated_at=now)
    try:
        async with db.begin_nested():
            db.add(action)
            await db.flush()
    except IntegrityError:
        existing = await db.scalar(query)
        if existing is None:
            raise
        return existing, False
    return action, True


async def claim_send(db: AsyncSession, action_id: str, *, task_token: str, send_token: str,
                     now: datetime, lease_seconds: int = 180) -> bool:
    if not task_token or not send_token or lease_seconds <= 0:
        raise ValueError("Valid claim tokens and lease duration are required")
    # Take the task row lock in the same transaction as the action claim, so
    # cancellation cannot slip between task ownership validation and the claim.
    owner = await db.execute(update(BrainTask).where(
        BrainTask.task_id.in_(select(BrainAction.task_id).where(BrainAction.action_id == action_id)),
        BrainTask.claim_token == task_token, BrainTask.lease_until > now,
        or_((BrainTask.status == "running") & (BrainTask.deadline > now),
            (BrainTask.status == "finalizing") & exists(select(BrainAction.action_id).where(
                BrainAction.action_id == action_id, BrainAction.logical_step == "timeout", BrainAction.effect_slot == "notice"))),
    ).values(revision=BrainTask.revision + 1, updated_at=now))
    if not owner.rowcount:
        return False
    result = await db.execute(update(BrainAction).where(
        BrainAction.action_id == action_id, BrainAction.status == "prepared",
    ).values(status="sending", claim_token=send_token, lease_until=now + timedelta(seconds=lease_seconds), updated_at=now))
    return bool(result.rowcount)


async def confirm(db: AsyncSession, action_id: str, *, send_token: str, message_id: str, now: datetime) -> bool:
    if not send_token or not message_id:
        raise ValueError("Send token and platform receipt are required")
    # Even a cancelled task must record a confirmed send that already happened.
    result = await db.execute(update(BrainAction).where(
        BrainAction.action_id == action_id, BrainAction.status.in_(("sending", "delivery_unknown")),
        BrainAction.claim_token == send_token,
    ).values(status="confirmed", receipt_message_id=message_id, lease_until=None, updated_at=now))
    return bool(result.rowcount)


async def mark_unknown(db: AsyncSession, *, now: datetime) -> int:
    result = await db.execute(update(BrainAction).where(
        BrainAction.status == "sending", BrainAction.lease_until <= now,
    ).values(status="delivery_unknown", lease_until=None, updated_at=now))
    return int(result.rowcount or 0)
