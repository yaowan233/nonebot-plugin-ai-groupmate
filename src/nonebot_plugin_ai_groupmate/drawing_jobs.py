"""Durable drawing-job state transitions. External calls happen outside transactions.

Callers must commit a successful claim before invoking a provider or adapter.
Expired generation and delivery claims are terminal: their side effects may
already have happened, so recovery must never repeat them automatically.
"""

import re
import uuid
from typing import Any, Literal, cast
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from sqlalchemy import Select, CursorResult, update
from sqlalchemy.ext.asyncio import AsyncSession, async_scoped_session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .model import DrawingJob

DB = AsyncSession | async_scoped_session[AsyncSession]
LEASE_SECONDS = 180
_FILE_ID = re.compile(r"^[0-9a-f]{32}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class DrawingRequest:
    platform: str
    session_id: str
    is_private: bool
    bot_id: str
    bot_name: str
    user_id: str
    request_id: str
    reply_to_id: str | None
    fingerprint: str
    prompt: str
    reference_file_ids: tuple[str, ...]
    size: str
    transparent_background: bool
    model: str
    quality: str
    endpoint_hash: str
    deferred: bool = False
    image_options: dict | None = None


@dataclass(frozen=True)
class EnqueuedDrawing:
    job_id: str
    status: str


async def enqueue_drawing(db: DB, request: DrawingRequest, *, now: datetime | None = None) -> EnqueuedDrawing:
    """Idempotently persist a task; no provider or transport call occurs here."""
    if (not all((request.platform, request.session_id, request.bot_id, request.user_id, request.request_id, request.model))
            or not _FINGERPRINT.fullmatch(request.fingerprint) or not _FINGERPRINT.fullmatch(request.endpoint_hash) or not request.prompt.strip()
            or len(request.prompt) > 32000 or len(request.reference_file_ids) > 4
            or any(not _FILE_ID.fullmatch(ref) for ref in request.reference_file_ids)
            or request.size not in {"1024x1024", "1536x1024", "1024x1536", "auto"}
            or request.quality not in {"low", "medium", "high", "xhigh", "max", "auto"}):
        raise ValueError("invalid drawing request")
    time = now or utcnow()
    values = {
        "job_id": uuid.uuid4().hex, "platform": request.platform, "session_id": request.session_id,
        "is_private": request.is_private, "bot_id": request.bot_id, "bot_name": request.bot_name.replace("\x00", ""),
        "user_id": request.user_id, "request_id": request.request_id, "reply_to_id": request.reply_to_id,
        "fingerprint": request.fingerprint, "prompt": request.prompt.replace("\x00", ""),
        "reference_file_ids": list(request.reference_file_ids), "size": request.size,
        "transparent_background": request.transparent_background, "model": request.model,
        "quality": request.quality, "endpoint_hash": request.endpoint_hash,
        "image_options": request.image_options,
        "status": "held" if request.deferred else "queued", "created_at": time, "updated_at": time,
    }
    keys = ["platform", "session_id", "is_private", "user_id", "request_id", "fingerprint"]
    dialect = db.get_bind().dialect.name
    if dialect in {"sqlite", "postgresql"}:
        insert = sqlite_insert if dialect == "sqlite" else pg_insert
        await db.execute(insert(DrawingJob).values(**values).on_conflict_do_nothing(index_elements=keys))
    else:
        existing = await db.scalar(Select(DrawingJob).where(*(getattr(DrawingJob, key) == values[key] for key in keys)))
        if existing is None:
            db.add(DrawingJob(**values))
            await db.flush()
    result = await db.scalar(Select(DrawingJob).where(*(getattr(DrawingJob, key) == values[key] for key in keys)))
    if result is None:
        raise RuntimeError("drawing job insert failed")
    return EnqueuedDrawing(result.job_id, result.status)


async def _transition(db: DB, job_id: str, *, from_status: tuple[str, ...], token: str | None,
                      values: dict[str, Any]) -> bool:
    conditions = [DrawingJob.job_id == job_id, DrawingJob.status.in_(from_status)]
    if token is not None:
        conditions.append(DrawingJob.claim_token == token)
    result = await db.execute(update(DrawingJob).where(*conditions).values(**values))
    return cast(CursorResult[Any], result).rowcount == 1


async def claim_generation(db: DB, job_id: str, token: str, *, now: datetime | None = None) -> bool:
    time = now or utcnow()
    return await _transition(db, job_id, from_status=("queued",), token=None,
        values={"status": "running", "claim_token": token, "lease_until": time + timedelta(seconds=LEASE_SECONDS), "updated_at": time})


async def renew_claim(db: DB, job_id: str, token: str, *, now: datetime | None = None) -> bool:
    time = now or utcnow()
    return await _transition(db, job_id, from_status=("running", "cancel_requested", "delivering", "notifying_failed", "notifying_interrupted"), token=token,
        values={"lease_until": time + timedelta(seconds=LEASE_SECONDS), "updated_at": time})


async def complete_generation(db: DB, job_id: str, token: str, file_id: str, *, now: datetime | None = None) -> Literal["generated", "cancelled"] | None:
    if not _FILE_ID.fullmatch(file_id):
        raise ValueError("invalid artifact id")
    time = now or utcnow()
    values = {"status": "generated", "result_file_id": file_id, "claim_token": None, "lease_until": None, "updated_at": time}
    if await _transition(db, job_id, from_status=("running",), token=token, values=values):
        return "generated"
    values.update(status="cancelled", finished_at=time)
    if await _transition(db, job_id, from_status=("cancel_requested",), token=token, values=values):
        return "cancelled"
    return None


async def claim_delivery(db: DB, job_id: str, token: str, *, now: datetime | None = None) -> bool:
    time = now or utcnow()
    return await _transition(db, job_id, from_status=("generated",), token=None,
        values={"status": "delivering", "claim_token": token, "lease_until": time + timedelta(seconds=LEASE_SECONDS), "updated_at": time})


async def finish_delivery(db: DB, job_id: str, token: str, *, message_id: str | None,
                          error_code: str | None = None, warning_code: str | None = None,
                          now: datetime | None = None) -> bool:
    time = now or utcnow()
    return await _transition(db, job_id, from_status=("delivering",), token=token, values={
        "status": "delivery_unknown" if error_code else "completed", "result_message_id": message_id,
        "error_code": error_code or warning_code, "claim_token": None, "lease_until": None, "updated_at": time, "finished_at": time,
    })


async def fail_delivery_before_send(db: DB, job_id: str, token: str, error_code: str, *, now: datetime | None = None) -> bool:
    """The image was not sent; queue one notice without regenerating it."""
    time = now or utcnow()
    return await _transition(db, job_id, from_status=("delivering",), token=token, values={
        "status": "failed_notice_pending", "error_code": error_code, "claim_token": None, "lease_until": None,
        "updated_at": time, "finished_at": time,
    })


async def fail_generation(db: DB, job_id: str, token: str, error_code: str, *, outcome_unknown: bool = True, now: datetime | None = None) -> bool:
    time = now or utcnow()
    if await _transition(db, job_id, from_status=("cancel_requested",), token=token, values={
        "status": "cancelled", "error_code": error_code if outcome_unknown else None, "claim_token": None, "lease_until": None,
        "updated_at": time, "finished_at": time,
    }):
        return True
    return await _transition(db, job_id, from_status=("running",), token=token, values={
        "status": "interrupted_notice_pending" if outcome_unknown else "failed_notice_pending", "error_code": error_code, "claim_token": None, "lease_until": None,
        "updated_at": time, "finished_at": time,
    })


async def claim_failure_notice(db: DB, job_id: str, token: str, *, now: datetime | None = None) -> bool:
    """Claim the notice separately; an uncertain send is never repeated."""
    time = now or utcnow()
    values = {"claim_token": token, "lease_until": time + timedelta(seconds=LEASE_SECONDS), "updated_at": time}
    for pending, notifying in (("failed_notice_pending", "notifying_failed"), ("interrupted_notice_pending", "notifying_interrupted")):
        if await _transition(db, job_id, from_status=(pending,), token=None, values={**values, "status": notifying}):
            return True
    return False


async def finish_failure_notice(db: DB, job_id: str, token: str, *, message_id: str | None,
                                outcome_unknown: bool = False, now: datetime | None = None) -> bool:
    time = now or utcnow()
    for notifying, completed, uncertain in (("notifying_failed", "failed", "failed_notice_unknown"),
                                             ("notifying_interrupted", "interrupted", "interrupted_notice_unknown")):
        if await _transition(db, job_id, from_status=(notifying,), token=token, values={
            "status": uncertain if outcome_unknown else completed, "result_message_id": message_id,
            "claim_token": None, "lease_until": None, "updated_at": time, "finished_at": time,
        }):
            return True
    return False


async def acknowledge_cancelled_generation(db: DB, job_id: str, token: str, *, now: datetime | None = None) -> bool:
    time = now or utcnow()
    return await _transition(db, job_id, from_status=("cancel_requested",), token=token, values={
        "status": "cancelled", "claim_token": None, "lease_until": None, "updated_at": time, "finished_at": time,
    })


async def revise_queued_drawing(db: DB, job_id: str, *, platform: str, session_id: str,
                                is_private: bool, user_id: str, revision: str,
                                now: datetime | None = None) -> Literal["updated", "not_found", "already_started", "too_long", "changed"]:
    """Append a user's revision only before this scoped job is claimed."""
    revision = revision.replace("\x00", "").strip()
    if not revision or len(revision) > 32000:
        return "too_long"
    conditions = (DrawingJob.job_id == job_id, DrawingJob.platform == platform,
                  DrawingJob.session_id == session_id, DrawingJob.is_private == is_private,
                  DrawingJob.user_id == user_id)
    row = (await db.execute(Select(DrawingJob.status, DrawingJob.prompt).where(*conditions))).one_or_none()
    if row is None:
        return "not_found"
    status, original = row
    if status != "queued":
        return "already_started"
    revised = f"{original}\n补充修改：{revision}"
    if len(revised) > 32000:
        return "too_long"
    # Keep the original request fingerprint: replaying that request must still
    # resolve to this job instead of submitting a second paid generation.
    result = await db.execute(update(DrawingJob).where(
        *conditions, DrawingJob.status == "queued", DrawingJob.prompt == original,
    ).values(prompt=revised, updated_at=now or utcnow()))
    return "updated" if cast(CursorResult[Any], result).rowcount == 1 else "changed"


async def cancel_drawing(db: DB, job_id: str, *, platform: str, session_id: str, is_private: bool,
                         user_id: str, now: datetime | None = None) -> str | None:
    """Only the owner in the original conversation can cancel; never cancel delivery."""
    time = now or utcnow()
    conditions = (DrawingJob.job_id == job_id, DrawingJob.platform == platform, DrawingJob.session_id == session_id,
                  DrawingJob.is_private == is_private, DrawingJob.user_id == user_id)
    batch_id = await db.scalar(Select(DrawingJob.batch_id).where(*conditions))
    if batch_id:
        from .ordered_delivery_jobs import cancel_batch
        cancelled = await cancel_batch(db, batch_id, platform=platform, session_id=session_id,
                                       is_private=is_private, user_id=user_id)
        return "cancelled" if cancelled else None
    result = await db.execute(update(DrawingJob).where(*conditions, DrawingJob.status.in_(("held", "queued", "generated"))).values(
        status="cancelled", updated_at=time, finished_at=time))
    if cast(CursorResult[Any], result).rowcount == 1:
        return "cancelled"
    result = await db.execute(update(DrawingJob).where(*conditions, DrawingJob.status == "running").values(status="cancel_requested", updated_at=time))
    return "cancel_requested" if cast(CursorResult[Any], result).rowcount == 1 else None


async def recover_expired_claims(db: DB, *, now: datetime | None = None, limit: int = 100) -> tuple[int, int]:
    time = now or utcnow()
    if limit <= 0:
        return (0, 0)
    generation_ids = list((await db.scalars(Select(DrawingJob.job_id).where(
        DrawingJob.status.in_(("running", "cancel_requested")), DrawingJob.lease_until < time,
    ).order_by(DrawingJob.lease_until).limit(limit))).all())
    delivery_ids = list((await db.scalars(Select(DrawingJob.job_id).where(
        DrawingJob.status == "delivering", DrawingJob.lease_until < time,
    ).order_by(DrawingJob.lease_until).limit(limit))).all())
    notice_ids = list((await db.scalars(Select(DrawingJob.job_id).where(
        DrawingJob.status.in_(("notifying_failed", "notifying_interrupted")), DrawingJob.lease_until < time,
    ).order_by(DrawingJob.lease_until).limit(limit))).all())
    if not generation_ids and not delivery_ids and not notice_ids:
        return (0, 0)
    generation = await db.execute(update(DrawingJob).where(
        DrawingJob.job_id.in_(generation_ids), DrawingJob.status == "running", DrawingJob.lease_until < time,
    ).values(status="interrupted_notice_pending", error_code="generation_outcome_unknown", claim_token=None,
             lease_until=None, updated_at=time, finished_at=time))
    cancelled = await db.execute(update(DrawingJob).where(
        DrawingJob.job_id.in_(generation_ids), DrawingJob.status == "cancel_requested", DrawingJob.lease_until < time,
    ).values(status="cancelled", error_code="generation_outcome_unknown", claim_token=None,
             lease_until=None, updated_at=time, finished_at=time))
    delivery = await db.execute(update(DrawingJob).where(
        DrawingJob.job_id.in_(delivery_ids), DrawingJob.status == "delivering", DrawingJob.lease_until < time,
    ).values(status="delivery_unknown", error_code="delivery_outcome_unknown", claim_token=None,
             lease_until=None, updated_at=time, finished_at=time))
    for active, uncertain in (("notifying_failed", "failed_notice_unknown"),
                              ("notifying_interrupted", "interrupted_notice_unknown")):
        await db.execute(update(DrawingJob).where(
            DrawingJob.job_id.in_(notice_ids), DrawingJob.status == active, DrawingJob.lease_until < time,
        ).values(status=uncertain, claim_token=None, lease_until=None, updated_at=time, finished_at=time))
    return cast(CursorResult[Any], generation).rowcount + cast(CursorResult[Any], cancelled).rowcount, cast(CursorResult[Any], delivery).rowcount
