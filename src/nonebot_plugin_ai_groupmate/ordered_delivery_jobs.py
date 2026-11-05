"""Durable state transitions for replies waiting on drawing jobs."""

from datetime import timedelta

from sqlalchemy import Select, update

from .model import DrawingJob, OrderedDeliveryJob
from .drawing_jobs import utcnow


async def claim_ready(db, batch_id: str, token: str) -> str | None:
    """Claim a ready batch for preparation; sending is claimed separately."""
    batch = await db.get(OrderedDeliveryJob, batch_id)
    if batch is None or batch.status not in {"waiting", "failed_notice_pending"}:
        return None
    children = (await db.scalars(Select(DrawingJob).where(DrawingJob.batch_id == batch_id))).all()
    if batch.status == "failed_notice_pending" or not children:
        state = "notifying"
    elif any(job.status in {"cancelled", "failed_notice_pending", "interrupted_notice_pending", "failed", "interrupted"} for job in children):
        state = "notifying"
    elif all(job.status == "generated" and job.result_file_id for job in children):
        state = "preparing"
    else:
        return None
    result = await db.execute(update(OrderedDeliveryJob).where(
        OrderedDeliveryJob.job_id == batch_id, OrderedDeliveryJob.status == batch.status,
    ).values(status=state, claim_token=token, updated_at=utcnow(), error_code="drawing_unavailable" if state == "notifying" else None))
    if result.rowcount != 1:
        return None
    if state == "notifying":
        await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status.in_(("queued", "generated"))).values(
            status="cancelled", updated_at=utcnow(), finished_at=utcnow()))
        await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status == "running").values(
            status="cancel_requested", updated_at=utcnow()))
    return state


async def cancel_batch(db, batch_id: str, *, platform: str, session_id: str, is_private: bool, user_id: str) -> bool:
    now = utcnow()
    result = await db.execute(update(OrderedDeliveryJob).where(
        OrderedDeliveryJob.job_id == batch_id, OrderedDeliveryJob.platform == platform,
        OrderedDeliveryJob.session_id == session_id, OrderedDeliveryJob.is_private == is_private,
        OrderedDeliveryJob.user_id == user_id, OrderedDeliveryJob.status.in_(("waiting", "preparing", "failed_notice_pending")),
    ).values(status="cancelled", claim_token=None, updated_at=now, finished_at=now))
    if result.rowcount != 1:
        return False
    await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status.in_(("queued", "generated"))).values(
        status="cancelled", updated_at=now, finished_at=now))
    await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status == "running").values(
        status="cancel_requested", updated_at=now))
    return True


async def recover_expired(db) -> None:
    now = utcnow()
    # Preparing does not send. A new token makes an abandoned worker harmless.
    await db.execute(update(OrderedDeliveryJob).where(
        OrderedDeliveryJob.status == "preparing", OrderedDeliveryJob.updated_at < now - timedelta(minutes=10),
    ).values(status="waiting", claim_token=None, updated_at=now))
    # A stale dispatch may already have reached the adapter; never resend it.
    expired_ids = list((await db.scalars(Select(OrderedDeliveryJob.job_id).where(
        OrderedDeliveryJob.status.in_(("delivering", "notifying")),
        OrderedDeliveryJob.updated_at < now - timedelta(minutes=10),
    ).limit(100))).all())
    for batch_id in expired_ids:
        result = await db.execute(update(OrderedDeliveryJob).where(
            OrderedDeliveryJob.job_id == batch_id, OrderedDeliveryJob.status.in_(("delivering", "notifying")),
            OrderedDeliveryJob.updated_at < now - timedelta(minutes=10),
        ).values(status="delivery_unknown", claim_token=None, error_code="worker_interrupted", updated_at=now, finished_at=now))
        if result.rowcount:
            await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status == "generated").values(
                status="delivery_unknown", error_code="delivery_outcome_unknown", updated_at=now, finished_at=now))
    # Held jobs have not reached the provider; abandoned preparations can expire.
    await db.execute(update(DrawingJob).where(DrawingJob.status == "held", DrawingJob.batch_id.is_(None),
        DrawingJob.created_at < now - timedelta(hours=1)).values(status="cancelled", updated_at=now, finished_at=now))
