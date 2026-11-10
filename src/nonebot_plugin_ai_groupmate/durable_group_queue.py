"""Persist OneBot V11 addressed group inputs, including the debounce window.

Only requests that have not started are replayed. A crash after execution starts
may have produced external side effects, so those jobs require manual review.
The queue currently assumes a single active bot process per database.
"""

from uuid import uuid4
from typing import cast
from datetime import datetime, timezone, timedelta
from dataclasses import replace

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import CursorResult
from nonebot_plugin_orm import get_session
from nonebot_plugin_uninfo.model import Session
from nonebot_plugin_uninfo.params import get_interface

from .model import GroupReplyJob
from .reply_request import ReplyRequest
from .message_metadata import is_reliable_platform_id


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def snapshot(request: ReplyRequest) -> dict | None:
    try:
        from nonebot.adapters.onebot.v11 import GroupMessageEvent
    except ImportError:
        return None
    if not isinstance(request.event, GroupMessageEvent) or not is_reliable_platform_id(request.message_id):
        return None
    return {
        "event": request.event.model_dump(mode="json"), "session": request.session.dump(),
        "request_id": request.request_id, "bot_name": request.bot_name,
        "user_id": request.user_id, "user_name": request.user_name,
        "is_tome": request.is_tome, "is_continuous": request.is_continuous,
        "reply_to_id": request.reply_to_id, "message_id": request.message_id,
        "input_batch": request.input_batch, "mentioned_user_ids": request.mentioned_user_ids,
        "normalized_parts": request.normalized_parts,
        "meme_required": request.meme_required, "meme_send_count": request.meme_send_count,
        "reaction_required": request.reaction_required,
    }


async def stage(request: ReplyRequest, text: str) -> tuple[ReplyRequest, bool]:
    """Commit a message before the in-memory debounce can acknowledge it."""
    payload = snapshot(request)
    if payload is None:
        return request, True
    job_id = uuid4().hex
    payload["input_batch"] = [(request.message_id, text)]
    timestamp = now()
    async with get_session() as db:
        try:
            async with db.begin_nested():
                db.add(GroupReplyJob(job_id=job_id, platform=request.bot.adapter.get_name(), bot_id=str(request.bot.self_id),
                                     scene_id=str(request.session.scene.id), user_id=request.user_id,
                                     message_id=request.message_id, payload=payload, status="staging",
                                     created_at=timestamp, updated_at=timestamp))
                await db.flush()
        except IntegrityError:
            return request, False
        await db.commit()
    return replace(request, durable_job_id=job_id), True


async def admit(request: ReplyRequest) -> bool:
    if request.durable_job_id is None:
        return True
    payload = snapshot(request)
    assert payload is not None
    ids = [mid for mid, _ in request.input_batch if is_reliable_platform_id(mid)]
    async with get_session() as db:
        result = await db.execute(update(GroupReplyJob).where(
            GroupReplyJob.job_id == request.durable_job_id, GroupReplyJob.status == "staging",
        ).values(payload=payload, status="queued", updated_at=now()))
        if not cast(CursorResult, result).rowcount:
            await db.rollback()
            return False
        await db.execute(update(GroupReplyJob).where(
            GroupReplyJob.bot_id == str(request.bot.self_id), GroupReplyJob.scene_id == str(request.session.scene.id),
            GroupReplyJob.user_id == request.user_id, GroupReplyJob.message_id.in_(ids),
            GroupReplyJob.job_id != request.durable_job_id, GroupReplyJob.status == "staging",
        ).values(status="merged", updated_at=now()))
        await db.commit()
    return True


async def transition(job_id: str, old: str, new: str) -> bool:
    async with get_session() as db:
        result = await db.execute(update(GroupReplyJob).where(
            GroupReplyJob.job_id == job_id, GroupReplyJob.status == old,
        ).values(status=new, updated_at=now()))
        await db.commit()
        return bool(cast(CursorResult, result).rowcount)


async def cancel_staged(bot_id: str, scene_id: str, user_id: str) -> None:
    async with get_session() as db:
        await db.execute(update(GroupReplyJob).where(
            GroupReplyJob.bot_id == bot_id, GroupReplyJob.scene_id == scene_id,
            GroupReplyJob.user_id == user_id, GroupReplyJob.status == "staging",
        ).values(status="cancelled", updated_at=now()))
        await db.commit()


async def recover_startup() -> None:
    async with get_session() as db:
        await db.execute(update(GroupReplyJob).where(GroupReplyJob.status == "running").values(status="interrupted", updated_at=now()))
        staged = list((await db.scalars(select(GroupReplyJob).where(GroupReplyJob.status == "staging")
                                        .order_by(GroupReplyJob.created_at))).all())
        batches: dict[tuple[str, str, str, str, str], list[list[GroupReplyJob]]] = {}
        for job in staged:
            key = (job.platform, job.bot_id, job.scene_id, job.user_id, job.payload.get("reply_to_id") or "")
            groups = batches.setdefault(key, [])
            if not groups or job.created_at - groups[-1][0].created_at > timedelta(seconds=10) or len(groups[-1]) >= 20:
                groups.append([])
            groups[-1].append(job)
        for groups in batches.values():
            for group in groups:
                latest = group[-1]
                payload = dict(latest.payload)
                payload["input_batch"] = [item for job in group for item in job.payload["input_batch"]]
                payload["is_tome"] = any(job.payload["is_tome"] for job in group)
                payload["is_continuous"] = any(job.payload["is_continuous"] for job in group)
                latest.payload = payload
                latest.status = "queued"
                latest.updated_at = now()
                for job in group[:-1]:
                    job.status = "merged"
                    job.updated_at = now()
        await db.commit()


async def queued(limit: int = 50) -> list[GroupReplyJob]:
    async with get_session() as db:
        jobs = list((await db.scalars(select(GroupReplyJob).where(GroupReplyJob.status == "queued")
                                     .order_by(GroupReplyJob.created_at).limit(limit))).all())
        db.expunge_all()
        return jobs


def restore(job: GroupReplyJob, bot) -> ReplyRequest:
    from nonebot.adapters.onebot.v11 import GroupMessageEvent

    if str(bot.self_id) != job.bot_id or bot.adapter.get_name() != job.platform:
        raise ValueError("The original bot is unavailable")
    data = job.payload
    event = GroupMessageEvent.model_validate(data["event"])
    session = Session.load(data["session"])
    interface = get_interface(bot)
    if interface is None or str(session.scene.id) != job.scene_id or str(session.user.id) != job.user_id:
        raise ValueError("The saved request cannot be restored")
    return ReplyRequest(request_id=data["request_id"], session=session, interface=interface, bot=bot, event=event,
                        bot_name=data["bot_name"], user_id=data["user_id"], user_name=data["user_name"],
                        is_tome=data["is_tome"], is_continuous=data["is_continuous"], reply_to_id=data["reply_to_id"],
                        message_id=data["message_id"], input_batch=tuple(tuple(x) for x in data["input_batch"]),
                        mentioned_user_ids=tuple(data["mentioned_user_ids"]),
                        normalized_parts=tuple(data["normalized_parts"]) if data["normalized_parts"] is not None else None,
                        meme_required=data["meme_required"], meme_send_count=data["meme_send_count"],
                        reaction_required=data["reaction_required"], durable_job_id=job.job_id)
