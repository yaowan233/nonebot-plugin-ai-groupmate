"""Deliver a claimed ordered reply exactly once through the normal adapter."""

import uuid
import asyncio
from pathlib import Path
from typing import cast

from sqlalchemy import Select, update, CursorResult
from nonebot.log import logger
from nonebot_plugin_orm import get_session
from nonebot_plugin_alconna import Target, UniMessage

from .model import DrawingJob, ChatHistory, OrderedDeliveryJob
from .drawing_jobs import utcnow
from .participation import delivery_context
from .agent.delivery import PublishedImage, OutboundIdentity, send_and_finalize
from .agent.file_content import image_preview
from .agent.outbound_plan import ReplyPlan, compose_reply
from .agent.file_workspace import FileWorkspace
from .ordered_delivery_jobs import claim_ready


async def run_ordered_delivery(batch_id: str, *, root: Path) -> bool:
    claim_token = uuid.uuid4().hex
    async with get_session() as db:
        state = await claim_ready(db, batch_id, claim_token)
        if state is None:
            await db.rollback()
            return False
        batch = await db.get(OrderedDeliveryJob, batch_id, populate_existing=True)
        if batch is None:
            await db.rollback()
            return False
        children = list((await db.scalars(Select(DrawingJob).where(DrawingJob.batch_id == batch_id))).all())
        db.expunge(batch)
        for child in children:
            db.expunge(child)
        await db.commit()
    workspace = FileWorkspace(root, session_id=batch.session_id, user_id=batch.user_id, bot_id=batch.bot_id,
                              is_private=batch.is_private, peer_bot_ids=())
    identity = OutboundIdentity(batch.platform, "private" if batch.is_private else "group", batch.bot_id, batch.bot_name, batch.request_id)
    target = Target(id=batch.session_id, private=batch.is_private, self_id=batch.bot_id)
    started = False
    error_code = None
    receipt = None
    try:
        if state == "notifying":
            content = "这组图片暂时没能全部准备好，所以没法按原来的排版一起发出来。"
            message = UniMessage.text(content)
            if batch.reply_to_id:
                message = UniMessage.reply(batch.reply_to_id) + message
            def record_notice(message_id):
                return [ChatHistory(session_id=batch.session_id, user_id=batch.bot_name, user_name=batch.bot_name,
                                    content_type="bot", content=f"id: {message_id}\n{content}")]
            record = record_notice
            publications = []
            file_ids = []
        else:
            job_files = {job.job_id: job.result_file_id for job in children if job.status == "generated" and job.result_file_id}
            resolved_parts = []
            for source in batch.parts:
                part = dict(source)
                if part.get("type") == "image" and part.get("resource_id", "").startswith("drawing:"):
                    file_id = job_files.get(part["resource_id"].partition(":")[2])
                    if not file_id:
                        raise ValueError("drawing artifact missing")
                    part["resource_id"] = "file:" + file_id
                resolved_parts.append(part)
            plan = ReplyPlan.model_validate({"parts": resolved_parts, "reply_to": batch.reply_to_id})
            images = {}
            file_ids = []
            publications = []
            names = {}
            total_bytes = 0
            for part in plan.parts:
                if part.type != "image":
                    continue
                assert part.resource_id is not None
                kind, _, file_id = part.resource_id.partition(":")
                if kind != "file":
                    raise ValueError("invalid image resource")
                artifact = await asyncio.to_thread(workspace.get, file_id)
                data = await asyncio.to_thread(workspace.read, file_id)
                total_bytes += len(data)
                if total_bytes > 30 * 1024 * 1024:
                    raise ValueError("message images too large")
                await asyncio.to_thread(image_preview, data, 0)
                images[part.resource_id] = data
                names[part.resource_id] = artifact.name
                file_ids.append(file_id)
                publications.append(PublishedImage(workspace, file_id))
            message = compose_reply(plan, images, batch.reply_to_id)
            if batch.platform == "OneBot V11":
                await message.export(adapter="OneBot V11", fallback=False)
            def record_message(message_id):
                header = f"id: {message_id}\n" + (f"回复id: {batch.reply_to_id}\n" if batch.reply_to_id else "")
                rows = []
                for part in plan.parts:
                    if part.type == "image":
                        assert part.resource_id is not None
                        file_id = part.resource_id.partition(":")[2]
                        content = f"[图片文件] {names[part.resource_id]} file_id={file_id}"
                    elif part.type == "mention":
                        assert part.user_id is not None
                        content = "@" + part.user_id
                    else:
                        assert part.text is not None
                        content = part.text
                    rows.append(ChatHistory(session_id=batch.session_id, user_id=batch.bot_name, user_name=batch.bot_name,
                                            content_type="bot", content=header + content))
                return rows
            record = record_message
        if state == "preparing":
            # The transition and cancellation contend on the same row. Once
            # dispatch is claimed, cancellation can no longer promise success.
            async with get_session() as db:
                result = await db.execute(update(OrderedDeliveryJob).where(
                    OrderedDeliveryJob.job_id == batch_id, OrderedDeliveryJob.status == "preparing",
                    OrderedDeliveryJob.claim_token == claim_token,
                ).values(status="delivering", updated_at=utcnow()))
                await db.commit()
            if cast(CursorResult, result).rowcount != 1:
                return False
            state = "delivering"
        token = delivery_context.set(((batch.bot_id, batch.session_id), False) if not batch.is_private else None)
        try:
            started = True
            receipt = await send_and_finalize(None, batch.session_id, message,
                send_options={"target": target, "fallback": False}, record=record,
                publications=publications, image_file_ids=file_ids, outbound_identity=identity,
                reply_to_id=batch.reply_to_id, session_factory=get_session)
        finally:
            delivery_context.reset(token)
    except asyncio.CancelledError:
        error_code = "delivery_outcome_unknown" if started else "preparation_interrupted"
        raise
    except Exception as error:
        error_code = "delivery_outcome_unknown" if started else "resource_unavailable"
        logger.warning(f"组合消息交付失败 job_id={batch_id} error_type={type(error).__name__}")
    finally:
        now = utcnow()
        status = ("delivery_unknown" if started and receipt is None else
                  "failed_notice_pending" if error_code and state == "preparing" else
                  "failed" if error_code or state == "notifying" else "completed")
        async with get_session() as db:
            result = await db.execute(update(OrderedDeliveryJob).where(
                OrderedDeliveryJob.job_id == batch_id, OrderedDeliveryJob.status == state,
                OrderedDeliveryJob.claim_token == claim_token,
            ).values(status=status, claim_token=None, error_code=error_code or ("drawing_unavailable" if state == "notifying" else None), result_message_id=receipt.message_id if receipt else None,
                     updated_at=now, finished_at=now))
            if cast(CursorResult, result).rowcount and state == "delivering" and status != "failed_notice_pending":
                await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status == "generated").values(
                    status=status, error_code=error_code, result_message_id=receipt.message_id if receipt else None,
                    updated_at=now, finished_at=now))
            if cast(CursorResult, result).rowcount and state == "notifying":
                for original, terminal in (("failed_notice_pending", "failed"), ("interrupted_notice_pending", "interrupted")):
                    if status == "delivery_unknown":
                        terminal += "_notice_unknown"
                    await db.execute(update(DrawingJob).where(DrawingJob.batch_id == batch_id, DrawingJob.status == original).values(
                        status=terminal, updated_at=now, finished_at=now))
            await db.commit()
    return receipt is not None and state == "delivering"
