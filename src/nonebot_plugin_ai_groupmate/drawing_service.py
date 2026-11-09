"""Provider and transport adapters for durable drawing jobs."""

import asyncio
import hashlib
from pathlib import Path

from sqlalchemy import Select
from nonebot_plugin_orm import get_session
from nonebot_plugin_alconna import Target, UniMessage

from .model import DrawingJob, ChatHistory
from .config import ScopedConfig
from .participation import delivery_context
from .agent.delivery import PublishedImage, OutboundIdentity, send_and_finalize
from .drawing_runner import DrawingDelivery, DrawingPreflightFailed, DrawingGenerationRejected, DrawingCancelledBeforeProvider, DrawingDeliveryPreflightFailed
from .message_metadata import is_reliable_platform_id
from .agent.reply_tools import image_reply_rows, compose_image_reply
from .agent.file_workspace import FileTaskError, FileWorkspace
from .agent.image_generation import ImageGenerationError, request_image, prepare_markup, prepare_reference, resolve_image_credentials


def endpoint_hash(endpoint: str) -> str:
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()


def workspace_for_job(root: Path, job: DrawingJob) -> FileWorkspace:
    return FileWorkspace(root, session_id=job.session_id, user_id=job.user_id, bot_id=job.bot_id,
                         is_private=job.is_private, peer_bot_ids=())


async def generate_job(job: DrawingJob, *, root: Path, config: ScopedConfig) -> str:
    """Load frozen references, call the configured provider once and save its output."""
    if not config.image_generation_enabled:
        raise DrawingPreflightFailed("drawing_disabled")
    cfg = config.model_copy(update={"image_generation_model": job.model, "image_generation_quality": job.quality}, deep=True)
    try:
        _, base = resolve_image_credentials(cfg)
    except (ValueError, RuntimeError) as error:
        raise DrawingPreflightFailed("provider_unavailable") from error
    if endpoint_hash(base) != job.endpoint_hash:
        raise DrawingPreflightFailed("provider_changed")
    workspace = workspace_for_job(root, job)
    references = []
    options = dict(job.image_options or {})
    try:
        for file_id in job.reference_file_ids:
            data = await asyncio.to_thread(workspace.read, file_id)
            references.append(await asyncio.to_thread(prepare_reference, data))
        markup_file_id = options.pop("markup_file_id", None)
        if markup_file_id:
            if not references:
                raise ValueError("missing original image")
            markup_data = await asyncio.to_thread(workspace.read, markup_file_id)
            options["markup"] = await asyncio.to_thread(prepare_markup, markup_data, references[0][0])
    except (OSError, ValueError) as error:
        raise DrawingPreflightFailed("reference_unavailable") from error
    # Synchronous rollout groups share the same provider limit with background jobs.
    from .agent.image_generation_tools import _generation_gate
    async with _generation_gate:
        async with get_session() as db:
            status = await db.scalar(Select(DrawingJob.status).where(
                DrawingJob.job_id == job.job_id, DrawingJob.claim_token == job.claim_token,
            ))
        if status == "cancel_requested":
            raise DrawingCancelledBeforeProvider()
        if status != "running":
            raise RuntimeError("drawing claim no longer valid")
        try:
            generated = await asyncio.wait_for(
                request_image(cfg, job.prompt, references, size=job.size, transparent_background=job.transparent_background, **options),
                timeout=cfg.image_generation_timeout_seconds,
            )
        except ImageGenerationError as error:
            if error.code in {"image_request_rejected", "image_auth_failed", "image_access_denied", "image_quota_exhausted"}:
                raise DrawingGenerationRejected(error.code) from error
            raise
    artifact = await asyncio.to_thread(workspace.save, generated.data, f"gpt-image.{generated.output_format}",
                                       content_type=f"image/{generated.output_format}", source_key=f"drawing-job:{job.job_id}")
    return artifact.file_id


async def deliver_job(job: DrawingJob, *, root: Path) -> DrawingDelivery:
    """Send to the original conversation; save history and publication after acknowledgement."""
    try:
        if not job.result_file_id:
            raise ValueError("drawing has no artifact")
        workspace = workspace_for_job(root, job)
        artifact = await asyncio.to_thread(workspace.get, job.result_file_id)
        data = await asyncio.to_thread(workspace.read, job.result_file_id)
    except (OSError, ValueError, FileTaskError) as error:
        raise DrawingDeliveryPreflightFailed("generated_image_unavailable") from error
    quoted_id = job.reply_to_id if is_reliable_platform_id(job.reply_to_id) else None
    message = compose_image_reply(UniMessage.image(raw=data), quoted_id=quoted_id)
    target = Target(id=job.session_id, private=job.is_private, self_id=job.bot_id)
    publication = PublishedImage(workspace, artifact.file_id)
    def record(message_id: str):
        return image_reply_rows(session_id=job.session_id, bot_name=job.bot_name,
                                message_id=message_id, quoted_id=quoted_id,
                                image_content=f"[图片文件] {artifact.name} file_id={artifact.file_id}")
    token = delivery_context.set(((job.bot_id, job.session_id), False) if not job.is_private else None)
    try:
        identity = OutboundIdentity(job.platform, "private" if job.is_private else "group", job.bot_id, job.bot_name, job.request_id)
        result = await send_and_finalize(None, job.session_id, message, send_options={"target": target, "fallback": False},
                                         record=record, publish=publication.publish, publication=publication.describe,
                                         outbound_identity=identity, reply_to_id=quoted_id, session_factory=get_session)
    finally:
        delivery_context.reset(token)
    warning = None if result.history_saved and result.publication_saved and result.structured_saved else "delivery_record_incomplete"
    return DrawingDelivery(result.message_id, warning)


async def deliver_failure_notice(job: DrawingJob) -> DrawingDelivery:
    """Tell the requester what happened without repeating the paid request."""
    if job.status == "notifying_failed":
        if job.error_code == "reference_unavailable":
            content = "参考图现在读不到了，可以重新发一下吗？"
        elif job.error_code == "generated_image_unavailable":
            content = "图已经画好了，但图片文件现在读不到了，要不再试一次？"
        elif job.error_code == "image_request_rejected":
            content = "这次绘图请求没被服务接受，换个描述再试试？"
        elif job.error_code == "image_quota_exhausted":
            content = "绘图服务现在限流或额度不足，稍后再试试吧。"
        else:
            content = "绘图暂时用不了，等设置恢复后再试试吧。"
    else:
        content = "刚才画图中断了，我没法确认服务有没有收到；你想重试再告诉我。"
    quoted_id = job.reply_to_id if is_reliable_platform_id(job.reply_to_id) else None
    message = UniMessage.text(content)
    if quoted_id:
        message = UniMessage.reply(quoted_id) + message
    target = Target(id=job.session_id, private=job.is_private, self_id=job.bot_id)
    def record(message_id: str) -> list[ChatHistory]:
        header = f"id: {message_id}\n" + (f"回复id: {quoted_id}\n" if quoted_id else "")
        return [ChatHistory(session_id=job.session_id, user_id=job.bot_name, user_name=job.bot_name,
                            content_type="bot", content=header + content)]
    token = delivery_context.set(((job.bot_id, job.session_id), False) if not job.is_private else None)
    try:
        identity = OutboundIdentity(job.platform, "private" if job.is_private else "group", job.bot_id, job.bot_name, job.request_id)
        result = await send_and_finalize(None, job.session_id, message, send_options={"target": target, "fallback": False},
                                         record=record, outbound_identity=identity, reply_to_id=quoted_id,
                                         session_factory=get_session)
    finally:
        delivery_context.reset(token)
    return DrawingDelivery(result.message_id, None if result.history_saved and result.structured_saved else "delivery_record_incomplete")
