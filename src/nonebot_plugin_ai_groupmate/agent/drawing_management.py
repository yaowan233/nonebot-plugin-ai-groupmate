"""View and cancel only the current user's drawing jobs in this conversation."""

from datetime import timedelta

from sqlalchemy import Select
from langchain_core.tools import tool

from ..model import DrawingJob, OrderedDeliveryJob
from ..ordered_delivery_jobs import cancel_batch
from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from ..drawing_jobs import utcnow, cancel_drawing, revise_queued_drawing

_JOB_MESSAGES = {
    "held": "等待加入图文组合",
    "queued": "等待绘图",
    "running": "正在绘图",
    "cancel_requested": "正在取消，已开始的绘图可能仍会计费",
    "generated": "图片已生成，等待发送",
    "delivering": "正在发送图片",
    "completed": "图片已发送",
    "cancelled": "已取消",
    "failed": "绘图未开始，请检查原因后重新请求",
    "interrupted": "绘图结果未知，不会自动重复请求",
    "failed_notice_pending": "绘图未开始，等待通知",
    "interrupted_notice_pending": "绘图结果未知，等待通知",
    "notifying_failed": "绘图未开始，正在通知",
    "notifying_interrupted": "绘图结果未知，正在通知",
    "failed_notice_unknown": "绘图未开始，通知结果未知，不会自动重复发送",
    "interrupted_notice_unknown": "绘图和通知结果未知，不会自动重复请求或发送",
    "delivery_unknown": "图片发送结果未知，不会自动重复发送",
}
_ERROR_MESSAGES = {
    "drawing_disabled": "绘图功能已关闭",
    "provider_unavailable": "绘图服务配置不可用",
    "provider_changed": "绘图服务配置已变更",
    "reference_unavailable": "参考图已失效或无法读取",
    "generated_image_unavailable": "图片已生成，但文件已失效或无法读取",
    "image_request_rejected": "绘图服务拒绝了请求",
    "image_auth_failed": "绘图服务认证失败",
    "image_access_denied": "绘图服务拒绝访问",
    "image_quota_exhausted": "绘图服务限流或额度不足",
    "generation_outcome_unknown": "绘图请求可能已提交",
    "delivery_outcome_unknown": "发送请求可能已提交",
    "delivery_record_incomplete": "图片已发送，但本地记录尚未全部保存",
}


def _job_status_message(status: str, error_code: str | None) -> str:
    if error_code == "generated_image_unavailable":
        return {
            "failed_notice_pending": "图片已生成但无法读取，等待通知",
            "notifying_failed": "图片已生成但无法读取，正在通知",
            "failed": "图片已生成但无法读取，已通知",
            "failed_notice_unknown": "图片已生成但无法读取，通知结果未知，不会自动重复发送",
        }.get(status, _JOB_MESSAGES.get(status, "状态未知"))
    if error_code in {"image_request_rejected", "image_auth_failed", "image_access_denied", "image_quota_exhausted"}:
        return {
            "failed_notice_pending": "绘图服务拒绝了请求，等待通知",
            "notifying_failed": "绘图服务拒绝了请求，正在通知",
            "failed": "绘图服务拒绝了请求，已通知",
            "failed_notice_unknown": "绘图请求被拒绝，通知结果未知，不会自动重复发送",
        }.get(status, _JOB_MESSAGES.get(status, "状态未知"))
    return _JOB_MESSAGES.get(status, "状态未知")


def create_drawing_management_tools(db, *, platform: str, session_id: str, is_private: bool,
                                    user_id: str, request_id: str | None):
    attempted_revisions: set[tuple[str, str]] = set()

    async def active() -> bool:
        return request_id is None or await is_request_active(session_id, request_id)

    def conditions():
        return (DrawingJob.platform == platform, DrawingJob.session_id == session_id,
                DrawingJob.is_private == is_private, DrawingJob.user_id == user_id)

    @tool("get_drawing_jobs")
    async def get_drawing_jobs() -> str:
        """查看当前用户在本会话最近的绘图任务和状态，不会重新生成。"""
        if not await active():
            return tool_skipped("request_expired", "请求已过期。")
        rows = (await db.scalars(Select(DrawingJob).where(*conditions(), DrawingJob.created_at >= utcnow() - timedelta(days=1))
                                 .order_by(DrawingJob.created_at.desc()).limit(5))).all()
        data = [{"job_id": row.job_id, "status": row.status, "status_message": _job_status_message(row.status, row.error_code),
                 "description": row.prompt[:120], "created_at": row.created_at.isoformat(), "error_code": row.error_code,
                 "batch_id": row.batch_id,
                 "error_message": _ERROR_MESSAGES.get(row.error_code) if row.error_code else None} for row in rows]
        batches = (await db.scalars(Select(OrderedDeliveryJob).where(
            OrderedDeliveryJob.platform == platform, OrderedDeliveryJob.session_id == session_id,
            OrderedDeliveryJob.is_private == is_private, OrderedDeliveryJob.user_id == user_id,
            OrderedDeliveryJob.created_at >= utcnow() - timedelta(days=1),
        ).order_by(OrderedDeliveryJob.created_at.desc()).limit(5))).all()
        return tool_success("drawing_jobs_loaded", "已读取本会话的绘图和组合回复任务；结果未知的任务不会自动重试。",
                            data={"jobs": data, "batches": [{"job_id": batch.job_id, "status": batch.status} for batch in batches]})

    @tool("cancel_drawing_job")
    async def cancel_drawing_job(job_id: str | None = None) -> str:
        """取消自己的绘图任务或组合回复；有多个进行中的任务时先查询并指定 job_id。"""
        if not await active():
            return tool_skipped("request_expired", "请求已过期，未取消任务。")
        if job_id is None:
            rows = (await db.scalars(Select(DrawingJob.job_id).where(*conditions(),
                    DrawingJob.status.in_(("held", "queued", "running", "generated", "cancel_requested")))
                    .order_by(DrawingJob.created_at.desc()).limit(2))).all()
            if len(rows) != 1:
                return tool_failure("drawing_not_unique", "没有唯一的进行中绘图任务，请先查询并指定任务。")
            job_id = str(rows[0])
        batch_cancelled = await cancel_batch(db, job_id, platform=platform, session_id=session_id,
                                             is_private=is_private, user_id=user_id)
        status = "cancelled" if batch_cancelled else await cancel_drawing(db, job_id, platform=platform, session_id=session_id,
                                                                           is_private=is_private, user_id=user_id)
        await db.commit()
        if status is None:
            return tool_failure("drawing_not_cancellable", "任务不存在、不可访问或已经无法取消。")
        message = ("已请求取消；绘图服务若已开始处理仍可能计费，结果不会自动发送。"
                   if status == "cancel_requested" else "已取消绘图任务。")
        return tool_success("drawing_cancelled", message, data={"job_id": job_id, "job_status": status})

    @tool("revise_drawing_job")
    async def revise_drawing_job(revision: str, job_id: str | None = None) -> str:
        """向自己正在排队的绘图任务补充修改；多个进行中任务时先查询并指定 job_id。不会重启已开始的绘图。"""
        if not await active():
            return tool_skipped("request_expired", "请求已过期，未修改绘图任务。")
        if not revision.strip() or len(revision) > 32000:
            return tool_failure("invalid_revision", "请提供有效的绘图修改要求。")
        if job_id is None:
            rows = (await db.scalars(Select(DrawingJob.job_id).where(*conditions(),
                    DrawingJob.status.in_(("queued", "running", "generated", "cancel_requested")))
                    .order_by(DrawingJob.created_at.desc()).limit(2))).all()
            if len(rows) != 1:
                return tool_failure("drawing_not_unique", "没有唯一的进行中绘图任务，请先查询并指定任务。")
            job_id = str(rows[0])
        revision_key = (str(job_id), revision.replace("\x00", "").strip())
        if revision_key in attempted_revisions:
            return tool_skipped("duplicate_revision", "本轮已尝试过这项修改，未重复追加。", delivery_state="not_attempted")
        attempted_revisions.add(revision_key)
        status = await revise_queued_drawing(db, job_id, platform=platform, session_id=session_id,
                                             is_private=is_private, user_id=user_id, revision=revision)
        await db.commit()
        if status == "updated":
            return tool_success("drawing_revised", "已将修改要求加入原有排队任务，不会重复生成。",
                                data={"job_id": job_id, "job_status": "queued"})
        if status == "not_found":
            return tool_failure("drawing_not_found", "任务不存在或不属于当前会话。")
        if status == "too_long":
            return tool_failure("invalid_revision", "绘图要求过长或为空，未修改任务。")
        return tool_failure("drawing_already_started", "任务已开始或状态发生变化，未修改，也未重新生成。")

    return [get_drawing_jobs, cancel_drawing_job, revise_drawing_job]
