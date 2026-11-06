"""Execute one claimed drawing stage using isolated database sessions."""

import time
import uuid
import asyncio
from typing import Literal, TypeVar
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from nonebot.log import logger
from nonebot_plugin_orm import get_session

from .model import DrawingJob
from .drawing_jobs import renew_claim, claim_delivery, fail_generation, finish_delivery, claim_generation, complete_generation, claim_failure_notice, finish_failure_notice, fail_delivery_before_send, acknowledge_cancelled_generation
from .message_trace import trace_id

HEARTBEAT_SECONDS = 30
T = TypeVar("T")


@dataclass(frozen=True)
class DrawingDelivery:
    message_id: str
    warning_code: str | None = None


class DrawingCancelledBeforeProvider(Exception):
    """Cancellation was observed before the external request began."""


class DrawingPreflightFailed(Exception):
    """A known failure before the provider was called."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class DrawingDeliveryPreflightFailed(Exception):
    """A generated image cannot be delivered and no send was attempted."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class DrawingGenerationRejected(Exception):
    """The provider explicitly rejected the request without generating an image."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


async def _claim(job_id: str, token: str, *, stage: Literal["generation", "delivery", "notice"]) -> DrawingJob | None:
    async with get_session() as db:
        claim = {"generation": claim_generation, "delivery": claim_delivery, "notice": claim_failure_notice}[stage]
        claimed = await claim(db, job_id, token)
        if not claimed:
            await db.rollback()
            return None
        job = await db.get(DrawingJob, job_id, populate_existing=True)
        if job is not None:
            db.expunge(job)
        # The claim must be durable before any external side effect starts.
        await db.commit()
        if job is not None:
            logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_{stage}_started job={job_id}")
        return job


async def _heartbeat(job_id: str, token: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        async with get_session() as db:
            ok = await renew_claim(db, job_id, token)
            await db.commit()
        if not ok:
            raise RuntimeError("drawing claim lost")


async def _execute_with_heartbeat(job_id: str, token: str, callback: Callable[[], Awaitable[T]]) -> T:
    execution = asyncio.ensure_future(callback())
    heartbeat = asyncio.create_task(_heartbeat(job_id, token))
    try:
        done, _ = await asyncio.wait({execution, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
        if heartbeat in done:
            await heartbeat
        return await execution
    finally:
        execution.cancel()
        heartbeat.cancel()
        await asyncio.gather(execution, heartbeat, return_exceptions=True)


async def run_generation(job_id: str, generate: Callable[[DrawingJob], Awaitable[str]]) -> bool:
    """Generate once; an interrupted attempt is never automatically submitted again."""
    token = uuid.uuid4().hex
    job = await _claim(job_id, token, stage="generation")
    if job is None:
        return False
    started = time.perf_counter()
    try:
        file_id = await _execute_with_heartbeat(job_id, token, lambda: generate(job))
        async with get_session() as db:
            status = await complete_generation(db, job_id, token, file_id)
            await db.commit()
        logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_generation_finished "
                    f"job={job_id} state={status or 'claim_lost'} duration_ms={(time.perf_counter() - started) * 1000:.0f}")
        return status is not None
    except DrawingCancelledBeforeProvider:
        async with get_session() as db:
            cancelled = await acknowledge_cancelled_generation(db, job_id, token)
            await db.commit()
        return cancelled
    except (DrawingPreflightFailed, DrawingGenerationRejected) as error:
        logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_generation_rejected job={job_id} code={error.code}")
        try:
            async with get_session() as db:
                cancelled = await acknowledge_cancelled_generation(db, job_id, token)
                if not cancelled:
                    await fail_generation(db, job_id, token, error.code, outcome_unknown=False)
                await db.commit()
        except Exception as update_error:
            logger.warning(f"绘图任务状态更新失败 job_id={job_id} error_type={type(update_error).__name__}")
        return False
    except asyncio.CancelledError:
        await _mark_generation_interrupted(job_id, token, "worker_cancelled")
        raise
    except Exception as error:
        logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_generation_unknown job={job_id}")
        logger.warning(f"绘图任务执行失败 job_id={job_id} error_type={type(error).__name__}")
        await _mark_generation_interrupted(job_id, token, "generation_outcome_unknown")
        return False


async def _mark_generation_interrupted(job_id: str, token: str, reason: str) -> None:
    try:
        async with get_session() as db:
            await fail_generation(db, job_id, token, reason)
            await db.commit()
    except Exception as error:
        # If the database is down, lease recovery will mark the outcome unknown.
        logger.warning(f"绘图任务状态更新失败 job_id={job_id} error_type={type(error).__name__}")


async def run_delivery(job_id: str, deliver: Callable[[DrawingJob], Awaitable[str | DrawingDelivery]]) -> bool:
    """Deliver once; a failed or interrupted send is recorded as outcome unknown."""
    token = uuid.uuid4().hex
    job = await _claim(job_id, token, stage="delivery")
    if job is None:
        return False
    started = time.perf_counter()
    try:
        result = await _execute_with_heartbeat(job_id, token, lambda: deliver(job))
        message_id = result.message_id if isinstance(result, DrawingDelivery) else result
        warning = result.warning_code if isinstance(result, DrawingDelivery) else None
        async with get_session() as db:
            completed = await finish_delivery(db, job_id, token, message_id=message_id, warning_code=warning)
            await db.commit()
        logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_delivery_finished "
                    f"job={job_id} completed={completed} duration_ms={(time.perf_counter() - started) * 1000:.0f}")
        return completed
    except DrawingDeliveryPreflightFailed as error:
        logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_delivery_unavailable job={job_id} code={error.code}")
        try:
            async with get_session() as db:
                await fail_delivery_before_send(db, job_id, token, error.code)
                await db.commit()
        except Exception as update_error:
            logger.warning(f"绘图交付状态更新失败 job_id={job_id} error_type={type(update_error).__name__}")
        return False
    except asyncio.CancelledError:
        await _mark_delivery_unknown(job_id, token, "worker_cancelled")
        raise
    except Exception as error:
        logger.warning(f"绘图交付结果未知 job_id={job_id} error_type={type(error).__name__}")
        await _mark_delivery_unknown(job_id, token, "delivery_outcome_unknown")
        return False


async def _mark_delivery_unknown(job_id: str, token: str, reason: str) -> None:
    try:
        async with get_session() as db:
            await finish_delivery(db, job_id, token, message_id=None, error_code=reason)
            await db.commit()
    except Exception as error:
        logger.warning(f"绘图交付状态更新失败 job_id={job_id} error_type={type(error).__name__}")


async def run_failure_notice(job_id: str, notify: Callable[[DrawingJob], Awaitable[str | DrawingDelivery]]) -> bool:
    """Notify once in the original session without resubmitting the provider request."""
    token = uuid.uuid4().hex
    job = await _claim(job_id, token, stage="notice")
    if job is None:
        return False
    started = time.perf_counter()
    try:
        result = await _execute_with_heartbeat(job_id, token, lambda: notify(job))
        message_id = result.message_id if isinstance(result, DrawingDelivery) else result
        async with get_session() as db:
            completed = await finish_failure_notice(db, job_id, token, message_id=message_id)
            await db.commit()
        logger.info(f"[MessageTrace] trace={trace_id(job.session_id, job.request_id)} stage=drawing_notice_finished "
                    f"job={job_id} completed={completed} duration_ms={(time.perf_counter() - started) * 1000:.0f}")
        return completed
    except asyncio.CancelledError:
        await _mark_notice_unknown(job_id, token)
        raise
    except Exception as error:
        logger.warning(f"绘图失败提示结果未知 job_id={job_id} error_type={type(error).__name__}")
        await _mark_notice_unknown(job_id, token)
        return False


async def _mark_notice_unknown(job_id: str, token: str) -> None:
    try:
        async with get_session() as db:
            await finish_failure_notice(db, job_id, token, message_id=None, outcome_unknown=True)
            await db.commit()
    except Exception as error:
        logger.warning(f"绘图提示状态更新失败 job_id={job_id} error_type={type(error).__name__}")
