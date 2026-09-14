import math
import uuid
import datetime
from typing import Any, Literal, cast

from sqlalchemy import CursorResult, func, select, update
from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_orm import get_session
from nonebot_plugin_alconna import Target, UniMessage
from nonebot_plugin_apscheduler import scheduler

from ..model import ChatHistory, ScheduledTask
from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from ..scheduled_tasks import utcnow

TaskStatus = Literal["scheduled", "running", "completed", "failed", "cancelled", "missed", "interrupted", "all"]


async def send_scheduled_text(
    session_id: str,
    content: str,
    *,
    is_private: bool,
    bot_id: str | None,
    bot_name: str,
) -> None:
    try:
        target = Target(
            id=session_id,
            private=is_private,
            self_id=bot_id,
        )
        result = await UniMessage.text(content).send(target=target)

        msg_id = "unknown"
        if result.msg_ids:
            raw_msg_id = result.msg_ids[-1].get("message_id") or result.msg_ids[
                -1
            ].get("msg_id")
            if raw_msg_id is not None:
                msg_id = str(raw_msg_id)

        async with get_session() as db_session:
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n" + content,
                user_name=bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()

        logger.info(f"[定时任务] 已发送到 {session_id}: {content}")
    except Exception as e:
        logger.error(f"[定时任务] 发送失败 {session_id}: {e}")
        raise


def _validate_delay(
    delay_minutes: float,
    delay_hours: float,
    *,
    label: str,
) -> tuple[float, str | None, str | None]:
    if any(not math.isfinite(value) or value < 0 for value in (delay_minutes, delay_hours)):
        return 0, "invalid_delay", "延迟时间必须是有限的非负数。"
    delay_seconds = delay_hours * 3600 + delay_minutes * 60
    if delay_seconds <= 0:
        return delay_seconds, "invalid_delay", "延迟时间必须大于 0。"
    if delay_seconds < 10:
        return delay_seconds, "delay_too_short", "延迟时间太短，至少需要 10 秒。"
    if delay_seconds > 7 * 24 * 3600:
        return (
            delay_seconds,
            "delay_too_long",
            f"延迟时间太长，当前最多支持 7 天内的定时{label}。",
        )
    return delay_seconds, None, None


def _task_data(task: ScheduledTask) -> dict[str, Any]:
    def format_time(value: datetime.datetime | None) -> str | None:
        return value.replace(tzinfo=datetime.timezone.utc).astimezone(scheduler.timezone).isoformat() if value else None

    return {
        "job_id": task.job_id,
        "task_type": task.task_type,
        "task" if task.task_type == "agent" else "content": task.content,
        "run_at": format_time(task.run_at),
        "status": task.status,
        "started_at": format_time(task.started_at),
        "finished_at": format_time(task.finished_at),
        "error": task.error,
    }


async def _expired_result(session_id: str, request_id: str | None) -> str | None:
    if request_id is not None and not await is_request_active(session_id, request_id):
        return tool_skipped("request_expired", "请求已过期，未执行定时任务操作。", delivery_state="not_attempted")
    return None


async def _create_task(
    session_id: str, request_id: str | None, *, is_private: bool, bot_id: str | None,
    bot_name: str, task_type: str, content: str, delay_minutes: float, delay_hours: float,
) -> str:
    if expired := await _expired_result(session_id, request_id):
        return expired
    content = content.strip()
    if not content:
        return tool_failure(
            "empty_task" if task_type == "agent" else "empty_content",
            "定时任务内容不能为空，未创建任务。", delivery_state="not_attempted",
        )
    delay_seconds, reason_code, error = _validate_delay(delay_minutes, delay_hours, label="任务")
    if error:
        return tool_failure(reason_code or "invalid_delay", error, delivery_state="not_attempted")
    now = utcnow()
    prefix = "ai_groupmate_agent_schedule" if task_type == "agent" else "ai_groupmate_schedule"
    task = ScheduledTask(
        job_id=f"{prefix}_{session_id}_{uuid.uuid4().hex}",
        session_id=session_id, is_private=is_private, bot_id=bot_id, bot_name=bot_name,
        task_type=task_type, content=content, run_at=now + datetime.timedelta(seconds=delay_seconds),
        status="scheduled", created_at=now, updated_at=now,
    )
    try:
        async with get_session() as session:
            session.add(task)
            await session.flush()
            data = _task_data(task)
            if expired := await _expired_result(session_id, request_id):
                await session.rollback()
                return expired
            await session.commit()
    except Exception as error:
        logger.warning(f"保存定时任务失败: session={session_id}, error_type={type(error).__name__}")
        return tool_failure("schedule_failed", "定时任务保存失败，请查询任务列表确认状态。", delivery_state="unknown")
    return tool_success(
        "schedule_created", f"定时任务已保存，将在 {data['run_at']} 执行。",
        data=data, delivery_state="completed",
    )


def create_schedule_message_tool(
    session_id: str, request_id: str | None, *, is_private: bool, bot_id: str | None, bot_name: str,
):
    @tool("schedule_message")
    async def schedule_message(content: str, delay_minutes: float = 0, delay_hours: float = 0) -> str:
        """安排 Bot 在几分钟或几小时后向当前群聊/私聊发送文本，任务持久保存到数据库。

        Args:
            content: 到点后发送的文本。
            delay_minutes: 延迟分钟数，可以是小数。
            delay_hours: 延迟小时数，和 delay_minutes 相加。
        """
        return await _create_task(
            session_id, request_id, is_private=is_private, bot_id=bot_id, bot_name=bot_name,
            task_type="message", content=content, delay_minutes=delay_minutes, delay_hours=delay_hours,
        )

    return schedule_message


def create_schedule_agent_task_tool(
    session_id: str, request_id: str | None, *, is_private: bool, bot_id: str | None,
):
    @tool("schedule_agent_task")
    async def schedule_agent_task(task: str, delay_minutes: float = 0, delay_hours: float = 0) -> str:
        """安排 Bot 到点后重新进入 Agent，使用工具完成任务；任务持久保存到数据库。

        Args:
            task: 到点后完成的任务描述，例如“查一下明天上海天气并提醒我带伞”。
            delay_minutes: 延迟分钟数，可以是小数。
            delay_hours: 延迟小时数，和 delay_minutes 相加。
        """
        return await _create_task(
            session_id, request_id, is_private=is_private, bot_id=bot_id, bot_name="",
            task_type="agent", content=task, delay_minutes=delay_minutes, delay_hours=delay_hours,
        )

    return schedule_agent_task


def create_schedule_management_tools(
    session_id: str, request_id: str | None, *, is_private: bool, bot_id: str | None,
):
    scope = (
        ScheduledTask.session_id == session_id,
        ScheduledTask.is_private == is_private,
        ScheduledTask.bot_id == bot_id,
    )

    def management_failure(error: Exception) -> str:
        logger.warning(f"管理定时任务失败: session={session_id}, error_type={type(error).__name__}")
        return tool_failure(
            "schedule_management_failed", "数据库操作失败，请重新查询任务列表确认当前状态。",
            delivery_state="unknown",
        )

    async def change_task(job_id: str, changes: dict[str, Any]) -> str:
        try:
            async with get_session() as session:
                # Only one of cancellation, editing and execution can win this update.
                result = await session.execute(
                    update(ScheduledTask).where(
                        *scope, ScheduledTask.job_id == job_id, ScheduledTask.status == "scheduled",
                    ).values(**changes, updated_at=utcnow())
                )
                if cast(CursorResult[Any], result).rowcount != 1:
                    await session.rollback()
                    return tool_failure(
                        "schedule_not_found", "当前会话中未找到该待执行任务，可能已开始执行、过期或取消。请重新查询列表。",
                        delivery_state="not_attempted",
                    )
                task = await session.get(ScheduledTask, job_id)
                assert task is not None
                data = _task_data(task)
                if expired := await _expired_result(session_id, request_id):
                    await session.rollback()
                    return expired
                await session.commit()
        except Exception as error:
            return management_failure(error)
        cancelled = changes.get("status") == "cancelled"
        return tool_success(
            "schedule_cancelled" if cancelled else "schedule_updated",
            "定时任务已取消，不会再执行。" if cancelled else "定时任务已修改。",
            data=data, delivery_state="completed",
        )

    @tool("list_scheduled_tasks")
    async def list_scheduled_tasks(offset: int = 0, limit: int = 10, status: TaskStatus = "scheduled") -> str:
        """从数据库查看当前会话、当前 Bot 的定时任务；按执行时间排序，可查历史状态。

        Args:
            offset: 跳过的任务数，从 0 开始；有更多任务时用 next_offset 继续查询。
            limit: 每页任务数，1 到 50，默认 10。
            status: 默认 scheduled 只查待执行；all 查全部。也可查询 running、completed、failed、cancelled、missed、interrupted。
        """
        if expired := await _expired_result(session_id, request_id):
            return expired
        if offset < 0 or not 1 <= limit <= 50:
            return tool_failure("invalid_pagination", "offset 必须非负，limit 必须在 1 到 50 之间。")
        filters = list(scope)
        if status != "all":
            filters.append(ScheduledTask.status == status)
        try:
            async with get_session() as session:
                total = await session.scalar(select(func.count()).select_from(ScheduledTask).where(*filters)) or 0
                tasks = (await session.scalars(
                    select(ScheduledTask).where(*filters)
                    .order_by(ScheduledTask.run_at, ScheduledTask.job_id).offset(offset).limit(limit)
                )).all()
                data = [_task_data(task) for task in tasks]
        except Exception as error:
            return management_failure(error)
        next_offset = offset + len(data)
        return tool_success(
            "schedules_listed", f"当前会话共有 {total} 个符合状态条件的定时任务。",
            data={"tasks": data, "total": total, "next_offset": next_offset if next_offset < total else None},
        )

    @tool("update_scheduled_task")
    async def update_scheduled_task(
        job_id: str, content: str | None = None,
        delay_minutes: float | None = None, delay_hours: float | None = None,
    ) -> str:
        """修改数据库中的待执行任务，保留 ID 和类型。先查询列表取得准确 ID。

        Args:
            job_id: 创建结果或列表中的完整 job_id，不要猜测。
            content: 新文本消息或新 Agent 任务描述；省略则保留原内容。
            delay_minutes: 从现在起多少分钟后执行，和 delay_hours 相加；两者都省略则保留原时间。
            delay_hours: 从现在起多少小时后执行；指定时间表示重新计时，不是在原时间上追加。
        """
        if expired := await _expired_result(session_id, request_id):
            return expired
        if content is None and delay_minutes is None and delay_hours is None:
            return tool_failure("no_changes", "请指定新内容或新的延迟时间。", delivery_state="not_attempted")
        changes: dict[str, Any] = {}
        if content is not None:
            content = content.strip()
            if not content:
                return tool_failure("empty_content", "任务内容不能为空，未修改任务。", delivery_state="not_attempted")
            changes["content"] = content
        if delay_minutes is not None or delay_hours is not None:
            delay_seconds, reason_code, error = _validate_delay(
                delay_minutes if delay_minutes is not None else 0,
                delay_hours if delay_hours is not None else 0, label="任务",
            )
            if error:
                return tool_failure(reason_code or "invalid_delay", error, delivery_state="not_attempted")
            changes["run_at"] = utcnow() + datetime.timedelta(seconds=delay_seconds)
        return await change_task(job_id, changes)

    @tool("cancel_scheduled_task")
    async def cancel_scheduled_task(job_id: str) -> str:
        """取消数据库中的待执行任务，保留取消记录。不能撤回已开始执行的任务。

        Args:
            job_id: 创建结果或列表中的完整 job_id，不确定时先查询，不要猜测。
        """
        if expired := await _expired_result(session_id, request_id):
            return expired
        return await change_task(job_id, {"status": "cancelled", "finished_at": utcnow()})

    return [list_scheduled_tasks, update_scheduled_task, cancel_scheduled_task]
