import uuid
import datetime
from collections.abc import Callable

from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_orm import get_session
from nonebot_plugin_alconna import Target, UniMessage
from nonebot_plugin_apscheduler import scheduler

from ..model import ChatHistory
from ..reply_guard import is_request_active


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


def _validate_delay(delay_minutes: float, delay_hours: float, *, label: str) -> tuple[float, str | None]:
    delay_seconds = delay_hours * 3600 + delay_minutes * 60
    if delay_seconds <= 0:
        return delay_seconds, "延迟时间必须大于 0。"
    if delay_seconds < 10:
        return delay_seconds, "延迟时间太短，至少需要 10 秒。"
    if delay_seconds > 7 * 24 * 3600:
        return delay_seconds, f"延迟时间太长，当前最多支持 7 天内的定时{label}。"
    return delay_seconds, None


def create_schedule_message_tool(
    session_id: str,
    request_id: str | None,
    *,
    is_private: bool,
    bot_id: str | None,
    bot_name: str,
):
    @tool("schedule_message")
    async def schedule_message(
        content: str,
        delay_minutes: float = 0,
        delay_hours: float = 0,
    ) -> str:
        """
        安排 bot 在几分钟或几小时后向当前群聊/私聊发送一条文本消息。
        Args:
            content: 到点后要发送的文本内容。
            delay_minutes: 延迟多少分钟，可以是小数。
            delay_hours: 延迟多少小时，可以和 delay_minutes 同时使用。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消定时任务。"

        content = content.strip()
        if not content:
            return "定时消息内容为空，未创建任务。"

        delay_seconds, error = _validate_delay(
            delay_minutes, delay_hours, label="消息"
        )
        if error:
            return error

        run_at = datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)
        job_id = f"ai_groupmate_schedule_{session_id}_{uuid.uuid4().hex}"

        scheduler.add_job(
            send_scheduled_text,
            "date",
            id=job_id,
            run_date=run_at,
            kwargs={
                "session_id": session_id,
                "content": content,
                "is_private": is_private,
                "bot_id": bot_id,
                "bot_name": bot_name,
            },
            misfire_grace_time=300,
        )

        return f"定时任务已创建，将在 {run_at.strftime('%Y-%m-%d %H:%M:%S')} 发送：{content}"

    return schedule_message


def create_schedule_agent_task_tool(
    session_id: str,
    request_id: str | None,
    *,
    is_private: bool,
    bot_id: str | None,
    run_agent_task: Callable[..., object],
):
    @tool("schedule_agent_task")
    async def schedule_agent_task(
        task: str,
        delay_minutes: float = 0,
        delay_hours: float = 0,
    ) -> str:
        """
        安排 bot 在几分钟或几小时后重新进入 agent，并允许到点后调用可用工具完成任务。

        Args:
            task: 到点后要完成的任务描述，例如“查一下明天上海天气并提醒我带伞”。
            delay_minutes: 延迟多少分钟，可以是小数。
            delay_hours: 延迟多少小时，可以和 delay_minutes 同时使用。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消定时任务。"

        task = task.strip()
        if not task:
            return "定时 agent 任务内容为空，未创建任务。"

        delay_seconds, error = _validate_delay(
            delay_minutes, delay_hours, label=" agent 任务"
        )
        if error:
            return error

        run_at = datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)
        job_id = f"ai_groupmate_agent_schedule_{session_id}_{uuid.uuid4().hex}"

        scheduler.add_job(
            run_agent_task,
            "date",
            id=job_id,
            run_date=run_at,
            kwargs={
                "session_id": session_id,
                "task": task,
                "is_private": is_private,
                "bot_id": bot_id,
            },
            misfire_grace_time=300,
        )

        return f"定时 agent 任务已创建，将在 {run_at.strftime('%Y-%m-%d %H:%M:%S')} 执行：{task}"

    return schedule_agent_task
