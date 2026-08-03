import asyncio

from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_orm import get_session

from ..reply_guard import is_request_active
from ..group_memory import update_group_memory as _update_group_memory

_background_update_tasks: dict[str, asyncio.Task[None]] = {}


async def _run_group_memory_update(
    session_id: str,
    *,
    bot_name: str,
    reason: str,
    timeout_seconds: float,
) -> None:
    try:
        async with get_session() as db_session:
            result = await asyncio.wait_for(
                _update_group_memory(
                    db_session,
                    session_id,
                    bot_name=bot_name,
                ),
                timeout=timeout_seconds,
            )
        logger.info(
            f"群 {session_id} 档案后台维护结束（理由: {reason}）: {result}"
        )
    except TimeoutError:
        logger.warning(
            f"群 {session_id} 档案后台维护超时（{timeout_seconds:.1f}s，理由: {reason}）"
        )
    except asyncio.CancelledError:
        logger.info(f"群 {session_id} 档案后台维护任务已取消")
        raise
    except Exception:
        logger.exception(f"群 {session_id} 档案后台维护失败（理由: {reason}）")


def _remove_finished_task(session_id: str, task: asyncio.Task[None]) -> None:
    if _background_update_tasks.get(session_id) is task:
        _background_update_tasks.pop(session_id, None)


def start_group_memory_update(
    session_id: str,
    *,
    bot_name: str,
    reason: str,
    timeout_seconds: float,
) -> bool:
    existing_task = _background_update_tasks.get(session_id)
    if existing_task is not None and not existing_task.done():
        logger.info(f"群 {session_id} 已有档案后台维护任务，跳过重复入队")
        return False

    task = asyncio.create_task(
        _run_group_memory_update(
            session_id,
            bot_name=bot_name,
            reason=reason,
            timeout_seconds=timeout_seconds,
        ),
        name=f"group-memory-update:{session_id}",
    )
    _background_update_tasks[session_id] = task
    task.add_done_callback(
        lambda finished_task: _remove_finished_task(session_id, finished_task)
    )
    return True


def create_group_memory_tool(
    session_id: str,
    request_id: str | None,
    *,
    bot_name: str,
    timeout_seconds: float,
):
    @tool("update_group_memory")
    async def update_group_memory(reason: str) -> str:
        """自主维护当前群的认知档案。

        当近期聊天出现值得长期记住的新话题、成员特征、内部梗/黑话或群氛围变化时调用。
        reason 是本次值得更新的简短理由。普通闲聊和一次性事件不要调用。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消群档案更新。"

        normalized_reason = reason.strip() or "未提供"
        started = start_group_memory_update(
            session_id,
            bot_name=bot_name,
            reason=normalized_reason,
            timeout_seconds=timeout_seconds,
        )
        if not started:
            return "当前群已有档案后台维护任务，本次已去重；不要播报此结果。"
        return "群档案后台维护任务已启动，当前对话无需等待；不要播报此结果。"

    return update_group_memory
