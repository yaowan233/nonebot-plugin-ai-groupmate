"""Submit a user's current cross-group request to the coordinator."""

import asyncio
from datetime import datetime, timezone, timedelta

from sqlalchemy import func, select
from langchain.tools import tool
from nonebot_plugin_orm import get_session

from ..model import BrainTask
from .repository import TaskInput, submit
from ..reply_guard import is_request_active
from .observations import configured_space
from ..message_metadata import is_reliable_platform_id
from ..agent.tool_results import tool_failure, tool_skipped, tool_success

_submission_lock = asyncio.Lock()


def create_coordination_tool(*, config, platform: str, bot_id: str, scene_id: str, user_id: str,
                             message_id: str | None, request_id: str | None, request_text: str):
    @tool("request_cross_group_coordination")
    async def request_cross_group_coordination(reason: str) -> str:
        """当本轮用户明确要求去另一个联动群询问时提交跨群任务；reason 简述任务目的。

        不用于普通闲聊、推测的跨群联系或历史消息中的要求。目标不明确时应先询问用户。
        """
        if request_id and not await is_request_active(scene_id, request_id):
            return tool_skipped("request_expired", "当前请求已过期，未创建任务。")
        if not configured_space(config.cross_group_spaces, scene_id):
            return tool_failure("space_unavailable", "当前群没有启用跨群联动。")
        if platform == "unknown" or not is_reliable_platform_id(message_id) or not request_text.strip():
            return tool_failure("source_unavailable", "当前消息无法可靠关联，未创建任务。")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with _submission_lock:
            async with get_session() as db:
                existing = await db.scalar(select(BrainTask).where(
                    BrainTask.brain_id == bot_id, BrainTask.platform == platform,
                    BrainTask.source_scene == scene_id, BrainTask.source_message == str(message_id),
                ))
                count = await db.scalar(select(func.count()).select_from(BrainTask).where(
                    BrainTask.created_at >= now - timedelta(hours=1), BrainTask.created_at <= now,
                ))
                if existing is None and (count or 0) >= 6:
                    return tool_failure("coordination_budget_exhausted", "跨群协调暂时繁忙，请稍后再试。")
                item = TaskInput(brain_id=bot_id, platform=platform, source_scene=scene_id,
                                 source_message=str(message_id), source_user=user_id, request=request_text[:4000],
                                 deadline=now + timedelta(minutes=30), transport_bot_id=bot_id)
                task, created = await submit(db, item, now=now)
                task_id, status = task.task_id, task.status
                if request_id and not await is_request_active(scene_id, request_id):
                    await db.rollback()
                    return tool_skipped("request_expired", "当前请求已过期，未创建任务。")
                await db.commit()
        return tool_success("coordination_queued", f"跨群任务状态：{status}。登记成功不代表已经发送或收到答复。",
                            data={"task_id": task_id, "created": created, "status": status})

    return request_cross_group_coordination
