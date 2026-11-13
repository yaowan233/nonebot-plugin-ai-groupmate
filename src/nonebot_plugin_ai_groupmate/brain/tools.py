"""Submit a user's current cross-group request to the coordinator."""

import json
import asyncio
from uuid import uuid4
from typing import Literal
from datetime import datetime, timezone, timedelta

from sqlalchemy import func, select
from langchain.tools import tool
from nonebot_plugin_orm import get_session

from ..model import BrainTask, BrainAction, ChatHistory
from .repository import TaskInput, submit
from .coordinator import review_direct_send
from ..reply_guard import is_request_active
from .observations import configured_space
from ..message_metadata import parse_message_metadata, is_reliable_platform_id
from ..agent.group_access import ReachableGroups, linked_groups, check_linked_group
from ..agent.tool_results import tool_failure, tool_skipped, tool_success
from ..cross_group_privacy import (
    public_context_text,
    asks_for_group_records,
    asks_for_personal_data,
    contains_personal_data,
    explicitly_shares_own_data,
)

_submission_lock = asyncio.Lock()


def create_coordination_tool(*, config, platform: str, bot_id: str, scene_id: str, user_id: str,
                             message_id: str | None, request_id: str | None, request_text: str,
                             reachable_groups: ReachableGroups | None = None,
                             review_model=None, source_history=(), user_name: str = ""):
    @tool("request_cross_group_coordination")
    async def request_cross_group_coordination(reason: str, operation: Literal["ask", "send"] = "ask",
                                               target_group_id: str = "", message: str = "") -> str:
        """按用户当前要求去联动群询问或发送消息；reason 说明结合当前对话理解的完整意图。

        询问并等待回答用 ask；只发消息用 send，填写目标群号 target_group_id 和实际要发送的 message。
        用户允许自行拟文时可以自然拟一句；message 只填正文，系统会注明来源和发起人。
        目标不明确先用 query_platform 的 list_groups 查找或询问。
        发送前会结合两群聊天判断，拒绝时把原因自然告知用户；登记不代表发出。
        """
        if request_id and not await is_request_active(scene_id, request_id):
            return tool_skipped("request_expired", "当前请求已过期，未创建任务。")
        if not configured_space(config.cross_group_spaces, scene_id):
            return tool_failure("space_unavailable", "当前群没有启用跨群联动。")
        if platform == "unknown" or not is_reliable_platform_id(message_id) or not request_text.strip():
            return tool_failure("source_unavailable", "当前消息无法可靠关联，未创建任务。")
        target = target_group_id.strip()
        content = message.strip()
        if operation == "ask" and asks_for_group_records(request_text):
            return tool_failure("cross_group_history_private", "不能替其他群成员索取或搬运该群的聊天记录。")
        if operation == "ask" and (contains_personal_data(request_text) or asks_for_personal_data(request_text)):
            return tool_failure("privacy_protected", "跨群询问涉及个人敏感信息，这次没有提交。")
        if operation == "send":
            if not target or target == scene_id or not content or len(content) > 1000 or "\x00" in content:
                return tool_failure("invalid_send", "请提供另一个联动群的群号及不超过1000字的发送内容。")
            error = await check_linked_group(scene_id, target, lambda: config.cross_group_spaces, reachable_groups)
            if error:
                return tool_failure(error, "目标群未联动，或无法确认当前账号是否在目标群。")
            if request_id and not await is_request_active(scene_id, request_id):
                return tool_skipped("request_expired", "当前请求已过期，未创建任务。")
            if contains_personal_data(content) and not explicitly_shares_own_data(request_text, content):
                return tool_failure("privacy_protected", "这段内容涉及个人敏感信息，无法确认本人明确授权，这次不跨群转发。")
            label = " ".join(user_name.split())[:80]
            if not label or contains_personal_data(label):
                label = "一位群友"
            outgoing = f"另一群的 {label} 托我转达：{content}"
            async with get_session() as db:
                existing = await db.scalar(select(BrainTask).where(
                    BrainTask.brain_id == bot_id, BrainTask.platform == platform,
                    BrainTask.source_scene == scene_id, BrainTask.source_message == str(message_id),
                ))
                if existing is not None:
                    prior = await db.scalar(select(BrainAction).where(
                        BrainAction.task_id == existing.task_id, BrainAction.logical_step == "send",
                    ))
                    if prior is None or prior.target_scene != target or prior.content != outgoing:
                        return tool_failure("task_conflict", "本条请求已有不同的跨群任务，请勿重复提交。")
                    return tool_success("coordination_queued", f"跨群任务状态：{existing.status}。登记不代表已发送；确认发送后会自动通知。",
                                        data={"task_id": existing.task_id, "created": False, "status": existing.status, "operation": operation})
            if review_model is None:
                return tool_failure("review_unavailable", "暂时无法判断是否适合跨群传话，这次没有发送。", retryable=True)

            def context(rows):
                result = []
                for row in rows:
                    if row.content_type not in {"text", "bot"}:
                        continue
                    _, _, body = parse_message_metadata(row.content or "")
                    if body.strip():
                        safe_body = public_context_text(body.strip())
                        result.append({"sender": row.user_name[:80] if safe_body == body.strip() else "[已隐藏]", "text": safe_body[:350],
                                       "time": row.created_at.isoformat(timespec="seconds")})
                return result[-10:]

            async with get_session() as db:
                target_rows = list((await db.scalars(select(ChatHistory).where(
                    ChatHistory.session_id == target, ChatHistory.content_type.in_(("text", "bot")),
                ).order_by(ChatHistory.msg_id.desc()).limit(10))).all())
                target_context = context(reversed(target_rows))
            try:
                review = await asyncio.wait_for(review_direct_send(
                    review_model, request=request_text, intent=reason, message=content,
                    source_scene=scene_id, target_scene=target,
                    source_messages=context(source_history), target_messages=target_context, requester=label,
                ), timeout=15)
            except Exception:
                return tool_failure("review_unavailable", "暂时无法判断是否适合跨群传话，这次没有发送。", retryable=True)
            if not review.send:
                reason_text = ("可能会重复刷屏，这次先不转发。" if any(word in review.reason for word in ("重复", "刷屏"))
                               else "我觉得现在不太合适跨群转达，就先不发了。")
                return tool_failure("relay_declined", reason_text)
            content = outgoing
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
                task_request = json.dumps({"current_user_message": request_text[:3000], "conversation_context": reason[:800]}, ensure_ascii=False)
                item = TaskInput(brain_id=bot_id, platform=platform, source_scene=scene_id,
                                 source_message=str(message_id), source_user=user_id, request=task_request,
                                 deadline=now + timedelta(minutes=2 if operation == "send" else 30), transport_bot_id=bot_id)
                task, created = await submit(db, item, now=now)
                if not created and operation == "ask":
                    prior_send = await db.scalar(select(BrainAction).where(BrainAction.task_id == task.task_id, BrainAction.logical_step == "send"))
                    if prior_send is not None:
                        await db.rollback()
                        return tool_failure("task_conflict", "本条请求已有不同的跨群任务，请勿重复提交。")
                if operation == "send":
                    if created:
                        db.add(BrainAction(action_id=uuid4().hex, task_id=task.task_id, logical_step="send", effect_slot="message",
                                           platform=platform, target_scene=target, content=content, status="prepared",
                                           claim_token=None, lease_until=None, receipt_message_id=None, created_at=now, updated_at=now))
                    else:
                        prior = await db.scalar(select(BrainAction).where(BrainAction.task_id == task.task_id, BrainAction.logical_step == "send"))
                        if prior is None or prior.target_scene != target or prior.content != content:
                            await db.rollback()
                            return tool_failure("task_conflict", "本条请求已有不同的跨群任务，请勿重复提交。")
                    if target not in linked_groups(scene_id, lambda: config.cross_group_spaces):
                        await db.rollback()
                        return tool_failure("group_not_allowed", "目标群已不在当前群的联动范围。")
                task_id, status = task.task_id, task.status
                if request_id and not await is_request_active(scene_id, request_id):
                    await db.rollback()
                    return tool_skipped("request_expired", "当前请求已过期，未创建任务。")
                await db.commit()
        return tool_success("coordination_queued", f"跨群任务状态：{status}。登记不代表已发送；确认发送后会自动通知。",
                            data={"task_id": task_id, "created": created, "status": status, "operation": operation})

    return request_cross_group_coordination
