import datetime
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Select
from nonebot.log import logger
from langchain.tools import tool
from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna import message_recall

from ..model import ChatHistory, StructuredMessage
from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from .history_format import parse_msg_meta
from .structured_context import StructuredSnapshot, visible_text

SELF_RECALL_WINDOW = datetime.timedelta(minutes=5)


@dataclass(frozen=True)
class RecallCandidate:
    message_id: str
    sender_name: str
    body: str
    is_self: bool
    created_at: datetime.datetime


def _extract_stored_message_id(content: str) -> str | None:
    first_line = content.splitlines()[0] if content else ""
    if not first_line.startswith("id:"):
        return None
    message_id = first_line.split(":", 1)[1].strip()
    if not message_id or message_id in {"system", "unknown"}:
        return None
    return message_id


def create_recall_message_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    *,
    bot_name: str,
    has_admin_permission: bool,
    bot: Bot | None,
    event: Event | None,
    reply_to_id: str | None = None,
    structured_platform: str | None = None,
    is_private: bool = False,
    bot_id: str | None = None,
):
    async def _recent_history() -> list[ChatHistory]:
        return list(
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(ChatHistory.session_id == session_id)
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(300)
                )
            )
            .scalars()
            .all()
        )

    async def _recent_candidates() -> list[RecallCandidate]:
        history = await _recent_history()
        if structured_platform is None:
            return [RecallCandidate(mid, getattr(row, "user_name", ""), parse_msg_meta(row.content)[2],
                                    row.content_type == "bot" and str(row.user_id) == str(bot_name), row.created_at)
                    for row in history if (mid := _extract_stored_message_id(row.content))]

        scope = "private" if is_private else "group"
        legacy_ids = {mid for row in history if (mid := _extract_stored_message_id(row.content))}
        # Resolve every legacy ID in both scopes so a record from another scope
        # can never authorize a recall through the compatibility path.
        async with db_session.begin_nested():
            scoped = list((await db_session.scalars(Select(StructuredMessage).where(
                StructuredMessage.platform == structured_platform, StructuredMessage.scope == scope,
                StructuredMessage.session_id == session_id,
            ).order_by(StructuredMessage.created_at.desc(), StructuredMessage.id.desc()).limit(300))).all())
            known = list((await db_session.scalars(Select(StructuredMessage).where(
                StructuredMessage.platform == structured_platform,
                StructuredMessage.scope.in_(("group", "private")),
                StructuredMessage.session_id == session_id, StructuredMessage.platform_id.in_(legacy_ids),
            ))).all()) if legacy_ids else []

        by_id = {row.platform_id: row for row in [*scoped, *known] if row.scope == scope}
        known_ids = {row.platform_id for row in known}
        candidates = []
        for row in by_id.values():
            if row.sender_kind not in {"user", "bot"} or not isinstance(row.parts, list):
                continue
            parts = tuple(dict(part) for part in row.parts if isinstance(part, dict))
            body = visible_text(StructuredSnapshot(row.platform_id, row.sender_id, row.reply_to_id, parts, {}))
            if body is None:
                # A video/file/unknown segment must not hide accompanying text.
                body = "".join(str(part.get("text", "")) for part in parts if part.get("type") in {"text", "card"})
            candidates.append(RecallCandidate(
                row.platform_id, row.sender_name, body,
                row.sender_kind == "bot" and bool(bot_id) and row.sender_id == str(bot_id), row.created_at,
            ))
        for row in history:
            mid = _extract_stored_message_id(row.content)
            if not mid or mid in by_id or mid in known_ids:
                continue
            candidates.append(RecallCandidate(
                mid, row.user_name, parse_msg_meta(row.content)[2],
                row.content_type == "bot" and str(row.user_id) == str(bot_name), row.created_at,
            ))
        return sorted(candidates, key=lambda row: row.created_at, reverse=True)

    @tool("recall_message")
    async def recall_message(
        target: Literal["latest_self", "reply", "content"] = "latest_self",
        target_text: str | None = None,
        sender_name: str | None = None,
        reason: str | None = None,
    ) -> str:
        """
        撤回当前会话历史中的一条消息。

        群聊管理员/群主权限下可以撤回他人消息；私聊或群聊普通权限下只能撤回 bot 自己发送且 5 分钟内的消息。

        Args:
            target: latest_self 撤回 bot 最近一条消息；reply 撤回用户本次引用的消息；content 按正文查找。
            target_text: target=content 时必填，原样摘取目标消息正文的完整内容或独特片段；匹配多条时不会撤回。
            sender_name: target=content 时可选，按聊天记录中的发送者名称进一步限定。
            reason: 撤回原因，用于日志和历史记录。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return tool_skipped(
                "request_expired",
                "请求已过期，已取消撤回。",
                delivery_state="not_attempted",
            )

        if bot is None or event is None:
            return tool_failure(
                "missing_context",
                "撤回失败：缺少 bot/event 上下文。",
                delivery_state="not_attempted",
            )

        if target == "reply" and not reply_to_id:
            return tool_failure(
                "missing_reply",
                "用户本次没有引用消息，请按消息正文指定目标。",
                delivery_state="not_attempted",
            )
        target_text = (target_text or "").strip()
        if target == "content" and not target_text:
            return tool_failure(
                "missing_target_text", "请提供要撤回的消息正文或独特片段。",
                delivery_state="not_attempted",
            )
        try:
            recent = await _recent_candidates()
        except Exception as error:
            logger.warning(f"读取撤回目标失败 error_type={type(error).__name__}")
            return tool_failure("history_unavailable", "撤回失败：暂时无法读取当前会话记录。", retryable=True,
                                delivery_state="not_attempted")
        candidates: dict[str, RecallCandidate] = {}
        for row in recent:
            message_id = row.message_id
            if target == "latest_self":
                matches = row.is_self
            elif target == "reply":
                matches = message_id == str(reply_to_id)
            else:
                matches = target_text in row.body
                if sender_name:
                    matches = matches and row.sender_name == sender_name
            if matches:
                candidates.setdefault(message_id, row)
                if target == "latest_self":
                    break
        if len(candidates) > 1:
            return tool_failure(
                "ambiguous_message", "有多条消息匹配，未执行撤回。请补充更完整的正文、发送者名称，或请用户引用目标消息。",
                delivery_state="not_attempted",
            )
        target_msg_id, history = next(iter(candidates.items()), (None, None))
        if history is None:
            return tool_failure(
                "message_not_found",
                "撤回失败：没有在当前会话历史中找到这条消息。",
                retryable=True,
                delivery_state="not_attempted",
            )

        is_bot_message = history.is_self
        if not has_admin_permission:
            if not is_bot_message:
                return tool_failure(
                    "permission_denied",
                    "当前会话权限只允许撤回 bot 自己发送的消息。",
                    delivery_state="not_attempted",
                )
            if datetime.datetime.now() - history.created_at > SELF_RECALL_WINDOW:
                return tool_failure(
                    "recall_window_expired",
                    "当前会话权限只允许撤回 bot 自己 5 分钟内发送的消息。",
                    delivery_state="not_attempted",
                )

        try:
            # The history lookup is complete.  Recalling through OneBot is
            # external I/O and must not occupy a database connection.
            await db_session.commit()
            await message_recall(message_id=target_msg_id, event=event, bot=bot)
        except ValueError as e:
            logger.warning(f"撤回消息失败，消息 ID 不被当前适配器支持: {target_msg_id} {e}")
            return tool_failure(
                "unsupported_message_id",
                "撤回失败：当前适配器不支持这个消息 ID 格式。",
                delivery_state="not_attempted",
            )
        except Exception as error:
            logger.warning(
                "撤回消息失败: "
                f"message_id={target_msg_id}, error_type={type(error).__name__}"
            )
            return tool_failure(
                "recall_failed",
                "撤回接口调用失败；操作结果可能未知，请勿立即重试。",
                delivery_state="unknown",
            )

        action_scope = "管理员撤回" if has_admin_permission else "撤回自己消息"
        chat_history = ChatHistory(
            session_id=session_id,
            user_id=bot_name,
            content_type="bot",
            content=(
                "id: system\n"
                f"已执行{action_scope}: "
                f"reason={reason or '未填写原因'}"
            ),
            user_name=bot_name,
        )
        db_session.add(chat_history)
        logger.info(
            f"已执行{action_scope}: message_id={target_msg_id}, reason={reason or '未填写原因'}"
        )
        return tool_success(
            "message_recalled",
            "已撤回目标消息。",
            data={"scope": action_scope},
            delivery_state="completed",
        )

    return recall_message
