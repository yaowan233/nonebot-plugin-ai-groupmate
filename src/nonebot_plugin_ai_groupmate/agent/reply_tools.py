import re
import difflib
import datetime
from typing import Any, Literal
from collections.abc import Callable

import jieba
from sqlalchemy import Select, or_
from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_alconna import Target, UniMessage

from ..model import ChatHistory
from ..reply_guard import is_request_active
from .member_tools import member_aliases
from .tool_results import tool_failure, tool_skipped, tool_success
from ..participation import record_group_delivery
from ..message_metadata import platform_message_ids, is_reliable_platform_id


async def resolve_reply_target(db_session, session_id: str, reply_to: str | None, *, current_message_id: str | None = None, replied_message_id: str | None = None) -> str | None:
    """Resolve a quote to a transport ID belonging to the bound conversation."""
    if reply_to is None:
        return None
    requested = reply_to.strip()
    target = current_message_id if requested == "current" else requested
    if not is_reliable_platform_id(target):
        raise ValueError("没有可引用的目标消息，请使用当前消息或当前会话中的有效消息ID。")
    target = str(target).strip()
    if target in {current_message_id, replied_message_id}:
        return target
    rows = await db_session.execute(
        Select(ChatHistory.content).where(
            ChatHistory.session_id == session_id,
            or_(
                ChatHistory.content.contains(f"id: {target}\n", autoescape=True),
                ChatHistory.content.contains(f"id:{target}\n", autoescape=True),
            ),
        ).order_by(ChatHistory.msg_id.asc()).limit(50)
    )
    if any(target in platform_message_ids(content) for content in rows.scalars().all()):
        return target
    raise ValueError("当前会话中找不到该消息，未执行引用回复。请核对平台消息ID，不要使用数据库记录ID。")


def compose_image_reply(image: UniMessage, text: str = "", quoted_id: str | None = None) -> UniMessage:
    """Build one native message chain for an image, optional text, and a quote."""
    message = UniMessage.text(text) + image if text else image
    return UniMessage.reply(quoted_id) + message if quoted_id is not None else message


def record_image_reply(db_session, *, session_id: str, bot_name: str, message_id: str, image_content: str, text: str = "", quoted_id: str | None = None, media_id: int | None = None) -> None:
    """Store multipart content under the single platform message's transport ID."""
    header = f"id: {message_id}\n" + (f"回复id: {quoted_id}\n" if quoted_id is not None else "")
    if text:
        db_session.add(ChatHistory(session_id=session_id, user_id=bot_name, user_name=bot_name, content_type="bot", content=header + text))
    db_session.add(ChatHistory(session_id=session_id, user_id=bot_name, user_name=bot_name, content_type="bot", content=header + image_content, media_id=media_id))


def create_reply_tool(
    db_session,
    session_id: str,
    request_id: str | None = None,
    interface: QryItrface | None = None,
    *,
    send_target: Target | None = None,
    bot_name: str,
    parse_msg_meta: Callable[[str], tuple[str | None, str | None, str]],
    group_members: list[Any] | None = None,
    repeat_text: str | None = None,
    current_message_id: str | None = None,
    replied_message_id: str | None = None,
):
    """
    核心工具：用于发送消息。
    """

    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _semantic_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        seq_ratio = difflib.SequenceMatcher(None, a, b).ratio()

        a_tokens = {t for t in jieba.lcut(a) if t.strip()}
        b_tokens = {t for t in jieba.lcut(b) if t.strip()}
        if not a_tokens or not b_tokens:
            return seq_ratio

        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        jaccard = inter / union if union else 0.0
        return max(seq_ratio, jaccard)

    def _dedupe_consecutive_lines(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text
        deduped: list[str] = []
        for line in lines:
            if deduped and deduped[-1] == line:
                continue
            deduped.append(line)
        return "\n".join(deduped)

    @tool("reply_user")
    async def reply_user(
        content: str,
        next_step: Literal["end", "continue"],
        reply_to: str | None = None,
    ) -> str:
        """
        向当前会话单独发送文字回复，不要直接返回正文。
        文字要和图片在同一条消息中时，使用发图工具的 text 参数，不要拆成两次发送。
        Args:
            content: 你想发送的内容。群内提及成员用@用户ID，ID后用空白或标点分隔；
                会生成平台原生@。未知ID或重名时先用get_group_members查询，也支持唯一昵称。
            reply_to: 可选引用回复。填 current 引用用户现在发出的这条消息；
                引用其他消息则传当前会话的已知平台消息ID。省略则普通发送，不要编造ID。
            next_step: 当前请求已处理并交付结果或明确说明阻塞后填 end；
                只有下一条会提供新信息时填 continue，最后一条必须填 end。
                end 表示本轮任务结束。不要发送“还在推/等下再回你”等空承诺；
                应先完成处理，给出结果，或明确说明无法完成及具体卡点。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return tool_skipped(
                "request_expired",
                "请求已过期，已取消发送。",
                delivery_state="not_attempted",
            )

        if not content or not content.strip():
            return tool_failure(
                "empty_content",
                "内容为空，未发送。",
                delivery_state="not_attempted",
            )

        if repeat_text is not None:
            expected = _normalize_text(repeat_text)
            actual = _normalize_text(content)
            if actual != expected or next_step != "end":
                return tool_failure(
                    "repeat_constraint_violation",
                    "当前是复读队形：只能原样复读并结束，或者调用 finish 保持沉默。",
                    delivery_state="not_attempted",
                )
            # 使用识别到的原文，避免模型添加不可见空白或改变换行。
            content = repeat_text

        try:
            try:
                quoted_id = await resolve_reply_target(db_session, session_id, reply_to, current_message_id=current_message_id, replied_message_id=replied_message_id)
            except ValueError as error:
                return tool_failure("invalid_reply_target", str(error), delivery_state="not_attempted")
            content = _dedupe_consecutive_lines(content.strip())
            normalized_content = _normalize_text(content)

            latest_bot_msg = (
                (
                    await db_session.execute(
                        Select(ChatHistory)
                        .where(
                            ChatHistory.session_id == session_id,
                            ChatHistory.content_type == "bot",
                        )
                        .order_by(ChatHistory.msg_id.desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if latest_bot_msg:
                _, latest_reply_id, latest_body = parse_msg_meta(latest_bot_msg.content)
                latest_normalized = _normalize_text(
                    latest_body or latest_bot_msg.content
                )
                recent = (
                    datetime.datetime.now() - latest_bot_msg.created_at
                    <= datetime.timedelta(seconds=90)
                )
                similarity = _semantic_similarity(latest_normalized, normalized_content)
                if recent and similarity >= 0.9 and latest_reply_id == quoted_id:
                    logger.info(
                        f"检测到近义重复回复(相似度={similarity:.2f})，已自动跳过"
                    )
                    return tool_skipped(
                        "duplicate_reply",
                        "检测到重复回复，已跳过发送。",
                        delivery_state="not_attempted",
                    )

            # The duplicate-check query has finished.  Sending through the
            # adapter can block, so release its pooled connection first.
            await db_session.commit()

            name_to_id: dict[str, str] = {}
            members = group_members
            if members is None and interface is not None:
                try:
                    members = await interface.get_members(SceneType.GROUP, session_id)
                except Exception as e:
                    logger.warning(f"获取群成员失败，降级为纯文本发送: {e}")
                    members = None
            if members is not None:
                aliases_to_ids: dict[str, set[str]] = {}
                for member in members:
                    target_id = str(member.id)
                    for alias in member_aliases(member):
                        aliases_to_ids.setdefault(alias, set()).add(target_id)
                name_to_id.update({alias: next(iter(ids)) for alias, ids in aliases_to_ids.items() if len(ids) == 1})
                name_to_id.update({str(member.id): str(member.id) for member in members})

            punctuation = "，。,.!！?？:：;；、)）]\"'”’"
            at_pattern = re.compile(rf"@(\d+(?=$|[\s{re.escape(punctuation)}])|[^\s@]+)")
            message: UniMessage | None = None

            def append_text(text: str):
                nonlocal message
                if not text:
                    return
                if message is None:
                    message = UniMessage.text(text)
                else:
                    message = message.text(text)

            def append_at(target_id: str):
                nonlocal message
                if message is None:
                    message = UniMessage.at(target_id)
                else:
                    message = message.at(target_id)

            cursor = 0
            for match in at_pattern.finditer(content):
                start, end = match.span()
                raw_name = match.group(1)
                mention_name = raw_name
                suffix = ""
                while mention_name and mention_name[-1] in punctuation:
                    suffix = mention_name[-1] + suffix
                    mention_name = mention_name[:-1]

                target_id = name_to_id.get(mention_name)
                if not target_id:
                    if mention_name.isdigit():
                        return tool_failure("invalid_mention_target", "当前群成员名单中无法确认该用户ID，未发送；请先用get_group_members核对。", delivery_state="not_attempted")
                    continue

                append_text(content[cursor:start])
                append_at(target_id)
                append_text(suffix)
                cursor = end

            append_text(content[cursor:])

            if request_id is not None and not await is_request_active(
                session_id, request_id
            ):
                return tool_skipped(
                    "request_expired",
                    "请求已过期，已取消发送。",
                    delivery_state="not_attempted",
                )

            outgoing = message or UniMessage.text(content)
            if quoted_id is not None:
                outgoing = UniMessage.reply(quoted_id) + outgoing
            res = await (
                outgoing.send(target=send_target)
                if send_target is not None
                else outgoing.send()
            )
            record_group_delivery(session_id)
            msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n" + (f"回复id: {quoted_id}\n" if quoted_id is not None else "") + content,
                user_name=bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"Bot已回复: {content}")
            return tool_success(
                "message_sent",
                "消息已发送。",
                data={"message_id": str(msg_id), "reply_to_message_id": quoted_id},
                delivery_state="completed",
            )
        except Exception as error:
            logger.warning(f"发送消息异常: error_type={type(error).__name__}")
            await db_session.rollback()
            return tool_failure(
                "send_failed",
                "消息发送失败；投递结果可能未知，请勿立即重试。",
                delivery_state="unknown",
            )

    return reply_user
