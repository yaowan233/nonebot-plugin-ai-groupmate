import re
from typing import Any, Literal
from collections.abc import Callable

from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_alconna import Target, UniMessage

from ..model import ChatHistory
from .delivery import OutboundIdentity, send_and_finalize
from .chat_text import plain_chat_text, wants_original_format
from ..reply_guard import is_request_active
from .member_tools import member_aliases
from .tool_results import tool_failure, tool_skipped, tool_success
from .outbound_plan import compose_segments
from ..message_metadata import is_reliable_platform_id
from ..message_repository import find_by_platform_id, structured_id_in_scope


async def resolve_reply_target(db_session, session_id: str, reply_to: str | None, *, current_message_id: str | None = None, replied_message_id: str | None = None, outbound_identity: OutboundIdentity | None = None) -> str | None:
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
    if outbound_identity is not None:
        found = await structured_id_in_scope(db_session, session_id=session_id, message_id=target,
                                             platform=outbound_identity.platform, scope=outbound_identity.scope)
        if found is True:
            return target
        if found is False:
            raise ValueError("当前会话中找不到该消息，未执行引用回复。请核对平台消息ID，不要使用数据库记录ID。")
    if await find_by_platform_id(db_session, session_id, target):
        return target
    raise ValueError("当前会话中找不到该消息，未执行引用回复。请核对平台消息ID，不要使用数据库记录ID。")


def compose_image_reply(image: UniMessage, text: str = "", quoted_id: str | None = None) -> UniMessage:
    """Build one native message chain for an image, optional text, and a quote."""
    return compose_segments(([UniMessage.text(text)] if text else []) + [image], quoted_id)


def image_reply_rows(*, session_id: str, bot_name: str, message_id: str, image_content: str, text: str = "", quoted_id: str | None = None, media_id: int | None = None) -> list[ChatHistory]:
    header = f"id: {message_id}\n" + (f"回复id: {quoted_id}\n" if quoted_id is not None else "")
    rows = []
    if text:
        rows.append(ChatHistory(session_id=session_id, user_id=bot_name, user_name=bot_name, content_type="bot", content=header + text))
    rows.append(ChatHistory(session_id=session_id, user_id=bot_name, user_name=bot_name, content_type="bot", content=header + image_content, media_id=media_id))
    return rows


def record_image_reply(db_session, **kwargs) -> None:
    """Compatibility writer for callers that manage their own transaction."""
    for row in image_reply_rows(**kwargs):
        db_session.add(row)


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
    current_user_id: str | None = None,
    repeat_text: str | None = None,
    request_text: str = "",
    current_message_id: str | None = None,
    replied_message_id: str | None = None,
    recovery_enabled: bool = True,
    outbound_identity: OutboundIdentity | None = None,
):
    """
    核心工具：用于发送消息。
    """
    attempted_sends: set[tuple[str, str | None]] = set()

    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

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
            content: 你想发送的内容。提及本轮发言者用 @current；群内其他成员用@用户ID，ID后用空白或标点分隔；
                会生成平台原生@。未知ID或重名时先用query_platform的list_members查询，也支持唯一昵称。
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

        send_started = False
        try:
            try:
                quoted_id = await resolve_reply_target(db_session, session_id, reply_to, current_message_id=current_message_id, replied_message_id=replied_message_id, outbound_identity=outbound_identity)
            except ValueError as error:
                return tool_failure("invalid_reply_target", str(error), delivery_state="not_attempted")
            if repeat_text is None:
                content = plain_chat_text(content.strip(), request_text=request_text)
                if not wants_original_format(request_text) and "```" not in content and "~~~" not in content:
                    content = _dedupe_consecutive_lines(content)
            current_pattern = r"(?<![\w@])@current(?=$|[\s，。,.!！?？:：;；、)）\]\"'”’])"
            if re.search(current_pattern, content, re.IGNORECASE):
                if not current_user_id:
                    return tool_failure("invalid_mention_target", "没有可确认的本轮发言者，未发送。", delivery_state="not_attempted")
                content = re.sub(current_pattern, lambda _: "@" + str(current_user_id), content, flags=re.IGNORECASE)
            normalized_content = _normalize_text(content)
            if (normalized_content, quoted_id) in attempted_sends:
                return tool_skipped("duplicate_reply", "本轮已尝试发送该消息，请勿重复发送。", delivery_state="not_attempted")

            # Sending can block; release the pooled connection first.
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
            if current_user_id:
                name_to_id[str(current_user_id)] = str(current_user_id)

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
                        return tool_failure("invalid_mention_target", "当前群成员名单中无法确认该用户ID，未发送；请先用query_platform的list_members核对。", delivery_state="not_attempted")
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
            outgoing = compose_segments([outgoing], quoted_id)
            def record(msg_id: str) -> list[ChatHistory]:
                return [ChatHistory(
                    session_id=session_id, user_id=bot_name, content_type="bot",
                    content=f"id: {msg_id}\n" + (f"回复id: {quoted_id}\n" if quoted_id is not None else "") + content,
                    user_name=bot_name,
                )]
            attempted_sends.add((normalized_content, quoted_id))
            send_started = True
            options = {"target": send_target} if send_target is not None else {}
            delivery = await send_and_finalize(db_session, session_id, outgoing, send_options=options, record=record,
                                               recovery_enabled=recovery_enabled, outbound_identity=outbound_identity,
                                               reply_to_id=quoted_id)
            logger.info(f"Bot已回复: {content}")
            return tool_success(
                "message_sent",
                delivery.note or "消息已发送。",
                data={"message_id": delivery.message_id, "reply_to_message_id": quoted_id, "history_saved": delivery.history_saved},
                delivery_state="completed",
            )
        except Exception as error:
            logger.warning(f"发送消息异常: error_type={type(error).__name__}")
            try:
                await db_session.rollback()
            except Exception as rollback_error:
                logger.warning(f"发送事务回滚失败 error_type={type(rollback_error).__name__}")
            return tool_failure(
                "send_failed",
                "消息发送失败；投递结果可能未知，请勿立即重试。" if send_started else "消息尚未发送。",
                delivery_state="unknown" if send_started else "not_attempted",
            )

    return reply_user
