import re
import json
import difflib
import datetime
from typing import Any, Literal
from collections.abc import Callable

import jieba
from sqlalchemy import Select
from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_alconna import Target, UniMessage

from ..model import ChatHistory
from ..reply_guard import is_request_active


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

    def _result(status: Literal["sent", "skipped", "failed"], message: str) -> str:
        return json.dumps(
            {"status": status, "message": message},
            ensure_ascii=False,
        )

    @tool("reply_user")
    async def reply_user(
        content: str,
        next_step: Literal["end", "continue"],
    ) -> str:
        """
        向当前群聊发送文本回复。
        注意：如果你想对用户说话，必须调用这个工具。不要直接返回文本。
        Args:
            content: 你想发送的内容。
            next_step: 发送后是否还需要继续思考并发送下一条。单条回复填 end；
                只有下一条会提供新信息时填 continue，最后一条必须填 end。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return _result("skipped", "请求已过期，已取消发送。")

        if not content or not content.strip():
            return _result("failed", "内容为空，未发送。")

        try:
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
                _, _, latest_body = parse_msg_meta(latest_bot_msg.content)
                latest_normalized = _normalize_text(
                    latest_body or latest_bot_msg.content
                )
                recent = (
                    datetime.datetime.now() - latest_bot_msg.created_at
                    <= datetime.timedelta(seconds=90)
                )
                similarity = _semantic_similarity(latest_normalized, normalized_content)
                if recent and similarity >= 0.9:
                    logger.info(
                        f"检测到近义重复回复(相似度={similarity:.2f})，已自动跳过"
                    )
                    return _result("skipped", "检测到重复回复，已跳过发送。")

            name_to_id: dict[str, str] = {}
            members = group_members
            if members is None and interface is not None:
                try:
                    members = await interface.get_members(SceneType.GROUP, session_id)
                except Exception as e:
                    logger.warning(f"获取群成员失败，降级为纯文本发送: {e}")
                    members = None
            if members is not None:
                for member in members:
                    target_id = str(member.id)
                    aliases = {
                        getattr(member, "name", None),
                        getattr(member, "nick", None),
                        getattr(getattr(member, "user", None), "name", None),
                        getattr(getattr(member, "user", None), "nick", None),
                    }
                    for alias in aliases:
                        if alias:
                            name_to_id[str(alias)] = target_id

            at_pattern = re.compile(r"@([^\s@]+)")
            punctuation = "，。,.!！?？:：;；、)）]\"'”’"
            message: UniMessage | None = None

            def append_text(text: str):
                nonlocal message
                if not text:
                    return
                if message is None:
                    message = UniMessage.text(text)
                else:
                    message = message.text(text)

            def append_at(target_id: str) -> bool:
                nonlocal message
                try:
                    if message is None:
                        message = UniMessage.at(target_id)
                    else:
                        message = message.at(target_id)
                    return True
                except Exception:
                    return False

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
                    continue

                append_text(content[cursor:start])
                if not append_at(target_id):
                    append_text("@" + mention_name)
                append_text(suffix)
                cursor = end

            append_text(content[cursor:])

            if request_id is not None and not await is_request_active(
                session_id, request_id
            ):
                return _result("skipped", "请求已过期，已取消发送。")

            outgoing = message or UniMessage.text(content)
            res = await (
                outgoing.send(target=send_target)
                if send_target is not None
                else outgoing.send()
            )
            msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n" + content,
                user_name=bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"Bot已回复: {content}")
            return _result("sent", "消息已发送。")
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            await db_session.rollback()
            return _result("failed", f"发送失败: {e}")

    return reply_user
