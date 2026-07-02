import re
import random
from typing import Literal

from nonebot.log import logger
from langchain.tools import tool
from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna import message_reaction

from ..model import ChatHistory
from ..reply_guard import is_request_active

ReactionMood = Literal[
    "like",
    "laugh",
    "surprise",
    "sad",
    "angry",
    "ok",
    "love",
    "question",
    "awkward",
    "clap",
    "plead",
    "thanks",
    "good_job",
    "shock",
    "smirk",
    "tease",
    "proud",
    "excited",
    "unhappy",
]

REACTION_EMOJI_MAP: dict[str, tuple[str, ...]] = {
    "like": ("76", "201", "389", "424"),
    "laugh": ("178", "182", "193", "387", "378"),
    "surprise": ("0", "180", "424"),
    "sad": ("5", "9", "15", "38", "49", "107", "194", "379", "382"),
    "angry": ("11", "326", "365"),
    "ok": ("124", "377", "381", "398"),
    "love": ("66", "305", "319", "383"),
    "question": ("32", "367"),
    "awkward": ("10", "27", "264"),
    "clap": ("99", "375"),
    "plead": ("106", "111"),
    "thanks": ("118", "63", "78", "409"),
    "good_job": ("356", "299", "306", "353", "380", "424"),
    "shock": ("26", "325"),
    "smirk": ("20", "101", "178"),
    "tease": ("102", "103", "178", "271"),
    "proud": ("4", "16", "306"),
    "excited": ("180", "400", "401"),
    "unhappy": ("15", "194"),
}


def is_onebot_context(bot: Bot | None, event: Event | None) -> bool:
    candidates: list[str] = []
    for obj in (bot, event):
        if obj is None:
            continue
        candidates.extend(
            [
                type(obj).__module__,
                type(obj).__qualname__,
                str(getattr(obj, "type", "")),
            ]
        )

    adapter = getattr(bot, "adapter", None) if bot is not None else None
    if adapter is not None:
        candidates.extend([type(adapter).__module__, type(adapter).__qualname__])
        get_name = getattr(adapter, "get_name", None)
        if callable(get_name):
            try:
                candidates.append(str(get_name()))
            except Exception:
                pass

    return any("onebot" in candidate.lower() for candidate in candidates)


def create_reaction_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    bot_name: str,
    bot: Bot | None,
    event: Event | None,
):
    """
    创建 OneBot 消息表情回复工具，底层使用 nonebot_plugin_alconna.message_reaction。
    """

    def _normalize_reaction_message_id(raw_message_id: str | None) -> str | None:
        if not raw_message_id:
            return None

        candidate = str(raw_message_id).strip()
        if not candidate:
            return None

        if candidate.lower() in {
            "current_event",
            "current",
            "event",
            "none",
            "null",
            "system",
        }:
            logger.debug(f"忽略无效表情回复目标消息占位符: {candidate}")
            return None

        meta_match = re.search(r"\bid[:：]\s*([A-Za-z0-9_.-]{1,128})", candidate)
        if meta_match:
            return meta_match.group(1)

        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate):
            return candidate

        logger.debug(f"忽略不像消息 id 的表情回复目标: {candidate[:80]}")
        return None

    def _get_event_message_id() -> str | None:
        if event is None:
            return None
        message_id = getattr(event, "message_id", None)
        if message_id is None:
            return None
        message_id = str(message_id).strip()
        return message_id or None

    def _is_supported_onebot_reaction_message_id(message_id: str | None) -> bool:
        return bool(message_id and message_id.isdigit())

    @tool("add_message_reaction")
    async def add_message_reaction(
        mood: ReactionMood,
        target_msg_id: str | None = None,
        delete: bool = False,
        emoji: str | None = None,
        count: int = 1,
    ) -> str:
        """
        给某条消息添加或取消表情回复。

        当只需要用一个表情表达态度时使用这个工具，不要再调用 reply_user 发送重复文本。

        Args:
            mood: 表情回复表达的态度。基础 mood: like/laugh/surprise/sad/angry/ok。扩展 mood: love/question/awkward/clap/plead/thanks/good_job/shock/smirk/tease/proud/excited/unhappy。
            target_msg_id: 目标消息 id，来自聊天记录里的 "id: xxxxx"。通常不要传；不传时默认给当前触发 bot 回复的这条消息添加。
            delete: 是否取消这个表情回复，默认 False。
            emoji: 可选的适配器原始表情 ID/名称覆盖值。通常不要传，除非明确知道平台支持的表情 ID。
            count: 同一 mood 下连续添加几个不同表情，范围 1-3。传了 emoji 时只添加该 emoji。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消表情回复。"

        mood_key = str(mood).strip()
        mood_emojis = REACTION_EMOJI_MAP.get(mood_key)
        reaction_count = max(1, min(int(count or 1), 3))
        if emoji:
            reaction_emojis = [str(emoji).strip()]
        elif mood_emojis:
            reaction_emojis = random.sample(
                list(mood_emojis), k=min(reaction_count, len(mood_emojis))
            )
        else:
            reaction_emojis = []
        reaction_emojis = [item for item in reaction_emojis if item]

        if not reaction_emojis:
            supported = ", ".join(REACTION_EMOJI_MAP)
            return f"表情回复失败: 未知 mood {mood_key!r}，可选值: {supported}"

        if bot is None or event is None:
            return "表情回复失败: 缺少 bot/event 上下文，无法调用 alconna message_reaction。"
        if not is_onebot_context(bot, event):
            return "表情回复失败: 当前适配器不是 OneBot，不支持消息表情回复。"

        message_id = _normalize_reaction_message_id(target_msg_id)
        if message_id and not _is_supported_onebot_reaction_message_id(message_id):
            logger.info(
                f"跳过表情回复：当前 OneBot reaction 接口不支持非数字消息 ID {message_id!r}"
            )
            return "表情回复跳过: 目标消息 ID 不是数字，底层 OneBot reaction 接口不支持。"
        if not message_id:
            event_message_id = _get_event_message_id()
            if not _is_supported_onebot_reaction_message_id(event_message_id):
                logger.info(
                    f"跳过表情回复：当前 OneBot reaction 接口不支持非数字消息 ID {event_message_id!r}"
                )
                return "表情回复跳过: 当前平台的消息 ID 不是数字，底层 OneBot reaction 接口不支持。"
        if not message_id:
            logger.debug("未指定表情回复目标消息，使用当前触发事件的消息 id")

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消表情回复。"

        async def _apply_reaction(
            reaction_emoji: str, target_message_id: str | None
        ) -> None:
            await message_reaction(
                reaction_emoji,
                message_id=target_message_id,
                event=event,
                bot=bot,
                delete=delete,
            )

        try:
            for reaction_emoji in reaction_emojis:
                try:
                    await _apply_reaction(reaction_emoji, message_id)
                except Exception as e:
                    if message_id and "msg not found" in str(e).lower():
                        logger.warning(
                            f"表情回复目标消息 {message_id} 不存在，回退到当前触发事件消息"
                        )
                        message_id = None
                        await _apply_reaction(reaction_emoji, None)
                    else:
                        raise
            action = "取消" if delete else "添加"
            reacted_message_desc = f"消息 {message_id}" if message_id else "当前触发消息"
            reaction_emoji_text = ",".join(reaction_emojis)
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=bot_name,
                content_type="bot",
                content=f"id: system\n已对{reacted_message_desc} {action}表情回复: mood={mood_key}, emoji={reaction_emoji_text}",
                user_name=bot_name,
            )
            db_session.add(chat_history)
            logger.info(
                f"已对{reacted_message_desc} {action}表情回复 mood={mood_key}, emoji={reaction_emoji_text}"
            )
            return f"已对{reacted_message_desc} {action}表情回复 mood={mood_key}, emoji={reaction_emoji_text}"
        except Exception as e:
            logger.error(f"表情回复工具执行失败: {e}")
            return f"表情回复失败: {e}"

    return add_message_reaction
