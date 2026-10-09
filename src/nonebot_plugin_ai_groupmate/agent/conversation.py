import datetime
from dataclasses import dataclass
from collections.abc import Callable

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import BaseMessage

from ..model import ChatHistory, ChatHistorySchema
from ..message_metadata import (
    BOT_ECHO_LOOKBACK,
    platform_message_ids,
    deduplicate_bot_echoes,
)

ACTIVE_THREAD_TTL = datetime.timedelta(minutes=10)
ACTIVE_THREAD_MAX_MESSAGES = 24


@dataclass
class ActiveConversationThread:
    messages: list[BaseMessage]
    last_msg_id: int
    updated_at: datetime.datetime


active_conversation_threads: dict[str, ActiveConversationThread] = {}

FormatHistory = Callable[
    [list[ChatHistorySchema], int, dict[str, str] | None, list[ChatHistorySchema] | None],
    list[BaseMessage],
]


def trim_thread_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    if len(messages) <= ACTIVE_THREAD_MAX_MESSAGES:
        return messages
    return messages[-ACTIVE_THREAD_MAX_MESSAGES:]


def get_active_thread(session_id: str) -> ActiveConversationThread | None:
    thread = active_conversation_threads.get(session_id)
    if not thread:
        return None
    if datetime.datetime.now() - thread.updated_at > ACTIVE_THREAD_TTL:
        active_conversation_threads.pop(session_id, None)
        return None
    return thread


def build_append_only_history(
    session_id: str,
    history: list[ChatHistorySchema],
    *,
    format_history: FormatHistory,
    user_roles: dict[str, str] | None = None,
    extra_inline_images: list[ChatHistorySchema] | None = None,
) -> tuple[list[BaseMessage], list[ChatHistorySchema], bool]:
    thread = get_active_thread(session_id)
    # Compare the full supplied window before slicing at last_msg_id, otherwise
    # an echo arriving after the previous bot turn loses its canonical evidence.
    history = deduplicate_bot_echoes(history)
    if not history:
        return [], [], False

    if not thread:
        return (
            format_history(history, 3, user_roles, extra_inline_images),
            history,
            False,
        )

    new_history = [msg for msg in history if msg.msg_id > thread.last_msg_id]
    if not new_history:
        return list(thread.messages), [], True

    new_messages = format_history(new_history, 3, user_roles, extra_inline_images)
    return list(thread.messages) + new_messages, new_history, True


async def update_active_thread(
    db_session: AsyncSession,
    session_id: str,
    base_messages: list[BaseMessage],
    input_max_msg_id: int,
    *,
    format_history: FormatHistory,
) -> None:
    await db_session.flush()
    new_rows = (
        (
            await db_session.execute(
                Select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .where(ChatHistory.msg_id > input_max_msg_id)
                .order_by(ChatHistory.msg_id.asc())
            )
        )
        .scalars()
        .all()
    )
    if new_rows:
        new_history = [ChatHistorySchema.model_validate(row) for row in new_rows]
        last_msg_id = max(msg.msg_id for msg in new_history)
        if any(
            msg.content_type != "bot" and platform_message_ids(msg.content)
            for msg in new_history
        ):
            # A delayed echo can sit beyond the input boundary while its bot
            # record is already in base_messages. Load bounded evidence only;
            # these older bot records must never be appended to the cache again.
            earliest = min(msg.created_at for msg in new_history) - BOT_ECHO_LOOKBACK
            latest = max(msg.created_at for msg in new_history) + BOT_ECHO_LOOKBACK
            earlier_bot_rows = (
                (
                    await db_session.execute(
                        Select(ChatHistory).where(
                            ChatHistory.session_id == session_id,
                            ChatHistory.content_type == "bot",
                            ChatHistory.msg_id <= input_max_msg_id,
                            ChatHistory.created_at >= earliest,
                            ChatHistory.created_at <= latest,
                        )
                    )
                )
                .scalars()
                .all()
            )
            earlier_bots = [ChatHistorySchema.model_validate(row) for row in earlier_bot_rows]
            new_history = [
                msg
                for msg in deduplicate_bot_echoes([*earlier_bots, *new_history])
                if msg.msg_id > input_max_msg_id
            ]
        base_messages = base_messages + format_history(new_history, 0, None, None)
    else:
        last_msg_id = input_max_msg_id

    active_conversation_threads[session_id] = ActiveConversationThread(
        messages=trim_thread_messages(base_messages),
        last_msg_id=last_msg_id,
        updated_at=datetime.datetime.now(),
    )
