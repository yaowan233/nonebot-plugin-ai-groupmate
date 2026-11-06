"""Session-scoped access to messages stored in the legacy history format.

Only this module knows how to locate transport IDs inside old records. Callers
must not infer identity from message text or from IDs in other sessions.
"""

from datetime import datetime

from sqlalchemy import Select, func
from nonebot.log import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .model import ChatHistory, StructuredMessage, HistoryMessageIndex, HistoryIndexProgress
from .message_metadata import platform_message_ids, is_reliable_platform_id


async def structured_id_in_scope(
    db_session: AsyncSession, *, session_id: str, message_id: str, platform: str, scope: str,
) -> bool | None:
    """Resolve a new message without parsing legacy content; None means unknown."""
    try:
        async with db_session.begin_nested():
            rows = (await db_session.execute(Select(StructuredMessage.scope).where(
                StructuredMessage.platform == platform, StructuredMessage.scope.in_(("group", "private")),
                StructuredMessage.session_id == session_id, StructuredMessage.platform_id == message_id,
            ))).all()
        if not rows:
            return None
        return any(row.scope == scope for row in rows)
    except Exception as error:
        logger.warning(f"查询结构化消息失败，无法安全确认引用范围 error_type={type(error).__name__}")
        raise ValueError("暂时无法确认消息所属会话，请稍后再试。") from None


async def find_by_platform_id(
    db_session: AsyncSession, session_id: str, message_id: str | None
) -> list[ChatHistory]:
    if not is_reliable_platform_id(message_id):
        return []
    target = str(message_id).strip()
    if "\n" in target or "\r" in target:
        return []
    indexed = await db_session.execute(
        Select(ChatHistory).join(HistoryMessageIndex, HistoryMessageIndex.history_id == ChatHistory.msg_id)
        .where(HistoryMessageIndex.session_id == session_id, HistoryMessageIndex.platform_id == target,
               ChatHistory.session_id == session_id)
        .order_by(ChatHistory.msg_id.asc())
    )
    matches = [row for row in indexed.scalars().all() if target in platform_message_ids(row.content)]
    coverage = await db_session.get(HistoryIndexProgress, session_id, populate_existing=True)
    if coverage is not None and coverage.complete and len(target) <= 255:
        return matches
    if coverage is None:
        cutoff = await db_session.scalar(Select(func.max(ChatHistory.msg_id))) or 0
        values = {"session_id": session_id, "cutoff_id": cutoff, "last_created_at": None,
                  "last_history_id": 0, "complete": False, "updated_at": datetime.now()}
        dialect = db_session.get_bind().dialect.name
        if dialect in {"sqlite", "postgresql"}:
            insert = sqlite_insert if dialect == "sqlite" else pg_insert
            await db_session.execute(insert(HistoryIndexProgress).values(**values).on_conflict_do_nothing(index_elements=["session_id"]))
        else:
            db_session.add(HistoryIndexProgress(**values))
    # This is a candidate filter only. Exact matching below excludes reply IDs,
    # body text, partial IDs and user-supplied lookalikes. The session predicate
    # also prevents a message ID from resolving across conversations.
    result = await db_session.execute(
        Select(ChatHistory)
        .where(
            ChatHistory.session_id == session_id,
            ChatHistory.content.contains(target, autoescape=True),
        )
        .order_by(ChatHistory.msg_id.asc())
    )
    # During incremental rollout older rows have no index entries. A transport
    # ID can be reused, so an indexed hit alone does not prove that every part
    # of the historical message was indexed. Merge until coverage is verified.
    legacy = [row for row in result.scalars().all() if target in platform_message_ids(row.content)]
    return sorted({row.msg_id: row for row in [*matches, *legacy]}.values(), key=lambda row: row.msg_id)
