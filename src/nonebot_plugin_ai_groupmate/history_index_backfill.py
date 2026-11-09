"""Incrementally index old history for conversations that need quoted lookup."""

from datetime import datetime

from sqlalchemy import Select, or_, and_, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .model import ChatHistory, HistoryMessageIndex, HistoryIndexProgress
from .message_metadata import platform_message_ids

DEFAULT_BATCH_SIZE = 500


async def has_pending_backfill(db, *, allowed_sessions: list[str] | None = None) -> bool:
    statement = Select(HistoryIndexProgress.session_id).where(HistoryIndexProgress.complete.is_(False))
    if allowed_sessions is not None:
        statement = statement.where(HistoryIndexProgress.session_id.in_(allowed_sessions))
    return await db.scalar(statement.limit(1)) is not None


async def backfill_one(db, *, batch_size: int = DEFAULT_BATCH_SIZE,
                       allowed_sessions: list[str] | None = None) -> tuple[str | None, int, bool]:
    """Index one bounded batch in the caller's transaction, locking rows during indexing."""
    if not 1 <= batch_size <= 1000:
        raise ValueError("invalid backfill batch size")
    dialect = db.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError("unsupported history index database")
    if dialect == "postgresql":
        await db.execute(text("SET LOCAL lock_timeout = '2s'"))
    pending = Select(HistoryIndexProgress).where(HistoryIndexProgress.complete.is_(False))
    if allowed_sessions is not None:
        pending = pending.where(HistoryIndexProgress.session_id.in_(allowed_sessions))
    progress = await db.scalar(pending.order_by(HistoryIndexProgress.updated_at, HistoryIndexProgress.session_id)
                               .limit(1).with_for_update(skip_locked=True))
    if progress is None:
        return None, 0, False
    statement = Select(ChatHistory.msg_id, ChatHistory.created_at, ChatHistory.content).where(
        ChatHistory.session_id == progress.session_id, ChatHistory.msg_id <= progress.cutoff_id,
    )
    if progress.last_created_at is not None:
        statement = statement.where(or_(ChatHistory.created_at > progress.last_created_at,
            and_(ChatHistory.created_at == progress.last_created_at, ChatHistory.msg_id > progress.last_history_id)))
    rows = (await db.execute(statement.order_by(ChatHistory.created_at, ChatHistory.msg_id)
                             .limit(batch_size).with_for_update())).all()
    values = [{"history_id": row.msg_id, "session_id": progress.session_id, "platform_id": value}
              for row in rows for value in platform_message_ids(row.content) if len(value) <= 255]
    if values:
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        await db.execute(insert(HistoryMessageIndex).values(values).on_conflict_do_nothing(
            index_elements=["history_id", "platform_id"]))
    if rows:
        progress.last_created_at = rows[-1].created_at
        progress.last_history_id = rows[-1].msg_id
    progress.complete = len(rows) < batch_size
    progress.updated_at = datetime.now()
    return progress.session_id, len(rows), progress.complete
