"""Maintain exact message IDs in the same transaction as new history rows."""

from sqlalchemy import event, delete, inspect

from .model import ChatHistory, HistoryMessageIndex
from .message_metadata import platform_message_ids


def _insert_ids(connection, history: ChatHistory) -> None:
    values = [{"history_id": history.msg_id, "session_id": history.session_id, "platform_id": value}
              for value in platform_message_ids(history.content) if len(value) <= 255]
    if values:
        connection.execute(HistoryMessageIndex.__table__.insert(), values)


@event.listens_for(ChatHistory, "after_insert")
def _index_new_history(_mapper, connection, history: ChatHistory) -> None:
    _insert_ids(connection, history)


@event.listens_for(ChatHistory, "after_update")
def _reindex_changed_history(_mapper, connection, history: ChatHistory) -> None:
    state = inspect(history)
    if not (state.attrs.content.history.has_changes() or state.attrs.session_id.history.has_changes()):
        return
    connection.execute(delete(HistoryMessageIndex).where(HistoryMessageIndex.history_id == history.msg_id))
    _insert_ids(connection, history)
