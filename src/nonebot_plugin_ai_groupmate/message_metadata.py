"""Interpret stored transport headers without inferring a sender from its nickname."""

import re
from typing import TypeVar, Protocol
from datetime import datetime, timedelta
from collections.abc import Sequence

_HEADER = re.compile(r"^(id|回复id|alias_id|aliasid)\s*:\s*(.*)$")
_UNRELIABLE_IDS = {"", "0", "unknown", "system", "none", "null", "undefined"}
BOT_ECHO_LOOKBACK = timedelta(hours=1)


class _HistoryRecord(Protocol):
    session_id: str
    content_type: str
    content: str
    created_at: datetime


_Record = TypeVar("_Record", bound=_HistoryRecord)


def is_reliable_platform_id(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in _UNRELIABLE_IDS


def _metadata_and_body(content: str) -> tuple[dict[str, list[str]], str]:
    # Only the leading transport header is metadata. A user's later "id:" line
    # remains ordinary speech, and Unicode line separators inside it stay intact.
    lines = content.split("\n")
    first = _HEADER.fullmatch(lines[0].rstrip("\r")) if lines else None
    if first is None or first[1] != "id":
        return {}, content.strip()

    metadata: dict[str, list[str]] = {"id": [first[2].strip()]}
    body_start = 1
    for line in lines[1:]:
        match = _HEADER.fullmatch(line.rstrip("\r"))
        if match is None or match[1] == "id":
            break
        metadata.setdefault(match[1], []).append(match[2].strip())
        body_start += 1
    return metadata, "\n".join(lines[body_start:]).strip()


def parse_message_metadata(content: str) -> tuple[str | None, str | None, str]:
    metadata, body = _metadata_and_body(content)
    own_ids = metadata.get("id", [])
    reply_ids = metadata.get("回复id", [])
    return (
        own_ids[0] if own_ids else None,
        reply_ids[0] if reply_ids else None,
        body,
    )


def platform_message_ids(content: str) -> frozenset[str]:
    metadata, _ = _metadata_and_body(content)
    return frozenset(
        value
        for key in ("id", "alias_id", "aliasid")
        for value in metadata.get(key, [])
        if is_reliable_platform_id(value)
    )


def deduplicate_bot_echoes(history: Sequence[_Record]) -> list[_Record]:
    """Keep stored bot records and drop adapter echoes of those same messages.

    Different sessions, unknown IDs and ordinary users' multipart messages are
    deliberately left alone. No content similarity or nickname guessing is used.
    """
    bot_times: dict[tuple[str, str], list[datetime]] = {}
    for message in history:
        if message.content_type == "bot":
            for platform_id in platform_message_ids(message.content):
                bot_times.setdefault((message.session_id, platform_id), []).append(
                    message.created_at
                )
    return [
        message
        for message in history
        if message.content_type == "bot"
        or not any(
            abs(message.created_at - bot_time) <= BOT_ECHO_LOOKBACK
            for platform_id in platform_message_ids(message.content)
            for bot_time in bot_times.get((message.session_id, platform_id), [])
        )
    ]
