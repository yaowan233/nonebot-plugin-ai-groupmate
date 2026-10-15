"""Find observable group interactions; retain evidence rather than assign traits."""

import unicodedata
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Iterable

from .message_metadata import (
    BOT_ECHO_LOOKBACK,
    platform_message_ids,
    deduplicate_bot_echoes,
    parse_message_metadata,
)


@dataclass
class HistoryMessage:
    msg_id: int
    session_id: str
    user_id: str
    user_name: str
    content_type: str
    content: str
    created_at: datetime
    media_id: int | None = None


def prepare_history(
    messages: Iterable[HistoryMessage],
    bot_user_ids: Iterable[str] = (),
) -> tuple[list[HistoryMessage], set[str]]:
    """Recognize bot sender IDs through two reliable transport echo pairs.

    Never infer identity from names or matching message text. Known IDs can be
    supplied by the adapter. Return chronological, deduplicated source records.
    """
    rows = sorted(messages, key=lambda row: (row.created_at, row.msg_id))
    bot_times: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for row in rows:
        if row.content_type == "bot":
            for platform_id in platform_message_ids(row.content):
                bot_times[row.session_id, platform_id].append(row.created_at)
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.content_type != "text":
            continue
        for platform_id in platform_message_ids(row.content):
            if any(abs(row.created_at - when) <= BOT_ECHO_LOOKBACK for when in bot_times.get((row.session_id, platform_id), ())):
                pairs[str(row.user_id)].add(platform_id)
    bot_ids = {str(uid) for uid in bot_user_ids}
    bot_ids.update(uid for uid, ids in pairs.items() if len(ids) >= 2)
    return deduplicate_bot_echoes(rows), bot_ids


def message_evidence(row: HistoryMessage, bot_ids: set[str], *, max_chars: int = 420) -> dict:
    _, reply_id, body = parse_message_metadata(row.content)
    return {
        "msg_id": row.msg_id,
        "user_id": str(row.user_id),
        "user_name": row.user_name[:80],
        "created_at": row.created_at.isoformat(),
        "role": "bot" if row.content_type == "bot" or str(row.user_id) in bot_ids else "user",
        "reply_platform_id": reply_id,
        "content": body[:max_chars],
        "content_truncated": len(body) > max_chars,
    }


def find_repeat_interactions(
    messages: Iterable[HistoryMessage],
    bot_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Discover exact cross-member repeat chains and first-visible speakers.

    NFKC equality, >=3 distinct users, <=10 minutes from the first message.
    Repetition is a candidate signal, not evidence of originality or a ranking.
    """
    by_text: dict[str, list[HistoryMessage]] = defaultdict(list)
    names: dict[str, str] = {}
    volumes: Counter[str] = Counter()
    for row in messages:
        uid = str(row.user_id)
        if row.content_type != "text" or uid in bot_ids:
            continue
        names[uid] = row.user_name[:80]
        volumes[uid] += 1
        body = unicodedata.normalize("NFKC", parse_message_metadata(row.content)[2]).strip()
        if row.media_id is not None or not 8 <= len(body) <= 180 or body.startswith(("[", "【", "@", "/")):
            continue
        if any(token in body for token in ("file_id=", "forward_id:", "需要查看", "需要了解")):
            continue
        by_text[body].append(row)

    events: list[dict] = []
    started: dict[str, list[dict]] = defaultdict(list)
    for rows in by_text.values():
        chain: list[HistoryMessage] = []

        def flush() -> None:
            members: dict[str, HistoryMessage] = {}
            for row in chain:
                members.setdefault(str(row.user_id), row)
            if len(members) < 3:
                return
            event = {
                "kind": "cross_member_repeat",
                "first_message": message_evidence(chain[0], bot_ids, max_chars=180),
                "end_at": chain[-1].created_at.isoformat(),
                "participant_count": len(members),
                "occurrence_count": len(chain),
                "participants": [{"user_id": uid, "user_name": row.user_name[:80], "source_msg_id": row.msg_id} for uid, row in members.items()],
                "source_msg_ids": [row.msg_id for row in chain],
            }
            events.append(event)
            started[str(chain[0].user_id)].append(event)

        for row in rows:
            if chain and row.created_at - chain[0].created_at > timedelta(minutes=10):
                flush()
                chain = []
            chain.append(row)
        flush()
    events.sort(key=lambda event: (-event["participant_count"], -event["occurrence_count"], event["first_message"]["created_at"], event["first_message"]["msg_id"]))
    candidates = [
        {
            "user_id": uid,
            "user_name": names[uid],
            "text_messages_in_window": volumes[uid],
            "first_visible_events": len(items),
            "event_days": len({item["first_message"]["created_at"][:10] for item in items}),
            "example_msg_ids": [item["first_message"]["msg_id"] for item in sorted(items, key=lambda item: -item["participant_count"])[:2]],
        }
        for uid, items in started.items()
    ]
    candidates.sort(key=lambda item: (-item["first_visible_events"], -item["event_days"], item["user_id"]))
    return events, candidates
