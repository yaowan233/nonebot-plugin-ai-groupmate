"""Find observable group interactions; retain evidence rather than assign traits."""

import re
import unicodedata
from typing import Protocol
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


class _QueryMessage(Protocol):
    @property
    def msg_id(self) -> int: ...

    @property
    def user_id(self) -> str: ...

    @property
    def content_type(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def created_at(self) -> datetime: ...


_HISTORY_TOPIC = re.compile(
    r"聊天|聊过|聊了|聊什么|发言|讨论|说过|提过|提到过|群友|成员|复读|梗|来历|出处|约定|观点|看法|态度|共同经历|时间线|上下文|前后文"
    r"|群.{0,12}(?:排名|排行|历史)|历史|以前|过去|之前|往事|上次|神人|整活|经典|语录"
)
_QUERY_INTENT = re.compile(
    r"^(?:(?:那|那么|请|帮我|帮忙|给我|麻烦|再|继续|顺便|你|能不能|可以|能|用\S+)[，,：:\s]*)*"
    r"(?:查|搜|找|翻翻|翻一下|追溯|核对|确认|对比|比较|回忆|展开|梳理|复盘|统计|盘点|回顾|整理|分析|总结|排名|排行|排个|排一下|排一|列|举|看看|看一下|给个)"
    r"|^(?:(?:请|帮我|麻烦)\s*)?(?:把|将).{1,100}(?:整理|总结|梳理|复盘|做成|串成|列成|画成|汇总)"
    r"|^(?:(?:那|那么|请|帮我|麻烦)\s*)?(?:给|为).{1,60}(?:排个|排一下|排名|排行|做(?:个|一份).{0,8}(?:回顾|总结|统计|对比))"
    r"|谁(?:是|最|算|的|有)|哪些|哪几|有没有|说过什么"
)
_HISTORY_QUESTION = re.compile(
    r"(?:聊|讨论|说|提|约定).{0,10}(?:什么|啥|何时|什么时候|哪些|哪几)"
    r"|(?:谁|哪位|哪个).{0,8}(?:先说|说过|提过|提到)"
    r"|(?:梗|说法|话).{0,8}(?:怎么来|哪来|出自|来源)"
    r"|(?:什么时候|何时).{0,8}(?:约好|说好|讨论|提到)"
)
_HISTORY_FOLLOWUP = re.compile(r"^(?:那|那么)?(?:后来呢|然后呢|前后文呢|还有谁|还有其他人吗)[？?。！!\s]*$")
_QUOTED_TEXT = re.compile(r'“[^”]*”|「[^」]*」|‘[^’]*’|"[^"\n]*"')


def interaction_history_requested(
    messages: Iterable[_QueryMessage],
    user_id: str,
    *,
    bot_name: str = "",
) -> bool:
    """Conservatively expose expensive history tools for the current user's query.

    This is a deterministic entry gate, not perfect semantic intent detection.
    Brief history follow-ups may reuse that same user's query within two hours.
    """
    rows = sorted((row for row in messages if str(row.user_id) == str(user_id) and row.content_type == "text"), key=lambda row: row.msg_id)
    if not rows:
        return False

    def text(row: _QueryMessage) -> str:
        body = parse_message_metadata(row.content)[2]
        body = re.sub(r"^@\S+\s+", "", body)
        if bot_name and body.startswith(bot_name):
            body = body[len(bot_name) :].lstrip("，,：: ")
        return body

    current = text(rows[-1])
    if re.match(r"^(?:那|你|请|还是|现在|再|先)*\s*(?:不要|别|不用|停止|取消)", current):
        return False

    def explicit_query(body: str) -> bool:
        unquoted = _QUOTED_TEXT.sub("[引用]", body)
        return bool((_QUERY_INTENT.search(unquoted) or _HISTORY_QUESTION.search(unquoted)) and _HISTORY_TOPIC.search(body))

    if explicit_query(current):
        return True
    unquoted = _QUOTED_TEXT.sub("[引用]", current)
    if not (_QUERY_INTENT.search(unquoted) or _HISTORY_FOLLOWUP.fullmatch(unquoted)):
        return False
    return any(timedelta(0) <= rows[-1].created_at - row.created_at <= timedelta(hours=2) and explicit_query(text(row)) for row in rows[-4:-1])


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
