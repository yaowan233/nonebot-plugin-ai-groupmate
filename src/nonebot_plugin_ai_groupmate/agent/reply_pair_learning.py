"""Reconstruct attributable human reply candidates; models judge relevance later."""

import hashlib
from typing import Literal
from datetime import datetime
from collections.abc import Sequence

from pydantic import Field, BaseModel, ConfigDict, StrictBool

from ..model import ChatHistorySchema
from ..message_metadata import platform_message_ids, parse_message_metadata

MAX_PAIR_CANDIDATES = 8
MAX_REPLY_PAIRS = 64
ADJACENT_PAIR_MAX_SECONDS = 60


class PairMessage(BaseModel):
    msg_id: int
    user_id: str
    text: str = Field(min_length=2, max_length=240)
    platform_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class ReplyPair(BaseModel):
    id: str
    relation: Literal["quoted", "adjacent"]
    kind: Literal["response", "continuation"]
    trigger: PairMessage
    reply: PairMessage
    updated_at: datetime
    disabled: bool = False


class ReplyPairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str
    is_reply_pair: StrictBool
    transferable: StrictBool


def build_reply_pair_candidates(
    history: Sequence[ChatHistorySchema],
    sources: Sequence[tuple[ChatHistorySchema, str]],
    session_id: str,
    *, after_reply_id: int = 0, after_quoted_reply_id: int | None = None, include_continuations: bool = True,
) -> list[ReplyPair]:
    """Explicit replies take precedence; never silently re-pair an unknown quote.

    Eligibility is supplied by the existing human-only style filter. Ineligible
    messages break adjacency, rather than connecting unrelated conversations.
    """
    eligible = {row.msg_id: (row, body) for row, body in sources if row.session_id == session_id}
    ordered = sorted((row for row in history if row.session_id == session_id), key=lambda row: (row.created_at, row.msg_id))
    by_platform: dict[str, list[ChatHistorySchema]] = {}
    for row in ordered:
        for platform_id in platform_message_ids(row.content):
            by_platform.setdefault(platform_id, []).append(row)

    def unambiguous(rows: list[ChatHistorySchema]) -> ChatHistorySchema | None:
        identities = {(row.user_id, row.content_type, parse_message_metadata(row.content)[2]) for row in rows}
        if len(identities) != 1:
            return None
        return next((row for row in reversed(rows) if row.msg_id in eligible), None)

    pairs: list[ReplyPair] = []
    previous: ChatHistorySchema | None = None
    for row in ordered:
        item = eligible.get(row.msg_id)
        if item is None or row.content_type != "text":
            previous = None
            continue
        _, body = item
        own_ids = platform_message_ids(row.content)
        if any(unambiguous(by_platform[pid]) is not row for pid in own_ids):
            previous = None
            continue
        _, quoted_id, _ = parse_message_metadata(row.content)
        trigger = None
        relation: Literal["quoted", "adjacent"] = "adjacent"
        if quoted_id is not None:
            relation = "quoted"
            trigger = unambiguous(by_platform.get(quoted_id, []))
        elif previous is not None and 0 <= (row.created_at - previous.created_at).total_seconds() <= ADJACENT_PAIR_MAX_SECONDS:
            trigger = previous
        previous = row
        cursor = after_quoted_reply_id if relation == "quoted" and after_quoted_reply_id is not None else after_reply_id
        if (row.msg_id <= cursor or trigger is None or trigger.msg_id >= row.msg_id
                or trigger.created_at > row.created_at or trigger.msg_id not in eligible):
            continue
        trigger_body = eligible[trigger.msg_id][1]
        kind: Literal["response", "continuation"] = "continuation" if trigger.user_id == row.user_id else "response"
        if kind == "continuation" and not include_continuations:
            continue
        key = hashlib.sha256(f"{session_id}\0{trigger.msg_id}\0{row.msg_id}\0{relation}".encode()).hexdigest()[:16]
        pairs.append(ReplyPair(
            id=key, relation=relation, kind=kind,
            trigger=PairMessage(msg_id=trigger.msg_id, user_id=trigger.user_id, text=trigger_body,
                                platform_ids=sorted(platform_message_ids(trigger.content)), created_at=trigger.created_at),
            reply=PairMessage(msg_id=row.msg_id, user_id=row.user_id, text=body,
                              platform_ids=sorted(own_ids), created_at=row.created_at),
            updated_at=row.created_at,
        ))
    # Each relation has an independent cursor: prioritizing explicit quotes
    # cannot advance past as-yet-unreviewed adjacent examples.
    return sorted(pairs, key=lambda pair: (pair.relation != "quoted", pair.reply.msg_id))[:MAX_PAIR_CANDIDATES]


def pair_prompt_payload(pairs: Sequence[ReplyPair]) -> list[dict]:
    """Models see immutable IDs and anonymous turns, not authors' identities."""
    return [{"pair_id": pair.id, "relation": pair.relation, "kind": pair.kind,
             "trigger_id": pair.trigger.msg_id, "trigger_text": pair.trigger.text,
             "reply_id": pair.reply.msg_id, "reply_text": pair.reply.text} for pair in pairs]


def validated_pair_judgments(judgments: Sequence[ReplyPairJudgment], pairs: Sequence[ReplyPair]) -> dict[str, ReplyPairJudgment]:
    valid_ids = {pair.id for pair in pairs}
    result = {}
    duplicates = set()
    for item in judgments:
        if item.pair_id not in valid_ids:
            continue
        if item.pair_id in result:
            duplicates.add(item.pair_id)
        result[item.pair_id] = item
    return {key: item for key, item in result.items() if key not in duplicates}
