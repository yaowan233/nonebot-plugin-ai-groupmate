"""Request and queue state shared by input batching and reply scheduling."""

import time
import asyncio
from collections import deque
from contextvars import ContextVar
from dataclasses import field, replace, dataclass

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import Uninfo, QryItrface
from .message_batching import current_input_batch
from .message_metadata import is_reliable_platform_id


current_transport_message_id: ContextVar[str | None] = ContextVar("current_transport_message_id", default=None)


def resolve_current_message_id(event: Event | None) -> str | None:
    """Use the ID captured at ingress, with adapter-event fallback for old callers."""
    candidate = current_transport_message_id.get()
    if is_reliable_platform_id(candidate):
        return str(candidate).strip()
    for message_id, _ in reversed(current_input_batch.get()):
        if is_reliable_platform_id(message_id):
            return str(message_id).strip()
    candidate = str(getattr(event, "message_id", "") or "") or None
    return str(candidate).strip() if is_reliable_platform_id(candidate) else None


@dataclass
class ReplyRequest:
    request_id: str
    session: Uninfo
    interface: QryItrface
    bot: Bot
    event: Event
    bot_name: str
    user_id: str
    user_name: str | None
    is_tome: bool
    is_continuous: bool
    reply_to_id: str | None
    proactive_meme_only: bool = False
    meme_required: bool = False
    meme_send_count: int = 1
    proactive_reaction_only: bool = False
    reaction_required: bool = False
    repeat_text: str | None = None
    proactive_observation: bool = False
    input_batch: tuple[tuple[str, str], ...] = ()
    mentioned_user_ids: tuple[str, ...] = ()
    created_monotonic: float = field(default_factory=time.monotonic)
    message_id: str | None = None
    normalized_parts: tuple[dict[str, str], ...] | None = None


@dataclass
class GroupReplyState:
    running: bool = False
    latest: ReplyRequest | None = None
    addressed: deque[ReplyRequest] = field(default_factory=deque)
    addressed_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    drawing_requests: set[str] = field(default_factory=set)
    drawing_resumes: deque[tuple[str, asyncio.Event]] = field(default_factory=deque)
    active: ReplyRequest | None = None
    task: asyncio.Task | None = None


def merge_reply_requests(previous: ReplyRequest, latest: ReplyRequest) -> ReplyRequest:
    """Preserve the whole burst's addressing, quote and first-arrival time."""
    return replace(latest, is_tome=previous.is_tome or latest.is_tome,
                   is_continuous=previous.is_continuous or latest.is_continuous,
                   meme_required=previous.meme_required or latest.meme_required,
                   meme_send_count=max(previous.meme_send_count, latest.meme_send_count),
                   reply_to_id=latest.reply_to_id or previous.reply_to_id,
                   reaction_required=previous.reaction_required or latest.reaction_required,
                   mentioned_user_ids=tuple(dict.fromkeys((*previous.mentioned_user_ids, *latest.mentioned_user_ids))),
                   created_monotonic=min(previous.created_monotonic, latest.created_monotonic))
