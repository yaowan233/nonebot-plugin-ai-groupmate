"""Decide if an active group is eligible for a bounded observation."""

from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    turn_id: str
    at: datetime
    explicitly_addressed: bool


@dataclass(frozen=True)
class ObservationState:
    interactions: tuple[Interaction, ...]
    last_interaction: datetime | None
    active: bool
    new_human_messages: int
    next_due: datetime
    enabled: bool = True
    paused: bool = False
    recent_message_activity: bool = False


def evaluate(state: ObservationState, *, now: datetime, budget_available: bool) -> tuple[bool, bool]:
    """Return (is_active, is_due); callers persist state and reserve budget atomically.

    Interactions must represent completed human/Bot exchanges. Sending extra
    messages or observing a group is never itself an interaction.
    """
    if not state.enabled:
        return False, False
    recent = [item for item in state.interactions if now - timedelta(hours=1) <= item.at <= now]
    qualifies = len({item.turn_id for item in recent}) >= 2 and any(item.explicitly_addressed for item in recent)
    keep_active = state.active and state.last_interaction is not None and now - timedelta(hours=2) < state.last_interaction <= now
    active = qualifies or keep_active or (state.recent_message_activity and state.new_human_messages >= 10)
    return active, bool(active and not state.paused and budget_available and state.new_human_messages >= 10 and state.next_due <= now)
