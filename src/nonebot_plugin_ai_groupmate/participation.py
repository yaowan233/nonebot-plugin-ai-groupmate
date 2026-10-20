"""Bounded, event-driven group participation; never starts an idle conversation."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import field, dataclass
from collections.abc import Callable


@dataclass
class Participation:
    touched: float
    last_sent: float = float("-inf")
    last_check: float = float("-inf")
    engaged_until: float = 0
    quiet_until: float = 0
    credit: float = 0
    unanswered: int = 0
    last_backoff: float = 0
    pending_since: float | None = None
    messages_since: int = 0
    activity: deque[tuple[float, bool]] = field(default_factory=lambda: deque(maxlen=40))
    proactive_sends: deque[float] = field(default_factory=lambda: deque(maxlen=20))


class ParticipationPolicy:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.states: OrderedDict[tuple[str, str], Participation] = OrderedDict()

    def state(self, key: tuple[str, str]) -> Participation:
        now = self.clock()
        state = self.states.pop(key, None)
        if state is None or now - state.touched > 1800:
            state = Participation(touched=now)
        if state.unanswered and now - state.last_backoff >= 600:
            state.unanswered = max(0, state.unanswered - int((now - state.last_backoff) // 600))
            state.last_backoff = now
        state.touched = now
        self.states[key] = state
        while len(self.states) > 2048:
            self.states.popitem(last=False)
        return state

    def engaged(self, key: tuple[str, str], *, expected_sent: float | None = None) -> None:
        state = self.state(key)
        # A slow classifier must not attribute a reply to a newer bot message.
        if expected_sent is not None and state.last_sent != expected_sent:
            return
        state.engaged_until = self.clock() + 300
        state.unanswered = 0
        state.pending_since = None

    def observe(self, key: tuple[str, str], *, addressed: bool, disengaged: bool) -> None:
        state = self.state(key)
        now = self.clock()
        state.activity.append((now, False))
        if disengaged:
            state.quiet_until = now + 300
            state.engaged_until = 0
            state.credit = 0
            state.pending_since = None
        elif addressed:
            state.quiet_until = 0
            self.engaged(key)
        elif state.pending_since is not None:
            state.messages_since += 1
            if state.messages_since >= 5 and now - state.pending_since >= 120:
                state.unanswered = min(state.unanswered + 1, 3)
                state.last_backoff = now
                state.pending_since = None
                state.engaged_until = 0

    def can_reply(self, key: tuple[str, str]) -> bool:
        state = self.state(key)
        now = self.clock()
        cooldown = (45 if now < state.engaged_until else 120) * (2 ** state.unanswered)
        recent = [bot for at, bot in state.activity if now - at < 600]
        return (
            now >= state.quiet_until
            and now - state.last_sent >= cooldown
            and sum(now - at < 600 for at in state.proactive_sends) < 3
            and not (sum(recent) >= 3 and sum(recent) / len(recent) >= 0.25)
        )

    def candidate(self, key: tuple[str, str], probability: float) -> bool:
        state = self.state(key)
        if probability <= 0:
            state.credit = 0
            return False
        now = self.clock()
        if not self.can_reply(key):
            state.credit = 0
            return False
        # Probability becomes a participation preference. Accumulation avoids
        # starving quiet groups forever, while elapsed time alone earns nothing.
        boost = 50 if now < state.engaged_until else 10
        state.credit = min(1.0, state.credit + min(probability * boost, 1.0))
        if state.credit < 0.999 or now - state.last_check < 20 * (2 ** state.unanswered):
            return False
        state.credit = 0
        state.last_check = now
        return True

    def sent(self, key: tuple[str, str], *, proactive: bool) -> None:
        state = self.state(key)
        now = self.clock()
        state.last_sent = now
        state.activity.append((now, True))
        state.credit = 0
        if proactive:
            state.engaged_until = now + 300
            # Multiple sends in one turn count toward the cap, but don't erase
            # the observation window for the first message.
            state.proactive_sends.append(now)
            if state.pending_since is None:
                state.pending_since = now
                state.messages_since = 0
        else:
            self.engaged(key)

    def context(self, key: tuple[str, str]) -> str:
        state = self.state(key)
        active = self.clock() < state.engaged_until
        return (
            f"Bot 最近参与对话：{'是' if active else '否'}；"
            f"近期主动发言后未发现明确承接的次数：{state.unanswered}。"
            "这只是频率信号，不能代替语义判断；后续有人发消息不等于回应 Bot。"
        )


participation = ParticipationPolicy()
# Inherited by tool tasks, isolated across concurrent groups and requests.
delivery_context: ContextVar[tuple[tuple[str, str], bool] | None] = ContextVar(
    "group_participation_delivery", default=None
)


def record_group_delivery(session_id: str) -> None:
    context = delivery_context.get()
    if context is not None and context[0][1] == session_id:
        participation.sent(context[0], proactive=context[1])


def group_delivery_allowed(session_id: str) -> bool:
    context = delivery_context.get()
    if context is None or not context[1] or context[0][1] != session_id:
        return True
    state = participation.state(context[0])
    now = participation.clock()
    return now >= state.quiet_until and sum(now - at < 600 for at in state.proactive_sends) < 3
