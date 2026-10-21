"""Occasional group observation scheduled from local message events."""

import time
import random
from typing import Generic, TypeVar
from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Callable

T = TypeVar("T")
GroupKey = tuple[str, str]
MIN_NEW_MESSAGES = 12
RECENT_ACTIVITY_SECONDS = 900
RECENT_PARTICIPATION_SECONDS = 2700


@dataclass
class _Watch(Generic[T]):
    payload: T
    due: float
    last_activity: float
    messages: int = 0
    paused: bool = False


class ObservationSchedule(Generic[T]):
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        delay: Callable[[], float] = lambda: random.uniform(2700, 5400),
    ):
        self.clock = clock
        self.delay = delay
        self.watches: OrderedDict[GroupKey, _Watch[T]] = OrderedDict()
        self.next_wake = 0.0

    def clear(self) -> None:
        self.watches.clear()
        self.next_wake = 0.0

    def participated(self, key: GroupKey) -> None:
        watch = self.watches.get(key)
        if watch is not None:
            watch.due = self.clock() + self.delay()
            watch.messages = 0

    def note(self, key: GroupKey, payload: T, *, addressed: bool = False, disengaged: bool = False) -> None:
        now = self.clock()
        watch = self.watches.pop(key, None)
        if watch is None or now - watch.last_activity > 21600:
            watch = _Watch(payload, now + self.delay(), now)
        watch.payload = payload
        watch.last_activity = now
        if addressed or disengaged:
            watch.paused = disengaged
            watch.due = now + self.delay()
            watch.messages = 0
        elif not watch.paused:
            watch.messages = min(MIN_NEW_MESSAGES, watch.messages + 1)
        self.watches[key] = watch
        while len(self.watches) > 256:
            self.watches.popitem(last=False)

    def take_due(self, allowed: Callable[[GroupKey], bool]) -> T | None:
        now = self.clock()
        for key, watch in list(self.watches.items()):
            if now - watch.last_activity > 21600:
                self.watches.pop(key)
        if now < self.next_wake:
            return None
        for key, watch in self.watches.items():
            if watch.paused or watch.due > now or watch.messages < MIN_NEW_MESSAGES or now - watch.last_activity > RECENT_ACTIVITY_SECONDS:
                continue
            # Declining or skipping a look also consumes the opportunity. Neither
            # timer ticks nor silence can repeatedly start model calls.
            watch.due = now + self.delay()
            watch.messages = 0
            if not allowed(key):
                continue
            self.next_wake = now + 300
            return watch.payload
        return None
