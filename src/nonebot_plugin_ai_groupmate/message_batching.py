"""Collect a speaker's short burst before admitting a request to the agent queue."""

import time
import asyncio
from typing import Generic, TypeVar
from contextvars import ContextVar
from dataclasses import field, dataclass
from collections.abc import Callable, Awaitable

T = TypeVar("T")
current_input_batch: ContextVar[tuple[tuple[str, str], ...]] = ContextVar("current_input_batch", default=())


@dataclass
class PendingInput(Generic[T]):
    request: T
    started: float
    deadline: float
    messages: list[tuple[str, str]] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class InputBatcher(Generic[T]):
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.pending: dict[tuple[str, ...], PendingInput[T]] = {}
        self.tasks: set[asyncio.Task] = set()

    async def submit(
        self, key: tuple[str, ...], request: T, *, message_id: str, text: str,
        delay: float, max_wait: float, merge: Callable[[T, T], T],
        deliver: Callable[[T], Awaitable[None]],
    ) -> None:
        batch = self.pending.get(key)
        if batch is None and (delay <= 0 or max_wait <= 0 or len(self.pending) >= 128):
            await deliver(request)
            return
        now = self.clock()
        if batch is not None:
            if message_id and any(mid == message_id for mid, _ in batch.messages):
                return
            batch.request = merge(batch.request, request)
            batch.messages.append((message_id, text))
            batch.deadline = min(batch.started + max_wait, now + delay)
            if len(batch.messages) >= 20:
                batch.deadline = now
            batch.changed.set()
            return
        batch = PendingInput(request, now, now + min(delay, max_wait), [(message_id, text)])
        self.pending[key] = batch
        task = asyncio.create_task(self._deliver_when_ready(key, batch, deliver))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _deliver_when_ready(self, key: tuple[str, ...], batch: PendingInput[T], deliver: Callable[[T], Awaitable[None]]) -> None:
        try:
            while True:
                if batch.cancelled:
                    return
                remaining = batch.deadline - self.clock()
                if remaining <= 0:
                    break
                batch.changed.clear()
                try:
                    await asyncio.wait_for(batch.changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
            self.pending.pop(key, None)  # A new burst cannot mutate an admitted task.
            token = current_input_batch.set(tuple(batch.messages))
            try:
                await deliver(batch.request)
            finally:
                current_input_batch.reset(token)
        finally:
            if self.pending.get(key) is batch:
                self.pending.pop(key, None)

    def discard_speaker(self, scope: tuple[str, str, str]) -> None:
        for key, batch in list(self.pending.items()):
            if key[:3] == scope:
                self.pending.pop(key, None)
                batch.cancelled = True
                batch.changed.set()

    async def close(self) -> None:
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
