import time
import asyncio
from weakref import WeakKeyDictionary
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from collections.abc import Callable, AsyncIterator

from nonebot.log import logger


@dataclass
class _GateLease:
    semaphore: asyncio.Semaphore
    owner: asyncio.Task | None
    held: bool = True


class ConcurrencyGate:
    """A loop-local semaphore with optional non-blocking admission."""

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = max(int(limit), 1)
        self._semaphores: WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = WeakKeyDictionary()
        self._lease: ContextVar[_GateLease | None] = ContextVar(f"{name}_lease", default=None)

    def configure(self, limit: int) -> None:
        normalized = max(int(limit), 1)
        if normalized == self.limit:
            return
        self.limit = normalized
        self._semaphores.clear()

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.limit)
            self._semaphores[loop] = semaphore
        return semaphore

    @asynccontextmanager
    async def slot(self, *, wait: bool = True) -> AsyncIterator[bool]:
        semaphore = self._get_semaphore()
        if not wait and semaphore.locked():
            yield False
            return

        started_at = time.perf_counter()
        await semaphore.acquire()
        waited_ms = (time.perf_counter() - started_at) * 1000
        if waited_ms >= 1000:
            logger.info(
                f"[并发控制] {self.name} 等待 {waited_ms:.0f}ms 后获得执行槽"
            )
        lease = _GateLease(semaphore, asyncio.current_task())
        token = self._lease.set(lease)
        try:
            yield True
        finally:
            self._lease.reset(token)
            if lease.held:
                semaphore.release()

    @asynccontextmanager
    async def suspend_slot(self) -> AsyncIterator[None]:
        """Release this run's slot during slow work, then reacquire before continuing.

        Tool tasks inherit the lease object. A tool timeout must reacquire too:
        its parent Agent can catch the timeout and continue using other tools.
        """
        lease = self._lease.get()
        if lease is None or not lease.held:
            yield
            return
        lease.held = False
        lease.semaphore.release()
        try:
            yield
        finally:
            if lease.owner is None or not getattr(lease.owner, "cancelling", lambda: 0)():
                await lease.semaphore.acquire()
                lease.held = True


agent_run_gate = ConcurrencyGate("Agent", 4)
background_image_gate = ConcurrencyGate("后台图片", 2)
maintenance_gate = ConcurrencyGate("维护任务", 1)

current_reply_suspension: ContextVar[Callable[[], AbstractAsyncContextManager[None]] | None] = ContextVar("reply_suspension", default=None)


@asynccontextmanager
async def suspend_reply_for_image() -> AsyncIterator[None]:
    """Keep the drawing request alive while allowing ordinary replies to run."""
    suspend_group = current_reply_suspension.get()
    async with agent_run_gate.suspend_slot():
        if suspend_group is None:
            yield
        else:
            async with suspend_group():
                yield


def configure_concurrency(
    *,
    agent_limit: int,
    background_image_limit: int,
    maintenance_limit: int,
) -> None:
    agent_run_gate.configure(agent_limit)
    background_image_gate.configure(background_image_limit)
    maintenance_gate.configure(maintenance_limit)
