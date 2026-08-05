import time
import asyncio
from weakref import WeakKeyDictionary
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from nonebot.log import logger


class ConcurrencyGate:
    """A loop-local semaphore with optional non-blocking admission."""

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = max(int(limit), 1)
        self._semaphores: WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = WeakKeyDictionary()

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
        try:
            yield True
        finally:
            semaphore.release()


agent_run_gate = ConcurrencyGate("Agent", 4)
background_image_gate = ConcurrencyGate("后台图片", 2)
maintenance_gate = ConcurrencyGate("维护任务", 1)


def configure_concurrency(
    *,
    agent_limit: int,
    background_image_limit: int,
    maintenance_limit: int,
) -> None:
    agent_run_gate.configure(agent_limit)
    background_image_gate.configure(background_image_limit)
    maintenance_gate.configure(maintenance_limit)
