"""Schedule addressed and autonomous replies without coupling queue rules to ingress."""

import asyncio
from contextlib import asynccontextmanager
from collections import deque
from collections.abc import Callable, Awaitable

from nonebot import logger

from .concurrency import current_reply_suspension
from .reply_guard import activate_request_id, deactivate_request_id, set_latest_request_id
from .reply_request import ReplyRequest, GroupReplyState


class GroupReplyScheduler:
    def __init__(self, execute: Callable[[ReplyRequest], Awaitable[None]],
                 concurrency_limit: Callable[[], int], *, max_addressed: int = 3) -> None:
        self.execute = execute
        self.concurrency_limit = concurrency_limit
        self.max_addressed = max_addressed
        self.states: dict[str, GroupReplyState] = {}
        self.lock = asyncio.Lock()

    def _start_background_locked(self, group_id: str, state: GroupReplyState) -> None:
        state.running = True
        state.task = asyncio.create_task(self.run_background(group_id))

    def _start_addressed_locked(self, group_id: str, state: GroupReplyState) -> None:
        limit = max(int(self.concurrency_limit()), 1)
        while len(state.addressed_tasks) - len(state.drawing_requests) < limit:
            if state.drawing_resumes:
                request_id, resumed = state.drawing_resumes.popleft()
                if request_id in state.addressed_tasks and request_id in state.drawing_requests:
                    state.drawing_requests.remove(request_id)
                    resumed.set()
                continue
            if not state.addressed:
                break
            request = state.addressed.popleft()
            task = asyncio.create_task(self._run_addressed(group_id, request),
                                       name=f"ai-groupmate-addressed:{group_id}:{request.request_id}")
            state.addressed_tasks[request.request_id] = task

    @asynccontextmanager
    async def _suspend_drawing(self, group_id: str, request_id: str, owner: asyncio.Task | None):
        async with self.lock:
            state = self.states[group_id]
            state.drawing_requests.add(request_id)
            self._start_addressed_locked(group_id, state)
        logger.info(f"[绘图调度] 等待绘图时让出聊天名额 session={group_id} request={request_id}")
        try:
            yield
        finally:
            if owner is None or not getattr(owner, "cancelling", lambda: 0)():
                resumed = asyncio.Event()
                async with self.lock:
                    state.drawing_resumes.append((request_id, resumed))
                    self._start_addressed_locked(group_id, state)
                await resumed.wait()

    async def _run_addressed(self, group_id: str, request: ReplyRequest) -> None:
        owner = asyncio.current_task()
        token = current_reply_suspension.set(lambda: self._suspend_drawing(group_id, request.request_id, owner))
        try:
            await activate_request_id(group_id, request.request_id)
            await self.execute(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"群 {group_id} 明确提问处理失败 request={request.request_id}")
        finally:
            current_reply_suspension.reset(token)
            await deactivate_request_id(group_id, request.request_id)
            async with self.lock:
                state = self.states.get(group_id)
                if state is None:
                    return
                state.addressed_tasks.pop(request.request_id, None)
                state.drawing_requests.discard(request.request_id)
                state.drawing_resumes = deque((rid, event) for rid, event in state.drawing_resumes if rid != request.request_id)
                self._start_addressed_locked(group_id, state)
                if (not state.addressed and not state.addressed_tasks and state.latest is not None
                        and (not state.running or state.task is None or state.task.done())):
                    self._start_background_locked(group_id, state)

    async def enqueue(self, group_id: str, request: ReplyRequest) -> bool:
        """Addressed requests have priority; autonomous work keeps only the latest."""
        async with self.lock:
            state = self.states.setdefault(group_id, GroupReplyState())
            active_is_addressed = bool(state.addressed_tasks)
            if getattr(request, "proactive_observation", False) and (
                active_is_addressed or state.addressed or state.running or state.latest is not None
            ):
                return False
            if request.is_tome or request.is_continuous:
                addressed_count = len(state.addressed) + len(state.addressed_tasks)
                if addressed_count >= self.max_addressed:
                    logger.warning(f"群 {group_id} 待处理的定向请求已达上限，忽略额外请求")
                    return False
                state.addressed.append(request)
                state.latest = None
                if state.active is not None and state.task is not None:
                    # Cleanup may overlap several mentions; revoke only the
                    # background turn, preserving every addressed reply.
                    await deactivate_request_id(group_id, state.active.request_id)
                    state.task.cancel()
                    logger.info(f"群 {group_id} 收到明确 @，已取消后台回复")
                self._start_addressed_locked(group_id, state)
                return True
            if active_is_addressed or state.addressed:
                logger.debug(f"群 {group_id} 存在待处理的定向请求，忽略后台回复采样")
                return False
            state.latest = request
            if not state.running:
                self._start_background_locked(group_id, state)
            elif not state.task or state.task.done():
                logger.warning(f"群 {group_id} 回复 worker 不可用，已重新启动")
                self._start_background_locked(group_id, state)
            elif state.active is not None:
                await set_latest_request_id(group_id, request.request_id)
                state.task.cancel()
                logger.info(f"群 {group_id} 已将后台回复切换到最新消息")
            return True

    async def run_background(self, group_id: str) -> None:
        try:
            while True:
                async with self.lock:
                    state = self.states.get(group_id)
                    if not state or state.addressed or state.addressed_tasks:
                        return
                    request = state.latest
                    state.latest = None
                    state.active = request
                if request is None:
                    break
                await set_latest_request_id(group_id, request.request_id)
                await self.execute(request)
        finally:
            async with self.lock:
                state = self.states.get(group_id)
                if state:
                    state.running = False
                    state.active = None
                    state.task = None
                    self._start_addressed_locked(group_id, state)
                    if not state.addressed and not state.addressed_tasks and state.latest is not None:
                        self._start_background_locked(group_id, state)
