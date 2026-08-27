import asyncio
from types import SimpleNamespace
from typing import Any, cast
from contextlib import asynccontextmanager

import pytest


@pytest.mark.asyncio
async def test_concurrency_gate_bounds_parallel_work():
    from nonebot_plugin_ai_groupmate.concurrency import ConcurrencyGate

    gate = ConcurrencyGate("test", 2)
    active = 0
    peak = 0

    async def run_one() -> None:
        nonlocal active, peak
        async with gate.slot() as admitted:
            assert admitted is True
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(run_one() for _ in range(20)))

    assert peak == 2


@pytest.mark.asyncio
async def test_maintenance_gate_can_skip_instead_of_queueing():
    from nonebot_plugin_ai_groupmate.concurrency import ConcurrencyGate

    gate = ConcurrencyGate("maintenance-test", 1)
    async with gate.slot() as first_admitted:
        assert first_admitted is True
        async with gate.slot(wait=False) as second_admitted:
            assert second_admitted is False


@pytest.mark.asyncio
async def test_cancelled_gate_holder_releases_slot():
    from nonebot_plugin_ai_groupmate.concurrency import ConcurrencyGate

    gate = ConcurrencyGate("cancel-test", 1)
    entered = asyncio.Event()

    async def hold_slot() -> None:
        async with gate.slot():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_slot())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def acquire_again() -> None:
        async with gate.slot() as admitted:
            assert admitted is True

    await asyncio.wait_for(acquire_again(), timeout=0.1)


@pytest.mark.asyncio
async def test_queued_image_task_opens_session_only_after_slot(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.concurrency import background_image_gate

    original_limit = background_image_gate.limit
    background_image_gate.configure(1)
    events: list[str] = []

    @asynccontextmanager
    async def fake_get_session():
        events.append("open")
        try:
            yield object()
        finally:
            events.append("close")

    async def fake_process_image_message(*args, **kwargs) -> None:
        events.append("process")

    monkeypatch.setattr(plugin, "get_session", fake_get_session)
    monkeypatch.setattr(plugin, "process_image_message", fake_process_image_message)

    try:
        async with background_image_gate.slot():
            task = asyncio.create_task(
                plugin._process_image_task(
                    None, None, None, None, None, None, "id: 1\n"
                )
            )
            await asyncio.sleep(0.01)
            assert events == []

        await task
        assert events == ["open", "process", "close"]
    finally:
        background_image_gate.configure(original_limit)


@pytest.mark.asyncio
async def test_group_workers_wait_for_agent_slot_before_running(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.concurrency import agent_run_gate

    original_limit = agent_run_gate.limit
    agent_run_gate.configure(2)
    active = 0
    peak = 0

    async def fake_handle_reply_logic(*args, **kwargs) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    monkeypatch.setattr(plugin, "handle_reply_logic", fake_handle_reply_logic)
    plugin._group_reply_states.clear()

    try:
        for index in range(20):
            group_id = f"group-{index}"
            request = SimpleNamespace(
                request_id=f"request-{index}",
                session=None,
                interface=None,
                bot=None,
                event=None,
                bot_name="bot",
                user_id=f"user-{index}",
                user_name=f"user-{index}",
                is_tome=True,
                is_continuous=False,
                reply_to_id=None,
            )
            plugin._group_reply_states[group_id] = plugin.GroupReplyState(
                running=True,
                latest=cast(Any, request),
            )

        await asyncio.gather(
            *(
                plugin._run_group_reply_worker(group_id)
                for group_id in list(plugin._group_reply_states)
            )
        )

        assert peak == 2
    finally:
        plugin._group_reply_states.clear()
        agent_run_gate.configure(original_limit)


@pytest.mark.asyncio
async def test_addressed_requests_queue_without_background_preemption(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    invalidated_requests: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    def request(
        request_id: str,
        user_id: str,
        *,
        is_tome: bool = False,
        is_continuous: bool = False,
        proactive_meme_only: bool = False,
    ):
        return cast(
            Any,
            SimpleNamespace(
                request_id=request_id,
                session=None,
                interface=None,
                bot=None,
                event=None,
                bot_name="bot",
                user_id=user_id,
                user_name=user_id,
                is_tome=is_tome,
                is_continuous=is_continuous,
                reply_to_id=None,
                proactive_meme_only=proactive_meme_only,
                repeat_text=None,
            ),
        )

    async def fake_handle_reply_logic(*args, **kwargs) -> None:
        started.set()
        await release.wait()

    async def fake_set_latest_request_id(group_id: str, request_id: str) -> None:
        invalidated_requests.append(request_id)

    monkeypatch.setattr(plugin, "handle_reply_logic", fake_handle_reply_logic)
    monkeypatch.setattr(plugin, "set_latest_request_id", fake_set_latest_request_id)
    plugin._group_reply_states.clear()
    direct_request = request("direct-1", "user-1", is_tome=True)
    accepted = await plugin._queue_group_reply_request(
        "group-1",
        direct_request,
    )
    assert accepted is True
    await asyncio.wait_for(started.wait(), timeout=0.2)

    try:
        accepted = await plugin._queue_group_reply_request(
            "group-1",
            request("background-1", "user-2", proactive_meme_only=True),
        )

        assert accepted is False
        assert invalidated_requests == []
        assert "direct-1" in plugin._group_reply_states["group-1"].addressed_tasks

        accepted = await plugin._queue_group_reply_request(
            "group-1",
            request("continuous-1", "user-1", is_continuous=True),
        )

        assert accepted is True
        assert invalidated_requests == []
        assert set(plugin._group_reply_states["group-1"].addressed_tasks) == {
            "direct-1",
            "continuous-1",
        }

        accepted = await plugin._queue_group_reply_request(
            "group-1",
            request("direct-2", "user-2", is_tome=True),
        )

        assert accepted is True
        assert invalidated_requests == []
        assert set(plugin._group_reply_states["group-1"].addressed_tasks) == {
            "direct-1",
            "continuous-1",
        }
        addressed = plugin._group_reply_states["group-1"].addressed
        assert [item.request_id for item in addressed] == ["direct-2"]
    finally:
        release.set()
        while True:
            state = plugin._group_reply_states["group-1"]
            tasks = list(state.addressed_tasks.values())
            if not tasks and not state.addressed:
                break
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0)
        plugin._group_reply_states.clear()


@pytest.mark.asyncio
async def test_group_runs_two_addressed_requests_concurrently_and_queues_third(
    monkeypatch,
):
    import nonebot_plugin_ai_groupmate as plugin

    started: list[str] = []
    release = asyncio.Event()
    two_started = asyncio.Event()
    all_finished = asyncio.Event()
    finished = 0

    def request(request_id: str):
        return cast(
            Any,
            SimpleNamespace(
                request_id=request_id,
                session=None,
                interface=None,
                bot=None,
                event=None,
                bot_name="bot",
                user_id=request_id,
                user_name=request_id,
                is_tome=True,
                is_continuous=False,
                reply_to_id=None,
                proactive_meme_only=False,
                repeat_text=None,
            ),
        )

    async def fake_handle_reply_logic(request_id: str, *args, **kwargs) -> None:
        nonlocal finished
        started.append(request_id)
        if len(started) == 2:
            two_started.set()
        await release.wait()
        finished += 1
        if finished == 3:
            all_finished.set()

    monkeypatch.setattr(plugin, "handle_reply_logic", fake_handle_reply_logic)
    monkeypatch.setattr(plugin.plugin_config, "agent_max_concurrency_per_group", 2)
    plugin._group_reply_states.clear()

    try:
        for index in range(1, 4):
            accepted = await plugin._queue_group_reply_request(
                "group-1",
                request(f"direct-{index}"),
            )
            assert accepted is True

        await asyncio.wait_for(two_started.wait(), timeout=0.2)
        await asyncio.sleep(0)
        assert started == ["direct-1", "direct-2"]

        release.set()
        await asyncio.wait_for(all_finished.wait(), timeout=0.2)
        assert started == ["direct-1", "direct-2", "direct-3"]
    finally:
        release.set()
        state = plugin._group_reply_states.get("group-1")
        tasks = (
            []
            if state is None
            else [state.task, *state.addressed_tasks.values()]
        )
        await asyncio.gather(
            *(task for task in tasks if task is not None),
            return_exceptions=True,
        )
        plugin._group_reply_states.clear()


@pytest.mark.asyncio
async def test_small_pool_survives_many_slow_agent_runs(tmp_path):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from nonebot_plugin_ai_groupmate.concurrency import ConcurrencyGate

    database_path = (tmp_path / "small-pool.db").as_posix()
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path}",
        pool_size=2,
        max_overflow=0,
        # Windows CI 上首次 SQLite checkout 偶尔超过 50ms；测试关注的是
        # 慢 I/O 期间连接是否已归还，而不是把调度抖动当成连接池耗尽。
        pool_timeout=0.5,
    )
    session_factory = async_sessionmaker(engine)
    gate = ConcurrencyGate("small-pool-agent", 4)

    async def run_agent() -> None:
        async with gate.slot():
            async with session_factory() as session:
                await session.execute(text("select 1"))
                await session.commit()

                # Simulate slow LLM I/O after returning the connection.
                await asyncio.sleep(0.08)

                await session.execute(text("select 1"))
                await session.commit()

    try:
        await asyncio.gather(*(run_agent() for _ in range(20)))
        assert cast(Any, engine.sync_engine.pool).checkedout() == 0
    finally:
        await engine.dispose()
