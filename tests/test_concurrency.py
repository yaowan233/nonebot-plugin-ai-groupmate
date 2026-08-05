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
async def test_small_pool_survives_many_slow_agent_runs(tmp_path):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from nonebot_plugin_ai_groupmate.concurrency import ConcurrencyGate

    database_path = (tmp_path / "small-pool.db").as_posix()
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path}",
        pool_size=2,
        max_overflow=0,
        pool_timeout=0.05,
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
