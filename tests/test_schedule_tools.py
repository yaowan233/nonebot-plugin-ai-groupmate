import json
import asyncio
import datetime
from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture(autouse=True)
async def task_database(tmp_path, monkeypatch):
    from nonebot_plugin_apscheduler import scheduler

    from nonebot_plugin_ai_groupmate import scheduled_tasks
    from nonebot_plugin_ai_groupmate.agent import schedule_tools
    from nonebot_plugin_ai_groupmate.model import ScheduledTask

    poll_job = scheduler.get_job("ai_groupmate_poll_scheduled_tasks")
    if poll_job is not None:
        poll_job.pause()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.sqlite3'}")
    async with engine.begin() as connection:
        await connection.run_sync(ScheduledTask.__table__.create)
    db = SimpleNamespace(engine=engine, sessions=async_sessionmaker(engine))
    monkeypatch.setattr(schedule_tools, "get_session", lambda: db.sessions())
    monkeypatch.setattr(scheduled_tasks, "get_session", lambda: db.sessions())
    monkeypatch.setattr(scheduled_tasks, "get_bots", lambda: {"bot-1": object(), "bot-2": object()})
    monkeypatch.setattr(scheduled_tasks, "_running_tasks", {})
    monkeypatch.setattr(scheduled_tasks, "_stopping", False)
    yield db
    await scheduled_tasks.stop_scheduled_tasks()
    await db.engine.dispose()
    if poll_job is not None:
        poll_job.resume()


def management_tools(*, session_id="group-1", is_private=False, bot_id="bot-1", request_id=None):
    from nonebot_plugin_ai_groupmate.agent.schedule_tools import create_schedule_management_tools

    return {
        tool.name: tool for tool in create_schedule_management_tools(
            session_id, request_id, is_private=is_private, bot_id=bot_id,
        )
    }


async def create_task(task_type="message", *, session_id="group-1", is_private=False, bot_id="bot-1", delay_minutes=10):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    if task_type == "message":
        tool = schedule_tools.create_schedule_message_tool(
            session_id, None, is_private=is_private, bot_id=bot_id, bot_name="小夏",
        )
        args = {"content": "喝水提醒"}
    else:
        tool = schedule_tools.create_schedule_agent_task_tool(
            session_id, None, is_private=is_private, bot_id=bot_id,
        )
        args = {"task": "查询天气"}
    result = json.loads(await tool.ainvoke({**args, "delay_minutes": delay_minutes}))
    assert result["status"] == "succeeded"
    assert result["delivery_state"] == "completed"
    return result["data"]


async def record(db, job_id):
    from nonebot_plugin_ai_groupmate.model import ScheduledTask

    async with db.sessions() as session:
        return await session.get(ScheduledTask, job_id)


async def set_record(db, job_id, **values):
    from nonebot_plugin_ai_groupmate.model import ScheduledTask

    async with db.sessions() as session:
        await session.execute(update(ScheduledTask).where(ScheduledTask.job_id == job_id).values(**values))
        await session.commit()


async def make_due(db, job_id):
    from nonebot_plugin_ai_groupmate.scheduled_tasks import utcnow

    await set_record(db, job_id, run_at=utcnow() - datetime.timedelta(seconds=1))


@pytest.mark.parametrize("task_type", ["message", "agent"])
async def test_database_management_lifecycle_survives_reconnect(task_database, task_type):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools
    from nonebot_plugin_ai_groupmate.scheduled_tasks import utcnow

    tools = management_tools()
    created = await create_task(task_type)
    job_id = created["job_id"]
    original = await record(task_database, job_id)
    assert original.status == "scheduled"
    key = "content" if task_type == "message" else "task"

    # Recreate the engine/session factory as after a process restart. No jobs need restoring.
    url = task_database.engine.url
    await task_database.engine.dispose()
    task_database.engine = create_async_engine(url)
    task_database.sessions = async_sessionmaker(task_database.engine)
    listed = json.loads(await management_tools()["list_scheduled_tasks"].ainvoke({}))
    assert listed["data"]["total"] == 1
    assert listed["data"]["tasks"][0] == created

    edited = json.loads(await tools["update_scheduled_task"].ainvoke({"job_id": job_id, "content": " 新内容 "}))
    assert edited["reason_code"] == "schedule_updated"
    assert edited["data"][key] == "新内容"
    assert (await record(task_database, job_id)).run_at == original.run_at
    before = utcnow()
    edited = json.loads(await tools["update_scheduled_task"].ainvoke({"job_id": job_id, "delay_minutes": 30}))
    after = utcnow()
    updated = await record(task_database, job_id)
    assert before + datetime.timedelta(minutes=30) <= updated.run_at <= after + datetime.timedelta(minutes=30)
    assert updated.content == "新内容"
    assert updated.bot_id == original.bot_id
    assert updated.bot_name == original.bot_name
    assert updated.task_type == task_type
    assert edited["data"]["job_id"] == job_id
    assert datetime.datetime.fromisoformat(edited["data"]["run_at"]).utcoffset() == datetime.datetime.now(schedule_tools.scheduler.timezone).utcoffset()

    cancelled = json.loads(await tools["cancel_scheduled_task"].ainvoke({"job_id": job_id}))
    assert cancelled["reason_code"] == "schedule_cancelled"
    assert (await record(task_database, job_id)).status == "cancelled"
    listed = json.loads(await tools["list_scheduled_tasks"].ainvoke({}))
    assert listed["data"] == {"tasks": [], "total": 0, "next_offset": None}
    history = json.loads(await tools["list_scheduled_tasks"].ainvoke({"status": "all"}))
    assert history["data"]["tasks"][0]["status"] == "cancelled"
    assert history["data"]["tasks"][0]["finished_at"] is not None
    repeated = json.loads(await tools["cancel_scheduled_task"].ainvoke({"job_id": job_id}))
    assert repeated["reason_code"] == "schedule_not_found"


@pytest.mark.parametrize("other_context", [
    {"session_id": "group-10"}, {"is_private": True}, {"bot_id": "bot-2"}, {"bot_id": None},
])
async def test_database_management_cannot_access_other_context(task_database, other_context):
    own = await create_task()
    foreign = await create_task(**other_context)
    tools = management_tools()
    listed = json.loads(await tools["list_scheduled_tasks"].ainvoke({"status": "all"}))
    assert [task["job_id"] for task in listed["data"]["tasks"]] == [own["job_id"]]
    for name, args in [
        ("update_scheduled_task", {"content": "不能修改"}), ("cancel_scheduled_task", {}),
    ]:
        result = json.loads(await tools[name].ainvoke({"job_id": foreign["job_id"], **args}))
        assert result["reason_code"] == "schedule_not_found"
        assert result["delivery_state"] == "not_attempted"
    assert (await record(task_database, foreign["job_id"])).content == foreign["content"]


async def test_database_list_pagination():
    later = await create_task(delay_minutes=20)
    earlier = await create_task("agent", delay_minutes=5)
    tools = management_tools()
    first = json.loads(await tools["list_scheduled_tasks"].ainvoke({"limit": 1}))["data"]
    assert first["total"] == 2
    assert first["tasks"][0]["job_id"] == earlier["job_id"]
    second = json.loads(await tools["list_scheduled_tasks"].ainvoke({"limit": 1, "offset": first["next_offset"]}))["data"]
    assert second["tasks"][0]["job_id"] == later["job_id"]
    assert second["next_offset"] is None


@pytest.mark.parametrize(("args", "reason"), [
    ({}, "no_changes"),
    ({"content": " "}, "empty_content"),
    ({"delay_minutes": 0}, "invalid_delay"),
    ({"delay_minutes": -1}, "invalid_delay"),
    ({"delay_minutes": 0.01}, "delay_too_short"),
    ({"delay_hours": 169}, "delay_too_long"),
    ({"delay_minutes": float("nan")}, "invalid_delay"),
    ({"delay_hours": float("inf")}, "invalid_delay"),
    ({"content": "不能部分修改", "delay_minutes": -1}, "invalid_delay"),
])
async def test_invalid_update_does_not_change_database(task_database, args, reason):
    created = await create_task()
    original = await record(task_database, created["job_id"])
    result = json.loads(await management_tools()["update_scheduled_task"].ainvoke({"job_id": created["job_id"], **args}))
    assert result["reason_code"] == reason
    assert result["delivery_state"] == "not_attempted"
    unchanged = await record(task_database, created["job_id"])
    assert unchanged.content == original.content
    assert unchanged.run_at == original.run_at


@pytest.mark.parametrize("args", [{"offset": -1}, {"limit": 0}, {"limit": 51}])
async def test_list_rejects_invalid_pagination(args):
    result = json.loads(await management_tools()["list_scheduled_tasks"].ainvoke(args))
    assert result["reason_code"] == "invalid_pagination"


@pytest.mark.parametrize("task_type", ["message", "agent"])
@pytest.mark.parametrize("delay", [0, -1, float("nan"), float("inf")])
async def test_creation_rejects_invalid_delay(task_type, delay):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    if task_type == "message":
        tool = schedule_tools.create_schedule_message_tool("group-1", None, is_private=False, bot_id="bot-1", bot_name="bot")
        args = {"content": "提醒"}
    else:
        tool = schedule_tools.create_schedule_agent_task_tool("group-1", None, is_private=False, bot_id="bot-1")
        args = {"task": "查天气"}
    result = json.loads(await tool.ainvoke({**args, "delay_minutes": delay}))
    assert result["reason_code"] == "invalid_delay"


async def test_expired_request_cannot_create_or_manage_tasks(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    async def expired(*args):
        return False

    created = await create_task()
    monkeypatch.setattr(schedule_tools, "is_request_active", expired)
    tools = management_tools(request_id="expired")
    tools["schedule_message"] = schedule_tools.create_schedule_message_tool(
        "group-1", "expired", is_private=False, bot_id="bot-1", bot_name="bot",
    )
    for name, args in [
        ("list_scheduled_tasks", {}),
        ("update_scheduled_task", {"job_id": created["job_id"], "content": "不能修改"}),
        ("cancel_scheduled_task", {"job_id": created["job_id"]}),
        ("schedule_message", {"content": "不能创建", "delay_minutes": 1}),
    ]:
        result = json.loads(await tools[name].ainvoke(args))
        assert result["reason_code"] == "request_expired"
        assert result["status"] == "skipped"
    assert (await record(task_database, created["job_id"])).content == created["content"]


async def test_expiration_during_database_write_rolls_back(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    created = await create_task()
    checks = iter([True, False])

    async def active(*args):
        return next(checks)

    monkeypatch.setattr(schedule_tools, "is_request_active", active)
    result = json.loads(await management_tools(request_id="expires")["cancel_scheduled_task"].ainvoke({"job_id": created["job_id"]}))
    assert result["status"] == "skipped"
    assert (await record(task_database, created["job_id"])).status == "scheduled"


@pytest.mark.parametrize("operation", ["list", "update", "cancel", "create"])
async def test_database_errors_return_failure_protocol(task_database, monkeypatch, operation):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    created = await create_task()

    @asynccontextmanager
    async def failed_session():
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(schedule_tools, "get_session", failed_session)
    if operation == "create":
        tool = schedule_tools.create_schedule_message_tool("group-1", None, is_private=False, bot_id="bot-1", bot_name="bot")
        args = {"content": "提醒", "delay_minutes": 1}
    else:
        tool = management_tools()["list_scheduled_tasks" if operation == "list" else f"{operation}_scheduled_task"]
        args = {} if operation == "list" else {"job_id": created["job_id"]}
        if operation == "update":
            args["content"] = "新内容"
    result = json.loads(await tool.ainvoke(args))
    assert result["status"] == "failed"
    assert result["reason_code"] == ("schedule_failed" if operation == "create" else "schedule_management_failed")


@pytest.mark.parametrize("model_name", ["GroupModelConfig", "PrivateModelConfig"])
async def test_idle_poll_does_not_lock_unrelated_sqlite_writes(monkeypatch, model_name):
    import uuid

    from sqlalchemy import Update, delete
    from sqlalchemy.pool import AsyncAdaptedQueuePool
    from sqlalchemy.ext.asyncio import AsyncSession

    from nonebot_plugin_ai_groupmate import model, scheduled_tasks

    writer_started = asyncio.Event()
    release_writer = asyncio.Event()

    class PollSession(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            result = await super().execute(statement, *args, **kwargs)
            if isinstance(statement, Update):
                # Pin the interleaving: the poll has a write transaction, while
                # another connection attempts the DELETE reported by CI.
                writer_started.set()
                await release_writer.wait()
            return result

    engine = create_async_engine(
        f"sqlite+aiosqlite:///file:poll-lock-{uuid.uuid4().hex}?mode=memory&cache=shared&uri=true",
        poolclass=AsyncAdaptedQueuePool,
    )
    config_model = getattr(model, model_name)
    async with engine.begin() as connection:
        await connection.run_sync(model.ScheduledTask.__table__.create)
        await connection.run_sync(config_model.__table__.create)
    monkeypatch.setattr(scheduled_tasks, "get_session", async_sessionmaker(engine, class_=PollSession))
    monkeypatch.setattr(scheduled_tasks, "get_bots", lambda: {})
    poll = asyncio.create_task(scheduled_tasks.poll_scheduled_tasks())
    writing = asyncio.create_task(writer_started.wait())
    try:
        done, _ = await asyncio.wait({poll, writing}, timeout=3, return_when=asyncio.FIRST_COMPLETED)
        assert done, "Idle poll did not finish or reach a database write"
        async with async_sessionmaker(engine)() as session:
            await session.execute(delete(config_model))
            await session.commit()
    finally:
        release_writer.set()
        writing.cancel()
        await asyncio.gather(poll, writing, return_exceptions=True)
        await engine.dispose()
    await poll


@pytest.mark.parametrize("task_type", ["message", "agent"])
async def test_restarted_worker_executes_latest_database_content_once(task_database, monkeypatch, task_type):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task(task_type)
    job_id = created["job_id"]
    await management_tools()["update_scheduled_task"].ainvoke({"job_id": job_id, "content": "更新后的任务"})
    await make_due(task_database, job_id)
    received = []

    async def execute(task):
        received.append((task.task_type, task.content, task.bot_id, task.session_id))

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", execute)
    url = task_database.engine.url
    await task_database.engine.dispose()
    task_database.engine = create_async_engine(url)
    task_database.sessions = async_sessionmaker(task_database.engine)
    await scheduled_tasks.poll_scheduled_tasks()
    await asyncio.gather(*list(scheduled_tasks._running_tasks.values()))
    await scheduled_tasks.poll_scheduled_tasks()
    assert received == [(task_type, "更新后的任务", "bot-1", "group-1")]
    persisted = await record(task_database, job_id)
    assert persisted.status == "completed"
    assert persisted.started_at is not None
    assert persisted.finished_at is not None
    assert persisted.lease_until is None


async def test_concurrent_workers_only_execute_task_once(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    received = []

    async def execute(task):
        received.append(task.job_id)

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", execute)
    await asyncio.gather(*(scheduled_tasks.run_scheduled_task(created["job_id"]) for _ in range(4)))
    assert received == [created["job_id"]]


async def test_cancelled_and_rescheduled_tasks_are_not_executed_from_stale_selection(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    cancelled, rescheduled = await create_task(), await create_task()
    for task in [cancelled, rescheduled]:
        await make_due(task_database, task["job_id"])
    tools = management_tools()
    await tools["cancel_scheduled_task"].ainvoke({"job_id": cancelled["job_id"]})
    await tools["update_scheduled_task"].ainvoke({"job_id": rescheduled["job_id"], "delay_hours": 1})

    async def forbidden(task):
        pytest.fail("Cancelled or rescheduled task executed")

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", forbidden)
    for task in [cancelled, rescheduled]:
        await scheduled_tasks.run_scheduled_task(task["job_id"])
    assert (await record(task_database, cancelled["job_id"])).status == "cancelled"
    assert (await record(task_database, rescheduled["job_id"])).status == "scheduled"


async def test_running_task_cannot_be_edited_or_cancelled(task_database):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    claimed = await scheduled_tasks._claim_task(created["job_id"], "worker")
    assert claimed is not None
    for name, args in [("update_scheduled_task", {"content": "禁止修改"}), ("cancel_scheduled_task", {})]:
        result = json.loads(await management_tools()[name].ainvoke({"job_id": created["job_id"], **args}))
        assert result["reason_code"] == "schedule_not_found"
    assert (await record(task_database, created["job_id"])).status == "running"


async def test_cancel_racing_execution_has_one_winner(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    received = []

    async def execute(task):
        received.append(task.job_id)

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", execute)
    cancelled, _ = await asyncio.gather(
        management_tools()["cancel_scheduled_task"].ainvoke({"job_id": created["job_id"]}),
        scheduled_tasks.run_scheduled_task(created["job_id"]),
    )
    if json.loads(cancelled)["status"] == "succeeded":
        assert received == []
        assert (await record(task_database, created["job_id"])).status == "cancelled"
    else:
        assert received == [created["job_id"]]
        assert (await record(task_database, created["job_id"])).status == "completed"


async def test_offline_bot_waits_and_does_not_route_to_another_bot(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task(bot_id="offline")
    await make_due(task_database, created["job_id"])
    await scheduled_tasks.poll_scheduled_tasks()
    assert scheduled_tasks._running_tasks == {}
    assert (await record(task_database, created["job_id"])).status == "scheduled"
    assert await scheduled_tasks._claim_task(created["job_id"], "wrong-bot") is None
    monkeypatch.setattr(scheduled_tasks, "get_bots", lambda: {"offline": object()})
    assert await scheduled_tasks._claim_task(created["job_id"], "right-bot") is not None


async def test_missed_and_abandoned_tasks_are_persisted_without_replay(task_database):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    expired, abandoned = await create_task(), await create_task()
    old = scheduled_tasks.utcnow() - datetime.timedelta(minutes=6)
    await set_record(task_database, expired["job_id"], run_at=old)
    await set_record(task_database, abandoned["job_id"], status="running", lease_until=old, claim_token="dead-worker")
    await scheduled_tasks.poll_scheduled_tasks()
    assert scheduled_tasks._running_tasks == {}
    assert (await record(task_database, expired["job_id"])).status == "missed"
    assert (await record(task_database, abandoned["job_id"])).status == "interrupted"
    history = json.loads(await management_tools()["list_scheduled_tasks"].ainvoke({"status": "all"}))
    assert {task["status"] for task in history["data"]["tasks"]} == {"missed", "interrupted"}


async def test_maintenance_rechecks_rescheduled_tasks_and_renewed_leases(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    expired, abandoned = await create_task(), await create_task()
    now = scheduled_tasks.utcnow()
    old = now - datetime.timedelta(minutes=6)
    future = now + datetime.timedelta(hours=1)
    await set_record(task_database, expired["job_id"], run_at=old)
    await set_record(task_database, abandoned["job_id"], status="running", lease_until=old, claim_token="live-worker")
    sessions_opened = 0

    @asynccontextmanager
    async def get_session():
        nonlocal sessions_opened
        sessions_opened += 1
        if sessions_opened == 2:
            # Another connection changes both rows after discovery, before the
            # maintenance write transaction opens.
            await set_record(task_database, expired["job_id"], run_at=future)
            await set_record(task_database, abandoned["job_id"], lease_until=future)
        async with task_database.sessions() as session:
            yield session

    monkeypatch.setattr(scheduled_tasks, "get_session", get_session)
    await scheduled_tasks.poll_scheduled_tasks()
    assert sessions_opened == 2
    assert (await record(task_database, expired["job_id"])).status == "scheduled"
    assert (await record(task_database, abandoned["job_id"])).status == "running"
    assert scheduled_tasks._running_tasks == {}


async def test_maintenance_processes_backlog_in_bounded_batches(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    first, second = await create_task(), await create_task()
    old = scheduled_tasks.utcnow() - datetime.timedelta(minutes=6)
    await set_record(task_database, first["job_id"], run_at=old)
    await set_record(task_database, second["job_id"], run_at=old)
    monkeypatch.setattr(scheduled_tasks, "MAINTENANCE_BATCH_SIZE", 1)
    await scheduled_tasks.poll_scheduled_tasks()
    assert sorted([(await record(task_database, task["job_id"])).status for task in [first, second]]) == ["missed", "scheduled"]
    assert scheduled_tasks._running_tasks == {}
    await scheduled_tasks.poll_scheduled_tasks()
    assert all([(await record(task_database, task["job_id"])).status == "missed" for task in [first, second]])
    assert scheduled_tasks._running_tasks == {}


async def test_execution_failure_is_recorded_and_not_retried(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])

    async def failed(task):
        raise RuntimeError("send failed")

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", failed)
    await scheduled_tasks.run_scheduled_task(created["job_id"])
    persisted = await record(task_database, created["job_id"])
    assert persisted.status == "failed"
    assert "RuntimeError" in persisted.error
    await scheduled_tasks.poll_scheduled_tasks()
    assert scheduled_tasks._running_tasks == {}


async def test_shutdown_marks_inflight_task_interrupted(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    started = asyncio.Event()

    async def execute(task):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", execute)
    await scheduled_tasks.poll_scheduled_tasks()
    await asyncio.wait_for(started.wait(), timeout=3)
    await scheduled_tasks.stop_scheduled_tasks()
    assert (await record(task_database, created["job_id"])).status == "interrupted"
    assert scheduled_tasks._running_tasks == {}


async def test_heartbeat_renews_claim(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    token = "worker"
    await scheduled_tasks._claim_task(created["job_id"], token)
    original = (await record(task_database, created["job_id"])).lease_until
    future = original + datetime.timedelta(seconds=1)
    monkeypatch.setattr(scheduled_tasks, "utcnow", lambda: future)
    monkeypatch.setattr(scheduled_tasks, "HEARTBEAT_SECONDS", 0.01)
    heartbeat = asyncio.create_task(scheduled_tasks._heartbeat(created["job_id"], token))
    try:
        async def wait_for_renewal():
            while (await record(task_database, created["job_id"])).lease_until == original:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_for_renewal(), timeout=3)
        assert (await record(task_database, created["job_id"])).lease_until == future + datetime.timedelta(seconds=scheduled_tasks.LEASE_SECONDS)
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)


async def test_lease_failure_stops_execution_without_replay(task_database, monkeypatch):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    stopped = asyncio.Event()

    async def execute(task):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def failed_heartbeat(*args):
        raise RuntimeError("database disconnected")

    monkeypatch.setattr(scheduled_tasks, "execute_scheduled_task", execute)
    monkeypatch.setattr(scheduled_tasks, "_heartbeat", failed_heartbeat)
    await scheduled_tasks.run_scheduled_task(created["job_id"])
    assert stopped.is_set()
    assert (await record(task_database, created["job_id"])).status == "failed"


async def test_shutdown_prevents_pending_tasks_from_starting(task_database):
    from nonebot_plugin_ai_groupmate import scheduled_tasks

    created = await create_task()
    await make_due(task_database, created["job_id"])
    await scheduled_tasks.stop_scheduled_tasks()
    await scheduled_tasks.poll_scheduled_tasks()
    assert (await record(task_database, created["job_id"])).status == "scheduled"
    assert scheduled_tasks._running_tasks == {}


@pytest.mark.parametrize("task_type", ["message", "agent"])
async def test_dispatch_preserves_saved_target_and_payload(task_database, monkeypatch, task_type):
    from nonebot_plugin_ai_groupmate import agent, scheduled_tasks
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    created = await create_task(task_type, is_private=True, bot_id="bot-2")
    received = []

    async def execute(*args, **kwargs):
        received.append((args, kwargs))

    monkeypatch.setattr(schedule_tools, "send_scheduled_text", execute)
    monkeypatch.setattr(agent, "_run_scheduled_agent_task", execute)
    await scheduled_tasks.execute_scheduled_task(await record(task_database, created["job_id"]))
    args, kwargs = received[0]
    assert args == ("group-1", created["content" if task_type == "message" else "task"])
    assert kwargs["bot_id"] == "bot-2"
    assert kwargs["is_private"] is True
    if task_type == "message":
        assert kwargs["bot_name"] == "小夏"


def test_schedule_management_skill_and_side_effect_guards():
    from nonebot_plugin_ai_groupmate.agent import _build_builtin_agent_skills
    from nonebot_plugin_ai_groupmate.agent.graph import SIDE_EFFECT_TOOL_NAMES

    skill = next(skill for skill in _build_builtin_agent_skills(
        is_private=False, has_admin_permission=False, mute_tool_instruction="", meme_similar_enabled=False,
    ) if skill.name == "schedule_tools")
    assert isinstance(skill.prompt, str)
    for name in management_tools():
        assert name in skill.prompt
    assert {"update_scheduled_task", "cancel_scheduled_task"} <= SIDE_EFFECT_TOOL_NAMES


def test_scheduled_task_migration_matches_model_and_downgrades():
    import importlib

    from alembic import op
    from sqlalchemy import inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from nonebot_plugin_ai_groupmate.model import ScheduledTask

    migration = importlib.import_module("nonebot_plugin_ai_groupmate.migrations.c8f2a6d1e905_add_scheduled_tasks")
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            table = ScheduledTask.__table__
            columns = {column["name"]: column for column in inspect(connection).get_columns(table.name)}
            assert set(columns) == set(table.columns.keys())
            assert {index["name"] for index in inspect(connection).get_indexes(table.name)} == {index.name for index in table.indexes}
            for column in table.columns:
                assert columns[column.name]["nullable"] == column.nullable
            assert op.get_bind() is connection
            migration.downgrade()
            assert table.name not in inspect(connection).get_table_names()
    engine.dispose()
