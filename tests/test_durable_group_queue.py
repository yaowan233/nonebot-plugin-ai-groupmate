from uuid import uuid4
from types import SimpleNamespace
from typing import Any, cast
from dataclasses import replace

import pytest

from tests.fake import fake_group_message_event_v11


def make_request(group: str, message_id: int):
    from nonebot_plugin_uninfo import SceneType
    from nonebot.adapters.onebot.v11 import Message
    from nonebot_plugin_uninfo.model import User, Scene, Session

    from nonebot_plugin_ai_groupmate.reply_request import ReplyRequest

    event = fake_group_message_event_v11(group_id=int(group), message_id=message_id, original_message=Message("test"))
    bot = SimpleNamespace(self_id="1", adapter=SimpleNamespace(get_name=lambda: "OneBot V11"))
    session = Session(self_id="1", adapter="OneBot V11", scope="QQClient",
                      scene=Scene(id=group, type=SceneType.GROUP), user=User(id=str(event.user_id)))
    return ReplyRequest(request_id=uuid4().hex, session=session, interface=cast(Any, object()), bot=cast(Any, bot),
                        event=event, bot_name="bot", user_id=session.user.id, user_name="Alice",
                        is_tome=True, is_continuous=False, reply_to_id=None, message_id=str(message_id))


@pytest.mark.asyncio
async def test_staging_merge_and_restart_preserve_messages(monkeypatch):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import durable_group_queue as queue
    from nonebot_plugin_ai_groupmate.model import GroupReplyJob

    group = str(int(uuid4().hex[:8], 16))
    first, _ = await queue.stage(make_request(group, 101), "第一句")
    second, _ = await queue.stage(make_request(group, 102), "第二句")
    assert first.durable_job_id
    assert second.durable_job_id
    duplicate, inserted = await queue.stage(make_request(group, 101), "重复")
    assert not inserted
    assert duplicate.durable_job_id is None
    second = replace(second, input_batch=(("101", "第一句"), ("102", "第二句")))
    assert second.durable_job_id is not None
    assert await queue.admit(second)
    async with get_session() as db:
        old = await db.get(GroupReplyJob, first.durable_job_id)
        job = await db.get(GroupReplyJob, second.durable_job_id)
        assert old
        assert old.status == "merged"
        assert job
        assert job.status == "queued"
        monkeypatch.setattr(queue, "get_interface", lambda _: object())
        recovered = queue.restore(job, second.bot)
        assert recovered.input_batch == second.input_batch
        assert recovered.event.get_message().extract_plain_text() == "test"
        assert recovered.session.scene.id == group
    assert await queue.transition(second.durable_job_id, "queued", "running")
    await queue.recover_startup()
    async with get_session() as db:
        job = await db.get(GroupReplyJob, second.durable_job_id)
        assert job
        assert job.status == "interrupted"
    assert not await queue.transition(second.durable_job_id, "queued", "running")


@pytest.mark.asyncio
async def test_restart_recovers_staged_requests_and_cancellation():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import durable_group_queue as queue
    from nonebot_plugin_ai_groupmate.model import GroupReplyJob

    group = str(int(uuid4().hex[:8], 16))
    first, _ = await queue.stage(make_request(group, 201), "问一下")
    second, _ = await queue.stage(make_request(group, 202), "不用了")
    assert first.durable_job_id
    assert second.durable_job_id
    await queue.cancel_staged("1", group, first.user_id)
    await queue.recover_startup()
    async with get_session() as db:
        for request in (first, second):
            job = await db.get(GroupReplyJob, request.durable_job_id)
            assert job
            assert job.status == "cancelled"
    other, _ = await queue.stage(make_request(group, 203), "重启前的问题")
    assert other.durable_job_id
    await queue.recover_startup()
    async with get_session() as db:
        job = await db.get(GroupReplyJob, other.durable_job_id)
        assert job
        assert job.status == "queued"


@pytest.mark.asyncio
async def test_recovery_reconstructs_a_partial_burst():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import durable_group_queue as queue
    from nonebot_plugin_ai_groupmate.model import GroupReplyJob

    group = str(int(uuid4().hex[:8], 16))
    first, _ = await queue.stage(make_request(group, 301), "前半句")
    second, _ = await queue.stage(make_request(group, 302), "后半句")
    await queue.recover_startup()
    async with get_session() as db:
        old = await db.get(GroupReplyJob, first.durable_job_id)
        new = await db.get(GroupReplyJob, second.durable_job_id)
        assert old is not None
        assert new is not None
        assert old.status == "merged"
        assert new.status == "queued"
        assert new.payload["input_batch"] == [["301", "前半句"], ["302", "后半句"]]


@pytest.mark.asyncio
async def test_only_claimed_job_executes_and_failed_execution_is_not_replayed(monkeypatch):
    from unittest.mock import AsyncMock

    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import durable_group_queue as queue
    from nonebot_plugin_ai_groupmate.model import GroupReplyJob

    group = str(int(uuid4().hex[:8], 16))
    request, _ = await queue.stage(make_request(group, 401), "问题")
    assert request.durable_job_id
    request = replace(request, input_batch=(("401", "问题"),))
    assert await queue.admit(request)
    execute = AsyncMock(side_effect=RuntimeError("failed after possible side effect"))
    monkeypatch.setattr(plugin, "_execute_reply_request_inner", execute)
    with pytest.raises(RuntimeError):
        await plugin._execute_reply_request(request)
    await plugin._execute_reply_request(request)
    execute.assert_awaited_once()
    async with get_session() as db:
        row = await db.get(GroupReplyJob, request.durable_job_id)
        assert row is not None
        assert row.status == "interrupted"


@pytest.mark.asyncio
async def test_ingress_and_poller_dispatch_a_saved_onebot_request(monkeypatch):
    from unittest.mock import AsyncMock

    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import durable_group_queue as queue
    from nonebot_plugin_ai_groupmate.model import GroupReplyJob

    group = str(int(uuid4().hex[:8], 16))
    request = make_request(group, 501)
    dispatched = AsyncMock(return_value=True)
    monkeypatch.setattr(plugin, "_durable_queue_initialized", True)
    monkeypatch.setattr(plugin, "_queue_group_reply_request", dispatched)
    monkeypatch.setattr(plugin.plugin_config, "group_message_debounce_seconds", 0)
    await plugin._submit_reply_input(group, request, "501", "真实提问")
    assert dispatched.await_args is not None
    saved = dispatched.await_args.args[1]
    assert saved.durable_job_id is not None
    async with get_session() as db:
        job = await db.get(GroupReplyJob, saved.durable_job_id)
        assert job is not None
        assert job.status == "queued"
    monkeypatch.setattr(plugin, "get_bots", lambda: {"1": request.bot})
    monkeypatch.setattr(queue, "get_interface", lambda _: object())
    dispatched.reset_mock()
    await plugin._poll_durable_group_queue()
    assert dispatched.await_args is not None
    assert dispatched.await_args.args[1].input_batch == (("501", "真实提问"),)
