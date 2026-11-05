import json
import asyncio
from uuid import uuid4
from types import SimpleNamespace
from typing import Any, cast

import pytest


def reply_request(request_id: str):
    return cast(Any, SimpleNamespace(
        request_id=request_id, session=None, interface=None, bot=None,
        event=None, bot_name="bot", user_id=request_id, user_name=request_id,
        is_tome=True, is_continuous=False, reply_to_id=None,
    ))


@pytest.mark.parametrize("chat_group", ["group", "other-group"])
@pytest.mark.parametrize("hold_chat_slot", [False, True])
async def test_chat_completes_while_image_provider_is_still_waiting(monkeypatch, tmp_path, chat_group, hold_chat_slot):
    from test_image_generation import png, tool_fixture

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.concurrency import agent_run_gate
    from nonebot_plugin_ai_groupmate.reply_guard import is_request_active
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    module, _, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path, request_id="draw")
    drawing = asyncio.Event()
    image_ready = asyncio.Event()
    chatted = asyncio.Event()
    delivered = asyncio.Event()
    release_chat = asyncio.Event()
    old_limit = agent_run_gate.limit
    agent_run_gate.configure(1)
    monkeypatch.setattr(plugin.plugin_config, "agent_max_concurrency_per_group", 1)
    plugin._group_reply_states.clear()

    async def slow_image(*args, **kwargs):
        drawing.set()
        await image_ready.wait()
        return GeneratedImage(png(), {})

    provider.side_effect = slow_image

    async def handle(request, *args, **kwargs):
        if request.request_id == "draw":
            result = json.loads(await generate.ainvoke({"prompt": "draw a cat"}))
            assert result["status"] == "succeeded"
            assert await is_request_active("group", "draw")
            delivered.set()
        else:
            assert not image_ready.is_set()
            chatted.set()
            if hold_chat_slot:
                await release_chat.wait()

    monkeypatch.setattr(plugin, "handle_reply_request", handle)
    try:
        assert await plugin._queue_group_reply_request("group", reply_request("draw"))
        await asyncio.wait_for(drawing.wait(), 1)
        assert await plugin._queue_group_reply_request(chat_group, reply_request("chat"))
        await asyncio.wait_for(chatted.wait(), 0.3)
        assert not delivered.is_set()
        image_ready.set()
        if hold_chat_slot:
            await asyncio.sleep(0.02)
            assert not delivered.is_set(), "Drawing must reclaim its execution slot before resuming"
            release_chat.set()
        await asyncio.wait_for(delivered.wait(), 1)
        provider.assert_awaited_once()
    finally:
        image_ready.set()
        release_chat.set()
        for _ in range(5):
            tasks = [t for s in plugin._group_reply_states.values() for t in s.addressed_tasks.values()]
            if not tasks:
                break
            await asyncio.gather(*tasks, return_exceptions=True)
        plugin._group_reply_states.clear()
        agent_run_gate.configure(old_limit)


async def test_same_group_chat_completes_while_background_drawing_runs(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import drawing_poll, drawing_service
    from nonebot_plugin_ai_groupmate.concurrency import agent_run_gate
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage
    from nonebot_plugin_ai_groupmate.agent.image_generation_tools import create_image_generation_tool

    group, bot_id = f"background-chat-{uuid4()}", f"background-bot-{uuid4()}"
    cfg = image_config()
    queued, drawing, finish_drawing, chatted = (asyncio.Event() for _ in range(4))
    old_limit = agent_run_gate.limit
    agent_run_gate.configure(1)
    monkeypatch.setattr(plugin.plugin_config, "agent_max_concurrency_per_group", 1)
    plugin._group_reply_states.clear()
    workspace = FileWorkspace(tmp_path, session_id=group, user_id="alice", bot_id=bot_id, is_private=False, peer_bot_ids=())

    async def provider(*_args, **_kwargs):
        drawing.set()
        await finish_drawing.wait()
        return GeneratedImage(png(), {})

    async def handle(request, *_args, **_kwargs):
        if request.request_id == "draw":
            async with get_session() as db:
                tool = create_image_generation_tool(db, group, "draw", config=cfg, workspace=workspace, pic_dir=tmp_path,
                    current_message_id="original", replied_message_id=None, current_user_id="alice", background=True,
                    bot_id=bot_id, bot_name="miyuki", platform="qq", is_private=False)
                result = json.loads(await tool.ainvoke({"prompt": "画一只猫"}))
                assert result["reason_code"] == "drawing_queued"
                queued.set()
        else:
            assert not finish_drawing.is_set()
            chatted.set()

    monkeypatch.setattr(drawing_service, "request_image", provider)
    monkeypatch.setattr(drawing_poll, "get_bots", lambda: {bot_id: SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))})
    monkeypatch.setattr(plugin, "handle_reply_request", handle)
    drawing_poll.start_drawing_jobs()
    try:
        assert await plugin._queue_group_reply_request(group, reply_request("draw"))
        await asyncio.wait_for(queued.wait(), 3)
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        await asyncio.wait_for(drawing.wait(), 3)
        assert await plugin._queue_group_reply_request(group, reply_request("chat"))
        await asyncio.wait_for(chatted.wait(), 1)
        assert not finish_drawing.is_set()
    finally:
        finish_drawing.set()
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()
        for state in list(plugin._group_reply_states.values()):
            await asyncio.gather(*state.addressed_tasks.values(), return_exceptions=True)
        plugin._group_reply_states.clear()
        agent_run_gate.configure(old_limit)


@pytest.mark.parametrize("cancel_child", [False, True])
async def test_cancelled_drawing_does_not_leak_or_overfill_agent_slots(cancel_child):
    from nonebot_plugin_ai_groupmate.concurrency import ConcurrencyGate

    gate = ConcurrencyGate("cancel-drawing", 1)
    suspended = asyncio.Event()

    async def drawing():
        async with gate.suspend_slot():
            suspended.set()
            await asyncio.Event().wait()

    async def run():
        async with gate.slot():
            if cancel_child:
                child = asyncio.create_task(drawing())
                await suspended.wait()
                child.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await child
                # A cancelled tool's Agent still owns exactly one slot.
                async with gate.slot(wait=False) as admitted:
                    assert admitted is False
            else:
                await drawing()

    task = asyncio.create_task(run())
    await suspended.wait()
    if not cancel_child:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    async with gate.slot(wait=False) as admitted:
        assert admitted is True
        async with gate.slot(wait=False) as second:
            assert second is False
