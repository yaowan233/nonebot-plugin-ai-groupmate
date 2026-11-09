import re
import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest


def test_trace_id_is_stable_scoped_and_does_not_expose_identifiers():
    from nonebot_plugin_ai_groupmate.message_trace import trace_id

    value = trace_id("private-session-marker", "private-request-marker")
    assert re.fullmatch(r"[0-9a-f]{16}", value)
    assert value == trace_id("private-session-marker", "private-request-marker")
    assert value != trace_id("another-session", "private-request-marker")
    assert value != trace_id("private-session-marker", "another-request")
    assert trace_id("private-session-marker", None) == "-"


@pytest.mark.asyncio
async def test_worker_trace_measures_real_global_slot_wait(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.concurrency import agent_run_gate
    from nonebot_plugin_ai_groupmate.message_trace import trace_id

    async def handle(*args):
        return None

    monkeypatch.setattr(plugin, "handle_reply_request", handle)
    monkeypatch.setattr(plugin, "message_pilot_enabled", lambda *args, **kwargs: True)
    request = SimpleNamespace(
        request_id="private-request", session=SimpleNamespace(scene=SimpleNamespace(id="group", type=None)),
        interface=None, bot=None, event=None, bot_name="bot", user_id="user", user_name=None,
        is_tome=True, is_continuous=False, reply_to_id=None, mentioned_user_ids=("member",),
        created_monotonic=plugin.time.monotonic(),
    )
    logs = []
    sink = plugin.logger.add(lambda message: logs.append(message.record["message"]))
    old_limit = agent_run_gate.limit
    agent_run_gate.configure(1)
    try:
        async with agent_run_gate.slot():
            task = asyncio.create_task(plugin._execute_reply_request(cast(Any, request)))
            await asyncio.sleep(0.03)
            assert not task.done()
        await task
    finally:
        agent_run_gate.configure(old_limit)
        plugin.logger.remove(sink)
    line = next(line for line in logs if "stage=worker_admitted" in line)
    assert f"trace={trace_id('group', 'private-request')}" in line
    match = re.search(r"global_slot_wait_ms=(\d+)", line)
    assert match is not None
    assert int(match.group(1)) >= 20
    assert "private-request" not in line
