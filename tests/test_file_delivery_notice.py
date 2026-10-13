import json
import asyncio
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport_error", "timeout", "raised_tool_error"])
@pytest.mark.parametrize("notice_fails", [False, True])
async def test_unknown_file_delivery_notifies_once_without_retry(tmp_path, monkeypatch, failure, notice_fails):
    from nonebot_plugin_alconna import Target, UniMessage

    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    file_attempts = []
    notices = []

    async def send(message, **kwargs):
        file_attempts.append(message)
        if failure == "timeout":
            await asyncio.sleep(10)
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(UniMessage, "send", send)
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), add=lambda row: None)
    file_tools = create_file_tools(db, "notice-group", None, root=tmp_path, user_id="user", bot_id="bot", is_private=False, bot_name="bot", send_target=Target("notice-group"))
    tools = {item.name: item for item in file_tools}
    created = json.loads(await tools["create_file"].ainvoke({"filename": "result.txt", "content": "汇总结果"}))
    file_id = created["data"]["file_id"]

    if failure == "raised_tool_error":
        @tool("send_file")
        async def send_file(file_id: str) -> str:
            """An unexpected failure after dispatch."""
            file_attempts.append(file_id)
            raise RuntimeError("dispatch failed")

        tools["send_file"] = send_file

    @tool("reply_user")
    async def reply_user(content: str, next_step: Literal["end", "continue"]) -> str:
        """Send a text notice."""
        notices.append(content)
        assert next_step == "end"
        if notice_fails:
            raise RuntimeError("text transport unavailable")
        return tool_success("reply_sent", "notice sent", delivery_state="completed")

    class Model:
        calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, **kwargs):
            self.calls += 1
            assert self.calls == 1, "Unknown file delivery must not resume model/tool execution"
            return AIMessage(content="", tool_calls=[{"id": "file-send", "name": "send_file", "args": {"file_id": file_id}}])

    model = Model()
    graph = build_chat_graph(model, [*tools.values(), reply_user], "Send the saved file.", db_session=db, limits=AgentRunLimits(max_llm_calls=1, tool_timeout_seconds=0.05))
    result = await graph.ainvoke({"messages": [HumanMessage(content="把结果文件发给我")], "session_id": "notice-group", "request_id": None})

    assert len(file_attempts) == 1
    assert len(notices) == 1
    assert "文件" in notices[0]
    assert "未确认送达" in notices[0]
    assert "已发送" not in notices[0]
    assert model.calls == 1
    assert result["called_finish"] >= 1
    assert len(result["completed_side_effect_keys"]) == 1
    assert result["reply_count"] == (0 if notice_fails else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expired", "returned_unknown", "notice_timeout"])
async def test_delivery_notice_respects_expiry_and_never_retries(monkeypatch, failure):
    from nonebot_plugin_ai_groupmate.agent import graph as module
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_failure

    attempts = []
    monkeypatch.setattr(module, "is_request_active", AsyncMock(return_value=False))

    @tool("reply_user")
    async def reply_user(content: str, next_step: Literal["end", "continue"]) -> str:
        """A failed text notice must not be retried."""
        attempts.append(content)
        if failure == "notice_timeout":
            await asyncio.sleep(10)
        return tool_failure("reply_failed", "unknown", delivery_state="unknown")

    node = module._make_file_delivery_notice_node(reply_user, module.AgentRunLimits(tool_timeout_seconds=0.01), None)
    result = await node({"session_id": "notice-group", "request_id": "expired" if failure == "expired" else None})
    assert len(attempts) == (0 if failure == "expired" else 1)
    assert result["file_delivery_notice_pending"] is False
    assert result["called_finish"] == 1
    assert "messages" not in result
