import json
import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from nonebot_plugin_ai_groupmate.agent.graph import AgentState


def _state(message: AIMessage, *, tool_count: int = 0) -> "AgentState":
    return {
        "messages": [message],
        "session_id": "group-1",
        "request_id": None,
        "reply_count": 0,
        "tool_count": tool_count,
        "reply_this_round": 0,
        "reply_requires_continuation": False,
        "reaction_this_round": 0,
        "called_finish": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_cached_tokens": 0,
        "llm_cache_creation_tokens": 0,
        "llm_call_count": 0,
        "llm_total_tokens": 0,
        "tool_timeout_count": 0,
        "tool_timeout_names": [],
        "tool_result_truncation_count": 0,
        "side_effect_duplicate_count": 0,
        "completed_side_effect_keys": [],
        "active_skills": [],
        "required_side_effect_completed": False,
        "required_side_effect_unavailable": False,
        "required_side_effect_success_count": 0,
        "required_side_effect_target_count": 1,
        "image_input_disabled": False,
    }


@pytest.mark.asyncio
async def test_tool_limit_is_enforced_before_execution(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import graph as graph_module

    calls: list[str] = []

    @tool("count_call")
    async def count_call(label: str) -> str:
        """Record a tool call for testing."""
        calls.append(label)
        return label

    monkeypatch.setattr(graph_module, "MAX_TOOL_COUNT", 2)
    tool_node = graph_module._make_tool_node(
        {count_call.name: count_call},
        [count_call],
        {},
        graph_module.AgentRunLimits(),
    )
    result = await tool_node(
        _state(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "count_call", "args": {"label": "first"}, "id": "1"},
                    {"name": "count_call", "args": {"label": "second"}, "id": "2"},
                ]
            ),
            tool_count=1,
        )
    )

    assert calls == ["first"]
    assert result["tool_count"] == 2
    assert "上限" in result["messages"][-1].content


class _ToolSpyModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = iter(responses)
        self.bound_tool_names: list[tuple[str, ...]] = []
        self.invoke_count = 0
        self.invoke_messages = []

    def bind_tools(self, tools):
        self.bound_tool_names.append(tuple(tool.name for tool in tools))
        return self

    async def ainvoke(self, messages):
        self.invoke_count += 1
        self.invoke_messages.append(messages)
        return next(self.responses)


class _InvalidImageThenResponseModel:
    def __init__(self, response: AIMessage):
        self.response = response
        self.invoke_messages = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(messages)
        if len(self.invoke_messages) == 1:
            raise RuntimeError(
                "Error code: 400 - The image format is illegal and cannot be opened"
            )
        return self.response


@pytest.mark.asyncio
async def test_invalid_image_error_retries_with_text_only_messages():
    from nonebot_plugin_ai_groupmate.agent.graph import (
        AgentRunLimits,
        _make_agent_node,
    )

    model = _InvalidImageThenResponseModel(AIMessage(content="recovered"))
    agent_node = _make_agent_node(
        model,
        [],
        "system",
        {},
        AgentRunLimits(),
    )
    state = _state(AIMessage(content=[
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
    ]))

    result = await agent_node(state)

    assert len(model.invoke_messages) == 2
    assert any(
        isinstance(item, dict) and item.get("type") == "image_url"
        for message in model.invoke_messages[0]
        if isinstance(message.content, list)
        for item in message.content
    )
    assert not any(
        isinstance(item, dict) and item.get("type") == "image_url"
        for message in model.invoke_messages[1]
        if isinstance(message.content, list)
        for item in message.content
    )
    assert result["image_input_disabled"] is True
    assert result["llm_call_count"] == 2


@pytest.mark.asyncio
async def test_agent_tracks_provider_usage_without_openai_callback():
    from nonebot_plugin_ai_groupmate.agent.graph import (
        AgentRunLimits,
        _make_agent_node,
    )

    model = _ToolSpyModel([
        AIMessage(
            content="done",
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            },
        )
    ])
    agent_node = _make_agent_node(model, [], "system", {}, AgentRunLimits())

    result = await agent_node(_state(AIMessage(content="question")))

    assert result["llm_input_tokens"] == 120
    assert result["llm_output_tokens"] == 30
    assert result["llm_total_tokens"] == 150


@pytest.mark.asyncio
async def test_skill_only_exposes_its_tools_after_loading():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    @tool("load_agent_skill")
    async def load_agent_skill(skill_name: str) -> str:
        """Load a skill instruction for testing."""
        return f"loaded {skill_name}"

    @tool("advanced_tool")
    async def advanced_tool() -> str:
        """A tool that requires the advanced skill."""
        return "done"

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_agent_skill",
                        "args": {"skill_name": "advanced"},
                        "id": "load-1",
                    }
                ]
            ),
            AIMessage(content="", tool_calls=[{"name": "advanced_tool", "args": {}, "id": "tool-1"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(
        model,
        [load_agent_skill, advanced_tool, finish],
        "system",
        base_tools=[load_agent_skill, finish],
        tools_by_skill={"advanced": [advanced_tool]},
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert model.bound_tool_names == [
        ("load_agent_skill", "finish"),
        ("load_agent_skill", "finish", "advanced_tool"),
    ]
    assert result["active_skills"] == ["advanced"]


@pytest.mark.asyncio
async def test_empty_model_response_is_corrected_once():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    sent: list[str] = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Send a corrected reply for testing."""
        sent.append(content)
        return "sent"

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(content=""),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "reply_user",
                    "args": {"content": "corrected", "next_step": "end"},
                    "id": "reply-1",
                }],
            ),
        ]
    )
    graph = build_chat_graph(model, [reply_user, finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert sent == ["corrected"]
    assert model.invoke_count == 2
    assert result["llm_call_count"] == 2


@pytest.mark.asyncio
async def test_repeated_empty_model_response_does_not_loop_forever():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel([AIMessage(content=""), AIMessage(content="")])
    graph = build_chat_graph(model, [finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert model.invoke_count == 2
    assert result["llm_call_count"] == 2


@pytest.mark.asyncio
async def test_reply_is_deferred_until_other_tool_work_finishes():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    events: list[str] = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Send a reply for testing."""
        events.append(f"reply:{content}")
        return "sent"

    @tool("load_agent_skill")
    async def load_agent_skill(skill_name: str) -> str:
        """Load a skill for testing."""
        events.append(f"load:{skill_name}")
        return f"loaded {skill_name}"

    @tool("save_preference")
    async def save_preference() -> str:
        """Persist a preference for testing."""
        events.append("save")
        return "saved"

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "premature", "next_step": "end"},
                        "id": "reply-early",
                    },
                    {
                        "name": "load_agent_skill",
                        "args": {"skill_name": "memory"},
                        "id": "load-1",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "save_preference",
                    "args": {},
                    "id": "save-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "reply_user",
                    "args": {"content": "saved", "next_step": "end"},
                    "id": "reply-final",
                }],
            ),
        ]
    )
    graph = build_chat_graph(
        model,
        [reply_user, load_agent_skill, save_preference],
        "system",
        base_tools=[reply_user, load_agent_skill],
        tools_by_skill={"memory": [save_preference]},
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert events == ["load:memory", "save", "reply:saved"]
    assert result["reply_count"] == 1
    assert result["active_skills"] == ["memory"]


@pytest.mark.asyncio
async def test_reply_with_end_stops_without_an_extra_model_call():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    sent: list[str] = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Send a reply for testing."""
        sent.append(content)
        return "sent"

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "one message", "next_step": "end"},
                        "id": "reply-1",
                    }
                ],
            )
        ]
    )
    graph = build_chat_graph(model, [reply_user], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert sent == ["one message"]
    assert model.invoke_count == 1
    assert result["reply_requires_continuation"] is False


@pytest.mark.asyncio
async def test_reply_with_continue_returns_to_model_for_the_next_message():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    sent: list[str] = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Send a reply for testing."""
        sent.append(content)
        return "sent"

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "first message", "next_step": "continue"},
                        "id": "reply-1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "second message", "next_step": "end"},
                        "id": "reply-2",
                    }
                ],
            ),
        ]
    )
    graph = build_chat_graph(model, [reply_user], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert sent == ["first message", "second message"]
    assert model.invoke_count == 2
    assert result["reply_count"] == 2


@pytest.mark.asyncio
async def test_failed_final_reply_returns_to_model_instead_of_ending():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Fail a reply for testing."""
        return json.dumps({"status": "failed", "message": "send failed"})

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "message", "next_step": "end"},
                        "id": "reply-1",
                    }
                ],
            ),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(model, [reply_user, finish], "system")

    await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert model.invoke_count == 2


@pytest.mark.asyncio
async def test_duplicate_side_effect_is_executed_once():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    calls: list[str] = []

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Record a side effect for testing."""
        calls.append(pic_id)
        return "sent"

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "send_meme_image", "args": {"pic_id": "42"}, "id": "send-1"},
                    {"name": "send_meme_image", "args": {"pic_id": "42"}, "id": "send-2"},
                ],
            ),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(model, [send_meme_image, finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert calls == ["42"]
    assert result["side_effect_duplicate_count"] == 1


@pytest.mark.asyncio
async def test_timed_out_side_effect_ends_run_without_retry():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph

    calls: list[str] = []

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Time out after the side effect may have been dispatched."""
        calls.append(pic_id)
        if len(calls) == 1:
            await asyncio.sleep(0.05)
        return "sent"

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    repeated_call = {"name": "send_meme_image", "args": {"pic_id": "42"}}
    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-1"}]),
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-2"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(
        model,
        [send_meme_image, finish],
        "system",
        limits=AgentRunLimits(tool_timeout_seconds=0.01),
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert calls == ["42"]
    assert model.invoke_count == 1
    assert result["tool_timeout_count"] == 1
    assert result["tool_timeout_names"] == ["send_meme_image"]
    assert result["side_effect_duplicate_count"] == 0
    assert result["called_finish"] == 1
    assert len(result["completed_side_effect_keys"]) == 1


@pytest.mark.asyncio
async def test_unknown_delivery_side_effect_ends_run_without_retry():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    calls: list[str] = []

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Report that dispatch started but its outcome is unknown."""
        calls.append(pic_id)
        return json.dumps({
            "schema_version": 1,
            "ok": False,
            "status": "failed",
            "reason_code": "delivery_failed",
            "message": "发送结果未知。",
            "retryable": False,
            "delivery_state": "unknown",
        })

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    repeated_call = {"name": "send_meme_image", "args": {"pic_id": "42"}}
    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-1"}]),
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-2"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(model, [send_meme_image, finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert calls == ["42"]
    assert model.invoke_count == 1
    assert result["called_finish"] == 1
    assert len(result["completed_side_effect_keys"]) == 1


@pytest.mark.asyncio
async def test_raised_side_effect_error_ends_run_without_retry():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    calls: list[str] = []

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Raise after dispatch may have started."""
        calls.append(pic_id)
        raise RuntimeError("provider disconnected")

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    repeated_call = {"name": "send_meme_image", "args": {"pic_id": "42"}}
    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-1"}]),
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-2"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(model, [send_meme_image, finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert calls == ["42"]
    assert model.invoke_count == 1
    assert result["called_finish"] == 1
    assert len(result["completed_side_effect_keys"]) == 1
    tool_result = json.loads(result["messages"][-1].content)
    assert tool_result["retryable"] is False
    assert tool_result["delivery_state"] == "unknown"


@pytest.mark.asyncio
async def test_raised_tool_error_reports_safe_failure_reason_to_model():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    @tool("lookup_data")
    async def lookup_data() -> str:
        """Raise an unexpected provider error."""
        raise ConnectionError("authorization=secret-value")

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel([
        AIMessage(content="", tool_calls=[{
            "name": "lookup_data",
            "args": {},
            "id": "lookup-1",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "finish",
            "args": {},
            "id": "finish-1",
        }]),
    ])
    graph = build_chat_graph(model, [lookup_data, finish], "system")

    await graph.ainvoke(_state(AIMessage(content="placeholder")))

    failure = json.loads(model.invoke_messages[1][-1].content)
    assert failure["reason_code"] == "tool_execution_failed"
    assert failure["retryable"] is True
    assert failure["data"]["error_type"] == "ConnectionError"
    assert "ConnectionError" in failure["message"]
    assert "secret-value" not in model.invoke_messages[1][-1].content


@pytest.mark.asyncio
async def test_failed_side_effect_can_be_retried():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    calls: list[str] = []

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Report one failure, then complete the same side effect."""
        calls.append(pic_id)
        status = "failed" if len(calls) == 1 else "sent"
        return json.dumps({"status": status})

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    repeated_call = {"name": "send_meme_image", "args": {"pic_id": "42"}}
    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-1"}]),
            AIMessage(content="", tool_calls=[{**repeated_call, "id": "send-2"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(model, [send_meme_image, finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert calls == ["42", "42"]
    assert result["side_effect_duplicate_count"] == 0


@pytest.mark.asyncio
async def test_required_meme_send_blocks_text_and_finish_until_image_is_sent():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    events: list[str] = []

    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """Return one approved meme candidate."""
        events.append(f"search:{description}")
        return json.dumps({
            "success": True,
            "images": [{"pic_id": "42", "description": "震惊"}],
        })

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Send one approved meme candidate."""
        events.append(f"send:{pic_id}")
        return json.dumps({"status": "sent", "message": "sent"})

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(content="没找到，下次补上"),
            AIMessage(
                content="",
                tool_calls=[{"name": "finish", "args": {}, "id": "finish-early"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_meme_image",
                    "args": {"description": "震惊"},
                    "id": "search-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "send_meme_image",
                    "args": {"pic_id": "42"},
                    "id": "send-1",
                }],
            ),
        ]
    )
    graph = build_chat_graph(
        model,
        [search_meme_image, send_meme_image, finish],
        "system",
        required_side_effect_tool="send_meme_image",
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert events == ["search:震惊", "send:42"]
    assert result["required_side_effect_completed"] is True
    assert result["called_finish"] == 1
    assert model.invoke_count == 4


@pytest.mark.asyncio
async def test_required_multi_meme_send_waits_for_distinct_images():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    events: list[str] = []

    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """Return three approved meme candidates."""
        events.append(f"search:{description}")
        return json.dumps({
            "success": True,
            "count": 3,
            "images": [
                {"pic_id": "41", "description": "第一张"},
                {"pic_id": "42", "description": "第二张"},
                {"pic_id": "43", "description": "第三张"},
            ],
        })

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """Send one approved meme candidate."""
        events.append(f"send:{pic_id}")
        return json.dumps({"status": "sent", "message": "sent"})

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel([
        AIMessage(content="", tool_calls=[{
            "name": "search_meme_image",
            "args": {"description": "龙图"},
            "id": "search-1",
        }]),
        *[
            AIMessage(content="", tool_calls=[{
                "name": "send_meme_image",
                "args": {"pic_id": str(pic_id)},
                "id": f"send-{pic_id}",
            }])
            for pic_id in (41, 42, 43)
        ],
    ])
    graph = build_chat_graph(
        model,
        [search_meme_image, send_meme_image, finish],
        "system",
        required_side_effect_tool="send_meme_image",
        required_side_effect_count=3,
    )
    state = _state(AIMessage(content="placeholder"))
    state["required_side_effect_target_count"] = 3

    result = await graph.ainvoke(state)

    assert events == ["search:龙图", "send:41", "send:42", "send:43"]
    assert result["required_side_effect_success_count"] == 3
    assert result["required_side_effect_target_count"] == 3
    assert result["required_side_effect_completed"] is True
    assert result["called_finish"] == 1
    assert model.invoke_count == 4


@pytest.mark.asyncio
async def test_required_meme_send_may_finish_when_search_pool_is_empty():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph

    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """Report that the candidate pool is truly empty."""
        return json.dumps({
            "success": False,
            "images": [],
            "reason_code": "no_candidates",
        })

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """This must not be called without a candidate."""
        raise AssertionError(pic_id)

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_meme_image",
                    "args": {"description": "震惊"},
                    "id": "search-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}],
            ),
        ]
    )
    graph = build_chat_graph(
        model,
        [search_meme_image, send_meme_image, finish],
        "system",
        required_side_effect_tool="send_meme_image",
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert result["required_side_effect_unavailable"] is True
    assert result["required_side_effect_completed"] is False
    assert result["called_finish"] == 1


@pytest.mark.asyncio
async def test_long_tool_results_are_truncated_before_the_next_model_call():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph

    @tool("search_web")
    async def search_web(query: str) -> str:
        """Return an oversized result for testing."""
        return "x" * 100

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "search-1"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(
        model,
        [search_web, finish],
        "system",
        limits=AgentRunLimits(tool_result_max_chars=20),
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert result["tool_result_truncation_count"] == 1


@pytest.mark.asyncio
async def test_timed_out_search_cannot_finish_without_replying_to_user():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph

    replies: list[str] = []
    generated_reply = "我刚才联网查询时超时了，你可以给我一个更短的关键词再试。"

    @tool("search_web")
    async def search_web(query: str) -> str:
        """Take too long for testing."""
        await asyncio.sleep(0.05)
        return query

    @tool("reply_user")
    async def reply_user(content: str, next_step: str = "end") -> str:
        """Record the user-visible reply for testing."""
        replies.append(content)
        return next_step

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "search-1"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
            AIMessage(content="", tool_calls=[{
                "name": "reply_user",
                "args": {"content": generated_reply, "next_step": "end"},
                "id": "reply-1",
            }]),
        ]
    )
    graph = build_chat_graph(
        model,
        [search_web, reply_user, finish],
        "system",
        limits=AgentRunLimits(tool_timeout_seconds=0.001),
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert model.invoke_count == 3
    assert result["tool_timeout_count"] == 1
    assert result["tool_timeout_names"] == ["search_web"]
    assert result["reply_count"] == 1
    assert replies == [generated_reply]


@pytest.mark.asyncio
async def test_provider_reported_search_timeout_cannot_finish_without_replying():
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_failure

    replies: list[str] = []
    generated_reply = "联网搜索刚才超时了，我可以换一组关键词继续帮你查。"

    @tool("search_web")
    async def search_web(query: str) -> str:
        """Return a timeout already handled by the search provider wrapper."""
        return tool_failure(
            "timeout",
            "联网搜索暂时超时，可以缩短关键词后重试一次。",
            retryable=True,
        )

    @tool("reply_user")
    async def reply_user(content: str, next_step: str = "end") -> str:
        """Record the user-visible reply for testing."""
        replies.append(content)
        return next_step

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{
                "name": "search_web",
                "args": {"query": "x"},
                "id": "search-1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "finish",
                "args": {},
                "id": "finish-1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "reply_user",
                "args": {"content": generated_reply, "next_step": "end"},
                "id": "reply-1",
            }]),
        ]
    )
    graph = build_chat_graph(model, [search_web, reply_user, finish], "system")

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert result["tool_timeout_count"] == 1
    assert result["tool_timeout_names"] == ["search_web"]
    assert result["reply_count"] == 1
    assert replies == [generated_reply]


@pytest.mark.asyncio
async def test_llm_call_budget_stops_before_another_model_turn():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph

    @tool("search_web")
    async def search_web(query: str) -> str:
        """Return a search result for testing."""
        return query

    @tool("finish")
    def finish() -> str:
        """End this test graph."""
        return ""

    model = _ToolSpyModel(
        [
            AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "search-1"}]),
            AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "finish-1"}]),
        ]
    )
    graph = build_chat_graph(
        model,
        [search_web, finish],
        "system",
        limits=AgentRunLimits(max_llm_calls=1),
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert model.invoke_count == 1
    assert result["llm_call_count"] == 1


def test_token_budget_ends_the_loop():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, _should_continue

    state = _state(AIMessage(content="placeholder"))
    state["llm_total_tokens"] = 10

    assert _should_continue(state, AgentRunLimits(max_total_tokens=10)) == "end"


@pytest.mark.asyncio
async def test_partial_rollback_session_is_recovered_before_tool_execution():
    from nonebot_plugin_ai_groupmate.agent.graph import _recover_db_session

    class _Session:
        is_active = False

        def __init__(self):
            self.rollback_count = 0

        async def rollback(self):
            self.rollback_count += 1
            self.is_active = True

    session = _Session()
    await _recover_db_session(session)

    assert session.rollback_count == 1
    assert session.is_active is True


@pytest.mark.asyncio
async def test_successful_tool_commits_before_returning_to_model():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, _make_tool_node

    class _Session:
        def __init__(self):
            self.commit_count = 0

        async def commit(self):
            self.commit_count += 1

    @tool("database_tool")
    async def database_tool() -> str:
        """Pretend to write through the shared database session."""
        return "done"

    session = _Session()
    tool_node = _make_tool_node(
        {database_tool.name: database_tool},
        [database_tool],
        {},
        AgentRunLimits(),
        db_session=session,
    )

    await tool_node(
        _state(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "database_tool", "args": {}, "id": "tool-1"}
                ],
            )
        )
    )

    assert session.commit_count == 1
