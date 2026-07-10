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
        "llm_cached_tokens": 0,
        "llm_cache_creation_tokens": 0,
        "llm_call_count": 0,
        "llm_total_tokens": 0,
        "tool_timeout_count": 0,
        "tool_result_truncation_count": 0,
        "side_effect_duplicate_count": 0,
        "completed_side_effect_keys": [],
        "active_skills": [],
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

    def bind_tools(self, tools):
        self.bound_tool_names.append(tuple(tool.name for tool in tools))
        return self

    async def ainvoke(self, messages):
        self.invoke_count += 1
        return next(self.responses)


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
async def test_timed_out_tool_returns_control_to_the_model():
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph

    @tool("search_web")
    async def search_web(query: str) -> str:
        """Take too long for testing."""
        await asyncio.sleep(0.05)
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
        limits=AgentRunLimits(tool_timeout_seconds=0.001),
    )

    result = await graph.ainvoke(_state(AIMessage(content="placeholder")))

    assert model.invoke_count == 2
    assert result["tool_timeout_count"] == 1


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
