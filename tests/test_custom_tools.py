import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _context(agent_tool_context):
    return agent_tool_context(
        db_session=None,
        session_id="group-1",
        request_id="req-1",
        user_id="user-1",
        user_name="tester",
        interface=None,
        send_target=None,  # type: ignore[arg-type]
        is_private=False,
    )


@pytest.mark.asyncio
async def test_registered_agent_skill_can_be_loaded_on_demand():
    from nonebot_plugin_ai_groupmate.agent.custom_tools import (
        AgentSkill,
        AgentToolBundle,
        AgentToolContext,
        register_agent_tool,
        build_agent_skill_index,
        clear_registered_agent_tools,
        create_agent_skill_loader_tool,
        build_registered_agent_extensions,
    )

    clear_registered_agent_tools()

    @register_agent_tool
    def build_skill(ctx: AgentToolContext) -> AgentToolBundle:
        async def prompt(context: AgentToolContext) -> str:
            return f"full prompt for {context.session_id}"

        return AgentToolBundle(
            instructions=["short tool instruction"],
            skills=[
                AgentSkill(
                    name="score_report",
                    description="score report workflow",
                    prompt=prompt,
                )
            ],
        )

    try:
        tools, instructions, skills = await build_registered_agent_extensions(_context(AgentToolContext))
        assert tools == []
        assert instructions == ["short tool instruction"]
        assert [skill.name for skill in skills] == ["score_report"]

        index = build_agent_skill_index(skills)
        assert "score_report" in index
        assert "score report workflow" in index
        assert "full prompt" not in index

        loader = create_agent_skill_loader_tool(skills, _context(AgentToolContext))
        assert loader is not None
        result = json.loads(await loader.ainvoke({"skill_name": "score_report"}))
        assert result["status"] == "succeeded"
        assert result["data"]["instructions"] == "full prompt for group-1"
    finally:
        clear_registered_agent_tools()


@pytest.mark.asyncio
async def test_registered_skill_tools_are_hidden_until_skill_is_loaded(monkeypatch):
    from langchain.tools import tool

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.agent.custom_tools import AgentSkill

    @tool("plugin_example")
    async def plugin_example() -> str:
        """Example plugin tool."""
        return "ok"

    skill = AgentSkill(name="plugin_skill", description="Example", prompt="Use plugin_example.", tools=[plugin_example])
    monkeypatch.setattr(agent.plugin_config, "agent_file_tools_enabled", False)
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context", "relevant_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "load_group_style_context", AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([plugin_example], [], [skill])))
    captured = {}

    def build(_model, tools, _prompt, **kwargs):
        captured.update(tools=tools, **kwargs)
        return object()

    monkeypatch.setattr(agent, "build_chat_graph", build)
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), "group", None, "user", "User", history=[],
                                  group_members=[])  # pyright: ignore[reportArgumentType]
    assert "plugin_example" not in {item.name for item in captured["base_tools"]}
    assert "plugin_example" in {item.name for item in captured["tools"]}
    assert [item.name for item in captured["tools_by_skill"]["plugin_skill"]] == ["plugin_example"]


@pytest.mark.asyncio
async def test_legacy_bundle_instructions_remain_visible_unless_explicitly_deferred():
    from nonebot_plugin_ai_groupmate.agent.custom_tools import (
        AgentToolBundle,
        AgentToolContext,
        register_agent_tool,
        clear_registered_agent_tools,
        build_registered_agent_extensions,
    )

    clear_registered_agent_tools()
    try:
        register_agent_tool(lambda _: AgentToolBundle(instructions=["legacy instruction"]))
        register_agent_tool(lambda _: AgentToolBundle(instructions=["deferred instruction"], defer_instructions=True))
        _, instructions, _ = await build_registered_agent_extensions(_context(AgentToolContext))
        assert instructions == ["legacy instruction"]
    finally:
        clear_registered_agent_tools()
