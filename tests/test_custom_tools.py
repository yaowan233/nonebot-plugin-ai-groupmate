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
        result = await loader.ainvoke({"skill_name": "score_report"})
        assert result == "full prompt for group-1"
    finally:
        clear_registered_agent_tools()
