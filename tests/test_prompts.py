import pytest


@pytest.mark.parametrize("is_private", [False, True])
def test_personality_setting_is_injected_as_fixed_knowledge(is_private):
    from nonebot_plugin_ai_groupmate.agent.prompts import build_chat_system_prompt

    result = build_chat_system_prompt(
        bot_name="bot",
        is_private=is_private,
        personality_setting="用户询问加群时，回复群号 123456789。",
        relation_context="relation",
        group_context="group",
        recent_relations_context="recent",
        permission_status="",
        mute_tool_instruction="",
        reaction_tool_instruction="",
    )

    assert "【自定义设定与固定知识】" in result.system_prompt
    assert "用户询问加群时，回复群号 123456789。" in result.system_prompt


def test_empty_personality_setting_does_not_add_fixed_knowledge_section():
    from nonebot_plugin_ai_groupmate.agent.prompts import build_chat_system_prompt

    result = build_chat_system_prompt(
        bot_name="bot",
        is_private=False,
        personality_setting="  ",
        relation_context="",
        group_context="",
        recent_relations_context="",
        permission_status="",
        mute_tool_instruction="",
        reaction_tool_instruction="",
    )

    assert "【自定义设定与固定知识】" not in result.system_prompt
