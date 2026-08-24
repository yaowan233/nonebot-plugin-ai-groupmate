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


def test_group_prompt_encourages_proactive_memes_with_serious_topic_boundary():
    from nonebot_plugin_ai_groupmate.agent.prompts import build_chat_system_prompt

    result = build_chat_system_prompt(
        bot_name="bot",
        is_private=False,
        personality_setting="",
        relation_context="",
        group_context="",
        recent_relations_context="",
        permission_status="",
        mute_tool_instruction="",
        reaction_tool_instruction="",
    )

    assert "主动考虑调用 `search_meme_image`" in result.system_prompt
    assert "不必等群友明确索要" in result.system_prompt
    assert "敏感或沉重话题" in result.system_prompt
    assert "没有合适候选就不发" in result.system_prompt


@pytest.mark.parametrize("is_private", [False, True])
def test_prompt_routes_public_current_facts_directly_to_web_search(is_private):
    from nonebot_plugin_ai_groupmate.agent.prompts import build_chat_system_prompt

    result = build_chat_system_prompt(
        bot_name="bot",
        is_private=is_private,
        personality_setting="",
        relation_context="",
        group_context="",
        recent_relations_context="",
        permission_status="",
        mute_tool_instruction="",
        reaction_tool_instruction="",
    )

    assert "直接调用 `search_web`" in result.system_prompt
    assert "不要改用 `search_history_context`" in result.system_prompt
    assert "不要向用户反问可通过联网查到的事实" in result.system_prompt
    assert "保留用户点名的品牌、产品、人物、事件和时间条件" in result.system_prompt
