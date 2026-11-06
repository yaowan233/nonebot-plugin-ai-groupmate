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
    assert "不用非等群友管你要" in result.system_prompt
    assert "敏感或者沉重的话题，必须认真对待" in result.system_prompt
    assert "要是没搜到合适的就算了" in result.system_prompt
    assert "别拿表情包去替代必须用文字说清楚的内容" in result.system_prompt


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

    search_rule = next(line for line in result.system_prompt.splitlines() if "search_web" in line and "search_history_context" in line)
    assert "直接调 `search_web`" in search_rule
    assert "别错用成 `search_history_context`" in search_rule
    assert "反问给" in search_rule
    assert ("网上能直接查到的事" if is_private else "联网就能查到的事实") in search_rule
    assert "保留好用户" in search_rule
    for condition in ("品牌", "产品", "人物", "事件", "时间"):
        assert condition in search_rule
