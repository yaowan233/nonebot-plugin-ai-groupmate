import pytest


@pytest.mark.parametrize("meme_similar_enabled", [True, False])
def test_meme_prompts_only_reference_bound_similarity_tool(
    meme_similar_enabled: bool,
):
    from nonebot_plugin_ai_groupmate.agent import (
        _build_builtin_agent_skills,
        _scheduled_meme_tool_instruction,
    )

    skills = _build_builtin_agent_skills(
        is_private=True,
        has_admin_permission=False,
        mute_tool_instruction="",
        meme_similar_enabled=meme_similar_enabled,
    )
    meme_skill = next(skill for skill in skills if skill.name == "meme_tools")
    assert isinstance(meme_skill.prompt, str)

    scheduled_instruction = _scheduled_meme_tool_instruction(
        meme_similar_enabled=meme_similar_enabled,
    )
    if meme_similar_enabled:
        assert "search_similar_meme_by_id" in meme_skill.prompt
        assert "search_similar_meme_by_id" in scheduled_instruction
    else:
        assert "search_similar_meme_by_id" not in meme_skill.prompt
        assert "search_similar_meme_by_id" not in scheduled_instruction
        assert "纯文本向量模式" in meme_skill.prompt
        assert "`search_meme_image`" in scheduled_instruction


def test_send_meme_description_does_not_assume_similarity_tool(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.meme_tools import create_send_meme_tool

    send_tool = create_send_meme_tool(
        object(),
        "group-1",
        pic_dir=tmp_path,
        bot_name="小助手",
    )

    assert "search_similar_meme_by_id" not in send_tool.description
