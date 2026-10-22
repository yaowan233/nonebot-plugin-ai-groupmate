import json
from datetime import datetime


def message(number, user, text, **overrides):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    values = {
        "msg_id": number, "session_id": "group", "user_id": user, "user_name": user,
        "content_type": "text", "content": text, "created_at": datetime(2026, 10, 22),
    }
    values.update(overrides)
    return ChatHistorySchema(**values)


def test_style_samples_exclude_bots_other_groups_and_transport_headers():
    from nonebot_plugin_ai_groupmate.agent.chat_style import build_group_style_context

    history = [
        message(1, "a", "id: 1\n回复id: 99\n笑死了"),
        message(2, "b", "真的假的"),
        message(3, "a", "有点东西"),
        message(4, "self", "很高兴为您服务", content_type="bot"),
        message(5, "self", "请问还有什么需要"),
        message(6, "connected", "另一个bot"),
        message(7, "c", "别的群", session_id="other"),
        message(8, "c", "图片说明", content_type="image"),
        message(9, "c", "https://example.com"),
    ]
    result = build_group_style_context(history, "group", ("connected",))
    samples = json.loads(result.splitlines()[-1])
    assert [s["text"] for s in samples] == ["笑死了", "真的假的", "有点东西"]
    assert samples[0]["speaker"] == samples[2]["speaker"]


def test_style_samples_bound_dominant_speakers_and_deduplicate():
    from nonebot_plugin_ai_groupmate.agent.chat_style import build_group_style_context

    history = [message(i, str(i % 5), f"说点啥{i}") for i in range(100)]
    history += [message(101, "4", "说点啥99")]
    samples = json.loads(build_group_style_context(history, "group").splitlines()[-1])
    assert len(samples) == 12
    assert len({s["text"] for s in samples}) == 12
    assert all(sum(s["speaker"] == who["speaker"] for s in samples) <= 3 for who in samples)


def test_style_does_not_invent_group_consensus_from_one_speaker():
    from nonebot_plugin_ai_groupmate.agent.chat_style import build_group_style_context

    assert build_group_style_context([], "group") == ""
    assert build_group_style_context(
        [message(i, "a", f"话术{i}") for i in range(10)], "group"
    ) == ""


def test_style_context_is_dynamic_and_not_injected_into_private_chat():
    from nonebot_plugin_ai_groupmate.agent.prompts import build_chat_system_prompt

    kwargs = {
        "bot_name": "bot", "personality_setting": "", "relation_context": "",
        "group_context": "", "recent_relations_context": "", "permission_status": "",
        "mute_tool_instruction": "", "reaction_tool_instruction": "",
        "group_style_context": "unique group style samples",
    }
    group = build_chat_system_prompt(is_private=False, **kwargs)
    private = build_chat_system_prompt(is_private=True, **kwargs)
    assert kwargs["group_style_context"] in group.system_prompt
    assert kwargs["group_style_context"] in group.dynamic_context_parts
    assert kwargs["group_style_context"] not in private.system_prompt
