import datetime
from types import SimpleNamespace

import pytest


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.value


class _Session:
    def __init__(self, value):
        self.value = value

    async def execute(self, _statement):
        return _ScalarResult(self.value)


@pytest.mark.asyncio
async def test_extreme_negative_relation_keeps_normal_requests_fair():
    from nonebot_plugin_ai_groupmate.agent.context import get_user_relation_context

    relation = SimpleNamespace(
        user_name="tester",
        favorability=-100,
        tags=["讨厌", "曾经争吵"],
        get_status_desc=lambda: "明显疏远",
    )

    context = await get_user_relation_context(_Session(relation), "10001", "tester")

    assert "对对方明确提出的正常请求仍要礼貌、公平并尽力完整回应" in context
    assert "当前消息的实际内容优先于历史好感度和标签" in context
    assert "不得因为负好感度或负面标签讽刺、羞辱、敌视、故意敷衍、无视对方" in context
    assert "可能片面或过时，不得用于决定回复态度" in context
    assert "死敌" not in context
    assert "语气带刺" not in context
    assert "道歉和实质性补偿" not in context


@pytest.mark.asyncio
async def test_recent_relation_summary_does_not_expose_scores_or_negative_tags():
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema
    from nonebot_plugin_ai_groupmate.agent.context import get_recent_relations_context

    relation = SimpleNamespace(
        user_id="10001",
        favorability=-50,
        tags=["骗子", "讨厌"],
        get_status_desc=lambda: "保持距离",
    )
    history = [
        ChatHistorySchema(
            msg_id=1,
            session_id="group-1",
            user_id="10001",
            content_type="text",
            content="hello",
            created_at=datetime.datetime(2026, 7, 29, 12, 0),
            user_name="tester",
        )
    ]

    context = await get_recent_relations_context(_Session([relation]), history)

    assert "- tester: 保持距离" in context
    assert "当前发言优先" in context
    assert "-50" not in context
    assert "骗子" not in context
    assert "讨厌" not in context


@pytest.mark.parametrize("is_private", [False, True])
def test_chat_prompt_limits_relationship_effects(is_private):
    from nonebot_plugin_ai_groupmate.agent.prompts import build_chat_system_prompt

    result = build_chat_system_prompt(
        bot_name="bot",
        is_private=is_private,
        personality_setting="",
        relation_context="历史关系上下文",
        group_context="",
        recent_relations_context="",
        permission_status="",
        mute_tool_instruction="",
        reaction_tool_instruction="",
    )

    assert "人际关系和印象标签只影响语气亲疏、主动性和玩笑尺度" in result.system_prompt
    assert "不得因此降低正常请求的回答质量" in result.system_prompt


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-100, "明显疏远"),
        (-50, "保持距离"),
        (-20, "稍显克制"),
        (0, "陌生/普通"),
        (50, "亲近/好友"),
        (100, "最亲近的人"),
    ],
)
def test_relation_status_descriptions_are_non_hostile(score, expected):
    from nonebot_plugin_ai_groupmate.model import UserRelation

    relation = UserRelation(
        user_id="10001",
        user_name="tester",
        favorability=score,
        tags=[],
    )

    assert relation.get_status_desc() == expected
