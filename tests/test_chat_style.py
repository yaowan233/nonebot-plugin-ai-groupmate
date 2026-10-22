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


def test_style_learns_conversation_instead_of_requests_addressed_to_the_bot():
    from nonebot_plugin_ai_groupmate.agent.chat_style import build_group_style_context

    history = [
        message(1, "a", "笑死了"), message(2, "b", "有点东西"), message(3, "c", "卡了"),
        message(4, "a", "miyuki 帮我画一个石像"),
        message(5, "b", "@miyuki 修改一下配色"),
        message(6, "c", "帮我生成一个表格"),
        message(7, "b", "修改一下，改成侧视图"),
    ]
    result = build_group_style_context(history, "group", bot_names=("miyuki",))
    assert [sample["text"] for sample in json.loads(result.splitlines()[-1])] == ["笑死了", "有点东西", "卡了"]


def test_style_without_enough_casual_samples_does_not_fill_with_tasks():
    from nonebot_plugin_ai_groupmate.agent.chat_style import build_group_style_context

    history = [message(1, "a", "miyuki 查一下天气"), message(2, "b", "帮我画一张图"), message(3, "c", "miyuki 好厉害")]
    assert build_group_style_context(history, "group", bot_names=("miyuki",)) == ""


async def test_style_uses_older_human_chat_when_reply_window_contains_only_images(monkeypatch):
    from datetime import timedelta
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from nonebot_plugin_orm import get_session
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistory, ChatHistorySchema

    group = "style-window-" + uuid4().hex
    now = datetime.now()
    casual = ["这也能接上（", "我还真没想到（", "有点离谱（", "我超，还能这样（", "这回真绷不住了（", "怎么还能这么玩（"]
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "build_chat_graph", lambda *args, **kwargs: object())
    async with get_session() as session:
        for index, body in enumerate(casual):
            session.add(ChatHistory(session_id=group, user_id=f"human-{index % 3}", user_name="User", content_type="text", content=body, created_at=now - timedelta(days=2, minutes=index)))
        images = [ChatHistory(session_id=group, user_id="image-sender", user_name="User", content_type="image", content="picture.png", created_at=now - timedelta(seconds=index)) for index in range(20)]
        session.add_all(images)
        await session.flush()
        history = [ChatHistorySchema.model_validate(row) for row in images]
        await session.commit()
        _, _, dynamic = await agent.create_chat_graph(session, group, None, "human-0", "User", history=history)
        assert "本群" in dynamic
        assert any(body in dynamic for body in casual), "reply-window images must not erase the group's older speech samples"


def test_style_statistics_require_multiple_people_and_dates():
    from datetime import timedelta
    from nonebot_plugin_ai_groupmate.agent.chat_style import build_group_style_context

    history = [message(i, f"human-{i % 4}", f"这操作我还真没想到{i}（", created_at=datetime(2026, 10, 22) - timedelta(days=i // 12)) for i in range(36)]
    context = build_group_style_context(history, "group")
    assert "36 条、4 位真人、3 个日期" in context
    assert "多位成员跨日出现的收尾形式" in context
    assert len(json.loads(context.splitlines()[-1])) == 12
    one_person = [message(i, "one", f"我又卡了{i}（", created_at=datetime(2026, 10, 22) - timedelta(days=i // 12)) for i in range(36)]
    assert build_group_style_context(one_person, "group") == ""


async def test_style_loader_keeps_groups_separate_and_expires_old_chat():
    from datetime import timedelta
    from uuid import uuid4
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_ai_groupmate.agent.chat_style import load_group_style_context
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    group = "style-isolation-" + uuid4().hex
    now = datetime.now()
    async with get_session() as session:
        for index, body in enumerate(("这下真绷不住了（", "我还以为能过呢（", "就差那么一点（")):
            session.add(ChatHistory(session_id=group, user_id=str(index), user_name="User", content_type="text", content=body, created_at=now - timedelta(days=2)))
            session.add(ChatHistory(session_id=group + "other", user_id=str(index), user_name="User", content_type="text", content=f"绝对不该串群{index}", created_at=now - timedelta(days=1)))
            session.add(ChatHistory(session_id=group, user_id=str(index), user_name="User", content_type="text", content=f"绝对不该保留八天前{index}", created_at=now - timedelta(days=8)))
        await session.commit()
        context = await load_group_style_context(session, group, now=now)
        assert "就差那么一点（" in context
        assert "绝对不该" not in context
