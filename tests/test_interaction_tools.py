import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def history_db(tmp_path, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import interaction_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'history.sqlite3'}")
    async with engine.begin() as connection:
        await connection.run_sync(ChatHistory.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(interaction_tools, "get_session", sessions)
    yield sessions
    await engine.dispose()


async def seed(sessions, rows):
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    async with sessions() as session:
        for index, overrides in enumerate(rows):
            session.add(
                ChatHistory(
                    **{"msg_id": index + 1, "session_id": "group-1", "user_id": str(index + 1), "user_name": f"成员{index + 1}", "content_type": "text", "content": "默认测试消息正文", "media_id": None, "created_at": datetime(2026, 7, 9, 15, 50) + timedelta(seconds=index), **overrides},
                )
            )
        await session.commit()


def tools(*, request_id=None):
    from nonebot_plugin_ai_groupmate.agent.interaction_tools import create_interaction_history_tools

    return {tool.name: tool for tool in create_interaction_history_tools("group-1", request_id, bot_user_ids=("bot-id",))}


async def invoke(tool, **kwargs):
    return json.loads(await tool.ainvoke(kwargs))


async def test_discovers_interaction_without_evaluation_keyword_and_keeps_roles(history_db):
    await seed(
        history_db,
        [
            {"user_id": "1", "content": "我把袖子看成别的了"},
            {"user_id": "2", "content": "你这是想出幻觉了吧"},
            {"user_id": "1", "content": "你这是想出幻觉了吧"},
            {"user_id": "3", "content": "你这是想出幻觉了吧"},
            {"user_id": "bot-id", "content": "你这是想出幻觉了吧"},
            {"session_id": "other-group", "user_id": "9", "content": "你这是想出幻觉了吧"},
        ],
    )
    bundle = tools()
    result = await invoke(bundle["search_group_interactions"])
    event = result["data"]["events"][0]
    assert event["first_message"]["user_id"] == "2"
    assert event["participant_count"] == 3
    assert event["source_msg_ids"] == [2, 3, 4]
    context = await invoke(bundle["read_member_history"], anchor_msg_id=2)
    assert context["data"]["source_message"]["user_id"] == "2"
    assert context["data"]["source_message"]["content"] == "你这是想出幻觉了吧"
    assert context["data"]["messages"][0]["content"] == "我把袖子看成别的了"
    assert context["data"]["messages"][-1]["role"] == "bot"
    assert all(item["msg_id"] != 6 for item in context["data"]["messages"])
    assert (await invoke(bundle["read_member_history"], anchor_msg_id=6))["reason_code"] == "source_not_found"
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    async with history_db() as session:
        assert await session.scalar(select(func.count()).select_from(ChatHistory)) == 6


async def test_names_are_resolved_by_id_and_ambiguous_names_are_not_merged(history_db):
    await seed(
        history_db,
        [
            {"user_id": "1", "user_name": "旧昵称", "content": "这个文档改到第九版了"},
            {"user_id": "1", "user_name": "小王", "content": "新版最终修改版还是要改"},
            {"user_id": "2", "user_name": "小王", "content": "另一个人的消息不能归过去"},
            {"session_id": "other-group", "user_id": "3", "user_name": "旧昵称"},
        ],
    )
    bundle = tools()
    ambiguous = await invoke(bundle["read_member_history"], member="小王")
    assert ambiguous["reason_code"] == "ambiguous_member"
    result = await invoke(bundle["read_member_history"], member="旧昵称", keyword="版")
    assert [item["user_id"] for item in result["data"]["messages"]] == ["1", "1"]
    assert {item["user_name"] for item in result["data"]["messages"]} == {"旧昵称", "小王"}


async def test_date_windows_pagination_and_result_budget(history_db, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import interaction_tools

    await seed(history_db, [{"user_id": "1", "content": f"第{i}条" + "长正文" * 200} for i in range(20)])
    monkeypatch.setattr(interaction_tools, "RESULT_MAX_CHARS", 2200)
    bundle = tools()
    first = await invoke(bundle["read_member_history"], member="1", limit=12, end_date="2026-07-09")
    second = await invoke(bundle["read_member_history"], member="1", limit=12, before_msg_id=first["data"]["next_before_msg_id"])
    assert first["data"]["messages"]
    assert second["data"]["messages"]
    assert {item["msg_id"] for item in first["data"]["messages"]}.isdisjoint(item["msg_id"] for item in second["data"]["messages"])
    assert all(item["content_truncated"] for item in first["data"]["messages"])
    assert len(json.dumps(first, ensure_ascii=False)) <= 2200
    invalid = await invoke(bundle["search_group_interactions"], start_date="2026-7-9")
    assert invalid["reason_code"] == "invalid_date"
    empty = await invoke(bundle["read_member_history"], member="1", start_date="2026-07-10")
    assert empty["data"]["messages"] == []


async def test_reliable_echo_identity_excludes_other_messages_from_same_bot(history_db):
    await seed(
        history_db,
        [
            {"content_type": "bot", "content": "id: p1\n机器人回执一"},
            {"user_id": "adapter-bot", "content": "id: p1\n机器人回执一"},
            {"content_type": "bot", "content": "id: p2\n机器人回执二"},
            {"user_id": "adapter-bot", "content": "id: p2\n机器人回执二"},
            {"user_id": "adapter-bot", "content": "这句会被多个人跟读呢"},
            {"user_id": "1", "content": "这句会被多个人跟读呢"},
            {"user_id": "2", "content": "这句会被多个人跟读呢"},
        ],
    )
    result = await invoke(tools()["search_group_interactions"])
    assert result["data"]["candidate_events"] == 0


async def test_expired_request_does_not_read_database(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import interaction_tools

    async def inactive(*_):
        return False

    def no_session():
        pytest.fail("expired request accessed the database")

    monkeypatch.setattr(interaction_tools, "is_request_active", inactive)
    monkeypatch.setattr(interaction_tools, "get_session", no_session)
    for tool in tools(request_id="expired").values():
        assert (await invoke(tool))["status"] == "skipped"


def test_group_skill_exposes_interaction_guidance():
    from nonebot_plugin_ai_groupmate.agent import _build_builtin_agent_skills

    for private in (False, True):
        skills = _build_builtin_agent_skills(is_private=private, has_admin_permission=False, mute_tool_instruction="", meme_similar_enabled=False)
        skill = next(item for item in skills if item.name == "search_context_tools")
        assert isinstance(skill.prompt, str)
        assert ("search_group_interactions" in skill.prompt) == (not private)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("谁是群里的神人？", True),
        ("给这个群的神人排个前五，举真实事例。", True),
        ("给上周的聊天做个总结", True),
        ("为群友的经典发言排一下名", True),
        ("miyuki 你用Excel统计一下这个群的神人发言", True),
        ("查一下小王之前说过什么", True),
        ("回顾一下上周大家都聊了什么", True),
        ("群里最近在讨论什么", True),
        ("这个梗是怎么来的", True),
        ("谁最早提过这个梗", True),
        ("追溯一下这句话的出处", True),
        ("核对之前关于聚餐的约定", True),
        ("小王上次说什么时候面试", True),
        ("对比小王前后对这个游戏的看法", True),
        ("找小李上周关于面试的发言", True),
        ("把上周的聊天整理成时间线", True),
        ("展开这段讨论的前后文", True),
        ("我也是神人了", False),
        ("神人来了哈哈哈哈", False),
        ("给群友的礼物都准备好了", False),
        ("他刚才说“给这个群的神人排个前五”", False),
        ("不用给这个群的神人排个前五了", False),
        ("今天吃啥", False),
        ("不用查群里的神人了", False),
        ("他刚才说：查一下群里的神人", False),
        ("他刚才说“谁是群里的神人”", False),
        ("这个梗真好笑", False),
        ("小王之前说过他喜欢这个游戏", False),
        ("今天群里讨论得好热闹", False),
        ("他刚才说“回顾一下上周大家聊了什么”", False),
        ("别查之前的聊天了", False),
        ("我以前最喜欢这个游戏了", False),
    ],
)
def test_interaction_tools_entry_requires_current_user_query(message, expected):
    from nonebot_plugin_ai_groupmate.interaction_history import HistoryMessage, interaction_history_requested

    row = HistoryMessage(1, "group-1", "1", "小王", "text", message, datetime.now())
    assert interaction_history_requested([row], "1", bot_name="miyuki") == expected
    assert not interaction_history_requested([row], "2", bot_name="miyuki")


def test_short_followup_uses_only_same_user_recent_query():
    from nonebot_plugin_ai_groupmate.interaction_history import HistoryMessage, interaction_history_requested

    now = datetime.now()
    prior = HistoryMessage(1, "g", "1", "甲", "text", "谁是群里的神人", now - timedelta(minutes=1))
    followup = HistoryMessage(2, "g", "1", "甲", "text", "那你给个排名", now)
    assert interaction_history_requested([prior, followup], "1")
    assert not interaction_history_requested([followup], "1")
    old = HistoryMessage(1, "g", "1", "甲", "text", prior.content, now - timedelta(hours=3))
    assert not interaction_history_requested([old, followup], "1")


def test_general_history_followup_requires_recent_query_from_same_user():
    from nonebot_plugin_ai_groupmate.interaction_history import HistoryMessage, interaction_history_requested

    now = datetime.now()
    prior = HistoryMessage(1, "g", "1", "甲", "text", "核对之前关于聚餐的约定", now - timedelta(minutes=1))
    followup = HistoryMessage(2, "g", "1", "甲", "text", "那后来呢？", now)
    assert interaction_history_requested([prior, followup], "1")
    assert not interaction_history_requested([followup], "1")
    other_user = HistoryMessage(1, "g", "2", "乙", "text", prior.content, prior.created_at)
    assert not interaction_history_requested([other_user, followup], "1")


async def test_member_history_compares_views_without_a_repeat_event(history_db):
    from nonebot_plugin_ai_groupmate.interaction_history import HistoryMessage, interaction_history_requested

    await seed(
        history_db,
        [
            {"user_id": "1", "user_name": "小王", "content": "游戏A的养成太费时间，暂时不推荐。", "created_at": datetime(2026, 7, 1)},
            {"user_id": "2", "user_name": "小李", "content": "游戏A可以一起玩，我觉得不错。", "created_at": datetime(2026, 7, 2)},
            {"user_id": "1", "user_name": "小王的新昵称", "content": "游戏A更新后省事了，现在愿意推荐。", "created_at": datetime(2026, 7, 9)},
        ],
    )
    query = HistoryMessage(100, "group-1", "3", "丙", "text", "对比小王前后对游戏A的看法", datetime(2026, 7, 10))
    assert interaction_history_requested([query], "3")
    bundle = tools()
    result = await invoke(bundle["read_member_history"], member="小王", keyword="游戏A")
    evidence = result["data"]["messages"]
    assert [row["msg_id"] for row in evidence] == [3, 1]
    assert all(row["user_id"] == "1" for row in evidence)
    assert "现在愿意推荐" in evidence[0]["content"]
    assert "暂时不推荐" in evidence[1]["content"]
    interactions = await invoke(bundle["search_group_interactions"])
    assert interactions["data"]["candidate_events"] == 0


def test_disabled_entry_removes_interaction_skill_guidance():
    from nonebot_plugin_ai_groupmate.agent import _build_builtin_agent_skills

    skills = _build_builtin_agent_skills(is_private=False, has_admin_permission=False, mute_tool_instruction="", meme_similar_enabled=False, interaction_history_enabled=False)
    skill = next(item for item in skills if item.name == "search_context_tools")
    assert isinstance(skill.prompt, str)
    assert "search_group_interactions" not in skill.prompt
    assert "神人" not in skill.description


async def test_interaction_pages_and_changed_date_window_do_not_skip_sources(history_db, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import interaction_tools

    rows = []
    for day in range(3):
        for user in range(3):
            rows.append({"user_id": str(user + 1), "content": f"第{day}天的独立互动原话", "created_at": datetime(2026, 7, 9 + day, 12, user)})
    await seed(history_db, rows)
    monkeypatch.setattr(interaction_tools, "RESULT_MAX_CHARS", 2200)
    bundle = tools()
    offset = 0
    ids = []
    while offset is not None:
        result = await invoke(bundle["search_group_interactions"], offset=offset, limit=8)
        ids.extend(event["first_message"]["msg_id"] for event in result["data"]["events"])
        offset = result["data"]["next_offset"]
        assert len(json.dumps(result, ensure_ascii=False)) <= 2200
    assert ids == [1, 4, 7]
    dated = await invoke(bundle["search_group_interactions"], start_date="2026-07-11", end_date="2026-07-11")
    assert dated["data"]["candidate_events"] == 1
    assert dated["data"]["events"][0]["source_msg_ids"] == [7, 8, 9]


async def test_context_budget_preserves_anchor_and_scan_bound_is_visible(history_db, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import interaction_tools

    await seed(history_db, [{"content": f"第{i}条" + "长正文" * 150} for i in range(20)])
    monkeypatch.setattr(interaction_tools, "RESULT_MAX_CHARS", 1700)
    bundle = tools()
    context = await invoke(bundle["read_member_history"], anchor_msg_id=10)
    assert context["data"]["context_truncated"]
    assert any(item["msg_id"] == 10 for item in context["data"]["messages"])
    assert len(json.dumps(context, ensure_ascii=False)) <= 1700
    monkeypatch.setattr(interaction_tools, "SCAN_LIMIT", 5)
    scan = await invoke(bundle["search_group_interactions"])
    assert scan["data"]["scan_truncated"]
    assert scan["data"]["scanned_messages"] == 5
