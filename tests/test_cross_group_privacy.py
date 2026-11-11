import json
from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime
from unittest.mock import AsyncMock

import pytest


def test_hide_other_group_numbers_only():
    from nonebot_plugin_ai_groupmate.cross_group_privacy import hide_other_group_ids

    spaces = {"friends": ["371306167", "727030439"], "other": ["758450633"]}
    text = "本群 371306167，隔壁 727030439，另一群 758450633，数字 17270304390"
    assert hide_other_group_ids(text, "371306167", spaces) == (
        "本群 371306167，隔壁 [群号已隐藏]，另一群 [群号已隐藏]，数字 17270304390"
    )


async def test_text_reply_hides_linked_group_number(monkeypatch):
    import json
    from types import SimpleNamespace

    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(str(message))
        return SimpleNamespace(msg_ids=[{"message_id": "group-number-test"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    async with get_session() as db:
        reply = create_reply_tool(db, "371306167", bot_name="miyuki", parse_msg_meta=parse_message_metadata,
                                  group_members=[], cross_group_spaces={"friends": ["371306167", "727030439"]})
        result = json.loads(await reply.ainvoke({"content": "隔壁群是 727030439", "next_step": "end"}))
    assert result["ok"]
    assert "727030439" not in sent[0]
    assert "[群号已隐藏]" in sent[0]


@pytest.mark.parametrize("text", [
    "电话 13800138000", "电话 １３８００１３８０００", "手机 138 0013 8000",
    "联系 abc@example.com", "身份证 11010519491231002X", "我的家人住院了", "小明确诊了",
    "我住在上海市某某路某号", "微信号：private123",
])
def test_personal_details_are_detected(text):
    from nonebot_plugin_ai_groupmate.cross_group_privacy import contains_personal_data

    assert contains_personal_data(text)


@pytest.mark.parametrize("text", ["今天 osu 比赛八点开始", "群号 787856471", "大家晚上好", "怎样保护手机号不被泄露"])
def test_ordinary_topics_are_not_hidden(text):
    from nonebot_plugin_ai_groupmate.cross_group_privacy import contains_personal_data

    assert not contains_personal_data(text)


def test_only_explicit_self_disclosure_is_allowed():
    from nonebot_plugin_ai_groupmate.cross_group_privacy import asks_for_personal_data, explicitly_shares_own_data

    assert explicitly_shares_own_data("把我的手机号13800138000发给隔壁群", "手机号13800138000")
    assert not explicitly_shares_own_data("把小明的手机号13800138000发给隔壁群", "手机号13800138000")
    assert not explicitly_shares_own_data("把朋友说的“我住院了”转发到隔壁群", "我住院了")
    assert not explicitly_shares_own_data("把我的手机号发给隔壁群", "手机号13800138000")
    assert not explicitly_shares_own_data("把我的电话13800138000和小王的电话13900139000发给隔壁群", "13800138000和13900139000")
    assert not explicitly_shares_own_data("把我的电话13800138000和小王的13900139000发给隔壁群", "13800138000和13900139000")
    assert not explicitly_shares_own_data("把我的电话13800138000和小王的13900139000发给隔壁群", "13900139000")
    assert not explicitly_shares_own_data("转告隔壁我住院了和小王也住院了", "我住院了和小王也住院了")
    assert asks_for_personal_data("去隔壁问小明的手机号是多少")
    assert not asks_for_personal_data("去隔壁问怎样保护手机号不被泄露")


def test_internal_topic_does_not_include_specific_times_or_numbers():
    from nonebot_plugin_ai_groupmate.cross_group_privacy import is_general_topic

    assert is_general_topic("游戏")
    assert is_general_topic("旅行")
    assert not is_general_topic("大家在聊郊游")
    assert not is_general_topic("比赛晚上八点开始")
    assert not is_general_topic("比赛时间是20:00")


def test_linked_history_hides_sensitive_evidence_without_changing_local_history():
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    messages = [
        {"msg_id": 10, "user_name": "Alice", "user_id": "123", "content": "我的电话是13800138000"},
        {"msg_id": 9, "user_name": "Bob", "user_id": "456", "content": "今晚比赛八点开始"},
    ]
    linked = json.loads(tool_success("history_read", "读取完成", data={"scope": "linked_group", "messages": messages, "next_before_msg_id": 9}))
    assert linked["data"]["privacy_filtered"] is True
    assert "13800138000" not in json.dumps(linked, ensure_ascii=False)
    assert linked["data"]["messages"][0]["user_id"] == ""
    assert linked["data"]["messages"][1]["content"] == "今晚比赛八点开始"
    assert linked["data"]["next_before_msg_id"] == 9
    local = json.loads(tool_success("history_read", "读取完成", data={"scope": "current_group", "messages": messages}))
    assert "13800138000" in json.dumps(local, ensure_ascii=False)
    rag = json.loads(tool_success("history_found", "读取完成", data={"scope": "linked_group", "context": "Alice 的电话13800138000"}))
    assert "13800138000" not in json.dumps(rag, ensure_ascii=False)


@pytest.mark.asyncio
async def test_user_facing_tools_do_not_fetch_another_groups_history(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import common_tools
    from nonebot_plugin_ai_groupmate.agent.interaction_tools import create_interaction_history_tools

    spaces = lambda: {"friends": ["group-1", "group-2"]}
    reachable = AsyncMock(return_value={"group-1", "group-2"})
    search = AsyncMock(return_value="本群普通消息")
    monkeypatch.setattr(common_tools.DB, "search_chat", search)
    rag = common_tools.create_scoped_history_search_tool(
        "group-1", None, platform="qq", configured_spaces=spaces, reachable_groups=reachable,
    )
    denied = json.loads(await rag.ainvoke({"query": "最近聊什么", "target_group_id": "group-2"}))
    assert denied["reason_code"] == "cross_group_history_private"
    search.assert_not_awaited()
    local = json.loads(await rag.ainvoke({"query": "最近聊什么"}))
    assert local["status"] == "succeeded"
    search.assert_awaited_once()

    tools = create_interaction_history_tools("group-1", None, platform="qq", configured_spaces=spaces, reachable_groups=reachable)
    for tool in tools:
        args = {"target_group_id": "group-2"}
        if tool.name == "read_member_history":
            args["member"] = "someone"
        result = json.loads(await tool.ainvoke(args))
        assert result["reason_code"] == "cross_group_history_private"


@pytest.mark.asyncio
async def test_user_facing_queries_do_not_list_another_groups_members():
    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock())
    query = create_platform_query_tool(bot, "101", None, lambda: {"friends": ["101", "202"]})
    for args in ({"operation": "list_members", "group_id": "202"},
                 {"operation": "member_info", "group_id": "202", "user_id": "123"}):
        result = json.loads(await query.ainvoke(args))
        assert result["reason_code"] == "cross_group_members_private"
    bot.call_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_cross_group_relay_blocks_other_peoples_private_data_but_allows_explicit_self_share():
    from sqlalchemy import delete, select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainTask, BrainAction
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool

    source, target, bot = (uuid4().hex for _ in range(3))
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"send":true,"reason":""}')))

    def tool(request, message_id):
        return create_coordination_tool(
            config=SimpleNamespace(cross_group_spaces={"friends": [source, target]}), platform="qq", bot_id=bot,
            scene_id=source, user_id="alice", user_name="Alice", message_id=message_id, request_id=None,
            request_text=request, reachable_groups=AsyncMock(return_value={source, target}), review_model=model,
        )

    args = {"reason": "传话", "operation": "send", "target_group_id": target, "message": "手机号13800138000"}
    blocked = json.loads(await tool("把小明的手机号13800138000发给隔壁群", "other").ainvoke(args))
    assert blocked["reason_code"] == "privacy_protected"
    records = json.loads(await tool("去隔壁问他们最近聊了什么", "records").ainvoke({"reason": "询问记录", "operation": "ask"}))
    assert records["reason_code"] == "cross_group_history_private"
    model.ainvoke.assert_not_awaited()
    approved = json.loads(await tool("把我的手机号13800138000发给隔壁群", "self").ainvoke(args))
    assert approved["reason_code"] == "coordination_queued"
    model.ainvoke.assert_awaited_once()
    async with get_session() as db:
        ids = select(BrainTask.task_id).where(BrainTask.source_scene == source)
        await db.execute(delete(BrainAction).where(BrainAction.task_id.in_(ids)))
        await db.execute(delete(BrainTask).where(BrainTask.source_scene == source))
        await db.commit()


@pytest.mark.asyncio
async def test_relay_review_never_returns_target_groups_private_details():
    from sqlalchemy import delete
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, HistoryMessageIndex
    from nonebot_plugin_ai_groupmate.brain.tools import create_coordination_tool

    source, target, bot = (uuid4().hex for _ in range(3))
    async with get_session() as db:
        db.add(ChatHistory(session_id=target, user_id="bob", user_name="Bob", content_type="text",
                           content="id: target\n我家人的电话是13800138000"))
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content='{"send":false,"reason":"Bob 的家人住院了，电话是13800138000。"}')))
    tool = create_coordination_tool(
        config=SimpleNamespace(cross_group_spaces={"friends": [source, target]}), platform="qq", bot_id=bot,
        scene_id=source, user_id="alice", user_name="Alice", message_id="source", request_id=None,
        request_text="去隔壁发个问候", reachable_groups=AsyncMock(return_value={source, target}), review_model=model,
    )
    try:
        result = json.loads(await tool.ainvoke({"reason": "问候", "operation": "send", "target_group_id": target, "message": "大家好"}))
        assert result["reason_code"] == "relay_declined"
        assert "13800138000" not in json.dumps(result, ensure_ascii=False)
        assert "住院" not in json.dumps(result, ensure_ascii=False)
        context = json.loads(model.ainvoke.await_args.args[0][-1].content)
        assert "13800138000" not in json.dumps(context, ensure_ascii=False)
        assert context["目标群近期聊天"][0]["sender"] == "[已隐藏]"
    finally:
        async with get_session() as db:
            await db.execute(delete(HistoryMessageIndex).where(HistoryMessageIndex.session_id == target))
            await db.execute(delete(ChatHistory).where(ChatHistory.session_id == target))
            await db.commit()


@pytest.mark.asyncio
async def test_observation_never_exposes_a_safe_summary_backed_by_private_source():
    from sqlalchemy import select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainSceneObservation, BrainObservationMessage
    from nonebot_plugin_ai_groupmate.brain.observations import record_message, relevant_context

    now = datetime(2026, 11, 11, 12)
    source, target, bot = (uuid4().hex for _ in range(3))
    async with get_session() as db:
        await record_message(db, platform="qq", bot_id=bot, scene_id=source, message_id="private", user_id="alice",
                             text="周末郊游联系我的电话13800138000", now=now)
        await db.flush()
        row = await db.scalar(select(BrainObservationMessage).where(BrainObservationMessage.bot_id == bot))
        assert row is not None
        db.add(BrainSceneObservation(platform="qq", bot_id=bot, scene_id=source, last_message_id=row.id,
                                    next_due=now, active=True,
                                    digest=json.dumps([{"fact": "旅行", "source_ids": [row.id], "topic_only": True}], ensure_ascii=False),
                                    source_ids=[row.id], updated_at=now))
        await db.commit()
        result = await relevant_context(db, spaces={"friends": [source, target]}, platform="qq", bot_id=bot,
                                        scene_id=target, query="周末郊游", now=now)
    assert result == ""
