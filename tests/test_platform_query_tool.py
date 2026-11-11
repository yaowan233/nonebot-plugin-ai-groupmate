import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_onebot_group_info_and_linked_group_list():
    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    async def call(api, **kwargs):
        if api == "get_group_info":
            return {"group_id": kwargs["group_id"], "group_name": "当前测试群", "member_count": 15, "extra": "ignored"}
        if api == "get_group_list":
            return [{"group_id": 101, "group_name": "当前测试群"}, {"group_id": 202, "group_name": "联动测试群"},
                    {"group_id": 303, "group_name": "未联动群"}]
        raise AssertionError(api)

    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock(side_effect=call))
    tool = create_platform_query_tool(bot, "101", None, lambda: {"friends": ["101", "202"]})
    result = json.loads(await tool.ainvoke({"operation": "group_info"}))
    assert result["data"]["group"]["group_name"] == "当前测试群"
    assert "extra" not in result["data"]["group"]
    bot.call_api.assert_awaited_once_with("get_group_info", group_id=101)
    result = json.loads(await tool.ainvoke({"operation": "list_groups"}))
    assert {x["group_id"] for x in result["data"]["groups"]} == {"101", "202"}
    result = json.loads(await tool.ainvoke({"operation": "group_info", "group_id": "303"}))
    assert not result["ok"]
    assert result["reason_code"] == "group_not_allowed"
    assert bot.call_api.await_count == 2


@pytest.mark.asyncio
async def test_reachable_groups_are_the_intersection_of_membership_and_config():
    from nonebot_plugin_ai_groupmate.agent.platform_queries import reachable_group_ids

    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock(return_value=[
        {"group_id": 101, "group_name": "来源"}, {"group_id": 303, "group_name": "未授权"},
    ]))
    assert await reachable_group_ids(bot, {"101", "202"}) == {"101"}
    bot.call_api.assert_awaited_once_with("get_group_list")


@pytest.mark.asyncio
async def test_members_query_and_scope_change():
    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    spaces = {"friends": ["101", "202"]}
    async def call(api, **kwargs):
        if api == "get_group_member_info":
            return {"user_id": kwargs["user_id"], "nickname": "小猫", "card": "猫猫", "role": "member", "ignored": 1}
        return [{"user_id": 1, "nickname": "小猫"}, {"user_id": 2, "nickname": "小狗"}]
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock(side_effect=call))
    tool = create_platform_query_tool(bot, "101", None, lambda: spaces, allow_cross_group_members=True)
    result = json.loads(await tool.ainvoke({"operation": "member_info", "group_id": "202", "user_id": "123"}))
    assert result["data"]["member"]["nickname"] == "小猫"
    assert "ignored" not in result["data"]["member"]
    result = json.loads(await tool.ainvoke({"operation": "list_members", "keyword": "猫"}))
    assert [x["nickname"] for x in result["data"]["members"]] == ["小猫"]
    spaces.clear()
    result = json.loads(await tool.ainvoke({"operation": "member_info", "group_id": "202", "user_id": "123"}))
    assert result["reason_code"] == "group_not_allowed"
    assert bot.call_api.await_count == 2


@pytest.mark.asyncio
async def test_unknown_operation_invalid_user_and_failed_adapter_do_not_leak_errors():
    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock(side_effect=RuntimeError("secret")))
    tool = create_platform_query_tool(bot, "101", None, lambda: {})
    result = json.loads(await tool.ainvoke({"operation": "member_info", "user_id": "wrong"}))
    assert result["reason_code"] == "invalid_user_id"
    result = json.loads(await tool.ainvoke({"operation": "group_info"}))
    assert result["reason_code"] == "platform_query_failed"
    assert "secret" not in json.dumps(result)
    result = json.loads(await tool.ainvoke({"operation": "send_group_msg"}))
    assert result["reason_code"] == "unsupported_query"
    assert bot.call_api.await_count == 1


@pytest.mark.asyncio
async def test_other_adapter_can_implement_same_operations_with_non_numeric_ids(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent.platform_queries import QUERY_PROVIDERS, supports_platform_queries, create_platform_query_tool

    class OtherProvider:
        def __init__(self, bot):
            self.bot = bot

        async def query(self, operation, **kwargs):
            return {"group": {"group_id": kwargs["group_id"], "group_name": "其他平台群"}}

    monkeypatch.setitem(QUERY_PROVIDERS, "Other Adapter", OtherProvider)
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "Other Adapter"))
    assert supports_platform_queries(bot)
    tool = create_platform_query_tool(bot, "room:abc", None, lambda: {})
    result = json.loads(await tool.ainvoke({"operation": "group_info"}))
    assert result["data"]["group"]["group_id"] == "room:abc"


@pytest.mark.asyncio
async def test_group_scope_is_rechecked_after_network_call():
    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    spaces = {"friends": ["101", "202"]}

    async def call(api, **kwargs):
        spaces.clear()
        return {"group_id": 202, "group_name": "不再联动的群"}

    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock(side_effect=call))
    tool = create_platform_query_tool(bot, "101", None, lambda: spaces)
    result = json.loads(await tool.ainvoke({"operation": "group_info", "group_id": "202"}))
    assert result["reason_code"] == "group_not_allowed"
    assert "不再联动的群" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_agent_exposes_query_tool_for_onebot_group_only(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    captured = []
    monkeypatch.setattr(agent.plugin_config, "agent_file_tools_enabled", False)
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context", "relevant_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "load_group_style_context", AsyncMock(return_value=""))

    def build(_model, _tools, _prompt, **kwargs):
        captured.extend(item.name for item in kwargs["base_tools"])
        return object()

    monkeypatch.setattr(agent, "build_chat_graph", build)
    bot = SimpleNamespace(self_id="123", adapter=SimpleNamespace(get_name=lambda: "OneBot V11"))
    event = SimpleNamespace(message_id="source", is_tome=lambda: True, get_plaintext=lambda: "这群叫什么")
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), "101", None, "alice", "Alice",
                                  history=[], bot_id="123", bot=bot, event=event, group_members=[])  # pyright: ignore[reportArgumentType]
    assert "query_platform" in captured
    assert "get_group_members" not in captured
    captured.clear()
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), "101", None, "alice", "Alice",
                                  history=[], bot_id="123", bot=bot, event=event, group_members=[], is_private=True)  # pyright: ignore[reportArgumentType]
    assert "query_platform" not in captured


@pytest.mark.asyncio
async def test_unified_member_query_preserves_random_selection_and_bot_exclusion(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import member_tools
    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    members = [SimpleNamespace(id=uid, name="群友", nick=None, user=None) for uid in ["111", "222", "222", "333", "444"]]
    populations = []

    def sample(population, count):
        populations.append(population)
        return [population[-1]]

    monkeypatch.setattr(member_tools.random, "sample", sample)
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "Other Adapter"))
    tool = create_platform_query_tool(bot, "group-a", None, lambda: {}, group_members=members, bot_user_ids=["444"])
    result = json.loads(await tool.ainvoke({"operation": "list_members", "random_selection": True,
                                            "limit": 1, "exclude_user_ids": ["111"]}))
    assert [item["user_id"] for item in populations[0]] == ["222", "333"]
    assert result["data"]["members"][0]["user_id"] == "333"
    assert result["data"]["matched_count"] == 2


@pytest.mark.asyncio
async def test_unified_member_query_uses_generic_interface_and_reuses_snapshot():
    from nonebot_plugin_uninfo import SceneType

    from nonebot_plugin_ai_groupmate.agent.platform_queries import create_platform_query_tool

    members = [SimpleNamespace(id="member:123", name="群名片", nick=None, user=SimpleNamespace(name="Nickname", nick=None))]
    interface = SimpleNamespace(get_members=AsyncMock(return_value=members))
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "Other Adapter"))
    tool = create_platform_query_tool(bot, "room:abc", None, lambda: {}, interface=interface)
    for query in ("nickname", "群名", "@member:123"):
        result = json.loads(await tool.ainvoke({"operation": "list_members", "keyword": query}))
        assert result["data"]["members"] == [{"user_id": "member:123", "name": "群名片"}]
    interface.get_members.assert_awaited_once_with(SceneType.GROUP, "room:abc")


@pytest.mark.asyncio
async def test_onebot_direct_list_randomly_excludes_bots_and_duplicates(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import platform_queries

    rows = [{"user_id": 111, "nickname": "一"}, {"user_id": 222, "nickname": "二"},
            {"user_id": 222, "nickname": "二"}, {"user_id": 333, "nickname": "三"}, {"user_id": 444, "nickname": "机器人"}]
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=AsyncMock(return_value=rows))
    tool = platform_queries.create_platform_query_tool(bot, "101", None, lambda: {}, bot_user_ids=["444"])
    populations = []

    def sample(population, count):
        populations.append(population)
        return [population[-1]]

    monkeypatch.setattr(platform_queries.random, "sample", sample)
    result = json.loads(await tool.ainvoke({"operation": "list_members", "random_selection": True,
                                            "limit": 1, "exclude_user_ids": ["111"]}))
    assert [item["user_id"] for item in populations[0]] == ["222", "333"]
    assert result["data"]["members"][0]["user_id"] == "333"
    assert result["data"]["matched_count"] == 2
