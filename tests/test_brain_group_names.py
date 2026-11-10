from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_resolver_scopes_names_to_bot_and_allowed_groups_and_caches():
    from nonebot_plugin_ai_groupmate.brain.group_names import GroupNameResolver

    api = AsyncMock(return_value=[{"group_id": 100, "group_name": "甲群"},
                                  {"group_id": 200, "group_name": "乙群"},
                                  {"group_id": 300, "group_name": "无关联群"}])
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=api)
    resolver = GroupNameResolver()
    assert await resolver.resolve(bot, "OneBot V11", ("100", "200"), {"200": "手动乙群"}) == {"100": "甲群", "200": "手动乙群"}
    assert await resolver.resolve(bot, "OneBot V11", ("100",), {}) == {"100": "甲群"}
    api.assert_awaited_once_with("get_group_list")
    assert await resolver.resolve(bot, "OneBot V11", ("300",), {}) == {"300": "无关联群"}
    assert api.await_count == 2

    other_api = AsyncMock(return_value=[{"group_id": 100, "group_name": "另一机器人群名"}])
    other_bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=other_api)
    assert await resolver.resolve(other_bot, "OneBot V11", ("100",), {}) == {"100": "另一机器人群名"}
    other_api.assert_awaited_once_with("get_group_list")


@pytest.mark.asyncio
async def test_resolver_falls_back_to_manual_when_api_fails_or_adapter_is_unsupported():
    from nonebot_plugin_ai_groupmate.brain.group_names import GroupNameResolver

    api = AsyncMock(side_effect=TimeoutError())
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=api)
    resolver = GroupNameResolver()
    assert await resolver.resolve(bot, "OneBot V11", ("200", "201"), {"200": "乙群", "300": "其他群"}) == {"200": "乙群"}
    assert await resolver.resolve(bot, "OneBot V11", ("200", "201"), {"200": "乙群"}) == {"200": "乙群"}
    api.assert_awaited_once()
    assert await resolver.resolve(bot, "other", ("200",), {"200": "乙群"}) == {"200": "乙群"}
    api.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_passes_actual_bot_group_names_only_when_planning(monkeypatch):
    import json
    from uuid import uuid4
    from datetime import datetime, timezone, timedelta

    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.brain.repository import TaskInput, submit

    group_a, group_b, bot_id = [uuid4().hex for _ in range(3)]
    api = AsyncMock(return_value=[{"group_id": group_b, "group_name": "测试乙群"},
                                  {"group_id": "unrelated", "group_name": "其他群"}])
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "OneBot V11"), call_api=api)
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"kind": "ask", "target_scene": group_b, "text": "几点出发？"}))))
    monkeypatch.setattr(plugin.plugin_config, "cross_group_spaces", {"friends": [group_a, group_b]})
    monkeypatch.setattr(plugin.plugin_config, "cross_group_scene_names", {})
    monkeypatch.setattr(plugin, "get_bots", lambda: {bot_id: bot})
    monkeypatch.setattr(agent, "get_chat_model", lambda: model)
    monkeypatch.setattr(plugin, "_send_cross_group_message", AsyncMock(return_value="receipt"))
    await plugin._poll_cross_group_tasks()
    api.assert_not_awaited()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with get_session() as db:
        await submit(db, TaskInput("brain", "OneBot V11", group_a, "message", "alice", "去测试乙群问几点",
                                   now + timedelta(minutes=5), transport_bot_id=bot_id), now=now)
        await db.commit()
    await plugin._poll_cross_group_tasks()
    api.assert_awaited_once_with("get_group_list")
    payload = json.loads(model.ainvoke.await_args.args[0][1].content)
    assert payload["target_scenes"] == [{"id": group_b, "name": "测试乙群"}]
