import json
from uuid import uuid4
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["就你了 @123456", "@123456 就你了", "@123456，轮到你了", "@群名片！"])
async def test_member_id_from_directory_reaches_native_mention(monkeypatch, content):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import At, UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    sent = []

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "sent-mention"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    members = [SimpleNamespace(id="123456", name="群名片", nick="昵称", user=None)]
    async with get_session() as db:
        tool = create_reply_tool(db, f"mention-{uuid4()}", bot_name="miyuki", parse_msg_meta=parse_message_metadata, group_members=members)
        result = json.loads(await tool.ainvoke({"content": content, "next_step": "end"}))
    assert result["ok"]
    assert [segment.target for segment in sent[0] if isinstance(segment, At)] == ["123456"]


@pytest.mark.asyncio
async def test_random_roster_selection_is_deduplicated_excludes_bots_and_sends_native_at(monkeypatch):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import At, UniMessage

    from nonebot_plugin_ai_groupmate.agent import member_tools
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    members = [SimpleNamespace(id=uid, name="同名", nick=None, user=None) for uid in ["111", "222", "222", "333", "444"]]
    populations, sent = [], []

    def sample(population, count):
        populations.append(population)
        assert count == 1
        return [population[-1]]

    async def capture(message, *args, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "selected-mention"}])

    monkeypatch.setattr(member_tools.random, "sample", sample)
    monkeypatch.setattr(UniMessage, "send", capture)
    group = f"roster-{uuid4()}"
    select = member_tools.create_group_member_tool(group, group_members=members, bot_user_ids=["444"])
    result = json.loads(await select.ainvoke({"random_selection": True, "limit": 1, "exclude_user_ids": ["111"]}))
    assert [item["user_id"] for item in populations[0]] == ["222", "333"]
    assert result["data"]["matched_count"] == 2
    selected = result["data"]["members"][0]["user_id"]
    async with get_session() as db:
        reply = create_reply_tool(db, group, bot_name="miyuki", parse_msg_meta=parse_message_metadata, group_members=members)
        assert json.loads(await reply.ainvoke({"content": f"@{selected} 就你了", "next_step": "end"}))["ok"]
    assert [segment.target for segment in sent[0] if isinstance(segment, At)] == ["333"]


@pytest.mark.asyncio
async def test_member_lookup_filters_aliases_and_reuses_bound_group_snapshot():
    from unittest.mock import AsyncMock

    from nonebot_plugin_uninfo import SceneType, QryItrface

    from nonebot_plugin_ai_groupmate.agent.member_tools import create_group_member_tool

    members = [SimpleNamespace(id="123", name="群名片", nick=None, user=SimpleNamespace(name="Nickname", nick=None))]
    interface = AsyncMock(spec=QryItrface)
    interface.get_members = AsyncMock(return_value=members)
    tool = create_group_member_tool("bound-group", interface=interface)
    for query in ["nickname", "群名", "@123"]:
        result = json.loads(await tool.ainvoke({"keyword": query}))
        assert result["data"]["members"] == [{"user_id": "123", "name": "群名片"}]
    interface.get_members.assert_awaited_once_with(SceneType.GROUP, "bound-group")


@pytest.mark.asyncio
async def test_member_lookup_failure_is_not_a_successful_empty_draw():
    from unittest.mock import AsyncMock

    from nonebot_plugin_uninfo import QryItrface

    from nonebot_plugin_ai_groupmate.agent.member_tools import create_group_member_tool

    interface = AsyncMock(spec=QryItrface)
    interface.get_members = AsyncMock(side_effect=RuntimeError("unavailable"))
    tool = create_group_member_tool("bound-group", interface=interface)
    result = json.loads(await tool.ainvoke({"random_selection": True, "limit": 1}))
    assert not result["ok"]
    assert result["reason_code"] == "members_unavailable"


@pytest.mark.asyncio
async def test_unknown_member_id_does_not_claim_a_mention_was_sent(monkeypatch):
    from unittest.mock import AsyncMock

    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    send = AsyncMock()
    monkeypatch.setattr(UniMessage, "send", send)
    async with get_session() as db:
        tool = create_reply_tool(db, f"missing-member-{uuid4()}", bot_name="miyuki", parse_msg_meta=parse_message_metadata, group_members=[])
        result = json.loads(await tool.ainvoke({"content": "@999999 就你了", "next_step": "end"}))
    assert result["reason_code"] == "invalid_mention_target"
    assert result["delivery_state"] == "not_attempted"
    send.assert_not_awaited()
