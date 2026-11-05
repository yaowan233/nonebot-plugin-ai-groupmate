import json
import asyncio
from uuid import uuid4
from types import SimpleNamespace
from typing import Any, cast

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


def test_current_request_keeps_the_actual_at_target_id():
    from nonebot_plugin_ai_groupmate.agent import _build_current_request_boundary

    original = [SimpleNamespace(type="text", data={"text": "miyuki 帮他解除禁言 "}), SimpleNamespace(type="at", data={"qq": "1001"})]
    event = SimpleNamespace(self_id="9000", original_message=original, get_plaintext=lambda: "miyuki 帮他解除禁言")
    boundary = _build_current_request_boundary([], "requester", "User", cast(Any, event))
    assert "1001" in boundary, "Plaintext strips the @ segment, losing the moderation target"


async def test_unmute_defaults_to_one_explicitly_mentioned_member():
    from test_moderation_tools import FakeBot, FakeSession, _member

    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    bot = FakeBot()
    tool = moderation_tools.create_mute_tool(FakeSession(), "12345", None, None, "9000", bot_name="miyuki", bot=cast(Any, bot),
                                               group_members=[_member("9000", "miyuki", "admin"), _member("1001", "同名"), _member("1002", "同名")], mentioned_user_ids=("9000", "1001"))
    result = json.loads(await tool.ainvoke({"duration_seconds": 0, "reason": "按请求解除禁言"}))
    assert result["status"] == "succeeded"
    assert bot.calls == [{"group_id": 12345, "user_id": 1001, "duration": 0}]


async def test_unmute_with_multiple_mentions_does_not_guess_a_target():
    from test_moderation_tools import FakeBot, FakeSession, _member

    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    bot = FakeBot()
    tool = moderation_tools.create_mute_tool(FakeSession(), "12345", None, None, "9000", bot_name="miyuki", bot=cast(Any, bot),
                                               group_members=[_member("9000", "miyuki", "admin"), _member("1001", "A"), _member("1002", "B")], mentioned_user_ids=("9000", "1001", "1002"))
    result = json.loads(await tool.ainvoke({"duration_seconds": 0, "reason": "解除禁言"}))
    assert result["reason_code"] == "target_ambiguous"
    assert not bot.calls


def test_native_mentions_survive_adapter_bot_at_removal_and_skip_at_all():
    from nonebot_plugin_ai_groupmate.message_mentions import format_member_mention, event_mentioned_user_ids, event_text_with_mentions

    original = [SimpleNamespace(type="at", data={"qq": "9000"}), SimpleNamespace(type="text", data={"text": "帮他解除禁言 "}),
                SimpleNamespace(type="at", data={"qq": "1001"}), SimpleNamespace(type="at", data={"qq": "all"})]
    event = SimpleNamespace(original_message=original, get_message=lambda: [], get_plaintext=lambda: "帮他解除禁言")
    assert event_mentioned_user_ids(event) == ("9000", "1001")
    assert "@1001" in event_text_with_mentions(event)
    assert "1001" in format_member_mention("1001", "昵称")


async def test_input_batch_retains_earlier_native_mentions_for_tools(monkeypatch):
    from typing import Any, cast

    from nonebot_plugin_uninfo import SceneType

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.message_batching import InputBatcher
    from nonebot_plugin_ai_groupmate.message_mentions import current_mentioned_user_ids

    monkeypatch.setattr(plugin, "_input_batcher", InputBatcher())
    monkeypatch.setattr(plugin.plugin_config, "group_message_debounce_seconds", 0.01)
    queued = []

    async def queue(group, request):
        queued.append(request)
        return True

    monkeypatch.setattr(plugin, "_queue_group_reply_request", queue)

    def request(number, mentioned):
        return plugin.ReplyRequest(request_id=number, session=cast(Any, SimpleNamespace(scene=SimpleNamespace(type=SceneType.GROUP))), interface=cast(Any, None),
                                   bot=cast(Any, SimpleNamespace(self_id="9000")), event=cast(Any, SimpleNamespace()), bot_name="miyuki", user_id="u", user_name="User",
                                   is_tome=True, is_continuous=False, reply_to_id=None, mentioned_user_ids=mentioned)

    await plugin._submit_reply_input("group", request("1", ("9000", "1001")), "1", "miyuki 帮 @1001 解禁")
    await plugin._submit_reply_input("group", request("2", ()), "2", "就是解除禁言")
    await asyncio.gather(*list(plugin._input_batcher.tasks))
    assert len(queued) == 1
    assert queued[0].mentioned_user_ids == ("9000", "1001")

    async def handle(*args):
        assert current_mentioned_user_ids.get() == ("9000", "1001")

    monkeypatch.setattr(plugin, "handle_reply_request", handle)
    await plugin._execute_reply_request(queued[0])
    assert current_mentioned_user_ids.get() is None
    await plugin._input_batcher.close()
