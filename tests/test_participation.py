import asyncio
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

KEY = ("bot", "group")


@pytest.fixture
def policy():
    from nonebot_plugin_ai_groupmate.participation import ParticipationPolicy

    now = [1000.0]
    return ParticipationPolicy(lambda: now[0]), now


def test_default_accumulates_opportunities_and_zero_disables(policy):
    scheduler, now = policy
    assert not any(scheduler.candidate(KEY, 0.01) for _ in range(9))
    assert scheduler.candidate(KEY, 0.01)
    assert not any(scheduler.candidate(KEY, 0.1) for _ in range(100))
    now[0] += 20
    assert scheduler.candidate(KEY, 0.01)
    assert not scheduler.candidate(KEY, 0)
    now[0] += 1000
    # No opportunity is earned by silence.
    assert not scheduler.candidate(KEY, 0.01)


def test_recent_participation_follows_other_members_with_cooldown(policy):
    scheduler, now = policy
    scheduler.sent(KEY, proactive=False)
    assert not scheduler.candidate(KEY, 0.1)
    now[0] += 45
    scheduler.observe(KEY, addressed=False, disengaged=False)
    assert not scheduler.candidate(KEY, 0.01)
    scheduler.observe(KEY, addressed=False, disengaged=False)
    assert scheduler.candidate(KEY, 0.01)


def test_no_reply_backoff_requires_traffic_and_time_and_recovers(policy):
    scheduler, now = policy
    scheduler.sent(KEY, proactive=True)
    now[0] += 130
    assert scheduler.state(KEY).unanswered == 0
    for _ in range(4):
        scheduler.observe(KEY, addressed=False, disengaged=False)
    assert scheduler.state(KEY).unanswered == 0
    scheduler.observe(KEY, addressed=False, disengaged=False)
    assert scheduler.state(KEY).unanswered == 1
    assert not scheduler.can_reply(KEY)
    scheduler.engaged(KEY)
    assert scheduler.state(KEY).unanswered == 0
    assert scheduler.can_reply(KEY)


def test_bot_share_and_send_cap_prevent_flooding(policy):
    scheduler, now = policy
    for _ in range(3):
        scheduler.sent(KEY, proactive=True)
        now[0] += 50
    assert not scheduler.can_reply(KEY)
    for _ in range(40):
        scheduler.observe(KEY, addressed=False, disengaged=False)
    assert not scheduler.can_reply(KEY)  # cap remains even with enough human traffic
    now[0] += 601
    assert scheduler.can_reply(KEY)


def test_stop_and_late_classifier_do_not_reopen_conversation(policy):
    scheduler, now = policy
    scheduler.sent(KEY, proactive=True)
    previous = scheduler.state(KEY).last_sent
    now[0] += 1
    scheduler.sent(KEY, proactive=True)
    scheduler.engaged(KEY, expected_sent=previous)
    assert scheduler.state(KEY).pending_since is not None
    scheduler.observe(KEY, addressed=False, disengaged=True)
    now[0] += 60
    scheduler.engaged(KEY)
    assert not scheduler.can_reply(KEY)
    scheduler.observe(KEY, addressed=True, disengaged=False)
    assert scheduler.can_reply(KEY)


def test_state_is_scoped_bounded_and_expires(policy):
    scheduler, now = policy
    scheduler.sent(KEY, proactive=True)
    assert scheduler.can_reply(("other-bot", "group"))
    assert scheduler.can_reply(("bot", "other-group"))
    now[0] += 1801
    assert scheduler.state(KEY).last_sent == float("-inf")
    for number in range(2100):
        scheduler.state(("bot", str(number)))
    assert len(scheduler.states) == 2048


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "reply", "continuation"),
    [
        ('{"should_reply":true,"continues_bot_conversation":true}', True, True),
        ('{"should_reply":false,"continues_bot_conversation":true}', False, True),
        ('{"should_reply":true,"continues_bot_conversation":false}', True, False),
        ('{"should_reply":"false","continues_bot_conversation":true}', False, False),
        ("YES", False, False),
        ("invalid", False, False),
    ],
)
async def test_gatekeeper_structured_decision(monkeypatch, response, reply, continuation):
    from nonebot_plugin_ai_groupmate import agent

    class Model:
        async def ainvoke(self, messages, **kwargs):
            assert kwargs["max_tokens"] >= 64
            assert "普通闲聊本身不是拒绝理由" in messages[0].content
            assert "不是证据" in messages[0].content
            assert "[Bot] 刚才的话" in messages[1].content
            return AIMessage(content=response)

    monkeypatch.setattr(agent, "get_flash_model", Model)
    callback = Mock()
    assert await agent.check_if_should_reply(
        "[Bot] 刚才的话\n[群友 B] 那你刚才那个方案呢",
        "那你刚才那个方案呢", "bot",
        participation_context="近期参与过",
        on_continuation=callback,
    ) is reply
    assert callback.called is continuation


@pytest.mark.asyncio
async def test_delivery_context_isolated_and_stop_blocks_inflight(monkeypatch, policy):
    from importlib import import_module

    from nonebot_plugin_ai_groupmate.reply_guard import is_request_active, activate_request_id, deactivate_request_id

    module = import_module("nonebot_plugin_ai_groupmate.participation")
    scheduler, _ = policy
    monkeypatch.setattr(module, "participation", scheduler)
    await activate_request_id("group", "request")
    token = module.delivery_context.set((KEY, True))
    try:
        assert await is_request_active("group", "request")
        # A child tool inherits the request context.
        async def send():
            module.record_group_delivery("group")
        await asyncio.create_task(send())
        assert len(scheduler.state(KEY).proactive_sends) == 1
        module.record_group_delivery("other-group")
        assert len(scheduler.states) == 1
        scheduler.observe(KEY, addressed=False, disengaged=True)
        assert not await is_request_active("group", "request")
    finally:
        module.delivery_context.reset(token)
        await deactivate_request_id("group", "request")
    module.record_group_delivery("group")
    assert len(scheduler.state(KEY).proactive_sends) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeds", [True, False])
async def test_actual_reply_counts_only_after_send(monkeypatch, policy, succeeds):
    import json
    from types import SimpleNamespace
    from importlib import import_module
    from unittest.mock import AsyncMock

    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    module = import_module("nonebot_plugin_ai_groupmate.participation")
    scheduler, _ = policy
    monkeypatch.setattr(module, "participation", scheduler)
    result = SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: None))
    db = SimpleNamespace(execute=AsyncMock(return_value=result), commit=AsyncMock(), rollback=AsyncMock(), add=Mock())

    async def send(self, *args, **kwargs):
        assert scheduler.state(KEY).last_sent == float("-inf")
        if not succeeds:
            raise RuntimeError("adapter failed")
        return SimpleNamespace(msg_ids=[{"message_id": "sent-id"}])

    monkeypatch.setattr(UniMessage, "send", send)
    tool = create_reply_tool(db, "group", bot_name="Bot", parse_msg_meta=lambda text: (None, None, text))
    token = module.delivery_context.set((KEY, True))
    try:
        response = json.loads(await tool.ainvoke({"content": "一起试试这个办法吧", "next_step": "end"}))
    finally:
        module.delivery_context.reset(token)
    assert (response["status"] == "succeeded") is succeeds
    assert len(scheduler.state(KEY).proactive_sends) == int(succeeds)
