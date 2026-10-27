from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.parametrize("text", [
    "miyuki 回个小表情", "给这条消息点一个小表情", "给个消息表情",
    "回个表情", "用reaction回应一下", "回复一个表情",
])
def test_explicit_reaction_requests_reach_action_routing(text):
    import nonebot_plugin_ai_groupmate as plugin

    assert not plugin._is_explicit_meme_request(text)
    assert plugin._is_explicit_reaction_request(text)


@pytest.mark.parametrize("text", ["miyuki 给我回个㊗️", "回复一个❤️", "用😄回应一下"])
def test_requested_unicode_symbol_is_a_reaction_request(text):
    import nonebot_plugin_ai_groupmate as plugin

    assert plugin._is_explicit_reaction_request(text)


@pytest.mark.asyncio
@pytest.mark.parametrize(("requested", "expected_id"), [("㊗️", "12951"), ("❤️", "10084"), ("😄", "128516")])
async def test_exact_requested_symbol_overrides_mood_and_is_encoded_for_qq(monkeypatch, requested, expected_id):
    import json

    from nonebot.adapters import Bot, Event

    from nonebot_plugin_ai_groupmate.agent import reaction

    sent = []

    async def capture(emoji, **_kwargs):
        sent.append(emoji)

    monkeypatch.setattr(reaction, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(reaction, "message_reaction", capture)
    event = SimpleNamespace(message_id=-123, message_type="group", get_plaintext=lambda: "miyuki 给我回个" + requested)
    action = reaction.create_reaction_tool(SimpleNamespace(add=lambda _: None), "group", None, "miyuki", cast(Bot, SimpleNamespace()), cast(Event, event))
    result = json.loads(await action.ainvoke({"mood": "laugh", "emoji": "76"}))
    assert result["status"] == "succeeded"
    assert sent == [expected_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(("emoji", "expected_id"), [("㊗️", "12951"), ("❤️", "10084"), ("76", "76")])
async def test_model_can_choose_an_exact_symbol_without_a_mood(monkeypatch, emoji, expected_id):
    import json

    from nonebot.adapters import Bot, Event

    from nonebot_plugin_ai_groupmate.agent import reaction

    sent = []

    async def capture(symbol, **_kwargs):
        sent.append(symbol)

    monkeypatch.setattr(reaction, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(reaction, "message_reaction", capture)
    event = SimpleNamespace(message_id=123, message_type="group", get_plaintext=lambda: "这把终于过了")
    action = reaction.create_reaction_tool(SimpleNamespace(add=lambda _: None), "group", None, "miyuki", cast(Bot, SimpleNamespace()), cast(Event, event))
    result = json.loads(await action.ainvoke({"emoji": emoji}))
    assert result["status"] == "succeeded"
    assert sent == [expected_id]


@pytest.mark.parametrize("text", [
    "不要给这条消息点表情", "别回复表情", "你发一个表情包", "这个表情是什么意思",
    "别回个㊗️", "不要回复❤️", "回复个小表情包",
])
def test_non_action_or_image_requests_do_not_require_reaction(text):
    import nonebot_plugin_ai_groupmate as plugin

    assert not plugin._is_explicit_reaction_request(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["👨‍👩‍👧‍👦", "©️"])
async def test_unsupported_requested_symbol_is_not_replaced_with_random_mood(monkeypatch, symbol):
    import json

    from nonebot.adapters import Bot, Event

    from nonebot_plugin_ai_groupmate.agent import reaction

    async def no_send(*_args, **_kwargs):
        pytest.fail("An unsupported exact symbol must not become another reaction")

    monkeypatch.setattr(reaction, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(reaction, "message_reaction", no_send)
    event = SimpleNamespace(message_id=123, message_type="group", get_plaintext=lambda: "回个" + symbol)
    action = reaction.create_reaction_tool(SimpleNamespace(add=lambda _: None), "group", None, "miyuki", cast(Bot, SimpleNamespace()), cast(Event, event))
    result = json.loads(await action.ainvoke({"mood": "like"}))
    assert result["status"] == "failed"
    assert result["retryable"] is False
    assert result["delivery_state"] == "not_attempted"


@pytest.mark.asyncio
async def test_reply_strategy_forwards_required_reaction(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    class ReachedFactory(BaseException):
        pass

    captured = {}

    async def factory(*_args, **kwargs):
        captured.update(kwargs)
        raise ReachedFactory

    monkeypatch.setattr(agent, "create_chat_graph", factory)
    with pytest.raises(ReachedFactory):
        await agent.choice_response_strategy(cast(AsyncSession, SimpleNamespace()), "group", None, [], "alice", "Alice", reaction_required=True)
    assert captured.get("reaction_required") is True


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return next(self.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
async def test_required_reaction_executes_or_reports_verified_platform_limit(unsupported):
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_skipped, tool_success

    sent = []
    actions = []

    @tool("add_message_reaction")
    async def react(mood: str) -> str:
        """Apply a message reaction."""
        actions.append(mood)
        if unsupported:
            return tool_skipped("unsupported_message_id", "平台不支持这条消息", delivery_state="not_attempted")
        return tool_success("reaction_updated", "done", delivery_state="completed")

    @tool("reply_user")
    async def reply(content: str, next_step: str) -> str:
        """Send text."""
        sent.append(content)
        return tool_success("replied", "done", delivery_state="completed")

    @tool("finish")
    def finish() -> str:
        """Finish the turn."""
        return ""

    responses = [AIMessage(content="", tool_calls=[{"name": "add_message_reaction", "args": {"mood": "laugh"}, "id": "react"}]), AIMessage(content="这条消息平台不支持点表情")]
    if not unsupported:
        responses = [AIMessage(content="好，给你点了"), AIMessage(content="", tool_calls=[{"name": "finish", "args": {}, "id": "early"}]), responses[0]]
    graph = build_chat_graph(Model(responses), [react, reply, finish], "system", required_side_effect_tool="add_message_reaction")
    result = await graph.ainvoke({"messages": [HumanMessage(content="回个小表情")], "session_id": "group", "request_id": None})
    assert actions == ["laugh"]
    assert sent == (["这条消息平台不支持点表情"] if unsupported else [])
    assert result["required_side_effect_completed"] is (not unsupported)
