from __future__ import annotations

import json
import datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _history(
    msg_id: int,
    user_id: str,
    text: str,
    *,
    content_type: str = "text",
):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    return ChatHistorySchema(
        msg_id=msg_id,
        session_id="group-1",
        user_id=user_id,
        content_type=content_type,
        content=f"id: {msg_id}\n{text}",
        created_at=datetime.datetime.now(),
        user_name=user_id,
    )


def test_repeat_chain_requires_two_different_users_with_the_same_short_text():
    from nonebot_plugin_ai_groupmate import agent

    bot_context = _history(1, "bot", "我也觉得", content_type="bot")
    repeated = agent._detect_repeat_chain(
        [bot_context, _history(2, "user-1", "确实"), _history(3, "user-2", "确实")]
    )
    same_user = agent._detect_repeat_chain(
        [bot_context, _history(2, "user-1", "确实"), _history(3, "user-1", "确实")]
    )
    changed = agent._detect_repeat_chain(
        [bot_context, _history(2, "user-1", "确实"), _history(3, "user-2", "真的")]
    )
    mention = agent._detect_repeat_chain(
        [
            bot_context,
            _history(2, "user-1", "@Alice 确实"),
            _history(3, "user-2", "@Alice 确实"),
        ]
    )
    without_bot = agent._detect_repeat_chain(
        [_history(1, "user-1", "确实"), _history(2, "user-2", "确实")]
    )

    assert repeated == "确实"
    assert same_user is None
    assert changed is None
    assert mention is None
    assert without_bot is None


def test_repeat_chain_uses_an_independent_probability(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin.plugin_config, "repeat_probability", 0.15)
    monkeypatch.setattr(plugin.random, "random", lambda: 0.1)

    assert plugin._sample_repeat_reply(
        repeat_text="确实",
        addressed=False,
        continuous=False,
        command_like=False,
        is_group=True,
    )


@pytest.mark.asyncio
async def test_sampled_repeat_is_sent_without_calling_the_agent(monkeypatch):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin

    invoked: list[dict[str, str]] = []

    class FakeDbSession:
        async def commit(self):
            return None

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeDbSession()

        async def __aexit__(self, *args: object):
            return None

    class FakeReplyTool:
        async def ainvoke(self, payload: dict[str, str]) -> str:
            invoked.append(payload)
            return json.dumps({"status": "sent", "message": "ok"})

    async def forbidden_agent(*args: Any, **kwargs: Any):
        raise AssertionError("抽中的复读不应调用 Agent")

    monkeypatch.setattr(plugin, "get_session", lambda: FakeSessionContext())
    monkeypatch.setattr(plugin, "create_reply_tool", lambda *args, **kwargs: FakeReplyTool())
    monkeypatch.setattr(plugin, "choice_response_strategy", forbidden_agent)

    session = SimpleNamespace(
        scene=SimpleNamespace(type=SceneType.GROUP, id="group-1"),
        self_id="bot-1",
    )
    await plugin.handle_reply_logic(
        "request-1",
        cast(Uninfo, session),
        cast(QryItrface, object()),
        cast(Bot, object()),
        cast(Event, object()),
        "bot",
        "user-2",
        "Bob",
        False,
        False,
        None,
        repeat_text="确实",
    )

    assert invoked == [{"content": "确实", "next_step": "end"}]


@pytest.mark.asyncio
async def test_repeat_reply_guard_rejects_commentary():
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    reply_tool = create_reply_tool(
        object(),
        "group-1",
        bot_name="bot",
        parse_msg_meta=lambda content: (None, None, content),
        repeat_text="确实",
    )

    raw_result = await reply_tool.ainvoke(
        {"content": "复读是吧", "next_step": "end"}
    )
    result = json.loads(raw_result)

    assert result["status"] == "failed"
    assert "只能原样复读" in result["message"]


@pytest.mark.asyncio
async def test_repeat_graph_only_exposes_exact_reply_or_silence(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    captured_base_tools: list[str] = []

    class FakeSession:
        async def commit(self):
            return None

    async def empty_context(*args: Any, **kwargs: Any) -> str:
        return ""

    async def no_extensions(*args: Any, **kwargs: Any):
        return [], [], []

    def fake_build_chat_graph(model, tools, system_prompt, **kwargs):
        captured_base_tools.extend(tool.name for tool in kwargs["base_tools"])
        return object()

    monkeypatch.setattr(agent, "get_user_relation_context", empty_context)
    monkeypatch.setattr(agent, "get_group_context", empty_context)
    monkeypatch.setattr(agent, "get_recent_relations_context", empty_context)
    monkeypatch.setattr(agent, "build_registered_agent_extensions", no_extensions)
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "build_chat_graph", fake_build_chat_graph)

    await agent.create_chat_graph(
        FakeSession(),
        "group-1",
        None,
        "user-2",
        "Bob",
        history=[],
        is_private=False,
        repeat_text="确实",
    )

    assert captured_base_tools == ["reply_user", "finish"]
