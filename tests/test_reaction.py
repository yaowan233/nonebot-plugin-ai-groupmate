from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage


def test_unaddressed_message_does_not_sample_proactive_reaction(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin.plugin_config, "reply_probability", 0.01)
    monkeypatch.setattr(plugin.plugin_config, "proactive_reaction_probability", 0.05)
    monkeypatch.setattr(plugin.plugin_config, "proactive_meme_probability", 0.02)
    rolls = iter([0.5])
    monkeypatch.setattr(plugin.random, "random", lambda: next(rolls))

    result = plugin._sample_proactive_reply_modes(
        addressed=False,
        continuous=False,
        command_like=False,
        has_text=True,
        is_group=True,
        reaction_supported=True,
    )

    assert result == (False, False, False)


def test_explicit_reaction_request_detection():
    import nonebot_plugin_ai_groupmate as plugin

    assert plugin._is_explicit_reaction_request("给这条消息点个表情")
    assert plugin._is_explicit_reaction_request("回应表情")
    assert plugin._is_explicit_reaction_request("reaction一下")
    assert not plugin._is_explicit_reaction_request("这个表情是什么意思")
    assert not plugin._is_explicit_reaction_request("miyuki多发点表情包")
    assert plugin._is_explicit_meme_request("miyuki多发点表情包")
    assert plugin._is_explicit_meme_request("随便发点表情")
    assert plugin._is_explicit_meme_request("miyuki你发一下图")
    assert plugin._is_explicit_meme_request("图呢")
    assert plugin._is_explicit_meme_request("来几张龙图")
    assert plugin._is_explicit_meme_request("发五张卡通猫娘图")


def test_explicit_meme_send_count_is_parsed_and_capped():
    import nonebot_plugin_ai_groupmate as plugin

    assert plugin._get_explicit_meme_send_count("发一个表情包") == 1
    assert plugin._get_explicit_meme_send_count("发3张图") == 3
    assert plugin._get_explicit_meme_send_count("发五张卡通猫娘图") == 5
    assert plugin._get_explicit_meme_send_count("发10张表情包") == 5
    assert plugin._get_explicit_meme_send_count("多发点表情包") == 3
    assert plugin._get_explicit_meme_send_count("来几张龙图") == 3
    assert plugin._get_explicit_meme_send_count("今天聊点别的") == 1


@pytest.mark.asyncio
async def test_proactive_reaction_gatekeeper_uses_a_lightweight_boundary(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    captured_messages = []

    class FakeFlashModel:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return AIMessage(content="YES")

    monkeypatch.setattr(agent, "get_flash_model", lambda: FakeFlashModel())

    result = await agent.check_if_should_reply(
        "Alice: 笑死我了",
        "这条消息适合轻量回应",
        "bot",
        proactive_reaction_only=True,
    )

    assert result is True
    assert "主动 reaction 采样" in captured_messages[0].content
    assert "提问、求助" in captured_messages[0].content


@pytest.mark.asyncio
async def test_proactive_reaction_graph_only_exposes_reaction_or_finish(monkeypatch):
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
    monkeypatch.setattr(agent, "is_onebot_context", lambda bot, event: True)
    monkeypatch.setattr(agent, "build_chat_graph", fake_build_chat_graph)

    await agent.create_chat_graph(
        FakeSession(),
        "group-1",
        None,
        "user-1",
        "Alice",
        history=[],
        is_private=False,
        proactive_reaction_only=True,
    )

    assert captured_base_tools == ["add_message_reaction", "finish"]


@pytest.mark.asyncio
async def test_normal_onebot_graph_exposes_reaction_without_loading_skill(monkeypatch):
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
    monkeypatch.setattr(agent, "is_onebot_context", lambda bot, event: True)
    monkeypatch.setattr(agent, "build_chat_graph", fake_build_chat_graph)

    await agent.create_chat_graph(
        FakeSession(),
        "group-1",
        None,
        "user-1",
        "Alice",
        history=[],
        is_private=False,
    )

    assert "add_message_reaction" in captured_base_tools


@pytest.mark.asyncio
async def test_reaction_tool_applies_to_the_current_numeric_message(monkeypatch):
    from nonebot.adapters import Bot, Event

    from nonebot_plugin_ai_groupmate.agent import reaction

    calls: list[dict[str, Any]] = []
    added: list[Any] = []

    async def fake_message_reaction(emoji: str, **kwargs: Any) -> None:
        calls.append({"emoji": emoji, **kwargs})

    monkeypatch.setattr(reaction, "is_onebot_context", lambda bot, event: True)
    monkeypatch.setattr(reaction, "message_reaction", fake_message_reaction)

    db_session = SimpleNamespace(add=added.append)
    event = SimpleNamespace(message_id=123)
    tool = reaction.create_reaction_tool(
        db_session,
        "group-1",
        None,
        "bot",
        cast(Bot, SimpleNamespace()),
        cast(Event, event),
    )

    result = await tool.ainvoke({"mood": "like"})

    assert "已对当前触发消息" in result
    assert calls[0]["message_id"] is None
    assert calls[0]["event"] is event
    assert len(added) == 1
