from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage


def test_proactive_meme_sampling_is_independent_from_random_text_reply(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin.plugin_config, "reply_probability", 0.01)
    monkeypatch.setattr(plugin.plugin_config, "proactive_reaction_probability", 0.05)
    monkeypatch.setattr(plugin.plugin_config, "proactive_meme_probability", 0.02)
    rolls = iter([0.5, 0.5, 0.01])
    monkeypatch.setattr(plugin.random, "random", lambda: next(rolls))

    result = plugin._sample_proactive_reply_modes(
        addressed=False,
        continuous=False,
        command_like=False,
        has_text=True,
        is_group=True,
        reaction_supported=True,
    )

    assert result == (False, False, True)


def test_addressed_message_never_uses_proactive_sampling(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(
        plugin.random,
        "random",
        lambda: (_ for _ in ()).throw(AssertionError("不应进行随机采样")),
    )

    result = plugin._sample_proactive_reply_modes(
        addressed=True,
        continuous=False,
        command_like=False,
        has_text=True,
        is_group=True,
        reaction_supported=True,
    )

    assert result == (False, False, False)


@pytest.mark.asyncio
async def test_proactive_meme_gatekeeper_uses_a_casual_reaction_boundary(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    captured_messages = []

    class FakeFlashModel:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return AIMessage(content="YES")

    monkeypatch.setattr(agent, "get_flash_model", lambda: FakeFlashModel())

    result = await agent.check_if_should_reply(
        "Alice: 今天又加班",
        "想用表情包自然吐槽",
        "bot",
        proactive_meme_only=True,
    )

    assert result is True
    assert "低概率主动表情包采样" in captured_messages[0].content
    assert "敏感或沉重话题" in captured_messages[0].content


@pytest.mark.asyncio
async def test_proactive_meme_graph_only_exposes_meme_actions(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    captured_base_tools: list[str] = []
    captured_search_options: dict[str, Any] = {}

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

    original_create_search_meme_tool = agent.create_search_meme_tool

    def capture_search_options(*args: Any, **kwargs: Any):
        captured_search_options.update(kwargs)
        return original_create_search_meme_tool(*args, **kwargs)

    monkeypatch.setattr(agent, "get_user_relation_context", empty_context)
    monkeypatch.setattr(agent, "get_group_context", empty_context)
    monkeypatch.setattr(agent, "get_recent_relations_context", empty_context)
    monkeypatch.setattr(agent, "build_registered_agent_extensions", no_extensions)
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "build_chat_graph", fake_build_chat_graph)
    monkeypatch.setattr(agent, "create_search_meme_tool", capture_search_options)

    await agent.create_chat_graph(
        FakeSession(),
        "group-1",
        None,
        "user-1",
        "Alice",
        history=[],
        is_private=False,
        proactive_meme_only=True,
    )

    assert captured_base_tools == [
        "search_meme_image",
        "send_meme_image",
        "finish",
    ]
    assert captured_search_options["allow_context_fallback"] is True
