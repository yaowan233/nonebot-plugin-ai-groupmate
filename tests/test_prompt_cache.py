from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from langchain_core.messages import AIMessage, HumanMessage


def test_openrouter_gemini_enables_explicit_prompt_cache():
    from nonebot_plugin_ai_groupmate.agent.prompt_cache import (
        should_use_explicit_prompt_cache,
    )

    assert should_use_explicit_prompt_cache(
        enabled=True,
        api_format="openai",
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-3.7-flash",
    )
    assert should_use_explicit_prompt_cache(
        enabled=True,
        api_format="openai",
        base_url="https://openrouter.ai/api/v1",
        model="~google/gemini-flash-latest",
    )
    assert not should_use_explicit_prompt_cache(
        enabled=False,
        api_format="openai",
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-3.7-flash",
    )
    assert not should_use_explicit_prompt_cache(
        enabled=True,
        api_format="openai",
        base_url="https://example.com/v1",
        model="google/gemini-3.7-flash",
    )


def test_openrouter_session_key_is_stable_private_and_scoped():
    from nonebot_plugin_ai_groupmate.agent.prompt_cache import (
        build_openrouter_request_kwargs,
    )

    first = build_openrouter_request_kwargs(
        "https://openrouter.ai/api/v1",
        "123456789",
    )
    repeated = build_openrouter_request_kwargs(
        "https://openrouter.ai/api/v1",
        "123456789",
    )
    other = build_openrouter_request_kwargs(
        "https://openrouter.ai/api/v1",
        "987654321",
    )

    assert first == repeated
    assert first != other
    routed_session = first["extra_body"]["session_id"]
    assert routed_session.startswith("ai-groupmate-")
    assert "123456789" not in routed_session
    assert len(routed_session) <= 256
    assert build_openrouter_request_kwargs(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "123456789",
    ) == {}


def test_openrouter_payload_keeps_system_cache_control_marker():
    from langchain_openai import ChatOpenAI

    from nonebot_plugin_ai_groupmate.agent.prompt_cache import (
        build_system_messages,
    )

    model = ChatOpenAI(
        model="google/gemini-3.7-flash",
        api_key=SecretStr("test"),
        base_url="https://openrouter.ai/api/v1",
    )
    messages = [
        *build_system_messages("stable system prompt", use_cache_control=True),
        HumanMessage(content="dynamic user prompt"),
    ]
    payload = model._get_request_payload(messages)

    system_content = payload["messages"][0]["content"]
    assert system_content[0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][1]["content"] == "dynamic user prompt"


@pytest.mark.asyncio
async def test_agent_node_forwards_provider_request_kwargs():
    from nonebot_plugin_ai_groupmate.agent.graph import (
        AgentRunLimits,
        _make_agent_node,
    )

    calls: list[dict] = []

    class RequestSpyModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages, **kwargs):
            calls.append(kwargs)
            return AIMessage(content="done")

    agent_node = _make_agent_node(
        RequestSpyModel(),
        [],
        "system",
        {},
        AgentRunLimits(),
        lambda session_id: {"extra_body": {"session_id": session_id}},
    )
    await agent_node({
        "messages": [HumanMessage(content="hello")],
        "session_id": "group-1",
        "request_id": None,
        "reply_count": 0,
        "tool_count": 0,
        "reply_this_round": 0,
        "reply_requires_continuation": False,
        "reaction_this_round": 0,
        "called_finish": 0,
        "llm_cached_tokens": 0,
        "llm_cache_creation_tokens": 0,
        "llm_call_count": 0,
        "llm_total_tokens": 0,
        "tool_timeout_count": 0,
        "tool_timeout_names": [],
        "tool_result_truncation_count": 0,
        "side_effect_duplicate_count": 0,
        "completed_side_effect_keys": [],
        "active_skills": [],
        "required_side_effect_completed": False,
        "required_side_effect_unavailable": False,
        "required_side_effect_success_count": 0,
        "required_side_effect_target_count": 1,
        "image_input_disabled": False,
    })

    assert calls == [{"extra_body": {"session_id": "group-1"}}]


def test_openrouter_cache_write_tokens_are_recorded():
    from nonebot_plugin_ai_groupmate.usage import extract_cache_creation_tokens
    from nonebot_plugin_ai_groupmate.agent.graph import _log_llm_cache_usage

    response = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 10_000,
            "output_tokens": 20,
            "total_tokens": 10_020,
        },
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10_000,
                "completion_tokens": 20,
                "total_tokens": 10_020,
                "prompt_tokens_details": {
                    "cached_tokens": 1_400,
                    "cache_write_tokens": 8_000,
                },
            }
        },
    )

    usage = _log_llm_cache_usage(response)

    assert usage["cached_tokens"] == 1_400
    assert usage["cache_creation_tokens"] == 8_000
    assert extract_cache_creation_tokens(
        SimpleNamespace(cache_write_tokens=8_000)
    ) == 8_000
