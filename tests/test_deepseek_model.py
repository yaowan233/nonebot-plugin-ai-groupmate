import json
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import SecretStr
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage


@pytest.mark.parametrize("streaming", [False, True])
async def test_deepseek_replays_reasoning_across_tool_and_user_turns(streaming):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai
    from nonebot_plugin_ai_groupmate.agent.graph import _invoke_model_with_idle_timeout

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        turn = len(requests)
        if turn > 1:
            assistants = [m for m in body["messages"] if m["role"] == "assistant"]
            assert [m.get("reasoning_content") for m in assistants] == [
                "reasoning-1", *(["reasoning-2"] if turn == 3 else [])
            ]
        message: dict[str, Any] = {"role": "assistant", "content": "done" if turn > 1 else "", "reasoning_content": f"reasoning-{turn}"}
        if turn == 1:
            message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
        usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "prompt_cache_hit_tokens": 12}
        if not body.get("stream"):
            return httpx.Response(200, json={"id": "test", "model": "deepseek-flash", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage})
        # Separate reasoning chunks exercise aggregation, not just one parsed response.
        delta = {k: v for k, v in message.items() if k != "reasoning_content"}
        if "tool_calls" in delta:
            delta["tool_calls"][0]["index"] = 0
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "reasoning-"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": str(turn)}}]},
            {"choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
        data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})

    cfg = ScopedConfig(chat_model="deepseek-flash", chat_base_url="https://api.deepseek.com/v1", chat_api_key="test-key")
    model = create_chat_openai(cfg)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        sdk = AsyncOpenAI(api_key="test-key", base_url=cfg.chat_base_url, http_client=client, max_retries=0)
        model.root_async_client = sdk
        model.async_client = sdk.chat.completions
        bound = model.bind_tools([{"type": "function", "function": {"name": "lookup", "description": "Read fixture", "parameters": {"type": "object", "properties": {}}}}])
        history: list[BaseMessage] = [HumanMessage(content="Look up the fixture")]
        for turn in range(3):
            if streaming:
                response = await _invoke_model_with_idle_timeout(bound, history, timeout_seconds=5, session_id="test", request_kwargs={})
            else:
                response = await bound.ainvoke(history)
            assert response.additional_kwargs.get("reasoning_content") == f"reasoning-{turn + 1}"
            assert response.usage_metadata is not None
            assert response.usage_metadata.get("input_token_details", {}).get("cache_read") == 12
            assert "reasoning-" not in response.content
            history.append(response)
            if turn == 0:
                history.append(ToolMessage(content="fixture result", tool_call_id="call-1"))
            elif turn == 1:
                history.append(HumanMessage(content="Summarize that result"))
    assert len(requests) == 3


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_deepseek_request_settings(effort):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    cfg = ScopedConfig(chat_model="deepseek-flash", chat_base_url="https://api.deepseek.com/v1", chat_api_key="test", chat_reasoning_effort=effort, chat_max_tokens=4096)
    model = create_chat_openai(cfg)
    payload = model._get_request_payload([HumanMessage(content="hello")])
    assert payload["reasoning_effort"] == effort
    assert payload["extra_body"]["thinking"]["type"] == ("disabled" if effort == "none" else "enabled")
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload
    assert ("temperature" in payload) == (effort == "none")
    assert create_chat_openai(cfg, max_tokens=512)._get_request_payload("hello")["max_tokens"] == 512


def test_deepseek_auxiliary_models_disable_thinking_and_keep_token_cap():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    cfg = ScopedConfig(llm_api_key="test", llm_base_url="https://api.deepseek.com/v1", flash_model="deepseek-flash")
    payload = create_chat_openai(cfg, role="flash")._get_request_payload("hello")
    assert payload["extra_body"]["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == cfg.flash_max_tokens


@pytest.mark.parametrize("base_url", ["https://dashscope.aliyuncs.com/compatible-mode/v1", "https://openrouter.ai/api/v1", "https://api.deepseek.com.example.org/v1"])
def test_other_endpoints_do_not_receive_deepseek_parameters(base_url):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    cfg = ScopedConfig(chat_api_key="test", chat_base_url=base_url, chat_model="deepseek-flash")
    payload = create_chat_openai(cfg)._get_request_payload("hello")
    assert "extra_body" not in payload
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.7


def test_group_override_selects_deepseek_adapter_with_inherited_effort():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_llm
    from nonebot_plugin_ai_groupmate.deepseek import DeepSeekChatOpenAI
    from nonebot_plugin_ai_groupmate.group_model_config import ActiveGroupModelConfig, _resolve_active_chat_config

    global_config = ScopedConfig(chat_api_key="global-test", chat_reasoning_effort="high")
    scoped = ActiveGroupModelConfig(group_id="1", api_format="openai", base_url="https://api.deepseek.com/v1", api_key="scoped-test", chat_model="deepseek-flash", chat_multimodal=True)
    resolved = _resolve_active_chat_config(scoped, global_config)
    model = create_chat_llm(resolved)
    assert isinstance(model, DeepSeekChatOpenAI)
    assert model.reasoning_effort == "high"
    assert isinstance(model.openai_api_key, SecretStr)
    assert model.openai_api_key.get_secret_value() == "scoped-test"
    assert global_config.chat_model == "qwen3.7-plus"


def test_eval_runner_uses_production_deepseek_transport(monkeypatch):
    from evals.runner import _create_model, build_argument_parser
    from nonebot_plugin_ai_groupmate.deepseek import DeepSeekChatOpenAI

    monkeypatch.setenv("EVAL_API_KEY", "test")
    args = build_argument_parser().parse_args(["--model", "deepseek-flash", "--base-url", "https://api.deepseek.com/v1", "--reasoning-effort", "none"])
    model = _create_model(args, args.model)
    assert isinstance(model, DeepSeekChatOpenAI)
    assert model.reasoning_effort == "none"


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_generic_reasoning_and_output_cap_reach_compatible_endpoint(effort):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    cfg = ScopedConfig(chat_api_key="test", chat_base_url="https://gateway.example/v1", chat_model="reasoning-model", chat_reasoning_effort=effort, chat_max_tokens=16384)
    payload = create_chat_openai(cfg)._get_request_payload("hello")
    assert payload["reasoning_effort"] == effort
    assert payload["max_completion_tokens"] == 16384
    assert "extra_body" not in payload
    assert ("temperature" in payload) == (effort == "none")
    assert create_chat_openai(cfg, max_tokens=512)._get_request_payload("hello")["max_completion_tokens"] == 512


def test_deepseek_defaults_remain_low_with_default_cap():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    cfg = ScopedConfig(chat_api_key="test", chat_base_url="https://api.deepseek.com/v1")
    payload = create_chat_openai(cfg)._get_request_payload("hello")
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == 8192


def test_responses_tools_respect_explicit_effort():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    cfg = ScopedConfig(chat_api_key="test", chat_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", chat_responses_builtin_tools=["code_interpreter"], chat_reasoning_effort="high")
    model = create_chat_openai(cfg)
    assert model.reasoning == {"effort": "high"}
    assert model.reasoning_effort is None
