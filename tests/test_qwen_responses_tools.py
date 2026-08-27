from typing import Any

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage


def test_builtin_tool_config_accepts_webui_and_env_formats():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    csv_config = ScopedConfig.model_validate({
        "chat_responses_builtin_tools": "web_search， web_extractor,web_search"
    })
    json_config = ScopedConfig.model_validate({
        "chat_responses_builtin_tools": '["code_interpreter", "image_search"]'
    })

    assert csv_config.chat_responses_builtin_tools == [
        "web_search",
        "web_extractor",
    ]
    assert json_config.chat_responses_builtin_tools == [
        "code_interpreter",
        "image_search",
    ]

    with pytest.raises(ValidationError):
        ScopedConfig.model_validate({"chat_responses_builtin_tools": "t2i_search"})


def test_builtin_tools_require_official_dashscope_endpoint_and_add_dependency():
    from nonebot_plugin_ai_groupmate.config import (
        ScopedConfig,
        resolve_chat_responses_builtin_tools,
    )

    workspace_config = ScopedConfig(
        chat_base_url=(
            "https://workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        ),
        chat_responses_builtin_tools=["web_extractor"],
    )
    custom_provider_config = ScopedConfig(
        chat_base_url="https://example.com/v1",
        chat_responses_builtin_tools=["web_search"],
    )

    assert resolve_chat_responses_builtin_tools(workspace_config) == [
        {"type": "web_search"},
        {"type": "web_extractor"},
    ]
    assert resolve_chat_responses_builtin_tools(custom_provider_config) == []


def test_settings_page_renders_builtin_tools_as_priced_checkboxes():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.settings_ui import render_settings_page

    html = render_settings_page(
        ScopedConfig(chat_responses_builtin_tools=["web_search"]),
        ScopedConfig(),
        overridden_fields=set(),
        pending_restart_fields=set(),
        dashboard_path="/usage",
        settings_path="/settings",
    )

    assert 'value="web_search" checked' in html
    assert "联网搜索（北京 ¥4/千次）" in html
    assert "网页抓取（限免，依赖联网搜索）" in html
    assert 'type="hidden" data-setting="chat_responses_builtin_tools"' in html


def test_chat_openai_uses_responses_api_and_reasoning_for_code_interpreter():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_openai

    model = create_chat_openai(
        ScopedConfig(
            chat_model="qwen3.8-max",
            chat_api_key="test-key",
            chat_responses_builtin_tools=["code_interpreter"],
        )
    )

    assert model.use_responses_api is True
    assert model.output_version == "responses/v1"
    assert model.reasoning == {"effort": "medium"}

    provider_tools = [
        {"type": "web_search"},
        {"type": "web_extractor"},
        {"type": "web_search_image"},
        {"type": "image_search"},
    ]
    bound_model = model.bind(tools=provider_tools)
    assert bound_model.kwargs["tools"] == provider_tools


class _CodeInterpreterModel:
    def __init__(self):
        self.bound_tools: list[Any] = []
        self.messages: list[Any] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        self.messages = list(messages)
        return AIMessage(content=[{"type": "text", "text": "1728"}])


@pytest.mark.asyncio
async def test_code_interpreter_isolated_wrapper_returns_server_answer():
    from nonebot_plugin_ai_groupmate.agent.qwen_responses_tools import (
        create_qwen_code_interpreter_tool,
    )

    model = _CodeInterpreterModel()
    code_tool = create_qwen_code_interpreter_tool(model)

    result = await code_tool.ainvoke({"task": "计算 12 的 3 次方"})

    assert model.bound_tools == [{"type": "code_interpreter"}]
    assert model.messages[0].content == "计算 12 的 3 次方"
    assert result == "1728"
