import pytest
from pydantic import SecretStr, ValidationError


def test_vertex_chat_model_uses_vertex_settings_and_normalizes_openrouter_name(
    monkeypatch,
):
    import langchain_google_genai

    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_llm

    captured = {}

    class FakeVertexModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeVertexModel,
    )
    cfg = ScopedConfig(
        chat_api_format="vertex",
        chat_model="google/gemini-3.7-flash",
        chat_api_key="old-openrouter-key",
        chat_base_url="https://openrouter.ai/api/v1",
        vertex_project="project-id",
        vertex_location="global",
        vertex_api_key="vertex-key",
    )

    model = create_chat_llm(cfg)

    assert isinstance(model, FakeVertexModel)
    assert captured == {
        "model": "gemini-3.7-flash",
        "vertexai": True,
        "temperature": 0.7,
        "location": "global",
        "project": "project-id",
        "api_key": SecretStr("vertex-key"),
    }


def test_vertex_service_account_credentials_take_precedence(monkeypatch):
    import langchain_google_genai
    from google.oauth2 import service_account

    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_vertex_llm

    captured = {}
    fake_credentials = object()

    class FakeVertexModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_load(path, *, scopes):
        assert path == "/run/secrets/vertex.json"
        assert scopes == ["https://www.googleapis.com/auth/cloud-platform"]
        return fake_credentials

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeVertexModel,
    )
    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_file",
        fake_load,
    )
    cfg = ScopedConfig(
        chat_model="gemini-3.7-flash",
        vertex_api_key="ignored-key",
        vertex_credentials_path="/run/secrets/vertex.json",
    )

    create_vertex_llm(cfg)

    assert captured["credentials"] is fake_credentials
    assert "api_key" not in captured


def test_vertex_tagging_and_vision_have_output_limit(monkeypatch):
    import langchain_google_genai

    from nonebot_plugin_ai_groupmate.config import (
        ScopedConfig,
        create_vision_llm,
        create_tagging_llm,
    )

    calls = []

    class FakeVertexModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeVertexModel,
    )
    cfg = ScopedConfig(
        tagging_api_format="vertex",
        vision_api_format="vertex",
        vision_model="gemini-3.7-flash",
    )

    create_tagging_llm(cfg)
    create_vision_llm(cfg)

    assert [call["max_tokens"] for call in calls] == [1024, 1024]
    assert all(call["vertexai"] is True for call in calls)


def test_api_format_rejects_unknown_provider():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    with pytest.raises(ValidationError):
        ScopedConfig(chat_api_format="unknown")  # type: ignore[arg-type]


def test_settings_page_exposes_vertex_provider_and_fields():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.settings_ui import render_settings_page

    html = render_settings_page(
        ScopedConfig(chat_api_format="vertex", vertex_project="project-id"),
        ScopedConfig(),
        overridden_fields=set(),
        pending_restart_fields=set(),
        dashboard_path="/usage",
        settings_path="/usage/settings",
    )

    assert '<option value="vertex" selected>Google Vertex AI</option>' in html
    assert 'data-setting="vertex_project"' in html
    assert 'data-setting="vertex_credentials_path"' in html


def test_vertex_chat_does_not_forward_openrouter_request_options(monkeypatch):
    import nonebot_plugin_ai_groupmate.agent as agent_module

    monkeypatch.setattr(agent_module.plugin_config, "chat_api_format", "vertex")
    monkeypatch.setattr(
        agent_module.plugin_config,
        "chat_base_url",
        "https://openrouter.ai/api/v1",
    )

    assert agent_module._chat_request_kwargs("group-1") == {}
