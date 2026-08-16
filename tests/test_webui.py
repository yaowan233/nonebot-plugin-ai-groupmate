from types import SimpleNamespace
from datetime import datetime

import pytest


def test_dashboard_renders_agent_metrics():
    from nonebot_plugin_ai_groupmate.webui import _render_dashboard
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    data = {
        "days": 7,
        "filters": {"session_id": "", "user_id": ""},
        "total": {
            "requests": 2,
            "total_tokens": 300,
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "cached_tokens": 50,
            "cache_creation_tokens": 0,
            "estimated_cost": 0.001,
        },
        "agent": {
            "runs": 2,
            "llm_calls": 3,
            "tool_calls": 4,
            "duration_ms": 3000,
            "avg_duration_ms": 1500,
            "tool_timeouts": 1,
            "tool_timeout_tools": {"search_web": 1},
            "result_truncations": 2,
            "side_effect_deduplications": 1,
        },
        "by_session": [
            {
                "session_id": "group-1",
                "session_type": "group",
                "requests": 2,
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "cached_tokens": 50,
                "cache_creation_tokens": 0,
                "total_tokens": 300,
                "estimated_cost": 0.001,
            }
        ],
        "agent_by_session": [
            {
                "session_id": "group-1",
                "requests": 2,
                "agent_llm_calls": 3,
                "agent_tool_calls": 4,
                "agent_avg_duration_ms": 1500,
                "agent_tool_timeouts": 1,
                "agent_tool_timeout_tools": {"search_web": 1},
                "agent_result_truncations": 2,
                "agent_side_effect_deduplications": 1,
            }
        ],
        "agent_recent": [
            {
                "created_at": "2026-07-10T16:00:00",
                "session_id": "group-1",
                "agent_llm_calls": 3,
                "agent_tool_calls": 4,
                "agent_duration_ms": 1500,
                "agent_tool_timeouts": 1,
                "agent_tool_timeout_tools": {"search_web": 1},
                "agent_result_truncations": 2,
                "agent_side_effect_deduplications": 1,
            }
        ],
        "by_user": [],
        "by_model": [],
        "recent": [
            {
                "created_at": "2026-07-10T16:00:00",
                "session_id": "group-1",
                "user_id": "user-1",
                "user_name": "tester",
                "model": "test-model",
                "total_tokens": 300,
                "cached_tokens": 50,
                "cache_creation_tokens": 0,
                "estimated_cost": 0.001,
                "agent_llm_calls": 3,
                "agent_tool_calls": 4,
                "agent_duration_ms": 1500,
                "agent_tool_timeouts": 1,
                "agent_tool_timeout_tools": {"search_web": 1},
                "agent_result_truncations": 2,
                "agent_side_effect_deduplications": 1,
            }
        ],
    }

    html = _render_dashboard(
        data,
        path="/ai-groupmate/usage",
        token=None,
        config=ScopedConfig(),
    )

    assert "运行与用量概览" in html
    assert "Agent 运行" in html
    assert "已观测运行" in html
    assert "1.50 s" in html
    assert "分群对比" in html
    assert "分群效果对比" in html
    assert 'data-tab="groups"' in html
    assert 'data-tab="overview"' in html
    assert 'data-tab="recent"' in html
    assert 'data-tab="usage"' in html
    assert "LLM / 工具显示每次运行的平均调用数" in html
    assert "1.50 / 2.00" in html
    assert "group-1" in html
    assert "search_web×1" in html
    assert "localStorage.setItem" in html
    assert 'filterForm?.addEventListener("submit"' in html


@pytest.mark.asyncio
async def test_record_token_usage_stores_agent_metrics():
    from nonebot_plugin_ai_groupmate.usage import record_token_usage

    class _Session:
        def __init__(self):
            self.rows = []

        def add(self, row):
            self.rows.append(row)

    db_session = _Session()
    await record_token_usage(
        db_session,  # type: ignore[arg-type]
        session_id="group-1",
        session_type="group",
        user_id="user-1",
        user_name="tester",
        model="test-model",
        request_id="request-1",
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=20,
        cache_creation_tokens=0,
        total_tokens=150,
        estimated_cost=0.01,
        agent_llm_calls=2,
        agent_tool_calls=3,
        agent_duration_ms=1500,
        agent_tool_timeouts=1,
        agent_tool_timeout_tools=["search_web"],
        agent_result_truncations=2,
        agent_side_effect_deduplications=1,
    )

    row = db_session.rows[0]
    assert row.agent_llm_calls == 2
    assert row.agent_tool_calls == 3
    assert row.agent_duration_ms == 1500
    assert row.agent_tool_timeouts == 1
    assert row.agent_tool_timeout_tools == ["search_web"]
    assert row.agent_result_truncations == 2
    assert row.agent_side_effect_deduplications == 1


@pytest.mark.asyncio
async def test_dashboard_excludes_pre_metrics_rows_from_agent_statistics():
    from nonebot_plugin_ai_groupmate.usage import get_usage_dashboard_data
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    legacy_row = SimpleNamespace(
        created_at=datetime(2026, 7, 10),
        session_id="legacy-group",
        session_type="group",
        user_id="legacy-user",
        user_name="legacy",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=0,
        cache_creation_tokens=0,
        total_tokens=120,
        estimated_cost=0.01,
        agent_llm_calls=0,
        agent_tool_calls=0,
        agent_duration_ms=0,
        agent_tool_timeouts=0,
        agent_result_truncations=0,
        agent_side_effect_deduplications=0,
    )
    agent_row = SimpleNamespace(
        created_at=datetime(2026, 7, 11),
        session_id="agent-group",
        session_type="group",
        user_id="agent-user",
        user_name="agent",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=0,
        cache_creation_tokens=0,
        total_tokens=120,
        estimated_cost=0.01,
        agent_llm_calls=1,
        agent_tool_calls=1,
        agent_duration_ms=1_500,
        agent_tool_timeouts=2,
        agent_tool_timeout_tools=["search_web", "search_web"],
        agent_result_truncations=0,
        agent_side_effect_deduplications=0,
    )

    class _Result:
        def __init__(self, rows):
            self.rows = rows

        def scalars(self):
            return self

        def all(self):
            return self.rows

    class _Session:
        async def execute(self, _statement):
            return _Result([agent_row, legacy_row])

    data = await get_usage_dashboard_data(
        _Session(),  # type: ignore[arg-type]
        config=ScopedConfig(),
    )

    assert data["total"]["requests"] == 2
    assert data["agent"]["runs"] == 1
    assert data["agent"]["avg_duration_ms"] == 1_500
    assert data["agent"]["tool_timeout_tools"] == {"search_web": 2}
    assert data["agent_by_session"][0]["session_id"] == "agent-group"
    assert data["agent_by_session"][0]["agent_tool_timeout_tools"] == {
        "search_web": 2
    }
    assert [row["session_id"] for row in data["agent_recent"]] == ["agent-group"]


def test_settings_page_groups_all_fields_and_never_renders_secrets():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.settings_ui import (
        SETTING_GROUPS,
        render_settings_page,
    )
    from nonebot_plugin_ai_groupmate.runtime_config import (
        CONFIGURABLE_FIELDS,
    )

    config = ScopedConfig(
        llm_api_key="sk-do-not-render",
        qdrant_api_key="qdrant-do-not-render",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dimension=1536,
    )
    html = render_settings_page(
        config,
        ScopedConfig(),
        overridden_fields={"llm_api_key", "reply_probability"},
        pending_restart_fields={"qdrant_uri"},
        dashboard_path="/ai-groupmate/usage",
        settings_path="/ai-groupmate/usage/settings",
    )

    grouped_fields = {
        field_name
        for _, _, fields in SETTING_GROUPS
        for field_name in fields
    }
    assert grouped_fields == CONFIGURABLE_FIELDS
    assert "sk-do-not-render" not in html
    assert "qdrant-do-not-render" not in html
    assert "Qwen/Qwen3-Embedding-0.6B" in html
    assert "1536" in html
    assert 'data-setting="embedding_model"' in html
    assert 'data-setting="embedding_dimension"' in html
    assert "Qdrant 服务端须为 1.16 或更高版本" in html
    assert "Qdrant、Embedding 与 Rerank 连接配置重启后生效" in html
    assert "已配置，留空保持不变" in html
    assert "网页覆盖" in html
    assert "等待重启" in html
    assert "测试连接" in html


def test_settings_page_renders_unset_dimension_as_blank():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.settings_ui import render_settings_page

    html = render_settings_page(
        ScopedConfig(embedding_dimension=None),
        ScopedConfig(),
        overridden_fields=set(),
        pending_restart_fields=set(),
        dashboard_path="/ai-groupmate/usage",
        settings_path="/ai-groupmate/usage/settings",
    )
    # 未配置维度渲染为空白，不出现字面量 "None"。
    assert 'data-setting="embedding_dimension"' in html
    assert 'value="None"' not in html


def test_blank_embedding_dimension_submission_is_treated_as_unset():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.runtime_config import (
        preview_runtime_config_update,
    )

    # WebUI 提交空白输入时值为 ""，应转换为 None（保持未配置语义）。
    candidate = ScopedConfig.model_validate({"embedding_dimension": ""})
    assert candidate.embedding_dimension is None

    candidate, _overrides, _changed = preview_runtime_config_update({
        "embedding_dimension": "",
    })
    assert candidate.embedding_dimension is None


def test_runtime_config_update_is_validated_and_bootstrap_fields_are_blocked():
    from pydantic import ValidationError

    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.runtime_config import (
        CONFIGURABLE_FIELDS,
        RESTART_REQUIRED_FIELDS,
        preview_runtime_config_update,
    )

    candidate, overrides, changed = preview_runtime_config_update({
        "reply_probability": "0.25",
        "agent_max_llm_calls": "6",
    })
    assert candidate.reply_probability == 0.25
    assert candidate.agent_max_llm_calls == 6
    assert overrides["reply_probability"] == 0.25
    assert {"reply_probability", "agent_max_llm_calls"} <= changed

    with pytest.raises(ValueError, match="不支持通过网页修改"):
        preview_runtime_config_update({"usage_webui_token": "new-token"})
    candidate, overrides, changed = preview_runtime_config_update({
        "embedding_model": "custom-model",
        "embedding_dimension": "1536",
    })
    assert candidate.embedding_model == "custom-model"
    assert candidate.embedding_dimension == 1536
    assert overrides["embedding_model"] == "custom-model"
    assert overrides["embedding_dimension"] == 1536
    assert {"embedding_model", "embedding_dimension"} <= changed
    with pytest.raises(ValueError, match="不是密钥配置"):
        preview_runtime_config_update({}, clear_secrets={"usage_webui_token"})
    with pytest.raises(ValidationError):
        ScopedConfig.model_validate({"meme_embedding_mode": "invalid"})
    with pytest.raises(ValidationError):
        ScopedConfig.model_validate({"embedding_dimension": 0})

    assert {"embedding_model", "embedding_dimension"} <= CONFIGURABLE_FIELDS
    assert {
        "embedding_model",
        "embedding_dimension",
        "meme_embedding_mode",
        "qwen_token",
    } <= RESTART_REQUIRED_FIELDS


def test_connection_error_redacts_configured_secrets(monkeypatch):
    from nonebot_plugin_ai_groupmate.webui import _safe_connection_error
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    config = ScopedConfig(llm_api_key="sk-secret-value")
    detail = _safe_connection_error(
        RuntimeError("request failed with sk-secret-value"),
        config,
    )
    assert "sk-secret-value" not in detail
    assert "***" in detail

    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "google-secret-value")
    detail = _safe_connection_error(
        RuntimeError("request failed with google-secret-value"),
        config,
    )
    assert "google-secret-value" not in detail
    assert "***" in detail


def test_settings_routes_require_login_and_apply_updates(monkeypatch):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import nonebot_plugin_ai_groupmate.webui as webui_module
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    app = FastAPI()
    config = ScopedConfig(usage_webui_token="admin-password")
    changed_callbacks: list[set[str]] = []

    @asynccontextmanager
    async def fake_session():
        yield object()

    async def fake_save(_session, updates, *, clear_secrets=None):
        assert updates == {"reply_probability": "0.2"}
        assert clear_secrets == {"llm_api_key"}
        return {"reply_probability", "llm_api_key"}, set()

    monkeypatch.setattr(
        webui_module,
        "get_driver",
        lambda: SimpleNamespace(server_app=app),
    )
    monkeypatch.setattr(webui_module, "get_session", fake_session)
    monkeypatch.setattr(webui_module, "save_runtime_config_updates", fake_save)
    webui_module.register_usage_webui(
        config,
        on_config_change=changed_callbacks.append,
    )

    with TestClient(app) as client:
        login_page = client.get("/ai-groupmate/usage/settings")
        assert login_page.status_code == 200
        assert "请输入 WebUI 管理密码" in login_page.text
        assert "admin-password" not in login_page.text

        assert client.post(
            "/ai-groupmate/usage/settings/login",
            json={"token": "wrong"},
        ).status_code == 401
        login = client.post(
            "/ai-groupmate/usage/settings/login",
            json={"token": "admin-password"},
        )
        assert login.status_code == 200
        assert login.cookies.get(webui_module.SETTINGS_COOKIE_NAME)

        settings_page = client.get("/ai-groupmate/usage/settings")
        assert settings_page.status_code == 200
        assert "环境变量作为默认值" in settings_page.text

        saved = client.post(
            "/ai-groupmate/usage/settings/api",
            json={
                "updates": {"reply_probability": "0.2"},
                "clear_secrets": ["llm_api_key"],
            },
        )
        assert saved.status_code == 200
        assert saved.json()["restart_required"] is False
        assert changed_callbacks == [{"reply_probability", "llm_api_key"}]
