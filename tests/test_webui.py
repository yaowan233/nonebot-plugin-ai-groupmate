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
            "result_truncations": 2,
            "side_effect_deduplications": 1,
        },
        "by_session": [],
        "agent_by_session": [
            {
                "session_id": "group-1",
                "requests": 2,
                "agent_llm_calls": 3,
                "agent_tool_calls": 4,
                "agent_avg_duration_ms": 1500,
                "agent_tool_timeouts": 1,
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
    assert "Token 用量明细" in html
    assert "group-1" in html


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
        agent_result_truncations=2,
        agent_side_effect_deduplications=1,
    )

    row = db_session.rows[0]
    assert row.agent_llm_calls == 2
    assert row.agent_tool_calls == 3
    assert row.agent_duration_ms == 1500
    assert row.agent_tool_timeouts == 1
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
        agent_tool_timeouts=0,
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
    assert data["agent_by_session"][0]["session_id"] == "agent-group"
    assert [row["session_id"] for row in data["agent_recent"]] == ["agent-group"]
