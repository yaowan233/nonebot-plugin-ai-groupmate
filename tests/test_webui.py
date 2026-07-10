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

    assert "Agent 运行" in html
    assert "1.50 s" in html
    assert "结果截断 / 去重" in html
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
