import json


def test_tool_result_success_uses_versioned_envelope():
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    result = json.loads(tool_success(
        "operation_completed",
        "操作完成。",
        data={"value": 42},
        delivery_state="completed",
    ))

    assert result == {
        "schema_version": 1,
        "ok": True,
        "status": "succeeded",
        "reason_code": "operation_completed",
        "message": "操作完成。",
        "retryable": False,
        "delivery_state": "completed",
        "data": {"value": 42},
    }


def test_tool_result_failure_keeps_business_data_nested():
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_failure

    result = json.loads(tool_failure(
        "target_ambiguous",
        "目标不明确。",
        retryable=True,
        data={"candidates": ["1", "2"]},
        delivery_state="not_attempted",
    ))

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["retryable"] is True
    assert result["data"] == {"candidates": ["1", "2"]}
    assert "candidates" not in result


def test_tool_result_parser_normalizes_legacy_sent_status():
    from nonebot_plugin_ai_groupmate.agent.tool_results import (
        parse_tool_result,
        tool_result_status,
    )

    content = '{"status":"sent","message":"legacy extension"}'

    assert tool_result_status(content) == "succeeded"
    assert parse_tool_result(content) == {
        "status": "succeeded",
        "message": "legacy extension",
        "ok": True,
    }


def test_tool_result_parser_rejects_unstructured_text():
    from nonebot_plugin_ai_groupmate.agent.tool_results import (
        parse_tool_result,
        tool_result_status,
    )

    assert parse_tool_result("plain extension result") is None
    assert tool_result_status("plain extension result") is None
