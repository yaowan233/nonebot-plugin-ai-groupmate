import json

import pytest


@pytest.mark.asyncio
async def test_schedule_message_returns_protocol_and_job_metadata(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    calls: list[tuple[object, str, dict]] = []

    def fake_add_job(func, trigger: str, **kwargs):
        calls.append((func, trigger, kwargs))

    monkeypatch.setattr(schedule_tools.scheduler, "add_job", fake_add_job)
    tool = schedule_tools.create_schedule_message_tool(
        "group-1",
        None,
        is_private=False,
        bot_id="bot-1",
        bot_name="小夏",
    )

    result = json.loads(await tool.ainvoke({
        "content": "起来活动一下",
        "delay_minutes": 1,
    }))

    assert result["status"] == "succeeded"
    assert result["reason_code"] == "schedule_created"
    assert result["delivery_state"] == "completed"
    assert result["data"]["task_type"] == "message"
    assert result["data"]["job_id"].startswith("ai_groupmate_schedule_group-1_")
    assert len(calls) == 1
    assert calls[0][1] == "date"


@pytest.mark.asyncio
async def test_schedule_agent_rejects_invalid_delay_with_protocol():
    from nonebot_plugin_ai_groupmate.agent import schedule_tools

    tool = schedule_tools.create_schedule_agent_task_tool(
        "group-1",
        None,
        is_private=False,
        bot_id="bot-1",
        run_agent_task=lambda: None,
    )

    result = json.loads(await tool.ainvoke({
        "task": "查天气",
        "delay_minutes": 0,
        "delay_hours": 0,
    }))

    assert result["status"] == "failed"
    assert result["reason_code"] == "invalid_delay"
    assert result["delivery_state"] == "not_attempted"
