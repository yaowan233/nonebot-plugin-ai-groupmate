import json
from copy import deepcopy
from collections import deque

import pytest
from langchain_core.messages import AIMessage

from evals.runner import (
    run_case,
    build_report,
    load_dataset,
    build_judge_request_messages,
)


class _ScriptedModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = deque(responses)
        self.bound_tools: list[tuple[str, ...]] = []

    def bind_tools(self, tools):
        self.bound_tools.append(tuple(tool.name for tool in tools))
        return self

    async def ainvoke(self, messages):
        return self.responses.popleft()


def _case(case_id: str):
    dataset = load_dataset()
    return next(case for case in dataset["cases"] if case["id"] == case_id)


@pytest.mark.asyncio
async def test_runner_executes_skill_fixture_and_scores_trace():
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_agent_skill",
                        "args": {"skill_name": "search_context_tools"},
                        "id": "load-1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_expression",
                        "args": {"expression": "128*4+48"},
                        "id": "calc-1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "结果是 560", "next_step": "end"},
                        "id": "reply-1",
                    }
                ],
            ),
        ]
    )

    result = await run_case(_case("single_tool_001"), model)

    assert result["error"] is None
    assert result["evaluation"]["passed"] is True
    assert result["evaluation"]["score"] == 100
    assert result["active_skills"] == ["search_context_tools"]
    assert [trace["name"] for trace in result["tool_traces"]] == [
        "load_agent_skill",
        "calculate_expression",
        "reply_user",
    ]
    calculation_result = json.loads(result["tool_traces"][1]["result"])
    assert calculation_result["status"] == "succeeded"
    assert calculation_result["data"]["result"] == "560"
    assert result["response_text"] == "结果是 560"


@pytest.mark.asyncio
async def test_runner_marks_timeout_after_dispatch_as_delivery_unknown_once():
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "收到", "next_step": "end"},
                        "id": "reply-1",
                    }
                ],
            ),
        ]
    )

    result = await run_case(
        _case("failure_recovery_003"),
        model,
        tool_timeout_seconds=0.01,
    )

    reply_traces = [
        trace for trace in result["tool_traces"] if trace["name"] == "reply_user"
    ]
    assert len(reply_traces) == 1
    assert reply_traces[0]["status"] == "timeout_after_dispatch"
    assert reply_traces[0]["dispatched"] is True
    assert result["llm_call_count"] == 1
    assert result["evaluation"]["observed_outcome"] == "delivery_unknown"
    assert not result["evaluation"]["hard_failures"]


@pytest.mark.asyncio
async def test_runner_uses_expected_llm_limit_for_scoring_not_execution_cutoff():
    case = deepcopy(_case("single_tool_001"))
    case["expected"]["max_llm_calls"] = 1
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_agent_skill",
                    "args": {"skill_name": "search_context_tools"},
                    "id": "load-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "calculate_expression",
                    "args": {"expression": "128*4+48"},
                    "id": "calc-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "reply_user",
                    "args": {"content": "结果是 560", "next_step": "end"},
                    "id": "reply-1",
                }],
            ),
        ]
    )

    result = await run_case(case, model)

    assert result["response_text"] == "结果是 560"
    assert result["llm_call_count"] == 3
    assert result["evaluation"]["limits"]["llm_within_limit"] is False
    assert result["evaluation"]["components"]["efficiency"] == 5


@pytest.mark.asyncio
async def test_runner_accepts_any_declared_allowed_outcome():
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "reply_user",
                    "args": {"content": "确实", "next_step": "end"},
                    "id": "reply-1",
                }],
            )
        ]
    )

    result = await run_case(_case("conversation_005"), model)

    assert result["evaluation"]["passed"] is True
    assert result["evaluation"]["observed_outcome"] == "reply"
    assert result["evaluation"]["expected_outcomes"] == ["reply", "silent"]


@pytest.mark.asyncio
async def test_runner_uses_optional_semantic_judge():
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reply_user",
                        "args": {"content": "先歇十分钟，别硬撑", "next_step": "end"},
                        "id": "reply-1",
                    }
                ],
            )
        ]
    )
    judge = _ScriptedModel(
        [
            AIMessage(
                content=(
                    '{"semantic_checks":[{"index":0,"passed":true,'
                    '"reason":"回应简短且支持"}],"rubric_score":1,'
                    '"rubric_reason":"符合要求","critical_failure":false,'
                    '"critical_failure_reason":""}'
                )
            )
        ]
    )

    result = await run_case(
        _case("conversation_004"),
        model,
        judge_model=judge,
    )

    assert result["judge"]["semantic_checks"][0]["passed"] is True
    assert result["evaluation"]["judge_used"] is True
    assert result["evaluation"]["components"]["response_quality"] == 25


def test_judge_request_includes_authoritative_tool_results():
    execution = {
        "response_text": "版本号是 Nova 5.2",
        "tool_traces": [
            {
                "name": "search_web",
                "args": {"query": "Nova 发布会版本号"},
                "status": "ok",
                "dispatched": True,
                "result": "发布会公告：正式版本号为 Nova 5.2。",
            }
        ],
    }

    messages = build_judge_request_messages(_case("multi_tool_008"), execution)
    prompt = messages[1]["content"]
    payload = json.loads(prompt.rsplit("\n\n", 1)[1])

    assert "不得判定为捏造" in prompt
    assert payload["tool_calls"] == [
        {
            "name": "search_web",
            "args": {"query": "Nova 发布会版本号"},
            "status": "ok",
            "dispatched": True,
            "result": "发布会公告：正式版本号为 Nova 5.2。",
        }
    ]


@pytest.mark.asyncio
async def test_annual_report_can_use_two_progressive_replies():
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_agent_skill",
                    "args": {"skill_name": "profile_memory_tools"},
                    "id": "load-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "generate_and_send_annual_report",
                    "args": {},
                    "id": "report-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "reply_user",
                    "args": {
                        "content": "今年发了1840条，最常聊摄影、跑步、咖啡。",
                        "next_step": "continue",
                    },
                    "id": "reply-1",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "reply_user",
                    "args": {"content": "最活跃的是10月。", "next_step": "end"},
                    "id": "reply-2",
                }],
            ),
        ]
    )

    result = await run_case(_case("multi_tool_005"), model)

    assert result["evaluation"]["passed"] is True
    assert result["evaluation"]["score"] == 100
    assert result["evaluation"]["side_effect_checks"] == [
        {
            "name": "reply_user",
            "count": 2,
            "bounds": {"min": 1, "max": 2},
            "passed": True,
        }
    ]


def test_report_aggregates_category_metrics():
    dataset = load_dataset()
    result = {
        "category": "single_tool",
        "duration_ms": 120,
        "llm_call_count": 2,
        "tool_call_count": 2,
        "total_tokens": 100,
        "judge": None,
        "evaluation": {
            "score": 90,
            "passed": True,
            "hard_failures": [],
            "components": {"tools": 25, "efficiency": 10},
        },
    }

    report = build_report(
        dataset,
        [result],
        model_name="test-model",
        judge_model_name=None,
    )

    assert report["summary"]["pass_rate"] == 1
    assert report["summary"]["average_score"] == 90
    assert report["summary"]["total_tokens"] == 100
    assert report["summary"]["average_components"]["tools"] == 25
    assert report["categories"]["single_tool"]["runs"] == 1
