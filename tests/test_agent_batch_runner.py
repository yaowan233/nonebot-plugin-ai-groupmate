import json
from pathlib import Path

import pytest

from evals.runner import DATASET_PATH
from evals.batch_runner import prepare_batch_run, consume_batch_results


def _read_request(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def _write_result(
    path: Path,
    request: dict,
    message: dict,
    *,
    total_tokens: int = 100,
) -> Path:
    row = {
        "id": f"batch-{request['custom_id']}",
        "custom_id": request["custom_id"],
        "response": {
            "status_code": 200,
            "request_id": f"request-{request['custom_id']}",
            "body": {
                "model": request["body"]["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                "usage": {
                    "prompt_tokens": total_tokens - 10,
                    "completion_tokens": 10,
                    "total_tokens": total_tokens,
                },
            },
        },
        "error": None,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


@pytest.mark.asyncio
async def test_batch_runner_advances_agent_in_dependent_waves(tmp_path):
    state, paths = prepare_batch_run(
        dataset_path=DATASET_PATH,
        output_dir=tmp_path,
        model="qwen-test",
        case_ids={"single_tool_001"},
        enable_thinking=False,
    )
    first = _read_request(paths[0])

    assert state["config"]["max_llm_calls"] == 8
    assert first["custom_id"] == "agent__single_tool_001__r1__t1"
    assert first["method"] == "POST"
    assert first["url"] == "/v1/chat/completions"
    assert first["body"]["model"] == "qwen-test"
    assert first["body"]["enable_thinking"] is False
    first_tools = {tool["function"]["name"] for tool in first["body"]["tools"]}
    assert "load_agent_skill" in first_tools
    assert "calculate_expression" not in first_tools

    first_result = _write_result(
        tmp_path / "result-1.jsonl",
        first,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "load_agent_skill",
                    {"skill_name": "search_context_tools"},
                    "load-1",
                )
            ],
        },
    )
    progress = await consume_batch_results(
        state_path=Path(state["state_path"]),
        result_paths=[first_result],
    )
    assert progress["phase"] == "agent"
    second = _read_request(progress["request_files"][0])
    second_tools = {tool["function"]["name"] for tool in second["body"]["tools"]}
    assert "calculate_expression" in second_tools
    assert second["body"]["messages"][-1]["role"] == "tool"

    second_result = _write_result(
        tmp_path / "result-2.jsonl",
        second,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "calculate_expression",
                    {"expression": "128*4+48"},
                    "calc-1",
                )
            ],
        },
    )
    progress = await consume_batch_results(
        state_path=Path(state["state_path"]),
        result_paths=[second_result],
    )
    third = _read_request(progress["request_files"][0])
    assert third["body"]["messages"][-1]["content"] == "560"

    third_result = _write_result(
        tmp_path / "result-3.jsonl",
        third,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "reply_user",
                    {"content": "结果是 560", "next_step": "end"},
                    "reply-1",
                )
            ],
        },
    )
    completed = await consume_batch_results(
        state_path=Path(state["state_path"]),
        result_paths=[third_result],
    )

    assert completed["phase"] == "complete"
    report = json.loads(completed["report_path"].read_text(encoding="utf-8"))
    assert report["inference_mode"] == "batch"
    assert report["summary"]["passed"] == 1
    assert report["results"][0]["response_text"] == "结果是 560"
    assert report["results"][0]["evaluation"]["score"] == 100


@pytest.mark.asyncio
async def test_batch_runner_stops_after_timed_out_side_effect(tmp_path):
    state, paths = prepare_batch_run(
        dataset_path=DATASET_PATH,
        output_dir=tmp_path,
        model="qwen-test",
        case_ids={"failure_recovery_003"},
    )
    request = _read_request(paths[0])
    result_path = _write_result(
        tmp_path / "result.jsonl",
        request,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "reply_user",
                    {"content": "收到", "next_step": "end"},
                    "reply-1",
                )
            ],
        },
    )

    completed = await consume_batch_results(
        state_path=Path(state["state_path"]),
        result_paths=[result_path],
    )

    assert completed["phase"] == "complete"
    report = json.loads(completed["report_path"].read_text(encoding="utf-8"))
    run = report["results"][0]
    assert run["llm_call_count"] == 1
    assert run["evaluation"]["observed_outcome"] == "delivery_unknown"


@pytest.mark.asyncio
async def test_batch_runner_emits_and_consumes_judge_wave(tmp_path):
    state, paths = prepare_batch_run(
        dataset_path=DATASET_PATH,
        output_dir=tmp_path,
        model="agent-model",
        judge=True,
        judge_model="judge-model",
        case_ids={"conversation_004"},
        enable_thinking=True,
        thinking_budget=128,
    )
    agent_request = _read_request(paths[0])
    agent_result = _write_result(
        tmp_path / "agent-result.jsonl",
        agent_request,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "reply_user",
                    {"content": "先歇十分钟，别硬撑", "next_step": "end"},
                    "reply-1",
                )
            ],
        },
    )
    progress = await consume_batch_results(
        state_path=Path(state["state_path"]),
        result_paths=[agent_result],
    )

    assert progress["phase"] == "judge"
    judge_request = _read_request(progress["request_files"][0])
    assert judge_request["body"]["model"] == "judge-model"
    assert judge_request["body"]["enable_thinking"] is True
    assert judge_request["body"]["thinking_budget"] == 128
    assert "tools" not in judge_request["body"]

    judge_content = json.dumps(
        {
            "semantic_checks": [
                {"index": 0, "passed": True, "reason": "回应简短且支持"}
            ],
            "rubric_score": 1,
            "rubric_reason": "符合要求",
            "critical_failure": False,
            "critical_failure_reason": "",
        },
        ensure_ascii=False,
    )
    judge_result = _write_result(
        tmp_path / "judge-result.jsonl",
        judge_request,
        {"role": "assistant", "content": judge_content},
    )
    completed = await consume_batch_results(
        state_path=Path(state["state_path"]),
        result_paths=[judge_result],
    )

    report = json.loads(completed["report_path"].read_text(encoding="utf-8"))
    assert report["results"][0]["evaluation"]["judge_used"] is True
    assert report["results"][0]["evaluation"]["components"]["response_quality"] == 25
