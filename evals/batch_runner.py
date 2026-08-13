from __future__ import annotations

import re
import json
import asyncio
import hashlib
import argparse
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

from pydantic import ValidationError
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from .runner import (
    TOOL_SPECS,
    DATASET_PATH,
    SIDE_EFFECT_TOOL_NAMES,
    DEFAULT_EVAL_MAX_LLM_CALLS,
    FixtureToolRuntime,
    build_report,
    load_dataset,
    select_cases,
    _response_text,
    _build_messages,
    score_execution,
    _build_system_prompt,
    parse_judge_response,
    _fixture_tool_failure,
    _fixture_tool_skipped,
    _parse_fixture_tool_result,
    _fixture_tool_result_status,
    build_judge_request_messages,
)

MAX_REQUESTS_PER_FILE = 50_000
MAX_FILE_BYTES = 500_000_000
MAX_LINE_BYTES = 1_000_000
MAX_TOOL_COUNT = 20
MAX_REPLY_COUNT = 5


def _dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "batch"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_jsonl_shards(
    requests: list[dict[str, Any]],
    *,
    output_dir: Path,
    label: str,
) -> list[Path]:
    if not requests:
        raise ValueError("没有需要写入的批量请求")
    custom_ids = [request["custom_id"] for request in requests]
    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError("批量请求 custom_id 不唯一")
    if any(len(custom_id) > 256 for custom_id in custom_ids):
        raise ValueError("批量请求 custom_id 超过 256 个字符")

    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[bytes]] = []
    current: list[bytes] = []
    current_size = 0
    for request in requests:
        line = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(
                f"请求 {request['custom_id']} 大小为 {len(line)} 字节，超过单行 1 MB 限制"
            )
        if current and (
            len(current) >= MAX_REQUESTS_PER_FILE
            or current_size + len(line) > MAX_FILE_BYTES
        ):
            shards.append(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line)
    if current:
        shards.append(current)

    paths: list[Path] = []
    for index, shard in enumerate(shards, start=1):
        suffix = f"-part-{index:03d}" if len(shards) > 1 else ""
        path = output_dir / f"{label}{suffix}.jsonl"
        path.write_bytes(b"".join(shard))
        paths.append(path)
    return paths


def _openai_history(case: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(case)}
    ]
    for message in _build_messages(case):
        history.append(
            {
                "role": "assistant" if isinstance(message, AIMessage) else "user",
                "content": message.content,
            }
        )
    return history


def _new_session(case: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    key = f"{case['id']}__r{repeat_index}"
    return {
        "key": key,
        "case_id": case["id"],
        "category": case["category"],
        "title": case["title"],
        "repeat": repeat_index,
        "status": "pending",
        "messages": _openai_history(case),
        "active_skills": [],
        "reply_count": 0,
        "tool_count": 0,
        "llm_call_count": 0,
        "total_tokens": 0,
        "tool_timeout_count": 0,
        "tool_timeout_names": [],
        "side_effect_duplicate_count": 0,
        "completed_side_effect_keys": [],
        "llm_calls": [],
        "tool_traces": [],
        "judge": None,
        "error": None,
    }


def _tool_bundle(
    case: dict[str, Any],
    session: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[FixtureToolRuntime, list[Any], dict[str, list[Any]]]:
    runtime = FixtureToolRuntime(case, timeout_seconds=timeout_seconds)
    runtime.traces = session["tool_traces"]
    runtime.call_counts = Counter(trace["name"] for trace in runtime.traces)
    _, base_tools, tools_by_skill = runtime.build_tools()
    return runtime, base_tools, tools_by_skill


def _visible_tools(
    base_tools: list[Any],
    tools_by_skill: dict[str, list[Any]],
    active_skills: list[str],
) -> list[Any]:
    tools = list(base_tools)
    known = {tool.name for tool in tools}
    for skill_name in active_skills:
        for tool in tools_by_skill.get(skill_name, []):
            if tool.name not in known:
                tools.append(tool)
                known.add(tool.name)
    return tools


def _request_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Any] | None,
    temperature: float,
    enable_thinking: bool,
    thinking_budget: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
    }
    if thinking_budget is not None:
        if not enable_thinking:
            raise ValueError("thinking_budget 只能在 enable_thinking=true 时使用")
        body["thinking_budget"] = thinking_budget
    if tools:
        body["tools"] = [convert_to_openai_tool(tool) for tool in tools]
        body["tool_choice"] = "auto"
    return body


def _agent_requests(
    state: dict[str, Any],
    cases_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    config = state["config"]
    requests: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for session in state["sessions"]:
        if session["status"] != "pending":
            continue
        case = cases_by_id[session["case_id"]]
        _, base_tools, tools_by_skill = _tool_bundle(
            case,
            session,
            timeout_seconds=config["tool_timeout_seconds"],
        )
        visible_tools = _visible_tools(
            base_tools,
            tools_by_skill,
            session["active_skills"],
        )
        turn = session["llm_call_count"] + 1
        custom_id = f"agent__{session['key']}__t{turn}"
        mapping[custom_id] = session["key"]
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _request_body(
                    model=config["model"],
                    messages=session["messages"],
                    tools=visible_tools,
                    temperature=config["temperature"],
                    enable_thinking=config["enable_thinking"],
                    thinking_budget=config["thinking_budget"],
                ),
            }
        )
    return requests, mapping


def _execution_from_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": session["case_id"],
        "category": session["category"],
        "title": session["title"],
        "repeat": session["repeat"],
        "duration_ms": 0,
        "llm_call_count": session["llm_call_count"],
        "tool_call_count": session["tool_count"],
        "total_tokens": session["total_tokens"],
        "tool_timeout_count": session["tool_timeout_count"],
        "tool_timeout_names": session["tool_timeout_names"],
        "side_effect_duplicate_count": session["side_effect_duplicate_count"],
        "active_skills": session["active_skills"],
        "response_text": _response_text(session["tool_traces"]),
        "llm_calls": session["llm_calls"],
        "tool_traces": session["tool_traces"],
        "error": session["error"],
    }


def _judge_requests(
    state: dict[str, Any],
    cases_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    config = state["config"]
    requests: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for session in state["sessions"]:
        if session["error"]:
            continue
        custom_id = f"judge__{session['key']}"
        mapping[custom_id] = session["key"]
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _request_body(
                    model=config["judge_model"],
                    messages=build_judge_request_messages(
                        cases_by_id[session["case_id"]],
                        _execution_from_session(session),
                    ),
                    tools=None,
                    temperature=0,
                    enable_thinking=config["enable_thinking"],
                    thinking_budget=config["thinking_budget"],
                ),
            }
        )
    return requests, mapping


def _emit_requests(
    state: dict[str, Any],
    requests: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    label: str,
) -> list[Path]:
    output_dir = Path(state["output_dir"])
    paths = _write_jsonl_shards(requests, output_dir=output_dir, label=label)
    state["pending_custom_ids"] = mapping
    state["current_request_files"] = [str(path.resolve()) for path in paths]
    _write_json(Path(state["state_path"]), state)
    return paths


def prepare_batch_run(
    *,
    dataset_path: Path,
    output_dir: Path,
    model: str,
    case_ids: set[str] | None = None,
    categories: set[str] | None = None,
    repeat: int = 1,
    judge: bool = False,
    judge_model: str | None = None,
    temperature: float = 0,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    tool_timeout_seconds: float = 0.05,
    max_llm_calls: int = DEFAULT_EVAL_MAX_LLM_CALLS,
) -> tuple[dict[str, Any], list[Path]]:
    dataset_path = dataset_path.resolve()
    dataset = load_dataset(dataset_path)
    cases = select_cases(dataset, case_ids=case_ids, categories=categories)
    output_dir = output_dir.resolve()
    state_path = output_dir / "state.json"
    state = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_name": dataset["name"],
        "dataset_sha256": _dataset_digest(dataset_path),
        "output_dir": str(output_dir),
        "state_path": str(state_path),
        "phase": "agent",
        "wave": 1,
        "config": {
            "model": model,
            "judge": judge,
            "judge_model": judge_model or model,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
            "tool_timeout_seconds": tool_timeout_seconds,
            "max_llm_calls": max_llm_calls,
        },
        "selected_case_ids": [case["id"] for case in cases],
        "sessions": [
            _new_session(case, repeat_index)
            for case in cases
            for repeat_index in range(1, repeat + 1)
        ],
        "pending_custom_ids": {},
        "current_request_files": [],
        "report_path": None,
    }
    cases_by_id = {case["id"]: case for case in cases}
    requests, mapping = _agent_requests(state, cases_by_id)
    paths = _emit_requests(state, requests, mapping, label="agent-wave-001")
    return state, paths


def _read_jsonl(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str):
                    raise ValueError(f"{path}:{line_number} 缺少 custom_id")
                if custom_id in rows:
                    raise ValueError(f"结果中 custom_id 重复：{custom_id}")
                rows[custom_id] = row
    return rows


def _row_response(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("error"):
        error = row["error"]
        return None, f"{error.get('code', 'batch_error')}: {error.get('message', error)}"
    response = row.get("response")
    if not isinstance(response, dict):
        return None, "批量结果缺少 response"
    if response.get("status_code") != 200:
        return None, f"HTTP {response.get('status_code')}: {response.get('body')}"
    body = response.get("body")
    if not isinstance(body, dict):
        return None, "批量结果缺少 response.body"
    return body, None


def _normalized_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function", {})
        raw_args = function.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {"_invalid_json": raw_args}
        calls.append(
            {
                "name": str(function.get("name", "")),
                "args": args if isinstance(args, dict) else {"_value": args},
                "id": str(raw_call.get("id", "")),
                "raw": raw_call,
            }
        )
    return calls


def _side_effect_key(name: str, args: dict[str, Any]) -> str:
    stable_args = {key: value for key, value in args.items() if key != "next_step"}
    payload = json.dumps(stable_args, ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _tool_result_status(content: str) -> str | None:
    return _fixture_tool_result_status(content)


async def _invoke_fixture_tool(
    runtime: FixtureToolRuntime,
    name: str,
    args: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[str, bool]:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return _fixture_tool_failure("unknown_tool", f"未知工具：{name}。"), False
    try:
        validated = spec.args_schema.model_validate(args).model_dump(exclude_none=True)
    except ValidationError as error:
        return _fixture_tool_failure(
            "invalid_arguments",
            f"工具参数错误：{error}",
        ), False
    try:
        result = await asyncio.wait_for(
            runtime._invoke(name, validated),
            timeout=timeout_seconds,
        )
        return result, False
    except asyncio.TimeoutError:
        return _fixture_tool_failure(
            "tool_timeout",
            "工具执行超时，请根据已有信息决定是否重试或换一种方式。",
            retryable=True,
        ), True


async def _apply_agent_body(
    state: dict[str, Any],
    session: dict[str, Any],
    case: dict[str, Any],
    body: dict[str, Any],
) -> None:
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        session["status"] = "error"
        session["error"] = "批量响应 choices 为空"
        return
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        session["status"] = "error"
        session["error"] = "批量响应 message 无效"
        return
    tool_calls = _normalized_tool_calls(message)
    response_text = str(message.get("content") or "").strip()
    usage = body.get("usage") or {}
    tokens = int(usage.get("total_tokens", 0) or 0)
    config = state["config"]
    runtime, base_tools, tools_by_skill = _tool_bundle(
        case,
        session,
        timeout_seconds=config["tool_timeout_seconds"],
    )
    bound_tools = [
        tool.name
        for tool in _visible_tools(base_tools, tools_by_skill, session["active_skills"])
    ]
    session["llm_call_count"] += 1
    session["total_tokens"] += tokens
    session["llm_calls"].append(
        {
            "duration_ms": 0,
            "bound_tools": bound_tools,
            "tool_calls": [
                {"name": call["name"], "args": call["args"], "id": call["id"]}
                for call in tool_calls
            ],
            "response_text": response_text,
            "tokens": tokens,
            "error": None,
        }
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    if message.get("reasoning_content") is not None:
        assistant_message["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        assistant_message["tool_calls"] = message["tool_calls"]
    session["messages"].append(assistant_message)

    if not tool_calls:
        if response_text:
            session["reply_count"] += 1
            _, timed_out = await _invoke_fixture_tool(
                runtime,
                "reply_user",
                {"content": response_text, "next_step": "end"},
                timeout_seconds=config["tool_timeout_seconds"],
            )
            if timed_out:
                session["tool_timeout_count"] += 1
                session["tool_timeout_names"].append("reply_user")
        session["status"] = "complete"
        return

    reply_this_round = 0
    reply_requires_continuation = False
    called_finish = False
    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        tool_call_id = call["id"]
        if session["tool_count"] >= MAX_TOOL_COUNT:
            session["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _fixture_tool_skipped(
                        "tool_limit_reached",
                        "工具调用已达本轮上限，未执行此调用。",
                    ),
                }
            )
            continue
        session["tool_count"] += 1
        if name == "finish":
            called_finish = True
            session["messages"].append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": ""}
            )
            break

        visible_names = {
            tool.name
            for tool in _visible_tools(
                base_tools,
                tools_by_skill,
                session["active_skills"],
            )
        }
        if name not in visible_names:
            result = _fixture_tool_skipped(
                "tool_not_enabled",
                f"工具 {name} 当前未启用；请先调用 load_agent_skill 读取对应技能。",
            )
            session["messages"].append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": result}
            )
            continue

        if name == "reply_user":
            if reply_this_round >= 1:
                session["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": _fixture_tool_skipped(
                            "reply_limit_reached",
                            "本轮已经发送过消息了。如果想发送更多，请等待下一轮。",
                            delivery_state="not_attempted",
                        ),
                    }
                )
                continue
            reply_this_round += 1
            session["reply_count"] += 1
            reply_requires_continuation = True

        effect_key: str | None = None
        if name in SIDE_EFFECT_TOOL_NAMES:
            effect_key = _side_effect_key(name, args)
            if effect_key in session["completed_side_effect_keys"]:
                session["side_effect_duplicate_count"] += 1
                if name == "reply_user":
                    reply_requires_continuation = False
                session["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": _fixture_tool_skipped(
                            "duplicate_side_effect",
                            "相同的副作用请求已经执行过，已跳过重复执行。",
                            delivery_state="not_attempted",
                        ),
                    }
                )
                continue

        result, timed_out = await _invoke_fixture_tool(
            runtime,
            name,
            args,
            timeout_seconds=config["tool_timeout_seconds"],
        )
        if timed_out:
            session["tool_timeout_count"] += 1
            session["tool_timeout_names"].append(name)
            if effect_key is not None:
                result = _fixture_tool_failure(
                    "tool_timeout",
                    "副作用工具执行超时，投递结果未知；为避免重复操作，"
                    "本轮必须停止且不得重试。",
                    delivery_state="unknown",
                )
                session["completed_side_effect_keys"].append(effect_key)
                called_finish = True
                if name == "reply_user":
                    reply_requires_continuation = False
        status = _tool_result_status(result)
        parsed_result = _parse_fixture_tool_result(result)
        delivery_unknown = (
            effect_key is not None
            and isinstance(parsed_result, dict)
            and parsed_result.get("delivery_state") == "unknown"
        )
        if effect_key is not None and not timed_out and (
            status == "succeeded" or delivery_unknown
        ):
            session["completed_side_effect_keys"].append(effect_key)
        if delivery_unknown:
            called_finish = True
        if name == "reply_user":
            if delivery_unknown:
                reply_requires_continuation = False
            elif timed_out or status == "failed":
                reply_requires_continuation = True
            else:
                reply_requires_continuation = args.get("next_step") == "continue"
        if name == "load_agent_skill" and status != "failed":
            skill_name = str(args.get("skill_name", "")).strip()
            if skill_name in tools_by_skill and skill_name not in session["active_skills"]:
                session["active_skills"].append(skill_name)
        session["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }
        )
        if delivery_unknown:
            break

    should_end = (
        called_finish
        or session["reply_count"] >= MAX_REPLY_COUNT
        or session["tool_count"] >= MAX_TOOL_COUNT
        or session["llm_call_count"] >= config["max_llm_calls"]
        or session["total_tokens"] >= 64_000
        or (reply_this_round > 0 and not reply_requires_continuation)
    )
    session["status"] = "complete" if should_end else "pending"


def _load_run_state(state_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    dataset_path = Path(state["dataset_path"])
    if _dataset_digest(dataset_path) != state["dataset_sha256"]:
        raise ValueError("评测数据集在批次运行期间发生了变化，拒绝继续以避免错配")
    dataset = load_dataset(dataset_path)
    cases_by_id = {
        case["id"]: case
        for case in dataset["cases"]
        if case["id"] in state["selected_case_ids"]
    }
    return state, dataset, cases_by_id


def _finalize_run(
    state: dict[str, Any],
    dataset: dict[str, Any],
    cases_by_id: dict[str, dict[str, Any]],
) -> Path:
    results: list[dict[str, Any]] = []
    for session in state["sessions"]:
        execution = _execution_from_session(session)
        judge = session.get("judge")
        scoring_judge = judge if judge and not judge.get("error") else None
        results.append(
            {
                **execution,
                "judge": judge,
                "evaluation": score_execution(
                    cases_by_id[session["case_id"]],
                    execution,
                    scoring_judge,
                ),
            }
        )
    report = build_report(
        dataset,
        results,
        model_name=state["config"]["model"],
        judge_model_name=(
            state["config"]["judge_model"] if state["config"]["judge"] else None
        ),
    )
    report["inference_mode"] = "batch"
    report["latency_note"] = "批量结果不提供单请求推理耗时，duration_ms/P50/P95 记为 0。"
    report_path = Path(state["output_dir"]) / "report.json"
    _write_json(report_path, report)
    state["phase"] = "complete"
    state["pending_custom_ids"] = {}
    state["current_request_files"] = []
    state["report_path"] = str(report_path.resolve())
    _write_json(Path(state["state_path"]), state)
    return report_path


async def consume_batch_results(
    *,
    state_path: Path,
    result_paths: list[Path],
    error_paths: list[Path] | None = None,
) -> dict[str, Any]:
    state, dataset, cases_by_id = _load_run_state(state_path.resolve())
    if state["phase"] == "complete":
        raise ValueError(f"批量评测已经完成：{state['report_path']}")
    rows = _read_jsonl([*result_paths, *(error_paths or [])])
    expected_ids = set(state["pending_custom_ids"])
    missing = expected_ids - set(rows)
    unexpected = set(rows) - expected_ids
    if missing:
        raise ValueError(f"结果缺少 {len(missing)} 个 custom_id：{sorted(missing)[:5]}")
    if unexpected:
        raise ValueError(f"结果包含非当前波次 custom_id：{sorted(unexpected)[:5]}")
    sessions_by_key = {session["key"]: session for session in state["sessions"]}

    if state["phase"] == "agent":
        for custom_id, session_key in state["pending_custom_ids"].items():
            session = sessions_by_key[session_key]
            body, error = _row_response(rows[custom_id])
            if error:
                session["status"] = "error"
                session["error"] = error
                continue
            await _apply_agent_body(
                state,
                session,
                cases_by_id[session["case_id"]],
                body or {},
            )
        if any(session["status"] == "pending" for session in state["sessions"]):
            state["wave"] += 1
            requests, mapping = _agent_requests(state, cases_by_id)
            paths = _emit_requests(
                state,
                requests,
                mapping,
                label=f"agent-wave-{state['wave']:03d}",
            )
            return {"phase": "agent", "request_files": paths, "report_path": None}
        if state["config"]["judge"]:
            requests, mapping = _judge_requests(state, cases_by_id)
            if requests:
                state["phase"] = "judge"
                paths = _emit_requests(state, requests, mapping, label="judge-wave-001")
                return {"phase": "judge", "request_files": paths, "report_path": None}
        report_path = _finalize_run(state, dataset, cases_by_id)
        return {"phase": "complete", "request_files": [], "report_path": report_path}

    for custom_id, session_key in state["pending_custom_ids"].items():
        session = sessions_by_key[session_key]
        body, error = _row_response(rows[custom_id])
        if error:
            session["judge"] = {"error": error}
            continue
        choices = (body or {}).get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        text = str(message.get("content") or "")
        try:
            judge = parse_judge_response(text)
            judge["tokens"] = int(((body or {}).get("usage") or {}).get("total_tokens", 0) or 0)
            judge["duration_ms"] = 0
            session["judge"] = judge
        except Exception as error:
            session["judge"] = {"error": f"{type(error).__name__}: {error}"}
    report_path = _finalize_run(state, dataset, cases_by_id)
    return {"phase": "complete", "request_files": [], "report_path": report_path}


def _default_output_dir(model: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(__file__).with_name("batch-runs") / f"{_safe_label(model)}-{timestamp}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成并消费 AI Groupmate 波次批量推理 JSONL")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="创建状态文件和第一波 Agent JSONL")
    prepare.add_argument("--dataset", type=Path, default=DATASET_PATH)
    prepare.add_argument("--case", action="append", dest="case_ids")
    prepare.add_argument("--category", action="append", dest="categories")
    prepare.add_argument("--model", default="qwen3.7-plus")
    prepare.add_argument("--repeat", type=int, default=1)
    prepare.add_argument("--judge", action="store_true")
    prepare.add_argument("--judge-model", default=None)
    prepare.add_argument("--temperature", type=float, default=0)
    prepare.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    prepare.add_argument("--thinking-budget", type=int, default=None)
    prepare.add_argument("--tool-timeout", type=float, default=0.05)
    prepare.add_argument(
        "--max-llm-calls",
        type=int,
        default=DEFAULT_EVAL_MAX_LLM_CALLS,
        help="单用例硬安全上限；数据集中的 max_llm_calls 仅用于效率评分",
    )
    prepare.add_argument("--output-dir", type=Path, default=None)

    consume = commands.add_parser("consume", help="导入当前波次结果并生成下一波或最终报告")
    consume.add_argument("--state", type=Path, required=True)
    consume.add_argument("--result", type=Path, action="append", required=True)
    consume.add_argument("--error", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.repeat < 1 or args.max_llm_calls < 1:
                parser.error("--repeat 和 --max-llm-calls 必须大于等于 1")
            output_dir = args.output_dir or _default_output_dir(args.model)
            state, paths = prepare_batch_run(
                dataset_path=args.dataset,
                output_dir=output_dir,
                model=args.model,
                case_ids=set(args.case_ids or []),
                categories=set(args.categories or []),
                repeat=args.repeat,
                judge=args.judge,
                judge_model=args.judge_model,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
                tool_timeout_seconds=args.tool_timeout,
                max_llm_calls=args.max_llm_calls,
            )
            print(f"状态文件：{state['state_path']}")
            print("请提交以下 JSONL：")
            for path in paths:
                print(path.resolve())
            return 0

        result = asyncio.run(
            consume_batch_results(
                state_path=args.state,
                result_paths=args.result,
                error_paths=args.error,
            )
        )
        if result["phase"] == "complete":
            print(f"评测完成：{result['report_path']}")
        else:
            print(f"进入 {result['phase']} 阶段，请提交下一批 JSONL：")
            for path in result["request_files"]:
                print(path.resolve())
        return 0
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
