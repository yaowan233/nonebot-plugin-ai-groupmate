import re
import json
from typing import Any
from pathlib import Path
from collections import Counter

DATASET_PATH = Path(__file__).parents[1] / "evals" / "agent_cases.json"
EXPECTED_CATEGORIES = {
    "conversation": 10,
    "single_tool": 8,
    "multi_tool": 8,
    "memory_knowledge": 6,
    "side_effect": 4,
    "failure_recovery": 4,
}
VALID_OUTCOMES = {
    "reply",
    "silent",
    "action",
    "action_and_reply",
    "degraded_reply",
    "delivery_unknown",
}
VALID_RESPONSE_CHECKS = {
    "contains_all",
    "contains_any",
    "not_contains_any",
    "semantic",
}


def _load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _assert_bounds(value: Any, *, label: str) -> None:
    assert isinstance(value, dict), f"{label} must be an object"
    assert set(value) == {"min", "max"}, f"{label} must contain min/max"
    assert isinstance(value["min"], int)
    assert value["min"] >= 0
    assert isinstance(value["max"], int)
    assert value["max"] >= value["min"]


def test_agent_eval_dataset_shape_and_category_counts():
    dataset = _load_dataset()

    assert dataset["schema_version"] == 1
    assert dataset["category_targets"] == EXPECTED_CATEGORIES
    assert len(dataset["cases"]) == sum(EXPECTED_CATEGORIES.values()) == 40

    counts = Counter(case["category"] for case in dataset["cases"])
    assert counts == Counter(EXPECTED_CATEGORIES)


def test_agent_eval_case_ids_are_unique_and_stable():
    cases = _load_dataset()["cases"]
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))
    for case in cases:
        assert re.fullmatch(r"[a-z_]+_\d{3}", case["id"])
        assert case["id"].startswith(f"{case['category']}_")


def test_agent_eval_cases_follow_schema():
    cases = _load_dataset()["cases"]

    for case in cases:
        case_id = case["id"]
        assert set(case) == {
            "id",
            "category",
            "title",
            "tags",
            "input",
            "tool_fixtures",
            "faults",
            "expected",
        }
        assert case["title"].strip()
        assert case["tags"]
        assert all(isinstance(tag, str) and tag for tag in case["tags"])

        input_data = case["input"]
        required_input_fields = {
            "scene",
            "bot_name",
            "current_time",
            "personality_setting",
            "has_admin_permission",
            "messages",
        }
        assert required_input_fields <= set(input_data), case_id
        assert set(input_data) <= required_input_fields | {"group_memory"}, case_id
        assert input_data["scene"] in {"group", "private"}
        assert isinstance(input_data["has_admin_permission"], bool)
        assert input_data["messages"]
        assert input_data["messages"][-1]["speaker"] == "user"

        message_ids: set[str] = set()
        for message in input_data["messages"]:
            assert set(message) == {
                "id",
                "speaker",
                "user_id",
                "user_name",
                "content_type",
                "content",
                "reply_to",
            }, case_id
            assert message["speaker"] in {"user", "bot"}
            assert message["content_type"] in {"text", "image"}
            assert message["id"] not in message_ids
            if message["reply_to"] is not None:
                assert message["reply_to"] in message_ids, case_id
            message_ids.add(message["id"])

        for fixture in case["tool_fixtures"]:
            assert set(fixture) == {"tool", "call", "result"}, case_id
            assert fixture["tool"]
            assert fixture["call"] >= 1

        for fault in case["faults"]:
            assert set(fault) == {"tool", "call", "kind"}, case_id
            assert fault["tool"]
            assert fault["call"] >= 1
            assert fault["kind"]

        expected = case["expected"]
        required_expected_fields = {
            "outcome",
            "required_tools",
            "optional_tools",
            "forbidden_tools",
            "ordered_tools",
            "tool_call_counts",
            "side_effects",
            "max_llm_calls",
            "max_tool_calls",
            "response_checks",
            "rubric",
        }
        assert required_expected_fields <= set(expected), case_id
        assert set(expected) <= required_expected_fields | {"allowed_outcomes"}, case_id
        assert expected["outcome"] in VALID_OUTCOMES
        allowed_outcomes = expected.get("allowed_outcomes", [expected["outcome"]])
        assert allowed_outcomes, case_id
        assert expected["outcome"] in allowed_outcomes, case_id
        assert set(allowed_outcomes) <= VALID_OUTCOMES, case_id
        assert expected["max_llm_calls"] >= 1
        assert expected["max_tool_calls"] >= 0
        assert expected["rubric"]

        required_tools = set(expected["required_tools"])
        optional_tools = set(expected["optional_tools"])
        forbidden_tools = set(expected["forbidden_tools"])
        assert not required_tools & optional_tools, case_id
        assert not required_tools & forbidden_tools, case_id
        assert not optional_tools & forbidden_tools, case_id
        assert set(expected["ordered_tools"]) <= required_tools | optional_tools, case_id

        for tool_name, bounds in expected["tool_call_counts"].items():
            assert tool_name in required_tools | optional_tools, case_id
            _assert_bounds(bounds, label=f"{case_id}.tool_call_counts.{tool_name}")
        for effect_name, bounds in expected["side_effects"].items():
            _assert_bounds(bounds, label=f"{case_id}.side_effects.{effect_name}")

        for check in expected["response_checks"]:
            check_type = check.get("type")
            assert check_type in VALID_RESPONSE_CHECKS, case_id
            if check_type == "semantic":
                assert set(check) == {"type", "criterion"}
                assert check["criterion"].strip()
            else:
                assert set(check) == {"type", "values"}
                assert check["values"]


def test_required_tools_have_deterministic_fixture_or_fault():
    cases = _load_dataset()["cases"]

    for case in cases:
        covered_tools = {
            item["tool"] for item in [*case["tool_fixtures"], *case["faults"]]
        }
        missing = set(case["expected"]["required_tools"]) - covered_tools
        assert not missing, f"{case['id']} missing fixtures/faults for {sorted(missing)}"


def test_strictly_silent_cases_have_no_external_side_effects():
    cases = _load_dataset()["cases"]

    for case in cases:
        expected = case["expected"]
        allowed_outcomes = set(expected.get("allowed_outcomes", [expected["outcome"]]))
        if allowed_outcomes != {"silent"}:
            continue
        assert expected["side_effects"] == {}, case["id"]
        assert "reply_user" in expected["forbidden_tools"], case["id"]
