import json
from typing import Any, Literal
from collections.abc import Mapping

ToolResultStatus = Literal["succeeded", "skipped", "failed"]
DeliveryState = Literal["not_attempted", "completed", "unknown"]

TOOL_RESULT_SCHEMA_VERSION = 1


def tool_result(
    status: ToolResultStatus,
    reason_code: str,
    message: str,
    *,
    retryable: bool = False,
    data: Mapping[str, Any] | None = None,
    delivery_state: DeliveryState | None = None,
) -> str:
    """Serialize the shared result envelope used by all built-in agent tools."""
    payload: dict[str, Any] = {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "ok": status == "succeeded",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "retryable": retryable,
    }
    if delivery_state is not None:
        payload["delivery_state"] = delivery_state
    if data is not None:
        payload["data"] = dict(data)
    return json.dumps(payload, ensure_ascii=False)


def tool_success(
    reason_code: str,
    message: str,
    *,
    data: Mapping[str, Any] | None = None,
    delivery_state: DeliveryState | None = None,
) -> str:
    return tool_result(
        "succeeded",
        reason_code,
        message,
        data=data,
        delivery_state=delivery_state,
    )


def tool_skipped(
    reason_code: str,
    message: str,
    *,
    data: Mapping[str, Any] | None = None,
    delivery_state: DeliveryState | None = None,
) -> str:
    return tool_result(
        "skipped",
        reason_code,
        message,
        data=data,
        delivery_state=delivery_state,
    )


def tool_failure(
    reason_code: str,
    message: str,
    *,
    retryable: bool = False,
    data: Mapping[str, Any] | None = None,
    delivery_state: DeliveryState | None = None,
) -> str:
    return tool_result(
        "failed",
        reason_code,
        message,
        retryable=retryable,
        data=data,
        delivery_state=delivery_state,
    )


def parse_tool_result(content: str) -> dict[str, Any] | None:
    """Parse a versioned result, or normalize a legacy status for extensions."""
    if content.strip().lower() == "sent":
        return {"status": "succeeded", "ok": True}
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None

    status = parsed.get("status")
    if status == "sent":
        return {**parsed, "status": "succeeded", "ok": True}
    legacy_success = parsed.get("success")
    if isinstance(legacy_success, bool):
        legacy_data = {
            key: value
            for key, value in parsed.items()
            if key not in {"success", "reason_code", "reason", "message"}
        }
        return {
            "status": "succeeded" if legacy_success else "failed",
            "ok": legacy_success,
            "reason_code": parsed.get("reason_code"),
            "message": parsed.get("message") or parsed.get("reason"),
            "data": legacy_data,
        }
    if status not in {"succeeded", "skipped", "failed"}:
        return None
    if "ok" not in parsed:
        parsed = {**parsed, "ok": status == "succeeded"}
    return parsed


def tool_result_status(content: str) -> ToolResultStatus | None:
    parsed = parse_tool_result(content)
    if parsed is None:
        return None
    status = parsed.get("status")
    return status if status in {"succeeded", "skipped", "failed"} else None
