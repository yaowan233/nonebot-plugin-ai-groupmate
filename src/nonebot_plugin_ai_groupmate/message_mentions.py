"""Keep native mention IDs intact across recording, input batching and tools."""

from typing import Any
from contextvars import ContextVar

from .structured_message import normalize_parts

current_mentioned_user_ids: ContextVar[tuple[str, ...] | None] = ContextVar("mentioned_user_ids", default=None)
current_normalized_message: ContextVar[tuple[Any, tuple[dict[str, str], ...]] | None] = ContextVar(
    "normalized_message", default=None,
)


def _segments(event: Any) -> list[Any]:
    original = getattr(event, "original_message", None)
    if original and any(getattr(segment, "type", "") not in {"reply", ""} for segment in original):
        return list(original)
    try:
        return list(event.get_message()) or list(original or ())
    except Exception:
        return list(original or ())


def mentioned_user_ids(parts: tuple[dict[str, str], ...] | list[dict[str, str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        target for part in parts
        if part["type"] == "mention" and (target := part.get("target", "").strip())
        and target.casefold() not in {"all", "here"}
    ))


def _normalized_parts(event: Any) -> tuple[dict[str, str], ...] | list[dict[str, str]]:
    current = current_normalized_message.get()
    if current is not None and current[0] is event:
        return current[1]
    return normalize_parts(event, _segments(event))


def event_mentioned_user_ids(event: Any) -> tuple[str, ...]:
    return mentioned_user_ids(_normalized_parts(event))


def format_member_mention(target: str, name: str = "") -> str:
    return f"@{name}（user_id={target}） " if name and name != target else f"@{target} "


def event_text_with_mentions(event: Any) -> str:
    try:
        plaintext = event.get_plaintext().strip()
    except Exception:
        plaintext = ""
    segments = _normalized_parts(event)
    if not any(part["type"] == "mention" and part.get("target", "").strip().casefold() not in {"", "all", "here"} for part in segments):
        return plaintext
    parts = []
    for segment in segments:
        kind = segment["type"]
        if kind == "mention":
            target = segment.get("target", "").strip()
            if target.casefold() not in {"", "all", "here"}:
                parts.append(format_member_mention(target))
        elif kind == "text":
            parts.append(segment.get("text", ""))
        elif kind in {"image", "mface", "face", "video", "record", "audio", "file"}:
            parts.append(f"[{kind}]")
    return "".join(parts).strip() or plaintext
