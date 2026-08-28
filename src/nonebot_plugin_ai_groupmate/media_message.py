from __future__ import annotations

import json
from typing import Any, Literal
from dataclasses import field, dataclass
from collections.abc import Iterable

MediaKind = Literal["audio", "video"]

MEDIA_HEADERS: dict[MediaKind, str] = {
    "audio": "【语音消息（内容未读取）】",
    "video": "【视频消息（内容未读取）】",
}
MEDIA_MESSAGE_ID_PREFIX = "attachment_message_id: "


def format_media_markers(
    message_id: str,
    *,
    audio_count: int = 0,
    video_count: int = 0,
) -> str:
    """Represent media without fetching or sending it to a model."""
    encoded_id = json.dumps(str(message_id), ensure_ascii=False)
    markers: list[str] = []
    media_counts: tuple[tuple[MediaKind, int], ...] = (
        ("audio", audio_count),
        ("video", video_count),
    )
    for kind, count in media_counts:
        if count <= 0:
            continue
        tool_name = f"read_{kind}_message"
        count_hint = f"（共 {count} 个）" if count > 1 else ""
        markers.append(
            f"{MEDIA_HEADERS[kind]}{count_hint}\n"
            f"{MEDIA_MESSAGE_ID_PREFIX}{encoded_id}\n"
            f"需要了解内容时调用 {tool_name}。"
        )
    return "\n".join(markers)


def extract_media_message_refs(
    contents: Iterable[str],
) -> dict[str, set[MediaKind]]:
    """Extract trusted lazy-media markers from stored chat content."""
    refs: dict[str, set[MediaKind]] = {}
    header_kinds: dict[str, MediaKind] = {
        header: kind for kind, header in MEDIA_HEADERS.items()
    }
    for content in contents:
        lines = content.splitlines()
        stored_message_id = (
            lines[0].split(":", 1)[1].strip()
            if lines and lines[0].startswith("id:")
            else ""
        )
        for index, line in enumerate(lines):
            if not line.startswith(MEDIA_MESSAGE_ID_PREFIX) or index == 0:
                continue
            header = lines[index - 1].split("（共 ", 1)[0]
            kind = header_kinds.get(header)
            if kind is None:
                continue
            try:
                message_id = json.loads(
                    line.removeprefix(MEDIA_MESSAGE_ID_PREFIX)
                )
            except (json.JSONDecodeError, TypeError):
                continue
            if (
                isinstance(message_id, str)
                and message_id
                and message_id == stored_message_id
            ):
                refs.setdefault(message_id, set()).add(kind)
    return refs


@dataclass
class LazyMediaRegistry:
    message_refs: dict[str, set[MediaKind]]
    forwarded_refs: dict[str, tuple[MediaKind, dict[str, Any]]] = field(
        default_factory=dict
    )
    _next_ref: int = 1

    def register_forwarded(self, kind: MediaKind, data: dict[str, Any]) -> str:
        media_ref = f"forward-media-{self._next_ref}"
        self._next_ref += 1
        self.forwarded_refs[media_ref] = (kind, dict(data))
        return media_ref

    def allows_message(self, message_id: str, kind: MediaKind) -> bool:
        return kind in self.message_refs.get(message_id, set())

    def forwarded_source(
        self,
        media_ref: str,
        kind: MediaKind,
    ) -> dict[str, Any] | None:
        entry = self.forwarded_refs.get(media_ref)
        if entry is None or entry[0] != kind:
            return None
        return dict(entry[1])
