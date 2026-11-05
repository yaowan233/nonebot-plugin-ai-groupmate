"""Load scoped structured message snapshots for model history formatting."""

import json
from dataclasses import field, dataclass, replace

from sqlalchemy import Select
from nonebot.log import logger

from ..model import MediaStorage, ChatHistorySchema, StructuredMessage, HistoryMessageIndex
from ..message_mentions import format_member_mention
from ..message_metadata import platform_message_ids
from ..media_message import format_media_markers
from ..forward_message import format_forward_marker


@dataclass(frozen=True)
class StructuredSnapshot:
    platform_id: str
    sender_id: str
    reply_to_id: str | None
    parts: tuple[dict[str, str], ...]
    media_paths: dict[int, str]
    history_ids: tuple[int, ...] = ()
    history_platform_ids: dict[int, frozenset[str]] = field(default_factory=dict)


def merge_snapshots(
    existing: dict[tuple[str, str], StructuredSnapshot],
    updates: dict[tuple[str, str], StructuredSnapshot],
) -> None:
    """Preserve bindings when separate history windows contain the same message."""
    for key, current in updates.items():
        previous = existing.get(key)
        if previous is not None:
            current = replace(
                current,
                media_paths={**previous.media_paths, **current.media_paths},
                history_ids=tuple(dict.fromkeys((*previous.history_ids, *current.history_ids))),
                history_platform_ids={**previous.history_platform_ids, **current.history_platform_ids},
            )
        existing[key] = current


def visible_text(snapshot: StructuredSnapshot) -> str | None:
    """Render message parts for textual quote context; preserve legacy-only media paths."""
    parts: list[str] = []
    for part in snapshot.parts:
        kind = part.get("type", "")
        if kind in {"text", "card"}:
            parts.append(part.get("text", ""))
        elif kind == "mention":
            parts.append(format_member_mention(part.get("target", ""), part.get("name", "")))
        elif kind == "image":
            parts.append("[图片]")
        elif kind in {"face", "mface"}:
            parts.append("[表情]")
        elif kind == "file" and part.get("file_id") and part.get("name"):
            parts.append("[文件：" + json.dumps(part["name"], ensure_ascii=False) + "]")
        else:
            return None
    rendered = "".join(parts)
    return rendered if rendered else None


def visible_quote_text(snapshot: StructuredSnapshot) -> str | None:
    """Render known quote parts and keep lazy media references scoped to the message."""
    if not snapshot.parts:
        return None
    output: list[str] = []
    audio_count = video_count = 0
    for part in snapshot.parts:
        kind = part.get("type", "")
        if kind in {"text", "card"} and not part.get("text"):
            continue
        fragment = visible_text(replace(snapshot, parts=(part,)))
        if fragment is not None:
            output.append(fragment)
        elif kind in {"record", "audio"}:
            audio_count += 1
            output.append("[语音]")
        elif kind == "video":
            video_count += 1
            output.append("[视频]")
        elif kind == "file":
            output.append("[文件]")
        elif kind == "forward":
            output.append(format_forward_marker(part.get("forward_id")))
        else:
            return None
    if audio_count or video_count:
        output.append("\n" + format_media_markers(snapshot.platform_id, audio_count=audio_count, video_count=video_count))
    return "".join(output) or None


async def load_structured_context(db, history: list[ChatHistorySchema], *, session_id: str,
                                  scope: str, platform: str, include_media: bool = True) -> dict[tuple[str, str], StructuredSnapshot]:
    """Return only records for the current platform, conversation and sender."""
    history_rows = [row for row in history if row.session_id == session_id]
    if not history_rows:
        return {}
    indexed = (await db.execute(Select(HistoryMessageIndex.history_id, HistoryMessageIndex.platform_id).where(
        HistoryMessageIndex.session_id == session_id,
        HistoryMessageIndex.history_id.in_([row.msg_id for row in history_rows]),
    ))).all()
    indexed_ids: dict[int, set[str]] = {}
    for history_id, platform_id in indexed:
        indexed_ids.setdefault(history_id, set()).add(platform_id)
    bindings: dict[tuple[str, str], list[int]] = {}
    known_ids: dict[int, frozenset[str]] = {}
    for row in history_rows:
        # Old records without an index still use the compatibility parser.
        message_ids = indexed_ids.get(row.msg_id)
        if message_ids is None:
            message_ids = platform_message_ids(row.content)
        known_ids[row.msg_id] = frozenset(message_ids)
        for message_id in message_ids:
            bindings.setdefault((message_id, row.user_id), []).append(row.msg_id)
    ids = {message_id for message_id, _ in bindings}
    if not ids:
        return {}
    result = await db.execute(Select(StructuredMessage).where(
        StructuredMessage.platform == platform, StructuredMessage.scope == scope,
        StructuredMessage.session_id == session_id, StructuredMessage.platform_id.in_(ids),
        StructuredMessage.sender_kind.in_(("user", "bot")),
    ))
    rows = list(result.scalars().all())
    media_ids = ({int(part["media_id"]) for row in rows for part in row.parts
                  if isinstance(part, dict) and str(part.get("media_id", "")).isdigit()}
                 if include_media else set())
    paths: dict[int, str] = {}
    if media_ids:
        result = await db.execute(Select(MediaStorage.media_id, MediaStorage.file_path).where(MediaStorage.media_id.in_(media_ids)))
        paths = {media_id: file_path for media_id, file_path in result.all() if file_path}
    return {
        (row.platform_id, row.sender_name if row.sender_kind == "bot" else row.sender_id): StructuredSnapshot(
            platform_id=row.platform_id, sender_id=row.sender_id, reply_to_id=row.reply_to_id,
            parts=tuple(dict(part) for part in row.parts if isinstance(part, dict)), media_paths=paths,
            history_ids=tuple(bindings.get((row.platform_id, row.sender_name if row.sender_kind == "bot" else row.sender_id), ())),
            history_platform_ids={history_id: known_ids[history_id] for history_id in
                                  bindings.get((row.platform_id, row.sender_name if row.sender_kind == "bot" else row.sender_id), ())},
        ) for row in rows
    }


async def load_structured_context_safely(db, history: list[ChatHistorySchema], *, session_id: str,
                                         scope: str, platform: str,
                                         include_media: bool = True) -> dict[tuple[str, str], StructuredSnapshot] | None:
    """Return None on failure so the caller can rebuild using legacy history."""
    try:
        async with db.begin_nested():
            return await load_structured_context(db, history, session_id=session_id, scope=scope, platform=platform,
                                                 include_media=include_media)
    except Exception as error:
        logger.warning(f"读取结构化消息失败，继续使用旧记录 error_type={type(error).__name__}")
        return None
