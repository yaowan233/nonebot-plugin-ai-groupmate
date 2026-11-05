"""Normalize native messages once, preserving ordered transport semantics."""

import hashlib
from io import BytesIO
from typing import Any
from datetime import datetime

from sqlalchemy import Select
from nonebot.log import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_scoped_session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .model import StructuredMessage
from .shared_cards import parse_shared_segment
from .message_metadata import is_reliable_platform_id


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "")


def adapter_name(bot: Any) -> str:
    try:
        value = bot.adapter.get_name()
        return _clean(value)[:64] or "unknown"
    except (AttributeError, TypeError):
        return "unknown"


def image_source_id(segment: Any) -> str:
    """Read a stable source identity without storing adapter URLs or IDs."""
    original = getattr(segment, "origin", None) or segment
    data = getattr(original, "data", {})
    data = data if isinstance(data, dict) else {}
    source = data.get("file") or data.get("file_id") or data.get("id") or \
        getattr(original, "id", "") or data.get("url") or getattr(original, "url", "")
    if isinstance(source, BytesIO):
        source = source.getvalue()
    if isinstance(source, (bytes, bytearray)) and source:
        return "raw-sha256:" + hashlib.sha256(source).hexdigest()
    if source:
        return _clean(source)
    raw = data.get("raw") or getattr(original, "raw", None)
    if isinstance(raw, BytesIO):
        raw = raw.getvalue()
    if isinstance(raw, (bytes, bytearray)) and raw:
        return "raw-sha256:" + hashlib.sha256(raw).hexdigest()
    return ""


def _message_segments(event: Any, fallback_segments: Any) -> Any:
    original = getattr(event, "original_message", None)
    if original and any(getattr(segment, "type", "") not in {"reply", ""} for segment in original):
        return original
    return fallback_segments


def normalize_parts(event: Any, fallback_segments: Any) -> list[dict[str, str]]:
    """Use original adapter segments when available, including mentions stripped by adapters."""
    segments = _message_segments(event, fallback_segments)
    parts: list[dict[str, str]] = []
    for segment in segments:
        kind = getattr(segment, "type", "")
        data = getattr(segment, "data", {})
        if not isinstance(data, dict):
            data = {}
        if card := parse_shared_segment(segment):
            parts.append({"type": "card", "text": _clean(card.text)})
        elif kind == "text":
            text = data.get("text")
            if text is None:
                text = getattr(segment, "text", "")
            parts.append({"type": "text", "text": _clean(text if not callable(text) else "")})
        elif kind == "at":
            target = getattr(segment, "target", None) or data.get("qq") or data.get("target") or data.get("user_id")
            parts.append({"type": "mention", "target": _clean(target)})
        elif kind in {"face", "mface"}:
            part = {"type": kind}
            face_id = data.get("id") or data.get("face_id") or data.get("emoji_id")
            if face_id is not None:
                part["face_id"] = _clean(face_id)[:128]
            parts.append(part)
        elif kind in {"forward", "reference"}:
            forward_id = _clean(data.get("id") or getattr(segment, "id", ""))
            part = {"type": "forward"}
            if forward_id and len(forward_id) <= 255 and "\n" not in forward_id and "\r" not in forward_id:
                part["forward_id"] = forward_id
            parts.append(part)
        elif kind in {"image", "video", "record", "audio", "file"}:
            # Resource access is resolved separately; avoid storing adapter URLs or raw payloads.
            parts.append({"type": "image" if kind == "image" else kind})
        elif kind not in {"reply", ""}:
            parts.append({"type": "unsupported", "kind": _clean(kind)[:64]})
    return parts


async def record_inbound(
    db: AsyncSession | async_scoped_session[AsyncSession], *, scope: str, session_id: str, platform_id: str,
    sender_id: str, sender_name: str, reply_to_id: str | None,
    parts: list[dict[str, str]], platform: str = "unknown", sender_kind: str = "user", bot_id: str = "",
    created_at: datetime | None = None,
) -> bool:
    """Insert idempotently in the caller's transaction; unknown IDs stay legacy-only."""
    if not is_reliable_platform_id(platform_id):
        return False
    values = {"platform": _clean(platform), "scope": scope, "session_id": _clean(session_id), "platform_id": _clean(platform_id).strip(),
              "sender_id": _clean(sender_id), "sender_name": _clean(sender_name),
              "sender_kind": _clean(sender_kind), "bot_id": _clean(bot_id),
              "reply_to_id": _clean(reply_to_id) or None, "parts": parts, "created_at": created_at or datetime.now()}
    dialect = db.get_bind().dialect.name
    if dialect in {"postgresql", "sqlite"}:
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        statement = insert(StructuredMessage).values(**values).on_conflict_do_nothing(
            index_elements=["platform", "scope", "session_id", "platform_id"]
        )
        result = await db.execute(statement)
        return bool(getattr(result, "rowcount", 0))
    existing = await db.scalar(Select(StructuredMessage.id).where(
        StructuredMessage.platform == platform, StructuredMessage.scope == scope, StructuredMessage.session_id == session_id,
        StructuredMessage.platform_id == values["platform_id"],
    ))
    if existing is not None:
        return False
    db.add(StructuredMessage(**values))
    return True


async def record_inbound_safely(
    db: AsyncSession | async_scoped_session[AsyncSession], *, event: Any, fallback_segments: Any,
    mention_names: dict[str, str], file_ids: tuple[str | None, ...] = (),
    file_source_ids: tuple[str, ...] = (), file_names: tuple[str, ...] = (),
    image_source_ids: tuple[str, ...] = (), normalized_parts: tuple[dict[str, str], ...] | None = None,
    **metadata: Any,
) -> bool:
    """Keep a failed structured write from discarding compatibility history."""
    try:
        async with db.begin_nested():
            parts = ([dict(part) for part in normalized_parts] if normalized_parts is not None
                     else normalize_parts(event, fallback_segments))
            if not parts:
                # An empty snapshot would hide the compatibility record from
                # scoped readers even though it carries no usable content.
                return False
            for part in parts:
                if part.get("type") == "mention" and part.get("target") in mention_names:
                    part["name"] = mention_names[part["target"]]
            file_parts = [part for part in parts if part.get("type") == "file"]
            segments = _message_segments(event, fallback_segments)
            source_ids = []
            native_image_ids = []
            for segment in segments:
                data = getattr(segment, "data", {})
                data = data if isinstance(data, dict) else {}
                if getattr(segment, "type", "") == "file":
                    source_ids.append(_clean(data.get("file_id") or data.get("id") or getattr(segment, "id", "")))
                elif getattr(segment, "type", "") == "image":
                    native_image_ids.append(image_source_id(segment))
            image_parts = [part for part in parts if part.get("type") == "image"]
            if (image_source_ids and len(image_parts) == len(native_image_ids) == len(image_source_ids)
                    and all(native_image_ids) and sorted(native_image_ids) == sorted(map(_clean, image_source_ids))):
                for part, source_id in zip(image_parts, native_image_ids):
                    part["source_hash"] = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
            if (file_ids and len(file_parts) == len(file_ids) == len(source_ids) == len(file_source_ids)
                    and all(source_ids) and tuple(source_ids) == file_source_ids):
                for index, (part, file_id) in enumerate(zip(file_parts, file_ids)):
                    if file_id:
                        part["file_id"] = file_id
                        if len(file_names) == len(file_parts) and file_names[index]:
                            part["name"] = _clean(file_names[index])[:100]
            return await record_inbound(db, parts=parts, **metadata)
    except Exception as error:
        logger.warning(f"保存结构化消息失败，继续使用旧记录 error_type={type(error).__name__}")
        return False


async def bind_image_resource(
    db: AsyncSession | async_scoped_session[AsyncSession], *, scope: str, session_id: str, platform_id: str,
    image_index: int, image_count: int, media_id: int, platform: str = "unknown", sender_id: str | None = None,
    source_id: str | None = None, require_source_for_multi: bool = False,
) -> bool:
    """Bind a downloaded image using its source when the image counts agree.

    Call under the session's ingestion lock. Missing or ambiguous associations
    remain unbound instead of assigning one user's image to another message.
    """
    message = await db.scalar(Select(StructuredMessage).where(
        StructuredMessage.platform == platform, StructuredMessage.scope == scope, StructuredMessage.session_id == session_id,
        StructuredMessage.platform_id == platform_id,
    ))
    if message is None or (sender_id is not None and message.sender_id != sender_id):
        return False
    parts = [dict(part) for part in message.parts]
    positions = [i for i, part in enumerate(parts) if part.get("type") == "image"]
    if len(positions) != image_count or not 0 <= image_index < len(positions):
        return False
    position = positions[image_index]
    if any(parts[pos].get("source_hash") for pos in positions):
        if not source_id:
            return False
        source_hash = hashlib.sha256(_clean(source_id).encode("utf-8")).hexdigest()
        matches = [pos for pos in positions if parts[pos].get("source_hash") == source_hash]
        if not matches:
            return False
        if position not in matches:
            position = next((pos for pos in matches if not parts[pos].get("media_id")), matches[0])
    elif require_source_for_multi and image_count > 1:
        return False
    if parts[position].get("media_id") not in (None, "", str(media_id)):
        return False
    parts[position]["media_id"] = str(media_id)
    message.parts = parts
    return True


async def bind_image_resource_safely(db: AsyncSession | async_scoped_session[AsyncSession], **metadata: Any) -> bool:
    """Preserve legacy image history if the optional structured association fails."""
    try:
        async with db.begin_nested():
            return await bind_image_resource(db, **metadata)
    except Exception as error:
        logger.warning(f"关联结构化图片失败，继续使用旧记录 error_type={type(error).__name__}")
        return False
