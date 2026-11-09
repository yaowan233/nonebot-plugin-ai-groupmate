"""Resolve scoped structured messages and their image resources."""

import asyncio
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy import Select
from nonebot.log import logger

from ..model import MediaStorage, StructuredMessage
from .image_resources import ImageResources, ImageResourceError
from ..message_metadata import is_reliable_platform_id


@dataclass(frozen=True)
class ScopedStructuredMessage:
    session_id: str
    is_private: bool
    platform_id: str
    sender_id: str
    sender_name: str
    sender_kind: str
    parts: tuple[dict[str, str], ...]
    media_paths: dict[str, str]


@dataclass(frozen=True)
class ScopedMessageLookup:
    message: ScopedStructuredMessage | None
    inaccessible: bool = False


async def find_recent_scoped_user_image(
    db, *, session_id: str, platform: str, is_private: bool, user_id: str | None,
) -> tuple[str, datetime] | None:
    """Find a recent image from this user without reading another conversation."""
    if not platform or not user_id:
        return None
    try:
        async with db.begin_nested():
            rows = (await db.execute(Select(
                StructuredMessage.platform_id, StructuredMessage.created_at, StructuredMessage.parts,
            ).where(
                StructuredMessage.platform == platform, StructuredMessage.scope == ("private" if is_private else "group"),
                StructuredMessage.session_id == session_id, StructuredMessage.sender_kind == "user",
                StructuredMessage.sender_id == str(user_id),
            ).order_by(StructuredMessage.created_at.desc(), StructuredMessage.id.desc()).limit(200))).all()
            for platform_id, created_at, parts in rows:
                if isinstance(parts, list) and any(isinstance(part, dict) and part.get("type") == "image" for part in parts):
                    return platform_id, created_at
        return None
    except Exception as error:
        logger.warning(f"读取近期结构化图片失败 error_type={type(error).__name__}")
        raise ImageResourceError("unavailable") from None


async def load_scoped_messages(
    db, *, session_id: str, platform: str, is_private: bool, message_ids: list[str],
) -> dict[str, ScopedMessageLookup]:
    """Load a batch in one scoped lookup; missing IDs remain eligible for legacy reading."""
    targets = list(dict.fromkeys(str(value).strip() for value in message_ids
                                 if is_reliable_platform_id(value) and "\n" not in value and "\r" not in value))
    if not platform or not targets:
        return {}
    scope = "private" if is_private else "group"
    try:
        async with db.begin_nested():
            rows = (await db.scalars(Select(StructuredMessage).where(
                StructuredMessage.platform == platform, StructuredMessage.scope.in_(("group", "private")),
                StructuredMessage.session_id == session_id, StructuredMessage.platform_id.in_(targets),
            ))).all()
            scoped = {row.platform_id: row for row in rows if row.scope == scope}
            unavailable = {row.platform_id for row in rows if row.platform_id not in scoped}
            unavailable.update(target for target, row in scoped.items() if not isinstance(row.parts, list))
            parts_by_id = {target: tuple(dict(part) for part in row.parts if isinstance(part, dict))
                           for target, row in scoped.items() if target not in unavailable}
            media_ids = {int(part["media_id"]) for parts in parts_by_id.values() for part in parts
                         if part.get("type") == "image" and str(part.get("media_id", "")).isdigit()}
            paths = {}
            if media_ids:
                entries = (await db.execute(Select(MediaStorage.media_id, MediaStorage.file_path).where(
                    MediaStorage.media_id.in_(media_ids),
                ))).all()
                paths = {str(media_id): path for media_id, path in entries if path}
            found = {target: ScopedMessageLookup(ScopedStructuredMessage(
                session_id, is_private, target, scoped[target].sender_id, scoped[target].sender_name,
                scoped[target].sender_kind, parts, paths,
            )) for target, parts in parts_by_id.items()}
            found.update({target: ScopedMessageLookup(None, inaccessible=True) for target in unavailable})
            return found
    except ImageResourceError:
        raise
    except Exception as error:
        logger.warning(f"读取结构化消息失败，停止本次引用 error_type={type(error).__name__}")
        raise ImageResourceError("unavailable") from None


async def load_scoped_message(
    db, *, session_id: str, platform: str, is_private: bool, message_id: str | None,
) -> ScopedStructuredMessage | None:
    """Return None only if no structured record exists and legacy lookup is needed."""
    if not is_reliable_platform_id(message_id):
        return None
    target = str(message_id).strip()
    found = await load_scoped_messages(db, session_id=session_id, platform=platform, is_private=is_private,
                                       message_ids=[target])
    result = found.get(target)
    if result is not None and result.inaccessible:
        raise ImageResourceError("not_found")
    return result.message if result is not None else None


async def read_message_images(
    resources: ImageResources, message: ScopedStructuredMessage, *,
    max_bytes: int | None, max_images: int = 4, last_only: bool = False, image_index: int | None = None,
) -> list[bytes]:
    if resources.workspace is not None and (
        resources.workspace.session_id != message.session_id or resources.workspace.is_private != message.is_private
    ):
        raise ImageResourceError("not_found")
    image_parts = [part for part in message.parts if part.get("type") == "image"]
    if image_index is not None:
        if image_index < 1 or image_index > len(image_parts):
            raise ImageResourceError("not_found")
        image_parts = [image_parts[image_index - 1]]
    elif last_only:
        image_parts = image_parts[-1:]
    if len(image_parts) > max_images:
        raise ImageResourceError("too_many")
    images = []
    for part in image_parts:
        file_id = str(part.get("file_id") or "")
        media_path = message.media_paths.get(str(part.get("media_id") or ""))
        if file_id and message.sender_kind == "bot":
            data = await asyncio.to_thread(resources.read_publication, file_id, message.platform_id, max_bytes=max_bytes)
        elif media_path:
            data = await asyncio.to_thread(resources.read_media, media_path, max_bytes=max_bytes)
        else:
            raise ImageResourceError("not_found")
        images.append(data)
    return images


async def read_scoped_message_images(
    db, resources: ImageResources, *, session_id: str, platform: str, is_private: bool,
    message_id: str | None, max_bytes: int | None, max_images: int = 4, last_only: bool = False,
    image_index: int | None = None,
) -> list[bytes] | None:
    if resources.workspace is not None and (
        resources.workspace.session_id != session_id or resources.workspace.is_private != is_private
    ):
        raise ImageResourceError("not_found")
    message = await load_scoped_message(db, session_id=session_id, platform=platform,
                                        is_private=is_private, message_id=message_id)
    if message is None:
        return None
    return await read_message_images(resources, message, max_bytes=max_bytes, max_images=max_images, last_only=last_only, image_index=image_index)
