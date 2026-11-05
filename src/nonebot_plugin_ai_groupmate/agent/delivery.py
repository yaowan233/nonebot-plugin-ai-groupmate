"""Finalize a confirmed platform send without misreporting delivery as failed."""

import uuid
import asyncio
from types import SimpleNamespace
from typing import Any
from datetime import datetime
from dataclasses import dataclass, replace
from collections.abc import Callable, Awaitable

from nonebot.log import logger

from ..model import ChatHistory
from ..message_trace import trace_id
from ..participation import record_group_delivery
from .file_workspace import FileWorkspace
from ..message_metadata import is_reliable_platform_id
from ..delivery_recovery import Publication, HistoryEntry, RecoveryRecord, StructuredDelivery, apply_history, write_recovery, delete_recovery
from ..structured_message import adapter_name, normalize_parts


@dataclass(frozen=True)
class DeliveryReceipt:
    message_id: str
    history_saved: bool
    publication_saved: bool
    structured_saved: bool = True

    @property
    def note(self) -> str:
        if not self.history_saved or not self.publication_saved or not self.structured_saved:
            return "消息已发出，但本地记录未完全保存，请勿重复发送。"
        return ""


@dataclass(frozen=True)
class OutboundIdentity:
    platform: str
    scope: str
    bot_id: str
    bot_name: str
    request_id: str | None = None


@dataclass(frozen=True)
class PublishedImage:
    """Publication and recovery metadata for one image in its original scope."""

    workspace: FileWorkspace
    file_id: str

    async def publish(self, message_id: str) -> None:
        await asyncio.to_thread(self.workspace.publish_image, self.file_id, message_id)

    def describe(self, message_id: str) -> Publication:
        return Publication(session_id=self.workspace.session_id, user_id=self.workspace.user_id, bot_id=self.workspace.bot_id,
                           is_private=self.workspace.is_private, file_id=self.file_id, message_id=message_id)


def structured_delivery(identity: OutboundIdentity, session_id: str, message_id: str, message: Any,
                        reply_to_id: str | None, rows: list[ChatHistory], publication: Publication | None,
                        created_at: datetime, image_file_ids: list[str | None] | None = None,
                        image_media_ids: list[int | None] | None = None) -> StructuredDelivery | None:
    if not is_reliable_platform_id(message_id) or len(message_id) > 255 or "\n" in message_id or "\r" in message_id:
        return None
    parts = normalize_parts(SimpleNamespace(original_message=None), message)
    images = [part for part in parts if part.get("type") == "image"]
    media_ids = [row.media_id for row in rows if row.media_id is not None]
    if image_media_ids is None and len(images) == len(media_ids):
        for part, media_id in zip(images, media_ids):
            part["media_id"] = str(media_id)
    if len(images) == 1 and publication is not None:
        images[0]["file_id"] = publication.file_id
    if image_file_ids is not None:
        if len(images) != len(image_file_ids):
            raise ValueError("image metadata does not match message")
        for part, file_id in zip(images, image_file_ids):
            if file_id:
                part["file_id"] = file_id
    if image_media_ids is not None:
        if len(images) != len(image_media_ids):
            raise ValueError("image metadata does not match message")
        for part, media_id in zip(images, image_media_ids):
            if media_id is not None:
                part["media_id"] = str(media_id)
    return StructuredDelivery(platform=identity.platform, scope=identity.scope, session_id=session_id,
                              platform_id=message_id, sender_id=identity.bot_id, sender_name=identity.bot_name,
                              bot_id=identity.bot_id, reply_to_id=reply_to_id, parts=parts, created_at=created_at)


def receipt_message_id(receipt: Any) -> str:
    ids = getattr(receipt, "msg_ids", None) or []
    if not ids or not isinstance(ids[-1], dict):
        return "unknown"
    value = ids[-1].get("message_id") or ids[-1].get("msg_id")
    return str(value) if value is not None else "unknown"


async def send_and_finalize(
    db_session: Any, session_id: str, message: Any, *, send_options: dict[str, Any],
    record: Callable[[str], list[ChatHistory]],
    publish: Callable[[str], Awaitable[None]] | None = None,
    publication: Callable[[str], Publication] | None = None,
    recovery_enabled: bool = True,
    outbound_identity: OutboundIdentity | None = None,
    reply_to_id: str | None = None,
    session_factory: Callable[[], Any] | None = None,
    publications: list[PublishedImage] | None = None,
    image_file_ids: list[str | None] | None = None,
    image_media_ids: list[int | None] | None = None,
) -> DeliveryReceipt:
    """Send once, then record the acknowledged message through one delivery path."""
    receipt = await message.send(**send_options)

    return await finalize_delivery(db_session, session_id, receipt, record=record, publish=publish,
                                   publication=publication, recovery_enabled=recovery_enabled,
                                   outbound_identity=outbound_identity, outbound_message=message,
                                   reply_to_id=reply_to_id, session_factory=session_factory,
                                   publications=publications, image_file_ids=image_file_ids, image_media_ids=image_media_ids)


async def finalize_delivery(
    db_session: Any, session_id: str, receipt: Any, *,
    record: Callable[[str], list[ChatHistory]],
    publish: Callable[[str], Awaitable[None]] | None = None,
    publication: Callable[[str], Publication] | None = None,
    recovery_enabled: bool = True,
    outbound_identity: OutboundIdentity | None = None,
    outbound_message: Any = None,
    reply_to_id: str | None = None,
    session_factory: Callable[[], Any] | None = None,
    publications: list[PublishedImage] | None = None,
    image_file_ids: list[str | None] | None = None,
    image_media_ids: list[int | None] | None = None,
) -> DeliveryReceipt:
    """The adapter has returned successfully; never claim the send failed here."""
    message_id = receipt_message_id(receipt)
    receipt_bot = getattr(receipt, "bot", None)
    if outbound_identity is not None and receipt_bot is not None:
        actual_bot_id = str(getattr(receipt_bot, "self_id", "") or "")
        actual_platform = adapter_name(receipt_bot)
        if (actual_bot_id and actual_platform != "unknown" and
                outbound_identity.bot_id in {"", outbound_identity.bot_name, actual_bot_id}):
            outbound_identity = replace(outbound_identity, bot_id=actual_bot_id, platform=actual_platform)
    if outbound_identity is None or outbound_identity.scope != "private":
        try:
            record_group_delivery(session_id)
        except Exception as error:
            logger.warning(f"记录群参与状态失败 error_type={type(error).__name__}")
    journal_path = None
    rows: list[ChatHistory] = []
    recovery = None
    structured_expected = bool(recovery_enabled and outbound_identity is not None and outbound_message is not None
                               and is_reliable_platform_id(message_id) and len(message_id) <= 255
                               and "\n" not in message_id and "\r" not in message_id)
    try:
        rows = record(message_id)
        if recovery_enabled:
            created_at = datetime.now()
            entries = [HistoryEntry(session_id=row.session_id, user_id=row.user_id, user_name=row.user_name,
                                    content=row.content, media_id=row.media_id, created_at=created_at) for row in rows]
            published = publication(message_id) if publication and message_id != "unknown" else None
            published_many = [item.describe(message_id) for item in (publications or [])] if message_id != "unknown" else []
            structured = None
            if outbound_identity is not None and outbound_message is not None:
                try:
                    structured = structured_delivery(outbound_identity, session_id, message_id, outbound_message, reply_to_id, rows, published, created_at, image_file_ids, image_media_ids)
                except Exception as error:
                    logger.warning(f"构建发送结构化记录失败，继续保存旧记录 error_type={type(error).__name__}")
            recovery = RecoveryRecord(receipt_id=uuid.uuid4().hex, entries=entries, publication=published, publications=published_many, structured=structured)
            from .. import plugin_data_dir
            journal_path = await asyncio.to_thread(write_recovery, plugin_data_dir / "delivery_recovery", recovery)
    except Exception as error:
        logger.warning(f"保存发送恢复记录失败 error_type={type(error).__name__}")
    publication_saved = True
    structured_saved = not structured_expected
    if publish is not None:
        if message_id == "unknown":
            publication_saved = False
        else:
            try:
                await publish(message_id)
            except Exception as error:
                publication_saved = False
                logger.warning(f"保存图片发布关系失败 error_type={type(error).__name__}")
    if publications:
        if message_id == "unknown":
            publication_saved = False
        else:
            for item in publications:
                try:
                    await item.publish(message_id)
                except Exception as error:
                    publication_saved = False
                    logger.warning(f"保存图片发布关系失败 error_type={type(error).__name__}")
    active_session = db_session

    async def save_history(session: Any) -> tuple[bool, bool]:
        if not rows:
            return False, not structured_expected
        if recovery is not None and callable(getattr(session, "get_bind", None)):
            # Even if the journal could not be written, preserve the structured
            # record with the same receipt marker in the database transaction.
            saved = await apply_history(session, recovery)
            return True, saved and (not structured_expected or recovery.structured is not None)
        for row in rows:
            session.add(row)
        await session.commit()
        return True, not structured_expected

    try:
        if session_factory is None:
            history_saved, structured_saved = await save_history(active_session)
        else:
            async with session_factory() as session:
                active_session = session
                history_saved, structured_saved = await save_history(session)
    except Exception as error:
        history_saved = False
        logger.warning(f"保存已发送消息失败 error_type={type(error).__name__}")
        try:
            if active_session is not None:
                await active_session.rollback()
        except Exception as rollback_error:
            logger.warning(f"发送记录回滚失败 error_type={type(rollback_error).__name__}")
    # A journal without a structured payload cannot reconstruct one later.
    can_discard_journal = structured_saved or (recovery is not None and recovery.structured is None)
    if journal_path is not None and history_saved and can_discard_journal and (publication_saved or message_id == "unknown"):
        try:
            await asyncio.to_thread(delete_recovery, journal_path)
        except OSError as error:
            logger.warning(f"清理发送恢复记录失败 error_type={type(error).__name__}")
    if outbound_identity is not None and outbound_identity.request_id:
        logger.info(f"[MessageTrace] trace={trace_id(session_id, outbound_identity.request_id)} stage=delivery_ack "
                    f"history_saved={history_saved} publication_saved={publication_saved} structured_saved={structured_saved}")
    return DeliveryReceipt(message_id, history_saved, publication_saved, structured_saved)
