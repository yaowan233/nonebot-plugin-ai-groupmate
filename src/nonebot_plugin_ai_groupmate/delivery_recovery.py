"""Durably reconcile acknowledged deliveries without sending a message again."""

import os
import re
import time
import uuid
import asyncio
from typing import Any, cast
from pathlib import Path
from datetime import datetime, timedelta

from pydantic import Field, BaseModel
from sqlalchemy import Select, CursorResult, delete
from nonebot.log import logger
from nonebot_plugin_orm import get_session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .model import ChatHistory, DeliveryRecoveryReceipt, StructuredMessage
from .agent.file_workspace import FileWorkspace

_ID = re.compile(r"^[0-9a-f]{32}$")
_TEMP_NAME = re.compile(r"^[0-9a-f]{32}\.[0-9a-f]{32}\.tmp$")
MAX_JOURNAL_BYTES = 262144


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class HistoryEntry(BaseModel):
    session_id: str
    user_id: str
    user_name: str
    content: str
    media_id: int | None = None
    created_at: datetime

    def to_row(self) -> ChatHistory:
        return ChatHistory(session_id=self.session_id, user_id=self.user_id, user_name=self.user_name,
                           content_type="bot", content=self.content, media_id=self.media_id, created_at=self.created_at)


class Publication(BaseModel):
    session_id: str
    user_id: str
    bot_id: str
    is_private: bool
    file_id: str
    message_id: str


class StructuredDelivery(BaseModel):
    platform: str
    scope: str
    session_id: str
    platform_id: str
    sender_id: str
    sender_name: str
    bot_id: str
    reply_to_id: str | None = None
    parts: list[dict[str, str]]
    created_at: datetime


class RecoveryRecord(BaseModel):
    receipt_id: str
    entries: list[HistoryEntry] = Field(min_length=1, max_length=32)
    publication: Publication | None = None
    publications: list[Publication] = Field(default_factory=list, max_length=10)
    structured: StructuredDelivery | None = None


def write_recovery(root: Path, record: RecoveryRecord) -> Path:
    if not _ID.fullmatch(record.receipt_id):
        raise ValueError("invalid recovery id")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise OSError("unsafe recovery directory")
    data = record.model_dump_json().encode("utf-8")
    if len(data) > MAX_JOURNAL_BYTES:
        raise ValueError("recovery record too large")
    target = root / (record.receipt_id + ".json")
    temporary = root / (record.receipt_id + "." + uuid.uuid4().hex + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        try:
            _sync_directory(root)
        except OSError as error:
            # The file is already visible. Still use its receipt marker so a
            # later replay cannot duplicate history if directory sync failed.
            logger.warning(f"恢复目录同步失败 error_type={type(error).__name__}")
    finally:
        temporary.unlink(missing_ok=True)
    return target


def delete_recovery(path: Path) -> None:
    path.unlink(missing_ok=True)
    _sync_directory(path.parent)


async def apply_history(db: Any, record: RecoveryRecord) -> bool:
    """Insert a receipt and all compatibility rows atomically, once per receipt."""
    dialect = db.get_bind().dialect.name
    if dialect not in {"sqlite", "postgresql"}:
        raise RuntimeError("unsupported database for delivery recovery")
    insert = sqlite_insert if dialect == "sqlite" else pg_insert
    result = await db.execute(insert(DeliveryRecoveryReceipt).values(
        receipt_id=record.receipt_id, created_at=datetime.now(),
    ).on_conflict_do_nothing(index_elements=["receipt_id"]))
    if cast(CursorResult[Any], result).rowcount:
        for entry in record.entries:
            db.add(entry.to_row())
    structured_saved = True
    if record.structured is not None:
        # History and its idempotency marker survive a structured-only failure;
        # keep the journal so the optional snapshot can be reconciled later.
        try:
            async with db.begin_nested():
                from .structured_message import record_inbound
                inserted = await record_inbound(db, **record.structured.model_dump(), sender_kind="bot")
                if not inserted:
                    item = record.structured
                    existing = await db.scalar(Select(StructuredMessage).where(
                        StructuredMessage.platform == item.platform, StructuredMessage.scope == item.scope,
                        StructuredMessage.session_id == item.session_id, StructuredMessage.platform_id == item.platform_id,
                    ))
                    if existing is None or existing.sender_kind != "bot" or existing.sender_id != item.sender_id:
                        raise ValueError("structured delivery identity conflict")
        except Exception as error:
            structured_saved = False
            logger.warning(f"恢复发送结构化记录失败 error_type={type(error).__name__}")
    await db.commit()
    return structured_saved


def apply_publication(agent_files: Path, publication: Publication) -> None:
    if not _ID.fullmatch(publication.file_id) or not publication.message_id or publication.message_id == "unknown":
        raise ValueError("invalid publication")
    workspace = FileWorkspace(agent_files, session_id=publication.session_id, user_id=publication.user_id,
                              bot_id=publication.bot_id, is_private=publication.is_private, peer_bot_ids=())
    workspace.publish_image(publication.file_id, publication.message_id)


async def recover_pending(root: Path, agent_files: Path, *, limit: int = 20) -> int:
    """Replay local metadata only. This function never invokes a transport."""
    if not root.is_dir() or root.is_symlink():
        return 0
    for path in list(root.glob("*.tmp"))[:limit]:
        try:
            if not path.is_symlink() and _TEMP_NAME.fullmatch(path.name) and path.stat().st_mtime < time.time() - 86400:
                path.unlink(missing_ok=True)
        except OSError:
            pass
    recovered = 0
    candidates = []
    for path in root.glob("*.json"):
        try:
            if not path.is_symlink() and _ID.fullmatch(path.stem):
                candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for due_at, path in sorted(candidates)[:limit]:
        if due_at > time.time():
            break
        try:
            if path.stat().st_size > MAX_JOURNAL_BYTES:
                raise ValueError("recovery record too large")
            record = RecoveryRecord.model_validate_json(path.read_bytes())
            if record.receipt_id != path.stem:
                raise ValueError("recovery id mismatch")
            async with get_session() as db:
                structured_saved = await apply_history(db, record)
            if record.publication is not None:
                await asyncio.to_thread(apply_publication, agent_files, record.publication)
            for publication in record.publications:
                await asyncio.to_thread(apply_publication, agent_files, publication)
            if not structured_saved:
                raise RuntimeError("structured delivery recovery pending")
            await asyncio.to_thread(delete_recovery, path)
            recovered += 1
        except Exception as error:
            logger.warning(f"恢复发送记录失败 receipt_id={path.stem} error_type={type(error).__name__}")
            try:
                next_attempt = time.time() + 300
                os.utime(path, (next_attempt, next_attempt), follow_symlinks=False)
            except OSError:
                pass
    return recovered


async def prune_receipts(root: Path, *, now: datetime | None = None, limit: int = 100) -> int:
    """Remove old deduplication markers only when no recovery file remains."""
    if root.is_symlink() or limit <= 0:
        return 0
    cutoff = (now or datetime.now()) - timedelta(days=30)
    async with get_session() as db:
        ids = list((await db.scalars(Select(DeliveryRecoveryReceipt.receipt_id)
                    .where(DeliveryRecoveryReceipt.created_at < cutoff)
                    .order_by(DeliveryRecoveryReceipt.created_at).limit(limit))).all())
        removable = [receipt_id for receipt_id in ids if _ID.fullmatch(receipt_id)
                     and not (root / (receipt_id + ".json")).exists()]
        if not removable:
            return 0
        result = await db.execute(delete(DeliveryRecoveryReceipt).where(
            DeliveryRecoveryReceipt.receipt_id.in_(removable), DeliveryRecoveryReceipt.created_at < cutoff))
        await db.commit()
        return cast(CursorResult[Any], result).rowcount
