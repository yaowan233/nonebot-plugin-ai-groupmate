import math
import asyncio
import weakref
import datetime
from typing import Any
from dataclasses import dataclass

from sqlalchemy import Select, select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from .model import GlobalModelGroupUsage, GlobalModelPrivateUserUsage

_quota_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


def _get_quota_lock(scope: str, subject_id: str) -> asyncio.Lock:
    lock_key = f"{scope}:{subject_id}"
    lock = _quota_locks.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _quota_locks[lock_key] = lock
    return lock


@dataclass(frozen=True)
class GroupDailyQuotaStatus:
    allowed: bool
    used: int
    limit: int
    resets_at: datetime.datetime

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def _local_now(now: datetime.datetime | None = None) -> datetime.datetime:
    if now is None:
        return datetime.datetime.now().astimezone()
    return now if now.tzinfo is not None else now.astimezone()


def _reset_time(now: datetime.datetime) -> datetime.datetime:
    return datetime.datetime.combine(
        now.date() + datetime.timedelta(days=1),
        datetime.time.min,
        tzinfo=now.tzinfo,
    )


def _status(
    *,
    used: int,
    limit: int,
    now: datetime.datetime,
) -> GroupDailyQuotaStatus:
    return GroupDailyQuotaStatus(
        allowed=limit <= 0 or used < limit,
        used=used,
        limit=limit,
        resets_at=_reset_time(now),
    )


async def _get_daily_quota_status(
    db_session: AsyncSession,
    subject_id: str,
    limit: int,
    *,
    usage_model: type[Any],
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    local_now = _local_now(now)
    if limit <= 0:
        return _status(used=0, limit=limit, now=local_now)

    row = await db_session.get(usage_model, subject_id)
    used = (
        row.used_count
        if row is not None and row.usage_date == local_now.date()
        else 0
    )
    return _status(used=used, limit=limit, now=local_now)


async def get_group_daily_quota_status(
    db_session: AsyncSession,
    group_id: str,
    limit: int,
    *,
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    """Read a group's current public-model quota without consuming it."""
    return await _get_daily_quota_status(
        db_session,
        group_id,
        limit,
        usage_model=GlobalModelGroupUsage,
        now=now,
    )


async def get_private_user_daily_quota_status(
    db_session: AsyncSession,
    user_id: str,
    limit: int,
    *,
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    """Read a user's current private public-model quota without consuming it."""
    return await _get_daily_quota_status(
        db_session,
        user_id,
        limit,
        usage_model=GlobalModelPrivateUserUsage,
        now=now,
    )


async def _consume_daily_quota_once(
    db_session: AsyncSession,
    subject_id: str,
    limit: int,
    *,
    usage_model: type[Any],
    id_field: Any,
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    """Atomically reserve one public-model reply and finish the transaction."""
    local_now = _local_now(now)
    usage_date = local_now.date()
    if limit <= 0:
        return _status(used=0, limit=limit, now=local_now)

    # The conditional UPDATE is the quota boundary. Concurrent workers can only
    # increment while the stored value is below the current runtime limit.
    for _ in range(3):
        result = await db_session.execute(
            update(usage_model)
            .where(
                id_field == subject_id,
                usage_model.usage_date == usage_date,
                usage_model.used_count < limit,
            )
            .values(
                used_count=usage_model.used_count + 1,
                updated_at=local_now.replace(tzinfo=None),
            )
            .execution_options(synchronize_session=False)
        )
        if getattr(result, "rowcount", 0) > 0:
            used = await db_session.scalar(
                select(usage_model.used_count).where(id_field == subject_id)
            )
            await db_session.commit()
            return GroupDailyQuotaStatus(
                allowed=True,
                used=used if used is not None else 1,
                limit=limit,
                resets_at=_reset_time(local_now),
            )

        result = await db_session.execute(
            update(usage_model)
            .where(
                id_field == subject_id,
                usage_model.usage_date != usage_date,
            )
            .values(
                usage_date=usage_date,
                used_count=1,
                updated_at=local_now.replace(tzinfo=None),
            )
            .execution_options(synchronize_session=False)
        )
        if getattr(result, "rowcount", 0) > 0:
            await db_session.commit()
            return GroupDailyQuotaStatus(
                allowed=True,
                used=1,
                limit=limit,
                resets_at=_reset_time(local_now),
            )

        row = (
            (
                await db_session.execute(
                    Select(usage_model).where(id_field == subject_id)
                )
            )
            .scalars()
            .first()
        )
        if row is not None:
            if row.usage_date == usage_date and row.used_count >= limit:
                used = row.used_count
                await db_session.commit()
                return GroupDailyQuotaStatus(
                    allowed=False,
                    used=used,
                    limit=limit,
                    resets_at=_reset_time(local_now),
                )
            await db_session.rollback()
            continue

        db_session.add(
            usage_model(
                **{id_field.key: subject_id},
                usage_date=usage_date,
                used_count=1,
                updated_at=local_now.replace(tzinfo=None),
            )
        )
        try:
            await db_session.commit()
        except IntegrityError:
            # Another process inserted this subject between SELECT and INSERT.
            await db_session.rollback()
            continue
        return GroupDailyQuotaStatus(
            allowed=True,
            used=1,
            limit=limit,
            resets_at=_reset_time(local_now),
        )

    # A highly contended INSERT raced repeatedly. Re-read once so the caller
    # receives a safe denial instead of allowing the limit to be bypassed.
    status = await _get_daily_quota_status(
        db_session,
        subject_id,
        limit,
        usage_model=usage_model,
        now=local_now,
    )
    await db_session.commit()
    return GroupDailyQuotaStatus(
        allowed=False,
        used=status.used,
        limit=limit,
        resets_at=status.resets_at,
    )


async def _consume_daily_quota(
    db_session: AsyncSession,
    subject_id: str,
    limit: int,
    *,
    scope: str,
    usage_model: type[Any],
    id_field: Any,
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    async with _get_quota_lock(scope, subject_id):
        for attempt in range(8):
            try:
                return await _consume_daily_quota_once(
                    db_session,
                    subject_id,
                    limit,
                    usage_model=usage_model,
                    id_field=id_field,
                    now=now,
                )
            except OperationalError as error:
                await db_session.rollback()
                if "locked" not in str(error).lower() or attempt == 7:
                    raise
                await asyncio.sleep(min(0.01 * (2**attempt), 0.25))

    raise RuntimeError("公共模型额度预留重试次数耗尽")


async def consume_group_daily_quota(
    db_session: AsyncSession,
    group_id: str,
    limit: int,
    *,
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    """Atomically reserve one group reply."""
    return await _consume_daily_quota(
        db_session,
        group_id,
        limit,
        scope="group",
        usage_model=GlobalModelGroupUsage,
        id_field=GlobalModelGroupUsage.group_id,
        now=now,
    )


async def consume_private_user_daily_quota(
    db_session: AsyncSession,
    user_id: str,
    limit: int,
    *,
    now: datetime.datetime | None = None,
) -> GroupDailyQuotaStatus:
    """Atomically reserve one private reply for a user."""
    return await _consume_daily_quota(
        db_session,
        user_id,
        limit,
        scope="private-user",
        usage_model=GlobalModelPrivateUserUsage,
        id_field=GlobalModelPrivateUserUsage.user_id,
        now=now,
    )


def _quota_wait_text(
    status: GroupDailyQuotaStatus,
    now: datetime.datetime | None = None,
) -> str:
    local_now = _local_now(now)
    seconds = max(0, (status.resets_at - local_now).total_seconds())
    total_minutes = max(1, math.ceil(seconds / 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"约 {hours} 小时 {minutes} 分钟"
    if hours:
        return f"约 {hours} 小时"
    return f"约 {minutes} 分钟"


def build_quota_exhausted_message(
    status: GroupDailyQuotaStatus,
    *,
    now: datetime.datetime | None = None,
) -> str:
    wait_text = _quota_wait_text(status, now)
    return (
        f"本群今天的公共模型额度（{status.limit} 次）已用完。"
        "群主或管理员可发送 /配置群API 使用本群自己的 API；"
        f"公共额度将在 {wait_text}后恢复。"
    )


def build_private_quota_exhausted_message(
    status: GroupDailyQuotaStatus,
    *,
    now: datetime.datetime | None = None,
) -> str:
    wait_text = _quota_wait_text(status, now)
    return (
        f"你今天的私聊公共模型额度（{status.limit} 次）已用完。"
        "可发送 /配置个人API 使用自己的 API；"
        f"公共额度将在 {wait_text}后恢复。"
    )
