from __future__ import annotations

import json
import asyncio
import datetime
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from sqlalchemy import Select, func

from .model import UserRelation


@dataclass(frozen=True)
class NegativeRelationResetResult:
    affected_count: int
    backup_path: Path | None


async def count_negative_relations(db_session) -> int:
    """返回需要软重置的负好感度关系数量。"""
    result = await db_session.execute(
        Select(func.count(UserRelation.id)).where(UserRelation.favorability < 0)
    )
    return int(result.scalar_one())


def _serialize_relation(relation: UserRelation) -> dict[str, Any]:
    updated_at = relation.updated_at
    return {
        "id": relation.id,
        "user_id": relation.user_id,
        "user_name": relation.user_name,
        "favorability": relation.favorability,
        "tags": list(relation.tags or []),
        "updated_at": updated_at.isoformat() if updated_at else None,
    }


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


async def reset_negative_relations(
    db_session,
    backup_dir: Path,
    *,
    now: datetime.datetime | None = None,
) -> NegativeRelationResetResult:
    """备份并软重置所有负好感度关系；备份成功前绝不修改数据库。"""
    relations = list(
        (
            await db_session.execute(
                Select(UserRelation)
                .where(UserRelation.favorability < 0)
                .order_by(UserRelation.id)
            )
        )
        .scalars()
        .all()
    )
    if not relations:
        await db_session.rollback()
        return NegativeRelationResetResult(affected_count=0, backup_path=None)

    backup_time = now or datetime.datetime.now(datetime.timezone.utc).astimezone()
    timestamp = backup_time.strftime("%Y%m%dT%H%M%S%f%z")
    backup_path = backup_dir / f"negative-relations-{timestamp}.json"
    payload = {
        "schema_version": 1,
        "created_at": backup_time.isoformat(),
        "operation": "reset_negative_relations",
        "affected_count": len(relations),
        "relations": [_serialize_relation(relation) for relation in relations],
    }

    try:
        await asyncio.to_thread(_write_json_atomically, backup_path, payload)
    except Exception:
        await db_session.rollback()
        raise

    try:
        for relation in relations:
            relation.favorability = 0
            relation.tags = []
        await db_session.commit()
    except Exception:
        await db_session.rollback()
        raise

    return NegativeRelationResetResult(
        affected_count=len(relations), backup_path=backup_path
    )
