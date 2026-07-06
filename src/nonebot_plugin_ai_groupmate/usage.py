import datetime
from typing import Any

from sqlalchemy import Select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .model import TokenUsage


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def extract_cached_tokens(callback: Any) -> int:
    for attr in (
        "cached_tokens",
        "prompt_tokens_cached",
        "cache_read_input_tokens",
    ):
        value = getattr(callback, attr, None)
        if isinstance(value, int | float):
            return int(value)
    return 0


def estimate_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    callback_cost: float,
    input_cost_per_million: float,
    output_cost_per_million: float,
    cached_input_cost_per_million: float,
) -> float:
    if callback_cost > 0:
        return float(callback_cost)

    billed_prompt_tokens = max(prompt_tokens - cached_tokens, 0)
    return (
        billed_prompt_tokens / 1_000_000 * input_cost_per_million
        + cached_tokens / 1_000_000 * cached_input_cost_per_million
        + completion_tokens / 1_000_000 * output_cost_per_million
    )


async def record_token_usage(
    db_session: AsyncSession,
    *,
    session_id: str,
    session_type: str,
    user_id: str,
    user_name: str | None,
    model: str,
    request_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    total_tokens: int,
    estimated_cost: float,
) -> None:
    db_session.add(
        TokenUsage(
            session_id=session_id,
            session_type=session_type,
            user_id=user_id or "",
            user_name=user_name or "",
            model=model or "",
            request_id=request_id or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
        )
    )


def since_from_days(days: int) -> datetime.datetime:
    days = max(1, min(days, 3650))
    return datetime.datetime.now() - datetime.timedelta(days=days)


def _summary_columns() -> tuple:
    return (
        func.count(TokenUsage.id).label("requests"),
        func.coalesce(func.sum(TokenUsage.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(TokenUsage.completion_tokens), 0).label("completion_tokens"),
        func.coalesce(func.sum(TokenUsage.cached_tokens), 0).label("cached_tokens"),
        func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(TokenUsage.estimated_cost), 0.0).label("estimated_cost"),
    )


def _row_metrics(row: Any) -> dict[str, Any]:
    data = row._mapping if hasattr(row, "_mapping") else row
    return {
        "requests": _as_int(data["requests"]),
        "prompt_tokens": _as_int(data["prompt_tokens"]),
        "completion_tokens": _as_int(data["completion_tokens"]),
        "cached_tokens": _as_int(data["cached_tokens"]),
        "total_tokens": _as_int(data["total_tokens"]),
        "estimated_cost": float(data["estimated_cost"] or 0.0),
    }


async def get_usage_dashboard_data(
    db_session: AsyncSession,
    *,
    days: int = 7,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    since = since_from_days(days)
    filters = [TokenUsage.created_at >= since]
    if session_id:
        filters.append(TokenUsage.session_id == session_id)
    if user_id:
        filters.append(TokenUsage.user_id == user_id)

    total_row = (
        await db_session.execute(Select(*_summary_columns()).where(*filters))
    ).one()

    by_session_rows = (
        await db_session.execute(
            Select(TokenUsage.session_id, TokenUsage.session_type, *_summary_columns())
            .where(*filters)
            .group_by(TokenUsage.session_id, TokenUsage.session_type)
            .order_by(func.sum(TokenUsage.total_tokens).desc())
            .limit(50)
        )
    ).all()

    by_user_rows = (
        await db_session.execute(
            Select(TokenUsage.user_id, func.max(TokenUsage.user_name).label("user_name"), *_summary_columns())
            .where(*filters)
            .group_by(TokenUsage.user_id)
            .order_by(func.sum(TokenUsage.total_tokens).desc())
            .limit(50)
        )
    ).all()

    by_model_rows = (
        await db_session.execute(
            Select(TokenUsage.model, *_summary_columns())
            .where(*filters)
            .group_by(TokenUsage.model)
            .order_by(func.sum(TokenUsage.total_tokens).desc())
            .limit(30)
        )
    ).all()

    recent_rows = (
        await db_session.execute(
            Select(TokenUsage)
            .where(*filters)
            .order_by(TokenUsage.created_at.desc())
            .limit(100)
        )
    ).scalars().all()

    return {
        "days": days,
        "since": since.isoformat(),
        "filters": {"session_id": session_id or "", "user_id": user_id or ""},
        "total": _row_metrics(total_row),
        "by_session": [
            {
                "session_id": row.session_id,
                "session_type": row.session_type,
                **_row_metrics(row),
            }
            for row in by_session_rows
        ],
        "by_user": [
            {
                "user_id": row.user_id,
                "user_name": row.user_name or "",
                **_row_metrics(row),
            }
            for row in by_user_rows
        ],
        "by_model": [
            {
                "model": row.model or "unknown",
                **_row_metrics(row),
            }
            for row in by_model_rows
        ],
        "recent": [
            {
                "created_at": row.created_at.isoformat(),
                "session_id": row.session_id,
                "session_type": row.session_type,
                "user_id": row.user_id,
                "user_name": row.user_name,
                "model": row.model,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "cached_tokens": row.cached_tokens,
                "total_tokens": row.total_tokens,
                "estimated_cost": row.estimated_cost,
            }
            for row in recent_rows
        ],
    }
