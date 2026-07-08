import datetime
from typing import Any
from collections.abc import Sequence

from sqlalchemy import Select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .model import TokenUsage
from .config import ScopedConfig


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


def extract_cache_creation_tokens(callback: Any) -> int:
    for attr in (
        "cache_creation_input_tokens",
        "prompt_tokens_cache_creation",
        "cache_write_input_tokens",
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
    cache_creation_tokens: int = 0,
    callback_cost: float,
    input_cost_per_million: float,
    output_cost_per_million: float,
    cached_input_cost_per_million: float,
    cache_creation_input_cost_per_million: float | None = None,
    long_context_threshold_tokens: int = 256000,
    long_input_cost_per_million: float | None = None,
    long_output_cost_per_million: float | None = None,
    long_cached_input_cost_per_million: float | None = None,
    long_cache_creation_input_cost_per_million: float | None = None,
) -> float:
    if callback_cost > 0:
        return float(callback_cost)

    if prompt_tokens > long_context_threshold_tokens:
        input_cost_per_million = (
            long_input_cost_per_million
            if long_input_cost_per_million is not None
            else input_cost_per_million
        )
        output_cost_per_million = (
            long_output_cost_per_million
            if long_output_cost_per_million is not None
            else output_cost_per_million
        )
        cached_input_cost_per_million = (
            long_cached_input_cost_per_million
            if long_cached_input_cost_per_million is not None
            else cached_input_cost_per_million
        )
        cache_creation_input_cost_per_million = (
            long_cache_creation_input_cost_per_million
            if long_cache_creation_input_cost_per_million is not None
            else cache_creation_input_cost_per_million
        )

    cache_creation_input_cost_per_million = (
        cache_creation_input_cost_per_million
        if cache_creation_input_cost_per_million is not None
        else input_cost_per_million
    )
    billed_prompt_tokens = max(prompt_tokens - cached_tokens - cache_creation_tokens, 0)
    return (
        billed_prompt_tokens / 1_000_000 * input_cost_per_million
        + cached_tokens / 1_000_000 * cached_input_cost_per_million
        + cache_creation_tokens / 1_000_000 * cache_creation_input_cost_per_million
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
    cache_creation_tokens: int,
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
            cache_creation_tokens=cache_creation_tokens,
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
        func.coalesce(func.sum(TokenUsage.cache_creation_tokens), 0).label("cache_creation_tokens"),
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
        "cache_creation_tokens": _as_int(data["cache_creation_tokens"]),
        "total_tokens": _as_int(data["total_tokens"]),
        "estimated_cost": float(data["estimated_cost"] or 0.0),
    }


def estimate_usage_row_cost(row: TokenUsage, config: ScopedConfig) -> float:
    if row.estimated_cost > 0:
        return row.estimated_cost
    cached_input_cost = (
        config.chat_explicit_cached_input_cost_per_million
        if config.chat_explicit_prompt_cache
        else config.chat_cached_input_cost_per_million
    )
    long_cached_input_cost = (
        config.chat_long_explicit_cached_input_cost_per_million
        if config.chat_explicit_prompt_cache
        else config.chat_long_cached_input_cost_per_million
    )
    return estimate_cost(
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        cached_tokens=row.cached_tokens,
        cache_creation_tokens=row.cache_creation_tokens,
        callback_cost=0.0,
        input_cost_per_million=config.chat_input_cost_per_million,
        output_cost_per_million=config.chat_output_cost_per_million,
        cached_input_cost_per_million=cached_input_cost,
        cache_creation_input_cost_per_million=config.chat_cache_creation_input_cost_per_million,
        long_context_threshold_tokens=config.chat_long_context_threshold_tokens,
        long_input_cost_per_million=config.chat_long_input_cost_per_million,
        long_output_cost_per_million=config.chat_long_output_cost_per_million,
        long_cached_input_cost_per_million=long_cached_input_cost,
        long_cache_creation_input_cost_per_million=config.chat_long_cache_creation_input_cost_per_million,
    )


def _aggregate_rows(rows: Sequence[TokenUsage], key_fn) -> list[dict[str, Any]]:
    grouped: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key, labels = key_fn(row)
        item = grouped.setdefault(
            key,
            {
                **labels,
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_creation_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            },
        )
        item["requests"] += 1
        item["prompt_tokens"] += row.prompt_tokens
        item["completion_tokens"] += row.completion_tokens
        item["cached_tokens"] += row.cached_tokens
        item["cache_creation_tokens"] += row.cache_creation_tokens
        item["total_tokens"] += row.total_tokens
        item["estimated_cost"] += getattr(row, "_display_cost", row.estimated_cost)
    return sorted(grouped.values(), key=lambda item: item["total_tokens"], reverse=True)


async def get_usage_dashboard_data(
    db_session: AsyncSession,
    *,
    config: ScopedConfig,
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

    rows = (
        await db_session.execute(
            Select(TokenUsage)
            .where(*filters)
            .order_by(TokenUsage.created_at.desc())
        )
    ).scalars().all()
    for row in rows:
        row._display_cost = estimate_usage_row_cost(row, config)  # type: ignore[attr-defined]

    total = {
        "requests": len(rows),
        "prompt_tokens": sum(row.prompt_tokens for row in rows),
        "completion_tokens": sum(row.completion_tokens for row in rows),
        "cached_tokens": sum(row.cached_tokens for row in rows),
        "cache_creation_tokens": sum(row.cache_creation_tokens for row in rows),
        "total_tokens": sum(row.total_tokens for row in rows),
        "estimated_cost": sum(getattr(row, "_display_cost", row.estimated_cost) for row in rows),
    }
    by_session_rows = _aggregate_rows(
        rows,
        lambda row: (
            (row.session_id, row.session_type),
            {"session_id": row.session_id, "session_type": row.session_type},
        ),
    )[:50]
    by_user_rows = _aggregate_rows(
        rows,
        lambda row: (
            (row.user_id,),
            {"user_id": row.user_id, "user_name": row.user_name},
        ),
    )[:50]
    by_model_rows = _aggregate_rows(
        rows,
        lambda row: (
            (row.model,),
            {"model": row.model or "unknown"},
        ),
    )[:30]
    recent_rows = (
        await db_session.execute(
            Select(TokenUsage)
            .where(*filters)
            .order_by(TokenUsage.created_at.desc())
            .limit(100)
        )
    ).scalars().all()
    for row in recent_rows:
        row._display_cost = estimate_usage_row_cost(row, config)  # type: ignore[attr-defined]

    return {
        "days": days,
        "since": since.isoformat(),
        "filters": {"session_id": session_id or "", "user_id": user_id or ""},
        "total": total,
        "by_session": by_session_rows,
        "by_user": by_user_rows,
        "by_model": by_model_rows,
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
                "cache_creation_tokens": row.cache_creation_tokens,
                "total_tokens": row.total_tokens,
                "estimated_cost": getattr(row, "_display_cost", row.estimated_cost),
            }
            for row in recent_rows
        ],
    }
