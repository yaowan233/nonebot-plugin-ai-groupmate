import asyncio
import datetime
from typing import Any

from nonebot import get_plugin_config
from pydantic import ValidationError
from sqlalchemy import Select
from nonebot.log import logger
from sqlalchemy.ext.asyncio import AsyncSession

from .model import RuntimeConfigOverride
from .config import Config, ScopedConfig

BOOTSTRAP_FIELDS = frozenset({
    "usage_webui_enabled",
    "usage_webui_path",
    "usage_webui_token",
})
RESTART_REQUIRED_FIELDS = frozenset({
    "qdrant_uri",
    "qdrant_api_key",
    "embedding_api_key",
    "embedding_base_url",
    "meme_embedding_mode",
    "rerank_api_url",
    "rerank_api_key",
    "qwen_token",
})
SECRET_FIELDS = frozenset(
    name
    for name in ScopedConfig.model_fields
    if name.endswith(("_api_key", "_token"))
)
CONFIGURABLE_FIELDS = frozenset(ScopedConfig.model_fields) - BOOTSTRAP_FIELDS

_environment_config = get_plugin_config(Config).ai_groupmate
_runtime_config = _environment_config.model_copy(deep=True)
_overrides: dict[str, Any] = {}
_pending_restart_fields: set[str] = set()
_restart_baseline = {
    field_name: getattr(_runtime_config, field_name)
    for field_name in RESTART_REQUIRED_FIELDS
}
_update_lock = asyncio.Lock()


def get_runtime_config() -> ScopedConfig:
    return _runtime_config


def get_environment_config() -> ScopedConfig:
    return _environment_config.model_copy(deep=True)


def get_config_overrides() -> dict[str, Any]:
    return dict(_overrides)


def get_pending_restart_fields() -> set[str]:
    return set(_pending_restart_fields)


def _refresh_pending_restart_fields() -> set[str]:
    _pending_restart_fields.clear()
    _pending_restart_fields.update({
        field_name
        for field_name, applied_value in _restart_baseline.items()
        if getattr(_runtime_config, field_name) != applied_value
    })
    return set(_pending_restart_fields)


def mark_restart_fields_applied() -> None:
    for field_name in RESTART_REQUIRED_FIELDS:
        _restart_baseline[field_name] = getattr(_runtime_config, field_name)
    _pending_restart_fields.clear()


def _validated_config(overrides: dict[str, Any]) -> ScopedConfig:
    values = _environment_config.model_dump()
    values.update({
        key: value
        for key, value in overrides.items()
        if key in CONFIGURABLE_FIELDS
    })
    return ScopedConfig.model_validate(values)


def _replace_runtime_config(config: ScopedConfig) -> set[str]:
    changed: set[str] = set()
    for field_name in ScopedConfig.model_fields:
        old_value = getattr(_runtime_config, field_name)
        new_value = getattr(config, field_name)
        if old_value != new_value:
            changed.add(field_name)
            setattr(_runtime_config, field_name, new_value)
    return changed


def _normalized_overrides(config: ScopedConfig) -> dict[str, Any]:
    environment_values = _environment_config.model_dump()
    config_values = config.model_dump()
    return {
        field_name: config_values[field_name]
        for field_name in CONFIGURABLE_FIELDS
        if config_values[field_name] != environment_values[field_name]
    }


def preview_runtime_config_update(
    updates: dict[str, Any],
    *,
    clear_secrets: set[str] | None = None,
) -> tuple[ScopedConfig, dict[str, Any], set[str]]:
    unknown_fields = set(updates) - CONFIGURABLE_FIELDS
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"不支持通过网页修改这些配置：{names}")

    clear_secrets = clear_secrets or set()
    configurable_secret_fields = SECRET_FIELDS & CONFIGURABLE_FIELDS
    invalid_clear_fields = clear_secrets - configurable_secret_fields
    if invalid_clear_fields:
        names = ", ".join(sorted(invalid_clear_fields))
        raise ValueError(f"这些字段不是密钥配置：{names}")

    values = _runtime_config.model_dump()
    values.update(updates)
    for field_name in clear_secrets:
        values[field_name] = ""

    candidate = ScopedConfig.model_validate(values)
    overrides = _normalized_overrides(candidate)
    changed = {
        field_name
        for field_name in ScopedConfig.model_fields
        if getattr(candidate, field_name) != getattr(_runtime_config, field_name)
    }
    return candidate, overrides, changed


async def load_runtime_config_overrides(db_session: AsyncSession) -> set[str]:
    global _overrides

    result = await db_session.execute(
        Select(RuntimeConfigOverride).where(RuntimeConfigOverride.id == 1)
    )
    row = result.scalar_one_or_none()
    stored = row.overrides if row and isinstance(row.overrides, dict) else {}
    stored = {
        key: value
        for key, value in stored.items()
        if key in CONFIGURABLE_FIELDS
    }
    try:
        config = _validated_config(stored)
    except ValidationError:
        logger.exception("WebUI 持久化配置校验失败，已回退到环境变量配置")
        config = _environment_config.model_copy(deep=True)
        stored = {}

    _overrides = dict(stored)
    changed = _replace_runtime_config(config)
    _refresh_pending_restart_fields()
    return changed


async def save_runtime_config_updates(
    db_session: AsyncSession,
    updates: dict[str, Any],
    *,
    clear_secrets: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    global _overrides

    async with _update_lock:
        candidate, overrides, changed = preview_runtime_config_update(
            updates,
            clear_secrets=clear_secrets,
        )
        result = await db_session.execute(
            Select(RuntimeConfigOverride).where(RuntimeConfigOverride.id == 1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = RuntimeConfigOverride(
                id=1,
                overrides=overrides,
                updated_at=datetime.datetime.now(),
            )
            db_session.add(row)
        else:
            row.overrides = overrides
            row.updated_at = datetime.datetime.now()
        await db_session.commit()

        _overrides = overrides
        _replace_runtime_config(candidate)
        restart_fields = _refresh_pending_restart_fields()
        return changed, restart_fields


async def reset_runtime_config_overrides(
    db_session: AsyncSession,
) -> tuple[set[str], set[str]]:
    global _overrides

    async with _update_lock:
        result = await db_session.execute(
            Select(RuntimeConfigOverride).where(RuntimeConfigOverride.id == 1)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await db_session.delete(row)
        await db_session.commit()

        config = _environment_config.model_copy(deep=True)
        changed = _replace_runtime_config(config)
        _overrides = {}
        restart_fields = _refresh_pending_restart_fields()
        return changed, restart_fields
