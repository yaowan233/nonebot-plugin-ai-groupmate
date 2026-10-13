from typing import Any, cast

from sqlalchemy import JSON, Text, String
from sqlalchemy.types import TypeDecorator
from sqlalchemy.engine import Dialect
from nonebot_plugin_orm import Model


def _clean_json(value: Any) -> Any:
    """Copy JSON containers, removing NUL from strings without changing other values."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            safe_key = _clean_json(key)
            if safe_key in cleaned:
                raise ValueError("JSON keys collide after removing NUL characters")
            cleaned[safe_key] = _clean_json(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


class _NulSafeString(TypeDecorator[str]):
    impl = String
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: Dialect) -> str | None:
        return value.replace("\x00", "") if value is not None else None


class _NulSafeText(_NulSafeString):
    impl = Text
    cache_ok = True


class _NulSafeJSON(TypeDecorator[Any]):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        return _clean_json(value)

    def coerce_compared_value(self, op: Any, value: Any) -> Any:
        # JSON index operands must retain SQLAlchemy's native string/int types.
        return cast(JSON, self.impl).coerce_compared_value(op, value)


class NulSafeModel(Model):
    """Apply safe binding to this plugin's columns, including Core UPDATE/WHERE."""

    __abstract__ = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # NoneBot ORM checks inherited __abstract__ when assigning table names.
        if "__abstract__" not in cls.__dict__:
            cls.__abstract__ = False
        super().__init_subclass__(**kwargs)
        if cls.__abstract__:
            return
        for column in cls.__table__.columns:
            if isinstance(column.type, Text):
                column.type = _NulSafeText(column.type.length, collation=column.type.collation)
            elif isinstance(column.type, String):
                column.type = _NulSafeString(column.type.length, collation=column.type.collation)
            elif isinstance(column.type, JSON):
                column.type = _NulSafeJSON(none_as_null=column.type.none_as_null)
