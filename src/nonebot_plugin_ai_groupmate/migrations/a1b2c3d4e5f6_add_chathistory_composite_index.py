"""add chathistory composite index

迁移 ID: a1b2c3d4e5f6
父迁移: 811f4ae4bcd1
创建时间: 2026-04-02

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "811f4ae4bcd1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_index(
        "ix_chat_session_time",
        "nonebot_plugin_ai_groupmate_chathistory",
        ["session_id", "created_at"],
        unique=False,
    )


def downgrade(name: str = "") -> None:
    if name:
        return
    op.drop_index(
        "ix_chat_session_time",
        table_name="nonebot_plugin_ai_groupmate_chathistory",
    )
