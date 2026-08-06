"""add group media usage index

迁移 ID: d4e6f8a1b2c3
父迁移: c7f1a2d9e4b6
创建时间: 2026-08-06

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "d4e6f8a1b2c3"
down_revision: str | Sequence[str] | None = "c7f1a2d9e4b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_index(
        "ix_chat_session_type_media",
        "nonebot_plugin_ai_groupmate_chathistory",
        ["session_id", "content_type", "media_id"],
        unique=False,
    )


def downgrade(name: str = "") -> None:
    if name:
        return
    op.drop_index(
        "ix_chat_session_type_media",
        table_name="nonebot_plugin_ai_groupmate_chathistory",
    )
