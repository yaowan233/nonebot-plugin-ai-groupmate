"""add token usage cache creation

迁移 ID: f4c9b2e8a731
父迁移: d9f0a8c7e2b1
创建时间: 2026-07-08

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f4c9b2e8a731"
down_revision: str | Sequence[str] | None = "d9f0a8c7e2b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table("nonebot_plugin_ai_groupmate_tokenusage", schema=None) as batch_op:
        batch_op.add_column(sa.Column("cache_creation_tokens", sa.Integer(), nullable=False, server_default="0"))


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table("nonebot_plugin_ai_groupmate_tokenusage", schema=None) as batch_op:
        batch_op.drop_column("cache_creation_tokens")
