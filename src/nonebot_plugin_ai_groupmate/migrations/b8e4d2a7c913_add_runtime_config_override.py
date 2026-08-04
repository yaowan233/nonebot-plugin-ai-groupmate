"""add runtime config override

迁移 ID: b8e4d2a7c913
父迁移: e7a1c5d9b3f2
创建时间: 2026-08-04

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8e4d2a7c913"
down_revision: str | Sequence[str] | None = "e7a1c5d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    op.create_table(
        "nonebot_plugin_ai_groupmate_runtimeconfigoverride",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("overrides", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_nonebot_plugin_ai_groupmate_runtimeconfigoverride"),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if name:
        return

    op.drop_table("nonebot_plugin_ai_groupmate_runtimeconfigoverride")
