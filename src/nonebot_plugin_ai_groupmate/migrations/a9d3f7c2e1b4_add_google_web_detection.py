"""add Google Web Detection cache and usage tables

迁移 ID: a9d3f7c2e1b4
父迁移: e6a4b0c3d812
创建时间: 2026-09-02

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9d3f7c2e1b4"
down_revision: str | Sequence[str] | None = "e6a4b0c3d812"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    op.create_table(
        "nonebot_plugin_ai_groupmate_googlewebdetectioncache",
        sa.Column("image_hash", sa.String(length=64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "image_hash",
            name=op.f(
                "pk_nonebot_plugin_ai_groupmate_googlewebdetectioncache"
            ),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_googlewebdetectioncache",
        schema=None,
    ) as batch_op:
        batch_op.create_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_googlewebdetectioncache_created_at"
            ),
            ["created_at"],
            unique=False,
        )

    op.create_table(
        "nonebot_plugin_ai_groupmate_googlewebdetectionusage",
        sa.Column("usage_month", sa.String(length=7), nullable=False),
        sa.Column("used_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "usage_month",
            name=op.f(
                "pk_nonebot_plugin_ai_groupmate_googlewebdetectionusage"
            ),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if name:
        return

    op.drop_table("nonebot_plugin_ai_groupmate_googlewebdetectionusage")
    op.drop_table("nonebot_plugin_ai_groupmate_googlewebdetectioncache")
