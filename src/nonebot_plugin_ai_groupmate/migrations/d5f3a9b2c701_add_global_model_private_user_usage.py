"""add global model private user usage

迁移 ID: d5f3a9b2c701
父迁移: c4e2f8a1d305
创建时间: 2026-08-26

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5f3a9b2c701"
down_revision: str | Sequence[str] | None = "c4e2f8a1d305"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    op.create_table(
        "nonebot_plugin_ai_groupmate_globalmodelprivateuserusage",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("used_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "user_id",
            name=op.f(
                "pk_nonebot_plugin_ai_groupmate_globalmodelprivateuserusage"
            ),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_globalmodelprivateuserusage",
        schema=None,
    ) as batch_op:
        batch_op.create_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_globalmodelprivateuserusage_usage_date"
            ),
            ["usage_date"],
            unique=False,
        )


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_globalmodelprivateuserusage",
        schema=None,
    ) as batch_op:
        batch_op.drop_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_globalmodelprivateuserusage_usage_date"
            )
        )
    op.drop_table(
        "nonebot_plugin_ai_groupmate_globalmodelprivateuserusage"
    )
