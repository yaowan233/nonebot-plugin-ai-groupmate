"""add global model group usage

迁移 ID: c4e2f8a1d305
父迁移: b3a1d7e9c204
创建时间: 2026-08-25

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e2f8a1d305"
down_revision: str | Sequence[str] | None = "b3a1d7e9c204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    op.create_table(
        "nonebot_plugin_ai_groupmate_globalmodelgroupusage",
        sa.Column("group_id", sa.String(), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("used_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "group_id",
            name=op.f(
                "pk_nonebot_plugin_ai_groupmate_globalmodelgroupusage"
            ),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_globalmodelgroupusage",
        schema=None,
    ) as batch_op:
        batch_op.create_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_globalmodelgroupusage_usage_date"
            ),
            ["usage_date"],
            unique=False,
        )


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_globalmodelgroupusage",
        schema=None,
    ) as batch_op:
        batch_op.drop_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_globalmodelgroupusage_usage_date"
            )
        )
    op.drop_table("nonebot_plugin_ai_groupmate_globalmodelgroupusage")
