"""add group reply probability

迁移 ID: b3a1d7e9c204
父迁移: f8b3c1d6a204
创建时间: 2026-08-25

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3a1d7e9c204"
down_revision: str | Sequence[str] | None = "f8b3c1d6a204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_groupmodelconfig",
        schema=None,
    ) as batch_op:
        batch_op.add_column(
            sa.Column("reply_probability", sa.Float(), nullable=True)
        )


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_groupmodelconfig",
        schema=None,
    ) as batch_op:
        batch_op.drop_column("reply_probability")
