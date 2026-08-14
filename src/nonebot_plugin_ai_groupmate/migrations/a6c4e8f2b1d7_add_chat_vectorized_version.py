"""add chat vectorized version

迁移 ID: a6c4e8f2b1d7
父迁移: d4e6f8a1b2c3
创建时间: 2026-08-13

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a6c4e8f2b1d7"
down_revision: str | Sequence[str] | None = "d4e6f8a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_chathistory",
        schema=None,
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "vectorized_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.create_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_chathistory_vectorized_version"
            ),
            ["vectorized_version"],
            unique=False,
        )


def downgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_chathistory",
        schema=None,
    ) as batch_op:
        batch_op.drop_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_chathistory_vectorized_version"
            )
        )
        batch_op.drop_column("vectorized_version")
