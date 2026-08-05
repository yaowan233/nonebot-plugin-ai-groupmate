"""add media embedding version

迁移 ID: c7f1a2d9e4b6
父迁移: b8e4d2a7c913
创建时间: 2026-08-05

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7f1a2d9e4b6"
down_revision: str | Sequence[str] | None = "b8e4d2a7c913"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_mediastorage",
        schema=None,
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "embedding_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.create_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_mediastorage_embedding_version"
            ),
            ["embedding_version"],
            unique=False,
        )


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_mediastorage",
        schema=None,
    ) as batch_op:
        batch_op.drop_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_mediastorage_embedding_version"
            )
        )
        batch_op.drop_column("embedding_version")
