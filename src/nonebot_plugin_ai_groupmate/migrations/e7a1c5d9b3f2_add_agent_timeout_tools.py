"""add agent timeout tool details

迁移 ID: e7a1c5d9b3f2
父迁移: c2d7e4f1a9b0
创建时间: 2026-07-28

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7a1c5d9b3f2"
down_revision: str | Sequence[str] | None = "c2d7e4f1a9b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_tokenusage",
        schema=None,
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "agent_tool_timeout_tools",
                sa.JSON(),
                nullable=False,
                server_default="[]",
            )
        )


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_tokenusage",
        schema=None,
    ) as batch_op:
        batch_op.drop_column("agent_tool_timeout_tools")
