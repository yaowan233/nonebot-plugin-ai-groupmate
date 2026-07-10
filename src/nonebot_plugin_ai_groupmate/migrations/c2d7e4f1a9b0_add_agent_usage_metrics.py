"""add agent usage metrics

迁移 ID: c2d7e4f1a9b0
父迁移: f4c9b2e8a731
创建时间: 2026-07-10

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2d7e4f1a9b0"
down_revision: str | Sequence[str] | None = "f4c9b2e8a731"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table("nonebot_plugin_ai_groupmate_tokenusage", schema=None) as batch_op:
        batch_op.add_column(sa.Column("agent_llm_calls", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("agent_tool_calls", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("agent_duration_ms", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("agent_tool_timeouts", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("agent_result_truncations", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("agent_side_effect_deduplications", sa.Integer(), nullable=False, server_default="0"))


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table("nonebot_plugin_ai_groupmate_tokenusage", schema=None) as batch_op:
        batch_op.drop_column("agent_side_effect_deduplications")
        batch_op.drop_column("agent_result_truncations")
        batch_op.drop_column("agent_tool_timeouts")
        batch_op.drop_column("agent_duration_ms")
        batch_op.drop_column("agent_tool_calls")
        batch_op.drop_column("agent_llm_calls")
