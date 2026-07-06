"""add token usage

迁移 ID: d9f0a8c7e2b1
父迁移: a1b2c3d4e5f6
创建时间: 2026-07-06

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9f0a8c7e2b1"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    op.create_table(
        "nonebot_plugin_ai_groupmate_tokenusage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("session_type", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_nonebot_plugin_ai_groupmate_tokenusage")),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    with op.batch_alter_table("nonebot_plugin_ai_groupmate_tokenusage", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_model"), ["model"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_request_id"), ["request_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_session_id"), ["session_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_session_type"), ["session_type"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_user_id"), ["user_id"], unique=False)
        batch_op.create_index("ix_token_usage_session_time", ["session_id", "created_at"], unique=False)
        batch_op.create_index("ix_token_usage_user_time", ["user_id", "created_at"], unique=False)


def downgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table("nonebot_plugin_ai_groupmate_tokenusage", schema=None) as batch_op:
        batch_op.drop_index("ix_token_usage_user_time")
        batch_op.drop_index("ix_token_usage_session_time")
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_user_id"))
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_session_type"))
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_session_id"))
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_request_id"))
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_model"))
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_tokenusage_created_at"))

    op.drop_table("nonebot_plugin_ai_groupmate_tokenusage")
