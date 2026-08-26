"""add private model config and ticket scope

迁移 ID: e6a4b0c3d812
父迁移: d5f3a9b2c701
创建时间: 2026-08-26

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e6a4b0c3d812"
down_revision: str | Sequence[str] | None = "d5f3a9b2c701"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_pendinggroupconfig",
        schema=None,
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "scope",
                sa.String(length=16),
                server_default="group",
                nullable=False,
            )
        )
        batch_op.create_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_scope"
            ),
            ["scope"],
            unique=False,
        )

    op.create_table(
        "nonebot_plugin_ai_groupmate_privatemodelconfig",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("api_format", sa.String(length=16), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("api_key_ciphertext", sa.Text(), nullable=False),
        sa.Column("chat_model", sa.String(), nullable=False),
        sa.Column("chat_multimodal", sa.Boolean(), nullable=False),
        sa.Column("allow_global_fallback", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_tested_at", sa.DateTime(), nullable=True),
        sa.Column("last_test_status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint(
            "user_id",
            name=op.f(
                "pk_nonebot_plugin_ai_groupmate_privatemodelconfig"
            ),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if name:
        return

    op.drop_table("nonebot_plugin_ai_groupmate_privatemodelconfig")
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_pendinggroupconfig",
        schema=None,
    ) as batch_op:
        batch_op.drop_index(
            batch_op.f(
                "ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_scope"
            )
        )
        batch_op.drop_column("scope")
