"""add group api relay and model config tables

迁移 ID: f8b3c1d6a204
父迁移: a6c4e8f2b1d7
创建时间: 2026-08-24

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f8b3c1d6a204"
down_revision: str | Sequence[str] | None = "a6c4e8f2b1d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return

    op.create_table(
        "nonebot_plugin_ai_groupmate_relayinstanceidentity",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("relay_url", sa.String(), nullable=False),
        sa.Column("instance_id", sa.String(), nullable=False),
        sa.Column("instance_token_ciphertext", sa.Text(), nullable=False),
        sa.Column("public_key_jwk", sa.JSON(), nullable=False),
        sa.Column("private_key_ciphertext", sa.Text(), nullable=False),
        sa.Column("key_id", sa.String(), nullable=False),
        sa.Column("registered_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_nonebot_plugin_ai_groupmate_relayinstanceidentity"),
        ),
        sa.UniqueConstraint(
            "instance_id",
            name=op.f(
                "uq_nonebot_plugin_ai_groupmate_relayinstanceidentity_instance_id"
            ),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_table(
        "nonebot_plugin_ai_groupmate_pendinggroupconfig",
        sa.Column("ticket_id", sa.String(), nullable=False),
        sa.Column("group_id", sa.String(), nullable=False),
        sa.Column("operator_id", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "ticket_id",
            name=op.f("pk_nonebot_plugin_ai_groupmate_pendinggroupconfig"),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_pendinggroupconfig",
        schema=None,
    ) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_group_id"),
            ["group_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_operator_id"),
            ["operator_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_expires_at"),
            ["expires_at"],
            unique=False,
        )
    op.create_table(
        "nonebot_plugin_ai_groupmate_groupmodelconfig",
        sa.Column("group_id", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("api_format", sa.String(length=16), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("api_key_ciphertext", sa.Text(), nullable=False),
        sa.Column("chat_model", sa.String(), nullable=False),
        sa.Column("chat_multimodal", sa.Boolean(), nullable=False),
        sa.Column("allow_global_fallback", sa.Boolean(), nullable=False),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_tested_at", sa.DateTime(), nullable=True),
        sa.Column("last_test_status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint(
            "group_id",
            name=op.f("pk_nonebot_plugin_ai_groupmate_groupmodelconfig"),
        ),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if name:
        return

    op.drop_table("nonebot_plugin_ai_groupmate_groupmodelconfig")
    with op.batch_alter_table(
        "nonebot_plugin_ai_groupmate_pendinggroupconfig",
        schema=None,
    ) as batch_op:
        batch_op.drop_index(
            batch_op.f("ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_expires_at")
        )
        batch_op.drop_index(
            batch_op.f("ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_operator_id")
        )
        batch_op.drop_index(
            batch_op.f("ix_nonebot_plugin_ai_groupmate_pendinggroupconfig_group_id")
        )
    op.drop_table("nonebot_plugin_ai_groupmate_pendinggroupconfig")
    op.drop_table("nonebot_plugin_ai_groupmate_relayinstanceidentity")
