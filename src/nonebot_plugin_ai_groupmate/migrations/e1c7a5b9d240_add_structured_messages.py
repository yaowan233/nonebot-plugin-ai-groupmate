"""Add structured messages without changing existing chat history."""

import sqlalchemy as sa
from alembic import op

revision = "e1c7a5b9d240"
down_revision = "f2a6c8e4b913"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        "nonebot_plugin_ai_groupmate_structuredmessage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("platform_id", sa.String(255), nullable=False),
        sa.Column("sender_id", sa.String(255), nullable=False),
        sa.Column("sender_name", sa.Text(), nullable=False),
        sa.Column("sender_kind", sa.String(16), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=False),
        sa.Column("reply_to_id", sa.String(255), nullable=True),
        sa.Column("parts", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("platform", "scope", "session_id", "platform_id", name="uq_structured_message_scope_platform"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_structured_message_session_time", "nonebot_plugin_ai_groupmate_structuredmessage", ["platform", "scope", "session_id", "created_at"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_structuredmessage")
