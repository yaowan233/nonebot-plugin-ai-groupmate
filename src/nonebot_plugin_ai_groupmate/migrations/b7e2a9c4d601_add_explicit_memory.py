"""add scoped explicit user memories"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e2a9c4d601"
down_revision: str | Sequence[str] | None = "a9d3f7c2e1b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        "nonebot_plugin_ai_groupmate_explicitmemory",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=False),
        sa.Column("is_private", sa.Boolean(), nullable=False),
        sa.Column("title", sa.String(60), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("lifetime", sa.String(16), nullable=False),
        sa.Column("source_quote", sa.Text(), nullable=False),
        sa.Column("source_msg_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("session_id", "user_id", "bot_id", "is_private", "title", name="uq_explicit_memory_scope_title"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_explicit_memory_expiry", "nonebot_plugin_ai_groupmate_explicitmemory", ["lifetime", "expires_at", "last_used_at"])


def downgrade(name: str = "") -> None:
    if name:
        return
    op.drop_index("ix_explicit_memory_expiry", table_name="nonebot_plugin_ai_groupmate_explicitmemory")
    op.drop_table("nonebot_plugin_ai_groupmate_explicitmemory")
