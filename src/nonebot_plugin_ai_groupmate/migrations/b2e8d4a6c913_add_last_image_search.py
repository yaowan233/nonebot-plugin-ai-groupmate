"""Persist the most recent image search evidence per user and conversation."""

import sqlalchemy as sa
from alembic import op

revision = "b2e8d4a6c913"
down_revision = "a9d3f7c2e1b4"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        "nonebot_plugin_ai_groupmate_lastimagesearch",
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("session_id", "user_id"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_lastimagesearch")
