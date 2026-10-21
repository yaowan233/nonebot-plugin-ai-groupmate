"""add persistent per-bot group affinity"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9a2e7b4d610"
down_revision: str | Sequence[str] | None = "b7e2a9c4d601"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        "nonebot_plugin_ai_groupmate_groupaffinity",
        sa.Column("bot_id", sa.String(255), primary_key=True),
        sa.Column("session_id", sa.String(255), primary_key=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("impression", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("last_message_id", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if name:
        return
    op.drop_table("nonebot_plugin_ai_groupmate_groupaffinity")
