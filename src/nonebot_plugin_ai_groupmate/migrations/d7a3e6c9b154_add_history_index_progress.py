"""Track resumable, bounded history-index coverage by conversation."""

import sqlalchemy as sa
from alembic import op

revision = "d7a3e6c9b154"
down_revision = "c6f1d3b8a952"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    table = "nonebot_plugin_ai_groupmate_historyindexprogress"
    op.create_table(table,
        sa.Column("session_id", sa.String(255), primary_key=True),
        sa.Column("cutoff_id", sa.Integer(), nullable=False),
        sa.Column("last_created_at", sa.DateTime(), nullable=True),
        sa.Column("last_history_id", sa.Integer(), nullable=False),
        sa.Column("complete", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        info={"bind_key": "nonebot_plugin_ai_groupmate"})
    op.create_index("ix_history_index_progress_pending", table, ["complete", "updated_at"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_historyindexprogress")
