"""Persist scheduled messages and Agent tasks."""

import sqlalchemy as sa
from alembic import op

revision = "c8f2a6d1e905"
down_revision = "b2e8d4a6c913"
branch_labels = None
depends_on = None

TABLE = "nonebot_plugin_ai_groupmate_scheduledtask"


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        TABLE,
        sa.Column("job_id", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("is_private", sa.Boolean(), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=True),
        sa.Column("bot_name", sa.String(255), nullable=False),
        sa.Column("task_type", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("run_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("claim_token", sa.String(32), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_scheduled_task_due", TABLE, ["status", "run_at"])
    op.create_index("ix_scheduled_task_scope", TABLE, ["session_id", "is_private", "bot_id", "status"])
    op.create_index("ix_scheduled_task_lease", TABLE, ["status", "lease_until"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table(TABLE)
