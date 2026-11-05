"""Persist ordered replies waiting for background drawings."""

import sqlalchemy as sa
from alembic import op

revision = "e9f2a3b4c615"
down_revision = "d7a3e6c9b154"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    drawing = "nonebot_plugin_ai_groupmate_drawingjob"
    table = "nonebot_plugin_ai_groupmate_ordereddeliveryjob"
    op.add_column(drawing, sa.Column("batch_id", sa.String(32), nullable=True))
    op.create_index("ix_nonebot_plugin_ai_groupmate_drawingjob_batch_id", drawing, ["batch_id"])
    op.create_table(table,
        sa.Column("job_id", sa.String(32), primary_key=True),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("is_private", sa.Boolean(), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=False),
        sa.Column("bot_name", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("request_id", sa.String(255), nullable=False),
        sa.Column("reply_to_id", sa.String(255), nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("parts", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("claim_token", sa.String(32), nullable=True),
        sa.Column("result_message_id", sa.String(255), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("platform", "session_id", "is_private", "user_id", "request_id", "fingerprint", name="uq_ordered_delivery_request"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_ordered_delivery_status_created", table, ["status", "created_at"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_ordereddeliveryjob")
        op.drop_index("ix_nonebot_plugin_ai_groupmate_drawingjob_batch_id", table_name="nonebot_plugin_ai_groupmate_drawingjob")
        op.drop_column("nonebot_plugin_ai_groupmate_drawingjob", "batch_id")
