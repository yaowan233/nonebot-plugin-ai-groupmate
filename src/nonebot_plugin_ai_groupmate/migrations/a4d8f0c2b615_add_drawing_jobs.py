"""Persist drawing jobs independently from chat execution."""

import sqlalchemy as sa
from alembic import op

revision = "a4d8f0c2b615"
down_revision = "e1c7a5b9d240"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    table = "nonebot_plugin_ai_groupmate_drawingjob"
    op.create_table(
        table,
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
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("reference_file_ids", sa.JSON(), nullable=False),
        sa.Column("size", sa.String(32), nullable=False),
        sa.Column("transparent_background", sa.Boolean(), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("quality", sa.String(16), nullable=False),
        sa.Column("endpoint_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("claim_token", sa.String(32), nullable=True),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("result_file_id", sa.String(32), nullable=True),
        sa.Column("result_message_id", sa.String(255), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("platform", "session_id", "is_private", "user_id", "request_id", "fingerprint", name="uq_drawing_job_request"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_drawing_job_status_created", table, ["status", "created_at"])
    op.create_index("ix_drawing_job_status_lease", table, ["status", "lease_until"])
    op.create_index("ix_drawing_job_scope", table, ["platform", "session_id", "is_private", "user_id", "created_at"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_drawingjob")
