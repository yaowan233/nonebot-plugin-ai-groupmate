"""Add an incremental platform-message index without rewriting old history."""

import sqlalchemy as sa
from alembic import op

revision = "c6f1d3b8a952"
down_revision = "b5e9c2a6d741"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    table = "nonebot_plugin_ai_groupmate_historymessageindex"
    op.create_table(table,
        sa.Column("history_id", sa.Integer(), sa.ForeignKey("nonebot_plugin_ai_groupmate_chathistory.msg_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("platform_id", sa.String(255), primary_key=True),
        sa.Column("session_id", sa.String(255), nullable=False),
        info={"bind_key": "nonebot_plugin_ai_groupmate"})
    op.create_index("ix_history_message_lookup", table, ["session_id", "platform_id"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_historymessageindex")
