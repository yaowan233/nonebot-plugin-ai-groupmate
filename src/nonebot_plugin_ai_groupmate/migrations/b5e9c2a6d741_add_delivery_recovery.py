"""Add idempotency receipts for recovery of already acknowledged sends."""

import sqlalchemy as sa
from alembic import op

revision = "b5e9c2a6d741"
down_revision = "a4d8f0c2b615"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    table = "nonebot_plugin_ai_groupmate_deliveryrecoveryreceipt"
    op.create_table(table, sa.Column("receipt_id", sa.String(32), primary_key=True),
                    sa.Column("created_at", sa.DateTime(), nullable=False),
                    info={"bind_key": "nonebot_plugin_ai_groupmate"})
    op.create_index("ix_delivery_recovery_created_at", table, ["created_at"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_deliveryrecoveryreceipt")
