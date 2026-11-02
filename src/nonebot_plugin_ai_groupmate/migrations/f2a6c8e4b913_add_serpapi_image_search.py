"""Add isolated SerpApi image search cache and usage counters."""

import sqlalchemy as sa
from alembic import op

revision = "f2a6c8e4b913"
down_revision = "c8f2a6d1e905"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        "nonebot_plugin_ai_groupmate_serpapiimagecache",
        sa.Column("image_hash", sa.String(64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("image_hash"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index(
        "ix_nonebot_plugin_ai_groupmate_serpapiimagecache_created_at",
        "nonebot_plugin_ai_groupmate_serpapiimagecache",
        ["created_at"],
    )
    op.create_table(
        "nonebot_plugin_ai_groupmate_serpapiimageusage",
        sa.Column("usage_month", sa.String(7), nullable=False),
        sa.Column("used_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("usage_month"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_serpapiimageusage")
        op.drop_table("nonebot_plugin_ai_groupmate_serpapiimagecache")
