"""Persist optional image generation parameters for background jobs."""

import sqlalchemy as sa
from alembic import op

revision = "f3a7b9c2d614"
down_revision = "e9f2a3b4c615"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if not name:
        op.add_column("nonebot_plugin_ai_groupmate_drawingjob", sa.Column("image_options", sa.JSON(), nullable=True))


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_column("nonebot_plugin_ai_groupmate_drawingjob", "image_options")
