"""Track resumable animation annotation independently of embedding versions."""

import sqlalchemy as sa
from alembic import op

revision = "a9d4e7c2b615"
down_revision = "f3a7b9c2d614"
branch_labels = None
depends_on = None

TABLE = "nonebot_plugin_ai_groupmate_mediastorage"
INDEX = "ix_nonebot_plugin_ai_groupmate_mediastorage_animation_version"


def upgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table(TABLE) as batch:
        batch.add_column(sa.Column("animation_version", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("animation_attempted_at", sa.DateTime(), nullable=True))
        batch.create_index(INDEX, ["animation_version"])


def downgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_index(INDEX)
        batch.drop_column("animation_attempted_at")
        batch.drop_column("animation_version")
