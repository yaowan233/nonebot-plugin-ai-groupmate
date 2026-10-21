"""persist adaptive scheduling for inline affinity reviews"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d6b9a3e2c710"
down_revision: str | Sequence[str] | None = "c9a2e7b4d610"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table("nonebot_plugin_ai_groupmate_groupaffinity") as batch:
        batch.add_column(sa.Column("review_after", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("review_interval", sa.Integer(), nullable=False, server_default="900"))


def downgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table("nonebot_plugin_ai_groupmate_groupaffinity") as batch:
        batch.drop_column("review_interval")
        batch.drop_column("review_after")
