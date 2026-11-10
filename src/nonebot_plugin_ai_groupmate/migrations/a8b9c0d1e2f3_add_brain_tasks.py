"""Add durable cross-group task and reply correlation records."""

import sqlalchemy as sa
from alembic import op

revision = "a8b9c0d1e2f3"
down_revision = "f3a7b9c2d614"
branch_labels = None
depends_on = None


def upgrade(name: str = "") -> None:
    if name:
        return
    op.create_table(
        "nonebot_plugin_ai_groupmate_braintask",
        sa.Column("task_id", sa.String(32), primary_key=True),
        sa.Column("brain_id", sa.String(255), nullable=False),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("source_scene", sa.String(255), nullable=False),
        sa.Column("source_message", sa.String(255), nullable=False),
        sa.Column("source_user", sa.String(255), nullable=False),
        sa.Column("request", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("claim_token", sa.String(32)),
        sa.Column("lease_until", sa.DateTime()),
        sa.Column("deadline", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("brain_id", "platform", "source_scene", "source_message", name="uq_brain_task_source"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_nonebot_plugin_ai_groupmate_braintask_status", "nonebot_plugin_ai_groupmate_braintask", ["status"])
    op.create_table(
        "nonebot_plugin_ai_groupmate_brainwait",
        sa.Column("task_id", sa.String(32), sa.ForeignKey("nonebot_plugin_ai_groupmate_braintask.task_id"), primary_key=True),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("target_scene", sa.String(255), nullable=False),
        sa.Column("question_message", sa.String(255), nullable=False),
        sa.Column("answer_message", sa.String(255)),
        sa.Column("answer_user", sa.String(255)),
        sa.Column("answer_text", sa.Text()),
        sa.Column("deadline", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_brain_wait_lookup", "nonebot_plugin_ai_groupmate_brainwait", ["platform", "target_scene", "question_message", "status"])
    op.create_table(
        "nonebot_plugin_ai_groupmate_brainaction",
        sa.Column("action_id", sa.String(32), primary_key=True),
        sa.Column("task_id", sa.String(32), sa.ForeignKey("nonebot_plugin_ai_groupmate_braintask.task_id"), nullable=False),
        sa.Column("logical_step", sa.String(128), nullable=False),
        sa.Column("effect_slot", sa.String(128), nullable=False),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("target_scene", sa.String(255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("claim_token", sa.String(32)),
        sa.Column("lease_until", sa.DateTime()),
        sa.Column("receipt_message_id", sa.String(255)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("task_id", "logical_step", "effect_slot", "platform", "target_scene", name="uq_brain_action_effect"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_nonebot_plugin_ai_groupmate_brainaction_status", "nonebot_plugin_ai_groupmate_brainaction", ["status"])
    op.create_table(
        "nonebot_plugin_ai_groupmate_braininteraction",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=False),
        sa.Column("scene_id", sa.String(255), nullable=False),
        sa.Column("turn_id", sa.String(255), nullable=False),
        sa.Column("explicitly_addressed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("platform", "bot_id", "scene_id", "turn_id", name="uq_brain_interaction_turn"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_brain_interaction_scene_time", "nonebot_plugin_ai_groupmate_braininteraction", ["platform", "bot_id", "scene_id", "created_at"])
    op.create_table(
        "nonebot_plugin_ai_groupmate_brainobservationmessage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=False),
        sa.Column("scene_id", sa.String(255), nullable=False),
        sa.Column("message_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("platform", "bot_id", "scene_id", "message_id", name="uq_brain_observation_message"),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_brain_observation_scene_time", "nonebot_plugin_ai_groupmate_brainobservationmessage", ["platform", "bot_id", "scene_id", "created_at"])
    op.create_table(
        "nonebot_plugin_ai_groupmate_brainsceneobservation",
        sa.Column("platform", sa.String(64), primary_key=True),
        sa.Column("bot_id", sa.String(255), primary_key=True),
        sa.Column("scene_id", sa.String(255), primary_key=True),
        sa.Column("last_message_id", sa.Integer(), nullable=False),
        sa.Column("next_due", sa.DateTime(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("digest", sa.Text(), nullable=False),
        sa.Column("source_ids", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_table(
        "nonebot_plugin_ai_groupmate_brainobservationrun",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("platform", sa.String(64), nullable=False),
        sa.Column("bot_id", sa.String(255), nullable=False),
        sa.Column("scene_id", sa.String(255), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )
    op.create_index("ix_nonebot_plugin_ai_groupmate_brainobservationrun_started_at", "nonebot_plugin_ai_groupmate_brainobservationrun", ["started_at"])


def downgrade(name: str = "") -> None:
    if not name:
        op.drop_table("nonebot_plugin_ai_groupmate_brainobservationrun")
        op.drop_table("nonebot_plugin_ai_groupmate_brainsceneobservation")
        op.drop_table("nonebot_plugin_ai_groupmate_brainobservationmessage")
        op.drop_table("nonebot_plugin_ai_groupmate_braininteraction")
        op.drop_table("nonebot_plugin_ai_groupmate_brainaction")
        op.drop_table("nonebot_plugin_ai_groupmate_brainwait")
        op.drop_table("nonebot_plugin_ai_groupmate_braintask")
