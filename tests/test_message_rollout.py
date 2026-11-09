from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime

from langchain_core.messages import HumanMessage


def test_rollout_scope_defaults_and_switch():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.message_rollout import message_pilot_enabled

    cfg = ScopedConfig()
    assert message_pilot_enabled(cfg, "g", is_private=False)
    assert message_pilot_enabled(cfg, "g", is_private=True)
    cfg.message_refactor_pilot_private = False
    cfg.message_refactor_pilot_groups = ["g"]
    assert message_pilot_enabled(cfg, "g", is_private=False)
    assert not message_pilot_enabled(cfg, "other", is_private=False)
    cfg.message_refactor_pilot_groups = ["*"]
    assert message_pilot_enabled(cfg, "other", is_private=False)
    assert not message_pilot_enabled(cfg, "g", is_private=True)
    cfg.message_refactor_pilot_private = True
    assert message_pilot_enabled(cfg, "g", is_private=True)


def test_switch_rebuilds_cached_history():
    from test_history_metadata import _history

    from nonebot_plugin_ai_groupmate.agent.conversation import ActiveConversationThread, build_append_only_history, active_conversation_threads

    group = f"rollout-cache-{uuid4()}"
    active_conversation_threads[group] = ActiveConversationThread(
        messages=[HumanMessage(content="旧缓存")], last_msg_id=1, updated_at=datetime.now(), history_variant="structured")
    try:
        messages, _, reused = build_append_only_history(group, [_history(1, "text", "id: 1\n新的")],
            format_history=lambda *_: [HumanMessage(content="重新读取")], history_variant="legacy")
        assert not reused
        assert messages[0].content == "重新读取"
    finally:
        active_conversation_threads.pop(group, None)


def test_unbound_structured_image_falls_back_to_legacy(tmp_path):
    from test_history_metadata import _history
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    (tmp_path / "picture.png").write_bytes(png())
    text = _history(1, "text", "id: 1\n看图片")
    image = _history(2, "image", "id: 1\npicture.png", media_id=9)
    snapshot = StructuredSnapshot(platform_id="1", sender_id=text.user_id, reply_to_id=None,
                                  parts=({"type": "text", "text": "看"}, {"type": "image"}), media_paths={})
    messages = format_chat_history([text, image], pic_dir=tmp_path, bot_name="bot",
                                   structured_messages={("1", text.user_id): snapshot})
    assert isinstance(messages[0].content, list)
    assert any(isinstance(block, dict) and block.get("type") == "image_url" for block in messages[0].content)


async def test_disabled_recovery_does_not_create_journal(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.delivery import finalize_delivery

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    group = f"legacy-delivery-{uuid4()}"
    async with get_session() as db:
        receipt = await finalize_delivery(db, group, SimpleNamespace(msg_ids=[{"message_id": "123"}]),
            recovery_enabled=False,
            record=lambda mid: [ChatHistory(session_id=group, user_id="bot", user_name="bot",
                                            content_type="bot", content=f"id: {mid}\nhello")])
        assert receipt.history_saved
    assert not (tmp_path / "delivery_recovery").exists()


async def test_disabled_pilot_does_not_run_index_maintenance(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin.plugin_config, "message_refactor_pilot_groups", [])
    monkeypatch.setattr(plugin.plugin_config, "message_refactor_pilot_private", False)
    def forbidden():
        raise AssertionError("disabled pilot should not open a backfill session")
    monkeypatch.setattr(plugin, "get_session", forbidden)
    await plugin._backfill_history_index()
