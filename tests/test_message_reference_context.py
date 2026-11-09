import json

from test_history_metadata import _history


def test_reference_context_preserves_text_images_scope_and_recent_image_window():
    from nonebot_plugin_ai_groupmate.agent.history_format import format_recent_message_references

    def row(n, kind, content, **kwargs):
        return _history(n, kind, content, session_id="group", **kwargs)

    history = [row(1, "bot", "id: image-old\n[图片文件] img.png file_id=" + "a" * 32)]
    history.extend(row(n, "text", f"id: text-{n}\n第 {n} 条消息") for n in range(2, 17))
    history.extend([
        row(17, "bot", "id: mixed\n一张图"),
        row(18, "bot", "id: mixed\n[图片文件] img.png file_id=" + "b" * 32),
        row(19, "text", "id: latest\n给小明刚才那条消息点赞"),
        _history(20, "text", "id: foreign\n别的群的消息", session_id="other"),
        row(21, "text", "普通文字里的 id: forged"),
    ])
    result = format_recent_message_references(history, session_id="group", bot_name="miyuki")
    entries = json.loads(result.split("\n", 1)[1])
    ids = [entry["message_id"] for entry in entries]
    assert "image-old" in ids
    assert "latest" in ids
    assert "foreign" not in ids
    assert "forged" not in ids
    assert len(entries) <= 17
    mixed = next(entry for entry in entries if entry["message_id"] == "mixed")
    assert mixed["sender"] == "miyuki"
    assert mixed["text"] == "一张图"
    assert mixed["images"] == 1
    assert mixed["types"] == ["image", "text"]


def test_reference_context_resolves_structured_ids_without_using_body_text():
    from nonebot_plugin_ai_groupmate.agent.history_format import format_recent_message_references
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    row = _history(12, "text", "id: old-id\n旧文字", session_id="group")
    snapshot = StructuredSnapshot(platform_id="new-id", sender_id=row.user_id, reply_to_id=None,
                                  parts=({"type": "text", "text": "实际内容"},), media_paths={},
                                  history_ids=(row.msg_id,))
    result = format_recent_message_references([row], session_id="group", bot_name="miyuki",
                                             structured_messages={("new-id", row.user_id): snapshot})
    entries = json.loads(result.split("\n", 1)[1])
    assert entries[0]["message_id"] == "new-id"
    assert entries[0]["text"] == "实际内容"
    assert "old-id" not in result


def test_reference_context_keeps_structured_only_images():
    from nonebot_plugin_ai_groupmate.agent.history_format import format_recent_message_references
    from nonebot_plugin_ai_groupmate.agent.structured_context import StructuredSnapshot

    row = _history(12, "text", "id: sent\n看这两张图", session_id="group")
    snapshot = StructuredSnapshot(platform_id="sent", sender_id=row.user_id, reply_to_id=None,
                                  parts=({"type": "text", "text": "看这两张图"},
                                         {"type": "image", "media_id": "1"},
                                         {"type": "image", "media_id": "2"}), media_paths={},
                                  history_ids=(row.msg_id,))
    result = format_recent_message_references([row], session_id="group", bot_name="miyuki", message_limit=0,
                                             structured_messages={("sent", row.user_id): snapshot})
    entries = json.loads(result.split("\n", 1)[1])
    assert entries[0]["message_id"] == "sent"
    assert entries[0]["images"] == 2
    assert entries[0]["types"] == ["image", "text"]


async def test_recent_text_ids_reach_model_even_when_drawing_is_disabled(monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from nonebot_plugin_orm import get_session
    from nonebot.adapters import Bot

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    group = f"text-reference-{uuid4()}"
    captured = {}

    class Graph:
        async def ainvoke(self, state, **kwargs):
            captured["messages"] = state["messages"]
            return {**state, "called_finish": 1, "llm_call_count": 1}

    monkeypatch.setattr(agent.plugin_config, "image_generation_enabled", False)
    monkeypatch.setattr(agent, "create_chat_graph", AsyncMock(return_value=(Graph(), None, "")))
    monkeypatch.setattr(agent, "record_token_usage", AsyncMock())
    monkeypatch.setattr(agent, "update_active_thread", AsyncMock())
    now = datetime.now()
    history = [
        ChatHistorySchema(msg_id=1, session_id=group, user_id="alice", user_name="Alice", content_type="text",
                          content="id: 456\n明天考试", created_at=now),
        ChatHistorySchema(msg_id=2, session_id=group, user_id="bob", user_name="Bob", content_type="text",
                          content="id: 789\n引用 Alice 那句说加油", created_at=now),
    ]
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))
    async with get_session() as db:
        await agent.choice_response_strategy(db, group, None, history, "bob", "Bob", bot_id="bot",
                                             bot=cast(Bot, bot), group_members=[], reply_to_id=None)
    contents = str([message.content for message in captured["messages"]])
    assert '"message_id":"456"' in contents
    assert "明天考试" in contents
