import json
import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from nonebot.adapters import Bot, Event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sender", "age_minutes", "found", "expected"),
    [
        ("bot", 1, True, "message_recalled"),
        ("user", 1, True, "permission_denied"),
        ("bot", 6, True, "recall_window_expired"),
        ("bot", 1, False, "message_not_found"),
    ],
)
async def test_private_recall_permissions(monkeypatch, sender, age_minutes, found, expected):
    from nonebot_plugin_ai_groupmate.agent import recall_tools

    history = SimpleNamespace(
        content="id: 123\nhello",
        content_type=sender if sender == "bot" else "text",
        user_id=sender,
        created_at=datetime.datetime.now() - datetime.timedelta(minutes=age_minutes),
    )

    class Session:
        def __init__(self):
            self.added = []
            self.committed = False

        async def execute(self, statement):
            assert "private-1" in statement.compile().params.values()
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [history] if found else []))

        async def commit(self):
            self.committed = True

        def add(self, row):
            self.added.append(row)

    session = Session()
    bot = SimpleNamespace(self_id="9000")
    event = SimpleNamespace(message_type="private", user_id=1001)
    recall = AsyncMock()
    monkeypatch.setattr(recall_tools, "message_recall", recall)
    reader = recall_tools.create_recall_message_tool(
        session, "private-1", None,
        bot_name="bot", has_admin_permission=False, bot=cast(Bot, bot), event=cast(Event, event),
    )
    result = json.loads(await reader.ainvoke({"target": "content", "target_text": "hello"}))

    assert result["reason_code"] == expected
    if expected == "message_recalled":
        recall.assert_awaited_once_with(message_id="123", event=event, bot=bot)
        assert session.committed
        assert session.added[0].session_id == "private-1"
        assert result["delivery_state"] == "completed"
        assert "123" not in json.dumps(result)
        assert "123" not in session.added[0].content
    else:
        recall.assert_not_awaited()
        assert not session.added


@pytest.mark.parametrize(
    ("arguments", "rows", "reply_to_id", "expected_id", "expected_reason"),
    [
        ({}, [("2", "user", "request"), ("1", "bot", "hello")], None, "1", "message_recalled"),
        ({"target": "reply"}, [("1", "bot", "hello")], "1", "1", "message_recalled"),
        ({"target": "reply"}, [("1", "bot", "hello")], None, None, "missing_reply"),
        ({"target": "content", "target_text": "hello"}, [("2", "bot", "hello"), ("1", "bot", "hello")], None, None, "ambiguous_message"),
        ({"target": "content", "target_text": "hello", "sender_name": "user"}, [("2", "user", "hello"), ("1", "bot", "hello")], None, "2", "message_recalled"),
        ({"target": "content"}, [], None, None, "missing_target_text"),
        ({}, [("system", "bot", "action"), ("unknown", "bot", "hello")], None, None, "message_not_found"),
    ],
)
async def test_recall_resolves_target_without_exposing_ids(monkeypatch, arguments, rows, reply_to_id, expected_id, expected_reason):
    from nonebot_plugin_ai_groupmate.agent import recall_tools

    history = [SimpleNamespace(
        content=f"id: {mid}\n{body}", content_type="bot" if sender == "bot" else "text",
        user_id=sender, user_name=sender, created_at=datetime.datetime.now(),
    ) for mid, sender, body in rows]
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: history))),
        commit=AsyncMock(), add=lambda row: None,
    )
    bot, event = SimpleNamespace(), SimpleNamespace()
    recall = AsyncMock()
    monkeypatch.setattr(recall_tools, "message_recall", recall)
    reader = recall_tools.create_recall_message_tool(
        session, "group-1", None, bot_name="bot", has_admin_permission=True,
        bot=bot, event=event, reply_to_id=reply_to_id,
    )
    assert "target_msg_id" not in reader.args
    result = json.loads(await reader.ainvoke(arguments))
    assert result["reason_code"] == expected_reason
    assert "message_id" not in result.get("data", {})
    if expected_id is not None:
        recall.assert_awaited_once_with(message_id=expected_id, event=event, bot=bot)
    else:
        recall.assert_not_awaited()
