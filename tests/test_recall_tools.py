import json
import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


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
        bot_name="bot", has_admin_permission=False, bot=bot, event=event,
    )
    result = json.loads(await reader.ainvoke({"target_msg_id": "123"}))

    assert result["reason_code"] == expected
    if expected == "message_recalled":
        recall.assert_awaited_once_with(message_id="123", event=event, bot=bot)
        assert session.committed
        assert session.added[0].session_id == "private-1"
        assert result["delivery_state"] == "completed"
    else:
        recall.assert_not_awaited()
        assert not session.added
