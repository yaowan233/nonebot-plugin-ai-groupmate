import json
import datetime
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_reply_releases_database_connection_before_adapter_send(monkeypatch):
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool

    events: list[str] = []

    class _Result:
        def scalars(self):
            return self

        def first(self):
            return None

    class _Session:
        async def execute(self, statement):
            events.append("query")
            return _Result()

        async def commit(self):
            events.append("commit")

        async def rollback(self):
            events.append("rollback")

        def add(self, value):
            events.append("add")

    async def fake_send(self, *args, **kwargs):
        events.append("send")
        return SimpleNamespace(msg_ids=[{"message_id": "message-1"}])

    monkeypatch.setattr(UniMessage, "send", fake_send)
    reply_tool = create_reply_tool(
        _Session(),
        "group-1",
        bot_name="bot",
        parse_msg_meta=lambda content: (None, None, content),
        group_members=[],
    )

    result = json.loads(
        await reply_tool.ainvoke({"content": "hello", "next_step": "end"})
    )

    assert result["status"] == "sent"
    assert events == ["query", "commit", "send", "add"]


@pytest.mark.asyncio
async def test_chat_vectorization_releases_connection_before_qdrant(monkeypatch):
    from nonebot_plugin_ai_groupmate import utils

    events: list[str] = []

    class _Session:
        async def commit(self):
            events.append("commit")

        async def rollback(self):
            events.append("rollback")

    async def fake_split(*args, **kwargs):
        events.append("query")
        return [[
            SimpleNamespace(
                msg_id=1,
                user_name="tester",
                content="hello",
                created_at=datetime.datetime.now(),
            )
        ]]

    async def fake_insert(*args, **kwargs):
        events.append("qdrant")

    async def fake_update(*args, **kwargs):
        events.append("update")

    monkeypatch.setattr(utils, "split_chat_into_context_groups", fake_split)
    monkeypatch.setattr(utils, "insert_vectors_with_retry", fake_insert)
    monkeypatch.setattr(utils, "update_messages_in_batches", fake_update)

    result = await utils.process_and_vectorize_session_chats(_Session(), "group-1")

    assert result is not None
    assert events[:3] == ["query", "commit", "qdrant"]
