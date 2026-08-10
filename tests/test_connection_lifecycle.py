import json
import datetime
from types import SimpleNamespace
from typing import cast
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_startup_reconfigures_vector_db_for_persisted_meme_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    import nonebot_plugin_ai_groupmate as plugin

    calls: list[str] = []

    @asynccontextmanager
    async def fake_get_session():
        yield object()

    async def fake_load_runtime_config_overrides(_session):
        return {"meme_embedding_mode"}

    async def fake_reconfigure():
        calls.append("reconfigure")

    monkeypatch.setattr(plugin, "get_session", fake_get_session)
    monkeypatch.setattr(
        plugin,
        "load_runtime_config_overrides",
        fake_load_runtime_config_overrides,
    )
    monkeypatch.setattr(plugin, "_refresh_runtime_resources", lambda _fields: None)
    monkeypatch.setattr(plugin.DB, "reconfigure", fake_reconfigure)
    monkeypatch.setattr(
        plugin,
        "mark_restart_fields_applied",
        lambda: calls.append("marked"),
    )

    await plugin._load_webui_runtime_config()

    assert calls == ["reconfigure", "marked"]


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
    monkeypatch.setattr(utils.DB, "enabled", True)

    result = await utils.process_and_vectorize_session_chats(
        cast(AsyncSession, _Session()), "group-1"
    )

    assert result is not None
    assert events[:3] == ["query", "commit", "qdrant"]


@pytest.mark.asyncio
async def test_chat_vectorization_keeps_messages_pending_without_qdrant(monkeypatch):
    from nonebot_plugin_ai_groupmate import utils

    async def fail_if_history_is_queried(*args, **kwargs):
        raise AssertionError("Qdrant 未启用时不应查询或标记聊天记录")

    monkeypatch.setattr(utils.DB, "enabled", False)
    monkeypatch.setattr(
        utils,
        "split_chat_into_context_groups",
        fail_if_history_is_queried,
    )

    result = await utils.process_and_vectorize_session_chats(
        cast(AsyncSession, object()),
        "group-1",
    )

    assert result is None


@pytest.mark.asyncio
async def test_chat_vectorization_scheduler_skips_without_qdrant(monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin

    def fail_if_session_is_opened():
        raise AssertionError("Qdrant 未启用时不应启动会话向量化")

    monkeypatch.setattr(plugin.DB, "enabled", False)
    monkeypatch.setattr(plugin, "get_session", fail_if_session_is_opened)

    await plugin.vectorize_message_history()


@pytest.mark.asyncio
async def test_reply_logic_closes_history_sessions_before_agent_wait(monkeypatch):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import ResponseMessage
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    events: list[str] = []
    sessions: list[object] = []
    history = ChatHistorySchema(
        msg_id=1,
        session_id="group-1",
        user_id="user-1",
        content_type="text",
        content="id: 1\nhello",
        created_at=datetime.datetime.now(),
        user_name="tester",
        media_id=None,
        vectorized=False,
    )

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return [history]

    class _Session:
        def __init__(self, index: int) -> None:
            self.index = index

        async def execute(self, statement):
            events.append(f"query-{self.index}")
            return _Result()

        async def commit(self):
            events.append(f"commit-{self.index}")

    @asynccontextmanager
    async def fake_get_session():
        session = _Session(len(sessions) + 1)
        sessions.append(session)
        events.append(f"open-{session.index}")
        try:
            yield session
        finally:
            events.append(f"close-{session.index}")

    async def fake_load_agent_history(db_session, session_id):
        assert db_session is sessions[1]
        return [history]

    async def fake_choice_response_strategy(db_session, *args, **kwargs):
        assert db_session is sessions[2]
        assert events[:6] == [
            "open-1",
            "query-1",
            "commit-1",
            "close-1",
            "open-2",
            "commit-2",
        ]
        assert "close-2" in events
        return ResponseMessage(need_reply=False, text=None)

    monkeypatch.setattr(plugin, "get_session", fake_get_session)
    monkeypatch.setattr(plugin, "_load_agent_history", fake_load_agent_history)
    monkeypatch.setattr(
        plugin, "choice_response_strategy", fake_choice_response_strategy
    )

    fake_session = SimpleNamespace(
        scene=SimpleNamespace(id="group-1", type=SceneType.PRIVATE),
        self_id="bot-1",
    )
    await plugin.handle_reply_logic(
        "request-1",
        cast(Uninfo, fake_session),
        cast(QryItrface, SimpleNamespace()),
        cast(Bot, SimpleNamespace()),
        cast(Event, SimpleNamespace()),
        "bot",
        "user-1",
        "tester",
        True,
        False,
        None,
    )

    assert events[-1] == "close-3"
