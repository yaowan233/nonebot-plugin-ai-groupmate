from uuid import uuid4

import pytest


async def test_structured_quote_lookup_respects_scope_and_legacy_fallback():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.reply_tools import resolve_reply_target
    from nonebot_plugin_ai_groupmate.structured_message import record_inbound

    session = f"structured-lookup-{uuid4()}"
    group = OutboundIdentity("qq", "group", "bot", "miyuki")
    private = OutboundIdentity("qq", "private", "bot", "miyuki")
    async with get_session() as db:
        await record_inbound(db, platform="qq", scope="group", session_id=session, platform_id="new",
                             sender_id="user", sender_name="user", reply_to_id=None, parts=[{"type": "text", "text": "hello"}])
        await record_inbound(db, platform="qq", scope="group", session_id=session, platform_id="structured-only",
                             sender_id="user", sender_name="user", reply_to_id=None, parts=[{"type": "text", "text": "hello"}])
        db.add(ChatHistory(session_id=session, user_id="user", user_name="user", content_type="text", content="id: new\nhello"))
        db.add(ChatHistory(session_id=session, user_id="user", user_name="user", content_type="text", content="id: old\nlegacy"))
        await db.commit()
        assert await resolve_reply_target(db, session, "new", outbound_identity=group) == "new"
        assert await resolve_reply_target(db, session, "old", outbound_identity=group) == "old"
        with pytest.raises(ValueError, match="当前会话中找不到"):
            await resolve_reply_target(db, session, "new", outbound_identity=private)
        assert await resolve_reply_target(db, session, "structured-only", outbound_identity=group) == "structured-only"


async def test_structured_quote_read_error_keeps_legacy_lookup_usable():
    from sqlalchemy import Select, event
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.reply_tools import resolve_reply_target

    session = f"structured-read-error-{uuid4()}"
    async with get_session() as db:
        db.add(ChatHistory(session_id=session, user_id="user", user_name="user", content_type="text", content="id: old\nlegacy"))
        await db.commit()
        engine = db.get_bind()
        def fail_structured(_conn, _cursor, statement, _parameters, _context, _executemany):
            if "structuredmessage" in statement.lower():
                raise RuntimeError("simulated failure")
        event.listen(engine, "before_cursor_execute", fail_structured)
        try:
            identity = OutboundIdentity("qq", "group", "bot", "miyuki")
            assert await resolve_reply_target(db, session, "old", outbound_identity=identity) == "old"
            assert await db.scalar(Select(ChatHistory.msg_id).where(ChatHistory.session_id == session)) is not None
        finally:
            event.remove(engine, "before_cursor_execute", fail_structured)


async def test_quote_lookup_uses_exact_scoped_platform_and_alias_ids():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.reply_tools import resolve_reply_target
    from nonebot_plugin_ai_groupmate.agent.history_format import load_replied_message_histories

    group = f"lookup-{uuid4()}"
    async with get_session() as db:
        for scope, content, kind in [
            (group, "id: original\nalias_id: ali%_123\n图文", "text"),
            (group, "id: original\nalias_id: ali%_123\n图片文件: x.png", "image"),
            (group, "id: unrelated\n回复id: ali%_123\nali%_123", "text"),
            (group, "id: ali%_1234\n相似 ID", "text"),
            (group + "-other", "id: ali%_123\n其他群", "text"),
        ]:
            db.add(ChatHistory(session_id=scope, user_id="u", user_name="u", content_type=kind, content=content))
        await db.commit()

        rows = await load_replied_message_histories(db, group, "ali%_123")
        assert [r.content_type for r in rows] == ["text", "image"]
        assert await resolve_reply_target(db, group, "ali%_123") == "ali%_123"
        with pytest.raises(ValueError, match="当前会话中找不到"):
            await resolve_reply_target(db, group + "-other", "original")
        assert await load_replied_message_histories(db, group, "unknown") == []


async def test_indexed_lookup_and_alias_updates_are_transactional():
    from sqlalchemy import Select, event
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, HistoryMessageIndex
    from nonebot_plugin_ai_groupmate.message_repository import find_by_platform_id

    group = f"index-{uuid4()}"
    async with get_session() as db:
        row = ChatHistory(session_id=group, user_id="u", user_name="u", content_type="image",
                          content="id: primary\nalias_id: old\nimage.png")
        db.add(row)
        await db.commit()
        await db.refresh(row)
        row_id = row.msg_id
        statements = []
        engine = db.get_bind()
        def observe(_conn, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement.lower())
        event.listen(engine, "before_cursor_execute", observe)
        try:
            found = await find_by_platform_id(db, group, "old")
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert [item.msg_id for item in found] == [row_id]
        assert any("historymessageindex" in statement for statement in statements)
        # Until historical coverage is complete, legacy rows must also be searched.
        assert any(" like " in statement for statement in statements)
        row.content = "id: primary\nalias_id: new\nimage.png"
        await db.commit()
        assert [item.msg_id for item in await find_by_platform_id(db, group, "new")] == [row_id]
        assert await find_by_platform_id(db, group, "old") == []
        ids = (await db.scalars(Select(HistoryMessageIndex.platform_id).where(HistoryMessageIndex.history_id == row_id))).all()
        assert set(ids) == {"primary", "new"}


async def test_legacy_rows_without_index_still_resolve():
    from sqlalchemy import insert
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_repository import find_by_platform_id

    group = f"legacy-{uuid4()}"
    async with get_session() as db:
        # Core inserts represent historical rows created before the listener existed.
        await db.execute(insert(ChatHistory).values(session_id=group, user_id="u", user_name="u",
            content_type="text", content="id: legacy\nalias_id: old-alias\n保留历史"))
        await db.commit()
        rows = await find_by_platform_id(db, group, "old-alias")
        assert len(rows) == 1
        assert "保留历史" in rows[0].content


async def test_old_and_new_rows_with_same_platform_id_are_both_visible():
    from sqlalchemy import insert
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_repository import find_by_platform_id

    group = f"mixed-index-{uuid4()}"
    async with get_session() as db:
        await db.execute(insert(ChatHistory).values(session_id=group, user_id="old", user_name="old",
            content_type="image", content="id: duplicate\nold.png"))
        db.add(ChatHistory(session_id=group, user_id="new", user_name="new", content_type="text",
                           content="id: duplicate\nnew"))
        await db.commit()
        rows = await find_by_platform_id(db, group, "duplicate")
        assert [row.user_id for row in rows] == ["old", "new"]


async def test_history_and_index_roll_back_together():
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, HistoryMessageIndex

    group = f"rollback-index-{uuid4()}"
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="u", user_name="u", content_type="text", content="id: tx\nhello"))
        await db.flush()
        assert len((await db.scalars(Select(HistoryMessageIndex).where(HistoryMessageIndex.session_id == group))).all()) == 1
        await db.rollback()
        assert not (await db.scalars(Select(HistoryMessageIndex).where(HistoryMessageIndex.session_id == group))).all()


def test_message_index_migration_preserves_old_history():
    from importlib import import_module

    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = import_module("nonebot_plugin_ai_groupmate.migrations.c6f1d3b8a952_add_history_message_index")
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE nonebot_plugin_ai_groupmate_chathistory (msg_id INTEGER PRIMARY KEY, content TEXT)"))
        connection.execute(text("INSERT INTO nonebot_plugin_ai_groupmate_chathistory VALUES (1, 'old')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert "nonebot_plugin_ai_groupmate_historymessageindex" in inspect(connection).get_table_names()
            migration.downgrade()
        assert connection.execute(text("SELECT content FROM nonebot_plugin_ai_groupmate_chathistory")).scalar_one() == "old"
