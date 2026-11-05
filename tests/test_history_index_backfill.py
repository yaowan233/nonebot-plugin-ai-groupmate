from uuid import uuid4
from datetime import datetime

import pytest
from sqlalchemy import event, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def index_db(tmp_path):
    from nonebot_plugin_ai_groupmate.model import ChatHistory, HistoryMessageIndex, HistoryIndexProgress

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'index.sqlite'}")
    async with engine.begin() as conn:
        await conn.run_sync(ChatHistory.__table__.create)
        await conn.run_sync(HistoryMessageIndex.__table__.create)
        await conn.run_sync(HistoryIndexProgress.__table__.create)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_backfill_resumes_after_rollback_and_only_uses_index_when_complete(index_db):
    from nonebot_plugin_ai_groupmate.model import ChatHistory, HistoryMessageIndex, HistoryIndexProgress
    from nonebot_plugin_ai_groupmate.message_repository import find_by_platform_id
    from nonebot_plugin_ai_groupmate.history_index_backfill import backfill_one

    group = f"coverage-{uuid4()}"
    time = datetime(2026, 1, 1)
    async with index_db() as db:
        for name, content in [("old-a", "id: shared\nA"), ("old-b", "id: second\nalias_id: alias\nB"),
                              ("old-c", "id: third\nC")]:
            await db.execute(insert(ChatHistory).values(session_id=group, user_id=name, user_name=name,
                content_type="text", content=content, created_at=time))
        await db.commit()
        assert [row.user_id for row in await find_by_platform_id(db, group, "shared")] == ["old-a"]
        await db.commit()  # Persists the historical cutoff before new traffic.
        db.add(ChatHistory(session_id=group, user_id="new", user_name="new", content_type="text",
                           content="id: shared\nnew", created_at=time))
        await db.commit()
    async with index_db() as db:
        _, count, done = await backfill_one(db, batch_size=1)
        assert count == 1
        assert not done
        await db.rollback()  # A crashed batch must be safe to repeat.
    async with index_db() as db:
        progress = await db.get(HistoryIndexProgress, group)
        assert progress is not None
        assert progress.last_history_id == 0
        while True:
            selected, _, done = await backfill_one(db, batch_size=1)
            assert selected == group
            await db.commit()
            if done:
                break
        indexed = (await db.scalars(select(HistoryMessageIndex).where(HistoryMessageIndex.session_id == group))).all()
        assert len(indexed) == 5
        # New rows can have an older timestamp; the write-time listener still
        # indexes them after the historical checkpoint is complete.
        db.add(ChatHistory(session_id=group, user_id="late", user_name="late", content_type="text",
                           content="id: late-id\nlate", created_at=datetime(2025, 1, 1)))
        await db.commit()
        statements = []
        engine = db.get_bind()
        def observe(_conn, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement.lower())
        event.listen(engine, "before_cursor_execute", observe)
        try:
            rows = await find_by_platform_id(db, group, "shared")
            alias_rows = await find_by_platform_id(db, group, "alias")
            late_rows = await find_by_platform_id(db, group, "late-id")
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert [row.user_id for row in rows] == ["old-a", "new"]
        assert [row.user_id for row in alias_rows] == ["old-b"]
        assert [row.user_id for row in late_rows] == ["late"]
        assert not any(" like " in statement for statement in statements)
        alias_rows[0].content = "id: second\nalias_id: replacement\nB"
        await db.commit()
        assert await find_by_platform_id(db, group, "alias") == []
        assert [row.user_id for row in await find_by_platform_id(db, group, "replacement")] == ["old-b"]


async def test_long_legacy_id_remains_searchable_after_backfill(index_db):
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_repository import find_by_platform_id
    from nonebot_plugin_ai_groupmate.history_index_backfill import backfill_one

    group = f"long-id-{uuid4()}"
    message_id = "a" * 256
    async with index_db() as db:
        await db.execute(insert(ChatHistory).values(session_id=group, user_id="old", user_name="old",
                         content_type="text", content=f"id: {message_id}\nbody"))
        await db.commit()
        assert len(await find_by_platform_id(db, group, message_id)) == 1
        await db.commit()
        _, _, done = await backfill_one(db, batch_size=10)
        assert done
        await db.commit()
        assert len(await find_by_platform_id(db, group, message_id)) == 1


def test_progress_migration_is_additive():
    from importlib import import_module

    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = import_module("nonebot_plugin_ai_groupmate.migrations.d7a3e6c9b154_add_history_index_progress")
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE old_messages (id INTEGER PRIMARY KEY, content TEXT)"))
        connection.execute(text("INSERT INTO old_messages VALUES (1, 'unchanged')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert "nonebot_plugin_ai_groupmate_historyindexprogress" in inspect(connection).get_table_names()
            migration.downgrade()
        assert connection.execute(text("SELECT content FROM old_messages")).scalar_one() == "unchanged"
