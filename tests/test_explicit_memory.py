import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def memory_db(tmp_path, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import explicit_memory
    from nonebot_plugin_ai_groupmate.model import ExplicitMemory

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'memories.sqlite3'}")
    async with engine.begin() as conn:
        await conn.run_sync(ExplicitMemory.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(explicit_memory, "get_session", sessions)
    yield sessions
    await engine.dispose()


def tools(source="接下来模仿雪人的文笔聊天", *, user="user-1", group="group-1", bot="bot-1", private=False):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema
    from nonebot_plugin_ai_groupmate.agent.explicit_memory import create_explicit_memory_tools

    history = [ChatHistorySchema(msg_id=40, session_id=group, user_id=user, user_name="测试用户", content_type="text", content="id: 123\n" + source, created_at=datetime.now())]
    return {tool.name: tool for tool in create_explicit_memory_tools(group, user, bot, private, history, None)}


async def save(source="接下来模仿雪人的文笔聊天", *, title="聊天文风", lifetime="temporary", kind="instruction", **scope):
    return json.loads(await tools(source, **scope)["save_explicit_memory"].ainvoke({"title": title, "source_quote": source, "lifetime": lifetime, "kind": kind}))


async def context(sessions, *, user="user-1", group="group-1", bot="bot-1", private=False, **kwargs):
    from nonebot_plugin_ai_groupmate.agent.explicit_memory import get_explicit_memory_context

    async with sessions() as db:
        result = await get_explicit_memory_context(db, group, user, bot, private, **kwargs)
        await db.commit()
        return result


async def rows(sessions):
    from nonebot_plugin_ai_groupmate.model import ExplicitMemory

    async with sessions() as db:
        return list((await db.scalars(select(ExplicitMemory))).all())


async def test_style_survives_history_rotation_and_new_tool_instance(memory_db):
    result = await save()
    assert result["ok"]
    later_tools = tools("第六十条消息：你怎么看？")
    read = json.loads(await later_tools["read_explicit_memories"].ainvoke({}))
    assert read["data"]["memories"][0]["content"] == "接下来模仿雪人的文笔聊天"
    assert "模仿雪人" in await context(memory_db)
    assert (await rows(memory_db))[0].source_msg_id == 40


@pytest.mark.parametrize("scope", [{"user": "user-2"}, {"group": "group-2"}, {"bot": "bot-2"}, {"private": True}])
async def test_memory_never_crosses_scope(memory_db, scope):
    await save()
    assert "模仿雪人" not in await context(memory_db, **scope)
    result = json.loads(await tools("恢复正常聊天", **scope)["forget_explicit_memory"].ainvoke({"title": "聊天文风"}))
    assert result["data"]["deleted_count"] == 0
    assert len(await rows(memory_db)) == 1


async def test_invented_or_old_source_cannot_be_saved(memory_db):
    result = json.loads(await tools("今天吃面条")["save_explicit_memory"].ainvoke({"title": "爱好", "source_quote": "以后记住我喜欢猫", "lifetime": "long_term"}))
    assert result["reason_code"] == "invalid_source"
    assert await rows(memory_db) == []


@pytest.mark.parametrize("source", ["记住我喜欢猫", "接下来一直模仿雪人", "今天记住我喜欢猫"])
async def test_ambiguous_and_temporary_requests_cannot_become_permanent(memory_db, source):
    result = await save(source, lifetime="long_term")
    assert result["reason_code"] == "missing_long_term_request"
    assert await rows(memory_db) == []


async def test_correction_overwrites_and_cancel_removes(memory_db):
    await save("以后叫我小陈", title="称呼", lifetime="long_term")
    await save("以后改叫我阿陈", title="称呼", lifetime="long_term")
    assert len(await rows(memory_db)) == 1
    active = await context(memory_db)
    assert "阿陈" in active
    assert "小陈" not in active
    result = json.loads(await tools("不要这个称呼了")["forget_explicit_memory"].ainvoke({"title": "称呼"}))
    assert result["ok"]
    assert result["data"]["deleted_count"] == 1
    assert "阿陈" not in await context(memory_db)
    assert await rows(memory_db) == []


async def test_idle_expiry_deletes_without_reintroducing_old_request(memory_db):
    await save()
    now = datetime.now() + timedelta(minutes=31)
    assert "模仿雪人" not in await context(memory_db, now=now)
    assert await rows(memory_db) == []


async def test_activity_refreshes_idle_but_not_absolute_limit(memory_db):
    await save()
    initial = (await rows(memory_db))[0]
    await context(memory_db, now=initial.last_used_at + timedelta(minutes=20))
    refreshed = (await rows(memory_db))[0]
    assert refreshed.last_used_at > initial.last_used_at
    assert refreshed.expires_at == initial.expires_at
    assert "模仿雪人" in await context(memory_db, now=initial.last_used_at + timedelta(minutes=40))
    from nonebot_plugin_ai_groupmate.model import ExplicitMemory
    async with memory_db() as db:
        await db.execute(update(ExplicitMemory).values(last_used_at=initial.expires_at - timedelta(minutes=1)))
        await db.commit()
    assert "模仿雪人" not in await context(memory_db, now=initial.expires_at)
    assert await rows(memory_db) == []


async def test_proactive_bot_activity_does_not_extend_temporary_memory(memory_db):
    await save()
    initial = (await rows(memory_db))[0]
    await context(memory_db, now=initial.last_used_at + timedelta(minutes=20), touch=False)
    assert (await rows(memory_db))[0].last_used_at == initial.last_used_at


async def test_long_term_fact_is_persisted_but_body_only_retrieved_when_needed(memory_db):
    await save("以后记住我的猫叫汤圆", title="宠物名字", lifetime="long_term", kind="fact")
    prompt = await context(memory_db, now=datetime.now() + timedelta(days=90))
    assert "宠物名字" in prompt
    assert "汤圆" not in prompt
    result = json.loads(await tools("我的猫叫什么")["read_explicit_memories"].ainvoke({"query": "宠物"}))
    assert result["data"]["memories"][0]["content"] == "以后记住我的猫叫汤圆"
    assert len(await rows(memory_db)) == 1


async def test_cleanup_runs_without_any_new_chat(memory_db):
    from nonebot_plugin_ai_groupmate.agent.explicit_memory import cleanup_explicit_memories
    await save()
    await save("以后记住我喜欢猫", title="爱好", lifetime="long_term", kind="fact")
    async with memory_db() as db:
        await cleanup_explicit_memories(db, now=datetime.now() + timedelta(hours=25))
        await db.commit()
    assert [row.title for row in await rows(memory_db)] == ["爱好"]


async def test_delete_all_requires_explicit_request(memory_db):
    await save()
    failure = json.loads(await tools("忘掉聊天文风")["forget_explicit_memory"].ainvoke({"target": "all"}))
    assert not failure["ok"]
    assert len(await rows(memory_db)) == 1
    success = json.loads(await tools("忘掉我的全部记忆")["forget_explicit_memory"].ainvoke({"target": "all"}))
    assert success["ok"]
    assert await rows(memory_db) == []


async def test_negative_request_cannot_authorize_bulk_delete(memory_db):
    await save()
    result = json.loads(await tools("不要忘掉我的全部记忆")["forget_explicit_memory"].ainvoke({"target": "all"}))
    assert not result["ok"]
    assert len(await rows(memory_db)) == 1


async def test_failed_commit_never_reports_saved(memory_db, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession
    async def fail_commit(self):
        raise RuntimeError("simulated database failure")
    monkeypatch.setattr(AsyncSession, "commit", fail_commit)
    result = await save()
    assert result["reason_code"] == "memory_write_failed"
    assert not result["ok"]
    assert await rows(memory_db) == []


async def test_today_expires_by_midnight(memory_db):
    await save("今天模仿雪人聊天")
    row = (await rows(memory_db))[0]
    assert row.expires_at is not None
    assert row.expires_at.hour == row.expires_at.minute == row.expires_at.second == 0


async def test_memory_limit_does_not_silently_discard_permanent_items(memory_db):
    for index in range(12):
        assert (await save(f"以后记住第{index}项", title=f"事项{index}", lifetime="long_term", kind="fact"))["ok"]
    result = await save("以后记住第十三项", title="事项13", lifetime="long_term", kind="fact")
    assert result["reason_code"] == "memory_limit"
    assert len(await rows(memory_db)) == 12


def test_migration_creates_model_schema_and_downgrades():
    from sqlalchemy import inspect, create_engine
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    from nonebot_plugin_ai_groupmate.model import ExplicitMemory
    from nonebot_plugin_ai_groupmate.migrations import b7e2a9c4d601_add_explicit_memory as migration

    engine = create_engine("sqlite://")
    with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()
        schema = inspect(conn)
        table = ExplicitMemory.__tablename__
        assert {column["name"] for column in schema.get_columns(table)} == set(ExplicitMemory.__table__.columns.keys())
        assert schema.get_unique_constraints(table)[0]["name"] == "uq_explicit_memory_scope_title"
        assert schema.get_indexes(table)[0]["name"] == "ix_explicit_memory_expiry"
        migration.downgrade()
        assert not inspect(conn).has_table(table)
    engine.dispose()
