from types import SimpleNamespace
from typing import Any
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def source_db(tmp_path, monkeypatch):
    import nonebot_plugin_orm

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sources.sqlite3'}")
    async with engine.begin() as connection:
        await connection.run_sync(ChatHistory.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(nonebot_plugin_orm, "get_session", sessions)
    async with sessions() as session:
        session.add_all([
            ChatHistory(msg_id=1, session_id="g", user_id="100", user_name="旧昵称", content_type="text", content="id: transport-1\n回复id: transport-0\n我之前反对这个计划", media_id=None, created_at=datetime(2026, 1, 1)),
            ChatHistory(msg_id=2, session_id="other", user_id="999", user_name="旧昵称", content_type="text", content="其他群的私密原话", media_id=None, created_at=datetime(2026, 1, 2)),
            ChatHistory(msg_id=3, session_id="g", user_id="100", user_name="新昵称", content_type="text", content="现在支持这个计划", media_id=None, created_at=datetime(2026, 1, 3)),
            ChatHistory(msg_id=4, session_id="g", user_id="200", user_name="新昵称", content_type="text", content="同名另一个人的计划", media_id=None, created_at=datetime(2026, 1, 4)),
            ChatHistory(msg_id=5, session_id="g", user_id="bot-id", user_name="机器人", content_type="bot", content="这是机器人的计划", media_id=None, created_at=datetime(2026, 1, 5)),
            ChatHistory(msg_id=6, session_id="g", user_id="300", user_name="多行\n昵称\u2028", content_type="text", content="开头\n" + "无关前文" * 600 + "最终的蓝鲸口令在这里\u2028结束", media_id=None, created_at=datetime(2026, 1, 6)),
        ])
        await session.commit()
    yield sessions
    await engine.dispose()


def operator_for(points):
    from nonebot_plugin_ai_groupmate.memory import VectorDBOperator

    operator: Any = object.__new__(VectorDBOperator)
    operator.enabled = True
    operator.chat_col = "chat_collection"
    operator.rerank_url = ""
    operator.rerank_key = ""
    operator._ensure_collections = AsyncMock()
    operator._get_text_embedding = AsyncMock(return_value=[0.1, 0.2])
    operator.client = SimpleNamespace(query_points=AsyncMock(return_value=SimpleNamespace(points=points)))
    return operator


async def test_old_index_returns_verified_ids_aliases_and_roles(source_db):
    operator = operator_for([SimpleNamespace(payload={"text": "旧昵称: 计划\n新昵称: 计划", "msg_ids": [1, 3, 4, 5]})])
    result = await operator.search_chat("计划", "g")
    assert '[user_id="100" msg_id=1 role=user] "旧昵称"' in result
    assert '[user_id="100" msg_id=3 role=user] "新昵称"' in result
    assert '[user_id="200" msg_id=4 role=user] "新昵称"' in result
    assert '[user_id="bot-id" msg_id=5 role=bot]' in result
    assert "我之前反对这个计划" in result
    assert "transport-1" not in result
    operator._get_text_embedding.assert_awaited_once_with("计划")


async def test_missing_or_foreign_sources_do_not_shift_owners(source_db):
    operator = operator_for([SimpleNamespace(payload={"text": "其他群的私密原话\n不存在的内容", "msg_ids": [1, 2, 99999, 3]})])
    result = await operator.search_chat("计划", "g")
    assert "部分来源缺失" in result
    assert '[user_id="100" msg_id=3 role=user]' in result
    assert "其他群的私密原话" not in result
    assert "不存在的内容" not in result
    assert "999" not in result


async def test_deleted_or_foreign_hits_do_not_resurrect_vector_text(source_db):
    operator = operator_for([SimpleNamespace(payload={"text": "不应复活的旧原文", "msg_ids": [2, 99999]})])
    assert await operator.search_chat("旧原文", "g") == "未找到相关历史记录"


async def test_legacy_hit_without_source_ids_is_explicitly_unverified():
    operator = operator_for([SimpleNamespace(payload={"text": "Alice: 计划"})])
    operator._read_chat_sources = AsyncMock(side_effect=AssertionError("no IDs to query"))
    result = await operator.search_chat("计划", "g")
    assert "来源未核实" in result
    assert "不得按昵称合并" in result
    assert "Alice: 计划" in result
    operator._read_chat_sources.assert_not_awaited()


async def test_source_lookup_failure_does_not_invent_ids():
    operator = operator_for([SimpleNamespace(payload={"text": "Alice: 计划", "msg_ids": [1]})])
    operator._read_chat_sources = AsyncMock(side_effect=RuntimeError("unavailable"))
    result = await operator.search_chat("计划", "g")
    assert "来源未核实" in result
    assert "user_id=" not in result


async def test_long_multiline_hit_keeps_matching_body_and_identity(source_db):
    operator = operator_for([SimpleNamespace(payload={"text": "旧切片：最终的蓝鲸口令在这里", "msg_ids": [6]})])
    result = await operator.search_chat("蓝鲸口令", "g")
    assert '[user_id="300" msg_id=6 role=user]' in result
    assert '"多行\\n昵称\\u2028"' in result
    assert "蓝鲸口令" in result
    assert len(result.splitlines()) == 2
    assert len(result) < 1000


async def test_source_ids_are_bound_to_reranked_text(source_db):
    operator = operator_for([
        SimpleNamespace(payload={"text": "与问题无关", "msg_ids": [4]}),
        SimpleNamespace(payload={"text": "计划", "msg_ids": [1, 3]}),
    ])
    result = await operator.search_chat("计划", "g")
    assert result.index("msg_id=1") < result.index("msg_id=4")
    assert '[user_id="200" msg_id=4 role=user]' in result
