from types import SimpleNamespace
from typing import Any
from datetime import datetime, timedelta

import pytest


def test_chat_query_expands_relative_time_to_exact_dates():
    from nonebot_plugin_ai_groupmate.memory import expand_chat_search_query

    expanded = expand_chat_search_query(
        "昨天和上周谁讨论过部署？",
        now=datetime(2026, 8, 13, 12, 0, 0),
    )

    assert "昨天=2026-08-12" in expanded
    assert "上周=2026-08-03至2026-08-09" in expanded


def test_chat_query_parses_exact_relative_time_range():
    from nonebot_plugin_ai_groupmate.memory import parse_chat_search_time_range

    result = parse_chat_search_time_range(
        "昨天谁提到了部署？",
        now=datetime(2026, 8, 13, 12, 0, 0),
    )

    assert result == (
        int(datetime(2026, 8, 12, 0, 0, 0).timestamp()),
        int(datetime(2026, 8, 12, 23, 59, 59).timestamp()),
    )


def test_chat_query_merges_multiple_relative_time_ranges():
    from nonebot_plugin_ai_groupmate.memory import parse_chat_search_time_range

    result = parse_chat_search_time_range(
        "昨天和上周谁讨论过部署？",
        now=datetime(2026, 8, 13, 12, 0, 0),
    )

    assert result == (
        int(datetime(2026, 8, 3, 0, 0, 0).timestamp()),
        int(datetime(2026, 8, 12, 23, 59, 59).timestamp()),
    )


def test_chat_candidate_ranking_blends_exact_terms_and_deduplicates():
    from nonebot_plugin_ai_groupmate.memory import VectorDBOperator

    duplicate = "[2026-08-12 10:00:00] Alice: 部署口令是蓝鲸"
    points = [
        SimpleNamespace(payload={"text": "[2026-08-12 09:00:00] Bob: 今晚吃火锅"}),
        SimpleNamespace(payload={"text": "[2026-08-12 09:30:00] Carol: 服务器恢复了"}),
        SimpleNamespace(payload={"text": duplicate}),
        SimpleNamespace(payload={"text": duplicate}),
    ]

    ranked = VectorDBOperator._rank_chat_candidates("部署口令", points)

    assert ranked[0] == duplicate
    assert ranked.count(duplicate) == 1


@pytest.mark.asyncio
async def test_search_chat_returns_labeled_locally_reranked_fragments():
    from nonebot_plugin_ai_groupmate.memory import (
        CHAT_SEARCH_CANDIDATE_LIMIT,
        VectorDBOperator,
    )

    operator: Any = object.__new__(VectorDBOperator)
    operator.enabled = True
    operator.chat_col = "chat_collection"
    operator.rerank_url = ""
    operator.rerank_key = ""
    calls: dict[str, Any] = {}

    async def ensure_collections(_: set[str]) -> None:
        return None

    async def get_embedding(text: str) -> list[float]:
        calls["embedding_text"] = text
        return [0.1, 0.2]

    class FakeQdrantClient:
        async def query_points(self, **kwargs: Any) -> Any:
            calls["query"] = kwargs
            return SimpleNamespace(points=[
                SimpleNamespace(payload={
                    "text": "[2026-08-12 09:00:00] Bob: 今晚吃火锅",
                    "created_at": 1,
                }),
                SimpleNamespace(payload={
                    "text": "[2026-08-12 10:00:00] Alice: 部署口令是蓝鲸",
                    "created_at": 2,
                }),
            ])

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_embedding
    operator.client = FakeQdrantClient()

    result = await operator.search_chat("部署口令", "group-1")

    assert calls["embedding_text"] == "部署口令"
    assert calls["query"]["limit"] == CHAT_SEARCH_CANDIDATE_LIMIT
    assert calls["query"]["with_payload"] is True
    assert "【相关历史片段 1】" in result
    assert result.index("部署口令是蓝鲸") < result.index("今晚吃火锅")


@pytest.mark.asyncio
async def test_search_chat_v2_uses_dense_sparse_fusion_and_time_filter():
    from nonebot_plugin_ai_groupmate.memory import (
        CHAT_COLLECTION,
        CHAT_DENSE_VECTOR,
        CHAT_SPARSE_VECTOR,
        VectorDBOperator,
    )

    operator: Any = object.__new__(VectorDBOperator)
    operator.enabled = True
    operator.chat_col = CHAT_COLLECTION
    operator.rerank_url = ""
    operator.rerank_key = ""
    calls: dict[str, Any] = {}

    async def ensure_collections(_: set[str]) -> None:
        return None

    async def get_embedding(_: str) -> list[float]:
        return [0.1, 0.2]

    class FakeQdrantClient:
        async def query_points(self, **kwargs: Any) -> Any:
            calls.update(kwargs)
            return SimpleNamespace(points=[SimpleNamespace(payload={
                "text": "[2026-08-12 10:00:00] Alice: 部署口令是蓝鲸",
                "created_at": 1,
                "msg_ids": [1],
            })])

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_embedding
    operator.client = FakeQdrantClient()

    result = await operator.search_chat("昨天的部署口令", "group-1")

    assert "部署口令是蓝鲸" in result
    assert {item.using for item in calls["prefetch"]} == {
        CHAT_DENSE_VECTOR,
        CHAT_SPARSE_VECTOR,
    }
    must_conditions = calls["query_filter"].must
    assert {condition.key for condition in must_conditions} == {
        "session_id",
        "start_at",
        "end_at",
    }


@pytest.mark.asyncio
async def test_chat_v2_collection_has_dense_sparse_and_payload_indexes():
    import asyncio

    from nonebot_plugin_ai_groupmate.memory import (
        CHAT_COLLECTION,
        CHAT_DENSE_VECTOR,
        CHAT_SPARSE_VECTOR,
        VectorDBOperator,
    )

    operator: Any = object.__new__(VectorDBOperator)
    operator.chat_col = CHAT_COLLECTION
    operator.text_only = False
    operator.media_multivector_col = "media_collection_v3"
    operator.emb_model = "test-model"
    operator.text_embedding_dimension = 1024
    operator._init_lock = asyncio.Lock()
    operator._ready_collections = set()
    operator._collection_validation_errors = {}
    create_calls: list[dict[str, Any]] = []
    index_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, _: str) -> bool:
            return False

        async def create_collection(self, **kwargs: Any) -> None:
            create_calls.append(kwargs)

        async def create_payload_index(self, **kwargs: Any) -> None:
            index_calls.append(kwargs)

    operator.client = FakeQdrantClient()

    await operator._ensure_collections({CHAT_COLLECTION})

    assert set(create_calls[0]["vectors_config"]) == {CHAT_DENSE_VECTOR}
    assert set(create_calls[0]["sparse_vectors_config"]) == {CHAT_SPARSE_VECTOR}
    assert {call["field_name"] for call in index_calls} == {
        "session_id",
        "start_at",
        "end_at",
    }


@pytest.mark.asyncio
async def test_rag_scheduler_coalesces_repeated_session_activity(monkeypatch):
    import asyncio

    import nonebot_plugin_ai_groupmate as plugin

    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def fake_worker(session_id: str) -> None:
        calls.append(session_id)
        started.set()
        await release.wait()

    monkeypatch.setattr(plugin.DB, "enabled", True)
    monkeypatch.setattr(plugin, "_run_idle_rag_vectorization", fake_worker)
    plugin._rag_vectorization_tasks.clear()
    plugin._rag_last_activity.clear()

    plugin._schedule_rag_vectorization("group-1")
    plugin._schedule_rag_vectorization("group-1")
    await started.wait()

    assert calls == ["group-1"]
    assert len(plugin._rag_vectorization_tasks) == 1

    release.set()
    await next(iter(plugin._rag_vectorization_tasks.values()))
    await asyncio.sleep(0)
    assert plugin._rag_vectorization_tasks == {}


def test_chat_context_timestamp_uses_last_message_time():
    from nonebot_plugin_ai_groupmate.memory import _chat_context_created_at

    text = (
        "[2026-08-12 09:00:00] Alice: 第一条\n"
        "[2026-08-12 10:30:00] Bob: 第二条"
    )

    assert _chat_context_created_at(text, 0) == int(
        datetime(2026, 8, 12, 10, 30, 0).timestamp()
    )


def test_rag_message_content_removes_internal_message_metadata():
    from nonebot_plugin_ai_groupmate.utils import _rag_message_content

    message: Any = SimpleNamespace(
        content_type="text",
        content="id: 123\n回复id: 456\n真正需要检索的正文",
    )

    assert _rag_message_content(message) == "真正需要检索的正文"


@pytest.mark.asyncio
async def test_batch_insert_keeps_messages_pending_when_embeddings_are_missing():
    from nonebot_plugin_ai_groupmate.memory import (
        VectorDBOperator,
        EmbeddingProviderUnavailableError,
    )

    operator: Any = object.__new__(VectorDBOperator)
    operator.enabled = True
    operator.chat_col = "chat_collection"

    async def ensure_collections(_: set[str]) -> None:
        return None

    async def get_embeddings(_: list[str]) -> list[list[float]]:
        return []

    operator._ensure_collections = ensure_collections
    operator._get_batch_text_embeddings = get_embeddings

    with pytest.raises(EmbeddingProviderUnavailableError):
        await operator.batch_insert(["需要保留的消息"], "group-1")


@pytest.mark.asyncio
async def test_chat_chunking_keeps_overlap_but_not_across_time_gaps():
    from nonebot_plugin_ai_groupmate import utils

    started_at = datetime(2026, 8, 13, 10, 0, 0)
    messages = [
        SimpleNamespace(
            msg_id=index,
            session_id="group-1",
            user_id=str(index),
            content_type="text",
            content=f"消息 {index}",
            created_at=(
                started_at + timedelta(minutes=index)
                if index < 5
                else started_at + timedelta(hours=2, minutes=index)
            ),
            user_name=f"用户{index}",
            media_id=None,
            vectorized=False,
        )
        for index in range(1, 7)
    ]

    class FakeScalars:
        def all(self) -> list[Any]:
            return messages

    class FakeResult:
        def scalars(self) -> FakeScalars:
            return FakeScalars()

    class FakeSession:
        async def execute(self, _: object) -> FakeResult:
            return FakeResult()

    groups = await utils.split_chat_into_context_groups(
        FakeSession(),  # type: ignore[arg-type]
        "group-1",
        max_time_gap=timedelta(hours=1),
        max_token_count=1000,
        max_messages=3,
        overlap_messages=1,
    )

    assert [[message.msg_id for message in group] for group in groups] == [
        [1, 2, 3],
        [3, 4],
        [5, 6],
    ]


@pytest.mark.asyncio
async def test_chat_chunking_splits_a_single_oversized_message():
    from nonebot_plugin_ai_groupmate import utils

    message = SimpleNamespace(
        msg_id=1,
        session_id="group-1",
        user_id="1",
        content_type="text",
        content="很长的消息" * 500,
        created_at=datetime(2026, 8, 13, 10, 0, 0),
        user_name="用户1",
        media_id=None,
        vectorized=False,
        vectorized_version=0,
    )

    class FakeScalars:
        def all(self) -> list[Any]:
            return [message]

    class FakeResult:
        def scalars(self) -> FakeScalars:
            return FakeScalars()

    class FakeSession:
        async def execute(self, _: object) -> FakeResult:
            return FakeResult()

    groups = await utils.split_chat_into_context_groups(
        FakeSession(),  # type: ignore[arg-type]
        "group-1",
        max_token_count=100,
        overlap_messages=0,
    )

    assert len(groups) > 1
    assert all(
        utils.estimate_token_count(message.content) <= 100
        for group in groups
        for message in group
    )
