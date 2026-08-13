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


def test_chat_context_timestamp_uses_last_message_time():
    from nonebot_plugin_ai_groupmate.memory import _chat_context_created_at

    text = (
        "[2026-08-12 09:00:00] Alice: 第一条\n"
        "[2026-08-12 10:30:00] Bob: 第二条"
    )

    assert _chat_context_created_at(text, 0) == int(
        datetime(2026, 8, 12, 10, 30, 0).timestamp()
    )


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
