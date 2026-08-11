from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from typing_extensions import Self


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def json(self) -> dict[str, Any]:
        return self._data


class FakeAsyncClient:
    def __init__(self, response: FakeResponse, calls: list[dict[str, Any]], **_: Any):
        self.response = response
        self.calls = calls

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


@pytest.fixture
def memory_module():
    from nonebot_plugin_ai_groupmate import memory

    return memory


def make_operator(memory_module: Any):
    operator = object.__new__(memory_module.VectorDBOperator)
    operator.media_col = "media_collection"
    operator.media_multivector_col = "media_collection_v3"
    return operator


def mock_http_client(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    response_data: dict[str, Any],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    response = FakeResponse(response_data)
    monkeypatch.setattr(
        memory_module.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(response, calls, **kwargs),
    )
    return calls


@pytest.mark.asyncio
async def test_ensure_collections_creates_named_media_vectors(memory_module: Any):
    operator = make_operator(memory_module)
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    create_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, collection_name: str) -> bool:
            return collection_name != "media_collection_v3"

        async def get_collection(self, collection_name: str) -> Any:
            size = (
                memory_module.MEDIA_VECTOR_SIZE
                if collection_name == "media_collection"
                else operator.text_embedding_dimension
            )
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={},
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=size),
                    ),
                ),
            )

        async def update_collection(self, **_kwargs: Any) -> bool:
            return True

        async def create_collection(self, **kwargs: Any) -> None:
            create_calls.append(kwargs)

    operator.client = FakeQdrantClient()

    await operator._ensure_collections()

    assert len(create_calls) == 1
    vectors_config = create_calls[0]["vectors_config"]
    assert set(vectors_config) == {
        memory_module.MEDIA_TEXT_VECTOR,
        memory_module.MEDIA_IMAGE_VECTOR,
    }
    assert all(
        vector.size == memory_module.MEDIA_VECTOR_SIZE
        for vector in vectors_config.values()
    )


@pytest.mark.asyncio
async def test_qwen_vl_embedding_enables_explicit_fusion(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    vector = [0.1] * memory_module.MEDIA_VECTOR_SIZE
    calls = mock_http_client(
        memory_module,
        monkeypatch,
        {"output": {"embeddings": [{"type": "fusion", "embedding": vector}]}},
    )

    result = await make_operator(memory_module)._get_qwen_vl_embedding(
        text="熊猫头流泪，表达无奈",
        image_source="data:image/png;base64,AAAA",
    )

    assert result == vector
    payload = calls[0]["json"]
    assert payload == {
        "model": "qwen3-vl-embedding",
        "input": {
            "contents": [
                {"text": "熊猫头流泪，表达无奈"},
                {"image": "data:image/png;base64,AAAA"},
            ]
        },
        "parameters": {
            "dimension": memory_module.MEDIA_VECTOR_SIZE,
            "enable_fusion": True,
        },
    }


@pytest.mark.asyncio
async def test_qwen_vl_embedding_keeps_single_modality_independent(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    vector = [0.2] * memory_module.MEDIA_VECTOR_SIZE
    calls = mock_http_client(
        memory_module,
        monkeypatch,
        {"output": {"embeddings": [{"type": "vl", "embedding": vector}]}},
    )

    result = await make_operator(memory_module)._get_qwen_vl_embedding(text="震惊")

    assert result == vector
    payload = calls[0]["json"]
    assert payload["input"]["contents"] == [{"text": "震惊"}]
    assert payload["parameters"] == {"dimension": memory_module.MEDIA_VECTOR_SIZE}


@pytest.mark.asyncio
async def test_qwen_vl_embedding_rejects_multiple_vectors(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    vector = [0.3] * memory_module.MEDIA_VECTOR_SIZE
    mock_http_client(
        memory_module,
        monkeypatch,
        {
            "output": {
                "embeddings": [
                    {"type": "text", "embedding": vector},
                    {"type": "image", "embedding": vector},
                ]
            }
        },
    )

    result = await make_operator(memory_module)._get_qwen_vl_embedding(
        text="震惊",
        image_source="data:image/png;base64,AAAA",
    )

    assert result is None


@pytest.mark.asyncio
async def test_qwen_vl_embedding_returns_ordered_independent_pair(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    text_vector = [0.31] * memory_module.MEDIA_VECTOR_SIZE
    image_vector = [0.32] * memory_module.MEDIA_VECTOR_SIZE
    calls = mock_http_client(
        memory_module,
        monkeypatch,
        {
            "output": {
                "embeddings": [
                    {"index": 1, "type": "vl", "embedding": image_vector},
                    {"index": 0, "type": "vl", "embedding": text_vector},
                ]
            }
        },
    )

    result = await make_operator(memory_module)._get_qwen_vl_independent_pair(
        "熊猫头流泪",
        "data:image/png;base64,AAAA",
    )

    assert result == (text_vector, image_vector)
    payload = calls[0]["json"]
    assert payload["input"]["contents"] == [
        {"text": "熊猫头流泪"},
        {"image": "data:image/png;base64,AAAA"},
    ]
    assert payload["parameters"] == {"dimension": memory_module.MEDIA_VECTOR_SIZE}


@pytest.mark.asyncio
async def test_insert_media_reports_embedding_failure(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    upsert_called = False

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_embedding(*_: Any, **__: Any) -> None:
        return None

    class FakeQdrantClient:
        async def upsert(self, **_: Any) -> None:
            nonlocal upsert_called
            upsert_called = True

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_independent_pair = get_embedding
    operator.client = FakeQdrantClient()

    assert await operator.insert_media(1, "data:image/png;base64,AAAA", "描述") is False
    assert upsert_called is False


@pytest.mark.asyncio
async def test_insert_media_waits_for_confirmed_upsert(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    vector = [0.4] * memory_module.MEDIA_VECTOR_SIZE
    upsert_calls: list[dict[str, Any]] = []

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_embedding(*_: Any, **__: Any) -> tuple[list[float], list[float]]:
        return vector, vector

    class FakeQdrantClient:
        async def upsert(self, **kwargs: Any) -> None:
            upsert_calls.append(kwargs)

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_independent_pair = get_embedding
    operator.client = FakeQdrantClient()

    assert await operator.insert_media(7, "data:image/png;base64,AAAA", "描述") is True
    assert upsert_calls[0]["wait"] is True
    assert upsert_calls[0]["collection_name"] == "media_collection_v3"
    point = upsert_calls[0]["points"][0]
    assert point.vector == {
        memory_module.MEDIA_TEXT_VECTOR: vector,
        memory_module.MEDIA_IMAGE_VECTOR: vector,
    }
    assert point.payload["embedding_version"] == memory_module.MEDIA_EMBEDDING_VERSION


def test_meme_results_exclude_recent_ids_before_fallback(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        memory_module.random,
        "choices",
        lambda population, **_: [population[0]],
    )
    points = [
        SimpleNamespace(id=1, score=0.90),
        SimpleNamespace(id=2, score=0.85),
        SimpleNamespace(id=3, score=0.80),
        SimpleNamespace(id=4, score=0.75),
    ]

    result = memory_module.VectorDBOperator._diversify_meme_results(
        points,
        exclude_ids={1, 2},
        limit=3,
    )

    assert result == [3, 4, 1]


def test_meme_search_routes_use_rrf_and_reward_cross_route_hits(
    memory_module: Any,
):
    primary = [
        SimpleNamespace(id=1, score=0.9),
        SimpleNamespace(id=2, score=0.8),
    ]
    legacy = [
        SimpleNamespace(id=2, score=0.95),
        SimpleNamespace(id=3, score=0.9),
    ]

    result = memory_module.VectorDBOperator._merge_meme_search_routes(
        primary,
        legacy,
    )

    assert [media_id for media_id, _ in result] == [2, 1, 3]


def test_weighted_meme_routes_combine_text_visual_and_legacy_hits(
    memory_module: Any,
):
    text = [SimpleNamespace(id=1, score=0.9), SimpleNamespace(id=2, score=0.8)]
    visual = [SimpleNamespace(id=3, score=0.95), SimpleNamespace(id=1, score=0.7)]
    legacy = [SimpleNamespace(id=4, score=0.9)]

    result = memory_module.VectorDBOperator._merge_weighted_meme_search_routes([
        (text, 1.0),
        (visual, memory_module.MEME_CONTEXT_VISUAL_ROUTE_WEIGHT),
        (legacy, memory_module.MEME_LEGACY_ROUTE_WEIGHT),
    ])

    assert [media_id for media_id, _ in result] == [1, 2, 3, 4]


def test_meme_search_expands_dragon_image_jargon(memory_module: Any):
    expanded = memory_module.expand_meme_search_terms("来张龙图")

    assert "黑白熊猫头" in expanded
    assert "人脸熊猫头" in expanded
    assert "不是动物龙" in expanded
    assert "用户原始搜索：来张龙图" in expanded
    assert memory_module.expand_meme_search_terms(expanded) == expanded
    assert memory_module.expand_meme_search_terms("黑色龙图案") == "黑色龙图案"
    assert memory_module.expand_meme_search_terms("末影龙图像") == "末影龙图像"


@pytest.mark.asyncio
async def test_meme_search_embeds_expanded_jargon(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    embedded_texts: list[str] = []

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_embedding(**kwargs: Any) -> list[float]:
        embedded_texts.append(kwargs["text"])
        return [0.1] * memory_module.MEDIA_VECTOR_SIZE

    async def search_routes(*_: Any, **__: Any) -> list[tuple[int, float]]:
        return []

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_embedding = get_embedding
    operator._search_media_routes = search_routes

    await operator.search_meme_candidates("发个龙图", strict_content_match=True)

    assert len(embedded_texts) == 1
    assert "黑白熊猫头" in embedded_texts[0]
    assert "用户原始搜索：发个龙图" in embedded_texts[0]


def test_legacy_route_reserves_independent_candidates_in_review_window(
    memory_module: Any,
):
    primary = [SimpleNamespace(id=media_id) for media_id in range(1, 21)]
    legacy = [SimpleNamespace(id=media_id) for media_id in range(101, 106)]
    candidates = [
        (media_id, 1.0 / (memory_module.MEME_RRF_K + rank))
        for rank, media_id in enumerate(range(1, 21), start=1)
    ] + [
        (media_id, memory_module.MEME_LEGACY_ROUTE_WEIGHT / (
            memory_module.MEME_RRF_K + rank
        ))
        for rank, media_id in enumerate(range(101, 106), start=1)
    ]

    result = memory_module.VectorDBOperator._reserve_meme_route_candidates(
        candidates,
        legacy,
        exclude_ids={int(point.id) for point in primary},
        quota=3,
        window=15,
    )

    assert [media_id for media_id, _ in result[:12]] == list(range(1, 13))
    assert [media_id for media_id, _ in result[12:15]] == [101, 102, 103]
    assert sorted(media_id for media_id, _ in result) == sorted(
        media_id for media_id, _ in candidates
    )


@pytest.mark.asyncio
async def test_media_route_timeout_keeps_successful_routes(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    operator = make_operator(memory_module)

    class FakeQdrantClient:
        async def query_points(self, **kwargs: Any) -> Any:
            if kwargs["collection_name"] == "media_collection":
                await asyncio.Event().wait()
            return SimpleNamespace(points=[SimpleNamespace(id=1, score=0.9)])

    operator.client = FakeQdrantClient()
    monkeypatch.setattr(
        memory_module,
        "MEME_QDRANT_ROUTE_TIMEOUT_SECONDS",
        0.01,
    )

    result = await operator._search_media_routes(
        [0.1] * memory_module.MEDIA_VECTOR_SIZE,
        vector_name=memory_module.MEDIA_TEXT_VECTOR,
        limit=5,
    )

    assert result
    assert result[0][0] == 1


@pytest.mark.asyncio
async def test_content_search_uses_lower_visual_route_weight(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    route_weights: list[float] = []

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_embedding(**_: Any) -> list[float]:
        return [0.1] * memory_module.MEDIA_VECTOR_SIZE

    async def search_routes(*_: Any, **kwargs: Any) -> list[tuple[int, float]]:
        route_weights.append(kwargs["visual_route_weight"])
        return []

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_embedding = get_embedding
    operator._search_media_routes = search_routes

    await operator.search_meme_candidates("随对话自然反应")
    await operator.search_meme_candidates("初音未来拿着葱", strict_content_match=True)

    assert route_weights == [
        memory_module.MEME_CONTEXT_VISUAL_ROUTE_WEIGHT,
        memory_module.MEME_CONTENT_VISUAL_ROUTE_WEIGHT,
    ]


def test_group_usage_boost_promotes_a_group_favorite(memory_module: Any):
    candidates = [
        (1, 0.9),
        (2, 0.8),
        (3, 0.7),
        (4, 0.6),
    ]

    result = memory_module.VectorDBOperator.apply_group_usage_boost(
        candidates,
        {3: 10},
    )

    assert [media_id for media_id, _ in result][:3] == [3, 1, 2]


@pytest.mark.asyncio
async def test_search_meme_uses_larger_candidate_pool_and_recent_exclusion(
    memory_module: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    operator = make_operator(memory_module)
    operator.enabled = True
    query_calls: list[dict[str, Any]] = []

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_embedding(**_: Any) -> list[float]:
        return [0.1] * memory_module.MEDIA_VECTOR_SIZE

    class FakeQdrantClient:
        async def query_points(self, **kwargs: Any) -> Any:
            query_calls.append(kwargs)
            if kwargs["collection_name"] == "media_collection":
                return SimpleNamespace(points=[])
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id=1, score=0.9),
                    SimpleNamespace(id=2, score=0.8),
                    SimpleNamespace(id=3, score=0.7),
                ]
            )

    monkeypatch.setattr(
        memory_module.random,
        "choices",
        lambda population, **_: [population[0]],
    )
    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_embedding = get_embedding
    operator.client = FakeQdrantClient()

    result = await operator.search_meme("无奈", limit=2, exclude_ids={1})

    assert result == [2, 3]
    assert len(query_calls) == 3
    assert query_calls[0]["collection_name"] == "media_collection_v3"
    assert query_calls[0]["using"] == memory_module.MEDIA_TEXT_VECTOR
    assert query_calls[0]["limit"] == memory_module.MEME_SEARCH_POOL_SIZE
    assert query_calls[0]["with_payload"] is False
    assert query_calls[0]["timeout"] == 8
    assert query_calls[1]["collection_name"] == "media_collection_v3"
    assert query_calls[1]["using"] == memory_module.MEDIA_IMAGE_VECTOR
    assert query_calls[2]["collection_name"] == "media_collection"
    assert "using" not in query_calls[2]


# ================= text 模式（meme_embedding_mode="text"） =================


@pytest.mark.asyncio
async def test_text_mode_ensure_collections_creates_text_collection(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    create_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, collection_name: str) -> bool:
            return collection_name == operator.chat_col

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={},
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=operator.text_embedding_dimension,
                        ),
                    ),
                ),
            )

        async def update_collection(self, **_kwargs: Any) -> bool:
            return True

        async def create_collection(self, **kwargs: Any) -> None:
            create_calls.append(kwargs)

    operator.client = FakeQdrantClient()

    await operator._ensure_collections()

    names = [c["collection_name"] for c in create_calls]
    assert memory_module.MEDIA_TEXT_COL in names
    assert "media_collection_v3" not in names
    text_call = next(c for c in create_calls if c["collection_name"] == memory_module.MEDIA_TEXT_COL)
    assert text_call["vectors_config"].size == memory_module.MEDIA_TEXT_VECTOR_SIZE


@pytest.mark.asyncio
async def test_existing_collection_without_metadata_is_backfilled(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator.media_text_col = memory_module.MEDIA_TEXT_COL
    operator._init_lock = asyncio.Lock()
    update_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata=None,
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=operator.text_embedding_dimension,
                        ),
                    ),
                ),
            )

        async def update_collection(self, **kwargs: Any) -> bool:
            update_calls.append(kwargs)
            return True

    operator.client = FakeQdrantClient()

    await operator._ensure_collections()

    assert len(update_calls) == 2
    assert all(
        call["metadata"] == {
            memory_module.EMBEDDING_MODEL_METADATA_KEY: operator.emb_model,
            memory_module.EMBEDDING_DIMENSION_METADATA_KEY: operator.text_embedding_dimension,
        }
        for call in update_calls
    )


@pytest.mark.asyncio
async def test_legacy_text_collection_without_metadata_rejects_changed_model(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator.media_text_col = memory_module.MEDIA_TEXT_COL
    operator.emb_model = "new-model-with-same-dimension"
    operator._init_lock = asyncio.Lock()
    update_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata=None,
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=operator.text_embedding_dimension,
                        ),
                    ),
                ),
            )

        async def update_collection(self, **kwargs: Any) -> bool:
            update_calls.append(kwargs)
            return True

    operator.client = FakeQdrantClient()

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections({operator.chat_col})

    assert update_calls == []


@pytest.mark.asyncio
async def test_metadata_backfill_failure_is_retryable(memory_module: Any):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator.media_text_col = memory_module.MEDIA_TEXT_COL
    operator._init_lock = asyncio.Lock()
    update_attempts = 0

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata=None,
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=operator.text_embedding_dimension,
                        ),
                    ),
                ),
            )

        async def update_collection(self, **_kwargs: Any) -> bool:
            nonlocal update_attempts
            update_attempts += 1
            if update_attempts == 1:
                raise ConnectionError("temporary qdrant outage")
            return True

    operator.client = FakeQdrantClient()

    with pytest.raises(memory_module.CollectionMetadataBackfillError):
        await operator._ensure_collections({operator.chat_col})
    assert operator._collection_validation_errors == {}

    await operator._ensure_collections({operator.chat_col})

    assert update_attempts == 2
    assert operator._collection_validation_errors == {}


@pytest.mark.asyncio
async def test_existing_collection_with_unrelated_metadata_is_backfilled(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator.media_text_col = memory_module.MEDIA_TEXT_COL
    operator._init_lock = asyncio.Lock()
    update_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={"owner": "other-plugin"},
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=operator.text_embedding_dimension,
                        ),
                    ),
                ),
            )

        async def update_collection(self, **kwargs: Any) -> bool:
            update_calls.append(kwargs)
            return True

    operator.client = FakeQdrantClient()

    await operator._ensure_collections()

    assert len(update_calls) == 2
    assert all(
        call["metadata"] == {
            memory_module.EMBEDDING_MODEL_METADATA_KEY: operator.emb_model,
            memory_module.EMBEDDING_DIMENSION_METADATA_KEY: operator.text_embedding_dimension,
        }
        for call in update_calls
    )


@pytest.mark.asyncio
async def test_collection_embedding_mismatch_is_rejected(memory_module: Any):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={
                        memory_module.EMBEDDING_MODEL_METADATA_KEY: "old-model",
                        memory_module.EMBEDDING_DIMENSION_METADATA_KEY: 1024,
                    },
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=1024),
                    ),
                ),
            )

    operator.client = FakeQdrantClient()

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections()


@pytest.mark.asyncio
async def test_chat_collection_mismatch_does_not_block_multimodal_media(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = False
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()

    def collection_info(
        metadata: dict[str, str | int],
        vectors: Any,
    ) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                metadata=metadata,
                params=SimpleNamespace(vectors=vectors),
            ),
        )

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, collection_name: str) -> Any:
            if collection_name == operator.chat_col:
                return collection_info(
                    {
                        memory_module.EMBEDDING_MODEL_METADATA_KEY: "old-model",
                        memory_module.EMBEDDING_DIMENSION_METADATA_KEY: 1024,
                    },
                    SimpleNamespace(size=1024),
                )
            if collection_name == operator.media_col:
                return collection_info(
                    {
                        memory_module.EMBEDDING_MODEL_METADATA_KEY:
                            memory_module.QWEN_VL_EMBEDDING_MODEL,
                        memory_module.EMBEDDING_DIMENSION_METADATA_KEY:
                            memory_module.MEDIA_VECTOR_SIZE,
                    },
                    SimpleNamespace(size=memory_module.MEDIA_VECTOR_SIZE),
                )
            return collection_info(
                {
                    memory_module.EMBEDDING_MODEL_METADATA_KEY:
                        memory_module.QWEN_VL_EMBEDDING_MODEL,
                    memory_module.EMBEDDING_DIMENSION_METADATA_KEY:
                        memory_module.MEDIA_VECTOR_SIZE,
                },
                {
                    memory_module.MEDIA_TEXT_VECTOR: SimpleNamespace(
                        size=memory_module.MEDIA_VECTOR_SIZE,
                    ),
                    memory_module.MEDIA_IMAGE_VECTOR: SimpleNamespace(
                        size=memory_module.MEDIA_VECTOR_SIZE,
                    ),
                },
            )

    operator.client = FakeQdrantClient()

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections({operator.chat_col})

    await operator._ensure_collections(
        {operator.media_col, operator.media_multivector_col},
        validate_text_embedding=False,
    )

    assert operator._get_ready_collections() == {
        operator.media_col,
        operator.media_multivector_col,
    }


@pytest.mark.asyncio
async def test_legacy_media_mismatch_does_not_block_v3_insert_or_search(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = False
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    vector = [0.5] * memory_module.MEDIA_VECTOR_SIZE
    created_collections: list[str] = []
    upsert_collections: list[str] = []
    query_collections: list[str] = []

    class FakeQdrantClient:
        async def collection_exists(self, collection_name: str) -> bool:
            return collection_name == operator.media_col

        async def get_collection(self, collection_name: str) -> Any:
            assert collection_name == operator.media_col
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={
                        memory_module.EMBEDDING_MODEL_METADATA_KEY:
                            memory_module.QWEN_VL_EMBEDDING_MODEL,
                        memory_module.EMBEDDING_DIMENSION_METADATA_KEY: 1024,
                    },
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=1024),
                    ),
                ),
            )

        async def create_collection(self, **kwargs: Any) -> None:
            created_collections.append(kwargs["collection_name"])

        async def upsert(self, **kwargs: Any) -> None:
            upsert_collections.append(kwargs["collection_name"])

        async def query_points(self, **kwargs: Any) -> Any:
            query_collections.append(kwargs["collection_name"])
            return SimpleNamespace(points=[SimpleNamespace(id=1, score=0.9)])

    async def get_qwen_vl_independent_pair(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[float], list[float]]:
        return vector, vector

    async def get_qwen_vl_embedding(**_kwargs: Any) -> list[float]:
        return vector

    operator.client = FakeQdrantClient()
    operator._get_qwen_vl_independent_pair = get_qwen_vl_independent_pair
    operator._get_qwen_vl_embedding = get_qwen_vl_embedding

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections(
            {operator.media_col},
            validate_text_embedding=False,
        )

    assert await operator.insert_media(1, "data:image/png;base64,AAAA", "描述")
    candidates = await operator.search_meme_candidates("无奈")

    assert created_collections == [operator.media_multivector_col]
    assert upsert_collections == [operator.media_multivector_col]
    assert query_collections == [
        operator.media_multivector_col,
        operator.media_multivector_col,
    ]
    assert [media_id for media_id, _ in candidates] == [1]


@pytest.mark.asyncio
async def test_concurrent_collection_validation_failure_is_not_retried(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    get_collection_started = asyncio.Event()
    allow_failure = asyncio.Event()
    get_collection_calls = 0

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            nonlocal get_collection_calls
            get_collection_calls += 1
            get_collection_started.set()
            await allow_failure.wait()
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={
                        memory_module.EMBEDDING_MODEL_METADATA_KEY: "old-model",
                        memory_module.EMBEDDING_DIMENSION_METADATA_KEY: 1024,
                    },
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=1024),
                    ),
                ),
            )

    operator.client = FakeQdrantClient()
    first = asyncio.create_task(operator._ensure_collections())
    await get_collection_started.wait()
    second = asyncio.create_task(operator._ensure_collections())
    await asyncio.sleep(0)
    allow_failure.set()

    results = await asyncio.gather(first, second, return_exceptions=True)

    assert all(
        isinstance(result, memory_module.CollectionEmbeddingConfigMismatchError)
        for result in results
    )
    assert get_collection_calls == 1


@pytest.mark.asyncio
async def test_partial_collection_metadata_is_rejected(memory_module: Any):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return True

        async def get_collection(self, _collection_name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={
                        memory_module.EMBEDDING_MODEL_METADATA_KEY: operator.emb_model,
                    },
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=operator.text_embedding_dimension,
                        ),
                    ),
                ),
            )

    operator.client = FakeQdrantClient()

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections()

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections()


@pytest.mark.asyncio
async def test_text_collections_use_configured_embedding_dimension(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator.text_embedding_dimension = 1536
    operator._init_lock = asyncio.Lock()
    create_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return False

        async def create_collection(self, **kwargs: Any) -> None:
            create_calls.append(kwargs)

        async def create_payload_index(self, **_kwargs: Any) -> None:
            return None

    operator.client = FakeQdrantClient()

    await operator._ensure_collections()

    sizes = {
        call["collection_name"]: call["vectors_config"].size
        for call in create_calls
    }
    assert sizes == {
        "chat_collection": 1536,
        memory_module.MEDIA_TEXT_COL: 1536,
    }


@pytest.mark.asyncio
async def test_text_embedding_rejects_unexpected_dimension(memory_module: Any):
    operator = make_operator(memory_module)
    operator.text_embedding_dimension = 1024
    operator.emb_model = "test-model"
    embedding_requests = 0

    class FakeEmbeddings:
        async def create(self, **_kwargs: Any):
            nonlocal embedding_requests
            embedding_requests += 1
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.5] * 1536)]
            )

    operator.emb_client = SimpleNamespace(embeddings=FakeEmbeddings())

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._get_text_embedding("hello")
    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._get_text_embedding("hello again")

    assert embedding_requests == 1


@pytest.mark.asyncio
async def test_wrong_embedding_dimension_does_not_create_collections(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator.text_embedding_dimension = 1536
    operator._init_lock = asyncio.Lock()
    qdrant_calls: list[str] = []
    embedding_requests = 0

    class FakeEmbeddings:
        async def create(self, **_kwargs: Any):
            nonlocal embedding_requests
            embedding_requests += 1
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.5] * 1024)]
            )

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            qdrant_calls.append("collection_exists")
            return False

        async def create_collection(self, **_kwargs: Any) -> None:
            qdrant_calls.append("create_collection")

    operator.emb_client = SimpleNamespace(embeddings=FakeEmbeddings())
    operator.client = FakeQdrantClient()

    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections()
    with pytest.raises(memory_module.CollectionEmbeddingConfigMismatchError):
        await operator._ensure_collections()

    assert embedding_requests == 1
    assert qdrant_calls == []


@pytest.mark.asyncio
async def test_embedding_probe_outage_is_soft_failure(memory_module: Any):
    operator = make_operator(memory_module)
    operator.text_only = True
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    probe_attempts = 0
    created_collections: list[str] = []

    class FakeEmbeddings:
        async def create(self, **_kwargs: Any):
            nonlocal probe_attempts
            probe_attempts += 1
            raise ConnectionError("temporary embedding outage")

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return False

        async def create_collection(self, **kwargs: Any) -> None:
            created_collections.append(kwargs["collection_name"])

        async def create_payload_index(self, **_kwargs: Any) -> None:
            return None

    operator.emb_client = SimpleNamespace(embeddings=FakeEmbeddings())
    operator.client = FakeQdrantClient()

    await operator._ensure_collections({operator.chat_col})

    assert probe_attempts == 1
    assert created_collections == [operator.chat_col]


@pytest.mark.asyncio
async def test_multimodal_media_remains_available_when_text_embedding_is_unavailable(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = False
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    embedding_requests = 0
    created_collections: list[str] = []
    upsert_collections: list[str] = []
    vector = [0.5] * memory_module.MEDIA_VECTOR_SIZE

    class FakeEmbeddings:
        async def create(self, **_kwargs: Any):
            nonlocal embedding_requests
            embedding_requests += 1
            raise ConnectionError("embedding provider unavailable")

    class FakeQdrantClient:
        async def collection_exists(self, _collection_name: str) -> bool:
            return False

        async def create_collection(self, **kwargs: Any) -> None:
            created_collections.append(kwargs["collection_name"])

        async def create_payload_index(self, **_kwargs: Any) -> None:
            return None

        async def upsert(self, **kwargs: Any) -> None:
            upsert_collections.append(kwargs["collection_name"])

    async def get_qwen_vl_independent_pair(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[float], list[float]]:
        return vector, vector

    operator.emb_client = SimpleNamespace(embeddings=FakeEmbeddings())
    operator.client = FakeQdrantClient()
    operator._get_qwen_vl_independent_pair = get_qwen_vl_independent_pair

    await operator._ensure_collections({operator.chat_col})

    assert await operator.insert_media(1, "data:image/png;base64,AAAA", "描述")
    assert embedding_requests == 1
    assert created_collections == [
        operator.chat_col,
        operator.media_multivector_col,
    ]
    assert upsert_collections == [operator.media_multivector_col]


@pytest.mark.asyncio
async def test_insert_chat_stores_vector_from_configured_model(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.chat_col = "chat_collection"
    operator.emb_model = "Qwen/Qwen3-Embedding-0.6B"
    operator.text_embedding_dimension = 1536
    embedding_requests: list[dict[str, Any]] = []
    upsert_calls: list[dict[str, Any]] = []

    class FakeEmbeddings:
        async def create(self, **kwargs: Any):
            embedding_requests.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.5] * 1536)]
            )

    class FakeQdrantClient:
        async def upsert(self, **kwargs: Any) -> None:
            upsert_calls.append(kwargs)

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    operator.emb_client = SimpleNamespace(embeddings=FakeEmbeddings())
    operator.client = FakeQdrantClient()
    operator._ensure_collections = ensure_collections

    await operator.insert_chat("hello", "group-1")

    assert embedding_requests == [{
        "input": ["hello"],
        "model": "Qwen/Qwen3-Embedding-0.6B",
    }]
    assert len(upsert_calls) == 1
    assert len(upsert_calls[0]["points"][0].vector) == 1536


@pytest.mark.asyncio
async def test_text_mode_insert_media_uses_text_embedding(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True
    operator.media_embedding_version = memory_module.MEDIA_TEXT_EMBEDDING_VERSION
    vector = [0.5] * memory_module.MEDIA_TEXT_VECTOR_SIZE
    upsert_calls: list[dict[str, Any]] = []

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_text_embedding(text: str) -> list[float]:
        return vector

    class FakeQdrantClient:
        async def upsert(self, **kwargs: Any) -> None:
            upsert_calls.append(kwargs)

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_text_embedding
    operator.client = FakeQdrantClient()

    assert await operator.insert_media(1, "data:image/png;base64,AAAA", "熊猫头流泪") is True
    assert len(upsert_calls) == 1
    assert upsert_calls[0]["collection_name"] == memory_module.MEDIA_TEXT_COL
    point = upsert_calls[0]["points"][0]
    assert point.id == 1
    assert point.vector == vector
    assert (
        point.payload["embedding_version"]
        == memory_module.MEDIA_TEXT_EMBEDDING_VERSION
    )


@pytest.mark.asyncio
async def test_insert_media_skips_when_disabled(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = False

    assert await operator.insert_media(1, "data:image/png;base64,AAAA", "描述") is False


@pytest.mark.asyncio
async def test_text_mode_insert_media_skips_missing_description(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    operator._ensure_collections = ensure_collections

    assert await operator.insert_media(1, "data:image/png;base64,AAAA", "") is False


@pytest.mark.asyncio
async def test_text_mode_insert_media_skips_embedding_failure(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_text_embedding(text: str) -> None:
        return None

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_text_embedding

    assert (
        await operator.insert_media(1, "data:image/png;base64,AAAA", "熊猫头流泪")
        is False
    )


@pytest.mark.asyncio
async def test_text_mode_search_meme_candidates_single_route(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True
    vector = [0.5] * memory_module.MEDIA_TEXT_VECTOR_SIZE
    query_calls: list[dict[str, Any]] = []

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_text_embedding(text: str) -> list[float]:
        return vector

    class FakeQdrantClient:
        async def query_points(self, **kwargs: Any) -> SimpleNamespace:
            query_calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id=7, score=0.9),
                    SimpleNamespace(id=8, score=0.8),
                ]
            )

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_text_embedding
    operator.client = FakeQdrantClient()

    result = await operator.search_meme_candidates("无奈", limit=10)

    assert result == [(7, 0.9), (8, 0.8)]
    assert len(query_calls) == 1
    assert query_calls[0]["collection_name"] == memory_module.MEDIA_TEXT_COL
    assert "using" not in query_calls[0]


@pytest.mark.asyncio
async def test_text_mode_search_similar_meme_disabled(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True

    result = await operator.search_similar_meme("/tmp/whatever.png")

    assert result == []


@pytest.mark.asyncio
async def test_search_meme_candidates_skips_when_disabled(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = False

    assert await operator.search_meme_candidates("无奈") == []


@pytest.mark.asyncio
async def test_text_mode_search_meme_candidates_embedding_failure(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_text_embedding(text: str) -> None:
        return None

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_text_embedding

    assert await operator.search_meme_candidates("无奈") == []


@pytest.mark.asyncio
async def test_text_mode_search_meme_candidates_empty_result(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = True
    vector = [0.5] * memory_module.MEDIA_TEXT_VECTOR_SIZE

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_text_embedding(text: str) -> list[float]:
        return vector

    class FakeQdrantClient:
        async def query_points(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(points=[])

    operator._ensure_collections = ensure_collections
    operator._get_text_embedding = get_text_embedding
    operator.client = FakeQdrantClient()

    assert await operator.search_meme_candidates("无奈") == []


@pytest.mark.asyncio
async def test_search_similar_meme_skips_when_disabled(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = False

    assert await operator.search_similar_meme("/tmp/whatever.png") == []


def test_configure_forces_text_mode_when_qwen_token_missing(
    memory_module: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        memory_module,
        "plugin_config",
        SimpleNamespace(
            qdrant_uri="http://127.0.0.1:6333",
            qwen_token="",
            meme_embedding_mode="multimodal",
            embedding_api_key="k",
            embedding_base_url="http://emb",
            rerank_api_url="",
            rerank_api_key="",
            qdrant_api_key="",
        ),
    )
    operator = object.__new__(memory_module.VectorDBOperator)

    operator._configure()

    assert operator.text_only is True
    assert operator.effective_meme_embedding_mode == "text"
    assert operator.media_embedding_version == memory_module.MEDIA_TEXT_EMBEDDING_VERSION


def test_configure_keeps_multimodal_when_qwen_token_set(
    memory_module: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        memory_module,
        "plugin_config",
        SimpleNamespace(
            qdrant_uri="http://127.0.0.1:6333",
            qwen_token="sk-xxx",
            meme_embedding_mode="multimodal",
            embedding_api_key="k",
            embedding_base_url="http://emb",
            rerank_api_url="",
            rerank_api_key="",
            qdrant_api_key="",
        ),
    )
    operator = object.__new__(memory_module.VectorDBOperator)

    operator._configure()

    assert operator.text_only is False
    assert operator.effective_meme_embedding_mode == "multimodal"
    assert (
        operator.media_embedding_version
        == memory_module.MEDIA_MULTIMODAL_EMBEDDING_VERSION
    )


def test_configure_strips_duplicate_embeddings_suffix(
    memory_module: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        memory_module,
        "plugin_config",
        SimpleNamespace(
            qdrant_uri="http://127.0.0.1:6333",
            qwen_token="sk-xxx",
            meme_embedding_mode="multimodal",
            embedding_api_key="k",
            embedding_base_url="http://emb/v1/embeddings",
            rerank_api_url="",
            rerank_api_key="",
            qdrant_api_key="",
        ),
    )
    operator = object.__new__(memory_module.VectorDBOperator)

    operator._configure()

    base_url = str(operator.emb_client.base_url).rstrip("/")
    assert base_url.endswith("/embeddings") is False
    assert base_url == "http://emb/v1"


def test_configure_uses_configured_text_embedding_model(
    memory_module: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        memory_module,
        "plugin_config",
        SimpleNamespace(
            qdrant_uri="http://127.0.0.1:6333",
            qwen_token="sk-xxx",
            meme_embedding_mode="multimodal",
            embedding_api_key="k",
            embedding_base_url="http://emb",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_dimension=1536,
            rerank_api_url="",
            rerank_api_key="",
            qdrant_api_key="",
        ),
    )
    operator = object.__new__(memory_module.VectorDBOperator)

    operator._configure()

    assert operator.emb_model == "Qwen/Qwen3-Embedding-0.6B"
    assert operator.text_embedding_dimension == 1536


@pytest.mark.asyncio
async def test_ensure_collections_creates_chat_and_multimodal_collections(
    memory_module: Any,
):
    operator = make_operator(memory_module)
    operator.text_only = False
    operator.chat_col = "chat_collection"
    operator._init_lock = asyncio.Lock()
    create_calls: list[str] = []

    class FakeQdrantClient:
        async def collection_exists(self, collection_name: str) -> bool:
            return False

        async def create_collection(self, **kwargs: Any) -> None:
            create_calls.append(kwargs["collection_name"])

        async def create_payload_index(self, **kwargs: Any) -> None:
            return None

    operator.client = FakeQdrantClient()

    await operator._ensure_collections()

    assert "chat_collection" in create_calls
    assert operator.media_col in create_calls
    assert operator.media_multivector_col in create_calls


@pytest.mark.asyncio
async def test_search_similar_meme_multimodal_path(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    operator.text_only = False
    vector = [0.1] * memory_module.MEDIA_VECTOR_SIZE

    async def ensure_collections(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_qwen_vl_embedding(**kwargs: Any) -> list[float]:
        return vector

    async def search_routes(*_: Any, **__: Any) -> list[tuple[int, float]]:
        return [(1, 0.9), (2, 0.8)]

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_embedding = get_qwen_vl_embedding
    operator._search_media_routes = search_routes

    result = await operator.search_similar_meme("/tmp/whatever.png", limit=6)

    assert set(result) == {1, 2}
