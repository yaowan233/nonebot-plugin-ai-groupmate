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
    operator._collections_ready = False
    operator._init_lock = asyncio.Lock()
    create_calls: list[dict[str, Any]] = []

    class FakeQdrantClient:
        async def collection_exists(self, collection_name: str) -> bool:
            return collection_name != "media_collection_v3"

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

    async def ensure_collections() -> None:
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

    async def ensure_collections() -> None:
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

    async def ensure_collections() -> None:
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

    async def ensure_collections() -> None:
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

    async def ensure_collections() -> None:
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
