from __future__ import annotations

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
async def test_insert_media_reports_embedding_failure(memory_module: Any):
    operator = make_operator(memory_module)
    operator.enabled = True
    upsert_called = False

    async def ensure_collections() -> None:
        return None

    async def get_embedding(**_: Any) -> None:
        return None

    class FakeQdrantClient:
        async def upsert(self, **_: Any) -> None:
            nonlocal upsert_called
            upsert_called = True

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_embedding = get_embedding
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

    async def get_embedding(**_: Any) -> list[float]:
        return vector

    class FakeQdrantClient:
        async def upsert(self, **kwargs: Any) -> None:
            upsert_calls.append(kwargs)

    operator._ensure_collections = ensure_collections
    operator._get_qwen_vl_embedding = get_embedding
    operator.client = FakeQdrantClient()

    assert await operator.insert_media(7, "data:image/png;base64,AAAA", "描述") is True
    assert upsert_calls[0]["wait"] is True
    point = upsert_calls[0]["points"][0]
    assert point.payload["embedding_version"] == memory_module.MEDIA_EMBEDDING_VERSION
