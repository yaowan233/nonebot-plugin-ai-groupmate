from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI


@pytest.fixture
def chat_operator(monkeypatch):
    from nonebot_plugin_ai_groupmate import memory

    operator: Any = object.__new__(memory.VectorDBOperator)
    operator.enabled = True
    operator.chat_col = memory.CHAT_COLLECTION
    operator.emb_model = "test-model"
    operator.configured_embedding_dimension = None
    operator.text_embedding_dimension = 2
    operator._ensure_collections = AsyncMock()
    operator.client = SimpleNamespace(upsert=AsyncMock(return_value=SimpleNamespace(status="completed")))
    monkeypatch.setattr(memory.asyncio, "sleep", AsyncMock())
    return operator


@pytest.mark.asyncio
async def test_embedding_connect_timeout_retries_only_failed_sub_batch(monkeypatch):
    from nonebot_plugin_ai_groupmate import memory

    requests: list[list[str]] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        import json

        inputs = json.loads(request.content)["input"]
        requests.append(inputs)
        if inputs == ["text-20"] and len(requests) <= 3:
            raise httpx.ConnectTimeout("connection timed out", request=request)
        return httpx.Response(200, json={
            "data": [
                {"embedding": [0.1, 0.2], "index": index, "object": "embedding"}
                for index, _ in enumerate(inputs)
            ],
            "model": "test-model", "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    operator: Any = object.__new__(memory.VectorDBOperator)
    operator.emb_model = "test-model"
    operator.configured_embedding_dimension = None
    operator.text_embedding_dimension = 2
    sleep = AsyncMock()
    monkeypatch.setattr(memory.asyncio, "sleep", sleep)
    async with AsyncOpenAI(
        api_key="test-key", base_url="https://embedding.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        operator.emb_client = client
        vectors = await operator._get_batch_text_embeddings([f"text-{i}" for i in range(21)])

    assert len(vectors) == 21
    assert [len(inputs) for inputs in requests] == [20, 1, 1, 1]
    delays = [call.args[0] for call in sleep.call_args_list]
    assert 5.0 in delays
    assert 10.0 in delays


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "connection", 408, 503, 401, 400])
async def test_embedding_retry_budget_does_not_multiply_sdk_or_outer_retries(
    chat_operator, monkeypatch, failure,
):
    from nonebot_plugin_ai_groupmate import utils, memory

    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "timeout":
            raise httpx.ConnectTimeout("timed out", request=request)
        if failure == "connection":
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(failure, json={"error": {"message": "test failure"}})

    monkeypatch.setattr(utils, "DB", chat_operator)
    async with AsyncOpenAI(
        api_key="test-key", base_url="https://embedding.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        chat_operator.emb_client = client
        with pytest.raises(memory.EmbeddingProviderUnavailableError):
            await utils.insert_vectors_with_retry(["pending message"], "embedding-failure")
        assert client.max_retries == 2  # The shared client is not reconfigured.

    expected_attempts = 1 if failure in {400, 401} else 3
    assert len(requests) == expected_attempts
    assert all(request.extensions["timeout"]["read"] == 60.0 for request in requests)
    chat_operator.client.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_qdrant_retries_only_failed_write_chunk(chat_operator):
    from qdrant_client.http.exceptions import UnexpectedResponse

    chat_operator._get_batch_text_embeddings = AsyncMock(return_value=[[0.1, 0.2]] * 65)
    completed = SimpleNamespace(status="completed")
    chat_operator.client.upsert.side_effect = [
        completed,
        UnexpectedResponse(408, "Request Timeout", b"timed out", httpx.Headers()),
        completed, completed,
    ]
    await chat_operator.batch_insert([f"message-{i}" for i in range(65)], "group-1")

    chat_operator._get_batch_text_embeddings.assert_awaited_once()
    batches = [call.kwargs["points"] for call in chat_operator.client.upsert.call_args_list]
    assert [len(batch) for batch in batches] == [32, 32, 32, 1]
    assert batches[1] == batches[2]
    ids = [point.id for batch in [batches[0], batches[2], batches[3]] for point in batch]
    assert len(set(ids)) == 65


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [408, 401, "wrapped_timeout", "wait_timeout", "acknowledged", "unknown"])
async def test_failed_qdrant_write_keeps_sql_messages_pending(chat_operator, monkeypatch, failure):
    from nonebot_plugin_orm import get_session
    from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

    from nonebot_plugin_ai_groupmate import utils
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    chat_operator._get_batch_text_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    upsert = chat_operator.client.upsert
    if isinstance(failure, int):
        upsert.side_effect = UnexpectedResponse(failure, "test failure", b"test", httpx.Headers())
    elif failure == "wrapped_timeout":
        upsert.side_effect = ResponseHandlingException(httpx.ReadTimeout("timed out"))
    else:
        upsert.return_value = SimpleNamespace(status=failure)
    monkeypatch.setattr(utils, "DB", chat_operator)

    session_id = f"qdrant-failure-{failure}"
    async with get_session() as session:
        row = ChatHistory(
            session_id=session_id, user_id="tester", user_name="tester",
            content_type="text", content="preserve this message", vectorized=False,
            vectorized_version=0,
        )
        session.add(row)
        await session.commit()
        try:
            result = await utils.process_and_vectorize_session_chats(session, session_id)
            await session.refresh(row)
            assert result is not None
            assert result["failed_groups"] == 1
            assert result["processed_groups"] == 0
            assert row.vectorized is False
            assert row.vectorized_version == 0
        finally:
            await session.delete(row)
            await session.commit()

    assert upsert.await_count == (1 if failure in {401, "unknown"} else 3)
    chat_operator._get_batch_text_embeddings.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("retry_after", "expected_delay"), [("12", 12.0), ("999999", 30.0), ("invalid", 2.0)])
async def test_qdrant_rate_limit_retry_after_is_bounded(chat_operator, retry_after, expected_delay):
    from qdrant_client.http.exceptions import UnexpectedResponse

    from nonebot_plugin_ai_groupmate import memory

    chat_operator._get_batch_text_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    chat_operator.client.upsert.side_effect = [
        UnexpectedResponse(429, "Too Many Requests", b"limited", httpx.Headers({"Retry-After": retry_after})),
        SimpleNamespace(status="completed"),
    ]
    await chat_operator.batch_insert(["pending message"], "rate-limited-group")
    assert isinstance(memory.asyncio.sleep, AsyncMock)
    memory.asyncio.sleep.assert_awaited_once_with(expected_delay)


@pytest.mark.asyncio
async def test_qdrant_cancellation_is_not_retried(chat_operator):
    import asyncio

    chat_operator._get_batch_text_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    chat_operator.client.upsert.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await chat_operator.batch_insert(["pending message"], "cancelled-group")
    chat_operator.client.upsert.assert_awaited_once()
