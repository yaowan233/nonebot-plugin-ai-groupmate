import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("private", [True, False])
async def test_search_then_send_to_current_conversation(monkeypatch, private):
    from nonebot_plugin_alconna import Target, UniMessage

    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

    search = AsyncMock()
    search.ainvoke.return_value = {"images": [
        {"url": "https://example.com/cat.png", "description": "小猫"},
        {"url": "https://example.com/cat.png", "description": "重复"},
        {"url": "file:///etc/passwd"},
    ]}

    def factory(**kwargs):
        assert kwargs["include_images"] is True
        assert kwargs["include_image_descriptions"] is True
        return search

    monkeypatch.setattr(module, "TavilySearch", factory)
    added = []
    session = SimpleNamespace(commit=AsyncMock(), add=added.append)
    target = Target("123", private=private)
    sent = []

    async def send(message, **kwargs):
        session.commit.assert_awaited_once()
        sent.append((message, kwargs))
        return SimpleNamespace(msg_ids=[{"message_id": 456}])

    monkeypatch.setattr(UniMessage, "send", send)
    search_tool, send_tool, preview_tool = module.create_web_image_tools(
        session, "123", None, tavily_api_key="test", bot_name="bot", send_target=target,
    )
    rejected = json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": "guessed"}))
    assert rejected["reason_code"] == "image_not_available"
    result = json.loads(await search_tool.ainvoke({"query": "小猫"}))
    assert len(result["data"]["images"]) == 1
    image_id = result["data"]["images"][0]["image_id"]
    await preview_tool.ainvoke({"image_ids": [image_id]})
    delivered = json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": image_id}))
    assert delivered["delivery_state"] == "completed"
    assert sent[0][1]["target"] is target
    assert sent[0][0][0].url == "https://example.com/cat.png"
    assert added[0].content.startswith("id: 456\n")
    assert "小猫" in added[0].content
    duplicate = json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": image_id}))
    assert duplicate["reason_code"] == "duplicate_image"
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_search_failures_and_send_limit(monkeypatch):
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

    search = AsyncMock()
    monkeypatch.setattr(module, "TavilySearch", lambda **kwargs: search)
    session = SimpleNamespace(commit=AsyncMock(), add=lambda row: None)
    search_tool, send_tool, preview_tool = module.create_web_image_tools(session, "1", None, tavily_api_key="test", bot_name="bot")
    search.ainvoke.return_value = {"images": []}
    assert json.loads(await search_tool.ainvoke({"query": "cat"}))["reason_code"] == "no_images"
    search.ainvoke.side_effect = TimeoutError()
    assert json.loads(await search_tool.ainvoke({"query": "cat"}))["reason_code"] == "timeout"
    search.ainvoke.side_effect = None
    search.ainvoke.return_value = {"images": [f"https://example.com/{i}.png" for i in range(6)]}
    result = json.loads(await search_tool.ainvoke({"query": "cat"}))
    assert len(result["data"]["images"]) == 5
    send = AsyncMock(return_value=SimpleNamespace(msg_ids=[]))
    monkeypatch.setattr(UniMessage, "send", send)
    for i in range(1, 5):
        await preview_tool.ainvoke({"image_ids": [f"web-image-{i}"]})
    for i in range(1, 4):
        assert json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": f"web-image-{i}"}))["status"] == "succeeded"
    assert json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": "web-image-4"}))["reason_code"] == "image_send_limit"
    assert send.await_count == 3


@pytest.mark.asyncio
async def test_missing_key_expired_request_and_unknown_delivery(monkeypatch):
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

    session = SimpleNamespace(commit=AsyncMock(), add=lambda row: None)
    missing, _, _ = module.create_web_image_tools(session, "1", None, tavily_api_key="", bot_name="bot")
    assert json.loads(await missing.ainvoke({"query": "cat"}))["reason_code"] == "missing_api_key"
    search = AsyncMock()
    search.ainvoke.return_value = {"images": ["https://example.com/cat.png"]}
    monkeypatch.setattr(module, "TavilySearch", lambda **kwargs: search)
    active = AsyncMock(return_value=True)
    monkeypatch.setattr(module, "is_request_active", active)
    search_tool, send_tool, preview_tool = module.create_web_image_tools(session, "1", "request", tavily_api_key="test", bot_name="bot")
    await search_tool.ainvoke({"query": "cat"})
    await preview_tool.ainvoke({"image_ids": ["web-image-1"]})
    active.return_value = False
    assert json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": "web-image-1"}))["reason_code"] == "request_expired"
    active.return_value = True
    send = AsyncMock(side_effect=RuntimeError("adapter failed"))
    monkeypatch.setattr(UniMessage, "send", send)
    assert json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": "web-image-1"}))["delivery_state"] == "unknown"
    assert json.loads(await send_tool.ainvoke({"allow_unverified": True, "image_id": "web-image-1"}))["reason_code"] == "duplicate_image"
    send.assert_awaited_once()


@pytest.mark.parametrize("as_exception", [False, True])
@pytest.mark.asyncio
async def test_web_image_search_reports_quota_exhaustion(monkeypatch, as_exception):
    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

    search = AsyncMock()
    if as_exception:
        search.ainvoke.side_effect = RuntimeError("Error 432: tvly-secret-value")
    else:
        search.ainvoke.return_value = {"error": RuntimeError("Error 432: tvly-secret-value")}
    monkeypatch.setattr(module, "TavilySearch", lambda **kwargs: search)
    search_tool, _, _ = module.create_web_image_tools(None, "group-1", None, tavily_api_key="test", bot_name="bot")
    raw = await search_tool.ainvoke({"query": "cat"})
    result = json.loads(raw)
    assert result["reason_code"] == "quota_exhausted"
    assert result["retryable"] is False
    assert result["data"]["http_status"] == 432
    assert "tvly-secret-value" not in raw


def test_image_send_is_registered_as_side_effect():
    from nonebot_plugin_ai_groupmate.agent.graph import SIDE_EFFECT_TOOL_NAMES

    assert "send_web_image" in SIDE_EFFECT_TOOL_NAMES


@pytest.mark.parametrize("text", ["网上搜一张猫咪图片发给我", "帮我搜图 初音未来", "搜索风景图片然后发两张", "去网上找一张壁纸发给我"])
def test_online_image_request_does_not_force_local_meme_tools(text):
    from nonebot_plugin_ai_groupmate import _is_explicit_meme_request

    assert not _is_explicit_meme_request(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_images", [True, False])
async def test_preview_supplies_actual_image_or_vision_summary(monkeypatch, supports_images):
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module
    from nonebot_plugin_ai_groupmate.agent.graph import _normalize_tool_result

    search = AsyncMock()
    search.ainvoke.return_value = {"images": ["https://example.com/cat.png"]}
    monkeypatch.setattr(module, "TavilySearch", lambda **kwargs: search)
    uri = "data:image/png;base64,Ynl0ZXM="
    download = AsyncMock(return_value=uri)
    monkeypatch.setattr(module, "_download_preview", download)
    summarize = AsyncMock(return_value="实际画面是一只猫")
    session = SimpleNamespace(commit=AsyncMock(), add=lambda row: None)
    search_tool, send_tool, preview = module.create_web_image_tools(
        session, "1", None, tavily_api_key="test", bot_name="bot",
        supports_images=supports_images, image_summarizer=summarize,
    )
    await search_tool.ainvoke({"query": "cat"})
    assert json.loads(await send_tool.ainvoke({"image_id": "web-image-1"}))["reason_code"] == "preview_required"
    result = await preview.ainvoke({"image_ids": ["web-image-1"]})
    text, blocks = _normalize_tool_result(result)
    if supports_images:
        assert blocks is not None
        assert {"type": "image_url", "image_url": {"url": uri}} in blocks
        summarize.assert_not_awaited()
    else:
        assert "实际画面是一只猫" in text
        summarize.assert_awaited_once_with([{"type": "image_url", "image_url": {"url": uri}}])
    await preview.ainvoke({"image_ids": ["web-image-1"]})
    download.assert_awaited_once()
    sent = []

    async def send(message, **kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[])

    monkeypatch.setattr(UniMessage, "send", send)
    result = json.loads(await send_tool.ainvoke({"image_id": "web-image-1"}))
    assert result["status"] == "succeeded"
    assert sent[0][0].raw == b"bytes"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["no_vision", "download", "summary"])
async def test_preview_fallback_requires_explicit_unverified_send(monkeypatch, failure):
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

    search = AsyncMock()
    search.ainvoke.return_value = {"images": ["https://example.com/cat.png"]}
    monkeypatch.setattr(module, "TavilySearch", lambda **kwargs: search)
    download = AsyncMock(return_value="data:image/png;base64,Ynl0ZXM=")
    if failure == "download":
        download.side_effect = ValueError("broken")
    monkeypatch.setattr(module, "_download_preview", download)
    session = SimpleNamespace(commit=AsyncMock(), add=lambda row: None)
    search_tool, send_tool, preview = module.create_web_image_tools(
        session, "1", None, tavily_api_key="test", bot_name="bot",
        supports_images=failure == "download",
        image_summarizer=AsyncMock(return_value="") if failure == "summary" else None,
    )
    await search_tool.ainvoke({"query": "cat"})
    assert json.loads(await preview.ainvoke({"image_ids": ["forged"]}))["status"] == "failed"
    result = json.loads(await preview.ainvoke({"image_ids": ["web-image-1"]}))
    assert result["data"]["images"][0]["status"] == "description_only"
    assert json.loads(await send_tool.ainvoke({"image_id": "web-image-1"}))["reason_code"] == "unverified_image"
    monkeypatch.setattr(UniMessage, "send", AsyncMock(return_value=SimpleNamespace(msg_ids=[])))
    result = json.loads(await send_tool.ainvoke({"image_id": "web-image-1", "allow_unverified": True}))
    assert result["data"]["inspection_status"] == "description_only"


@pytest.mark.asyncio
async def test_preview_download_validates_images_limits_and_redirects(monkeypatch):
    import io

    import httpx
    from PIL import Image

    from nonebot_plugin_ai_groupmate.agent import image_search_tools as module

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        return httpx.Response(200, content=payload)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(module.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("127.0.0.1" if host == "127.0.0.1" else "8.8.8.8", port))])
    assert (await module._download_preview("https://example.com/image")).startswith("data:image/")
    with pytest.raises(ValueError, match="non_public"):
        await module._download_preview("https://example.com/redirect")
    assert "http://127.0.0.1/private" not in requests
    monkeypatch.setattr(module, "MAX_PREVIEW_BYTES", 1)
    with pytest.raises(ValueError, match="image_too_large"):
        await module._download_preview("https://example.com/image")
    monkeypatch.setattr(module, "MAX_PREVIEW_BYTES", 1024)
    payload = b"not an image"
    with pytest.raises(ValueError, match="invalid_image"):
        await module._download_preview("https://example.com/image")
