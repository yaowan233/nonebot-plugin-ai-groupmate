from __future__ import annotations

import json
import uuid
import asyncio
import logging
import datetime
from io import BytesIO
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image


def image_bytes(format="PNG", size=(64, 64)):
    output = BytesIO()
    Image.new("RGB", size, "pink").save(output, format=format)
    return output.getvalue()


def test_upload_preparation_preserves_small_original_and_converts_large_or_animated_images():
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import MAX_UPLOAD_BYTES, prepare_serpapi_image

    small = image_bytes()
    assert prepare_serpapi_image(small) == (small, "png", "image/png")
    noisy = Image.effect_noise((2200, 2200), 100).convert("RGB")
    output = BytesIO()
    noisy.save(output, format="PNG")
    assert len(output.getvalue()) > MAX_UPLOAD_BYTES
    for original in (output.getvalue(), image_bytes("GIF")):
        data, extension, mime = prepare_serpapi_image(original)
        assert 0 < len(data) <= MAX_UPLOAD_BYTES
        assert (extension, mime) == ("jpg", "image/jpeg")
        with Image.open(BytesIO(data)) as image:
            assert image.format == "JPEG"


def test_invalid_image_is_rejected():
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import prepare_serpapi_image
    from nonebot_plugin_ai_groupmate.agent.web_image_search import WebDetectionError

    with pytest.raises(WebDetectionError, match="source_image_invalid"):
        prepare_serpapi_image(b"not an image")


def test_normalization_keeps_character_evidence_without_api_links_or_credentials():
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import normalize_lens_response

    result = normalize_lens_response({
        "search_metadata": {"json_endpoint": "https://serpapi.com/searches/private?api_key=secret"},
        "exact_matches": [
            {"title": "<b>Akali</b> &amp; League of Legends", "link": "https://example.org/akali", "source": "Character wiki", "snippet": "Akali, the Rogue Assassin", "serpapi_link": "private"},
            {"title": "duplicate", "link": "https://example.org/akali"},
        ],
        "visual_matches": [
            {"title": "bad", "link": "javascript:alert(1)"},
            {"title": "secret endpoint", "link": "https://serpapi.com/search?api_key=secret"},
            {"title": "bad credential", "link": "https://example.org/?api_key=secret"},
            {"title": "Similar costume", "link": "https://example.org/similar"},
            {"title": "overflow", "link": "https://example.org/overflow"},
        ],
        "organic_results": [{"title": "Akali official", "link": "https://example.org/official", "snippet": "secret should be redacted"}],
    }, max_results=1, api_key="secret")
    assert result["exact_matches"] == [{"title": "Akali & League of Legends", "url": "https://example.org/akali", "source": "Character wiki", "snippet": "Akali, the Rogue Assassin"}]
    assert result["visual_matches"] == [{"title": "Similar costume", "url": "https://example.org/similar"}]
    assert "secret" not in json.dumps(result)
    assert "search_metadata" not in result


async def test_multipart_upload_then_lens_uses_original_bytes_and_uploaded_id(caplog):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import request_serpapi_lens, prepare_serpapi_image

    original = image_bytes()
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/image":
            assert request.method == "POST"
            assert original in request.content
            assert b'name="image"' in request.content
            assert b'name="api_key"' in request.content
            return httpx.Response(200, json={"image_id": "uploaded-image"})
        assert request.url.params["engine"] == "google_lens"
        assert request.url.params["image_id"] == "uploaded-image"
        assert request.url.params["type"] == "all"
        assert request.url.params["api_key"] == "test-secret"
        assert "url" not in request.url.params
        return httpx.Response(200, json={"search_metadata": {"status": "Success"}, "visual_matches": [{"title": "Akali", "link": "https://example.org/akali"}]})

    with caplog.at_level(logging.INFO, logger="httpx"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await request_serpapi_lens(prepare_serpapi_image(original), ScopedConfig(serpapi_api_key="test-secret"), client=client)
    assert len(requests) == 2
    assert result["visual_matches"] == [{"title": "Akali", "url": "https://example.org/akali"}]
    # A missing response section is not evidence that a separate exact search failed.
    assert "exact_matches" not in result
    assert "test-secret" not in caplog.text


@pytest.mark.parametrize(("status", "body", "expected"), [
    (401, {"error": "secret"}, "serpapi_auth_failed"),
    (429, {"error": "secret"}, "serpapi_rate_limited"),
    (500, {"error": "secret"}, "serpapi_request_failed"),
    (200, {"error": "secret"}, "serpapi_search_failed"),
    (200, {}, "invalid_provider_response"),
])
async def test_upload_errors_stop_before_search_without_exposing_provider_text(status, body, expected):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import request_serpapi_lens
    from nonebot_plugin_ai_groupmate.agent.web_image_search import WebDetectionError

    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(WebDetectionError) as exc:
            await request_serpapi_lens((b"image", "jpg", "image/jpeg"), ScopedConfig(serpapi_api_key="secret"), client=client)
    assert len(calls) == 1
    assert exc.value.reason_code == expected
    assert "secret" not in exc.value.user_message


@pytest.mark.parametrize("body", [{"search_metadata": {"status": "Processing"}}, {"search_metadata": {"status": "Success"}, "error": "verify request"}, []])
async def test_lens_verification_and_malformed_responses_are_not_cached_as_success(body):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import request_serpapi_lens
    from nonebot_plugin_ai_groupmate.agent.web_image_search import WebDetectionError

    def handle(request):
        return httpx.Response(200, json={"image_id": "uploaded"} if request.method == "POST" else body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(WebDetectionError):
            await request_serpapi_lens((b"image", "jpg", "image/jpeg"), ScopedConfig(serpapi_api_key="secret"), client=client)


async def test_upload_and_search_share_one_deadline():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.serpapi_search import request_serpapi_lens
    from nonebot_plugin_ai_groupmate.agent.web_image_search import WebDetectionError

    async def handle(request):
        await asyncio.sleep(0.035)
        return httpx.Response(200, json={"image_id": "uploaded"} if request.method == "POST" else {"search_metadata": {"status": "Success"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(WebDetectionError, match="serpapi_unavailable"):
            await request_serpapi_lens((b"image", "jpg", "image/jpeg"), ScopedConfig(serpapi_api_key="secret", serpapi_timeout_seconds=0.05), client=client)


async def test_provider_caches_and_monthly_limits_are_isolated():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import SerpApiImageUsage, GoogleWebDetectionUsage
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.web_image_search import WebDetectionQuotaExceeded, perform_web_detection

    sha = uuid.uuid4().hex.ljust(64, "0")
    now = datetime.datetime(2097, 10, 1)
    google = AsyncMock(return_value={"best_guess_labels": [{"label": "cartoon"}]})
    lens = AsyncMock(return_value={"visual_matches": [{"title": "Akali", "url": "https://example.org/akali"}]})
    config = ScopedConfig(serpapi_api_key="secret", serpapi_monthly_limit=1)
    async with get_session() as db:
        await perform_web_detection(db, sha, b"image", ScopedConfig(google_web_detection_enabled=True), provider=google, now=now)
        first = await perform_web_detection(db, sha, b"image", config, provider=lens, now=now)
        cached = await perform_web_detection(db, sha, b"image", config, provider=lens, now=now)
        assert first.cached is False
        assert cached.cached is True
        assert cached.result == lens.return_value
        with pytest.raises(WebDetectionQuotaExceeded):
            await perform_web_detection(db, uuid.uuid4().hex.ljust(64, "0"), b"image", config, provider=lens, now=now)
        serpapi_usage = await db.get(SerpApiImageUsage, "2097-10")
        google_usage = await db.get(GoogleWebDetectionUsage, "2097-10")
        assert serpapi_usage is not None
        assert google_usage is not None
        assert serpapi_usage.used_count == 1
        assert google_usage.used_count == 1
    lens.assert_awaited_once()


async def test_serpapi_tool_binds_quoted_image_and_persists_character_evidence(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import serpapi_search as provider
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.web_image_search import create_last_image_search_tool, create_reverse_image_search_tool

    original = image_bytes()
    (tmp_path / "character.png").write_bytes(original)
    fake = AsyncMock(return_value={"provider": "serpapi_google_lens", "visual_matches": [{"title": "Akali", "url": "https://example.org/akali"}]})
    monkeypatch.setattr(provider, "request_serpapi_lens", fake)
    group = uuid.uuid4().hex
    async with get_session() as db:
        media = MediaStorage(file_hash=uuid.uuid4().hex, file_path="character.png", description="character")
        db.add(media)
        await db.flush()
        db.add(ChatHistory(session_id=group, user_id="other", content_type="image", content="id: quoted-image\ncharacter.png", user_name="Other", media_id=media.media_id))
        await db.commit()
        search = create_reverse_image_search_tool(db, group, None, "alice", reply_to_id="quoted-image", pic_dir=tmp_path, config=ScopedConfig(serpapi_api_key="secret"))
        result = json.loads(await search.ainvoke({}))
        assert result["ok"] is True
        assert result["data"]["source_platform_message_ids"] == ["quoted-image"]
        assert result["data"]["provider"] == "serpapi_google_lens"
        assert "Akali" in json.dumps(result)
        fake.assert_awaited_once()
        assert fake.call_args.args[0][0] == original
        last = json.loads(await create_last_image_search_tool(db, group, "alice").ainvoke({}))
        assert last["data"]["search_result"] == result


def test_serpapi_key_is_masked_in_runtime_settings():
    from nonebot_plugin_ai_groupmate.runtime_config import SECRET_FIELDS, CONFIGURABLE_FIELDS
    assert "serpapi_api_key" in SECRET_FIELDS
    assert "serpapi_api_key" in CONFIGURABLE_FIELDS
