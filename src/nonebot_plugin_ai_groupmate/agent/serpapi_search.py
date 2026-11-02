"""SerpApi Google Lens: upload cached image bytes, return bounded source evidence."""

from __future__ import annotations

import re
import asyncio
import logging
from io import BytesIO
from urllib.parse import urlsplit, parse_qsl
from collections.abc import Mapping

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import ScopedConfig
from .web_image_search import WebDetectionError, _safe_web_url, _clean_page_title

IMAGE_UPLOAD_URL = "https://serpapi.com/image"
LENS_SEARCH_URL = "https://serpapi.com/search.json"
MAX_UPLOAD_BYTES = 500_000


class _RedactSerpApiKey(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "serpapi.com" in message:
            record.msg = re.sub(r"(?i)(api_key=)[^&\s\"']+", r"\1[REDACTED]", message)
            record.args = ()
        return True


# HTTPX logs complete GET URLs at INFO; Lens requires its key in the query.
logging.getLogger("httpx").addFilter(_RedactSerpApiKey())


def prepare_serpapi_image(image_bytes: bytes) -> tuple[bytes, str, str]:
    """Preserve small supported originals; compress other images below 500 KB."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("image dimensions too large")
            image.seek(0)
            image.load()
            formats = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"), "WEBP": ("webp", "image/webp")}
            if image.format in formats and len(image_bytes) <= MAX_UPLOAD_BYTES:
                extension, mime = formats[image.format]
                return image_bytes, extension, mime
            oriented = ImageOps.exif_transpose(image).convert("RGBA")
            flattened = Image.new("RGB", oriented.size, "white")
            flattened.paste(oriented, mask=oriented.getchannel("A"))
            flattened.thumbnail((2048, 2048))
            for _ in range(6):
                for quality in (90, 80, 70, 60):
                    output = BytesIO()
                    flattened.save(output, format="JPEG", quality=quality, optimize=True)
                    data = output.getvalue()
                    if len(data) <= MAX_UPLOAD_BYTES:
                        return data, "jpg", "image/jpeg"
                flattened = flattened.resize((max(1, flattened.width * 3 // 4), max(1, flattened.height * 3 // 4)))
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        pass
    raise WebDetectionError("source_image_invalid", "图片无法解码或压缩，请换一张有效图片。")


def normalize_lens_response(body: Mapping, *, max_results: int = 5, api_key: str = "") -> dict[str, object]:
    """Keep names, descriptions and original source links; discard API metadata."""
    limit = max(1, min(max_results, 10))

    def clean_text(value: object) -> str:
        text = _clean_page_title(value) if isinstance(value, str) else ""
        return text.replace(api_key, "[REDACTED]") if api_key else text

    def clean_url(value: object) -> str | None:
        url = _safe_web_url(value)
        if not url:
            return None
        parsed = urlsplit(url)
        if parsed.hostname == "serpapi.com" or (parsed.hostname or "").endswith(".serpapi.com"):
            return None
        if any(key.lower() in {"api_key", "access_token", "token"} for key, _ in parse_qsl(parsed.query)):
            return None
        return None if api_key and api_key in url else url

    result: dict[str, object] = {"provider": "serpapi_google_lens"}
    for key in ("exact_matches", "visual_matches", "organic_results"):
        if key not in body:
            continue
        entries = body.get(key)
        normalized = []
        seen = set()
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, Mapping):
                continue
            url = clean_url(entry.get("link"))
            title = clean_text(entry.get("title"))
            if not url or not title or url in seen:
                continue
            seen.add(url)
            item = {"title": title, "url": url}
            for field in ("source", "snippet"):
                text = clean_text(entry.get(field))
                if text:
                    item[field] = text
            normalized.append(item)
            if len(normalized) >= limit:
                break
        result[key] = normalized
    return result


def _response_body(response: httpx.Response, *, upload: bool = False) -> Mapping:
    # Never expose provider error text or a request URL containing the key.
    if response.status_code in {401, 403}:
        raise WebDetectionError("serpapi_auth_failed", "SerpApi 密钥无效或权限不足，请管理员检查配置。")
    if response.status_code == 429:
        raise WebDetectionError("serpapi_rate_limited", "SerpApi 配额或请求速率受限，请稍后重试。", retryable=True)
    if response.status_code >= 400:
        raise WebDetectionError("serpapi_request_failed", "SerpApi 搜图请求失败。", retryable=response.status_code >= 500)
    try:
        body = response.json()
    except ValueError:
        raise WebDetectionError("invalid_provider_response", "SerpApi 返回了无法解析的结果。") from None
    if not isinstance(body, Mapping):
        raise WebDetectionError("invalid_provider_response", "SerpApi 返回了无法解析的结果。")
    metadata = body.get("search_metadata")
    status = metadata.get("status") if isinstance(metadata, Mapping) else None
    if body.get("error") or (not upload and status != "Success"):
        raise WebDetectionError("serpapi_search_failed", "SerpApi 未能完成搜图，请稍后重试。", retryable=True)
    return body


async def request_serpapi_lens(
    prepared: tuple[bytes, str, str],
    config: ScopedConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, object]:
    key = config.serpapi_api_key.strip()
    if not key:
        raise WebDetectionError("not_configured", "尚未配置 SerpApi 密钥。")

    async def search(http: httpx.AsyncClient) -> dict[str, object]:
        data, extension, mime = prepared
        uploaded = _response_body(await http.post(
            IMAGE_UPLOAD_URL,
            data={"api_key": key},
            files={"image": (f"image.{extension}", data, mime)},
        ), upload=True)
        image_id = uploaded.get("image_id")
        if not isinstance(image_id, str) or not image_id or len(image_id) > 2048:
            raise WebDetectionError("invalid_provider_response", "SerpApi 未返回有效的图片上传标识。")
        body = _response_body(await http.get(LENS_SEARCH_URL, params={
            "engine": "google_lens", "image_id": image_id, "type": "all", "hl": "en", "api_key": key,
        }))
        return normalize_lens_response(body, max_results=config.serpapi_max_results, api_key=key)

    async def run() -> dict[str, object]:
        if client is not None:
            return await search(client)
        async with httpx.AsyncClient(timeout=config.serpapi_timeout_seconds, follow_redirects=False) as http:
            return await search(http)

    try:
        return await asyncio.wait_for(run(), timeout=config.serpapi_timeout_seconds)
    except (httpx.HTTPError, asyncio.TimeoutError):
        raise WebDetectionError("serpapi_unavailable", "SerpApi 搜图连接失败或超时，请稍后重试。", retryable=True) from None
