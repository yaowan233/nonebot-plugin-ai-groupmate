"""Bounded GPT Images transport; credentials and provider bodies never escape errors."""

import io
import json
import base64
import asyncio
import binascii
from typing import TYPE_CHECKING
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from ..config import ScopedConfig

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


class ImageGenerationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    usage: dict[str, int]
    output_format: str = "png"


def resolve_image_credentials(config: "ScopedConfig") -> tuple[str, str]:
    """Never pair a dedicated provider address with an unrelated fallback key."""
    dedicated_key = config.image_generation_api_key.strip()
    dedicated_base = config.image_generation_base_url.strip()
    main_base = (config.chat_base_url or config.llm_base_url).strip()
    if dedicated_key:
        key, base = dedicated_key, dedicated_base or "https://api.openai.com/v1"
    elif config.chat_api_format == "openai" and (not dedicated_base or dedicated_base.rstrip("/") == main_base.rstrip("/")):
        key, base = (config.chat_api_key or config.llm_api_key).strip(), dedicated_base or main_base
    else:
        raise ImageGenerationError("missing_image_api_key", "请为绘图服务配置匹配的 API Key 和地址。")
    parsed = urlsplit(base)
    if not key:
        raise ImageGenerationError("missing_image_api_key", "绘图服务没有配置 API Key。")
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ImageGenerationError("invalid_image_api_url", "绘图 API 地址必须是无账号、查询参数或片段的 HTTPS 地址。")
    return key, base.rstrip("/")


def prepare_reference(data: bytes) -> tuple[bytes, str, str]:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageGenerationError("invalid_reference", "参考图为空或超过 10 MiB。")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageGenerationError("invalid_reference", "参考图尺寸过大。")
            animated = getattr(image, "n_frames", 1) > 1
            image.verify()
            image_format = image.format
        if image_format in {"PNG", "JPEG", "WEBP"} and not animated:
            suffix = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[image_format]
            return data, suffix, Image.MIME[image_format]
        # Animated formats contribute their first frame only.
        with Image.open(io.BytesIO(data)) as image:
            output = io.BytesIO()
            image.convert("RGBA").save(output, format="PNG")
            converted = output.getvalue()
        if len(converted) > MAX_IMAGE_BYTES:
            raise ImageGenerationError("invalid_reference", "转换后的参考图超过 10 MiB。")
        return converted, "png", "image/png"
    except ImageGenerationError:
        raise
    except Exception:
        raise ImageGenerationError("invalid_reference", "参考文件不是可读取的图片。") from None


def prepare_markup(data: bytes, first_reference: bytes) -> tuple[bytes, str, str]:
    """Validate the separate transparent annotation layer before a paid request."""
    try:
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("markup size")
        with Image.open(io.BytesIO(first_reference)) as original, Image.open(io.BytesIO(data)) as layer:
            if (layer.format != "PNG" or getattr(layer, "n_frames", 1) != 1
                    or layer.width * layer.height > 16_777_216):
                raise ValueError("markup format")
            oriented_original = ImageOps.exif_transpose(original)
            oriented_layer = ImageOps.exif_transpose(layer)
            if (oriented_original.width * oriented_original.height > 16_777_216
                    or oriented_original.size != oriented_layer.size
                    or oriented_layer.convert("RGBA").getchannel("A").getextrema()[0] == 255):
                raise ValueError("markup dimensions or transparency")
    except Exception:
        raise ImageGenerationError("invalid_markup", "标注层需要与原图同尺寸，并且是含透明区域的独立 PNG。") from None
    return data, "png", "image/png"


def decode_generated_image(payload: object, output_format: str = "png") -> GeneratedImage:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or len(payload["data"]) != 1:
        raise ImageGenerationError("invalid_image_response", "绘图服务没有返回一张有效图片。")
    item = payload["data"][0]
    encoded = item.get("b64_json") if isinstance(item, dict) else None
    if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ImageGenerationError("invalid_image_response", "绘图服务返回的图片为空、过大或格式不受支持。")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ImageGenerationError("invalid_image_response", "绘图服务返回的图片编码无效。") from None
    if output_format == "png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ImageGenerationError("invalid_image_response", "绘图服务没有返回请求的 PNG 图片。")
        prepare_reference(data)
    elif output_format == "gif":
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "GIF" or not 2 <= getattr(image, "n_frames", 1) <= 16 or image.width * image.height * image.n_frames > MAX_IMAGE_PIXELS:
                    raise ValueError("invalid GIF")
                for frame in range(image.n_frames):
                    image.seek(frame)
                    image.load()
        except Exception:
            raise ImageGenerationError("invalid_image_response", "绘图服务没有返回有效的动态 GIF。") from None
    else:
        raise ImageGenerationError("invalid_image_response", "不支持的图片格式。")
    raw_usage = payload.get("usage")
    usage = {key: value for key, value in raw_usage.items() if key in {"input_tokens", "output_tokens", "total_tokens"} and isinstance(value, int) and not isinstance(value, bool) and value >= 0} if isinstance(raw_usage, dict) else {}
    return GeneratedImage(data, usage, output_format)


async def request_image(config: "ScopedConfig", prompt: str, references: list[tuple[bytes, str, str]], *, size: str, transparent_background: bool,
                        complexity: str | None = None, output_format: str = "png", frame_count: int | None = None, frame_duration_ms: int | None = None,
                        markup: tuple[bytes, str, str] | None = None) -> GeneratedImage:
    if (complexity not in {None, "low", "medium", "high"} or output_format not in {"png", "gif"}
            or (output_format == "gif" and (references or transparent_background))
            or (output_format != "gif" and (frame_count is not None or frame_duration_ms is not None))
            or (frame_count is not None and (type(frame_count) is not int or not 2 <= frame_count <= 16))
            or (frame_duration_ms is not None and (type(frame_duration_ms) is not int or not 20 <= frame_duration_ms <= 2000))
            or (markup is not None and (not references or output_format != "png"))
            or ((complexity is not None or output_format == "gif" or markup is not None) and not config.image_generation_model.startswith("gpt-image-2.6-"))):
        raise ImageGenerationError("invalid_image_arguments", "所选模型或参数不支持此绘图方式。")
    if markup is not None:
        markup = prepare_markup(markup[0], references[0][0])
    key, base = resolve_image_credentials(config)
    params = {
        "model": config.image_generation_model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": config.image_generation_quality,
        "output_format": output_format,
        "background": "transparent" if transparent_background else "auto",
    }
    if complexity is not None:
        params["complexity"] = complexity
    if frame_count is not None:
        params["frame_count"] = frame_count
    if frame_duration_ms is not None:
        params["frame_duration_ms"] = frame_duration_ms
    endpoint = "edits" if references else "generations"
    kwargs: dict = {"json": params}
    if references:
        kwargs = {
            "data": {key: str(value) for key, value in params.items()},
            "files": [("image[]", (f"reference-{index}.{suffix}", data, mime)) for index, (data, suffix, mime) in enumerate(references, 1)],
        }
        if markup is not None:
            kwargs["files"].append(("markup", ("markup.png", markup[0], "image/png")))
    try:
        async with httpx.AsyncClient(timeout=config.image_generation_timeout_seconds, trust_env=False, follow_redirects=False) as client:
            async with client.stream("POST", base + "/images/" + endpoint, headers={"Authorization": "Bearer " + key}, **kwargs) as response:
                if response.status_code != 200:
                    code, message = {
                        400: ("image_request_rejected", "绘图请求被拒绝，请检查描述、模型和参数；不要原样重试。"),
                        401: ("image_auth_failed", "绘图 API Key 无效。"),
                        403: ("image_access_denied", "当前 Key 没有该绘图模型的权限。"),
                        429: ("image_quota_exhausted", "绘图服务限流或额度不足。"),
                    }.get(response.status_code, ("image_provider_failed", "绘图服务暂时不可用或接口不兼容。"))
                    raise ImageGenerationError(code, message)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ImageGenerationError("image_response_too_large", "绘图响应过大，已停止读取。")
        return await asyncio.to_thread(decode_generated_image, json.loads(body), output_format)
    except ImageGenerationError:
        raise
    except (TimeoutError, httpx.TimeoutException):
        raise ImageGenerationError("image_timeout", "绘图等待超时，服务端可能已计费；请勿自动重复生成。") from None
    except Exception:
        raise ImageGenerationError("image_provider_failed", "绘图响应无法读取；请勿自动重复生成。") from None
