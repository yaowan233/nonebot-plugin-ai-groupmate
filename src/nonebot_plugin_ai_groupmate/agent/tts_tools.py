"""On-demand DeepSeek speech synthesis and native voice delivery."""

import io
import re
import wave
import struct
from typing import Literal

import httpx
from langchain.tools import tool
from nonebot_plugin_alconna import UniMessage

from ..model import ChatHistory
from .delivery import OutboundIdentity, send_and_finalize
from ..deepseek import is_deepseek_endpoint
from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from ..cross_group_privacy import hide_other_group_ids

TTS_URL = "https://api.deepseek.com/v1/tools/tts"
MAX_TEXT_CHARS = 500
MAX_AUDIO_BYTES = 12 * 1024 * 1024
_VOICE_REQUEST = re.compile(r"语音|声音|朗读|念出来|读出来|念一念|念一下|说出来|说句话|开口说|用嘴说|voice|tts", re.IGNORECASE)


def wants_voice(text: str) -> bool:
    return bool(_VOICE_REQUEST.search(text or ""))


def deepseek_tts_key(config, chat_config) -> str:
    """Do not send credentials for a different provider to DeepSeek."""
    if config.deepseek_tts_api_key:
        return config.deepseek_tts_api_key
    for candidate in (chat_config, config):
        if is_deepseek_endpoint(candidate.chat_base_url or candidate.llm_base_url):
            if candidate.chat_api_key:
                return candidate.chat_api_key
            if is_deepseek_endpoint(candidate.llm_base_url) and candidate.llm_api_key:
                return candidate.llm_api_key
    if is_deepseek_endpoint(config.llm_base_url) and config.llm_api_key:
        return config.llm_api_key
    return config.deepseek_search_api_key


def finalize_streamed_wav(audio: bytes) -> bytes:
    """Fix DeepSeek's unknown-length streaming WAV header and reject truncation."""
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("invalid WAV")
    output = bytearray(audio)
    riff_size = struct.unpack_from("<I", output, 4)[0]
    if riff_size != 0xFFFFFFFF and riff_size + 8 != len(output):
        raise ValueError("incomplete WAV")
    offset = 12
    found_data = False
    while offset + 8 <= len(output):
        tag = bytes(output[offset:offset + 4])
        size = struct.unpack_from("<I", output, offset + 4)[0]
        if tag == b"data":
            actual = len(output) - offset - 8
            if not actual or actual % 2 or (size != 0xFFFFFFFF and size != actual):
                raise ValueError("incomplete WAV audio")
            struct.pack_into("<I", output, offset + 4, actual)
            found_data = True
            break
        if size == 0xFFFFFFFF or offset + 8 + size > len(output):
            raise ValueError("invalid WAV chunk")
        offset += 8 + size + (size % 2)
    if not found_data:
        raise ValueError("missing WAV data")
    struct.pack_into("<I", output, 4, len(output) - 8)
    with wave.open(io.BytesIO(output), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise ValueError("unsupported WAV format")
    return bytes(output)


async def synthesize_speech(text: str, key: str, voice: str) -> bytes:
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
        async with client.stream("POST", TTS_URL, headers={"Authorization": f"Bearer {key}"},
                                 json={"text": text, "voice_id": voice, "format": "wav", "filter_markdown": True}) as response:
            response.raise_for_status()
            audio = bytearray()
            async for chunk in response.aiter_bytes():
                audio.extend(chunk)
                if len(audio) > MAX_AUDIO_BYTES:
                    raise ValueError("audio too large")
    return finalize_streamed_wav(bytes(audio))


def create_voice_tool(db_session, session_id: str, request_id: str | None, *, config, chat_config,
                      send_target, bot_name: str, outbound_identity: OutboundIdentity | None = None,
                      recovery_enabled: bool = True, cross_group_spaces: dict[str, list[str]] | None = None):
    attempted = False

    @tool("send_voice_message")
    async def send_voice_message(text: str, voice: Literal["stella", "tide"] | None = None,
                                 next_step: Literal["end", "continue"] = "end") -> str:
        """把要说的话合成语音并发送。text 是实际朗读内容；voice 可选 stella 女声、tide 男声，留空用默认音色。发送完成就将 next_step 设为 end。"""
        nonlocal attempted
        if attempted:
            return tool_skipped("voice_already_attempted", "本轮已尝试发送语音，请勿重复发送。", delivery_state="not_attempted")
        if request_id is not None and not await is_request_active(session_id, request_id):
            return tool_skipped("request_expired", "本轮请求已结束。", delivery_state="not_attempted")
        text = hide_other_group_ids(text.strip(), session_id, cross_group_spaces or {})
        if not text or len(text) > MAX_TEXT_CHARS:
            return tool_failure("invalid_text", f"语音内容须为 1 到 {MAX_TEXT_CHARS} 字。", delivery_state="not_attempted")
        key = deepseek_tts_key(config, chat_config)
        if not key:
            return tool_failure("tts_not_configured", "还没有配置 DeepSeek 语音密钥。", delivery_state="not_attempted")
        try:
            await db_session.commit()
            audio = await synthesize_speech(text, key, voice or config.deepseek_tts_voice)
        except httpx.HTTPStatusError as error:
            return tool_failure("tts_api_error", f"语音合成服务返回 {error.response.status_code}，这次没有发送语音。", delivery_state="not_attempted")
        except (httpx.HTTPError, ValueError, wave.Error, EOFError):
            return tool_failure("tts_synthesis_failed", "语音合成失败，这次没有发送语音。", delivery_state="not_attempted")
        if request_id is not None and not await is_request_active(session_id, request_id):
            return tool_skipped("request_expired", "本轮请求已结束，语音没有发送。", delivery_state="not_attempted")
        message = UniMessage.voice(raw=audio, mimetype="audio/wav", name="voice.wav")
        attempted = True
        def record(message_id: str):
            return [ChatHistory(session_id=session_id, user_id=bot_name, user_name=bot_name,
                                content_type="bot", content=f"id: {message_id}\n[语音] {text}")]
        try:
            receipt = await send_and_finalize(db_session, session_id, message, send_options={"target": send_target, "fallback": False},
                                              record=record, recovery_enabled=recovery_enabled, outbound_identity=outbound_identity)
        except Exception:
            return tool_failure("voice_delivery_unknown", "语音是否送达无法确认，请勿自动重发。", delivery_state="unknown")
        return tool_success("voice_sent", receipt.note or "语音已发送。", data={"message_id": receipt.message_id}, delivery_state="completed")

    return send_voice_message
