import io
import json
import wave
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


def streamed_wav():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 2400)
    data = bytearray(out.getvalue())
    struct.pack_into("<I", data, 4, 0xffffffff)
    struct.pack_into("<I", data, 40, 0xffffffff)
    return bytes(data)


def test_streamed_wav_length_is_repaired_and_truncated_wav_rejected():
    from nonebot_plugin_ai_groupmate.agent.tts_tools import finalize_streamed_wav

    fixed = finalize_streamed_wav(streamed_wav())
    with wave.open(io.BytesIO(fixed), "rb") as wav:
        assert wav.getnframes() == 2400
        assert len(wav.readframes(2400)) == 4800
    with pytest.raises(ValueError, match="incomplete WAV"):
        finalize_streamed_wav(fixed[:-2])
    with pytest.raises(ValueError, match="invalid WAV"):
        finalize_streamed_wav(b"not a wav")


async def test_synthesis_uses_official_endpoint_and_repairs_wav(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import tts_tools

    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=streamed_wav())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(tts_tools.httpx, "AsyncClient", lambda **_: client)
    audio = await tts_tools.synthesize_speech("你好", "fake-key", "stella")
    assert str(calls[0].url) == tts_tools.TTS_URL
    assert calls[0].headers["authorization"] == "Bearer fake-key"
    assert json.loads(calls[0].content) == {"text":"你好","voice_id":"stella","format":"wav","filter_markdown":True}
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnframes() == 2400


def test_credentials_only_use_deepseek_keys():
    from nonebot_plugin_ai_groupmate.agent.tts_tools import deepseek_tts_key

    def config(**overrides):
        return SimpleNamespace(**({"deepseek_tts_api_key":"","deepseek_search_api_key":"", "chat_base_url":"",
                                   "llm_base_url":"https://other.example/v1","chat_api_key":"other", "llm_api_key":""} | overrides))
    cfg = config()
    assert deepseek_tts_key(cfg, cfg) == ""
    ds = config(chat_base_url="https://api.deepseek.com/v1", chat_api_key="ds-key")
    assert deepseek_tts_key(cfg, ds) == "ds-key"
    assert deepseek_tts_key(config(deepseek_tts_api_key="tts-key"), ds) == "tts-key"
    mixed = config(chat_base_url="https://api.deepseek.com/v1", chat_api_key="", llm_api_key="other-key")
    assert deepseek_tts_key(mixed, mixed) == ""


async def test_voice_tool_sends_native_voice_once_and_records_spoken_text(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import tts_tools
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    cfg = ScopedConfig(deepseek_tts_api_key="test", deepseek_tts_voice="stella")
    synthesis = AsyncMock(return_value=streamed_wav())
    monkeypatch.setattr(tts_tools, "synthesize_speech", synthesis)
    sent = []
    async def deliver(db, session_id, message, **kwargs):
        sent.append(message)
        records = kwargs["record"]("platform-id")
        assert records[0].content == "id: platform-id\n[语音] 你好呀"
        return SimpleNamespace(note="", message_id="platform-id")
    monkeypatch.setattr(tts_tools, "send_and_finalize", deliver)
    voice = tts_tools.create_voice_tool(SimpleNamespace(commit=AsyncMock()), "group", None, config=cfg, chat_config=cfg,
                                        send_target=None, bot_name="miyuki")
    result = json.loads(await voice.ainvoke({"text":"你好呀"}))
    assert result["delivery_state"] == "completed"
    assert len(sent) == 1
    assert sent[0][0].type == "voice"
    assert synthesis.await_args is not None
    assert synthesis.await_args.args[2] == "stella"
    again = json.loads(await voice.ainvoke({"text":"你好呀"}))
    assert again["status"] == "skipped"


async def test_voice_failure_never_claims_delivery(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import tts_tools
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    cfg = ScopedConfig(deepseek_tts_api_key="test")
    monkeypatch.setattr(tts_tools, "synthesize_speech", AsyncMock(side_effect=httpx.ConnectError("failed")))
    deliver = AsyncMock()
    monkeypatch.setattr(tts_tools, "send_and_finalize", deliver)
    voice = tts_tools.create_voice_tool(SimpleNamespace(commit=AsyncMock()), "group", None, config=cfg, chat_config=cfg,
                                        send_target=None, bot_name="miyuki")
    result = json.loads(await voice.ainvoke({"text":"你好呀"}))
    assert result["delivery_state"] == "not_attempted"
    deliver.assert_not_awaited()

async def test_voice_is_exported_as_onebot_record(monkeypatch):
    from nonebot_plugin_alconna import UniMessage
    from nonebot_plugin_alconna.uniseg.adapters.onebot11.exporter import Onebot11MessageExporter

    voice = UniMessage.voice(raw=streamed_wav(), mimetype="audio/wav")[0]
    segment = await Onebot11MessageExporter().media(voice, None)
    assert segment.type == "record"
    assert segment.data["file"].startswith("base64://")


async def test_voice_delivery_graph_counts_message_and_stops(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from nonebot_plugin_ai_groupmate.agent import tts_tools
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, build_chat_graph

    cfg = ScopedConfig(deepseek_tts_api_key="test")
    monkeypatch.setattr(tts_tools, "synthesize_speech", AsyncMock(return_value=streamed_wav()))
    send = AsyncMock(return_value=SimpleNamespace(note="", message_id="platform-id"))
    monkeypatch.setattr(tts_tools, "send_and_finalize", send)
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    voice = tts_tools.create_voice_tool(db, "group", None, config=cfg, chat_config=cfg, send_target=None, bot_name="miyuki")
    class Model:
        calls = 0
        def bind_tools(self, _tools):
            return self
        async def ainvoke(self, _messages, **_kwargs):
            self.calls += 1
            return AIMessage(content="", tool_calls=[{"id": "voice-1", "name": "send_voice_message", "args": {"text": "你好呀", "next_step": "end"}}])
    model = Model()
    graph = build_chat_graph(model, [voice], "按请求发送语音", db_session=db, limits=AgentRunLimits())
    result = await graph.ainvoke({"messages": [HumanMessage(content="用语音说你好")], "session_id": "group", "request_id": None})
    assert send.await_count == 1
    assert result["reply_count"] == 1
    assert model.calls == 1


async def test_delivery_failure_is_unknown_and_not_repeated(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import tts_tools
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    cfg = ScopedConfig(deepseek_tts_api_key="test")
    monkeypatch.setattr(tts_tools, "synthesize_speech", AsyncMock(return_value=streamed_wav()))
    send = AsyncMock(side_effect=RuntimeError("unknown"))
    monkeypatch.setattr(tts_tools, "send_and_finalize", send)
    voice = tts_tools.create_voice_tool(SimpleNamespace(commit=AsyncMock()), "group", None, config=cfg, chat_config=cfg,
                                        send_target=None, bot_name="miyuki")
    assert json.loads(await voice.ainvoke({"text": "你好"}))["delivery_state"] == "unknown"
    assert json.loads(await voice.ainvoke({"text": "你好"}))["status"] == "skipped"
    assert send.await_count == 1
