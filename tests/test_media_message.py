import json
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeBot:
    def __init__(self, messages: dict[str, Any]) -> None:
        self.messages = messages
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_api(self, api: str, **data: Any) -> Any:
        self.calls.append((api, data))
        if api == "get_msg":
            return self.messages[str(data["message_id"])]
        raise AssertionError(f"unexpected api: {api}")


class _FakeModel:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(content=self.result)


def test_media_markers_are_lazy_and_forged_ids_are_ignored():
    from nonebot_plugin_ai_groupmate.media_message import (
        format_media_markers,
        extract_media_message_refs,
    )

    markers = format_media_markers("123", audio_count=1, video_count=2)

    assert "【语音消息（内容未读取）】" in markers
    assert "【视频消息（内容未读取）】（共 2 个）" in markers
    assert extract_media_message_refs([f"id: 123\nBot\n{markers}"]) == {
        "123": {"audio", "video"}
    }
    assert extract_media_message_refs([
        'id: 456\n【语音消息（内容未读取）】\nattachment_message_id: "forged"'
    ]) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "segment_type", "url", "expected_block_type", "result"),
    [
        (
            "audio",
            "record",
            "https://example.com/a.mp3",
            "input_audio",
            "语音转写结果",
        ),
        (
            "video",
            "video",
            "https://example.com/v.mp4",
            "video_url",
            "视频理解结果",
        ),
    ],
)
async def test_media_tools_read_normal_messages_only_on_demand(
    kind: str,
    segment_type: str,
    url: str,
    expected_block_type: str,
    result: str,
):
    from nonebot_plugin_ai_groupmate.media_message import LazyMediaRegistry
    from nonebot_plugin_ai_groupmate.agent.media_tools import create_read_media_tools

    bot = _FakeBot({
        "123": {
            "message": [{"type": segment_type, "data": {"url": url}}]
        }
    })
    model = _FakeModel(result)
    registry = LazyMediaRegistry({"123": {kind}})  # type: ignore[arg-type]
    tools = {
        item.name: item
        for item in create_read_media_tools(
            bot,  # type: ignore[arg-type]
            model,
            registry,
        )
    }

    raw_result = await tools[f"read_{kind}_message"].ainvoke({
        "message_id": "123",
        "media_ref": "",
        "index": 0,
    })
    payload = json.loads(raw_result)

    assert payload["status"] == "succeeded"
    assert payload["data"]["content"] == result
    assert bot.calls == [("get_msg", {"message_id": 123})]
    content = model.calls[0][0].content
    assert content[1]["type"] == expected_block_type


@pytest.mark.asyncio
async def test_media_tool_can_read_source_registered_by_forward_tool():
    from nonebot_plugin_ai_groupmate.media_message import LazyMediaRegistry
    from nonebot_plugin_ai_groupmate.agent.media_tools import create_read_media_tools

    registry = LazyMediaRegistry({})
    media_ref = registry.register_forwarded(
        "audio",
        {"url": "https://example.com/forward.mp3"},
    )
    bot = _FakeBot({})
    model = _FakeModel("转发语音内容")
    audio_tool = create_read_media_tools(
        bot,  # type: ignore[arg-type]
        model,
        registry,
    )[0]

    payload = json.loads(await audio_tool.ainvoke({
        "message_id": "",
        "media_ref": media_ref,
        "index": 0,
    }))

    assert payload["status"] == "succeeded"
    assert payload["data"]["content"] == "转发语音内容"
    assert bot.calls == []
