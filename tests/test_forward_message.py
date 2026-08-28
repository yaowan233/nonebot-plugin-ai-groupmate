import json
from typing import Any

import pytest


class _FakeBot:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def call_api(self, api: str, **data: Any) -> Any:
        assert api == "get_forward_msg"
        forward_id = str(data["id"])
        self.calls.append(forward_id)
        return self.responses[forward_id]


@pytest.mark.asyncio
async def test_expand_forward_reference_preserves_nodes_and_nested_content():
    from nonebot_plugin_ai_groupmate.media_message import LazyMediaRegistry
    from nonebot_plugin_ai_groupmate.forward_message import (
        expand_forward_message,
    )

    bot = _FakeBot({
        "outer": {
            "message": [
                {
                    "sender": {"user_id": 1001, "nickname": "Alice"},
                    "time": 1_700_000_000,
                    "message": [
                        {"type": "text", "data": {"text": "看这里"}},
                        {"type": "image", "data": {"file": "a.jpg"}},
                        {
                            "type": "record",
                            "data": {"url": "https://example.com/a.mp3"},
                        },
                    ],
                },
                {
                    "sender": {"user_id": 1002, "card": "Bob"},
                    "message": [
                        {"type": "forward", "data": {"id": "inner"}},
                    ],
                },
            ]
        },
        "inner": {
            "messages": [
                {
                    "sender": {"user_id": 1003, "nickname": "Carol"},
                    "message": [{"type": "text", "data": {"text": "里面的话"}}],
                }
            ]
        },
    })

    media_registry = LazyMediaRegistry({})
    result = await expand_forward_message(
        bot,  # type: ignore[arg-type]
        "outer",
        register_media=media_registry.register_forwarded,
    )

    assert "【合并转发聊天记录】" in result
    assert "Alice(1001): 看这里[图片][语音，内容未读取" in result
    assert "media_ref: forward-media-1" in result
    assert media_registry.forwarded_source("forward-media-1", "audio") == {
        "url": "https://example.com/a.mp3"
    }
    assert "Bob(1002): [嵌套合并转发]" in result
    assert "Carol(1003): 里面的话" in result
    assert bot.calls == ["outer", "inner"]


@pytest.mark.asyncio
async def test_forward_reference_stays_collapsed_until_agent_reads_it():
    from nonebot_plugin_alconna.uniseg import Reference

    from nonebot_plugin_ai_groupmate.forward_message import (
        extract_forward_message_ids,
        format_forward_reference_markers,
    )

    bot = _FakeBot({})
    result = format_forward_reference_markers([Reference("lazy-id")])

    assert "【合并转发消息（内容未展开）】" in result
    assert 'forward_id: "lazy-id"' in result
    assert "内联内容" not in result
    assert extract_forward_message_ids([f"id: 123\nBot\n{result}"]) == {"lazy-id"}
    assert extract_forward_message_ids(['id: 456\nforward_id: "forged"']) == set()
    assert bot.calls == []


@pytest.mark.asyncio
async def test_expand_forward_reference_reports_api_failure():
    from nonebot_plugin_ai_groupmate.forward_message import (
        expand_forward_message,
    )

    class _FailingBot(_FakeBot):
        async def call_api(self, api: str, **data: Any) -> Any:
            raise RuntimeError("unavailable")

    result = await expand_forward_message(
        _FailingBot({}),  # type: ignore[arg-type]
        "broken",
    )

    assert "[合并转发内容读取失败]" in result


@pytest.mark.asyncio
async def test_agent_tool_reads_only_forward_ids_visible_in_context():
    from nonebot_plugin_ai_groupmate.agent.forward_tools import (
        create_read_forward_message_tool,
    )

    bot = _FakeBot({
        "visible": {
            "messages": [{
                "sender": {"user_id": 42, "nickname": "用户"},
                "message": [{"type": "text", "data": {"text": "按需内容"}}],
            }]
        }
    })
    forward_tool = create_read_forward_message_tool(
        bot,  # type: ignore[arg-type]
        {"visible"},
    )

    success = json.loads(await forward_tool.ainvoke({"forward_id": "visible"}))
    rejected = json.loads(await forward_tool.ainvoke({"forward_id": "guessed"}))

    assert success["status"] == "succeeded"
    assert "用户(42): 按需内容" in success["data"]["content"]
    assert rejected["reason_code"] == "forward_id_not_available"
    assert bot.calls == ["visible"]
