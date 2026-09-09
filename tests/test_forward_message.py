import json
from typing import Any, cast

import pytest
from nonebot.adapters import Bot


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
async def test_forward_tool_returns_image_and_text_to_model():
    from nonebot_plugin_ai_groupmate.agent.graph import _normalize_tool_result, _build_extra_content_message
    from nonebot_plugin_ai_groupmate.agent.forward_tools import create_read_forward_message_tool

    url = "https://example.com/forward.png"
    bot = _FakeBot({"mixed": {"messages": [{"message": [
        {"type": "image", "data": {"url": url}},
        {"type": "text", "data": {"text": "图片后面的文字"}},
    ]}]}})
    reader = create_read_forward_message_tool(cast(Bot, bot), {"mixed"})
    result = await reader.ainvoke({"forward_id": "mixed"})
    text, blocks = _normalize_tool_result(result)

    assert "图片后面的文字" in text
    assert blocks is not None
    message = await _build_extra_content_message(blocks, supports_images=True, image_summarizer=None)
    assert isinstance(message.content, list)
    assert {"type": "image_url", "image_url": {"url": url}} in message.content

    async def summarize(content):
        assert {"type": "image_url", "image_url": {"url": url}} in content
        return "图片中的文字"

    fallback = await _build_extra_content_message(blocks, supports_images=False, image_summarizer=summarize)
    assert "图片中的文字" in fallback.content


@pytest.mark.asyncio
async def test_forward_images_nested_limit_and_unavailable():
    from nonebot_plugin_ai_groupmate.agent.graph import _normalize_tool_result
    from nonebot_plugin_ai_groupmate.agent.forward_tools import create_read_forward_message_tool

    bot = _FakeBot({
        "outer": {"messages": [{"message": [
            {"type": "text", "data": {"text": "外层文字"}},
            {"type": "forward", "data": {"id": "inner"}},
        ]}]},
        "inner": {"messages": [{"message": [
            {"type": "image", "data": {}},
            *[{"type": "image", "data": {"url": f"https://example.com/{i}.png"}} for i in range(4)],
            {"type": "text", "data": {"text": "内层文字"}},
        ]}]},
    })
    result = await create_read_forward_message_tool(cast(Bot, bot), {"outer"}).ainvoke({"forward_id": "outer"})
    text, blocks = _normalize_tool_result(result)
    assert "外层文字" in text
    assert "内层文字" in text
    assert "转发图片 1，无法读取" in text
    assert "超过本次读取数量上限" in text
    assert blocks is not None
    assert len([block for block in blocks if isinstance(block, dict) and block.get("type") == "image_url"]) == 2


@pytest.mark.asyncio
async def test_forward_image_resolves_file_id_and_base64():
    import io
    import base64

    from PIL import Image

    from nonebot_plugin_ai_groupmate.agent.forward_tools import _forward_image_source

    class ImageBot:
        async def call_api(self, api, **data):
            assert api == "get_image"
            assert data == {"file": "image-id"}
            return {"data": {"url": "https://example.com/resolved.png"}}

    bot = cast(Bot, ImageBot())
    assert await _forward_image_source(bot, {"file": "image-id"}) == "https://example.com/resolved.png"
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    for prefix in ("base64://", "data:image/png;base64,"):
        result = await _forward_image_source(bot, {"file": prefix + encoded})
        assert result is not None
        assert result.startswith("data:image/")
    assert await _forward_image_source(bot, {"file": "base64://invalid"}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("image_first", [True, False])
async def test_agent_reads_text_in_cq_forward_with_large_image(image_first):
    from nonebot_plugin_ai_groupmate.agent.forward_tools import (
        create_read_forward_message_tool,
    )

    image = "[CQ:image,file=base64://" + "A" * 13_000 + "]"
    text = "图旁边的重要文字"
    content = image + text if image_first else text + image
    bot = _FakeBot({
        "mixed": {"messages": [
            {"sender": {"nickname": "Alice"}, "message": content},
            {"sender": {"nickname": "Bob"}, "message": "下一条文字"},
        ]},
    })
    reader = create_read_forward_message_tool(cast(Bot, bot), {"mixed"})

    result = json.loads(await reader.ainvoke({"forward_id": "mixed"}))
    body = result["data"]["content"]

    assert text in body
    assert "Bob: 下一条文字" in body
    marker = "[转发图片 1，无法读取]"
    assert (marker + text if image_first else text + marker) in body
    assert "base64://" not in body


@pytest.mark.asyncio
async def test_cq_forward_preserves_escaped_text_nested_ids_and_media():
    from nonebot_plugin_ai_groupmate.media_message import LazyMediaRegistry
    from nonebot_plugin_ai_groupmate.forward_message import expand_forward_message

    registry = LazyMediaRegistry({})
    bot = _FakeBot({
        "outer": {"messages": [{"message": (
            "&#91;CQ:image,file=literal&#93; &amp; "
            "[CQ:image,file=a.jpg]图片后文字"
            "[CQ:record,url=https://example.com/a?x=1&#44;2&amp;y=3]"
            "[CQ:forward,id=inner]"
        )}]},
        "inner": {"messages": [{"message": "内层文字"}]},
    })

    result = await expand_forward_message(
        cast(Bot, bot), register_media=registry.register_forwarded, forward_id="outer",
    )

    assert "[CQ:image,file=literal] & [图片]图片后文字" in result
    assert "内层文字" in result
    assert bot.calls == ["outer", "inner"]
    assert registry.forwarded_source("forward-media-1", "audio") == {
        "url": "https://example.com/a?x=1,2&y=3",
    }


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
