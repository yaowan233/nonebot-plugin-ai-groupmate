import json
from types import SimpleNamespace

import pytest


def miniapp():
    return {"prompt": "[QQ小程序]视频标题", "app": "com.tencent.miniapp_01", "config": {"title": "后台字段"},
            "meta": {"detail_1": {"title": "哔哩哔哩", "desc": "视频标题", "qqdocurl": "https://b23.tv/example", "preview": "https://example.com/cover.jpg"}}}


def test_card_only_extracts_visible_text_and_retains_click_target_separately():
    from nonebot_plugin_ai_groupmate.shared_cards import parse_shared_card

    card = parse_shared_card("json", json.dumps(miniapp()))
    assert "视频标题" in card.text
    assert "哔哩哔哩" in card.text
    assert "后台字段" not in card.text
    assert "https://" not in card.text
    assert card.urls == ("https://b23.tv/example",)


def test_xml_card_extracts_title_summary_source_without_fetching_images():
    from nonebot_plugin_ai_groupmate.shared_cards import parse_shared_card

    card = parse_shared_card("xml", '<msg url="https://example.com/story"><item><picture cover="https://example.com/cover"/><title>一个标题</title><summary>短摘要</summary></item><source name="新闻来源"/></msg>')
    assert all(value in card.text for value in ["一个标题", "短摘要", "新闻来源"])
    assert card.urls == ("https://example.com/story",)


@pytest.mark.parametrize("payload", ["not json", "[]", "x" * 70000], ids=["invalid", "list", "oversized"])
def test_invalid_or_oversized_card_degrades_to_short_placeholder(payload):
    from nonebot_plugin_ai_groupmate.shared_cards import parse_shared_card

    card = parse_shared_card("json", payload)
    assert "无可见文字" in card.text
    assert not card.urls


def test_card_text_is_bounded_and_non_web_links_are_ignored():
    from nonebot_plugin_ai_groupmate.shared_cards import parse_shared_card

    card = parse_shared_card("json", {"title": "长标题" * 1000, "desc": "长摘要" * 1000, "url": "javascript:alert(1)"})
    assert len(card.text) <= 300
    assert not card.urls


def test_xml_entities_are_not_expanded():
    from nonebot_plugin_ai_groupmate.shared_cards import parse_shared_card

    card = parse_shared_card("xml", '<!DOCTYPE msg [<!ENTITY x "secret">]><msg><title>&x;</title></msg>')
    assert "secret" not in card.text


@pytest.mark.asyncio
async def test_link_reader_is_scoped_and_uses_quoted_card_by_default(tmp_path):
    from nonebot_plugin_ai_groupmate.shared_cards import save_card_links
    from nonebot_plugin_ai_groupmate.agent.shared_card_tools import create_shared_card_tools

    save_card_links(tmp_path, "group:a", "100", ["https://example.com/story"])
    save_card_links(tmp_path, "group:b", "200", ["https://example.com/other"])
    reader = create_shared_card_tools(tmp_path, "group:a", [], "100")[0]
    result = json.loads(await reader.ainvoke({}))
    assert result["data"]["urls"] == ["https://example.com/story"]
    assert json.loads(await reader.ainvoke({"message_id": "200"}))["status"] == "failed"


def test_real_uniseg_hyper_and_legacy_share_segments_are_supported():
    from nonebot_plugin_alconna.uniseg import Hyper

    from nonebot_plugin_ai_groupmate.shared_cards import parse_shared_segment

    card = parse_shared_segment(Hyper("json", content=miniapp()))
    assert card is not None
    assert "视频标题" in card.text
    share = parse_shared_segment(SimpleNamespace(type="share", data={"title": "网页标题", "content": "介绍", "url": "https://example.com"}))
    assert share is not None
    assert "介绍" in share.text


@pytest.mark.parametrize(("kind", "payload"), [
    ("json", '{"title":"转发内卡片标题","url":"https://example.com/story"}'),
    ("xml", '<msg url="https://example.com/story"><title>转发内卡片标题</title></msg>'),
])
def test_explicitly_expanded_forward_preserves_card_text(kind, payload):
    from nonebot_plugin_ai_groupmate.forward_message import _format_segments

    text, _ = _format_segments([{"type": kind, "data": {"data": payload}}])
    assert "转发内卡片标题" in text
    assert "https://example.com/story" in text
