import pytest


@pytest.mark.parametrize("message", [
    "miyuki 来点绯玛丽的美图", "miyuki 来点摩卡兰的美图",
    "给张参考图", "发三张风景图片", "给我一张流程图", "图呢",
])
def test_generic_image_message_does_not_require_extra_local_memes(message):
    import nonebot_plugin_ai_groupmate as plugin

    assert not plugin._is_explicit_meme_request(message)
    assert plugin._get_explicit_meme_send_count(message) == 1


@pytest.mark.parametrize(("message", "count"), [
    ("来个表情包", 1), ("发三张表情包", 3), ("来点梗图", 3),
    ("随便发点表情", 3),
])
def test_actual_meme_message_keeps_required_delivery(message, count):
    import nonebot_plugin_ai_groupmate as plugin

    assert plugin._is_explicit_meme_request(message)
    assert plugin._get_explicit_meme_send_count(message) == count
