import datetime

import pytest


@pytest.mark.parametrize("content_type", ["text", "bot"])
def test_history_hides_platform_id(tmp_path, content_type):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    history = ChatHistorySchema(
        msg_id=1,
        session_id="group-1",
        content="id: 123456\nhello",
        content_type=content_type,
        user_id="bot" if content_type == "bot" else "1001",
        user_name="sender",
        media_id=None,
        created_at=datetime.datetime.now(),
    )
    messages = format_chat_history([history], pic_dir=tmp_path, bot_name="bot")

    assert "123456" not in messages[0].content
    assert "hello" in messages[0].content
