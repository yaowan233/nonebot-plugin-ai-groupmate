from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _history(msg_id, content_type, content, *, session_id="group-1", media_id=None):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    return ChatHistorySchema(
        msg_id=msg_id,
        session_id=session_id,
        user_id="miyuki" if content_type == "bot" else "adapter-bot-account",
        user_name="miyuki" if content_type == "bot" else "echo-account",
        content_type=content_type,
        content=content,
        created_at=datetime(2026, 10, 2, 16, 25),
        media_id=media_id,
    )


@pytest.mark.parametrize("echo_first", [False, True])
def test_formatter_keeps_bot_text_and_discards_cross_identity_echo(tmp_path, echo_first):
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    bot = _history(1, "bot", "id: 112801593\n只能说大概率是只短毛猫")
    echo = _history(2, "text", "id: 112801593\n只能说大概率是只短毛猫")
    real_repeat = _history(3, "text", "id: 112801594\n只能说大概率是只短毛猫")
    history = [echo, bot, real_repeat] if echo_first else [bot, echo, real_repeat]

    formatted = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki")

    assert len(formatted) == 2
    assert isinstance(formatted[0], AIMessage)
    assert isinstance(formatted[1], HumanMessage)


def test_formatter_does_not_reintroduce_bot_image_echo_as_human_inline_image(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    history = [
        _history(1, "bot", "id: -669237620\n图片文件: fish.png", media_id=12),
        _history(2, "image", "id: -669237620\nfish.png", media_id=12),
    ]
    formatted = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki")

    assert len(formatted) == 1
    assert isinstance(formatted[0], AIMessage)
    assert "[图片]" in formatted[0].content


def test_append_only_history_drops_delayed_echo_of_previous_bot_turn(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.conversation import (
        ActiveConversationThread,
        build_append_only_history,
        active_conversation_threads,
    )
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    bot = _history(1, "bot", "id: 10001\n我在这里")
    echo = _history(2, "text", "id: 10001\n我在这里")
    active_conversation_threads["group-1"] = ActiveConversationThread(
        messages=[AIMessage("我在这里")], last_msg_id=1, updated_at=datetime.now()
    )

    def format_history(history, max_images, roles, extra):
        return format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki")

    try:
        messages, appended, reused = build_append_only_history(
            "group-1", [bot, echo], format_history=format_history
        )
    finally:
        active_conversation_threads.pop("group-1", None)

    assert reused is True
    assert len(messages) == 1
    assert appended == []


def test_formatter_preserves_user_multimodal_parts_and_other_group_same_id(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    history = [
        _history(1, "bot", "id: 10\n另一群的话", session_id="other-group"),
        _history(2, "text", "id: 10\n看看这个"),
        _history(3, "image", "id: 10\npic.png", media_id=12),
    ]
    formatted = format_chat_history(
        history, pic_dir=tmp_path, bot_name="miyuki", max_inline_images=0
    )

    assert len(formatted) == 2
    assert isinstance(formatted[1], HumanMessage)
    assert "看看这个 [图片]" in formatted[1].content


@pytest.mark.parametrize("placeholder", ["unknown", "system", "0"])
def test_formatter_does_not_merge_unreliable_platform_ids(tmp_path, placeholder):
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    formatted = format_chat_history(
        [
            _history(1, "bot", f"id: {placeholder}\n上一句话"),
            _history(2, "text", f"id: {placeholder}\n真正的新话"),
        ],
        pic_dir=tmp_path,
        bot_name="miyuki",
    )

    assert len(formatted) == 2
    assert isinstance(formatted[-1], HumanMessage)


def test_formatter_recognizes_bot_alias_ids_without_leaking_headers(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    formatted = format_chat_history(
        [
            _history(1, "bot", "id: message-1\nalias_id: alias-1\naliasid: alias-2\n你好"),
            _history(2, "text", "id: alias-1\n你好"),
            _history(3, "text", "id: alias-2\n你好"),
        ],
        pic_dir=tmp_path, bot_name="miyuki",
    )

    assert len(formatted) == 1
    assert isinstance(formatted[0], AIMessage)
    assert formatted[0].content == "你好"


def test_old_platform_id_collision_does_not_hide_new_user_message(tmp_path):
    from datetime import timedelta

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    bot = _history(1, "bot", "id: 42\n很久前的回复")
    user = _history(2, "text", "id: 42\n新的问题")
    user.created_at = bot.created_at + timedelta(hours=2)

    formatted = format_chat_history([bot, user], pic_dir=tmp_path, bot_name="miyuki")

    assert len(formatted) == 2
    assert isinstance(formatted[-1], HumanMessage)
    assert "新的问题" in formatted[-1].content


@pytest.mark.asyncio
async def test_cache_update_does_not_append_echo_of_bot_before_input_boundary(tmp_path):
    from uuid import uuid4

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.conversation import update_active_thread, active_conversation_threads
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    session_id = f"cache-late-echo-{uuid4()}"
    def format_history(history, max_images, roles, extra):
        return format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki")

    async with get_session() as db:
        bot = ChatHistory(session_id=session_id, user_id="miyuki", user_name="miyuki", content_type="bot", content="id: 1200\n我在这里", media_id=None)
        db.add(bot)
        await db.flush()
        input_boundary = bot.msg_id
        db.add(ChatHistory(session_id=session_id, user_id="adapter-account", user_name="adapter-account", content_type="text", content="id: 1200\n我在这里", media_id=None))
        await db.commit()

        await update_active_thread(db, session_id, [AIMessage("我在这里")], input_boundary, format_history=format_history)

    thread = active_conversation_threads.pop(session_id)
    assert len(thread.messages) == 1
    assert isinstance(thread.messages[0], AIMessage)
    assert thread.last_msg_id > input_boundary
