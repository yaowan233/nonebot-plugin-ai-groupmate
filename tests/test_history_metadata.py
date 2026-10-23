from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text", "bot", "image"])
async def test_history_strips_nul_on_insert_and_update(content_type):
    from uuid import uuid4

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    async with get_session() as db:
        row = ChatHistory(
            session_id=f"history-nul-{uuid4()}",
            user_id="member-1",
            user_name="中\x00文🙂",
            content_type=content_type,
            content="id: 42\n太\x00像那谁\t🙂",
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        assert row.user_name == "中文🙂"
        assert row.content == "id: 42\n太像那谁\t🙂"

        row.user_name = "新\x00昵称"
        row.content = "id: 42\n\x00新正文\n"
        await db.commit()
        await db.refresh(row)
        assert row.user_name == "新昵称"
        assert row.content == "id: 42\n新正文\n"


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


@pytest.mark.parametrize("echo_first", [None, False, True])
@pytest.mark.parametrize("image_limit", [0, 3])
def test_sent_meme_meaning_survives_history_rebuild_without_image_bytes(tmp_path, echo_first, image_limit):
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    bot = _history(1, "bot", "id: 1222234946\n发送了图片，图片描述是: 黄色卡通角色，配文‘切换为奶龙模式’，表示进入轻松装傻的状态。\n图片文件: sent-meme.jpg", media_id=1056353)
    echo = _history(2, "image", "id: 1222234946\nsent-meme.jpg", media_id=1056353)
    followup = _history(3, "text", "id: 1222234947\n你刚才发的那张是什么意思")
    history = [bot] if echo_first is None else [echo, bot] if echo_first else [bot, echo]

    messages = format_chat_history([*history, followup], pic_dir=tmp_path, bot_name="miyuki", max_inline_images=image_limit)

    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert "切换为奶龙模式" in messages[0].content
    assert "轻松装傻" in messages[0].content
    assert "sent-meme.jpg" not in messages[0].content
    assert "图片文件:" not in messages[0].content
    assert isinstance(messages[1], HumanMessage)


@pytest.mark.parametrize("echo_first", [False, True])
def test_sent_meme_original_is_attached_once_in_provider_compatible_role(tmp_path, echo_first):
    import base64
    from io import BytesIO

    from PIL import Image

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    Image.new("RGB", (11, 7), "red").save(tmp_path / "sent-meme.png")
    bot = _history(1, "bot", "id: 1222234946\n图片文件: sent-meme.png", media_id=1056353)
    echo = _history(2, "image", "id: 1222234946\nsent-meme.png", media_id=1056353)
    followup = _history(3, "text", "id: 1222234947\n你刚才发的那张是什么意思")

    history = [echo, bot, followup] if echo_first else [bot, echo, followup]
    messages = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki")

    assert len(messages) == 3
    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], HumanMessage)
    blocks = messages[1].content
    assert isinstance(blocks, list)
    assert isinstance(blocks[0], dict)
    assert "miyuki" in blocks[0]["text"]
    assert "历史" in blocks[0]["text"]
    image_blocks = [part for part in blocks if isinstance(part, dict) and part["type"] == "image_url"]
    assert len(image_blocks) == 1
    raw = base64.b64decode(image_blocks[0]["image_url"]["url"].split(",", 1)[1])
    with Image.open(BytesIO(raw)) as original:
        assert original.size == (11, 7)
    assert "刚才发的" in messages[2].content


@pytest.mark.asyncio
async def test_sent_meme_original_survives_active_thread_update(tmp_path):
    from uuid import uuid4

    from PIL import Image
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.conversation import update_active_thread, active_conversation_threads
    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    Image.new("RGB", (11, 7), "red").save(tmp_path / "sent-meme.png")
    session_id = f"cache-sent-meme-{uuid4()}"

    def format_history(history, max_images, roles, extra):
        return format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki", max_inline_images=max_images, extra_inline_images=extra)

    try:
        async with get_session() as db:
            trigger = ChatHistory(session_id=session_id, user_id="user", user_name="user", content_type="text", content="id: 1200\n发个表情包")
            db.add(trigger)
            await db.flush()
            boundary = trigger.msg_id
            db.add(ChatHistory(session_id=session_id, user_id="miyuki", user_name="miyuki", content_type="bot", content="id: 1201\n图片文件: sent-meme.png", media_id=1056353))
            await db.commit()
            await update_active_thread(db, session_id, [HumanMessage("发个表情包")], boundary, format_history=format_history)

        thread = active_conversation_threads[session_id]
        assert any(isinstance(m.content, list) and any(p.get("type") == "image_url" for p in m.content if isinstance(p, dict)) for m in thread.messages)
        assert any(isinstance(m, AIMessage) for m in thread.messages)
    finally:
        active_conversation_threads.pop(session_id, None)


def test_sent_and_received_images_share_the_recent_inline_budget(tmp_path):
    import base64
    from io import BytesIO

    from PIL import Image

    from nonebot_plugin_ai_groupmate.agent.history_format import format_chat_history

    history = []
    for index, kind in enumerate(("bot", "image", "bot", "image"), start=1):
        filename = f"image-{index}.png"
        Image.new("RGB", (10 + index, 7), "red").save(tmp_path / filename)
        body = f"图片文件: {filename}" if kind == "bot" else filename
        history.append(_history(index, kind, f"id: {index}\n{body}", media_id=index))

    messages = format_chat_history(history, pic_dir=tmp_path, bot_name="miyuki", max_inline_images=2)
    sizes = []
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                assert isinstance(message, HumanMessage)
                raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                with Image.open(BytesIO(raw)) as original:
                    sizes.append(original.size)
    assert sizes == [(13, 7), (14, 7)]


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
