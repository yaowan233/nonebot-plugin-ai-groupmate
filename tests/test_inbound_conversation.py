import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def deliver_message(monkeypatch):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import SceneType
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin

    queued = AsyncMock(return_value=True)
    images = []
    monkeypatch.setattr(plugin, "get_bots", lambda: {"bot-1": None, "bot-2": None})
    monkeypatch.setattr(plugin, "_queue_group_reply_request", queued)
    monkeypatch.setattr(plugin, "_schedule_rag_vectorization", lambda *_: None)
    monkeypatch.setattr(plugin, "_load_repeat_chain_text", AsyncMock(return_value=None))
    monkeypatch.setattr(plugin, "_sample_proactive_reply_modes", lambda **_: (False, False, False))
    monkeypatch.setattr(plugin, "is_onebot_context", lambda *_: False)
    monkeypatch.setattr(plugin, "_start_background_image_task", lambda *args: images.append(args))
    monkeypatch.setattr(plugin.plugin_config, "bot_name", "咕咕")
    monkeypatch.setattr(plugin.plugin_config, "continuous_conversation_minutes", 5)
    plugin._continuous_conversation_until.clear()

    async def deliver(
        session_id,
        message_id,
        text,
        *,
        bot_id="bot-1",
        user_id="member-1",
        addressed=False,
        message=None,
        original_message=None,
        reply=None,
    ):
        monkeypatch.setattr(plugin, "get_message_id", lambda: message_id)
        event = SimpleNamespace(
            original_message=original_message or [],
            reply=reply,
            get_plaintext=lambda: text,
            is_tome=lambda: addressed,
        )
        session = SimpleNamespace(
            scene=SimpleNamespace(id=session_id, type=SceneType.GROUP),
            user=SimpleNamespace(id=user_id, name=user_id, nick=None),
            member=None,
            self_id=bot_id,
        )
        async with get_session() as db:
            await plugin.handle_message(
                db,
                message if message is not None else UniMessage.text(text),
                session,
                event,
                SimpleNamespace(self_id=bot_id),
                {},
                SimpleNamespace(get_members=AsyncMock(return_value=[])),
            )

    yield deliver, queued, images
    plugin._continuous_conversation_until.clear()


@pytest.mark.asyncio
async def test_followup_window_does_not_follow_user_to_another_bot(deliver_message):
    deliver, queued, _ = deliver_message
    await deliver("followup-bot-isolation", "1", "你好", addressed=True)
    queued.reset_mock()

    await deliver("followup-bot-isolation", "2", "接着聊", bot_id="bot-2")

    queued.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["不是和你说话", "我和另一个bot交流呐，别搞混了", "别插话", "ai别讲话", "闭嘴", "你闭嘴", "咕咕闭嘴"])
async def test_explicit_disengagement_clears_followup_without_reply(deliver_message, text):
    deliver, queued, _ = deliver_message
    await deliver(f"followup-exit-{text}", "1", "你好", addressed=True)
    queued.reset_mock()

    await deliver(f"followup-exit-{text}", "2", text, addressed=True)
    await deliver(f"followup-exit-{text}", "3", "继续刚才的话")

    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_unaccepted_followup_does_not_extend_window(deliver_message):
    import nonebot_plugin_ai_groupmate as plugin

    deliver, queued, _ = deliver_message
    await deliver("followup-no-refresh", "1", "你好", addressed=True)
    original = dict(plugin._continuous_conversation_until)
    queued.reset_mock()

    await deliver("followup-no-refresh", "2", "嗯")

    assert plugin._continuous_conversation_until == original


@pytest.mark.asyncio
@pytest.mark.parametrize("image_only", [False, True])
async def test_outbound_echo_is_deduplicated_across_sender_and_content_type(deliver_message, image_only):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    deliver, queued, images = deliver_message
    session_id = f"outbound-echo-{image_only}"
    async with get_session() as db:
        db.add(ChatHistory(
            session_id=session_id,
            user_id="咕咕",
            user_name="咕咕",
            content_type="bot",
            content="id: 468370284\n发送了图片" if image_only else "id: 468370284\n我也觉得",
            created_at=datetime.datetime.now() - datetime.timedelta(seconds=20),
        ))
        await db.commit()

    await deliver(
        session_id,
        "468370284",
        "" if image_only else "我也觉得",
        user_id="another-process-bot-account",
        message=UniMessage.image(raw=b"not-downloaded") if image_only else None,
    )

    async with get_session() as db:
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == session_id))).scalars().all()
        assert len(rows) == 1
        assert rows[0].content_type == "bot"
    queued.assert_not_awaited()
    assert images == []


@pytest.mark.asyncio
async def test_other_bot_can_still_address_current_bot(deliver_message):
    deliver, queued, _ = deliver_message

    await deliver("other-bot-participation", "fresh-message", "一起聊聊", user_id="another-process-bot-account", addressed=True)

    queued.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["at", "reply"])
async def test_explicit_route_to_other_connected_bot_stays_ignored(deliver_message, reference):
    deliver, queued, _ = deliver_message

    await deliver(
        f"other-bot-route-{reference}",
        "1",
        "你好",
        addressed=True,
        original_message=[SimpleNamespace(type="at", data={"qq": "bot-2"})] if reference == "at" else None,
        reply=SimpleNamespace(sender=SimpleNamespace(user_id="bot-2"), id="42") if reference == "reply" else None,
    )

    queued.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["他说‘别插话’，你觉得呢", "你怎么又来插话哈哈", "不要沉默，继续说", "他说‘闭嘴’，你觉得呢", "闭嘴哈哈，我开玩笑的"])
async def test_quoted_or_playful_words_do_not_disengage(deliver_message, text):
    deliver, queued, _ = deliver_message
    await deliver(f"followup-not-an-exit-{text}", "1", text, addressed=True)

    queued.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_command_to_someone_else_does_not_clear_our_followup(deliver_message):
    from nonebot_plugin_alconna import UniMessage

    deliver, queued, _ = deliver_message
    await deliver("followup-other-target", "1", "你好", addressed=True)
    queued.reset_mock()
    await deliver("followup-other-target", "2", "别插话", message=UniMessage.at("other-member").text("别插话"))
    queued.assert_not_awaited()

    await deliver("followup-other-target", "3", "继续我们的聊天")

    queued.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_text", ["别插话", "咕咕闭嘴"])
async def test_explicit_readdressing_restores_conversation_after_disengagement(deliver_message, stop_text):
    deliver, queued, _ = deliver_message
    session_id = f"followup-restart-{stop_text}"
    await deliver(session_id, "1", "你好", addressed=True)
    await deliver(session_id, "2", stop_text)
    queued.reset_mock()

    await deliver(session_id, "3", "回来聊一下", addressed=True)
    await deliver(session_id, "4", "继续")

    assert queued.await_count == 2


@pytest.mark.asyncio
async def test_human_text_record_does_not_hide_image_component_of_same_message(deliver_message):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    deliver, queued, images = deliver_message
    async with get_session() as db:
        db.add(ChatHistory(session_id="human-image-part", user_id="member-1", user_name="Alice", content_type="text", content="id: 42\n看这张图"))
        await db.commit()

    await deliver("human-image-part", "42", "", message=UniMessage.image(raw=b"not-downloaded"))

    assert len(images) == 1
    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_placeholder_platform_id_is_not_used_as_echo_evidence(deliver_message):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    deliver, queued, _ = deliver_message
    async with get_session() as db:
        db.add(ChatHistory(session_id="unknown-id", user_id="咕咕", user_name="咕咕", content_type="bot", content="id: unknown\n之前的回复"))
        await db.commit()

    await deliver("unknown-id", "unknown", "新问题", addressed=True)

    queued.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("accepted", "disengaged_while_waiting"), [(False, False), (True, False), (True, True)])
async def test_followup_refresh_requires_gatekeeper_acceptance(deliver_message, monkeypatch, accepted, disengaged_while_waiting):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin

    deliver, _, _ = deliver_message
    session_id = f"followup-gate-{accepted}-{disengaged_while_waiting}"
    await deliver(session_id, "1", "你好", addressed=True)
    original = dict(plugin._continuous_conversation_until)
    async def check_gatekeeper(*args, **kwargs):
        if disengaged_while_waiting:
            await deliver(session_id, "2", "不是和你说话", addressed=True)
        return accepted

    gatekeeper = AsyncMock(side_effect=check_gatekeeper)
    agent = AsyncMock(return_value=SimpleNamespace(need_reply=False, text=None))
    monkeypatch.setattr(plugin, "check_if_should_reply", gatekeeper)
    monkeypatch.setattr(plugin, "choice_response_strategy", agent)
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit_enabled", False)
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_private_user_limit_enabled", False)
    await plugin.handle_reply_logic(
        "test-request",
        cast(Uninfo, SimpleNamespace(scene=SimpleNamespace(id=session_id, type=SceneType.GROUP), self_id="bot-1")),
        cast(QryItrface, SimpleNamespace(get_members=AsyncMock(return_value=[]))),
        cast(Bot, SimpleNamespace(self_id="bot-1")),
        cast(Event, SimpleNamespace()),
        "咕咕",
        "member-1",
        "Alice",
        False,
        True,
        None,
    )

    gatekeeper.assert_awaited_once()
    if disengaged_while_waiting:
        assert plugin._continuous_conversation_until == {}
        agent.assert_not_awaited()
    elif accepted:
        assert plugin._continuous_conversation_until != original
        agent.assert_awaited_once()
    else:
        assert plugin._continuous_conversation_until == original
        agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_bot_id_outside_echo_window_does_not_hide_new_input(deliver_message):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    deliver, queued, _ = deliver_message
    async with get_session() as db:
        db.add(ChatHistory(
            session_id="old-platform-id",
            user_id="咕咕",
            user_name="咕咕",
            content_type="bot",
            content="id: 42\n旧的回复",
            created_at=datetime.datetime.now() - datetime.timedelta(hours=2),
        ))
        await db.commit()

    await deliver("old-platform-id", "42", "新问题", addressed=True)

    queued.assert_awaited_once()
    async with get_session() as db:
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == "old-platform-id"))).scalars().all()
        assert len(rows) == 2
        assert rows[-1].content_type == "text"
