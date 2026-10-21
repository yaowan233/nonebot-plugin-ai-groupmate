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
    monkeypatch.setattr(plugin, "note_group_activity", lambda *_, **__: None)
    monkeypatch.setattr(plugin, "_load_repeat_chain_text", AsyncMock(return_value=None))
    monkeypatch.setattr(plugin, "_sample_proactive_reply_modes", lambda **_: (False, False, False))
    monkeypatch.setattr(plugin, "is_onebot_context", lambda *_: False)
    monkeypatch.setattr(plugin, "_start_background_image_task", lambda *args: images.append(args))
    monkeypatch.setattr(plugin.plugin_config, "bot_name", "咕咕")
    monkeypatch.setattr(plugin.plugin_config, "continuous_conversation_minutes", 5)
    plugin._continuous_conversation_until.clear()
    plugin.participation.states.clear()
    plugin._observation_schedule.clear()

    async def deliver(
        session_id,
        message_id,
        text,
        *,
        bot_id="bot-1",
        user_id="member-1",
        user_name=None,
        member_nick=None,
        addressed=False,
        message=None,
        original_message=None,
        reply=None,
        is_private=False,
    ):
        monkeypatch.setattr(plugin, "get_message_id", lambda: message_id)
        event = SimpleNamespace(
            original_message=original_message or [],
            reply=reply,
            get_plaintext=lambda: text,
            is_tome=lambda: addressed,
        )
        session = SimpleNamespace(
            scene=SimpleNamespace(id=session_id, type=SceneType.PRIVATE if is_private else SceneType.GROUP),
            user=SimpleNamespace(id=user_id, name=user_name if user_name is not None else user_id, nick=None),
            member=SimpleNamespace(nick=member_nick) if member_nick is not None else None,
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
    plugin.participation.states.clear()
    plugin._observation_schedule.clear()


@pytest.mark.asyncio
async def test_only_new_group_text_feeds_affinity_with_public_api(deliver_message, monkeypatch):
    from uuid import uuid4
    from unittest.mock import Mock

    import nonebot_plugin_ai_groupmate as plugin

    deliver, _, _ = deliver_message
    note = Mock()
    monkeypatch.setattr(plugin, "note_group_activity", note)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda *_: 0.01)
    group = f"affinity-inbound-{uuid4()}"
    await deliver(group, "new", "最近有首歌很好听")
    await deliver(group, "new", "最近有首歌很好听")
    await deliver(group, "command", "/查看群API")
    await deliver(f"private-{uuid4()}", "private", "聊聊天", is_private=True)
    note.assert_called_once_with(("bot-1", group), interaction=False)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda *_: 0)
    await deliver(group, "disabled", "另一个话题")
    assert note.call_count == 1


@pytest.mark.asyncio
async def test_ordinary_chat_is_counted_locally_and_only_due_looks_are_queued(deliver_message, monkeypatch):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.observation import ObservationSchedule

    deliver, queued, _ = deliver_message
    now = [1000.0]
    observer = ObservationSchedule(lambda: now[0], lambda: 2700)
    monkeypatch.setattr(plugin, "_observation_schedule", observer)
    monkeypatch.setattr(plugin.plugin_config, "proactive_observation_enabled", True)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda *_: 0.01)
    gatekeeper, agent = AsyncMock(), AsyncMock()
    monkeypatch.setattr(plugin, "check_if_should_reply", gatekeeper)
    monkeypatch.setattr(plugin, "choice_response_strategy", agent)
    for number in range(12):
        await deliver("observation-inbound", str(number), f"最近听的第{number}首歌")
    await deliver("observation-inbound", "11", "最近听的第11首歌")  # Duplicate.
    await deliver("observation-inbound", "command", "/查看群API")
    await deliver("observation-private", "private", "私聊不观察", is_private=True)
    queued.reset_mock()  # Private delivery is an addressed request.
    assert observer.watches[("bot-1", "observation-inbound")].messages == 12
    assert len(observer.watches) == 1
    await plugin._occasionally_observe_groups()
    queued.assert_not_awaited()
    gatekeeper.assert_not_awaited()
    agent.assert_not_awaited()
    now[0] += 2700
    await deliver("observation-inbound", "latest", "这个谱面还挺好玩的")
    await plugin._occasionally_observe_groups()
    queued.assert_awaited_once()
    request = queued.await_args.args[1]
    assert request.proactive_observation
    assert not request.is_tome
    assert not request.is_continuous
    assert request.reply_to_id is None
    assert request.repeat_text is None
    queued.reset_mock()
    for _ in range(10):
        await plugin._occasionally_observe_groups()
    queued.assert_not_awaited()
    gatekeeper.assert_not_awaited()
    agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("suppressed", ["disabled", "probability_zero", "recent_reply", "recent_check", "disconnected", "stopped"])
async def test_due_observation_respects_live_controls(deliver_message, monkeypatch, suppressed):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.observation import ObservationSchedule
    from nonebot_plugin_ai_groupmate.participation import ParticipationPolicy

    deliver, queued, _ = deliver_message
    now = [1000.0]
    observer = ObservationSchedule(lambda: now[0], lambda: 2700)
    monkeypatch.setattr(plugin, "_observation_schedule", observer)
    monkeypatch.setattr(plugin, "participation", ParticipationPolicy(lambda: now[0]))
    monkeypatch.setattr(plugin.plugin_config, "proactive_observation_enabled", True)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda *_: 0.01)
    for number in range(12):
        await deliver("observation-controls", str(number), f"第{number}条新聊天")
    now[0] += 2700
    await deliver("observation-controls", "latest", "继续这个话题")
    if suppressed == "disabled":
        monkeypatch.setattr(plugin.plugin_config, "proactive_observation_enabled", False)
    elif suppressed == "probability_zero":
        monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda *_: 0)
    elif suppressed == "recent_reply":
        plugin.participation.sent(("bot-1", "observation-controls"), proactive=False)
    elif suppressed == "recent_check":
        plugin.participation.state(("bot-1", "observation-controls")).last_check = now[0]
    elif suppressed == "disconnected":
        monkeypatch.setattr(plugin, "get_bots", lambda: {})
    else:
        await deliver("observation-controls", "stop", "别插话", addressed=True)
    queued.reset_mock()
    await plugin._occasionally_observe_groups()
    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_observation_uses_only_bounded_recent_text_and_can_be_disabled_inflight(deliver_message, monkeypatch):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.participation import observation_guard, group_delivery_allowed

    session_id = "observation-bounded-context"
    async with get_session() as db:
        for number in range(30):
            db.add(ChatHistory(session_id=session_id, user_id="member-1", user_name="Alice", content_type="text", content=f"第{number}条" + "聊天正文" * 300, created_at=datetime.datetime.now() - datetime.timedelta(seconds=60 - number)))
        db.add(ChatHistory(session_id=session_id, user_id="member-1", user_name="Alice", content_type="image", content="图片不分析"))
        db.add(ChatHistory(session_id="other-group", user_id="member-1", user_name="Alice", content_type="text", content="别的群不读取"))
        await db.commit()
    monkeypatch.setattr(plugin.plugin_config, "proactive_observation_enabled", True)
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit_enabled", False)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda *_: 0.01)
    gatekeeper, history_loader = AsyncMock(), AsyncMock()
    monkeypatch.setattr(plugin, "check_if_should_reply", gatekeeper)
    monkeypatch.setattr(plugin, "_load_agent_history", history_loader)

    async def choose(db, group, request_id, history, *args, **kwargs):
        assert len(history) == 20
        assert all(row.session_id == session_id and row.content_type == "text" and len(row.content) <= 600 for row in history)
        assert kwargs["proactive_observation"] is True
        assert group_delivery_allowed(session_id)
        monkeypatch.setattr(plugin.plugin_config, "proactive_observation_enabled", False)
        assert not group_delivery_allowed(session_id)
        return SimpleNamespace(need_reply=False, text=None)

    agent = AsyncMock(side_effect=choose)
    monkeypatch.setattr(plugin, "choice_response_strategy", agent)
    await plugin.handle_reply_logic(
        "observation-test", cast(Uninfo, SimpleNamespace(scene=SimpleNamespace(id=session_id, type=SceneType.GROUP), self_id="bot-1")),
        cast(QryItrface, SimpleNamespace(get_members=AsyncMock(return_value=[]))), cast(Bot, SimpleNamespace(self_id="bot-1")),
        cast(Event, SimpleNamespace()), "咕咕", "member-1", "Alice", False, False, None, proactive_observation=True,
    )
    agent.assert_awaited_once()
    gatekeeper.assert_not_awaited()
    history_loader.assert_not_awaited()
    assert observation_guard.get() is None
    assert plugin.delivery_context.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_name", "member_nick", "text"),
    [
        ("\n\t\n\x07$ÿĀ\x11\x10\n\x06\x12\x04\n\t\n\x07$ÿĀ\x11\x10\x10\x00", None, "太像那谁"),
        ("Alice", "群\x00昵称", "太像那谁"),
        ("Alice", None, "太\x00像那谁\x00\n下一行\t🙂"),
        ("中文昵称🙂", None, "太像那谁\n下一行\t🙂"),
    ],
)
async def test_inbound_text_strips_nul_before_database_write(deliver_message, user_name, member_nick, text):
    from uuid import uuid4

    from sqlalchemy import Select, event
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory

    # SQLite accepts NUL, so enforce PostgreSQL's restriction at the SQL boundary.
    def reject_nul(_conn, _cursor, statement, parameters, _context, _executemany):
        if statement.startswith("INSERT INTO nonebot_plugin_ai_groupmate_chathistory"):
            if any(isinstance(value, str) and "\x00" in value for value in parameters):
                raise ValueError('invalid byte sequence for encoding "UTF8": 0x00')

    deliver, _, _ = deliver_message
    session_id = f"inbound-nul-{uuid4()}"
    async with get_session() as db:
        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", reject_nul)
        try:
            await deliver(session_id, "-2072206098", text, user_name=user_name, member_nick=member_nick)
            # The body fallback must compare the same sanitized text across bots.
            await deliver(session_id, "-2072206097", text, user_name=user_name, member_nick=member_nick, bot_id="bot-2")
            rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == session_id))).scalars().all()
            assert len(rows) == 1
            assert rows[0].user_name == (member_nick or user_name).replace("\x00", "")
            assert rows[0].content == f"id: -2072206098\n{text.replace(chr(0), '')}"
        finally:
            event.remove(engine, "before_cursor_execute", reject_nul)


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
@pytest.mark.parametrize("stop_while_judging", [False, True])
async def test_adaptive_reply_sees_bot_history_and_rechecks_stop(deliver_message, monkeypatch, stop_while_judging):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    deliver, _, _ = deliver_message
    session_id = f"adaptive-history-{stop_while_judging}"
    key = ("bot-1", session_id)
    async with get_session() as db:
        db.add(ChatHistory(session_id=session_id, user_id="bot-1", user_name="咕咕", content_type="bot", content="先把大任务拆成三个步骤。"))
        await db.commit()
    await deliver(session_id, "1", "你刚才说的三个步骤怎么分？", user_id="member-2")
    captured = []

    async def gatekeeper(history, current, *args, **kwargs):
        captured.append(history)
        assert "[Bot]" in history
        assert "先把大任务拆成三个步骤" in history
        assert "member-2" in history
        assert "三个步骤怎么分" in current
        assert kwargs["participation_context"]
        assert "affinity_review" not in kwargs
        if stop_while_judging:
            plugin.participation.observe(key, addressed=False, disengaged=True)
        kwargs["on_continuation"]()
        return True

    agent = AsyncMock(return_value=SimpleNamespace(need_reply=False, text=None))
    monkeypatch.setattr(plugin, "check_if_should_reply", gatekeeper)
    monkeypatch.setattr(plugin, "choice_response_strategy", agent)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda _: 0.01)
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit_enabled", False)
    await plugin.handle_reply_logic(
        "adaptive-request",
        cast(Uninfo, SimpleNamespace(scene=SimpleNamespace(id=session_id, type=SceneType.GROUP), self_id="bot-1")),
        cast(QryItrface, SimpleNamespace(get_members=AsyncMock(return_value=[]))),
        cast(Bot, SimpleNamespace(self_id="bot-1")),
        cast(Event, SimpleNamespace()),
        "咕咕", "member-2", "Bob", False, False, None,
    )
    assert len(captured) == 1
    assert agent.await_count == int(not stop_while_judging)
    assert plugin.delivery_context.get() is None


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
