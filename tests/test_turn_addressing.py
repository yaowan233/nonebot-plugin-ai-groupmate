import json
from uuid import uuid4
from types import SimpleNamespace
from typing import cast
from datetime import datetime
from unittest.mock import Mock, AsyncMock

import pytest
from nonebot.adapters import Bot, Event
from langchain_core.messages import AIMessage


def row(mid, user, body, *, kind="text"):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    return ChatHistorySchema(msg_id=mid, session_id="group", user_id=user, user_name=user,
                             content_type=kind, content=body, created_at=datetime.now())


def test_queued_turn_keeps_its_sender_transport_alias_and_image_parts():
    from nonebot_plugin_ai_groupmate.agent.turn_context import bind_turn_history, build_turn_context

    rows = [row(1, "alice", "id: 100\nalias_id: alice-alias\n给我解释一下"),
            row(2, "alice", "id: 100\n图.png", kind="image"),
            row(3, "bob", "id: 200\n小鲸鱼，介绍一下自己")]
    bound, current = bind_turn_history(rows, "alice", "alice-alias")
    assert current is rows[0]
    assert bound == rows[:2]
    data = json.loads(build_turn_context(rows, user_id="alice", user_name="Alice", current_text="给我解释一下", message_id="alice-alias"))
    assert data["latest_message"]["speaker_id"] == "alice"
    assert [m["speaker_id"] for m in data["recent_messages"]] == ["alice", "alice"]


def test_missing_trigger_does_not_fall_back_to_someone_elses_message():
    from nonebot_plugin_ai_groupmate.agent.turn_context import bind_turn_history, build_turn_context

    rows = [row(1, "bob", "id: 200\n我的问题")]
    assert bind_turn_history(rows, "alice", "100")[1] is None
    data = json.loads(build_turn_context(rows, user_id="alice", user_name="Alice", current_text="实际触发内容", message_id="100"))
    assert data["latest_message"]["speaker_id"] == "alice"
    assert data["latest_message"]["text"] == "实际触发内容"


@pytest.mark.asyncio
@pytest.mark.parametrize(("recipient", "continuous", "allowed", "engaged"), [
    ("other", False, False, False), ("other", True, False, False),
    ("unclear", False, False, False), ("group", False, True, False),
    ("group", True, False, False), ("bot", False, True, True), ("bot", True, True, True),
    ("invented", False, False, False), (None, False, False, False),
])
async def test_recipient_decision_controls_admission_and_continuation(monkeypatch, recipient, continuous, allowed, engaged):
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.agent.turn_context import build_turn_context

    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps({
        "recipient": recipient, "should_reply": True, "continues_bot_conversation": True,
    }))))
    monkeypatch.setattr(agent, "get_flash_model", lambda: model)
    callback = Mock()
    context = build_turn_context([row(1, "alice", "id: 100\n小鲸鱼能做什么")], user_id="alice", user_name="Alice", current_text="小鲸鱼能做什么", continuous=continuous)
    assert await agent.check_if_should_reply("history", "小鲸鱼能做什么", "miyuki", turn_context=context, is_continuous=continuous, on_continuation=callback) is allowed
    assert callback.called is engaged
    assert model.ainvoke.call_args.kwargs["max_tokens"] >= 160


@pytest.mark.asyncio
@pytest.mark.parametrize(("flag", "scope"), [("proactive_meme_only", "一张表情包"), ("proactive_reaction_only", "消息表情")])
async def test_recipient_classifier_keeps_lightweight_response_scope(monkeypatch, flag, scope):
    from nonebot_plugin_ai_groupmate import agent

    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='{"recipient":"group","should_reply":false,"continues_bot_conversation":false}')))
    monkeypatch.setattr(agent, "get_flash_model", lambda: model)
    assert not await agent.check_if_should_reply(
        "history", "current", "miyuki", turn_context="{}",
        proactive_meme_only=flag == "proactive_meme_only",
        proactive_reaction_only=flag == "proactive_reaction_only",
    )
    assert scope in model.ainvoke.call_args.args[0][0].content
    assert "认真求助" in model.ainvoke.call_args.args[0][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["followup", "ambient", "observation"])
async def test_other_recipient_blocks_actual_reply_pipeline(monkeypatch, mode):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    group = f"recipient-{uuid4()}"
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="text", content="id: 100\n小鲸鱼能做什么，介绍一下自己"))
        await db.commit()
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit_enabled", False)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda _: 0.01)
    monkeypatch.setattr(plugin, "_observation_allowed", lambda _: True)
    plugin._refresh_continuous_conversation(group, "alice", "bot")
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps({
        "recipient": "other", "should_reply": True, "continues_bot_conversation": True,
    }))))
    monkeypatch.setattr(agent, "get_flash_model", lambda: model)
    choose = AsyncMock(return_value=SimpleNamespace(need_reply=False, text=None))
    monkeypatch.setattr(plugin, "choice_response_strategy", choose)
    await plugin.handle_reply_logic(
        "request", cast(Uninfo, SimpleNamespace(scene=SimpleNamespace(id=group, type=SceneType.GROUP), self_id="bot")),
        cast(QryItrface, SimpleNamespace(get_members=AsyncMock(return_value=[]))), cast(Bot, SimpleNamespace(self_id="bot")),
        cast(Event, SimpleNamespace(message_id="100", get_plaintext=lambda: "小鲸鱼能做什么，介绍一下自己")),
        "miyuki", "alice", "Alice", False, mode == "followup", None, proactive_observation=mode == "observation",
    )
    model.ainvoke.assert_awaited_once()
    choose.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", ["event", "batch"])
async def test_queued_pipeline_does_not_promote_newer_users_input_to_current_request(monkeypatch, anchor):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.message_batching import current_input_batch

    group = f"queued-recipient-{uuid4()}"
    async with get_session() as db:
        db.add_all([
            ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="text", content="id: 100\nalias_id: first-alias\n你刚才第二步怎么做"),
            ChatHistory(session_id=group, user_id="bob", user_name="Bob", content_type="text", content="id: 200\n小鲸鱼介绍一下自己"),
        ])
        await db.commit()
    async def judge(messages, **_kwargs):
        data = json.loads(messages[1].content)
        assert data["latest_message"]["speaker_id"] == "alice"
        assert data["latest_message"]["text"] == "你刚才第二步怎么做"
        assert all(m["speaker_id"] != "bob" for m in data["recent_messages"])
        return AIMessage(content='{"recipient":"bot","should_reply":true,"continues_bot_conversation":true}')
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=judge))
    monkeypatch.setattr(agent, "get_flash_model", lambda: model)
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit_enabled", False)
    monkeypatch.setattr(plugin, "resolve_group_reply_probability", lambda _: 0.01)
    choose = AsyncMock(return_value=SimpleNamespace(need_reply=False, text=None))
    monkeypatch.setattr(plugin, "choice_response_strategy", choose)
    token = current_input_batch.set((("first-alias", "你刚才第二步怎么做"),)) if anchor == "batch" else None
    try:
        await plugin.handle_reply_logic(
            "request", cast(Uninfo, SimpleNamespace(scene=SimpleNamespace(id=group, type=SceneType.GROUP), self_id="bot")),
            cast(QryItrface, SimpleNamespace(get_members=AsyncMock(return_value=[]))), cast(Bot, SimpleNamespace(self_id="bot")),
            cast(Event, SimpleNamespace(message_id="first-alias" if anchor == "event" else None, get_plaintext=lambda: "你刚才第二步怎么做")),
            "miyuki", "alice", "Alice", False, False, None,
        )
    finally:
        if token is not None:
            current_input_batch.reset(token)
    model.ainvoke.assert_awaited_once()
    choose.assert_awaited_once()
    assert [m.user_id for m in choose.call_args.args[3]] == ["alice"]


def test_group_boundary_does_not_turn_third_party_speech_into_a_self_request():
    from nonebot_plugin_ai_groupmate.agent import _build_current_request_boundary

    history = [row(1, "alice", "id: 100\n小鲸鱼能做什么，介绍一下自己")]
    boundary = _build_current_request_boundary(history, "alice", "Alice", None)
    assert "最新发言，不自动构成给你的请求" in boundary
    assert "不代答、不自我介绍" in boundary
    assert "<latest_message>" in boundary
    assert "<current_request>" not in boundary
    assert "【发言对象】" not in _build_current_request_boundary(history, "alice", "Alice", None, is_private=True)
