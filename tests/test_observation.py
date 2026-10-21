from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, AsyncMock

import pytest

KEY = ("bot", "group")


@pytest.fixture
def schedule():
    from nonebot_plugin_ai_groupmate.observation import ObservationSchedule

    now = [1000.0]
    return ObservationSchedule[str](lambda: now[0], lambda: 2700), now


def test_time_new_messages_and_recent_activity_are_all_required(schedule):
    observer, now = schedule
    allowed = Mock(return_value=True)
    for number in range(11):
        observer.note(KEY, str(number))
    now[0] += 2700
    assert observer.take_due(allowed) is None  # Fewer than twelve messages.
    observer.note(KEY, "latest")
    assert observer.take_due(allowed) == "latest"
    allowed.assert_called_once_with(KEY)
    for _ in range(100):
        assert observer.take_due(allowed) is None
    now[0] += 2700
    assert observer.take_due(allowed) is None  # Time alone earns no look.
    for _ in range(12):
        observer.note(KEY, "new conversation")
    now[0] += 901
    assert observer.take_due(allowed) is None  # An idle group isn't awakened.
    observer.note(KEY, "active again")
    assert observer.take_due(allowed) == "active again"


def test_new_messages_do_not_bypass_interval_and_blocked_looks_back_off(schedule):
    observer, now = schedule
    for _ in range(12):
        observer.note(KEY, "chat")
    allowed = Mock(return_value=False)
    now[0] += 2699
    observer.note(KEY, "latest")
    assert observer.take_due(allowed) is None
    allowed.assert_not_called()
    now[0] += 1
    assert observer.take_due(allowed) is None
    allowed.assert_called_once()
    for _ in range(100):
        observer.note(KEY, "more chat")
        assert observer.take_due(allowed) is None
    assert allowed.call_count == 1


def test_stop_pauses_until_addressed_and_bot_participation_defers_looks(schedule):
    observer, now = schedule
    observer.note(KEY, "stop", disengaged=True)
    now[0] += 2700
    for _ in range(12):
        observer.note(KEY, "others talking")
    assert observer.take_due(lambda _: True) is None
    observer.note(KEY, "called bot again", addressed=True)
    for _ in range(12):
        observer.note(KEY, "chat")
    now[0] += 2000
    observer.participated(KEY)
    now[0] += 2700
    observer.note(KEY, "one new message")
    assert observer.take_due(lambda _: True) is None
    for _ in range(11):
        observer.note(KEY, "new conversation")
    assert observer.take_due(lambda _: True) == "new conversation"


def test_groups_are_scoped_globally_throttled_bounded_and_expire(schedule):
    observer, now = schedule
    keys = (KEY, ("other-bot", "group"), ("bot", "other-group"))
    for key in keys:
        for _ in range(12):
            observer.note(key, str(key))
    now[0] += 2700
    for key in keys:
        observer.note(key, str(key))
    assert observer.take_due(lambda _: True) == str(KEY)
    assert observer.take_due(lambda _: True) is None
    now[0] += 300
    assert observer.take_due(lambda _: True) == str(keys[1])
    for number in range(300):
        observer.note(("bot", str(number)), "bounded")
    assert len(observer.watches) == 256
    now[0] += 21601
    assert observer.take_due(lambda _: True) is None
    assert not observer.watches


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", ["running", "addressed", "addressed_tasks", "latest"])
async def test_observation_never_interrupts_existing_requests(monkeypatch, busy):
    import nonebot_plugin_ai_groupmate as plugin

    state = plugin.GroupReplyState()
    setattr(state, busy, True if busy == "running" else ["existing"])
    monkeypatch.setattr(plugin, "_group_reply_states", {"group": state})
    request = cast(plugin.ReplyRequest, SimpleNamespace(proactive_observation=True))
    assert not await plugin._queue_group_reply_request("group", request)
    assert getattr(state, busy)


@pytest.mark.asyncio
async def test_observation_prompt_has_no_current_request_or_reused_thread(monkeypatch):
    from datetime import datetime

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    captured = {}

    class Graph:
        async def ainvoke(self, state, **kwargs):
            captured["prompt"] = state["messages"][-1].content
            return {**state, "called_finish": 1, "llm_call_count": 1}

    for name in ("get_user_relation_context", "get_recent_relations_context", "get_group_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "build_chat_graph", lambda *_, **__: Graph())
    no_thread = Mock(side_effect=AssertionError("must not revive a thread or old request"))
    monkeypatch.setattr(agent, "build_append_only_history", no_thread)
    monkeypatch.setattr(agent, "_build_current_request_boundary", no_thread)
    thread_update = AsyncMock()
    avatar = AsyncMock()
    monkeypatch.setattr(agent, "update_active_thread", thread_update)
    monkeypatch.setattr(agent, "build_avatar_context_messages", avatar)
    monkeypatch.setattr(agent, "record_token_usage", AsyncMock())
    history = [ChatHistorySchema(msg_id=1, session_id="observation-prompt", user_id="member", user_name="Member", content_type="text", content="上次那个问题已经解决了", created_at=datetime.now())]
    async with get_session() as db:
        result = await agent.choice_response_strategy(db, "observation-prompt", None, history, "member", "Member", bot_id="bot", group_members=[], proactive_observation=True)
    assert not result.need_reply
    assert "没有当前用户请求" in captured["prompt"]
    assert 'next_step="end"' in captured["prompt"]
    assert "看完就调用 `finish`" in captured["prompt"]
    assert "追答旧问题" in captured["prompt"]
    no_thread.assert_not_called()
    thread_update.assert_not_awaited()
    avatar.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["silent", "reply", "invalid", "progress"])
async def test_observation_graph_stays_silent_or_sends_once_without_budget_notice(decision):
    from langchain_core.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage

    from nonebot_plugin_ai_groupmate.agent.graph import AgentRunLimits, TerminalMetadata, build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.common_tools import finish
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    sent = []

    @tool("reply_user")
    async def reply_user(content: str, next_step: str = "end") -> str:
        """Send one natural group message."""
        sent.append(content)
        return tool_success("sent", "sent")

    class Model:
        calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if decision == "progress":
                return AIMessage(content="我稍后再回你")
            if decision == "silent":
                calls = [{"name": "finish", "args": {"group_feeling": {"feeling": "neutral"}}, "id": "finish"}]
            elif decision == "reply":
                calls = [{"name": "reply_user", "args": {"content": "这个节奏我也喜欢", "next_step": "continue"}, "id": str(number)} for number in range(2)]
            else:
                calls = [{"name": "schedule_agent_task", "args": {"task": "继续查看"}, "id": str(self.calls)}]
            return AIMessage(content="", tool_calls=calls)

    metadata_saved = AsyncMock()
    model = Model()
    graph = build_chat_graph(
        model, [reply_user, finish], "可以只看看然后保持沉默",
        limits=AgentRunLimits(max_llm_calls=2, max_replies=1, finalize_on_budget_exhaustion=False),
        terminal_metadata=TerminalMetadata("group_feeling", {"type": "object"}, metadata_saved),
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="群友正在聊新歌")], "session_id": "observation", "request_id": None})
    assert len(sent) == int(decision == "reply")
    assert model.calls == (2 if decision in {"invalid", "progress"} else 1)
    assert not result.get("budget_finalization_attempted", False)
    if decision == "silent":
        metadata_saved.assert_awaited_once_with({"feeling": "neutral"})
