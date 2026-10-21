from math import log2
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture(autouse=True)
def isolated_affinity(monkeypatch):
    from collections import OrderedDict

    from nonebot_plugin_ai_groupmate import group_affinity as affinity

    monkeypatch.setattr(affinity, "_feelings", {})
    monkeypatch.setattr(affinity, "_activity", OrderedDict())
    monkeypatch.setattr(affinity, "_review_due", {})
    return affinity


def chat_history(*, start=1, one_speaker=False, group=None, count=12):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    group = group or f"affinity-{uuid4()}"
    history = [ChatHistorySchema(
        msg_id=index, session_id=group, user_id="alice" if one_speaker or index % 2 else "bob",
        user_name="Alice", content_type="text", content=f"id: transport-{index}\n一起聊音游的第{index}条消息",
        created_at=datetime.now(),
    ) for index in range(start, start + count)]
    return ("bot", group), history


def note_messages(affinity, key, *, interaction=False):
    for _ in range(12):
        affinity.note_group_activity(key, interaction=interaction)


def reflection(history, feeling="enjoying"):
    return {
        "feeling": feeling, "reason": "大家一起聊音游很投机，我愿意多聊一会儿",
        "evidence": [{"message_id": row.msg_id, "quote": row.content.split("\n")[-1]} for row in history[:2]],
        "desired_reply_probability": {"enjoying": 0.015, "neutral": None, "uncomfortable": 0.007}[feeling],
    }


async def prepared_review(affinity, *, history=None, key=None, now=None, interaction=False):
    if history is None:
        key, history = chat_history()
    assert key is not None
    note_messages(affinity, key, interaction=interaction)
    review = await affinity.prepare_affinity_review(key, history, now=now)
    assert review is not None
    return review, history


async def run_main_agent(review, value, *, terminal="reply_user", metadata=None):
    from langchain_core.tools import tool

    from nonebot_plugin_ai_groupmate import group_affinity as affinity
    from nonebot_plugin_ai_groupmate.agent.graph import TerminalMetadata, build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.common_tools import finish
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    sent = []
    @tool("reply_user")
    async def reply_user(content: str, next_step: str) -> str:
        """Send the ordinary final answer."""
        sent.append(content)
        return tool_success("message_sent", "sent")

    async def complete(value):
        await affinity.complete_affinity_review(review, value)

    class Model:
        def __init__(self):
            self.calls = []
            self.schemas = []
        def bind_tools(self, tools):
            self.schemas = tools
            return self
        async def ainvoke(self, messages):
            self.calls.append(messages)
            args = {"group_feeling": value}
            if terminal == "reply_user":
                args.update(content="这首歌我也喜欢", next_step="end")
            return AIMessage(content="", tool_calls=[{"name": terminal, "args": args, "id": "final"}])

    model = Model()
    graph = build_chat_graph(model, [reply_user, finish], "完整的人设和群档案", terminal_metadata=metadata or TerminalMetadata("group_feeling", affinity.reflection_schema(), complete))
    result = await graph.ainvoke({"messages": [HumanMessage(content="主循环已有的完整聊天与长期记忆")], "session_id": review.key[1], "request_id": None})
    return model, result, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["reply_user", "finish"])
async def test_main_agent_updates_affinity_with_exactly_one_model_call(isolated_affinity, terminal):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GroupAffinity

    affinity = isolated_affinity
    now = utcnow()
    key, history = chat_history(count=20)
    review, _ = await prepared_review(affinity, key=key, history=history, now=now)
    assert len(review.new_humans) == 20
    model, result, sent = await run_main_agent(review, reflection(history), terminal=terminal)
    assert len(model.calls) == 1
    assert result["llm_call_count"] == 1
    assert len(sent) == int(terminal == "reply_user")
    assert "完整的人设和群档案" in model.calls[0][0].content
    assert "完整聊天与长期记忆" in model.calls[0][-1].content
    for schema in model.schemas:
        assert "group_feeling" in schema["function"]["parameters"]["properties"]
    async with get_session() as db:
        row = await db.get(GroupAffinity, review.key)
        assert row is not None
        assert row.score == pytest.approx(log2(1.5))
        assert row.evidence == reflection(history)["evidence"]
        assert row.last_message_id == history[-1].msg_id
        affinity._feelings.clear()
        await affinity.load_group_affinities(db)
    assert affinity.affinity_multiplier(review.key) > 1
    assert affinity.affinity_multiplier(("other-bot", review.key[1])) == 1
    note_messages(affinity, review.key)
    assert await affinity.prepare_affinity_review(review.key, history, now=now + timedelta(hours=1)) is None


@pytest.mark.asyncio
async def test_ineligible_judgment_keeps_its_original_token_budget(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content='{"should_reply":true,"continues_bot_conversation":false}')
    monkeypatch.setattr(agent, "get_flash_model", lambda: model)
    assert await agent.check_if_should_reply("history", "current", "Bot", participation_context="频率允许")
    model.ainvoke.assert_awaited_once()
    assert model.ainvoke.call_args.kwargs["max_tokens"] == 64
    assert "顺带更新相处感受" not in model.ainvoke.call_args.args[0][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "private", "meme", "reaction", "repeat"])
async def test_main_review_uses_full_loaded_context_and_only_normal_group_turns(monkeypatch, isolated_affinity, mode):
    from types import SimpleNamespace

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.agent.graph import TerminalMetadata

    key, history = chat_history(count=20)
    for name in ("get_user_relation_context", "get_recent_relations_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value="关系上下文"))
    monkeypatch.setattr(agent, "get_group_context", AsyncMock(return_value="已有群文化档案"))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    personality = "完整人设" * 300 + "人设末尾也应保留"
    monkeypatch.setattr(agent.plugin_config, "personality_setting", personality)
    metadata = TerminalMetadata("group_feeling", isolated_affinity.reflection_schema(), AsyncMock())
    prepare = AsyncMock(return_value=(metadata, "主循环感受更新说明"))
    monkeypatch.setattr(agent, "prepare_main_group_review", prepare)
    captured = {}
    def build(model, tools, system_prompt, **kwargs):
        captured.update(kwargs)
        captured["system"] = system_prompt
        return object()
    monkeypatch.setattr(agent, "build_chat_graph", build)
    _, _, context = await agent.create_chat_graph(
        SimpleNamespace(commit=AsyncMock()), key[1], None, "user", "User", history=history, bot_id=key[0],
        is_private=mode == "private", proactive_meme_only=mode == "meme", proactive_reaction_only=mode == "reaction",
        repeat_text="哈哈哈" if mode == "repeat" else None,
    )
    if mode == "normal":
        prepare.assert_awaited_once_with(key[0], key[1], history, None)
        assert captured["terminal_metadata"] is metadata
        assert "主循环感受更新说明" in context
        assert "已有群文化档案" in context
        assert any(personality in str(message.content) for message in captured["system"])
    else:
        prepare.assert_not_awaited()
        assert captured["terminal_metadata"] is None


@pytest.mark.asyncio
async def test_main_review_validates_quotes_beyond_gatekeeper_truncation(isolated_affinity):
    affinity = isolated_affinity
    key, history = chat_history(count=20)
    history[0].content = "前面的聊天内容" * 250 + "末尾才说最近一起玩得很开心"
    review, _ = await prepared_review(affinity, key=key, history=history)
    value = reflection(history)
    value["evidence"][0]["quote"] = "末尾才说最近一起玩得很开心"
    assert await affinity.complete_affinity_review(review, value)


@pytest.mark.asyncio
async def test_expired_request_cannot_save_main_review(monkeypatch, isolated_affinity):
    from nonebot_plugin_ai_groupmate.agent import group_feeling

    affinity = isolated_affinity
    key, history = chat_history()
    note_messages(affinity, key)
    metadata, prompt = await group_feeling.prepare_main_group_review(key[0], key[1], history, "expired-request")
    assert metadata is not None
    assert "完整的人设" in prompt
    monkeypatch.setattr(group_feeling, "is_request_active", AsyncMock(return_value=False))
    await metadata.on_complete(reflection(history))
    assert affinity.affinity_multiplier(key) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("feeling", "one_speaker", "valid"), [
    ("uncomfortable", False, True), ("uncomfortable", True, False), ("neutral", True, True),
])
async def test_negative_feeling_requires_different_members(isolated_affinity, feeling, one_speaker, valid):
    affinity = isolated_affinity
    key, history = chat_history(one_speaker=one_speaker)
    review, _ = await prepared_review(affinity, key=key, history=history)
    assert await affinity.complete_affinity_review(review, reflection(history, feeling)) is valid
    assert (affinity.affinity_multiplier(key) < 1) is (valid and feeling == "uncomfortable")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unknown_id", "invented_quote", "bot_quote", "one_quote", "duplicate_quote", "invalid_feeling", "missing"])
async def test_bad_affinity_does_not_change_reply_or_trigger_retry(isolated_affinity, invalid):

    affinity = isolated_affinity
    key, history = chat_history()
    if invalid == "bot_quote":
        history[0].content_type = "bot"
    review, _ = await prepared_review(affinity, key=key, history=history)
    value = reflection(history)
    if invalid == "unknown_id":
        value["evidence"][0]["message_id"] += 999999
    elif invalid == "invented_quote":
        value["evidence"][0]["quote"] = "聊天中不存在的原话"
    elif invalid == "one_quote":
        value["evidence"] = value["evidence"][:1]
    elif invalid == "duplicate_quote":
        value["evidence"] = [value["evidence"][0], value["evidence"][0]]
    elif invalid == "invalid_feeling":
        value["feeling"] = "love"
    model, result, sent = await run_main_agent(review, None if invalid == "missing" else value)
    assert sent == ["这首歌我也喜欢"]
    assert affinity.affinity_multiplier(key) == 1
    assert len(model.calls) == 1
    assert result["llm_call_count"] == 1
    assert affinity._review_due[key] > review.reserved_until


@pytest.mark.asyncio
async def test_unchanged_feelings_back_off_then_new_experience_restores_short_interval(isolated_affinity):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GroupAffinity

    affinity = isolated_affinity
    now = utcnow()
    key, _ = chat_history()
    for index, expected in enumerate((1800, 3600, 7200, 7200)):
        _, history = chat_history(start=1 + index * 12, group=key[1])
        review, _ = await prepared_review(affinity, key=key, history=history, now=now)
        assert await affinity.complete_affinity_review(review, reflection(history, "neutral"), now=now)
        async with get_session() as db:
            row = await db.get(GroupAffinity, key)
            assert row is not None
            assert row.review_interval == expected
        now = affinity._review_due[key]
    _, history = chat_history(start=49, group=key[1])
    review, _ = await prepared_review(affinity, key=key, history=history, now=now)
    assert await affinity.complete_affinity_review(review, reflection(history), now=now)
    assert affinity._review_due[key] == now + timedelta(minutes=15)


@pytest.mark.asyncio
async def test_explicit_interaction_can_shorten_stable_interval_but_keeps_minimum(isolated_affinity):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GroupAffinity

    affinity = isolated_affinity
    key, history = chat_history()
    now = utcnow()
    async with get_session() as db:
        db.add(GroupAffinity(bot_id=key[0], session_id=key[1], updated_at=now, review_interval=7200, review_after=now + timedelta(hours=2)))
        await db.commit()
        await affinity.load_group_affinities(db)
    note_messages(affinity, key, interaction=True)
    assert await affinity.prepare_affinity_review(key, history, now=now + timedelta(minutes=14)) is None
    assert await affinity.prepare_affinity_review(key, history, now=now + timedelta(minutes=15)) is not None


@pytest.mark.asyncio
async def test_schedule_survives_restart_and_duplicate_decisions_are_reserved_once(isolated_affinity):
    from nonebot_plugin_orm import get_session

    affinity = isolated_affinity
    now = utcnow()
    review, history = await prepared_review(affinity, now=now)
    note_messages(affinity, review.key)
    assert await affinity.prepare_affinity_review(review.key, history, now=now) is None
    # Simulate process cache loss; the persisted reservation still enforces the interval.
    affinity._review_due.clear()
    async with get_session() as db:
        await affinity.load_group_affinities(db)
    assert await affinity.prepare_affinity_review(review.key, history, now=now) is None
    assert await affinity.complete_affinity_review(review, reflection(history), now=now)
    assert not await affinity.complete_affinity_review(review, reflection(history), now=now)
    assert affinity._feelings[review.key].score == pytest.approx(log2(1.5))


@pytest.mark.asyncio
async def test_activity_and_time_alone_never_call_models(monkeypatch, isolated_affinity):
    from nonebot_plugin_ai_groupmate import agent

    affinity = isolated_affinity
    model = AsyncMock()
    monkeypatch.setattr(agent, "get_flash_model", lambda: model)
    key, history = chat_history()
    for _ in range(11):
        affinity.note_group_activity(key)
    assert await affinity.prepare_affinity_review(key, history) is None
    affinity.note_group_activity(key)
    review = await affinity.prepare_affinity_review(key, history)
    assert review is not None
    # Preparation only reserves an inline field; it does not invoke a model.
    model.ainvoke.assert_not_awaited()
    for index in range(2100):
        affinity.note_group_activity(("bot", str(index)))
    assert len(affinity._activity) == 2048


@pytest.mark.asyncio
async def test_model_or_storage_failure_does_not_retry_or_erase_old_feeling(monkeypatch, isolated_affinity):
    from nonebot_plugin_ai_groupmate.agent.graph import TerminalMetadata

    affinity = isolated_affinity
    review, history = await prepared_review(affinity)
    previous = affinity.GroupFeeling(0.5, "聊得来", utcnow() - timedelta(hours=1))
    affinity._feelings[review.key] = previous
    metadata = TerminalMetadata("group_feeling", affinity.reflection_schema(), AsyncMock(side_effect=RuntimeError("storage unavailable")))
    model, result, sent = await run_main_agent(review, reflection(history), metadata=metadata)
    assert sent == ["这首歌我也喜欢"]
    assert len(model.calls) == 1
    assert result["llm_call_count"] == 1
    assert affinity._feelings[review.key] is previous


def test_affinity_is_bounded_scoped_and_old_impressions_fade(isolated_affinity):
    affinity = isolated_affinity
    now = utcnow()
    key = ("bot", "group")
    for score, expected in ((affinity.MAX_SCORE, 5), (affinity.MIN_SCORE, 0.1), (1, 2), (-1, 0.5), (0, 1)):
        affinity._feelings[key] = affinity.GroupFeeling(score, "旧印象", now)
        assert affinity.affinity_multiplier(key, now=now) == pytest.approx(expected)
        assert affinity._feelings[key].current_score(now + timedelta(days=30)) == score / 2
        assert affinity.affinity_multiplier(key, now=now + timedelta(days=3650)) == pytest.approx(1)
    assert affinity.affinity_multiplier(("other-bot", "group")) == 1
    assert affinity.affinity_multiplier(("bot", "other-group")) == 1


def test_feeling_changes_opportunities_but_keeps_disable_cooldown_and_caps(isolated_affinity):
    from nonebot_plugin_ai_groupmate.participation import ParticipationPolicy

    affinity = isolated_affinity
    now = [1000.0]
    sample = [0.015]
    policy = ParticipationPolicy(lambda: now[0], random_sample=lambda: sample[0])
    liked, neutral, disliked = ("bot", "liked"), ("bot", "neutral"), ("bot", "disliked")
    affinity._feelings[liked] = affinity.GroupFeeling(1, "很投机", utcnow())
    affinity._feelings[disliked] = affinity.GroupFeeling(-1, "不太自在", utcnow())
    assert policy.candidate(liked, 0.01)  # This group's current preference is 2%.
    assert not policy.candidate(neutral, 0.01)
    assert not policy.candidate(disliked, 0.01)
    sample[0] = 0.0075
    assert policy.candidate(neutral, 0.01)
    assert not policy.candidate(disliked, 0.01)
    sample[0] = 0.004
    assert policy.candidate(disliked, 0.01)
    assert not policy.candidate(liked, 0)
    policy.sent(liked, proactive=True)
    assert not policy.can_reply(liked)
    for _ in range(2):
        now[0] += 50
        policy.sent(liked, proactive=True)
    now[0] += 50
    assert not policy.can_reply(liked)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [0.001, 0.003, 0.005, 0.008, 0.013, 0.018, 0.02, 0.03, 0.05])
async def test_bot_can_choose_exact_probability_without_a_fixed_step(isolated_affinity, target):
    affinity = isolated_affinity
    now = utcnow()
    review, history = await prepared_review(affinity, now=now)
    value = reflection(history)
    value["desired_reply_probability"] = target
    assert await affinity.complete_affinity_review(review, value, now=now)
    assert 0.01 * affinity.affinity_multiplier(review.key, now=now) == pytest.approx(target)
    schema = review.schema()["properties"]["desired_reply_probability"]["anyOf"][0]
    assert schema["minimum"] == 0.001
    assert schema["maximum"] == 0.05
    assert "没有固定加减步长" in review.prompt()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [0, 0.0009, 0.0501, 0.1, -1, "0.015", True, float("nan"), float("inf")])
async def test_invalid_chosen_probability_preserves_current_preference(isolated_affinity, target):
    affinity = isolated_affinity
    review, history = await prepared_review(affinity)
    value = reflection(history)
    value["desired_reply_probability"] = target
    assert not await affinity.complete_affinity_review(review, value)
    assert affinity.affinity_multiplier(review.key) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("feeling", "one_speaker", "target"), [("neutral", False, 0.015), ("enjoying", True, 0.008)])
async def test_probability_changes_require_evidence_and_decreases_need_multiple_members(isolated_affinity, feeling, one_speaker, target):
    affinity = isolated_affinity
    key, history = chat_history(one_speaker=one_speaker)
    review, _ = await prepared_review(affinity, key=key, history=history)
    value = reflection(history, feeling)
    value["desired_reply_probability"] = target
    if feeling == "neutral":
        value["evidence"] = []
    assert not await affinity.complete_affinity_review(review, value)
    assert affinity.affinity_multiplier(key) == 1


@pytest.mark.asyncio
async def test_manual_base_is_authoritative_and_disabled_groups_cannot_be_adjusted(monkeypatch, isolated_affinity):
    affinity = isolated_affinity
    base = [0.04]
    monkeypatch.setattr(affinity, "resolve_group_reply_probability", lambda *_: base[0])
    review, history = await prepared_review(affinity)
    assert review.probability_bounds() == (0.004, 0.1)
    value = reflection(history)
    value["desired_reply_probability"] = 0.07
    base[0] = 0.02  # An administrator changed it while the model was thinking.
    assert not await affinity.complete_affinity_review(review, value)
    assert affinity.affinity_multiplier(review.key) == 1
    base[0] = 0
    assert await affinity.prepare_affinity_review(review.key, history) is None


@pytest.mark.asyncio
async def test_omitted_probability_does_not_apply_a_hidden_fixed_increment(isolated_affinity):
    affinity = isolated_affinity
    review, history = await prepared_review(affinity)
    value = reflection(history)
    value.pop("desired_reply_probability")
    assert await affinity.complete_affinity_review(review, value)
    assert affinity.affinity_multiplier(review.key) == 1


def test_affinity_migrations_round_trip_and_preserve_existing_impression(monkeypatch):
    from importlib import import_module

    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    initial = import_module("nonebot_plugin_ai_groupmate.migrations.c9a2e7b4d610_add_group_affinity")
    schedule = import_module("nonebot_plugin_ai_groupmate.migrations.d6b9a3e2c710_add_affinity_review_schedule")
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as conn:
            operations = Operations(MigrationContext.configure(conn))
            monkeypatch.setattr(initial, "op", operations)
            monkeypatch.setattr(schedule, "op", operations)
            initial.upgrade()
            table = "nonebot_plugin_ai_groupmate_groupaffinity"
            conn.execute(text(f"INSERT INTO {table} (bot_id,session_id,score,impression,evidence,last_message_id,updated_at) VALUES ('b','g',0.5,'old','[]',12,'2026-10-20 00:00:00')"))
            schedule.upgrade()
            assert set(inspect(conn).get_pk_constraint(table)["constrained_columns"]) == {"bot_id", "session_id"}
            row = conn.execute(text(f"SELECT score, review_after, review_interval FROM {table}")).one()
            assert tuple(row) == (0.5, None, 900)
            schedule.downgrade()
            assert "review_after" not in {column["name"] for column in inspect(conn).get_columns(table)}
            initial.downgrade()
            assert table not in inspect(conn).get_table_names()
    finally:
        engine.dispose()
