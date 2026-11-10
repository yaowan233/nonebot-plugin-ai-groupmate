import json
from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_active_scene_observes_once_and_shares_only_relevant_scoped_facts():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.observations import record_message, relevant_context, poll_observations, prune_observations, record_interaction

    now = datetime(2026, 11, 10, 12)
    group_a, group_b, group_c = uuid4().hex, uuid4().hex, uuid4().hex
    bot_id = uuid4().hex
    spaces = {"friends": [group_a, group_b]}
    async with get_session() as db:
        for index in range(2):
            assert await record_interaction(db, platform="qq", bot_id=bot_id, scene_id=group_a,
                                            turn_id=f"turn-{index}", explicitly_addressed=index == 0,
                                            now=now - timedelta(minutes=10 - index))
        for index in range(10):
            assert await record_message(db, platform="qq", bot_id=bot_id, scene_id=group_a,
                                        message_id=f"msg-{index}", user_id="alice", text="周末郊游八点出发",
                                        now=now - timedelta(minutes=1))
        assert not await record_message(db, platform="qq", bot_id=bot_id, scene_id=group_a,
                                        message_id="msg-0", user_id="alice", text="重复", now=now)
        await db.commit()

    async def observe(messages):
        rows = json.loads(messages[1].content)
        return SimpleNamespace(content=json.dumps([
            {"fact": "周末郊游八点出发", "source_ids": [rows[0]["id"]]},
            {"fact": "郊游的虚构细节", "source_ids": [999999999]},
        ], ensure_ascii=False))

    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=observe))
    kwargs = {"spaces": spaces, "connected": {bot_id: "qq"}, "paused": lambda *_: False, "model_factory": lambda: model, "now": now}
    assert await poll_observations(**kwargs)
    assert not await poll_observations(**kwargs)
    assert model.ainvoke.await_count == 1
    async with get_session() as db:
        context = await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id, scene_id=group_b, query="郊游几点出发", now=now)
        assert "周末郊游八点出发" in context
        assert "虚构细节" not in context
        assert group_a in context
        assert not await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id, scene_id=group_c, query="郊游", now=now)
        assert not await relevant_context(db, spaces=spaces, platform="other", bot_id=bot_id, scene_id=group_b, query="郊游", now=now)
        assert not await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id, scene_id=group_b, query="计算圆周率", now=now)
        assert not await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id, scene_id=group_b, query="郊游", now=now + timedelta(days=2))
        await prune_observations(db, now=now + timedelta(days=2))
        await db.commit()
        assert not await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id, scene_id=group_b, query="郊游", now=now)


@pytest.mark.asyncio
async def test_unrelated_or_ambiguous_groups_never_start_observation():
    from nonebot_plugin_ai_groupmate.brain.observations import configured_space, poll_observations

    assert configured_space({"one": ["a", "b"], "two": ["a", "c"]}, "a") is None
    model = SimpleNamespace(ainvoke=AsyncMock())
    assert not await poll_observations(spaces={}, connected={}, paused=lambda *_: False,
                                       model_factory=lambda: model, now=datetime(2026, 11, 10, 12))
    model.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_reply_handler_records_one_interaction(monkeypatch):
    from sqlalchemy import select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import SceneType

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory, BrainInteraction
    from nonebot_plugin_ai_groupmate.participation import record_group_delivery

    group, peer = uuid4().hex, uuid4().hex
    monkeypatch.setattr(plugin.plugin_config, "cross_group_spaces", {"test": [group, peer]})
    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit_enabled", False)
    monkeypatch.setattr(plugin, "adapter_name", lambda _bot: "test-platform")
    monkeypatch.setattr(plugin, "get_bots", lambda: {})
    async with get_session() as db:
        db.add(ChatHistory(session_id=group, user_id="alice", user_name="Alice", content_type="text", content="id: 100\n你好"))
        await db.commit()

    async def execute(*_args, **_kwargs):
        record_group_delivery(group)
        record_group_delivery(group)
        return SimpleNamespace(need_reply=False, text=None)

    monkeypatch.setattr(plugin, "choice_response_strategy", execute)
    await plugin.handle_reply_logic(
        "request-1", SimpleNamespace(scene=SimpleNamespace(id=group, type=SceneType.GROUP), self_id="bot"),  # pyright: ignore[reportArgumentType]
        SimpleNamespace(get_members=AsyncMock(return_value=[])), SimpleNamespace(self_id="bot"),  # pyright: ignore[reportArgumentType]
        SimpleNamespace(message_id="100", get_plaintext=lambda: "你好"), "bot", "alice", "Alice", True, False, None,  # pyright: ignore[reportArgumentType]
    )
    async with get_session() as db:
        rows = list((await db.scalars(select(BrainInteraction).where(BrainInteraction.scene_id == group))).all())
        assert len(rows) == 1
        assert rows[0].turn_id == "request-1"
        assert rows[0].explicitly_addressed
