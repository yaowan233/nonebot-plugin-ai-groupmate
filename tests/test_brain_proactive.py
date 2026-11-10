import json
from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest


def test_generic_summary_word_does_not_make_unrelated_topics_match():
    from nonebot_plugin_ai_groupmate.brain.proactive import _related

    assert not _related("有用户表示找不到名为 ; 5291494 的玩家，可能是输入错误。", "用户正在讨论角色图像创作")


@pytest.mark.asyncio
async def test_proactive_uses_current_target_messages_instead_of_stale_summary():
    from sqlalchemy import select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainSceneObservation, BrainObservationMessage
    from nonebot_plugin_ai_groupmate.brain.proactive import poll_proactive
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 14)
    source, target, bot = [uuid4().hex for _ in range(3)]
    async with get_session() as db:
        await record_message(db, platform="qq", bot_id=bot, scene_id=source, message_id="s", user_id="a",
                             text="查不到这位玩家", now=now)
        await record_message(db, platform="qq", bot_id=bot, scene_id=target, message_id="t", user_id="b",
                             text="什么是114514", now=now)
        await db.flush()
        rows = list((await db.scalars(select(BrainObservationMessage).where(BrainObservationMessage.bot_id == bot))).all())
        for row in rows:
            fact = "有用户表示找不到玩家" if row.scene_id == source else "用户讨论玩家的话题"
            db.add(BrainSceneObservation(platform="qq", bot_id=bot, scene_id=row.scene_id, last_message_id=row.id,
                next_due=now, active=True, digest=json.dumps([{"fact": fact, "source_ids": [row.id]}], ensure_ascii=False),
                source_ids=[row.id], updated_at=now))
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"send":true,"text":"看看玩家"}')))
    sender = AsyncMock(return_value="receipt")
    result = await poll_proactive(spaces=lambda: {"friends": [source, target]}, connected={bot: "qq"},
        allowed=lambda _bot, scene: scene == target, model_factory=lambda: model, sender=sender, now=now)
    assert result == "idle"
    model.ainvoke.assert_not_awaited()
    sender.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_scene", ["source", "target"])
async def test_proactive_does_not_use_stale_source_or_target_messages(stale_scene):
    from sqlalchemy import select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainSceneObservation, BrainObservationMessage
    from nonebot_plugin_ai_groupmate.brain.proactive import poll_proactive
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 16)
    source, target, bot = [uuid4().hex for _ in range(3)]
    async with get_session() as db:
        for scene, text, stale_minutes in ((source, "周末郊游早上八点出发", 61), (target, "周末郊游什么时候出发", 6)):
            time = now - timedelta(minutes=stale_minutes) if scene == (source if stale_scene == "source" else target) else now
            await record_message(db, platform="qq", bot_id=bot, scene_id=scene, message_id=uuid4().hex,
                                 user_id="user", text=text, now=time)
        await db.flush()
        rows = list((await db.scalars(select(BrainObservationMessage).where(BrainObservationMessage.bot_id == bot))).all())
        for row in rows:
            db.add(BrainSceneObservation(platform="qq", bot_id=bot, scene_id=row.scene_id, last_message_id=row.id,
                next_due=now, active=True, digest=json.dumps([{"fact": row.text, "source_ids": [row.id]}], ensure_ascii=False),
                source_ids=[row.id], updated_at=now))
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock())
    sender = AsyncMock()
    result = await poll_proactive(spaces=lambda: {"friends": [source, target]}, connected={bot: "qq"},
        allowed=lambda _bot, scene: scene == target, model_factory=lambda: model, sender=sender, now=now)
    assert result == "idle"
    model.ainvoke.assert_not_awaited()
    sender.assert_not_awaited()


@pytest.mark.asyncio
async def test_proactive_requires_related_backed_facts_and_never_retries_unknown_send():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import BrainProactive, BrainSceneObservation
    from nonebot_plugin_ai_groupmate.brain.proactive import poll_proactive
    from nonebot_plugin_ai_groupmate.brain.observations import record_message

    now = datetime(2026, 11, 10, 12)
    source, target, other, bot = [uuid4().hex for _ in range(4)]
    spaces = {"friends": [source, target], "other": [other, uuid4().hex]}
    async with get_session() as db:
        for scene, text in ((source, "周末郊游早上八点出发"), (target, "周末郊游还没决定几点出发"), (other, "周末郊游谁参加")):
            await record_message(db, platform="qq", bot_id=bot, scene_id=scene, message_id=uuid4().hex,
                                 user_id="user", text=text, now=now)
        await db.flush()
        from sqlalchemy import select

        from nonebot_plugin_ai_groupmate.model import BrainObservationMessage

        rows = list((await db.scalars(select(BrainObservationMessage).where(BrainObservationMessage.bot_id == bot))).all())
        for row in rows:
            db.add(BrainSceneObservation(platform="qq", bot_id=bot, scene_id=row.scene_id, last_message_id=row.id,
                                         next_due=now, active=True, digest=json.dumps([{"fact": row.text, "source_ids": [row.id]}], ensure_ascii=False),
                                         source_ids=[row.id], updated_at=now))
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content='{"send":true,"text":"隔壁群说周末郊游早上八点出发。"}')))
    sender = AsyncMock(side_effect=TimeoutError())
    resolver = AsyncMock(return_value={source: "周末活动群"})
    kwargs = {"spaces": lambda: spaces, "connected": {bot: "qq"}, "allowed": lambda _bot, scene: scene == target,
                  "model_factory": lambda: model, "sender": sender, "now": now, "name_resolver": resolver}
    assert await poll_proactive(**kwargs, source_allowed=lambda *_: False) == "idle"
    model.ainvoke.assert_not_awaited()
    assert await poll_proactive(**kwargs) == "unknown"
    assert sender.await_args is not None
    assert sender.await_args.args[2] == target
    assert "周末郊游早上八点出发" in sender.await_args.args[3]
    assert "来自群" not in sender.await_args.args[3]
    context = json.loads(model.ainvoke.await_args.args[0][1].content)
    assert context["来源群"] == "周末活动群"
    assert context["来源原话"] == ["周末郊游早上八点出发"]
    assert context["目标群刚才的发言"] == ["周末郊游还没决定几点出发"]
    assert await poll_proactive(**kwargs) == "idle"
    sender.assert_awaited_once()
    async with get_session() as db:
        entries = list((await db.scalars(select(BrainProactive).where(BrainProactive.bot_id == bot))).all())
        assert len(entries) == 1
        assert entries[0].status == "unknown"


@pytest.mark.asyncio
async def test_inactive_unrelated_or_unconfigured_groups_are_not_sent():
    from nonebot_plugin_ai_groupmate.brain.proactive import poll_proactive

    model = SimpleNamespace(ainvoke=AsyncMock())
    sender = AsyncMock()
    assert await poll_proactive(spaces=lambda: {}, connected={}, allowed=lambda *_: True,
                                model_factory=lambda: model, sender=sender,
                                now=datetime(2026, 11, 10, 12) + timedelta(days=1)) == "idle"
    model.ainvoke.assert_not_awaited()
    sender.assert_not_awaited()
