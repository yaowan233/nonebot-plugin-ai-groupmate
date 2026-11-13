import json
from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest


async def test_public_topic_from_linked_group_can_inform_current_conversation():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.observations import record_message, relevant_context, poll_observations, record_interaction

    now = datetime(2026, 11, 11, 12)
    source, target, bot_id = uuid4().hex, uuid4().hex, uuid4().hex
    spaces = {"friends": [source, target]}
    async with get_session() as db:
        for index in range(2):
            await record_interaction(db, platform="qq", bot_id=bot_id, scene_id=source,
                                     turn_id=f"public-{index}", explicitly_addressed=index == 0, now=now)
        for index in range(10):
            await record_message(db, platform="qq", bot_id=bot_id, scene_id=source,
                                 message_id=f"public-{index}", user_id="alice",
                                 text="osu! 新版本更新，谱面编辑器有新功能", now=now)
        await db.commit()

    async def observe(messages):
        if "审查线索" in messages[0].content:
            return SimpleNamespace(content='{"approved":[0]}')
        rows = json.loads(messages[1].content)
        return SimpleNamespace(content=json.dumps([{"topic": "游戏", "public_hint": "osu! 谱面编辑器更新",
                                                    "keywords": ["osu"], "source_ids": [rows[0]["id"]]}], ensure_ascii=False))

    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=observe))
    assert await poll_observations(spaces=spaces, connected={bot_id: "qq"}, paused=lambda *_: False,
                                   model_factory=lambda: model, now=now)
    async with get_session() as db:
        context = await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id,
                                         scene_id=target, query="最近 osu 有什么更新", now=now)
    assert "osu" in context.lower()
    assert "编辑器" in context
    assert source not in context


async def test_active_linked_group_is_observed_without_prior_bot_interaction():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.observations import record_message, poll_observations

    now = datetime(2026, 11, 11, 12)
    source, target, bot_id = uuid4().hex, uuid4().hex, uuid4().hex
    async with get_session() as db:
        for index in range(10):
            await record_message(db, platform="qq", bot_id=bot_id, scene_id=source,
                                 message_id=f"active-{index}", user_id="alice", text="今天在讨论游戏",
                                 now=now - timedelta(minutes=1))
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="[]")))
    assert await poll_observations(spaces={"friends": [source, target]}, connected={bot_id: "qq"},
                                   paused=lambda *_: False, model_factory=lambda: model, now=now)
    model.ainvoke.assert_awaited_once()


@pytest.mark.parametrize(("original", "hint"), [
    ("我最近在打 osu，病假还没结束", "osu 玩家病假"),
    ("张三的工资涨到两万了", "张三的工资涨到两万了"),
    ("osu 群里有人准备换工作", "osu 群友换工作"),
    ("osu 有个活动，联系电话 13812345678", "osu 的活动电话 13812345678"),
])
def test_private_source_cannot_become_public_inspiration(original, hint):
    from nonebot_plugin_ai_groupmate.brain.observations import _public_hint

    assert _public_hint({"public_hint": hint, "keywords": ["osu"]}, [original]) is None


def test_public_hint_must_have_keyword_grounded_in_source():
    from nonebot_plugin_ai_groupmate.brain.observations import _public_hint

    assert _public_hint({"public_hint": "某游戏更新", "keywords": ["某游戏"]}, ["另一款游戏更新"]) is None


def test_model_copied_public_sentence_is_reduced_to_short_topic_terms():
    from nonebot_plugin_ai_groupmate.brain.observations import _public_hint

    original = "osu! 新版本更新，谱面编辑器有新功能"
    item = {"public_hint": "osu! 新版本更新，谱面编辑器新增功能", "keywords": ["osu!", "新版本", "谱面编辑器", "更新"]}
    hint = _public_hint(item, [original])
    assert hint is not None
    assert hint["hint"] == "osu! / 新版本 / 谱面编辑器 / 更新"
    assert original not in hint["hint"]


def test_short_public_meme_can_be_used_as_a_topic():
    from nonebot_plugin_ai_groupmate.brain.observations import _public_hint

    hint = _public_hint({"public_hint": "我嘞个豆", "keywords": ["我嘞个豆"]}, ["我嘞个豆，太抽象了"])
    assert hint is not None
    assert hint["hint"] == "我嘞个豆"


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
            {"topic": "旅行", "source_ids": [rows[0]["id"]]},
            {"topic": "活动", "source_ids": [999999999]},
            {"topic": "旅行八点出发", "source_ids": [rows[0]["id"]]},
        ], ensure_ascii=False))

    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=observe))
    kwargs = {"spaces": spaces, "connected": {bot_id: "qq"}, "paused": lambda *_: False, "model_factory": lambda: model, "now": now}
    assert await poll_observations(**kwargs)
    assert not await poll_observations(**kwargs)
    assert model.ainvoke.await_count == 1
    async with get_session() as db:
        context = await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot_id, scene_id=group_b, query="郊游几点出发", now=now)
        assert "旅行" in context
        assert "八点" not in context
        assert "虚构细节" not in context
        assert group_a not in context
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

async def test_repeated_text_meme_can_inspire_without_matching_keywords():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.brain.observations import record_message, relevant_context, poll_observations

    now = datetime(2026, 11, 12, 12)
    source, target, other, bot = [uuid4().hex for _ in range(4)]
    spaces = {"friends": [source, target]}
    async with get_session() as db:
        for index in range(10):
            await record_message(db, platform="qq", bot_id=bot, scene_id=source,
                                 message_id=f"meme-{index}", user_id=f"user-{index % 2}",
                                 text="我嘞个豆", now=now)
        await db.commit()
    async def observe(messages):
        if "审查线索" in messages[0].content:
            return SimpleNamespace(content='{"approved":[0]}')
        rows = json.loads(messages[1].content)
        return SimpleNamespace(content=json.dumps([{"topic":"生活", "meme":"我嘞个豆", "usage":"惊讶时感叹",
                                                    "source_ids":[rows[0]["id"], rows[1]["id"]]}], ensure_ascii=False))
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=observe))
    assert await poll_observations(spaces=spaces, connected={bot: "qq"}, paused=lambda *_: False,
                                   model_factory=lambda: model, now=now)
    async with get_session() as db:
        context = await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot, scene_id=target,
                                         query="居然考了满分", now=now)
        assert "我嘞个豆" in context
        assert "惊讶时感叹" in context
        assert "自然使用" in context
        assert source not in context
        assert "user-" not in context
        assert not await relevant_context(db, spaces=spaces, platform="qq", bot_id=bot, scene_id=other,
                                          query="居然考了满分", now=now)


def test_meme_requires_multiple_users_and_cannot_include_private_details():
    from nonebot_plugin_ai_groupmate.brain.observations import _meme_hint

    item = {"meme": "我嘞个豆", "usage": "惊讶时感叹"}
    assert _meme_hint(item, ["我嘞个豆", "我嘞个豆"], ["a", "b"]) == item
    assert _meme_hint(item, ["我嘞个豆", "我嘞个豆"], ["a", "a"]) is None
    assert _meme_hint(item, ["我嘞个豆 群友住院", "我嘞个豆"], ["a", "b"]) is None
    assert _meme_hint({"meme": "张三的工资", "usage": "惊讶"}, ["张三的工资", "张三的工资"], ["a", "b"]) is None


def test_meme_template_preserves_fixed_text_and_replaces_variable_content():
    from nonebot_plugin_ai_groupmate.brain.observations import _meme_hint

    item = {"meme": "今天是疯狂星期四，我在肯德基{内容}，谁请我吃炸鸡", "usage": "开玩笑地邀人吃饭"}
    assert _meme_hint(item, ["今天是疯狂星期四，我在肯德基饿得走不动，谁请我吃炸鸡",
                             "今天是疯狂星期四，我在肯德基看菜单，谁请我吃炸鸡"], ["a", "b"]) == item
    assert _meme_hint(item, ["今天是疯狂星期四，我在肯德基饿得走不动，谁请我吃炸鸡",
                             "今天是疯狂星期四，我在肯德基看菜单，电话13812345678，谁请我吃炸鸡"], ["a", "b"]) is None
    assert _meme_hint(item, ["今天是疯狂星期四，我在肯德基饿得走不动，谁请我吃炸鸡",
                             "完全不相关的内容"], ["a", "b"]) is None
