from __future__ import annotations

import uuid
import datetime

import pytest


@pytest.mark.asyncio
async def test_group_usage_counts_exclude_other_groups_and_bot_sends():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.meme_tools import (
        _get_group_favorite_memes,
        _get_group_meme_usage_counts,
        _prepare_meme_context_review,
    )

    unique = uuid.uuid4().hex
    session_id = f"group-{unique}"
    other_session_id = f"other-{unique}"

    async with get_session() as db_session:
        popular = MediaStorage(
            file_hash=f"popular-{unique}",
            file_path="popular.png",
            references=3,
            description="群内常用表情包",
            vectorized=True,
            embedding_version=3,
        )
        occasional = MediaStorage(
            file_hash=f"occasional-{unique}",
            file_path="occasional.png",
            references=3,
            description="偶尔使用的表情包",
            vectorized=True,
            embedding_version=3,
        )
        db_session.add_all([popular, occasional])
        await db_session.flush()
        popular_id = popular.media_id
        occasional_id = occasional.media_id

        def history(
            media_id: int,
            *,
            target_session: str = session_id,
            content_type: str = "image",
        ) -> ChatHistory:
            return ChatHistory(
                session_id=target_session,
                user_id="bot" if content_type == "bot" else "user",
                content_type=content_type,
                content="image",
                user_name="bot" if content_type == "bot" else "member",
                media_id=media_id,
            )

        db_session.add_all(
            [
                history(popular_id),
                history(popular_id),
                history(popular_id),
                history(occasional_id),
                *[
                    history(occasional_id, target_session=other_session_id)
                    for _ in range(5)
                ],
                *[
                    history(occasional_id, content_type="bot")
                    for _ in range(5)
                ],
            ]
        )
        await db_session.commit()

        counts = await _get_group_meme_usage_counts(
            db_session,
            session_id,
            [popular_id, occasional_id],
        )
        favorites = await _get_group_favorite_memes(db_session, session_id)
        review_candidates, review_counts, favorite_ids = (
            await _prepare_meme_context_review(
                db_session,
                session_id,
                [(occasional_id, 0.9), (popular_id, 0.8)],
                set(),
            )
        )

    assert counts == {
        popular_id: 3,
        occasional_id: 1,
    }
    assert favorites == [(popular_id, 3)]
    assert {media_id for media_id, _ in review_candidates} == {
        popular_id,
        occasional_id,
    }
    assert review_counts == counts
    assert favorite_ids == {popular_id}


def test_group_aware_selection_reserves_one_approved_favorite_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    from nonebot_plugin_ai_groupmate import memory
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    monkeypatch.setattr(
        memory.random,
        "choices",
        lambda population, **_: [population[0]],
    )
    monkeypatch.setattr(meme_tools.random, "randrange", lambda _: 1)

    selected = meme_tools._select_group_aware_meme_ids(
        [(1, 0.9), (2, 0.8), (99, 0.75), (3, 0.7), (4, 0.6)],
        {2: 4, 99: 8},
        {99},
        set(),
    )

    assert selected == [2, 99, 1, 3, 4]


def test_group_favorite_cannot_bypass_context_review():
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    selected = meme_tools._select_group_aware_meme_ids(
        [(1, 0.9)],
        {99: 20},
        {99},
        set(),
    )

    assert selected == [1]


@pytest.mark.asyncio
async def test_context_reranker_filters_low_scores_and_unknown_ids():
    from nonebot_plugin_ai_groupmate.agent import meme_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    captured_messages = []

    class FakeReviewer:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return {
                "should_send": True,
                "candidates": [
                    {"pic_id": 2, "relevance": 0.92},
                    {"pic_id": 1, "relevance": 0.59},
                    {"pic_id": 999, "relevance": 1.0},
                ],
            }

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is meme_tools.MemeContextReview
            return FakeReviewer()

    history = [
        ChatHistorySchema(
            msg_id=1,
            session_id="group-1",
            user_id="user-1",
            content_type="text",
            content="id: 123\n今天又加班了",
            created_at=datetime.datetime.now(),
            user_name="Alice",
        )
    ]
    result = await meme_tools._rerank_meme_candidates_for_context(
        FakeModel(),
        search_intent="表达累到崩溃",
        history=history,
        candidates=[(1, "开心庆祝"), (2, "熊猫头累趴下")],
    )

    assert result == [(2, 0.92)]
    assert "今天又加班了" in captured_messages[1].content
    assert "id: 123" not in captured_messages[1].content


@pytest.mark.asyncio
async def test_context_reranker_fails_closed():
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    class BrokenModel:
        def with_structured_output(self, schema):
            raise RuntimeError("unavailable")

    result = await meme_tools._rerank_meme_candidates_for_context(
        BrokenModel(),
        search_intent="震惊",
        history=[],
        candidates=[(1, "震惊表情")],
    )

    assert result == []


@pytest.mark.asyncio
async def test_send_meme_rejects_an_id_outside_current_approval(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    send_tool = meme_tools.create_send_meme_tool(
        object(),
        "group-1",
        pic_dir=tmp_path,
        bot_name="bot",
        approved_meme_ids={1},
    )

    result = await send_tool.ainvoke({"pic_id": "42"})

    assert "未通过本轮搜索审核" in result
