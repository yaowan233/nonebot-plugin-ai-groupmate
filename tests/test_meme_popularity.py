from __future__ import annotations

import json
import uuid
import datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw


@pytest.mark.asyncio
async def test_group_usage_counts_exclude_other_groups_and_bot_sends():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.meme_tools import (
        _get_group_favorite_memes,
        _get_recent_sent_meme_ids,
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
        recent_ids = await _get_recent_sent_meme_ids(db_session, session_id)
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
    assert recent_ids == {popular_id, occasional_id}
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


def test_content_selection_preserves_match_ranking_without_random_favorite():
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    selected = meme_tools._select_content_meme_ids(
        [(7, 0.95), (8, 0.90), (99, 0.85)],
        {7},
        limit=3,
    )

    assert selected == [8, 99, 7]


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
                    {"pic_id": 3, "relevance": 0.45},
                    {"pic_id": 1, "relevance": 0.39},
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
        candidates=[
            (1, "开心庆祝"),
            (2, "熊猫头累趴下"),
            (3, "泛用疲惫表情"),
        ],
    )

    assert result == [(2, 0.92), (3, 0.45)]
    assert "允许带一点随机性" in captured_messages[0].content
    assert "今天又加班了" in captured_messages[1].content
    assert "id: 123" not in captured_messages[1].content


@pytest.mark.asyncio
async def test_content_reranker_treats_character_quote_and_meme_as_hard_constraints():
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    captured_messages = []

    class FakeReviewer:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return {
                "should_send": True,
                "candidates": [{"pic_id": 2, "relevance": 0.9}],
            }

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is meme_tools.MemeContextReview
            return FakeReviewer()

    result = await meme_tools._rerank_meme_candidates_for_context(
        FakeModel(),
        search_intent="初音未来拿着葱，玩甩葱歌的梗",
        history=[],
        candidates=[(1, "普通开心女孩"), (2, "初音未来拿着葱的甩葱歌梗")],
        match_type="content",
    )

    assert result == [(2, 0.9)]
    assert "硬条件" in captured_messages[0].content
    payload = json.loads(captured_messages[1].content)
    assert payload["search_request"] == "初音未来拿着葱，玩甩葱歌的梗"
    assert payload["match_type"] == "content"


@pytest.mark.asyncio
async def test_explicit_dragon_image_request_expands_jargon_for_search_and_review(
    monkeypatch,
):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    search_queries: list[str] = []
    review_queries: list[str] = []

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return [SimpleNamespace(media_id=1, description="黑白熊猫头表情包")]

    class FakeSession:
        async def commit(self):
            return None

        async def execute(self, statement):
            return FakeResult()

    async def no_recent(*args, **kwargs):
        return set()

    async def fake_search(description, *args, **kwargs):
        search_queries.append(description)
        return [(1, 0.9)]

    async def fake_prepare(*args, **kwargs):
        return [(1, "黑白熊猫头表情包")], {}, set()

    async def fake_review(*args, **kwargs):
        review_queries.append(kwargs["search_intent"])
        return [(1, 0.9)]

    monkeypatch.setattr(meme_tools, "_get_recent_sent_meme_ids", no_recent)
    monkeypatch.setattr(meme_tools.DB, "search_meme_candidates", fake_search)
    monkeypatch.setattr(meme_tools, "_prepare_meme_context_review", fake_prepare)
    monkeypatch.setattr(
        meme_tools,
        "_rerank_meme_candidates_for_context",
        fake_review,
    )

    tool = meme_tools.create_search_meme_tool(
        FakeSession(),
        "group-1",
        None,
        model=object(),
        history=[],
        default_match_type="content",
        explicit_request_text="来张龙图",
    )

    result = json.loads(await tool.ainvoke({
        "description": "龙图",
        "match_type": "content",
    }))

    assert result["success"] is True
    assert search_queries == review_queries
    assert "黑白熊猫头" in search_queries[0]
    assert "不是动物龙" in search_queries[0]


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
async def test_explicit_meme_request_preserves_original_text_in_multimodal_fallback(
    monkeypatch,
):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    search_calls: list[tuple[str, bool]] = []

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return [SimpleNamespace(media_id=1, description="群内常用表情包")]

    class FakeSession:
        async def commit(self):
            return None

        async def execute(self, statement):
            return FakeResult()

    async def no_recent(*args, **kwargs):
        return set()

    async def fake_search(description, *args, **kwargs):
        search_calls.append((description, kwargs["strict_content_match"]))
        return [(1, 0.9)]

    async def fake_prepare(*args, **kwargs):
        return [(1, "群内常用表情包")], {1: 5}, {1}

    async def reject_all(*args, **kwargs):
        return []

    monkeypatch.setattr(meme_tools, "_get_recent_sent_meme_ids", no_recent)
    monkeypatch.setattr(meme_tools.DB, "search_meme_candidates", fake_search)
    monkeypatch.setattr(meme_tools, "_prepare_meme_context_review", fake_prepare)
    monkeypatch.setattr(
        meme_tools,
        "_rerank_meme_candidates_for_context",
        reject_all,
    )

    approved: set[int] = set()
    tool = meme_tools.create_search_meme_tool(
        FakeSession(),
        "group-1",
        None,
        model=object(),
        history=[],
        approved_meme_ids=approved,
        allow_context_fallback=True,
        default_match_type="content",
        explicit_request_text="发一个初音未来拿着葱的表情包",
    )

    result = json.loads(await tool.ainvoke({
        "description": "开心的二次元表情",
        "match_type": "content",
    }))

    assert result["success"] is True
    assert result["match_type"] == "content"
    assert result["images"][0]["pic_id"] == 1
    assert "回退" in result["note"]
    assert approved == {1}
    assert search_calls == [
        (
            "开心的二次元表情\n用户原始找图要求：发一个初音未来拿着葱的表情包",
            True,
        )
    ]


@pytest.mark.asyncio
async def test_search_meme_deduplicates_same_picture_before_model_selection(
    tmp_path,
    monkeypatch,
):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    base_image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(base_image)
    draw.rectangle((8, 8, 55, 60), fill="navy")
    draw.ellipse((65, 15, 110, 65), fill="gold")
    other_image = Image.new("RGB", (120, 80), "black")
    ImageDraw.Draw(other_image).polygon(
        [(10, 70), (60, 5), (110, 70)],
        fill="white",
    )
    base_image.save(tmp_path / "same.png")
    base_image.resize((240, 160)).save(
        tmp_path / "same.jpg",
        quality=75,
    )
    other_image.save(tmp_path / "other.png")
    media = [
        SimpleNamespace(media_id=1, file_path="same.png", description="PNG 版本"),
        SimpleNamespace(media_id=2, file_path="same.jpg", description="JPEG 版本"),
        SimpleNamespace(media_id=3, file_path="other.png", description="另一张图"),
    ]

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return media

    class FakeSession:
        async def execute(self, _statement):
            return FakeResult()

        async def commit(self):
            return None

    async def no_recent(*_args, **_kwargs):
        return set()

    async def fake_search(*_args, **_kwargs):
        return [(1, 0.9), (2, 0.85), (3, 0.8)]

    async def fake_prepare(*_args, **_kwargs):
        return [(1, "PNG 版本"), (2, "JPEG 版本"), (3, "另一张图")], {}, set()

    async def approve_all(*_args, **_kwargs):
        return [(1, 0.95), (2, 0.94), (3, 0.9)]

    monkeypatch.setattr(meme_tools, "_get_recent_sent_meme_ids", no_recent)
    monkeypatch.setattr(meme_tools.DB, "search_meme_candidates", fake_search)
    monkeypatch.setattr(meme_tools, "_prepare_meme_context_review", fake_prepare)
    monkeypatch.setattr(
        meme_tools,
        "_rerank_meme_candidates_for_context",
        approve_all,
    )
    approved: set[int] = set()
    tool = meme_tools.create_search_meme_tool(
        FakeSession(),
        "group-1",
        None,
        model=object(),
        history=[],
        approved_meme_ids=approved,
        default_match_type="content",
        pic_dir=tmp_path,
    )

    result = json.loads(await tool.ainvoke({
        "description": "东方表情包",
        "match_type": "content",
    }))

    assert [image["pic_id"] for image in result["images"]] == [1, 3]
    assert result["count"] == 2
    assert approved == {1, 3}


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


@pytest.mark.asyncio
async def test_send_meme_honors_multi_send_limit_and_uses_distinct_ids(
    tmp_path,
    monkeypatch,
):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    pictures = []
    for pic_id in (1, 2):
        file_path = f"{pic_id}.png"
        (tmp_path / file_path).write_bytes(f"image-{pic_id}".encode())
        pictures.append(SimpleNamespace(
            media_id=pic_id,
            file_path=file_path,
            description=f"表情包 {pic_id}",
        ))

    class FakeResult:
        def __init__(self, picture):
            self.picture = picture

        def scalar(self):
            return self.picture

    class FakeSession:
        def __init__(self):
            self.remaining = iter(pictures)
            self.added = []
            self.commit_count = 0

        async def execute(self, statement):
            return FakeResult(next(self.remaining))

        async def commit(self):
            self.commit_count += 1

        def add(self, record):
            self.added.append(record)

    sends: list[bytes] = []

    class FakeOutgoingMessage:
        def __init__(self, raw: bytes):
            self.raw = raw

        async def send(self, target=None):
            sends.append(self.raw)
            return SimpleNamespace(msg_ids=[{"message_id": len(sends)}])

    monkeypatch.setattr(
        meme_tools.UniMessage,
        "image",
        staticmethod(lambda *, raw: FakeOutgoingMessage(raw)),
    )

    async def no_recent_memes(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(
        meme_tools,
        "_get_recent_sent_meme_ids",
        no_recent_memes,
    )

    session = FakeSession()
    approved = {1, 2, 3}
    send_tool = meme_tools.create_send_meme_tool(
        session,
        "group-1",
        pic_dir=tmp_path,
        bot_name="bot",
        approved_meme_ids=approved,
        max_sends=2,
    )

    first = json.loads(await send_tool.ainvoke({"pic_id": "1"}))
    # 模拟模型再次搜索后把同一候选重新加入审核集合；发送工具仍应硬去重。
    approved.add(1)
    repeated = json.loads(await send_tool.ainvoke({"pic_id": "1"}))
    second = json.loads(await send_tool.ainvoke({"pic_id": "2"}))
    blocked = json.loads(await send_tool.ainvoke({"pic_id": "3"}))

    assert first["status"] == "sent"
    assert repeated["status"] == "skipped"
    assert "已经发送过" in repeated["message"]
    assert second["status"] == "sent"
    assert blocked["status"] == "skipped"
    assert blocked["max_sends"] == 2
    assert sends == [b"image-1", b"image-2"]
    assert approved == {3}
    assert len(session.added) == 2
    assert session.commit_count == 2


@pytest.mark.asyncio
async def test_send_meme_rechecks_images_recently_sent_by_group_members(
    tmp_path,
    monkeypatch,
):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    class FakeSession:
        async def execute(self, _statement):
            raise AssertionError("最近图片已命中时不应继续读取图片记录")

    async def recently_sent(*_args, **_kwargs):
        return {42}

    monkeypatch.setattr(
        meme_tools,
        "_get_recent_sent_meme_ids",
        recently_sent,
    )
    approved = {42}
    send_tool = meme_tools.create_send_meme_tool(
        FakeSession(),
        "group-1",
        pic_dir=tmp_path,
        bot_name="bot",
        approved_meme_ids=approved,
        max_sends=2,
    )

    result = json.loads(await send_tool.ainvoke({"pic_id": "42"}))

    assert result["status"] == "skipped"
    assert "刚刚在群里出现过" in result["message"]
    assert approved == set()


@pytest.mark.asyncio
async def test_send_meme_rejects_same_picture_with_different_file_encoding(
    tmp_path,
    monkeypatch,
):
    from nonebot_plugin_ai_groupmate.agent import meme_tools

    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 55, 60), fill="navy")
    draw.ellipse((65, 15, 110, 65), fill="gold")
    png_buffer = BytesIO()
    jpeg_buffer = BytesIO()
    image.save(png_buffer, format="PNG")
    image.resize((240, 160)).save(jpeg_buffer, format="JPEG", quality=75)
    (tmp_path / "same.png").write_bytes(png_buffer.getvalue())
    (tmp_path / "same.jpg").write_bytes(jpeg_buffer.getvalue())

    pictures = iter([
        SimpleNamespace(
            media_id=1,
            file_path="same.png",
            description="PNG 版本",
        ),
        SimpleNamespace(
            media_id=2,
            file_path="same.jpg",
            description="JPEG 版本",
        ),
    ])

    class FakeResult:
        def __init__(self, picture):
            self.picture = picture

        def scalar(self):
            return self.picture

    class FakeSession:
        def __init__(self):
            self.added = []

        async def execute(self, _statement):
            return FakeResult(next(pictures))

        async def commit(self):
            return None

        def add(self, record):
            self.added.append(record)

    sent_images: list[bytes] = []

    class FakeOutgoingMessage:
        def __init__(self, raw: bytes):
            self.raw = raw

        async def send(self, target=None):
            sent_images.append(self.raw)
            return SimpleNamespace(msg_ids=[{"message_id": len(sent_images)}])

    async def no_recent_memes(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(
        meme_tools,
        "_get_recent_sent_meme_ids",
        no_recent_memes,
    )
    monkeypatch.setattr(
        meme_tools.UniMessage,
        "image",
        staticmethod(lambda *, raw: FakeOutgoingMessage(raw)),
    )
    approved = {1, 2}
    session = FakeSession()
    send_tool = meme_tools.create_send_meme_tool(
        session,
        "group-1",
        pic_dir=tmp_path,
        bot_name="bot",
        approved_meme_ids=approved,
        max_sends=2,
    )

    first = json.loads(await send_tool.ainvoke({"pic_id": "1"}))
    duplicate = json.loads(await send_tool.ainvoke({"pic_id": "2"}))

    assert first["status"] == "sent"
    assert duplicate["status"] == "skipped"
    assert "画面重复" in duplicate["message"]
    assert len(sent_images) == 1
    assert approved == set()
    assert len(session.added) == 1
