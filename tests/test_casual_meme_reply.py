import io
import json
import asyncio
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("image_only", [False, True])
async def test_reply_uses_one_reviewed_image_and_respects_cooldown(monkeypatch, image_only):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    messages = []

    async def capture(message, **kwargs):
        messages.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": str(len(messages))}])

    monkeypatch.setattr(UniMessage, "send", capture)
    group = str(uuid4())
    search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [
        {"pic_id": 4, "context_relevance": 0.8}, {"pic_id": 5, "context_relevance": 0.95},
    ]})))
    send = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("sent", "", delivery_state="completed")))
    async with get_session() as db:
        async def accompaniment(**kwargs):
            return await try_casual_meme(db, group, search, send, visual_verify=AsyncMock(return_value=True), **kwargs)

        reply = create_reply_tool(db, group, bot_name="bot", parse_msg_meta=parse_message_metadata, casual_meme_sender=accompaniment)
        result = json.loads(await reply.ainvoke({"content": "你才笨", "next_step": "end", "meme_intent": "假装生气", "meme_only": image_only}))
        assert result["status"] == "succeeded"
        send.ainvoke.assert_awaited_once_with({"pic_id": "5", "text": "" if image_only else "你才笨", "reply_to": None, "next_step": "end"})
        assert search.ainvoke.await_args.args[0]["reply_text"] == "你才笨"
        assert messages == []
        result = json.loads(await reply.ainvoke({"content": "再来", "next_step": "end", "meme_intent": "开心"}))
        assert result["status"] == "succeeded"
        assert len(messages) == 1
        assert search.ainvoke.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("delivery_state", "expect_fallback"), [("not_attempted", True), ("unknown", False)])
async def test_fallback_only_if_image_delivery_was_not_attempted(monkeypatch, delivery_state, expect_fallback):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_failure, tool_success

    messages = []

    async def capture(message, **kwargs):
        messages.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "1"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [{"pic_id": 4, "context_relevance": 0.8}]})))
    send = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_failure("failed", "", delivery_state=delivery_state)))
    group = str(uuid4())
    async with get_session() as db:
        async def accompaniment(**kwargs):
            return await try_casual_meme(db, group, search, send, visual_verify=AsyncMock(return_value=True), **kwargs)

        reply = create_reply_tool(db, group, bot_name="bot", parse_msg_meta=parse_message_metadata, casual_meme_sender=accompaniment)
        result = json.loads(await reply.ainvoke({"content": "后备文字", "next_step": "end", "meme_intent": "开心", "meme_only": True}))
        assert bool(messages) is expect_fallback
        assert result["delivery_state"] == ("completed" if expect_fallback else "unknown")


@pytest.mark.asyncio
async def test_low_score_never_sends_image():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [{"pic_id": 4, "context_relevance": 0.69}]})))
    send = SimpleNamespace(ainvoke=AsyncMock())
    async with get_session() as db:
        result = await try_casual_meme(db, str(uuid4()), search, send, intent="无语", text="后备", image_only=True, reply_to=None)
    assert result is None
    send.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_visual_rejection_falls_back_without_sending():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [{"pic_id": 4, "context_relevance": 0.9}]})))
    send = SimpleNamespace(ainvoke=AsyncMock())
    verify = AsyncMock(return_value=False)
    async with get_session() as db:
        result = await try_casual_meme(db, str(uuid4()), search, send, intent="无语", text="后备", image_only=False, reply_to=None, visual_verify=verify)
    assert result is None
    verify.assert_awaited_once()
    send.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsuitable_caption_does_not_reach_delivery():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [{"pic_id": 4, "context_relevance": 0.99, "description": "白兔图片，配文‘来操操’"}]})))
    send = SimpleNamespace(ainvoke=AsyncMock())
    verify = AsyncMock(return_value=True)
    async with get_session() as db:
        result = await try_casual_meme(db, str(uuid4()), search, send, intent="调侃偷牧草", text="等你来偷", image_only=False, reply_to=None, visual_verify=verify)
    assert result is None
    verify.assert_not_awaited()
    send.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_visual_verifier_checks_stills_and_animation_keyframes(tmp_path):
    from PIL import Image

    from nonebot_plugin_ai_groupmate.agent.casual_meme import verify_casual_image

    Image.new("RGB", (10, 10), "red").save(tmp_path / "still.png")
    frames = [Image.new("RGB", (10, 10), color) for color in ("red", "blue")]
    frames[0].save(tmp_path / "animated.gif", save_all=True, append_images=frames[1:], duration=20)
    db = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(file_path="still.png")), commit=AsyncMock())
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"approved":false}')))
    assert not await verify_casual_image(db, tmp_path, lambda: model, pic_id=1, intent="无语", text="后备")
    assert model.ainvoke.await_count == 1
    assert "data:image/" in model.ainvoke.await_args.args[0][1].content[1]["image_url"]["url"]
    model.ainvoke.return_value = SimpleNamespace(content='{"approved":true}')
    assert await verify_casual_image(db, tmp_path, lambda: model, pic_id=1, intent="无语", text="后备")
    db.get.return_value = SimpleNamespace(file_path="animated.gif")
    assert await verify_casual_image(db, tmp_path, lambda: model, pic_id=1, intent="无语", text="后备")
    assert model.ainvoke.await_count == 3
    animation_messages = model.ainvoke.await_args.args[0]
    assert "动图" in animation_messages[0].content
    assert "关键帧" in animation_messages[1].content[0]["text"]
    assert "data:image/" in animation_messages[1].content[1]["image_url"]["url"]


@pytest.mark.asyncio
async def test_incomplete_or_corrupt_animation_falls_back(monkeypatch, tmp_path):
    from nonebot_plugin_ai_groupmate.agent import casual_meme

    path = tmp_path / "animation.gif"
    path.write_bytes(b"bad image")
    db = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(file_path=path.name)), commit=AsyncMock())
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"approved":true}')))
    assert not await casual_meme.verify_casual_image(db, tmp_path, lambda: model, pic_id=1, intent="无语", text="后备")
    monkeypatch.setattr(casual_meme, "prepare_image", lambda *args, **kwargs: SimpleNamespace(animated=True, selected_frames=1))
    assert not await casual_meme.verify_casual_image(db, tmp_path, lambda: model, pic_id=1, intent="无语", text="后备")
    model.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_replies_share_group_cooldown():
    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    group = str(uuid4())
    search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [{"pic_id": 4, "context_relevance": 0.9}]})))

    async def slow_send(args):
        await asyncio.sleep(0.01)
        return tool_success("sent", "", delivery_state="completed")

    send = SimpleNamespace(ainvoke=AsyncMock(side_effect=slow_send))
    db = SimpleNamespace(scalar=AsyncMock(return_value=None), commit=AsyncMock())
    results = await asyncio.gather(*[
        try_casual_meme(db, group, search, send, intent="开心", text="好耶", image_only=False, reply_to=None, visual_verify=AsyncMock(return_value=True))
        for _ in range(2)
    ])
    assert sum(result is not None for result in results) == 1
    assert send.ainvoke.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("animated", [False, True])
async def test_casual_reply_delivers_and_records_image_with_text(monkeypatch, tmp_path, animated):
    from PIL import Image
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.meme_tools import create_send_meme_tool
    from nonebot_plugin_ai_groupmate.message_metadata import parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.casual_meme import try_casual_meme
    from nonebot_plugin_ai_groupmate.agent.reply_tools import create_reply_tool
    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_success

    image = io.BytesIO()
    if animated:
        frames = [Image.new("RGB", (10, 10), color) for color in ("red", "blue")]
        frames[0].save(image, format="GIF", save_all=True, append_images=frames[1:], duration=20)
    else:
        Image.new("RGB", (10, 10), "red").save(image, format="PNG")
    filename = "image.gif" if animated else "image.png"
    (tmp_path / filename).write_bytes(image.getvalue())
    messages = []

    async def capture(message, **kwargs):
        messages.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "123"}])

    monkeypatch.setattr(UniMessage, "send", capture)
    group = str(uuid4())
    async with get_session() as db:
        media = MediaStorage(file_hash=uuid4().hex * 2, file_path=filename, description="委屈")
        db.add(media)
        await db.commit()
        await db.refresh(media)
        search = SimpleNamespace(ainvoke=AsyncMock(return_value=tool_success("found", "", data={"images": [{"pic_id": media.media_id, "context_relevance": 0.9}]})))
        send = create_send_meme_tool(db, group, pic_dir=tmp_path, bot_name="bot", approved_meme_ids={media.media_id})

        async def accompaniment(**kwargs):
            return await try_casual_meme(db, group, search, send, visual_verify=AsyncMock(return_value=True), **kwargs)

        reply = create_reply_tool(db, group, bot_name="bot", parse_msg_meta=parse_message_metadata, casual_meme_sender=accompaniment)
        result = json.loads(await reply.ainvoke({"content": "你才笨", "next_step": "end", "meme_intent": "假装委屈"}))
        assert result["delivery_state"] == "completed"
        assert len(messages) == 1
        assert [part.type for part in messages[0]] == ["text", "image"]
        assert messages[0][1].raw_bytes == image.getvalue()
        await db.flush()
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == group))).scalars().all()
        assert len(rows) == 2
        assert {parse_message_metadata(row.content)[0] for row in rows} == {"123"}


@pytest.mark.asyncio
async def test_casual_search_only_reviews_local_candidates_and_rejects_weak_matches(monkeypatch):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import meme_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage

    group = str(uuid4())
    reviewed = []
    async with get_session() as db:
        local = MediaStorage(file_hash=uuid4().hex * 2, file_path="local.png", description="委屈")
        remote = MediaStorage(file_hash=uuid4().hex * 2, file_path="remote.png", description="委屈")
        db.add_all([local, remote])
        await db.flush()
        local_id, remote_id = local.media_id, remote.media_id
        db.add(ChatHistory(session_id=group, user_id="member", user_name="member", content_type="image", content="old", media_id=local_id))
        await db.commit()

        async def candidates(*args, **kwargs):
            return [(remote_id, 0.99), (local_id, 0.9)]

        async def prepare(db_session, session_id, candidates, recent_ids, **kwargs):
            return [(mid, "委屈") for mid, _ in candidates], {}, set()

        async def review(model, **kwargs):
            reviewed.append(kwargs)
            return [(local_id, 0.69)]

        monkeypatch.setattr(meme_tools.DB, "search_meme_candidates", candidates)
        monkeypatch.setattr(meme_tools, "_prepare_meme_context_review", prepare)
        monkeypatch.setattr(meme_tools, "_rerank_meme_candidates_for_context", review)
        monkeypatch.setattr(meme_tools, "_get_recent_sent_meme_ids", AsyncMock(return_value=set()))
        tool = meme_tools.create_search_meme_tool(db, group, None, model=object(), history=[], casual_reply=True, allow_context_fallback=True)
        result = json.loads(await tool.ainvoke({"description": "假装委屈", "reply_text": "你才笨"}))
        assert reviewed[0]["candidates"] == [(local_id, "委屈")]
        assert reviewed[0]["planned_reply"] == "你才笨"
        assert result["status"] == "failed"
