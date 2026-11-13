import json
from types import SimpleNamespace
from datetime import datetime

from PIL import Image


async def test_cross_group_meme_scope_and_visual_review(tmp_path):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.meme_tools import _scope_group_meme_candidates
    spaces = {"linked": ["scope-target", "scope-peer"]}
    async with get_session() as db:
        images = []
        for index in range(4):
            name = f"scope-{index}.png"
            Image.new("RGB", (30, 30), (index * 40, 20, 20)).save(tmp_path / name)
            images.append(MediaStorage(file_hash=f"scope-{datetime.now().timestamp()}-{index}", file_path=name, description="meme", references=1))
        db.add_all(images)
        await db.flush()
        ids = [image.media_id for image in images]
        for group, mid in zip(["scope-target", "scope-peer", "scope-peer", "scope-outside"], ids):
            db.add(ChatHistory(session_id=group, user_id="user", user_name="user", content_type="image", content="image", media_id=mid))
        await db.commit()
        class Model:
            async def ainvoke(self, messages, **kwargs):
                assert any(block.get("type") == "image_url" for block in messages[1].content)
                # A hallucinated outside ID cannot bypass the linked-group filter.
                return SimpleNamespace(content=json.dumps({"approved": [ids[1], ids[3]]}))
        candidates = [(mid, "meme") for mid in ids]
        result = await _scope_group_meme_candidates(db, "scope-target", candidates, spaces, tmp_path, Model)
        assert [mid for mid, _ in result] == ids[:2]
        assert await _scope_group_meme_candidates(db, "scope-target", candidates, spaces, tmp_path, lambda: None) == candidates[:1]
        class Broken:
            async def ainvoke(self, *_args, **_kwargs):
                raise RuntimeError("unavailable")
        assert await _scope_group_meme_candidates(db, "scope-target", candidates, spaces, tmp_path, Broken) == candidates[:1]
        assert await _scope_group_meme_candidates(db, "scope-target", candidates, {}, tmp_path, Model) == candidates[:1]

async def test_context_search_only_offers_approved_relevant_linked_memes(tmp_path, monkeypatch):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import meme_tools
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage

    Image.new("RGB", (40, 40), "red").save(tmp_path / "meme.png")
    async with get_session() as db:
        meme = MediaStorage(file_hash=f"pipeline-{datetime.now().timestamp()}", file_path="meme.png", description="笑脸", references=1)
        db.add(meme)
        await db.flush()
        mid = meme.media_id
        db.add(ChatHistory(session_id="pipeline-peer", user_id="u", user_name="u", content_type="image", content="image", media_id=mid))
        await db.commit()
        async def search(*_args, **_kwargs):
            return [(mid, 0.9)]
        async def prepare(*_args, **_kwargs):
            return [(mid, "笑脸")], {}, set()
        async def recent(*_args):
            return set()
        monkeypatch.setattr(meme_tools.DB, "search_meme_candidates", search)
        monkeypatch.setattr(meme_tools, "_prepare_meme_context_review", prepare)
        monkeypatch.setattr(meme_tools, "_get_recent_sent_meme_ids", recent)
        class Model:
            calls = 0
            async def ainvoke(self, messages, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"approved": [mid]}))
                return SimpleNamespace(content=json.dumps({"should_send": True, "candidates": [{"pic_id": mid, "relevance": 0.9}]}))
        model = Model()
        approved = set()
        tool = meme_tools.create_search_meme_tool(
            db, "pipeline-target", None, history=[], approved_meme_ids=approved, pic_dir=tmp_path,
            vision_model_factory=lambda: model, cross_group_spaces={"space": ["pipeline-target", "pipeline-peer"]}, context_only=True,
        )
        result = json.loads(await tool.ainvoke({"description": "开心", "match_type": "random"}))
        assert result["status"] == "succeeded"
        assert result["data"]["match_type"] == "context"
        assert approved == {mid}
        assert model.calls == 2
        assert "pipeline-peer" not in json.dumps(result)
