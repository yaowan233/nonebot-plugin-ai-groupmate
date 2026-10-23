import json
import base64
from io import BytesIO
from types import SimpleNamespace
from datetime import datetime

from PIL import Image


def test_contact_sheet_uses_real_pixels_and_skips_missing_and_outside_files(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.meme_visuals import build_meme_contact_sheet

    Image.new("RGB", (40, 40), "red").save(tmp_path / "red.png")
    Image.new("RGB", (40, 40), "blue").save(tmp_path / "blue.png")
    data, ids = build_meme_contact_sheet(tmp_path, [(7, "red.png"), (8, "missing.png"), (9, "blue.png"), (10, "../outside.png")])
    assert ids == {7, 9}
    assert data is not None
    with Image.open(BytesIO(base64.b64decode(data.split(",", 1)[1]))) as grid:
        assert grid.size == (768, 384)
        assert grid.getpixel((190, 50)) == (255, 0, 0)
        assert grid.getpixel((574, 50)) == (0, 0, 255)


async def test_visual_rerank_sees_images_and_can_reject_all(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import meme_tools
    from nonebot_plugin_ai_groupmate.agent.meme_visuals import build_meme_contact_sheet

    Image.new("RGB", (40, 40), "red").save(tmp_path / "candidate.png")
    data, _ = build_meme_contact_sheet(tmp_path, [(7, "candidate.png")])
    class Model:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 1200
            content = messages[1].content
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"] == data
            assert "拟发送的文字是安慰" in content[0]["text"]
            assert '"planned_reply": "先歇一会儿吧"' in content[0]["text"]
            return SimpleNamespace(content=json.dumps({"should_send": False, "candidates": [{"pic_id": 7, "relevance": 1.0}]}))

    assert await meme_tools._rerank_meme_candidates_for_context(
        Model(), search_intent="拟发送的文字是安慰", history=[], candidates=[(7, "表面像安慰，实际台词在嘲笑")], candidate_grid=data, planned_reply="先歇一会儿吧",
    ) == []


async def test_search_pipeline_selects_only_visible_candidates(tmp_path, monkeypatch):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import meme_tools
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    Image.new("RGB", (50, 50), "green").save(tmp_path / "visible.png")
    async with get_session() as db:
        rows = [MediaStorage(file_hash=f"visual-{datetime.now().timestamp()}-{i}", file_path=path, description="相同描述", references=3)
                for i, path in enumerate(("visible.png", "missing.png"))]
        db.add_all(rows)
        await db.flush()
        ids = [row.media_id for row in rows]
        await db.commit()

        async def recent(*_args):
            return set()
        async def search(*_args, **_kwargs):
            return [(mid, 0.9) for mid in ids]
        async def prepare(*_args, **_kwargs):
            return [(mid, "相同描述") for mid in ids], {}, set()
        monkeypatch.setattr(meme_tools, "_get_recent_sent_meme_ids", recent)
        monkeypatch.setattr(meme_tools.DB, "search_meme_candidates", search)
        monkeypatch.setattr(meme_tools, "_prepare_meme_context_review", prepare)

        class VisualModel:
            def with_structured_output(self, _schema):
                return self

            async def ainvoke(self, messages, **_kwargs):
                content = messages[1].content
                payload = json.loads(content[0]["text"].split("\n候选原图", 1)[0])
                assert [item["pic_id"] for item in payload["candidates"]] == [ids[0]]
                assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
                # A hallucinated/missing candidate may not receive approval.
                return SimpleNamespace(content=json.dumps({"should_send": True, "candidates": [{"pic_id": mid, "relevance": 0.9} for mid in ids]}))

        approved = set()
        tool = meme_tools.create_search_meme_tool(db, "visual-group", None, history=[], vision_model_factory=VisualModel,
                                                 approved_meme_ids=approved, pic_dir=tmp_path)
        result = json.loads(await tool.ainvoke({"description": "接一句安慰"}))
        assert result["status"] == "succeeded"
        assert approved == {ids[0]}


async def test_deepseek_visual_review_disables_thinking_only_for_this_call(monkeypatch):
    from pydantic import SecretStr
    from langchain_core.messages import AIMessage

    from nonebot_plugin_ai_groupmate.agent import meme_tools
    from nonebot_plugin_ai_groupmate.deepseek import DeepSeekChatOpenAI

    model = DeepSeekChatOpenAI(model="deepseek-flash", api_key=SecretStr("test"), base_url="https://api.deepseek.com",
                              reasoning_effort="high", extra_body={"thinking": {"type": "enabled"}})
    async def invoke(self, _messages, **kwargs):
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert kwargs["reasoning_effort"] == "none"
        assert kwargs["max_tokens"] == 1200
        return AIMessage(content='```json\n{"should_send":false,"candidates":[]}\n```')
    monkeypatch.setattr(DeepSeekChatOpenAI, "ainvoke", invoke)
    metadata = {}
    assert await meme_tools._rerank_meme_candidates_for_context(model, search_intent="安慰", history=[], candidates=[(1, "笑话")],
                                                               candidate_grid="data:image/png;base64,eA==", review_metadata=metadata) == []
    assert metadata["reviewed"] is True
    assert model.reasoning_effort == "high"
    assert model.extra_body == {"thinking": {"type": "enabled"}}
