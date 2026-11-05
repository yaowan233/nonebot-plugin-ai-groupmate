import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest


def message(number, user, text, **overrides):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema
    values = {"msg_id": number, "session_id": "group", "user_id": user, "user_name": user, "content_type": "text",
              "content": f"id: {number}\n{text}", "created_at": datetime(2026, 11, 1, 12) + timedelta(seconds=number)}
    return ChatHistorySchema(**{**values, **overrides})


def corpus():
    return [message(1, "a", "我们群说蓝票，指的是预约空闲机位的凭证"), message(2, "b", "蓝票就是机位预约凭证，我刚领了一张"),
            *[message(i, "a" if i % 2 else "b", f"刚才一起聊的普通话题{i}") for i in range(3, 13)]]


def extraction(**overrides):
    from nonebot_plugin_ai_groupmate.agent.jargon_learning import JargonExtraction
    return JargonExtraction.model_validate({"terms": [{"term": "蓝票", "meaning": "预约空闲机位的凭证", "source_ids": [1, 2], "confidence": "supported", **overrides}]})


class Model:
    def __init__(self, response=None):
        self.response = response if response is not None else extraction()
        self.calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_sources_exclude_bots_headers_other_groups_cards_and_commands():
    from nonebot_plugin_ai_groupmate.agent.jargon_learning import collect_sources
    rows = [message(1, "a", "蓝票是预约用的"), message(2, "b", "另一个群", session_id="other"),
            message(3, "connected", "蓝票就是错的"), message(4, "echo", "回声", content_type="bot"),
            message(5, "bot-alias", "回声", content="id: 4\n回声"), message(6, "a", "/命令"),
            message(7, "a", "【分享卡片文字】卡片"), message(8, "a", "原图", content_type="image")]
    sources = collect_sources(rows, "group", ("connected",))
    assert [source.msg_id for source in sources] == [1]
    assert sources[0].text == "蓝票是预约用的"


@pytest.mark.parametrize(("term", "text", "matches"), [("AP", "happy", False), ("AP", "cap", False), ("AP", "这把ＡＰ了", True),
                                                               ("蓝票", "蓝票还剩几张", True), ("蓝票", "机位还剩几个", False)])
def test_term_matching_respects_latin_boundaries_and_unicode(term, text, matches):
    from nonebot_plugin_ai_groupmate.agent.jargon_learning import contains_term
    assert contains_term(text, term) is matches


@pytest.mark.parametrize("change", [{"term": "凭空出现的词"}, {"source_ids": [1, 999]}, {"source_ids": [1, 3]}])
def test_hallucinated_terms_and_false_source_references_are_rejected(change):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    draft = extraction().terms[0].model_dump()
    draft.update(change)
    learned = module.JargonExtraction.model_validate({"terms": [draft]})
    bank = module.merge_extraction(module.JargonBank(session_id="group"), learned, module.collect_sources(corpus(), "group"), now=datetime.now())
    assert bank.entries == []


async def test_single_speaker_and_uncertain_meanings_are_not_injected(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    for rows, learned in [(corpus(), extraction(confidence="uncertain")),
                          ([row.model_copy(update={"user_id": "a"}) for row in corpus()], extraction())]:
        bank = module.merge_extraction(module.JargonBank(session_id="group"), learned, module.collect_sources(rows, "group"), now=datetime.now())
        assert bank.entries[0].status == "candidate"
        module._write_bank(module.bank_path(tmp_path, "group"), bank)
        assert await module.get_jargon_context(tmp_path, "group", "我想领蓝票") == ""


async def test_single_character_jargon_can_be_learned_from_supported_group_usage(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    rows = [message(1, "a", "寄在我们群指的是提前休息"), message(2, "b", "我要寄就是打算提前休息了")]
    bank = module.merge_extraction(module.JargonBank(session_id="group"), extraction(term="寄", meaning="提前休息"),
                                   module.collect_sources(rows, "group"), now=datetime.now(), reviewed=True)
    module._write_bank(module.bank_path(tmp_path, "group"), bank)
    assert "提前休息" in await module.get_jargon_context(tmp_path, "group", "我先寄一会儿")


async def test_learning_batches_discovery_and_review_without_lookup_model_calls(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    model = Model()
    now = datetime.now()
    assert await module.learn_jargon(tmp_path, "group", corpus(), lambda: model, now=now)
    context = await module.get_jargon_context(tmp_path, "group", "蓝票还能领吗")
    assert "预约空闲机位" in context
    assert "聊天推测" in context
    assert await module.get_jargon_context(tmp_path, "other", "蓝票还能领吗") == ""
    assert await module.get_jargon_context(tmp_path, "group", "晚上吃什么") == ""
    assert not await module.learn_jargon(tmp_path, "group", corpus(), lambda: model, now=now + timedelta(hours=1))
    assert model.calls == 2
    bank = module._read_bank(module.bank_path(tmp_path, "group"), "group")
    assert bank.last_source_id == 12
    assert [source.msg_id for source in bank.entries[0].sources] == [1, 2]
    assert bank.entries[0].sources[0].user_id == "a"


async def test_failed_learning_keeps_material_for_retry_and_persists_cooldown(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    now = datetime.now()
    failed = Model(RuntimeError("provider failed"))
    with pytest.raises(RuntimeError):
        await module.learn_jargon(tmp_path, "group", corpus(), lambda: failed, now=now)
    bank = module._read_bank(module.bank_path(tmp_path, "group"), "group")
    assert bank.last_source_id == 0
    good = Model()
    assert not await module.learn_jargon(tmp_path, "group", corpus(), lambda: good, now=now + timedelta(minutes=1))
    assert good.calls == 0
    assert await module.learn_jargon(tmp_path, "group", corpus(), lambda: good, now=now + timedelta(minutes=31))
    assert good.calls == 2


async def test_new_material_accumulates_until_threshold(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    model = Model()
    assert not await module.learn_jargon(tmp_path, "group", corpus()[:6], lambda: model)
    assert model.calls == 0
    assert await module.learn_jargon(tmp_path, "group", corpus(), lambda: model)


async def test_manual_correction_wins_even_if_automatic_learning_is_in_flight(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    entered = asyncio.Event()
    release = asyncio.Event()
    class DelayedModel(Model):
        async def ainvoke(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return extraction()
    task = asyncio.create_task(module.learn_jargon(tmp_path, "group", corpus(), DelayedModel))
    await asyncio.wait_for(entered.wait(), timeout=2)
    try:
        await module.correct_jargon(tmp_path, "group", "蓝票", "本周组队的暗号", user_id="owner", source_quote="蓝票是本周组队的暗号")
    finally:
        release.set()
        await task
    context = await module.get_jargon_context(tmp_path, "group", "蓝票呢")
    assert "本周组队的暗号" in context
    assert "预约空闲机位" not in context
    bank = module._read_bank(module.bank_path(tmp_path, "group"), "group")
    assert bank.entries[0].status == "manual"
    assert bank.entries[0].corrections[0].user_id == "owner"


async def test_disable_is_persistent_and_blocks_relearning(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    await module.correct_jargon(tmp_path, "group", "蓝票", "", user_id="a", source_quote="停用蓝票这个词", disabled=True)
    await module.learn_jargon(tmp_path, "group", corpus(), Model)
    assert await module.get_jargon_context(tmp_path, "group", "蓝票") == ""
    assert module._read_bank(module.bank_path(tmp_path, "group"), "group").entries[0].disabled


async def test_correction_tool_requires_current_literal_evidence_and_records_history(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    text = "蓝票不是门票，在我们群是预约空闲机位的凭证"
    tool = module.create_jargon_tool(tmp_path, "group", "alice", text, None)
    invalid = json.loads(await tool.ainvoke({"term": "蓝票", "source_quote": "历史里说蓝票是门票", "meaning": "门票"}))
    assert invalid["ok"] is False
    invalid_meaning = json.loads(await tool.ainvoke({"term": "蓝票", "source_quote": text, "meaning": "群主最喜欢的物品"}))
    assert invalid_meaning["ok"] is False
    negated = json.loads(await tool.ainvoke({"term": "蓝票", "source_quote": text, "meaning": "门票"}))
    assert negated["reason_code"] == "negated_meaning"
    result = json.loads(await tool.ainvoke({"term": "蓝票", "source_quote": text, "meaning": "预约空闲机位的凭证"}))
    assert result["ok"] is True
    no_disable = json.loads(await tool.ainvoke({"term": "蓝票", "source_quote": text, "disable": True}))
    assert no_disable["ok"] is False
    correction = module._read_bank(module.bank_path(tmp_path, "group"), "group").entries[0].corrections[0]
    assert correction.source_quote == text
    assert correction.meaning == "预约空闲机位的凭证"


async def test_stale_correction_cannot_write(monkeypatch, tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    monkeypatch.setattr(module, "is_request_active", AsyncMock(return_value=False))
    tool = module.create_jargon_tool(tmp_path, "group", "a", "蓝票是预约凭证", "expired")
    result = json.loads(await tool.ainvoke({"term": "蓝票", "meaning": "预约凭证", "source_quote": "蓝票是预约凭证"}))
    assert result["status"] == "skipped"
    assert not module.bank_path(tmp_path, "group").exists()


async def test_matching_is_bounded_expiring_and_does_not_include_all_terms(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    now = datetime.now()
    entries = [module.JargonEntry(term=f"词语{i}", meaning=f"含义{i}", status="inferred", validation_version=module.AUTO_VALIDATION_VERSION, updated_at=now) for i in range(8)]
    entries.extend([module.JargonEntry(term="旧词", meaning="旧意思", status="inferred", validation_version=module.AUTO_VALIDATION_VERSION, updated_at=now - timedelta(days=91)),
                    module.JargonEntry(term="蓝票", meaning="普通词条", status="inferred", validation_version=module.AUTO_VALIDATION_VERSION, updated_at=now),
                    module.JargonEntry(term="蓝票机", meaning="更长词条", status="inferred", validation_version=module.AUTO_VALIDATION_VERSION, updated_at=now)])
    module._write_bank(module.bank_path(tmp_path, "group"), module.JargonBank(session_id="group", entries=entries))
    assert await module.get_jargon_context(tmp_path, "group", "旧词") == ""
    context = await module.get_jargon_context(tmp_path, "group", "蓝票机")
    data = json.loads(context.splitlines()[-1])
    assert [item["term"] for item in data] == ["蓝票机"]
    context = await module.get_jargon_context(tmp_path, "group", "、".join(entry.term for entry in entries))
    assert len(json.loads(context.splitlines()[-1])) == 5


async def test_corrupt_bank_is_not_overwritten_as_empty(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    path = module.bank_path(tmp_path, "group")
    path.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        await module.learn_jargon(tmp_path, "group", corpus(), Model)
    assert path.read_text(encoding="utf-8") == "broken"


@pytest.mark.parametrize(("enabled", "is_private", "quoted"), [(True, False, False), (True, False, True), (False, False, False), (True, True, False)])
async def test_agent_only_exposes_current_group_meaning_as_dynamic_context(monkeypatch, tmp_path, enabled, is_private, quoted):
    from typing import cast

    from nonebot.adapters import Event
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.agent.jargon_learning import correct_jargon

    await correct_jargon(tmp_path / "group_jargon", "group", "蓝票", "预约机位的凭证", user_id="a", source_quote="蓝票是预约机位的凭证")
    captured = {}

    def build(model, tools, system_prompt, **kwargs):
        captured["tools"] = {item.name for item in kwargs["base_tools"]}
        captured["system"] = str(system_prompt)
        return object()

    async def empty(*args, **kwargs):
        return ""

    class FakeEvent:
        def get_plaintext(self):
            return "这是什么意思" if quoted else "蓝票用完了吗"

    monkeypatch.setattr(agent, "get_user_relation_context", empty)
    monkeypatch.setattr(agent, "get_group_context", empty)
    monkeypatch.setattr(agent, "get_recent_relations_context", empty)
    monkeypatch.setattr(agent, "load_group_style_context", empty)
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "get_flash_model", lambda: object())
    monkeypatch.setattr(agent, "build_chat_graph", build)
    monkeypatch.setattr(agent, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(agent.plugin_config, "group_jargon_learning_enabled", enabled)
    async with get_session() as db:
        _, _, context = await agent.create_chat_graph(db, "group", None, "a", "Alice", history=[message(9, "b", "蓝票没了")],
                                                      event=cast(Event, FakeEvent()), is_private=is_private, reply_to_id="9" if quoted else None)
    visible = enabled and not is_private
    assert ("预约机位的凭证" in context) is visible
    assert ("update_group_jargon" in captured["tools"]) is visible
    assert "预约机位的凭证" not in captured["system"]


async def test_background_scheduler_deduplicates_jobs_and_reads_committed_history(monkeypatch, tmp_path):
    from uuid import uuid4

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    group = str(uuid4())
    async with get_session() as db:
        for row in corpus():
            db.add(ChatHistory(session_id=group, user_id=row.user_id, user_name=row.user_name,
                               content_type="text", content=row.content, created_at=datetime.now()))
        await db.commit()
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def learn(root, session_id, history, model_factory, **kwargs):
        seen.extend(history)
        entered.set()
        await release.wait()

    monkeypatch.setattr(module, "learn_jargon", learn)
    assert module.schedule_jargon_learning(tmp_path, group, Model)
    assert not module.schedule_jargon_learning(tmp_path, group, Model)
    task = module._jobs[module.bank_path(tmp_path, group)]
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert len(seen) == 12
        assert {row.session_id for row in seen} == {group}
    finally:
        release.set()
        await task
    assert module.bank_path(tmp_path, group) not in module._jobs
