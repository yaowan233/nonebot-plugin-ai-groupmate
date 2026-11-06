import json
from datetime import datetime, timedelta

import pytest
from test_jargon_learning import Model, corpus, message, extraction


async def test_real_repeated_questions_do_not_verify_an_invented_meaning(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    rows = [message(1, "a", "miyuki hero是什么食物"), message(2, "b", "miyuki 今晚夜宵吃什么"), *corpus()[2:]]
    invented = extraction(term="miyuki", meaning="操作失误", source_ids=[1, 2])
    bank = module.merge_extraction(module.JargonBank(session_id="group"), invented, module.collect_sources(rows, "group"), now=datetime.now())
    module._write_bank(module.bank_path(tmp_path, "group"), bank)
    assert await module.get_jargon_context(tmp_path, "group", "miyuki 晚饭吃什么") == ""


async def test_old_auto_guesses_are_not_treated_as_reviewed_definitions(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    old = module.JargonEntry(term="miyuki", meaning="操作失误", status="inferred", updated_at=datetime.now())
    module._write_bank(module.bank_path(tmp_path, "group"), module.JargonBank(session_id="group", entries=[old]))
    assert await module.get_jargon_context(tmp_path, "group", "miyuki是什么") == ""


async def test_blind_reviewer_rejects_real_bot_name_mislearning(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    rows = [message(1, "a", "miyuki hero是什么食物"), message(2, "b", "miyuki 今晚夜宵吃什么"), *corpus()[2:]]
    proposer = Model(extraction(term="miyuki", meaning="操作失误"))
    reviewer = Model(module.JargonExtraction())
    assert await module.learn_jargon(tmp_path, "group", rows, lambda: proposer, review_model_factory=lambda: reviewer)
    assert proposer.calls == reviewer.calls == 1
    review_input = json.loads(reviewer.messages[-1].content)
    assert review_input["terms"] == ["miyuki"]
    assert "操作失误" not in reviewer.messages[-1].content
    assert await module.get_jargon_context(tmp_path, "group", "miyuki是什么") == ""


async def test_known_names_and_prefixed_bot_output_are_not_candidates(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    rows = [message(1, "a", "miyuki hero是什么食物"), message(2, "b", "miyuki 今晚夜宵吃什么"),
            *corpus()[2:], message(20, "third_party_bot", "[Lazybot] 暂支支持其他模式")]
    proposer = Model(extraction(term="miyuki", meaning="操作失误"))
    reviewer = Model()
    assert await module.learn_jargon(tmp_path, "group", rows, lambda: proposer, review_model_factory=lambda: reviewer, excluded_terms=("miyuki",))
    assert reviewer.calls == 0
    assert 20 not in {source.msg_id for source in module.collect_sources(rows, "group")}
    assert module._read_bank(module.bank_path(tmp_path, "group"), "group").entries == []


async def test_review_failure_does_not_activate_guesses_or_advance_cursor(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    now = datetime.now()
    for index in range(3):
        when = now + timedelta(hours=index * 2)
        with pytest.raises(TimeoutError):
            await module.learn_jargon(tmp_path, "group", corpus(), Model, review_model_factory=lambda: Model(TimeoutError()), now=when)
        bank = module._read_bank(module.bank_path(tmp_path, "group"), "group")
        assert bank.last_source_id == 0
        assert not bank.entries
        assert bank.consecutive_failures == index + 1
        assert bank.next_attempt_at == when + timedelta(minutes=30 * 2 ** index)


async def test_old_guesses_do_not_anchor_discovery_and_manual_definitions_survive(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import jargon_learning as module

    bank = module.JargonBank(session_id="group", entries=[module.JargonEntry(term="蓝票", meaning="一个完全错误的猜测", status="inferred", updated_at=datetime.now())])
    module._write_bank(module.bank_path(tmp_path, "group"), bank)
    model = Model()
    await module.learn_jargon(tmp_path, "group", corpus(), lambda: model)
    assert "一个完全错误的猜测" not in str(model.messages)
    context = await module.get_jargon_context(tmp_path, "group", "蓝票是什么")
    assert "预约空闲机位" in context
    saved = module._read_bank(module.bank_path(tmp_path, "group"), "group")
    assert saved.entries[0].validation_version == module.AUTO_VALIDATION_VERSION
    await module.correct_jargon(tmp_path, "group", "miyuki", "本群机器人的名字", user_id="a", source_quote="miyuki是本群机器人的名字")
    assert "本群机器人的名字" in await module.get_jargon_context(tmp_path, "group", "miyuki是什么")
    assert "本群机器人的名字" in await module.get_jargon_context(tmp_path, "group", "miyuki是什么", excluded_terms=("miyuki",))


def test_learning_model_has_own_budget_and_review_uses_main_model(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    monkeypatch.setattr(agent.plugin_config, "llm_api_key", "unit-test-key")
    agent.get_flash_model.cache_clear()
    agent.get_jargon_discovery_model.cache_clear()
    agent.get_jargon_review_model.cache_clear()
    try:
        discovery = agent.get_jargon_discovery_model()
        reviewer = agent.get_jargon_review_model()
        assert discovery.max_tokens == 1000
        assert reviewer.max_tokens == 1200
        assert discovery.max_retries == reviewer.max_retries == 0
        assert discovery.root_async_client.max_retries == reviewer.root_async_client.max_retries == 0
        assert discovery.root_async_client.timeout == reviewer.root_async_client.timeout == 40
        assert discovery.model_name == (agent.plugin_config.chat_model or agent.plugin_config.base_model)
        assert reviewer.model_name == (agent.plugin_config.chat_model or agent.plugin_config.base_model)
        assert agent.get_flash_model().max_tokens == agent.plugin_config.flash_max_tokens
    finally:
        agent.get_flash_model.cache_clear()
        agent.get_jargon_discovery_model.cache_clear()
        agent.get_jargon_review_model.cache_clear()
