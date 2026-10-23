import json
import asyncio
from typing import Any
from datetime import datetime, timedelta

import pytest


def message(number, user, text):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    return ChatHistorySchema(msg_id=number, session_id="group", user_id=user, user_name=user,
                             content_type="text", content=text, created_at=datetime.now())


def expression_bank():
    from nonebot_plugin_ai_groupmate.agent.style_learning import ExpressionBank, ExpressionUnit

    return ExpressionBank(session_id="group", last_source_id=100, expressions=[
        ExpressionUnit(id="surprise", situation="惊讶接梗", style="短反问，末尾轻停顿", sources=[{"id": 1, "user_id": "a"}], updated_at=datetime.now()),
        ExpressionUnit(id="explain", situation="认真解释", style="先答结论，再用短句说明原因", sources=[{"id": 2, "user_id": "b"}], updated_at=datetime.now()),
    ])


def test_expression_provenance_only_accepts_filtered_human_ids_and_preserves_disabling():
    from nonebot_plugin_ai_groupmate.agent.style_learning import ExpressionBank, ExpressionLearning, merge_expressions

    sources = [(message(1, "alice", "这也能过（"), "这也能过（"), (message(2, "bob", "我还以为卡了"), "我还以为卡了")]
    learned = ExpressionLearning.model_validate({"expressions": [
        {"situation": "惊讶", "style": "短句反问", "source_ids": [1, 2]},
        {"situation": "来源假的", "style": "不应保存", "source_ids": [1, 999]},
        {"situation": "自称alice", "style": "继承群友身份", "source_ids": [1]},
    ]})
    bank = merge_expressions(ExpressionBank(session_id="group"), learned, sources, now=datetime.now())
    assert len(bank.expressions) == 1
    assert [source["id"] for source in bank.expressions[0].sources] == [1, 2]
    assert bank.expressions[0].sources[0]["text"] == "这也能过（"
    bank.expressions[0].disabled = True
    bank.expressions[0].updated_at -= timedelta(days=100)
    bank = merge_expressions(bank, learned, sources, now=datetime.now())
    assert len(bank.expressions) == 1
    assert bank.expressions[0].disabled


async def test_style_selection_changes_with_turn_and_rejects_unknown_ids(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning

    learning._write_bank(learning.bank_path(tmp_path, "group"), expression_bank())
    prompts = []

    class Model:
        def with_structured_output(self, schema):
            assert schema is learning.ExpressionSelection
            return self

        async def ainvoke(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 400
            payload = json.loads(messages[1].content)
            prompts.append(payload)
            key = "explain" if "原理" in payload["recent_chat"] else "surprise"
            return {"expression_ids": [key, "forged"]}

    kwargs: dict[str, Any] = {"root": tmp_path, "session_id": "group", "sources": [], "model_factory": Model}
    surprised = await learning.contextual_style(recent_chat="这也能过？", **kwargs)
    serious = await learning.contextual_style(recent_chat="请解释原理", **kwargs)
    assert surprised is not None
    assert serious is not None
    assert "短反问" in surprised
    assert "先答结论" not in surprised
    assert "先答结论" in serious
    assert "短反问" not in serious
    assert len(prompts) == 2
    assert "sources" not in prompts[0]["candidates"][0]


@pytest.mark.parametrize("answer", [{"expression_ids": []}, {"expression_ids": ["forged"]}, {"wrong_schema": True}])
async def test_empty_or_failed_selection_does_not_refill_unrelated_examples(tmp_path, answer):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning

    learning._write_bank(learning.bank_path(tmp_path, "group"), expression_bank())

    class Model:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, *_args, **_kwargs):
            return answer

    result = await learning.contextual_style(root=tmp_path, session_id="group", sources=[], recent_chat="严肃求助", model_factory=Model)
    assert result == ""


async def test_background_learning_persists_evidence_and_can_be_loaded_next_turn(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning

    observed_budgets = []

    class Model:
        def with_structured_output(self, schema):
            self.schema = schema
            return self

        async def ainvoke(self, messages, **kwargs):
            observed_budgets.append(kwargs["max_tokens"])
            if self.schema is learning.ExpressionLearning:
                payload = json.loads(messages[1].content)
                if isinstance(payload, list):
                    assert len(payload) == 12
                else:
                    assert payload["candidates"][0]["source_ids"] == [1, 2]
                    assert len(payload["evidence"]) == 12
                return {"expressions": [{"situation": "惊讶", "style": "先短问，再接一句轻吐槽", "source_ids": [1, 2]}]}
            return {"expression_ids": [json.loads(messages[1].content)["candidates"][0]["id"]]}

    sources = [(message(i, f"human-{i % 3}", f"这也能过{i}（"), f"这也能过{i}（") for i in range(1, 13)]
    kwargs: dict[str, Any] = {"root": tmp_path, "session_id": "group", "sources": sources, "recent_chat": "怎么还过了", "model_factory": Model}
    assert await learning.contextual_style(**kwargs) is None
    await asyncio.gather(*list(learning._jobs.values()))
    saved = learning.ExpressionBank.model_validate_json(learning.bank_path(tmp_path, "group").read_text(encoding="utf-8"))
    assert saved.last_source_id == 12
    assert [item["id"] for item in saved.expressions[0].sources] == [1, 2]
    selected = await learning.contextual_style(**kwargs)
    assert selected is not None
    assert "轻吐槽" in selected
    assert observed_budgets == [1800, 1800, 400]
    assert await learning.contextual_style(**{**kwargs, "session_id": "other", "sources": []}) is None


async def test_disabled_expired_and_bot_sourced_expressions_are_not_selected(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning

    bank = expression_bank()
    bank.expressions[0].disabled = True
    bank.expressions[1].updated_at -= timedelta(days=100)
    learning._write_bank(learning.bank_path(tmp_path, "group"), bank)
    def must_not_call():
        raise AssertionError("no eligible expressions")
    assert await learning.contextual_style(root=tmp_path, session_id="group", sources=[], recent_chat="继续", model_factory=must_not_call) == ""
    bank = expression_bank()
    for unit in bank.expressions:
        unit.sources[0]["user_id"] = "known-bot"
    learning._write_bank(learning.bank_path(tmp_path, "group"), bank)
    assert await learning.contextual_style(root=tmp_path, session_id="group", sources=[], recent_chat="继续", model_factory=must_not_call, excluded_users=["known-bot"]) == ""
