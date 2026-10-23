import json
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError


def message(number, user, body, *, quote=None, platform_id=None, seconds=None, **overrides):
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    content = f"id: {platform_id or number}\n"
    if quote is not None:
        content += f"回复id: {quote}\n"
    values = {"msg_id": number, "session_id": "group", "user_id": user, "user_name": user,
              "content_type": "text", "content": content + body,
              "created_at": datetime.now().replace(microsecond=0) + timedelta(seconds=number if seconds is None else seconds)}
    return ChatHistorySchema(**{**values, **overrides})


def candidates(rows, *, excluded=(), after=0, quoted_after=None):
    from nonebot_plugin_ai_groupmate.agent.chat_style import _human_expressions
    from nonebot_plugin_ai_groupmate.agent.reply_pair_learning import build_reply_pair_candidates

    sources = _human_expressions(rows, "group", excluded)
    return build_reply_pair_candidates(rows, sources, "group", after_reply_id=after, after_quoted_reply_id=quoted_after)


def quoted_rows():
    return [message(1, "person-a", "折腾半天终于弄好了"), message(2, "person-b", "你还真给整出来了", quote="1")]


def test_explicit_quote_wins_over_nearest_message_and_keeps_source_ids():
    rows = [message(1, "a", "总算跑起来了"), message(2, "b", "晚上吃什么"), message(3, "c", "你还真弄好了", quote="1")]
    pair = next(pair for pair in candidates(rows) if pair.reply.msg_id == 3)
    assert pair.relation == "quoted"
    assert pair.kind == "response"
    assert pair.trigger.msg_id == 1
    assert pair.trigger.text == "总算跑起来了"
    assert pair.trigger.platform_ids == ["1"]
    assert pair.reply.platform_ids == ["3"]


@pytest.mark.parametrize("quote", ["missing", "0", "2", "future"])
def test_missing_invalid_self_and_future_quotes_never_fall_back_to_adjacent(quote):
    rows = [message(1, "a", "前面的消息"), message(2, "b", "后面的消息", quote=quote), message(3, "c", "未来的消息", platform_id="future")]
    assert all(pair.reply.msg_id != 2 for pair in candidates(rows))


def test_bot_other_group_and_ambiguous_platform_ids_cannot_supply_quote_evidence():
    for target in (message(1, "a", "其他群的话", session_id="other"), message(1, "bot", "机器人说的", content_type="bot")):
        rows = [target, message(2, "b", "引用后的回应", quote="1")]
        assert candidates(rows) == []
    rows = [message(1, "a", "一条消息", platform_id="same"), message(2, "b", "另一条消息", platform_id="same"),
            message(3, "c", "不能确定引用谁", quote="same")]
    assert candidates(rows) == []


def test_quote_matches_transport_alias_but_not_body_text():
    row = message(1, "a", "id: fake-id\n普通正文", content="id: real-id\nalias_id: alias-id\nid: fake-id\n普通正文")
    good = message(2, "b", "这是引用回应", quote="alias-id")
    bad = message(3, "c", "不是正文的引用", quote="fake-id")
    pairs = candidates([row, good, bad])
    assert any(pair.reply.msg_id == 2 and pair.trigger.msg_id == 1 for pair in pairs)
    assert all(pair.reply.msg_id != 3 for pair in pairs)


def test_adjacent_window_and_ineligible_message_breaks_prevent_bridging():
    assert candidates([message(1, "a", "前面说的话", seconds=0), message(2, "b", "隔很久说的话", seconds=61)]) == []
    rows = [message(1, "a", "前面说的话"), message(2, "bot", "机器人中间回复", content_type="bot"), message(3, "b", "后面的话")]
    assert candidates(rows) == []
    rows[1] = message(2, "c", "/一个命令")
    assert candidates(rows) == []
    rows[1] = message(2, "connected-bot", "其他机器人")
    assert candidates(rows, excluded=("connected-bot",)) == []


def test_same_person_continuation_is_not_labeled_another_person_response():
    pairs = candidates([message(1, "a", "我刚弄好了"), message(2, "a", "补充一下细节"), message(3, "b", "原来是这样啊")])
    assert pairs[0].kind == "continuation"
    assert pairs[1].kind == "response"


async def test_same_person_followup_is_not_trained_as_interpersonal_reply(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning
    from nonebot_plugin_ai_groupmate.agent.chat_style import _human_expressions

    rows = [message(1, "a", "这个还有后续"), message(2, "a", "补充一点细节")]
    def must_not_call():
        raise AssertionError("a continuation is not a human-to-human response")
    path = learning.bank_path(tmp_path, "group")
    await learning._learn(path, "group", _human_expressions(rows, "group"), must_not_call, rows)
    assert not path.exists()


def test_quote_priority_has_independent_progress_without_skipping_adjacent_pairs():
    from nonebot_plugin_ai_groupmate.agent.style_learning import ExpressionBank, ExpressionLearning, merge_reply_pairs
    from nonebot_plugin_ai_groupmate.agent.reply_pair_learning import MAX_PAIR_CANDIDATES

    rows = [message(i, f"user-{i % 4}", f"第{i}条普通聊天", quote="1" if i == 11 else None) for i in range(1, 13)]
    pairs = candidates(rows)
    assert len(pairs) == MAX_PAIR_CANDIDATES
    assert pairs[0].relation == "quoted"
    assert {pair.reply.msg_id for pair in pairs} == {11, *range(2, 9)}
    decisions = ExpressionLearning.model_validate({"reply_pairs": [{"pair_id": pair.id, "is_reply_pair": True, "transferable": True} for pair in pairs]})
    bank = merge_reply_pairs(ExpressionBank(session_id="group"), pairs, decisions, decisions, now=datetime.now())
    assert bank.pair_cursor == 8
    assert bank.quoted_pair_cursor == 11
    later = candidates(rows, after=bank.pair_cursor, quoted_after=bank.quoted_pair_cursor)
    assert {pair.reply.msg_id for pair in later} == {9, 10, 12}


def test_bank_migration_and_pair_judgments_require_explicit_booleans():
    from nonebot_plugin_ai_groupmate.agent.style_learning import ExpressionBank
    from nonebot_plugin_ai_groupmate.agent.reply_pair_learning import ReplyPairJudgment

    bank = ExpressionBank.model_validate({"session_id": "group", "expressions": [], "last_source_id": 123})
    assert bank.reply_pairs == []
    assert bank.pair_cursor == 0
    assert bank.quoted_pair_cursor == 0
    with pytest.raises(ValidationError):
        ReplyPairJudgment.model_validate({"pair_id": "p", "is_reply_pair": "false", "transferable": True})
    with pytest.raises(ValidationError):
        ReplyPairJudgment.model_validate({"pair_id": "p", "is_reply_pair": True})


def test_unknown_duplicate_and_partial_judgments_do_not_skip_cursor():
    from nonebot_plugin_ai_groupmate.agent.style_learning import ExpressionBank, ExpressionLearning, merge_reply_pairs

    pairs = candidates(quoted_rows())
    decision = {"pair_id": pairs[0].id, "is_reply_pair": True, "transferable": True}
    first = ExpressionLearning.model_validate({"reply_pairs": [decision]})
    bank = ExpressionBank(session_id="group", last_source_id=100)
    unknown = ExpressionLearning.model_validate({"reply_pairs": [{**decision, "pair_id": "forged"}]})
    partial = merge_reply_pairs(bank, pairs, first, unknown, now=datetime.now())
    assert partial.pair_cursor == 0
    assert partial.quoted_pair_cursor == 0
    assert partial.reply_pairs == []
    duplicated = ExpressionLearning.model_validate({"reply_pairs": [decision, decision]})
    assert merge_reply_pairs(bank, pairs, duplicated, first, now=datetime.now()).pair_cursor == 0
    accepted = merge_reply_pairs(bank, pairs, first, first, now=datetime.now())
    assert accepted.quoted_pair_cursor == 2
    assert accepted.pair_cursor == 0
    assert accepted.last_source_id == 100
    accepted.reply_pairs[0].disabled = True
    accepted = merge_reply_pairs(accepted, pairs, first, first, now=datetime.now())
    assert accepted.reply_pairs[0].disabled


@pytest.mark.parametrize("accepted", [True, False])
async def test_real_learning_pipeline_persists_original_pair_only_after_review(tmp_path, accepted):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning
    from nonebot_plugin_ai_groupmate.agent.chat_style import _human_expressions

    rows = quoted_rows()
    pair = candidates(rows)[0]
    calls = []
    class Model:
        def with_structured_output(self, schema):
            self.schema = schema
            return self

        async def ainvoke(self, messages, **kwargs):
            payload = json.loads(messages[1].content)
            calls.append(payload)
            if self.schema is learning.PairedExpressionSelection:
                assert payload["reply_pairs"][0]["trigger_text"] == rows[0].content.split("\n", 1)[1]
                return {"expression_ids": [], "pair_ids": [pair.id, "invented"]}
            assert kwargs["max_tokens"] == 3000
            assert payload["reply_pairs"][0]["kind"] == "response"
            return {"expressions": [{"situation": "接住对方刚完成的事情", "style": "用短反问表达意外", "source_ids": [2], "pair_ids": [pair.id]}],
                    "reply_pairs": [{"pair_id": pair.id, "is_reply_pair": True, "transferable": accepted if "candidates" in payload else True}]}

    sources = _human_expressions(rows, "group")
    path = learning.bank_path(tmp_path, "group")
    await learning._learn(path, "group", sources, Model, rows)
    bank = learning._read_bank(path, "group")
    assert bank.last_source_id == 0, "pair-only learning must not advance the single-message cursor"
    assert bank.quoted_pair_cursor == 2
    assert len(bank.reply_pairs) == int(accepted)
    assert len(bank.expressions) == int(accepted)
    if accepted:
        assert bank.reply_pairs[0].trigger.text == "折腾半天终于弄好了"
        assert bank.reply_pairs[0].reply.text == "你还真给整出来了"
        result = await learning.contextual_style(root=tmp_path, session_id="group", sources=[], recent_chat="我总算修好了", model_factory=Model)
        assert result is not None
        assert "真人接话案例" in result
        assert "折腾半天终于弄好了" in result
        assert "person-a" not in result
        assert "invented" not in result
    else:
        assert len(calls) == 2


async def test_disabled_or_bot_sourced_pair_cannot_be_injected(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning

    pair = candidates(quoted_rows())[0]
    bank = learning.ExpressionBank(session_id="group", reply_pairs=[pair], pair_cursor=2)
    def must_not_call():
        raise AssertionError("there is no eligible pair")
    path = learning.bank_path(tmp_path, "group")
    learning._write_bank(path, bank)
    assert await learning.contextual_style(root=tmp_path, session_id="group", sources=[], recent_chat="继续", model_factory=must_not_call,
                                          excluded_users=["person-a"]) == ""
    bank.reply_pairs[0].disabled = True
    learning._write_bank(path, bank)
    assert await learning.contextual_style(root=tmp_path, session_id="group", sources=[], recent_chat="继续", model_factory=must_not_call) == ""


async def test_pair_job_requires_judgments_and_does_not_skip_failed_batch(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning
    from nonebot_plugin_ai_groupmate.agent.chat_style import _human_expressions

    class IncompleteModel:
        def with_structured_output(self, schema):
            assert schema is learning.PairedExpressionLearning
            return self

        async def ainvoke(self, *_args, **_kwargs):
            return {"expressions": []}
    rows = quoted_rows()
    path = learning.bank_path(tmp_path, "group")
    await learning._learn(path, "group", _human_expressions(rows, "group"), IncompleteModel, rows)
    assert not path.exists()


async def test_explicitly_rejected_pair_is_processed_without_another_review(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning
    from nonebot_plugin_ai_groupmate.agent.chat_style import _human_expressions

    rows = quoted_rows()
    pair = candidates(rows)[0]
    calls = []
    class Model:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, *_args, **_kwargs):
            calls.append(1)
            return {"expressions": [], "reply_pairs": [{"pair_id": pair.id, "is_reply_pair": False, "transferable": False}]}
    path = learning.bank_path(tmp_path, "group")
    await learning._learn(path, "group", _human_expressions(rows, "group"), Model, rows)
    bank = learning._read_bank(path, "group")
    assert bank.quoted_pair_cursor == 2
    assert bank.reply_pairs == []
    assert len(calls) == 1


async def test_pair_and_single_learning_take_turns_without_advancing_each_others_cursor(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import style_learning as learning
    from nonebot_plugin_ai_groupmate.agent.chat_style import _human_expressions

    rows = [message(i, f"user-{i % 3}" if i > 1 else "starter", f"第{i}条普通聊天", quote="1" if i > 1 else None) for i in range(1, 15)]
    inputs = []
    class Model:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages, **_kwargs):
            payload = json.loads(messages[1].content)
            inputs.append(payload)
            if isinstance(payload, dict):
                assert payload["single_messages"] == []
                return {"expressions": [], "reply_pairs": [{"pair_id": pair["pair_id"], "is_reply_pair": True, "transferable": False}
                                                           for pair in payload["reply_pairs"]]}
            return {"expressions": []}
    path = learning.bank_path(tmp_path, "group")
    sources = _human_expressions(rows, "group")
    await learning._learn(path, "group", sources, Model, rows)
    first = learning._read_bank(path, "group")
    assert first.last_source_id == 0
    assert first.quoted_pair_cursor == 9
    assert first.prefer_pair_learning is False
    await learning._learn(path, "group", sources, Model, rows)
    second = learning._read_bank(path, "group")
    assert second.last_source_id == 14
    assert second.quoted_pair_cursor == first.quoted_pair_cursor
    assert second.prefer_pair_learning is True
    assert len(inputs) == 2
    assert isinstance(inputs[1], list)


async def test_production_style_loader_passes_persisted_quote_relationships(monkeypatch, tmp_path):
    from uuid import uuid4

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import chat_style
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    group = f"pair-loader-{uuid4()}"
    captured = []
    async def select(**kwargs):
        from nonebot_plugin_ai_groupmate.agent.reply_pair_learning import build_reply_pair_candidates

        pairs = build_reply_pair_candidates(kwargs["learning_history"], kwargs["sources"], group)
        captured.extend(pairs)
        return "接话关系已传入"
    monkeypatch.setattr(chat_style, "contextual_style", select)
    async with get_session() as db:
        now = datetime.now()
        db.add_all([ChatHistory(session_id=group, user_id="a", user_name="A", content_type="text", content="id: 100\n总算弄好了", created_at=now - timedelta(seconds=2)),
                    ChatHistory(session_id=group, user_id="b", user_name="B", content_type="text", content="id: 101\n回复id: 100\n你还真弄好了", created_at=now - timedelta(seconds=1))])
        await db.commit()
        result = await chat_style.load_group_style_context(db, group, model_factory=lambda: None, store_root=tmp_path)
    assert result == "接话关系已传入"
    assert len(captured) == 1
    assert captured[0].relation == "quoted"


async def test_pair_source_window_keeps_old_quote_targets_and_message_boundaries(monkeypatch, tmp_path):
    from uuid import uuid4

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import chat_style
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.reply_pair_learning import build_reply_pair_candidates

    group = f"pair-window-{uuid4()}"
    seen = []
    async def select(**kwargs):
        # The regular style sampler has intentionally capped this speaker.
        assert all(row.content.split("\n")[0] != "id: 100" for row, _ in kwargs["sources"])
        pairs = build_reply_pair_candidates(kwargs["learning_history"], kwargs["pair_sources"], group)
        seen.extend(pairs)
        return ""
    monkeypatch.setattr(chat_style, "contextual_style", select)
    now = datetime.now()
    async with get_session() as db:
        db.add_all([ChatHistory(session_id=group, user_id="a", user_name="A", content_type="text", content=f"id: {100+i}\n第{i}条真人聊天", created_at=now-timedelta(seconds=90-i)) for i in range(30)])
        db.add_all([
            ChatHistory(session_id=group, user_id="b", user_name="B", content_type="text", content="id: 200\n回复id: 100\n还真给你弄好了", created_at=now-timedelta(seconds=5)),
            ChatHistory(session_id=group, user_id="c", user_name="C", content_type="image", content="id: 201\nimage.png", created_at=now-timedelta(seconds=4)),
            ChatHistory(session_id=group, user_id="d", user_name="D", content_type="text", content="id: 202\n另一个话题", created_at=now-timedelta(seconds=3)),
        ])
        await db.commit()
        await chat_style.load_group_style_context(db, group, model_factory=lambda: None, store_root=tmp_path)
    assert any(pair.relation == "quoted" and pair.trigger.platform_ids == ["100"] for pair in seen)
    assert all(pair.reply.platform_ids != ["202"] for pair in seen)
