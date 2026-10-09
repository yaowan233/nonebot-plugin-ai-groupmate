import datetime
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def summary_model(monkeypatch):
    from nonebot_plugin_ai_groupmate import group_memory

    model = SimpleNamespace(ainvoke=AsyncMock())
    monkeypatch.setattr(group_memory, "get_summary_model", lambda: model)
    return model


@pytest.mark.asyncio
async def test_summary_at_unicode_character_limit_is_preserved_without_retry(summary_model):
    from nonebot_plugin_ai_groupmate import group_memory

    summary = "群" * 250 + "🧑" * 250
    summary_model.ainvoke.return_value = SimpleNamespace(content=summary)

    assert await group_memory._call_summary_model("", "Alice：一起打音游") == summary
    summary_model.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_overlong_summary_gets_one_compression_retry(summary_model):
    from nonebot_plugin_ai_groupmate import group_memory

    expected = "常见话题：音游和日常近况。"
    summary_model.ainvoke.side_effect = [SimpleNamespace(content="群" * 501), SimpleNamespace(content=expected)]

    assert await group_memory._call_summary_model("旧话题：音游", "Alice：一起打音游") == expected
    assert summary_model.ainvoke.await_count == 2
    retry_prompt = summary_model.ainvoke.await_args_list[1].args[0]
    assert "压缩" in retry_prompt[-1].content
    assert "Alice：一起打音游" in retry_prompt[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_result", [SimpleNamespace(content="群" * 501), SimpleNamespace(content=""), SimpleNamespace(content=[]), RuntimeError("compression failed")], ids=["overlong", "empty", "nontext", "failed"])
async def test_invalid_compression_is_skipped_without_truncation_or_more_retries(summary_model, retry_result):
    from nonebot_plugin_ai_groupmate import group_memory

    summary_model.ainvoke.side_effect = [SimpleNamespace(content="群" * 700), retry_result]

    assert await group_memory._call_summary_model("有效旧档案", "Alice：一起打音游") is None
    assert summary_model.ainvoke.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("long_existing", ["旧事实。" * 1000 + "不可出现在输入的尾巴", "无句段边界" * 5000], ids=["sentences", "unbroken"])
async def test_oversized_existing_memory_does_not_enter_prompt_unbounded(summary_model, long_existing):
    from nonebot_plugin_ai_groupmate import group_memory

    summary_model.ainvoke.return_value = SimpleNamespace(content="常见话题：音游。")
    await group_memory._call_summary_model(long_existing, "Alice：一起打音游")

    prompt = summary_model.ainvoke.await_args.args[0][-1].content
    assert len(prompt) < 2500
    assert "不可出现在输入的尾巴" not in prompt
    assert "省略" in prompt
    assert "Alice：一起打音游" in prompt
    if long_existing.startswith("无句段边界"):
        assert "无句段边界" not in prompt


@pytest.mark.asyncio
async def test_compression_reference_is_bounded_at_complete_sentence(summary_model):
    from nonebot_plugin_ai_groupmate import group_memory

    candidate = "有证据的完整句子。" + "巨长无终止语句" * 1000
    summary_model.ainvoke.side_effect = [
        SimpleNamespace(content=candidate),
        SimpleNamespace(content="常见话题：音游。"),
    ]

    assert await group_memory._call_summary_model("", "Alice：一起打音游") == "常见话题：音游。"

    retry_prompt = summary_model.ainvoke.await_args_list[1].args[0][-1].content
    assert len(retry_prompt) < 3000
    assert "有证据的完整句子。" in retry_prompt
    assert "巨长无终止语句" not in retry_prompt
    assert "省略" in retry_prompt
    assert "Alice：一起打音游" in retry_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("has_existing", [False, True])
async def test_failed_compression_leaves_persisted_memory_unchanged(summary_model, has_existing):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import group_memory
    from nonebot_plugin_ai_groupmate.model import ChatHistory, GroupMemory

    session_id = f"summary-limit-{uuid4()}"
    old_time = datetime.datetime.now() - datetime.timedelta(hours=1)
    summary_model.ainvoke.side_effect = [SimpleNamespace(content="群" * 700), SimpleNamespace(content="群" * 600)]
    async with get_session() as session:
        if has_existing:
            session.add(GroupMemory(session_id=session_id, summary="有效旧档案", msg_count_at_last_update=7, updated_at=old_time))
        session.add(ChatHistory(session_id=session_id, user_id="Alice", user_name="Alice", content_type="text", content="id: 101\n一起打音游"))
        await session.commit()

        result = await group_memory.update_group_memory(session, session_id, bot_name="咕咕")
        record = (await session.execute(Select(GroupMemory).where(GroupMemory.session_id == session_id))).scalar_one_or_none()

        if has_existing:
            assert record is not None
            assert (record.summary, record.msg_count_at_last_update, record.updated_at) == ("有效旧档案", 7, old_time)
        else:
            assert record is None
    assert "跳过" in result
    assert summary_model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_persistence_boundary_rejects_overlong_helper_result(monkeypatch):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import group_memory
    from nonebot_plugin_ai_groupmate.model import ChatHistory, GroupMemory

    session_id = f"summary-boundary-{uuid4()}"
    monkeypatch.setattr(group_memory, "_call_summary_model", AsyncMock(return_value="群" * 501))
    async with get_session() as session:
        session.add(ChatHistory(session_id=session_id, user_id="Alice", user_name="Alice", content_type="text", content="一起打音游"))
        await session.commit()
        result = await group_memory.update_group_memory(session, session_id, bot_name="咕咕")
        record = (await session.execute(Select(GroupMemory).where(GroupMemory.session_id == session_id))).scalar_one_or_none()

    assert record is None
    assert "跳过" in result


@pytest.mark.asyncio
async def test_known_bot_pollution_in_old_memory_is_removed_before_model(monkeypatch):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import group_memory
    from nonebot_plugin_ai_groupmate.model import ChatHistory, GroupMemory

    session_id = f"summary-clean-old-{uuid4()}"
    old_time = datetime.datetime.now() - datetime.timedelta(hours=1)
    summary_call = AsyncMock(return_value="常见话题：音游。")
    monkeypatch.setattr(group_memory, "_call_summary_model", summary_call)
    async with get_session() as session:
        session.add(GroupMemory(session_id=session_id, summary="常见话题：音游。\n- 咕咕是群里活跃成员\n- 标准回应：别得寸进尺", updated_at=old_time))
        session.add(ChatHistory(session_id=session_id, user_id="Alice", user_name="Alice", content_type="text", content="一起打音游"))
        await session.commit()

        await group_memory.update_group_memory(session, session_id, bot_name="咕咕")

    call_args = summary_call.await_args
    assert call_args is not None
    assert call_args.args[0] == "常见话题：音游。"
