from uuid import uuid4

import pytest
from sqlalchemy import Select, event


@pytest.fixture
async def postgres_text_session():
    from nonebot_plugin_orm import get_session

    # SQLite accepts NUL. Reject it at the SQL boundary as PostgreSQL does.
    def reject_nul(_conn, _cursor, statement, parameters, _context, _executemany):
        if statement.startswith(("INSERT", "UPDATE")):
            if any(isinstance(value, str) and "\x00" in value for value in parameters):
                raise ValueError('invalid byte sequence for encoding "UTF8": 0x00')

    async with get_session() as db:
        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", reject_nul)
        try:
            yield db
        finally:
            event.remove(engine, "before_cursor_execute", reject_nul)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_name", ["\n\x04\n\x02$)\n\x14\x12\x12原味果粒奶优\x10\x00", "中文昵称\n\t🙂", "\x00", None])
async def test_record_token_usage_commits_with_chat_history(postgres_text_session, user_name):
    from nonebot_plugin_ai_groupmate.model import TokenUsage, ChatHistory
    from nonebot_plugin_ai_groupmate.usage import record_token_usage

    db = postgres_text_session
    session_id = f"usage-nul-{uuid4()}"
    expected_name = (user_name or "").replace("\x00", "")
    db.add(ChatHistory(
        session_id=session_id,
        user_id="member-1",
        user_name=user_name or "",
        content_type="bot",
        content="id: 42\n你好",
    ))
    await record_token_usage(
        db,
        session_id=session_id,
        session_type="group",
        user_id="member-1",
        user_name=user_name,
        model="deepseek-flash",
        request_id=f"{session_id}:request-1",
        prompt_tokens=19147,
        completion_tokens=461,
        cached_tokens=16000,
        cache_creation_tokens=0,
        total_tokens=19608,
        estimated_cost=0.01093,
        agent_llm_calls=1,
        agent_tool_calls=1,
        agent_duration_ms=9030,
    )
    await db.commit()

    usage = (await db.execute(Select(TokenUsage).where(TokenUsage.session_id == session_id))).scalar_one()
    history = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == session_id))).scalar_one()
    assert usage.user_name == history.user_name == expected_name
    assert usage.model == "deepseek-flash"
    assert usage.prompt_tokens == 19147
    assert usage.completion_tokens == 461
    assert usage.cached_tokens == 16000
    assert usage.total_tokens == 19608
    assert usage.estimated_cost == pytest.approx(0.01093)
    assert usage.agent_llm_calls == usage.agent_tool_calls == 1
    assert usage.agent_duration_ms == 9030


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["TokenUsage", "UserRelation"])
async def test_user_name_is_cleaned_on_direct_insert_and_update(postgres_text_session, model_name):
    from nonebot_plugin_ai_groupmate import model

    db = postgres_text_session
    row_model = getattr(model, model_name)
    values = {"user_id": f"nickname-nul-{uuid4()}", "user_name": "原\x00味🙂"}
    if model_name == "TokenUsage":
        values["session_id"] = f"usage-nul-{uuid4()}"
    row = row_model(**values)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    assert row.user_name == "原味🙂"

    row.user_name = "新\x00昵称\n\t🙂"
    await db.commit()
    await db.refresh(row)
    assert row.user_name == "新昵称\n\t🙂"
