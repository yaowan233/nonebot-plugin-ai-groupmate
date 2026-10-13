from uuid import uuid4
from datetime import datetime

import pytest
from sqlalchemy import JSON, String, event, select, update


@pytest.fixture
async def postgres_safe_session():
    from nonebot_plugin_orm import get_session

    def reject_nul(_conn, _cursor, _statement, parameters, _context, _executemany):
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
@pytest.mark.parametrize(("model_name", "field"), [
    ("MediaStorage", "description"), ("GroupMemory", "summary"),
    ("ExplicitMemory", "title"), ("ExplicitMemory", "content"), ("ExplicitMemory", "source_quote"),
    ("ScheduledTask", "content"), ("ScheduledTask", "error"),
])
async def test_external_text_insert_update_and_lookup(postgres_safe_session, model_name, field):
    from nonebot_plugin_ai_groupmate import model

    db = postgres_safe_session
    row_model = getattr(model, model_name)
    unique = str(uuid4())
    now = datetime.now()
    values = {
        "MediaStorage": {"file_hash": unique, "file_path": "audit.png", "description": "normal"},
        "GroupMemory": {"session_id": unique, "summary": "normal"},
        "ExplicitMemory": {"session_id": unique, "user_id": unique, "bot_id": "bot", "is_private": False, "title": "normal", "content": "normal", "kind": "instruction", "lifetime": "temporary", "source_quote": "normal", "source_msg_id": 1, "created_at": now, "updated_at": now, "last_used_at": now},
        "ScheduledTask": {"job_id": unique, "session_id": unique, "is_private": False, "bot_id": None, "bot_name": "bot", "task_type": "message", "content": "normal", "run_at": now, "status": "scheduled", "created_at": now, "updated_at": now},
    }[model_name]
    values[field] = "外\x00部文本\n\t🙂"
    row = row_model(**values)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    assert getattr(row, field) == "外部文本\n\t🙂"
    column = getattr(row_model, field)
    found = (await db.execute(select(row_model).where(column == values[field]))).scalar_one()
    assert found is row
    await db.execute(update(row_model).where(column == values[field]).values({column: "新\x00文本\n🙂"}))
    await db.commit()
    await db.refresh(row)
    assert getattr(row, field) == "新文本\n🙂"


@pytest.mark.asyncio
async def test_nested_json_is_cleaned_without_mutating_input(postgres_safe_session):
    from nonebot_plugin_ai_groupmate.model import LastImageSearch

    db = postgres_safe_session
    payload = {"标\x00题": ["中\x00文🙂", {"url": "https://example.com/\x00image", "count": 3, "ok": True, "empty": None}], "escaped": r"\u0000"}
    row = LastImageSearch(session_id=str(uuid4()), user_id="user", result=payload)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    assert row.result == {"标题": ["中文🙂", {"url": "https://example.com/image", "count": 3, "ok": True, "empty": None}], "escaped": r"\u0000"}
    assert "标\x00题" in payload
    await db.execute(update(LastImageSearch).where(LastImageSearch.session_id == row.session_id).values(result={"nested": {"value": "新\x00值"}}))
    await db.commit()
    await db.refresh(row)
    assert row.result == {"nested": {"value": "新值"}}
    nested_value = (await db.execute(select(LastImageSearch.result["nested"]["value"].as_string()).where(LastImageSearch.session_id == row.session_id))).scalar_one()
    assert nested_value == "新值"


def test_every_plugin_text_and_json_column_has_safe_binding():
    from sqlalchemy.dialects.postgresql import dialect

    from nonebot_plugin_ai_groupmate import model

    pg = dialect()
    for row_model in vars(model).values():
        table = getattr(row_model, "__table__", None)
        if table is None:
            continue
        for column in table.columns:
            raw_type = getattr(column.type, "impl", column.type)
            if isinstance(raw_type, String):
                processor = column.type.bind_processor(pg)
                assert processor is not None, f"{table.name}.{column.name} has no safe binding"
                assert processor("中\x00文🙂") == "中文🙂"
                assert processor(None) is None
            elif isinstance(raw_type, JSON):
                processor = column.type.bind_processor(pg)
                assert processor is not None
                assert "\\u0000" not in processor({"nested": ["中\x00文"]})
            # Wrapping must preserve schema types, lengths and JSON NULL behavior.
            assert column.type.compile(dialect=pg) == raw_type.compile(dialect=pg)


def test_json_key_collisions_are_rejected_instead_of_overwriting_values():
    from sqlalchemy.dialects.postgresql import dialect

    from nonebot_plugin_ai_groupmate.model import LastImageSearch

    processor = LastImageSearch.__table__.c.result.type.bind_processor(dialect())
    assert processor is not None
    payload = {"title": "first", "ti\x00tle": "second"}
    with pytest.raises(ValueError, match="keys collide"):
        processor(payload)
    assert len(payload) == 2
