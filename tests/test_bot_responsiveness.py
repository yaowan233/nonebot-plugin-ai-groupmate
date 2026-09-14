import asyncio
import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_image_and_text_complete_with_one_database_connection(monkeypatch, tmp_path):
    import nonebot_plugin_ai_groupmate as plugin

    pool = asyncio.Semaphore(1)
    image_has_connection = asyncio.Event()
    continue_image = asyncio.Event()
    group_id = "image-text-lock-order"
    media = SimpleNamespace(media_id=1, references=1)

    class Session:
        def __init__(self, *, image=False):
            self.image = image
            self.connected = False
            self.calls = 0

        async def execute(self, _statement):
            if not self.connected:
                await pool.acquire()
                self.connected = True
            self.calls += 1
            if self.image and self.calls == 1:
                image_has_connection.set()
                await continue_image.wait()
            return SimpleNamespace(scalar_one_or_none=lambda: media if self.calls == 1 else None)

        def add(self, _row):
            pass

        async def flush(self):
            pass

        async def refresh(self, _row):
            pass

        async def commit(self):
            if self.connected:
                self.connected = False
                pool.release()

        rollback = commit

    image_session = Session(image=True)
    text_session = Session()
    monkeypatch.setattr(plugin, "image_fetch", AsyncMock(return_value=b"test image"))
    monkeypatch.setattr(plugin, "check_and_compress_image_bytes", lambda data, **_: data)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)

    async def text_message():
        # The same lock / transaction order used by handle_message.
        async with plugin._get_dedup_lock(group_id):
            await text_session.execute(None)
            await text_session.commit()

    image_task = asyncio.create_task(plugin.process_image_message(
        image_session, cast(Any, SimpleNamespace(id="image.png")), cast(Any, None), cast(Any, None), {},
        cast(Any, SimpleNamespace(scene=SimpleNamespace(id=group_id), user=SimpleNamespace(id="user"))),
        "tester", "id: 123\n",
    ))
    text_task = None
    try:
        await asyncio.wait_for(image_has_connection.wait(), timeout=1)
        text_task = asyncio.create_task(text_message())
        await asyncio.sleep(0)
        continue_image.set()
        await asyncio.wait_for(asyncio.gather(image_task, text_task), timeout=0.5)
    finally:
        image_task.cancel()
        if text_task is not None:
            text_task.cancel()
        await asyncio.gather(image_task, *([text_task] if text_task else []), return_exceptions=True)
        await image_session.rollback()
        await text_session.rollback()
        plugin._dedup_locks.pop(group_id, None)


@pytest.mark.asyncio
async def test_rag_tokenizer_loading_does_not_stop_event_loop(monkeypatch):
    from datetime import datetime

    from nonebot_plugin_ai_groupmate import utils
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    release = threading.Event()
    entered = threading.Event()
    encoder = SimpleNamespace(encode=lambda text: list(text), decode=lambda tokens: "".join(tokens))

    def slow_get_encoder():
        session.commit.assert_awaited()
        entered.set()
        release.wait(timeout=0.3)
        return encoder

    row = ChatHistorySchema(
        msg_id=1, session_id="tokenizer-loop", user_id="user", user_name="tester",
        content="hello", content_type="text", created_at=datetime.now(), media_id=None,
        vectorized=False,
    )
    result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [row]))
    session = SimpleNamespace(execute=AsyncMock(return_value=result), commit=AsyncMock())
    monkeypatch.setattr(utils, "get_encoder", slow_get_encoder)
    task = asyncio.create_task(utils.split_chat_into_context_groups(cast(Any, session), row.session_id))
    responsive = False

    def heartbeat():
        nonlocal responsive
        if not entered.is_set() and not task.done():
            asyncio.get_running_loop().call_later(0.01, heartbeat)
            return
        responsive = entered.is_set() and not task.done()
        release.set()

    handle = asyncio.get_running_loop().call_later(0.05, heartbeat)
    try:
        await task
        await asyncio.sleep(0.06)
        assert responsive, "RAG tokenizer blocked unrelated event-loop callbacks"
    finally:
        release.set()
        handle.cancel()


@pytest.mark.asyncio
async def test_image_ingestion_still_persists_and_deduplicates(monkeypatch, tmp_path):
    from sqlalchemy import delete, select
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage

    image_bytes = b"responsiveness-test-image"
    file_hash = plugin.generate_file_hash(image_bytes)
    group_id = "image-ingestion-responsiveness"
    monkeypatch.setattr(plugin, "image_fetch", AsyncMock(return_value=image_bytes))
    monkeypatch.setattr(plugin, "check_and_compress_image_bytes", lambda data, **_: data)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    session_info = SimpleNamespace(scene=SimpleNamespace(id=group_id), user=SimpleNamespace(id="user"))

    async with get_session() as session:
        try:
            for message_id in ["123", "456"]:
                await plugin.process_image_message(
                    session, cast(Any, SimpleNamespace(id="image.png")), cast(Any, None), cast(Any, None), {},
                    cast(Any, session_info), "tester", f"id: {message_id}\n",
                )
            media = (await session.execute(select(MediaStorage).where(MediaStorage.file_hash == file_hash))).scalar_one()
            history = (await session.execute(select(ChatHistory).where(ChatHistory.session_id == group_id))).scalar_one()
            assert media.references == 2
            assert history.media_id == media.media_id
            assert "id: 123\n" in history.content
            assert "alias_id: 456\n" in history.content
        finally:
            await session.rollback()
            await session.execute(delete(ChatHistory).where(ChatHistory.session_id == group_id))
            await session.execute(delete(MediaStorage).where(MediaStorage.file_hash == file_hash))
            await session.commit()
            plugin._dedup_locks.pop(group_id, None)
