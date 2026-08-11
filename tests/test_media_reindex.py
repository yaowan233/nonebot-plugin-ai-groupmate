from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from contextlib import asynccontextmanager

import pytest


class FakeScalars:
    def __init__(self, values: list[int]):
        self.values = values

    def all(self) -> list[int]:
        return self.values


class FakeResult:
    def __init__(self, values: list[int]):
        self.values = values

    def scalars(self) -> FakeScalars:
        return FakeScalars(self.values)


class FakeSession:
    def __init__(self, media: Any):
        self.media = media
        self.execute_count = 0
        self.commit_count = 0
        self.rollback_count = 0

    async def execute(self, _statement: Any) -> FakeResult:
        self.execute_count += 1
        if self.execute_count == 1:
            return FakeResult([])
        return FakeResult([self.media.media_id])

    async def get(self, _model: Any, media_id: int) -> Any:
        assert media_id == self.media.media_id
        return self.media

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    def add(self, _media: Any) -> None:
        return None


@pytest.mark.parametrize(
    ("stored_version", "active_version"),
    [(0, 3), (3, 4), (4, 3)],
    ids=["legacy", "multimodal-to-text", "text-to-multimodal"],
)
@pytest.mark.asyncio
async def test_vectorize_media_reindexes_mismatched_version_without_retagging(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    stored_version: int,
    active_version: int,
):
    import nonebot_plugin_ai_groupmate as plugin_module

    media = SimpleNamespace(
        media_id=42,
        file_path="legacy.png",
        description="熊猫头流泪，表达无奈",
        vectorized=True,
        embedding_version=stored_version,
    )
    session = FakeSession(media)
    (tmp_path / media.file_path).write_bytes(b"legacy-image")

    @asynccontextmanager
    async def fake_get_session():
        yield session

    insert_calls: list[tuple[int, str, str]] = []

    async def fake_insert_media(media_id: int, image_data: str, description: str) -> bool:
        insert_calls.append((media_id, image_data, description))
        return True

    def fail_if_tagging_model_is_used():
        raise AssertionError("旧表情包重建不应再次调用视觉标注模型")

    monkeypatch.setattr(plugin_module, "get_session", fake_get_session)
    monkeypatch.setattr(plugin_module, "pic_dir", tmp_path)
    monkeypatch.setattr(plugin_module.DB, "enabled", True)
    monkeypatch.setattr(
        plugin_module.DB,
        "media_embedding_version",
        active_version,
    )
    monkeypatch.setattr(plugin_module.DB, "insert_media", fake_insert_media)
    monkeypatch.setattr(plugin_module, "get_tagging_model", fail_if_tagging_model_is_used)

    await plugin_module.vectorize_media()

    assert len(insert_calls) == 1
    assert insert_calls[0][0] == media.media_id
    assert insert_calls[0][1].startswith("data:image/png;base64,")
    assert insert_calls[0][2] == media.description
    assert media.embedding_version == active_version
    assert session.rollback_count == 0


@pytest.mark.asyncio
async def test_missing_legacy_media_is_marked_current_and_not_retried(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    import nonebot_plugin_ai_groupmate as plugin_module

    media = SimpleNamespace(
        media_id=42,
        file_path="missing.png",
        description="已经丢失的旧表情包",
        vectorized=True,
        embedding_version=0,
    )
    session = FakeSession(media)

    @asynccontextmanager
    async def fake_get_session():
        yield session

    monkeypatch.setattr(plugin_module, "get_session", fake_get_session)
    monkeypatch.setattr(plugin_module, "pic_dir", tmp_path)
    monkeypatch.setattr(plugin_module.DB, "enabled", True)
    monkeypatch.setattr(plugin_module.DB, "media_embedding_version", 3)

    result = await plugin_module._process_media_vectorization(media.media_id)

    assert result == "missing"
    assert media.vectorized is True
    assert media.embedding_version == 3


@pytest.mark.asyncio
async def test_vectorize_media_uses_configured_concurrency(
    monkeypatch: pytest.MonkeyPatch,
):
    import nonebot_plugin_ai_groupmate as plugin_module

    class DiscoverySession:
        def __init__(self) -> None:
            self.execute_count = 0

        async def execute(self, _statement: Any) -> FakeResult:
            self.execute_count += 1
            return FakeResult([1, 2, 3, 4, 5] if self.execute_count == 1 else [])

        async def commit(self) -> None:
            return None

    session = DiscoverySession()

    @asynccontextmanager
    async def fake_get_session():
        yield session

    active = 0
    max_active = 0

    async def fake_process(_media_id: int) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "indexed"

    monkeypatch.setattr(plugin_module, "get_session", fake_get_session)
    monkeypatch.setattr(plugin_module.DB, "enabled", True)
    monkeypatch.setattr(plugin_module, "_process_media_vectorization", fake_process)
    monkeypatch.setattr(plugin_module.plugin_config, "media_vectorize_batch_size", 10)
    monkeypatch.setattr(plugin_module.plugin_config, "media_vectorize_min_references", 1)
    monkeypatch.setattr(plugin_module.plugin_config, "media_vectorize_concurrency", 3)

    await plugin_module._vectorize_media_impl()

    assert max_active == 3


@pytest.mark.asyncio
async def test_vectorize_media_summarizes_missing_legacy_files(
    monkeypatch: pytest.MonkeyPatch,
):
    import nonebot_plugin_ai_groupmate as plugin_module

    class DiscoverySession:
        def __init__(self) -> None:
            self.execute_count = 0

        async def execute(self, _statement: Any) -> FakeResult:
            self.execute_count += 1
            return FakeResult([] if self.execute_count == 1 else [1, 2])

        async def commit(self) -> None:
            return None

    @asynccontextmanager
    async def fake_get_session():
        yield DiscoverySession()

    async def fake_process(_media_id: int) -> str:
        return "missing"

    warning_messages: list[str] = []
    info_messages: list[str] = []
    fake_logger = SimpleNamespace(
        debug=lambda _message: None,
        info=info_messages.append,
        warning=warning_messages.append,
    )
    monkeypatch.setattr(plugin_module, "get_session", fake_get_session)
    monkeypatch.setattr(plugin_module, "logger", fake_logger)
    monkeypatch.setattr(plugin_module.DB, "enabled", True)
    monkeypatch.setattr(plugin_module, "_process_media_vectorization", fake_process)
    monkeypatch.setattr(plugin_module.plugin_config, "media_vectorize_batch_size", 10)
    monkeypatch.setattr(plugin_module.plugin_config, "media_vectorize_min_references", 1)
    monkeypatch.setattr(plugin_module.plugin_config, "media_vectorize_concurrency", 3)

    await plugin_module._vectorize_media_impl()

    assert warning_messages == [
        "本轮跳过 2 个文件缺失的历史表情包，已停止后续重建重试"
    ]
    assert info_messages[-1] == "本轮图片处理完成：成功 0，跳过 2，失败待重试 0"


@pytest.mark.asyncio
async def test_media_vectorization_skips_all_work_without_qdrant(
    monkeypatch: pytest.MonkeyPatch,
):
    import nonebot_plugin_ai_groupmate as plugin_module

    def fail_if_session_is_opened():
        raise AssertionError("Qdrant 未启用时不应查询媒体或启动模型任务")

    monkeypatch.setattr(plugin_module.DB, "enabled", False)
    monkeypatch.setattr(plugin_module, "get_session", fail_if_session_is_opened)

    await plugin_module.vectorize_media()
    await plugin_module._vectorize_media_impl()
    result = await plugin_module._process_media_vectorization(42)

    assert result == "disabled"
