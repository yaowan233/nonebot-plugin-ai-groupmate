import io
import json
import base64
import random
import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw, ImageFont


def animation(colors=("red", "blue", "red", "lime"), *, format="GIF", **kwargs):
    frames = [Image.new("RGB", (100, 80), color) for color in colors]
    stream = io.BytesIO()
    frames[0].save(stream, format=format, save_all=True, append_images=frames[1:], duration=100, loop=0, **kwargs)
    return stream.getvalue()


def decode_uri(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


def pixel(image: Image.Image, point: tuple[int, int]) -> tuple[int, ...]:
    return cast(tuple[int, ...], image.getpixel(point))


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG"])
def test_animation_preserves_time_order_across_formats(format):
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    prepared = prepare_image(animation(format=format))
    assert prepared.animated
    assert prepared.source_frames == prepared.selected_frames == 4
    assert prepared.mime_type == "image/jpeg"
    sheet = decode_uri(prepared.data_uri)
    assert max(sheet.size) <= 1536
    centers = [(sheet.width * x // 4, sheet.height * y // 4) for y in (1, 3) for x in (1, 3)]
    assert [max(range(3), key=lambda channel: pixel(sheet, center)[channel]) for center in centers] == [0, 2, 0, 1]


def test_long_animation_covers_the_whole_motion_in_nine_frames():
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    prepared = prepare_image(animation([(i, i, i) for i in range(0, 80, 2)]))
    assert prepared.source_frames == 40
    assert prepared.selected_frames == 9
    sheet = decode_uri(prepared.data_uri)
    shades = [pixel(sheet, (sheet.width * (2 * col + 1) // 6, sheet.height * (2 * row + 1) // 6))[0] for row in range(3) for col in range(3)]
    assert shades == sorted(shades)
    assert abs(shades[0]) <= 1
    assert abs(shades[-1] - 78) <= 1
    steps = [b - a for a, b in zip(shades, shades[1:])]
    assert max(steps) - min(steps) <= 12


def test_apng_default_image_is_not_a_playback_frame():
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    prepared = prepare_image(animation(["red", "blue", "lime"], format="PNG", default_image=True))
    assert prepared.source_frames == 3
    assert prepared.selected_frames == 2
    sheet = decode_uri(prepared.data_uri)
    pixels = [pixel(sheet, (sheet.width // 2, sheet.height * y // 4)) for y in (1, 3)]
    assert [max(range(3), key=lambda channel: rgb[channel]) for rgb in pixels] == [2, 1]


def test_still_looking_animation_keeps_its_animation_metadata():
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    prepared = prepare_image(animation(["red", (254, 0, 0)]))
    assert prepared.animated
    assert prepared.selected_frames == 1


@pytest.mark.parametrize("frame_count", [2, 12])
def test_small_text_changes_survive_animation_preview(frame_count):
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    font = ImageFont.load_default(size=40)
    frames = []
    for index in range(frame_count):
        image = Image.new("RGB", (512, 512), "white")
        ImageDraw.Draw(image).text((220, 230), f"{index:02d}", font=font, fill="black")
        frames.append(image)
    data = io.BytesIO()
    frames[0].save(data, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)

    prepared = prepare_image(data.getvalue())

    assert prepared.source_frames == frame_count
    assert prepared.selected_frames == min(frame_count, 9)


def test_long_held_frame_is_sampled_even_among_many_short_frames():
    from PIL import ImageChops

    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    frames = [Image.new("RGB", (64, 64), "white" if i % 2 else "#eeeeee") for i in range(310)]
    frames[30] = Image.new("RGB", (64, 64), "red")
    durations = [20] * len(frames)
    durations[30] = 10_000
    data = io.BytesIO()
    frames[0].save(data, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0)

    prepared = prepare_image(data.getvalue())

    assert prepared.source_frames == 310
    red, green, _ = decode_uri(prepared.data_uri).split()
    extrema = cast(tuple[int, int], ImageChops.subtract(red, green).getextrema())
    assert extrema[1] > 100


def test_static_transparency_orientation_and_size_limits():
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    stream = io.BytesIO()
    Image.new("RGBA", (2500, 1000), (255, 0, 0, 100)).save(stream, format="PNG")
    prepared = prepare_image(stream.getvalue())
    assert prepared.mime_type == "image/png"
    assert not prepared.animated
    with Image.open(io.BytesIO(prepared.data)) as result:
        assert max(result.size) == 2048
        assert pixel(result, (10, 10))[3] == 100
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (100, 80), "red").save(stream, format="JPEG", exif=exif)
    assert decode_uri(prepare_image(stream.getvalue()).data_uri).size == (80, 100)


def test_oversized_or_invalid_images_do_not_fall_back_to_the_first_frame(monkeypatch):
    from nonebot_plugin_ai_groupmate import image_preparation
    from nonebot_plugin_ai_groupmate.agent.history_format import _image_bytes_to_data_uri

    monkeypatch.setattr(image_preparation, "MAX_SOURCE_FRAMES", 3)
    assert _image_bytes_to_data_uri(animation(), source="too-many-frames") is None
    assert _image_bytes_to_data_uri(b"not an image", source="broken") is None
    monkeypatch.setattr(image_preparation, "MAX_IMAGE_PIXELS", 100)
    assert _image_bytes_to_data_uri(animation(["red"]), source="too-many-pixels") is None


def test_large_gif_survives_ingest_compression_without_losing_frames():
    from nonebot_plugin_ai_groupmate.utils import check_and_compress_image_bytes

    rng = random.Random(20261110)
    frames = [Image.frombytes("P", (1024, 1024), rng.randbytes(1024 * 1024)) for _ in range(3)]
    for frame in frames:
        frame.putpalette([channel for i in range(256) for channel in (i, i, i)])
    stream = io.BytesIO()
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=100, optimize=False)
    original = stream.getvalue()
    assert len(original) > 2 * 1024 * 1024
    compressed = check_and_compress_image_bytes(original)
    assert compressed == original
    with Image.open(io.BytesIO(compressed)) as result:
        assert getattr(result, "n_frames", 1) == 3


@pytest.mark.parametrize("format", ["WEBP", "PNG"])
def test_other_animation_formats_survive_compression(format):
    from nonebot_plugin_ai_groupmate.utils import check_and_compress_image_bytes

    original = animation(format=format)
    assert check_and_compress_image_bytes(original, max_size_mb=0.00001) == original


def test_history_preview_uses_all_frames_even_when_extension_is_wrong(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.history_format import get_image_data_uri

    (tmp_path / "qq.jpg").write_bytes(animation())
    sheet = decode_uri(get_image_data_uri("qq.jpg", pic_dir=tmp_path))
    assert sheet.width > 1000
    assert pixel(sheet, (sheet.width * 3 // 4, sheet.height // 4))[2] > 240


async def test_current_and_quoted_previews_are_built_off_the_event_loop(tmp_path, monkeypatch):
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    original_converter = agent._image_bytes_to_data_uri
    event_thread = threading.get_ident()
    calls = []

    def convert(raw, *, source):
        assert threading.get_ident() != event_thread
        calls.append(source)
        return original_converter(raw, source=source)

    monkeypatch.setattr(agent, "_image_bytes_to_data_uri", convert)
    monkeypatch.setattr(agent, "_chat_supports_images", lambda config: True)
    result = await agent._build_multimodal_prompt(
        "看这个动图",
        history=[],
        replied_extra=[],
        chat_config=ScopedConfig(),
        image_workspace=FileWorkspace(tmp_path, session_id="animation-prompt", user_id="u", bot_id="b", is_private=False),
        vision_metrics=agent.VisionRunMetrics(),
        structured_reply_images=[animation()],
        structured_reply_failed=False,
        structured_current_images=[animation()],
        structured_current_failed=False,
        structured_quote=None,
        quote_lookup_failed=False,
        reply_to_id="quoted",
        structured_messages={},
        include_legacy_current_images=True,
    )
    assert isinstance(result, list)
    images = [part["image_url"]["url"] for part in result if part["type"] == "image_url"]
    assert len(images) == 2
    assert all(decode_uri(uri).width > 1000 for uri in images)
    assert calls == ["structured_reply", "structured_current"]


async def test_ingest_uses_actual_format_and_requeues_a_recovered_original(tmp_path, monkeypatch):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import SceneType
    from nonebot_plugin_alconna import Image as IncomingImage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage
    from nonebot_plugin_ai_groupmate.image_preparation import ANIMATION_ANNOTATION_VERSION

    original = animation()
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    monkeypatch.setattr(plugin, "image_fetch", AsyncMock(return_value=original))
    session = SimpleNamespace(scene=SimpleNamespace(id="animation-ingest", type=SceneType.GROUP), user=SimpleNamespace(id="u"))
    async with get_session() as db:
        await plugin.process_image_message(db, IncomingImage(id="misleading.jpg"), cast(Any, None), cast(Any, None), {}, cast(Any, session), "user", "id: anim-ingest\n")
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".gif"
    assert files[0].read_bytes() == original
    async with get_session() as db:
        row = (await db.execute(Select(MediaStorage).where(MediaStorage.file_hash == plugin.generate_file_hash(original)))).scalar_one()
        media_id = row.media_id
        row.file_path = "missing-old.jpg"
        row.animation_version = ANIMATION_ANNOTATION_VERSION
        row.vectorized = True
        row.description = "旧描述"
        await db.commit()
    files[0].unlink()
    async with get_session() as db:
        await plugin.process_image_message(db, IncomingImage(id="misleading.jpg"), cast(Any, None), cast(Any, None), {}, cast(Any, session), "user", "id: anim-ingest-again\n")
        recovered = await db.get(MediaStorage, media_id)
        assert recovered is not None
        assert recovered.file_path == files[0].name
        assert recovered.animation_version == 0
        assert recovered.animation_attempted_at is None
    assert files[0].read_bytes() == original


@pytest.mark.parametrize("description", ["猫挥手", "动图：猫挥手", "动图: 猫挥手", "动图：动图: 猫挥手"])
def test_animation_mark_is_added_exactly_once(description):
    from nonebot_plugin_ai_groupmate.image_preparation import mark_animation_description

    assert mark_animation_description(description) == "动图：猫挥手"
    assert mark_animation_description("动图：  ") == ""


async def test_reannotation_updates_description_and_vectors_and_retries_failures(tmp_path, monkeypatch):
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    path = tmp_path / "old.jpg"
    original = animation()
    path.write_bytes(original)
    async with get_session() as db:
        row = MediaStorage(file_hash="old-animation-reannotation", file_path=path.name, description="红色画面", references=3, vectorized=True, embedding_version=0)
        db.add(row)
        await db.flush()
        media_id = row.media_id
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"is_meme": True, "description": "动图: 画面红蓝交替，最后变绿"}))))
    insert = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    monkeypatch.setattr(plugin.DB, "enabled", True)
    monkeypatch.setattr(plugin.DB, "insert_media", insert)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    assert await plugin._process_media_vectorization(media_id) == "failed"
    async with get_session() as db:
        unchanged = await db.get(MediaStorage, media_id)
        assert unchanged is not None
        assert unchanged.description == "红色画面"
        assert unchanged.animation_version == 0
    assert await plugin._process_media_vectorization(media_id) == "indexed"
    assert await plugin._process_media_vectorization(media_id) == "skipped"
    assert model.ainvoke.await_count == 2
    assert model.ainvoke.await_args is not None
    assert insert.await_args is not None
    content = model.ainvoke.await_args.args[0][0].content
    assert "关键帧" in content[0]["text"]
    assert decode_uri(content[1]["image_url"]["url"]).width > 1000
    assert insert.await_args.args[1] == content[1]["image_url"]["url"]
    assert insert.await_args.args[2] == "动图：画面红蓝交替，最后变绿"
    async with get_session() as db:
        updated = await db.get(MediaStorage, media_id)
        assert updated is not None
        assert updated.description == "动图：画面红蓝交替，最后变绿"
        assert updated.embedding_version == plugin.DB.media_embedding_version
        assert updated.animation_version == plugin.ANIMATION_ANNOTATION_VERSION
    assert path.read_bytes() == original


@pytest.fixture
async def animation_database(tmp_path, tmp_path_factory, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage, HistoryMessageIndex

    # Concurrent workers need independent SQL connections/transactions.
    database = tmp_path_factory.mktemp("animation-databases") / "test.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    async with engine.begin() as connection:
        await connection.run_sync(MediaStorage.__table__.create)
        await connection.run_sync(ChatHistory.__table__.create)
        await connection.run_sync(HistoryMessageIndex.__table__.create)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(plugin, "get_session", factory)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    monkeypatch.setattr(plugin.DB, "enabled", True)
    monkeypatch.setattr(plugin.DB, "media_embedding_version", 3)
    monkeypatch.setattr(plugin.plugin_config, "media_vectorize_min_references", 3)
    monkeypatch.setattr(plugin.plugin_config, "media_vectorize_batch_size", 20)
    monkeypatch.setattr(plugin.plugin_config, "media_vectorize_concurrency", 2)
    try:
        yield factory
    finally:
        await engine.dispose()


async def test_background_automatically_repairs_only_historical_memes_once(tmp_path, monkeypatch, animation_database):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage
    from nonebot_plugin_ai_groupmate.image_preparation import ANIMATION_ANNOTATION_VERSION

    rows = []
    for name, raw, references, description, version in (
        ("gif", animation(), 3, "旧描述", 0),
        ("webp", animation(format="WEBP"), 3, "旧描述", 0),
        ("apng", animation(format="PNG"), 3, "旧描述", 0),
        ("still", animation(["red"]), 3, "静态描述", 0),
        ("low-reference", animation(), 1, "旧描述", 0),
        ("not-meme", animation(), 3, "[图片]", 0),
        ("already-done", animation(), 3, "动图：已有动作描述", ANIMATION_ANNOTATION_VERSION),
        ("missing", None, 3, "旧描述", 0),
    ):
        if raw is not None:
            (tmp_path / name).write_bytes(raw)
        rows.append(MediaStorage(file_hash=name, file_path=name, description=description, references=references, vectorized=True, embedding_version=3, animation_version=version))
    async with animation_database() as db:
        db.add_all(rows)
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"is_meme":true,"description":"画面变化"}')))
    insert = AsyncMock(return_value=True)
    monkeypatch.setattr(plugin.DB, "insert_media", insert)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    await plugin._vectorize_media_impl()
    assert model.ainvoke.await_count == insert.await_count == 3
    async with animation_database() as db:
        for original in rows:
            updated = await db.get(MediaStorage, original.media_id)
            assert updated is not None
            if original.file_path in {"gif", "webp", "apng"}:
                assert updated.description == "动图：画面变化"
            if original.file_path in {"low-reference", "not-meme"}:
                assert updated.animation_version == 0
            else:
                assert updated.animation_version == ANIMATION_ANNOTATION_VERSION
        static = await db.get(MediaStorage, rows[3].media_id)
        assert static is not None
        assert static.description == "静态描述"
        assert static.embedding_version == 3
    await plugin._vectorize_media_impl()
    assert model.ainvoke.await_count == insert.await_count == 3


async def test_existing_animation_label_does_not_skip_old_vector_upgrade(tmp_path, monkeypatch, animation_database):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage
    from nonebot_plugin_ai_groupmate.image_preparation import ANIMATION_ANNOTATION_VERSION

    (tmp_path / "old.gif").write_bytes(animation())
    async with animation_database() as db:
        row = MediaStorage(file_hash="prefixed-old", file_path="old.gif", description="动图：旧描述",
                           references=3, vectorized=True, embedding_version=3, animation_version=0)
        db.add(row)
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"is_meme":true,"description":"新的动作描述"}')))
    insert = AsyncMock(return_value=True)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    monkeypatch.setattr(plugin.DB, "insert_media", insert)

    assert await plugin._process_media_vectorization(row.media_id) == "indexed"
    assert model.ainvoke.await_count == insert.await_count == 1
    async with animation_database() as db:
        updated = await db.get(MediaStorage, row.media_id)
        assert updated is not None
        assert updated.description == "动图：新的动作描述"
        assert updated.animation_version == ANIMATION_ANNOTATION_VERSION
    assert await plugin._process_media_vectorization(row.media_id) == "skipped"
    assert insert.await_count == 1


async def test_background_retries_failures_after_unattempted_records(tmp_path, monkeypatch, animation_database):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage
    from nonebot_plugin_ai_groupmate.image_preparation import ANIMATION_ANNOTATION_VERSION

    monkeypatch.setattr(plugin.plugin_config, "media_vectorize_batch_size", 1)
    (tmp_path / "old.gif").write_bytes(animation())
    async with animation_database() as db:
        rows = [MediaStorage(file_hash=f"retry-{i}", file_path="old.gif", description="旧描述", references=3, vectorized=True, embedding_version=3) for i in range(2)]
        db.add_all(rows)
        await db.commit()
    response = SimpleNamespace(content='{"is_meme":true,"description":"画面变化"}')
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[RuntimeError("temporary failure"), response, response]))
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    monkeypatch.setattr(plugin.DB, "insert_media", AsyncMock(return_value=True))
    await plugin._vectorize_media_impl()
    assert model.ainvoke.await_count == 1
    async with animation_database() as db:
        failed = await db.get(MediaStorage, rows[0].media_id)
        untouched = await db.get(MediaStorage, rows[1].media_id)
        assert failed is not None
        assert untouched is not None
        assert failed.animation_version == 0
        assert failed.animation_attempted_at is not None
        assert untouched.animation_attempted_at is None
    await plugin._vectorize_media_impl()
    async with animation_database() as db:
        failed = await db.get(MediaStorage, rows[0].media_id)
        completed = await db.get(MediaStorage, rows[1].media_id)
        assert failed is not None
        assert completed is not None
        assert failed.animation_version == 0
        assert completed.animation_version == ANIMATION_ANNOTATION_VERSION
    await plugin._vectorize_media_impl()
    async with animation_database() as db:
        retried = await db.get(MediaStorage, rows[0].media_id)
        assert retried is not None
        assert retried.animation_version == ANIMATION_ANNOTATION_VERSION
    await plugin._vectorize_media_impl()
    assert model.ainvoke.await_count == 3


@pytest.mark.parametrize("historical", [True, False])
async def test_backfill_preserves_known_stickers_but_new_images_are_still_classified(tmp_path, monkeypatch, animation_database, historical):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    (tmp_path / "pig.gif").write_bytes(animation())
    async with animation_database() as db:
        row = MediaStorage(file_hash="negative-classification", file_path="pig.gif", description="卡通小猪" if historical else "[图片]", references=3, vectorized=historical, embedding_version=3 if historical else 0)
        db.add(row)
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"is_meme":false,"description":"小猪反复翻转跳跃，适合撒娇卖萌。"}')))
    insert = AsyncMock(return_value=True)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    monkeypatch.setattr(plugin.DB, "insert_media", insert)
    await plugin._vectorize_media_impl()
    assert model.ainvoke.await_count == 1
    assert insert.await_count == (1 if historical else 0)
    async with animation_database() as db:
        updated = await db.get(MediaStorage, row.media_id)
        assert updated is not None
        assert updated.description == ("动图：小猪反复翻转跳跃，适合撒娇卖萌。" if historical else "[图片]")


def test_animation_progress_migration_preserves_historical_data():
    from importlib import import_module

    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = import_module("nonebot_plugin_ai_groupmate.migrations.a9d4e7c2b615_add_animation_annotation_progress")
    engine = create_engine("sqlite:///:memory:")
    table = "nonebot_plugin_ai_groupmate_mediastorage"
    with engine.begin() as connection:
        connection.execute(text(f"CREATE TABLE {table} (media_id INTEGER PRIMARY KEY, description TEXT NOT NULL)"))
        connection.execute(text(f"INSERT INTO {table} VALUES (1, '旧描述')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert connection.execute(text(f"SELECT description, animation_version, animation_attempted_at FROM {table}")).one() == ("旧描述", 0, None)
            connection.execute(text(f"INSERT INTO {table} (media_id, description) VALUES (2, '另一个描述')"))
            assert connection.execute(text(f"SELECT animation_version FROM {table} WHERE media_id=2")).scalar_one() == 0
            migration.downgrade()
        assert {column["name"] for column in inspect(connection).get_columns(table)} == {"media_id", "description"}
        assert connection.execute(text(f"SELECT description FROM {table} ORDER BY media_id")).scalars().all() == ["旧描述", "另一个描述"]
    engine.dispose()


@pytest.mark.parametrize("active_version", [3, 4], ids=["multimodal", "text"])
@pytest.mark.parametrize("already_annotated", [False, True])
async def test_embedding_upgrade_retags_old_animations_and_reuses_new_descriptions(tmp_path, monkeypatch, active_version, already_annotated):
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    path = tmp_path / "legacy.jpg"
    path.write_bytes(animation(format="WEBP"))
    description = "动图：猫挥手" if already_annotated else "猫"
    async with get_session() as db:
        row = MediaStorage(file_hash=f"upgrade-animation-{active_version}-{already_annotated}", file_path=path.name,
                           description=description, references=3, vectorized=True, embedding_version=0,
                           animation_version=plugin.ANIMATION_ANNOTATION_VERSION if already_annotated else 0)
        db.add(row)
        await db.flush()
        media_id = row.media_id
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"is_meme":true,"description":"猫挥手"}')))
    insert = AsyncMock(return_value=True)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    monkeypatch.setattr(plugin.DB, "enabled", True)
    monkeypatch.setattr(plugin.DB, "media_embedding_version", active_version)
    monkeypatch.setattr(plugin.DB, "insert_media", insert)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    assert await plugin._process_media_vectorization(media_id) == "indexed"
    assert model.ainvoke.await_count == (0 if already_annotated else 1)
    assert insert.await_args is not None
    assert insert.await_args.args[2] == "动图：猫挥手"
    assert decode_uri(insert.await_args.args[1]).width > 1000
    async with get_session() as db:
        updated = await db.get(MediaStorage, media_id)
        assert updated is not None
        assert updated.description == "动图：猫挥手"
        assert updated.embedding_version == active_version


async def test_animations_sharing_the_first_frame_remain_distinct(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.meme_tools import _deduplicate_meme_pic_ids

    (tmp_path / "a.gif").write_bytes(animation(["red", "blue"]))
    (tmp_path / "b.gif").write_bytes(animation(["red", "lime"]))
    media = {1: SimpleNamespace(file_path="a.gif"), 2: SimpleNamespace(file_path="b.gif")}
    assert await _deduplicate_meme_pic_ids([1, 2], media, tmp_path) == [1, 2]


async def test_visual_review_gets_readable_animated_candidates_and_a_frame_budget(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.meme_tools import _rerank_meme_candidates_for_context
    from nonebot_plugin_ai_groupmate.agent.meme_visuals import build_meme_review_images

    colors = [(i * 20, i * 20, i * 20) for i in range(13)]
    for i in range(3):
        (tmp_path / f"{i}.gif").write_bytes(animation(colors))
    blocks, ids = build_meme_review_images(tmp_path, [(i, f"{i}.gif") for i in range(3)])
    assert ids == {0, 1, 2}
    images = [part["image_url"]["url"] for part in blocks if part["type"] == "image_url"]
    assert len(images) == 3
    assert all(decode_uri(uri).width > 1000 for uri in images)

    class Model:
        async def ainvoke(self, messages, **kwargs):
            assert "关键帧" in messages[0].content
            assert messages[1].content[1:] == blocks
            return SimpleNamespace(content='{"should_send":true,"candidates":[{"pic_id":0,"relevance":0.9},{"pic_id":99,"relevance":1}]}')

    approved = await _rerank_meme_candidates_for_context(Model(), search_intent="画面变化", history=[], candidates=[(i, "动图") for i in range(3)], candidate_images=blocks)
    assert approved == [(0, 0.9)]


def test_animation_budget_keeps_later_static_candidates(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.meme_visuals import build_meme_review_images

    for i in range(2):
        (tmp_path / f"{i}.gif").write_bytes(animation([(j * 20,) * 3 for j in range(13)]))
    for i in range(2, 6):
        Image.new("RGB", (40, 40), "blue").save(tmp_path / f"{i}.png")
    candidates = [(i, f"{i}.gif" if i < 2 else f"{i}.png") for i in range(6)]
    blocks, ids = build_meme_review_images(tmp_path, candidates)
    assert ids == set(range(6))
    assert len([part for part in blocks if part["type"] == "image_url"]) == 3


def test_reduced_frame_budget_still_covers_the_whole_animation():
    from nonebot_plugin_ai_groupmate.image_preparation import prepare_image

    prepared = prepare_image(animation([(j * 20,) * 3 for j in range(13)]), max_frames=6)
    assert prepared.selected_frames == 6
    sheet = decode_uri(prepared.data_uri)
    assert pixel(sheet, (sheet.width // 4, sheet.height // 6))[0] <= 1
    assert abs(pixel(sheet, (sheet.width * 3 // 4, sheet.height * 5 // 6))[0] - 240) <= 1


@pytest.mark.parametrize("queue", ["new", "reindex", "animation"])
@pytest.mark.parametrize("failure", ["pixels", "frames", "decode_budget", "bomb", "invalid"])
async def test_permanent_preview_failure_does_not_block_the_queue(tmp_path, monkeypatch, animation_database, queue, failure):
    from collections import Counter

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import image_preparation
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    (tmp_path / "bad.gif").write_bytes(b"invalid" if failure == "invalid" else animation())
    Image.new("RGB", (1, 1), "red").save(tmp_path / "good.png")
    if failure == "bomb":
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    else:
        option = {"pixels": "MAX_IMAGE_PIXELS", "frames": "MAX_SOURCE_FRAMES", "decode_budget": "MAX_ANIMATION_PIXELS"}.get(failure)
        if option:
            monkeypatch.setattr(image_preparation, option, 1)
    monkeypatch.setattr(plugin.plugin_config, "media_vectorize_batch_size", 1)
    async with animation_database() as db:
        rows = [MediaStorage(file_hash=name, file_path=name, references=3, vectorized=queue != "new", description="[图片]" if queue == "new" else "旧描述", embedding_version=3 if queue == "animation" else 0) for name in ("bad.gif", "good.png")]
        db.add_all(rows)
        await db.commit()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"is_meme":true,"description":"红色图片"}')))
    insert = AsyncMock(return_value=True)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: model)
    monkeypatch.setattr(plugin.DB, "insert_media", insert)
    original_reader = plugin._read_prepared_image
    reads = Counter()

    def read(path, **kwargs):
        reads[path.name] += 1
        return original_reader(path, **kwargs)

    monkeypatch.setattr(plugin, "_read_prepared_image", read)
    await plugin._vectorize_media_impl()
    model.ainvoke.assert_not_awaited()
    insert.assert_not_awaited()
    async with animation_database() as db:
        skipped = await db.get(MediaStorage, rows[0].media_id)
        assert skipped is not None
        assert skipped.vectorized is True
        assert skipped.animation_version == plugin.ANIMATION_ANNOTATION_VERSION
        assert skipped.embedding_version == plugin.DB.media_embedding_version
    await plugin._vectorize_media_impl()
    await plugin._vectorize_media_impl()
    assert reads["bad.gif"] == 1
    assert reads["good.png"] == 1
    assert model.ainvoke.await_count == (1 if queue == "new" else 0)
    assert insert.await_count == (0 if queue == "animation" else 1)


async def test_file_io_failures_remain_retryable(tmp_path, monkeypatch, animation_database):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    Image.new("RGB", (1, 1), "red").save(tmp_path / "recover.png")
    async with animation_database() as db:
        row = MediaStorage(file_hash="io-error", file_path="recover.png", references=3, description="[图片]")
        db.add(row)
        await db.commit()
    original = plugin._read_prepared_image

    def unavailable(*args, **kwargs):
        raise OSError("temporary filesystem failure")

    monkeypatch.setattr(plugin, "_read_prepared_image", unavailable)
    assert await plugin._process_media_vectorization(row.media_id) == "failed"
    async with animation_database() as db:
        pending = await db.get(MediaStorage, row.media_id)
        assert pending is not None
        assert pending.vectorized is False
        assert pending.animation_version == 0
    monkeypatch.setattr(plugin, "_read_prepared_image", original)
    monkeypatch.setattr(plugin, "get_tagging_model", lambda: SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"is_meme":true,"description":"红色图片"}'))))
    monkeypatch.setattr(plugin.DB, "insert_media", AsyncMock(return_value=True))
    assert await plugin._process_media_vectorization(row.media_id) == "indexed"


async def test_static_backfill_does_not_encode_a_preview(tmp_path, monkeypatch, animation_database):
    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage

    Image.new("RGB", (100, 100), "red").save(tmp_path / "still.png")
    async with animation_database() as db:
        row = MediaStorage(file_hash="static-inspection", file_path="still.png", references=3, description="已有描述", vectorized=True, embedding_version=3)
        db.add(row)
        await db.commit()

    def fail_if_encoding(*args, **kwargs):
        raise AssertionError("static backfill must only inspect the header")

    monkeypatch.setattr(plugin, "prepare_image", fail_if_encoding)
    await plugin._vectorize_media_impl()
    async with animation_database() as db:
        checked = await db.get(MediaStorage, row.media_id)
        assert checked is not None
        assert checked.animation_version == plugin.ANIMATION_ANNOTATION_VERSION
        assert checked.description == "已有描述"


async def test_existing_mislabeled_gif_is_reused_without_an_orphan(tmp_path, monkeypatch, animation_database):
    from sqlalchemy import Select
    from nonebot_plugin_alconna import Image as IncomingImage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage

    raw = animation()
    digest = plugin.generate_file_hash(raw)
    old_name = digest + ".jpg"
    (tmp_path / old_name).write_bytes(raw)
    async with animation_database() as db:
        row = MediaStorage(file_hash=digest, file_path=old_name, references=3, description="动图：已有描述", vectorized=True, embedding_version=3, animation_version=plugin.ANIMATION_ANNOTATION_VERSION)
        db.add(row)
        await db.commit()
    monkeypatch.setattr(plugin, "image_fetch", AsyncMock(return_value=raw))
    session = SimpleNamespace(scene=SimpleNamespace(id="reuse-old-gif"), user=SimpleNamespace(id="u"))
    async with animation_database() as db:
        for message_id in ("701", "702"):
            await plugin.process_image_message(db, IncomingImage(id="no-extension"), cast(Any, None), cast(Any, None), {}, cast(Any, session), "user", f"id: {message_id}\n")
        stored = await db.get(MediaStorage, row.media_id)
        assert stored is not None
        assert stored.file_path == old_name
        assert stored.references == 5
        assert stored.animation_version == plugin.ANIMATION_ANNOTATION_VERSION
        history = (await db.execute(Select(ChatHistory))).scalars().all()
        assert history
        assert all(item.content.endswith(old_name) for item in history)
    assert {path.name for path in tmp_path.iterdir()} == {old_name}


async def test_file_embedding_queries_use_the_same_bounded_preview_as_indexing(tmp_path):
    from nonebot_plugin_ai_groupmate.memory import VectorDBOperator
    from nonebot_plugin_ai_groupmate.image_preparation import MAX_MODEL_BYTES, prepare_image

    rng = random.Random(20261110)
    frames = [Image.frombytes("P", (1024, 1024), rng.randbytes(1024 * 1024)) for _ in range(3)]
    for frame in frames:
        frame.putpalette([channel for shade in range(256) for channel in (shade, shade, shade)])
    output = io.BytesIO()
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=100, optimize=False)
    raw = output.getvalue()
    assert len(raw) > 2 * 1024 * 1024
    path = tmp_path / "mislabeled.jpg"
    path.write_bytes(raw)
    preview = prepare_image(raw)
    operator = VectorDBOperator.__new__(VectorDBOperator)
    operator.enabled = True
    operator.text_only = False
    operator.media_multivector_col = "test-media"
    operator.client = cast(Any, SimpleNamespace(upsert=AsyncMock()))
    operator._ensure_collections = AsyncMock()
    request = AsyncMock(side_effect=[[[0.1], [0.2]], [[0.3]]])
    operator._request_qwen_vl_embeddings = request
    operator._search_media_routes = AsyncMock(return_value=[(10, 0.9)])
    assert await operator.insert_media(1, preview.data_uri, "动画描述")
    assert await operator.search_similar_meme(str(path)) == [10]
    indexed_image = request.await_args_list[0].args[0][1]["image"]
    queried_image = request.await_args_list[1].args[0][0]["image"]
    assert queried_image == indexed_image == preview.data_uri
    assert len(base64.b64decode(queried_image.split(",", 1)[1])) <= MAX_MODEL_BYTES


@pytest.mark.parametrize("groupmate_head", ["f3a7b9c2d614", "a8b9c0d1e2f3"])
def test_animation_migration_upgrades_both_existing_heads(groupmate_head):
    from pathlib import Path
    from importlib import import_module

    from sqlalchemy import text, create_engine
    from alembic.script import ScriptDirectory
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = import_module("nonebot_plugin_ai_groupmate.migrations.a9d4e7c2b615_add_animation_annotation_progress")
    assert migration.__file__ is not None
    directory = str(Path(migration.__file__).parent)
    script = ScriptDirectory(dir=directory, version_locations=[directory])
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE nonebot_plugin_ai_groupmate_mediastorage (media_id INTEGER PRIMARY KEY, description TEXT NOT NULL)"))
        connection.execute(text("INSERT INTO nonebot_plugin_ai_groupmate_mediastorage VALUES (1, '旧描述')"))
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
        connection.execute(text("INSERT INTO alembic_version VALUES (:head),('d6b9a3e2c710')"), {"head": groupmate_head})
        context = MigrationContext.configure(connection, opts={"fn": lambda heads, ctx: script._upgrade_revs("heads", heads)})
        with Operations.context(context):
            context.run_migrations()
            context.run_migrations()
        assert set(connection.execute(text("SELECT version_num FROM alembic_version")).scalars()) == {"b1e2a3c4d5f6", "d6b9a3e2c710"}
        assert connection.execute(text("SELECT description, animation_version FROM nonebot_plugin_ai_groupmate_mediastorage")).one() == ("旧描述", 0)
    engine.dispose()


async def test_forward_urls_and_base64_use_animation_previews(tmp_path, monkeypatch):
    import httpx

    from nonebot_plugin_ai_groupmate.agent import image_search_tools
    from nonebot_plugin_ai_groupmate.agent.forward_tools import _forward_image_source
    from nonebot_plugin_ai_groupmate.agent.history_format import _image_bytes_to_data_uri

    raw = animation()
    preview = _image_bytes_to_data_uri(raw, source="forward-test")
    downloader = AsyncMock(return_value=preview)
    monkeypatch.setattr(image_search_tools, "_download_preview", downloader)
    bot = cast(Any, SimpleNamespace(call_api=AsyncMock(return_value={})))
    url_result = await _forward_image_source(bot, {"url": "https://example.com/qq-image"})
    assert url_result == preview
    downloader.assert_awaited_once_with("https://example.com/qq-image")
    assert decode_uri(url_result).width > 1000
    encoded = base64.b64encode(raw).decode("ascii")
    assert await _forward_image_source(bot, {"file": "base64://" + encoded}) == preview
    downloader.side_effect = httpx.ReadTimeout("timeout")
    assert await _forward_image_source(bot, {"url": "https://example.com/expired.gif"}) is None
