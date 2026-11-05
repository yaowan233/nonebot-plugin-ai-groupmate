import io
import json
import base64
from uuid import uuid4
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image
from test_image_generation import image_config


def animation():
    output = io.BytesIO()
    frames = [Image.new("RGB", (16, 16), color) for color in ("red", "blue")]
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=200, loop=0)
    return output.getvalue()


def overlay():
    output = io.BytesIO()
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    image.putpixel((3, 3), (255, 0, 0, 255))
    image.save(output, format="PNG")
    return output.getvalue()


async def test_markup_is_separate_multipart_and_validated(monkeypatch):
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent import image_generation as module
    captured = []
    async def handle(request):
        await request.aread()
        captured.append(request.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})
    client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    cfg = image_config(image_generation_model="gpt-image-2.6-sunburst")
    await module.request_image(cfg, "change the circled area", [module.prepare_reference(png())], size="auto",
                               transparent_background=False, complexity="low", markup=(overlay(), "png", "image/png"))
    assert b'name="markup"; filename="markup.png"' in captured[0]
    assert overlay() in captured[0]
    with pytest.raises(module.ImageGenerationError) as exc:
        await module.request_image(cfg, "change", [module.prepare_reference(png())], size="auto",
                                   transparent_background=False, markup=(png(), "png", "image/png"))
    assert exc.value.code == "invalid_markup"
    assert len(captured) == 1


async def test_markup_tool_uses_original_and_owned_transparent_layer(monkeypatch, tmp_path):
    from test_image_generation import png, tool_fixture
    _, workspace, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path, image_generation_model="gpt-image-2.6-sunburst")
    original = workspace.save(png(), "original.png")
    markup = workspace.save(overlay(), "markup.png")
    result = json.loads(await generate.ainvoke({"prompt": "change the circled area", "reference_file_ids": [original.file_id],
                                               "markup_file_id": markup.file_id, "complexity": "low"}))
    assert result["status"] == "succeeded"
    assert provider.await_args is not None
    assert provider.await_args.args[2][0][0] == png()
    assert provider.await_args.kwargs["markup"][0] == overlay()
    assert provider.await_args.kwargs["complexity"] == "low"


async def test_gif_defaults_to_auto_even_if_model_picks_a_static_size(monkeypatch, tmp_path):
    from test_image_generation import tool_fixture

    _, _, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path, image_generation_model="gpt-image-2.6-sunburst")
    result = json.loads(await generate.ainvoke({"prompt": "peaceful beach animation", "output_format": "gif",
                                               "size": "1536x1024", "frame_count": 16}))
    assert result["status"] == "succeeded"
    assert provider.await_args is not None
    assert provider.await_args.kwargs["size"] == "auto"


async def test_gif_keeps_explicitly_requested_size(monkeypatch, tmp_path):
    from test_image_generation import tool_fixture

    _, _, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path, image_generation_model="gpt-image-2.6-sunburst")
    result = json.loads(await generate.ainvoke({"prompt": "peaceful beach animation", "output_format": "gif",
                                               "size": "1024x1024", "size_explicit": True}))
    assert result["status"] == "succeeded"
    assert provider.await_args is not None
    assert provider.await_args.kwargs["size"] == "1024x1024"


async def test_gif_auto_size_survives_background_queue(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    group = f"gif-auto-{uuid4()}"
    cfg = image_config(image_generation_model="gpt-image-2.6-sunburst")
    workspace = FileWorkspace(tmp_path, session_id=group, user_id="alice", bot_id="bot", is_private=False)
    monkeypatch.setattr(image_generation_tools, "is_request_active", AsyncMock(return_value=True))
    async with get_session() as db:
        generate = image_generation_tools.create_image_generation_tool(
            db, group, "request-1", config=cfg, workspace=workspace, pic_dir=tmp_path,
            current_message_id="user-message", replied_message_id=None, current_user_id="alice",
            background=True, bot_id="bot", bot_name="miyuki", platform="qq", is_private=False,
        )
        result = json.loads(await generate.ainvoke({"prompt": "peaceful beach", "output_format": "gif",
                                                   "size": "1536x1024", "frame_count": 16}))
        assert result["reason_code"] == "drawing_queued"
        job = await db.get(DrawingJob, result["data"]["job_id"])
        assert job is not None and job.image_options is not None
        assert job.size == "auto"
        assert job.image_options["frame_count"] == 16


def test_image_options_migration_preserves_existing_rows(monkeypatch):
    from importlib import import_module

    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    migration = import_module("nonebot_plugin_ai_groupmate.migrations.f3a7b9c2d614_add_image_options")
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE nonebot_plugin_ai_groupmate_drawingjob (job_id TEXT PRIMARY KEY)"))
        connection.execute(text("INSERT INTO nonebot_plugin_ai_groupmate_drawingjob (job_id) VALUES ('old')"))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        columns = {col["name"] for col in inspect(connection).get_columns("nonebot_plugin_ai_groupmate_drawingjob")}
        assert "image_options" in columns
        assert connection.execute(text("SELECT image_options FROM nonebot_plugin_ai_groupmate_drawingjob WHERE job_id='old'")).scalar_one() is None
        migration.downgrade()
        assert connection.execute(text("SELECT job_id FROM nonebot_plugin_ai_groupmate_drawingjob")).scalar_one() == "old"


async def test_gif_request_and_response(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import image_generation as module

    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(animation()).decode()}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    cfg = image_config(image_generation_model="gpt-image-2.6-sunburst")
    result = await module.request_image(cfg, "a waving cat", [], size="auto", transparent_background=False,
                                        complexity="low", output_format="gif", frame_count=2, frame_duration_ms=200)
    assert result.data == animation()
    assert result.output_format == "gif"
    assert captured[0]["output_format"] == "gif"
    assert captured[0]["complexity"] == "low"
    assert captured[0]["frame_count"] == 2
    assert captured[0]["frame_duration_ms"] == 200


async def test_gif_rejects_references_before_provider():
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError, request_image
    cfg = image_config(image_generation_model="gpt-image-2.6-sunburst")
    with pytest.raises(ImageGenerationError) as exc:
        await request_image(cfg, "animate", [(png(), "png", "image/png")], size="auto",
                            transparent_background=False, output_format="gif")
    assert exc.value.code == "invalid_image_arguments"


async def test_background_gif_options_survive_queue_and_delivery(monkeypatch, tmp_path):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate import drawing_service as service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_delivery, run_generation
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage, resolve_image_credentials

    cfg = image_config(image_generation_model="gpt-image-2.6-sunburst")
    group = f"gif-{uuid4()}"
    options = {"output_format": "gif", "complexity": "low", "frame_count": 2}
    req = replace(request(), session_id=group, reference_file_ids=(), model=cfg.image_generation_model,
                  endpoint_hash=service.endpoint_hash(resolve_image_credentials(cfg)[1]), image_options=options)
    provider = AsyncMock(return_value=GeneratedImage(animation(), {}, "gif"))
    monkeypatch.setattr(service, "request_image", provider)
    sent = []
    async def send(message, *, target, fallback):
        sent.append(message)
        from types import SimpleNamespace
        return SimpleNamespace(msg_ids=[{"message_id": "gif-message"}])
    monkeypatch.setattr(UniMessage, "send", send)
    async with get_session() as db:
        queued = await enqueue_drawing(db, req)
        await db.commit()
    assert await run_generation(queued.job_id, lambda job: service.generate_job(job, root=tmp_path, config=cfg))
    assert provider.await_args is not None
    assert provider.await_args.kwargs["output_format"] == "gif"
    assert provider.await_args.kwargs["complexity"] == "low"
    assert await run_delivery(queued.job_id, lambda job: service.deliver_job(job, root=tmp_path))
    assert len(sent) == 1
    assert any(getattr(segment, "raw", None) == animation() for segment in sent[0] if segment.type == "image")
    async with get_session() as db:
        job = await db.get(DrawingJob, queued.job_id)
        assert job is not None and job.result_file_id is not None
        workspace = FileWorkspace(tmp_path, session_id=group, user_id=req.user_id, bot_id=req.bot_id, is_private=False, peer_bot_ids=())
        artifact = workspace.get(job.result_file_id)
        assert artifact.name.endswith(".gif")
        assert artifact.content_type == "image/gif"
        assert workspace.read(job.result_file_id) == animation()
        assert workspace.read_published_image(job.result_file_id, "gif-message") == animation()


async def test_background_markup_loads_owned_overlay(monkeypatch, tmp_path):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import png

    from nonebot_plugin_ai_groupmate import drawing_service as service
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage, resolve_image_credentials

    cfg = image_config(image_generation_model="gpt-image-2.6-sunburst")
    group = f"markup-{uuid4()}"
    req = replace(request(), session_id=group, model=cfg.image_generation_model,
                  endpoint_hash=service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    workspace = FileWorkspace(tmp_path, session_id=group, user_id=req.user_id, bot_id=req.bot_id, is_private=False, peer_bot_ids=())
    original = workspace.save(png(), "original.png")
    annotation = workspace.save(overlay(), "markup.png")
    req = replace(req, reference_file_ids=(original.file_id,), image_options={"markup_file_id": annotation.file_id, "complexity": "low"})
    provider = AsyncMock(return_value=GeneratedImage(png(), {}))
    monkeypatch.setattr(service, "request_image", provider)
    async with get_session() as db:
        job = await enqueue_drawing(db, req)
        await db.commit()
    assert await run_generation(job.job_id, lambda task: service.generate_job(task, root=tmp_path, config=cfg))
    assert provider.await_args is not None
    assert provider.await_args.kwargs["markup"][0] == overlay()
    assert provider.await_args.kwargs["complexity"] == "low"
