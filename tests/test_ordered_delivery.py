import io
import json
from typing import cast
from uuid import uuid4
from types import SimpleNamespace
from importlib import import_module

from PIL import Image
from sqlalchemy import Select, update, CursorResult


def image_bytes(color):
    output = io.BytesIO()
    Image.new("RGB", (3, 3), color).save(output, format="PNG")
    return output.getvalue()


async def test_ordered_message_sends_one_chain_and_records_all_images(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import StructuredMessage
    from nonebot_plugin_ai_groupmate.reply_guard import activate_request_id, deactivate_request_id
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_resources import ImageResources
    from nonebot_plugin_ai_groupmate.agent.ordered_reply_tool import create_ordered_reply_tool
    from nonebot_plugin_ai_groupmate.agent.structured_image_resources import read_scoped_message_images

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    sent = []
    async def send(message, **_kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "1001"}])
    monkeypatch.setattr(UniMessage, "send", send)
    group, request = uuid4().hex, uuid4().hex
    workspace = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    first, second = image_bytes("red"), image_bytes("blue")
    a = workspace.save(first, "a.png")
    b = workspace.save(second, "b.png")
    parts = [{"type": "text", "text": "开头"}, {"type": "image", "resource_id": "file:" + a.file_id},
             {"type": "text", "text": "中间"}, {"type": "image", "resource_id": "file:" + b.file_id},
             {"type": "text", "text": "结尾"}]
    await activate_request_id(group, request)
    try:
        async with get_session() as db:
            send_tool, _ = create_ordered_reply_tool(db, group, request, workspace=workspace, pic_dir=tmp_path,
                approved_meme_ids=set(), send_target=None, bot_name="miyuki", members=[], current_message_id="123",
                replied_message_id=None, outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki", request), recovery_enabled=True)
            result = json.loads(await send_tool.ainvoke({"parts": parts, "reply_to": "current"}))
            assert result["delivery_state"] == "completed"
            assert len(sent) == 1
            native = await sent[0].export(adapter="OneBot V11", fallback=False)
            assert [segment.type for segment in native] == ["reply", "text", "image", "text", "image", "text"]
            row = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
            assert row is not None
            assert row.reply_to_id == "123"
            assert [part["type"] for part in row.parts] == ["text", "image", "text", "image", "text"]
            assert [part["file_id"] for part in row.parts if part["type"] == "image"] == [a.file_id, b.file_id]
            resources = ImageResources(tmp_path, workspace)
            assert await read_scoped_message_images(db, resources, session_id=group, platform="qq", is_private=False,
                message_id="1001", max_bytes=None) == [first, second]
            assert await read_scoped_message_images(db, resources, session_id=group, platform="qq", is_private=False,
                message_id="1001", max_bytes=None, image_index=2) == [second]
    finally:
        await deactivate_request_id(group, request)


async def test_waiting_drawings_deliver_together_and_can_be_cancelled(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import DrawingJob, StructuredMessage, OrderedDeliveryJob
    from nonebot_plugin_ai_groupmate.reply_guard import activate_request_id, deactivate_request_id
    from nonebot_plugin_ai_groupmate.drawing_jobs import DrawingRequest, enqueue_drawing
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.ordered_delivery_jobs import cancel_batch
    from nonebot_plugin_ai_groupmate.agent.ordered_reply_tool import create_ordered_reply_tool
    from nonebot_plugin_ai_groupmate.ordered_delivery_service import run_ordered_delivery

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    sent = []
    async def send(message, **_kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "1002"}])
    monkeypatch.setattr(UniMessage, "send", send)
    group, request = uuid4().hex, uuid4().hex
    workspace = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    await activate_request_id(group, request)
    try:
        async with get_session() as db:
            jobs = []
            for index in range(2):
                job = await enqueue_drawing(db, DrawingRequest(platform="qq", session_id=group, is_private=False,
                    bot_id="bot", bot_name="miyuki", user_id="alice", request_id=request, reply_to_id="incoming",
                    fingerprint=str(index) * 64, prompt="cat", reference_file_ids=(), size="1024x1024",
                    transparent_background=False, model="model", quality="medium", endpoint_hash="a" * 64, deferred=True))
                jobs.append(job.job_id)
            await db.commit()
            send_tool, _ = create_ordered_reply_tool(db, group, request, workspace=workspace, pic_dir=tmp_path,
                approved_meme_ids=set(), send_target=None, bot_name="miyuki", members=[], current_message_id="incoming",
                replied_message_id=None, outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki", request), recovery_enabled=True)
            parts = [{"type": "image", "resource_id": "drawing:" + jobs[0]}, {"type": "text", "text": "中间"},
                     {"type": "image", "resource_id": "drawing:" + jobs[1]}]
            result = json.loads(await send_tool.ainvoke({"parts": parts, "reply_to": "current"}))
            batch_id = result["data"]["job_id"]
            assert not sent
            for job_id, color in zip(jobs, ("red", "blue")):
                artifact = workspace.save(image_bytes(color), "drawing.png")
                await db.execute(update(DrawingJob).where(DrawingJob.job_id == job_id).values(status="generated", result_file_id=artifact.file_id))
            await db.commit()
        assert await run_ordered_delivery(batch_id, root=tmp_path / "agent_files")
        assert len(sent) == 1
        async with get_session() as db:
            row = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
            assert row is not None
            assert [part["type"] for part in row.parts] == ["image", "text", "image"]
            batch = await db.get(OrderedDeliveryJob, batch_id)
            assert batch is not None
            assert batch.status == "completed"
            assert not await cancel_batch(db, batch_id, platform="qq", session_id=group, is_private=False, user_id="alice")
        assert not await run_ordered_delivery(batch_id, root=tmp_path / "agent_files")
        assert len(sent) == 1
    finally:
        await deactivate_request_id(group, request)


async def test_cancel_during_preparation_prevents_dispatch():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob, OrderedDeliveryJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import utcnow
    from nonebot_plugin_ai_groupmate.ordered_delivery_jobs import claim_ready, cancel_batch

    batch_id, job_id, group = uuid4().hex, uuid4().hex, uuid4().hex
    now = utcnow()
    async with get_session() as db:
        db.add(OrderedDeliveryJob(job_id=batch_id, platform="qq", session_id=group, is_private=False, bot_id="bot",
            bot_name="miyuki", user_id="alice", request_id="request", fingerprint="a" * 64,
            parts=[{"type": "image", "resource_id": "drawing:" + job_id}], status="waiting", created_at=now, updated_at=now))
        db.add(DrawingJob(job_id=job_id, batch_id=batch_id, platform="qq", session_id=group, is_private=False,
            bot_id="bot", bot_name="miyuki", user_id="alice", request_id="request", fingerprint="b" * 64,
            prompt="cat", reference_file_ids=[], size="1024x1024", transparent_background=False, model="model",
            quality="medium", endpoint_hash="c" * 64, status="generated", result_file_id="d" * 32,
            created_at=now, updated_at=now))
        await db.commit()
        assert await claim_ready(db, batch_id, "worker") == "preparing"
        await db.commit()
        assert await cancel_batch(db, batch_id, platform="qq", session_id=group, is_private=False, user_id="alice")
        await db.commit()
        result = await db.execute(update(OrderedDeliveryJob).where(OrderedDeliveryJob.job_id == batch_id,
            OrderedDeliveryJob.status == "preparing", OrderedDeliveryJob.claim_token == "worker").values(status="delivering"))
        assert cast(CursorResult, result).rowcount == 0
        drawing = await db.get(DrawingJob, job_id, populate_existing=True)
        assert drawing is not None and drawing.status == "cancelled"


async def test_prepared_memes_survive_next_search_and_form_one_message(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import MediaStorage, StructuredMessage
    from nonebot_plugin_ai_groupmate.reply_guard import activate_request_id, deactivate_request_id
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.ordered_reply_tool import create_ordered_reply_tool

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    sent = []
    async def send(message, **_kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "1004"}])
    monkeypatch.setattr(UniMessage, "send", send)
    group, request = uuid4().hex, uuid4().hex
    workspace = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "a.png").write_bytes(image_bytes("red"))
    (pics / "b.png").write_bytes(image_bytes("blue"))
    approved = set()
    await activate_request_id(group, request)
    try:
        async with get_session() as db:
            a = MediaStorage(file_hash="a" * 64, file_path="a.png", description="red")
            b = MediaStorage(file_hash="b" * 64, file_path="b.png", description="blue")
            db.add_all([a, b])
            await db.flush()
            ids = (a.media_id, b.media_id)
            await db.commit()
            sender, prepare = create_ordered_reply_tool(db, group, request, workspace=workspace, pic_dir=pics,
                approved_meme_ids=approved, send_target=None, bot_name="miyuki", members=[], current_message_id="123",
                replied_message_id=None, outbound_identity=OutboundIdentity("OneBot V11", "group", "bot", "miyuki", request), recovery_enabled=True)
            resources = []
            for media_id in ids:
                approved.clear()
                approved.add(media_id)
                result = json.loads(await prepare.ainvoke({"pic_id": str(media_id)}))
                resources.append(result["data"]["resource_id"])
            result = json.loads(await sender.ainvoke({"parts": [
                {"type": "image", "resource_id": resources[0]}, {"type": "text", "text": "中间"},
                {"type": "image", "resource_id": resources[1]}]}))
            assert result["delivery_state"] == "completed"
            assert result["data"]["meme_count"] == 2
            assert len(sent) == 1
            snapshot = await db.scalar(Select(StructuredMessage).where(StructuredMessage.session_id == group))
            assert snapshot is not None
            assert [part["type"] for part in snapshot.parts] == ["image", "text", "image"]
    finally:
        await deactivate_request_id(group, request)


async def test_web_images_require_preview_and_can_be_prepared(monkeypatch, tmp_path):
    import base64

    from nonebot_plugin_ai_groupmate.agent import image_search_tools
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    class Search:
        def __init__(self, **_kwargs):
            pass

        async def ainvoke(self, _query):
            return {"images": [{"url": "https://example.com/a.png", "description": "example"}]}

    async def preview(_url):
        return "data:image/png;base64," + base64.b64encode(image_bytes("red")).decode()

    monkeypatch.setattr(image_search_tools, "TavilySearch", Search)
    monkeypatch.setattr(image_search_tools, "_download_preview", preview)
    workspace = FileWorkspace(tmp_path, session_id="group", user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    tools = image_search_tools.create_web_image_tools(None, "group", None, tavily_api_key="fixture", bot_name="miyuki",
                                                     supports_images=True, workspace=workspace, include_prepare=True)
    by_name = {item.name: item for item in tools}
    search = json.loads(await by_name["search_web_images"].ainvoke({"query": "cat"}))
    image_id = search["data"]["images"][0]["image_id"]
    before = json.loads(await by_name["prepare_web_image"].ainvoke({"image_id": image_id}))
    assert before["reason_code"] == "preview_required"
    await by_name["preview_web_images"].ainvoke({"image_ids": [image_id]})
    prepared = json.loads(await by_name["prepare_web_image"].ainvoke({"image_id": image_id}))
    assert prepared["status"] == "succeeded"
    file_id = prepared["data"]["resource_id"].removeprefix("file:")
    assert workspace.read(file_id) == image_bytes("red")


async def test_multi_image_publication_recovers_without_sending_twice(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.reply_guard import activate_request_id, deactivate_request_id
    from nonebot_plugin_ai_groupmate.agent.delivery import OutboundIdentity
    from nonebot_plugin_ai_groupmate.delivery_recovery import recover_pending
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.ordered_reply_tool import create_ordered_reply_tool

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    sends = []
    async def send(message, **_kwargs):
        sends.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "1003"}])
    monkeypatch.setattr(UniMessage, "send", send)
    group, request = uuid4().hex, uuid4().hex
    workspace = FileWorkspace(tmp_path / "agent_files", session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    first, second = image_bytes("red"), image_bytes("blue")
    a, b = workspace.save(first, "a.png"), workspace.save(second, "b.png")
    publish = workspace.publish_image
    def fail_second(file_id, message_id):
        if file_id == b.file_id:
            raise OSError("simulated publication failure")
        publish(file_id, message_id)
    monkeypatch.setattr(workspace, "publish_image", fail_second)
    await activate_request_id(group, request)
    try:
        async with get_session() as db:
            send_tool, _ = create_ordered_reply_tool(db, group, request, workspace=workspace, pic_dir=tmp_path,
                approved_meme_ids=set(), send_target=None, bot_name="miyuki", members=[], current_message_id="incoming",
                replied_message_id=None, outbound_identity=OutboundIdentity("qq", "group", "bot", "miyuki", request), recovery_enabled=True)
            result = json.loads(await send_tool.ainvoke({"parts": [
                {"type": "image", "resource_id": "file:" + a.file_id},
                {"type": "text", "text": "中间"}, {"type": "image", "resource_id": "file:" + b.file_id}]}))
            assert result["delivery_state"] == "completed"
            assert "记录未完全保存" in result["message"]
        monkeypatch.setattr(workspace, "publish_image", publish)
        assert await recover_pending(tmp_path / "delivery_recovery", tmp_path / "agent_files") == 1
        assert workspace.read_published_image(a.file_id, "1003") == first
        assert workspace.read_published_image(b.file_id, "1003") == second
        assert len(sends) == 1
    finally:
        await deactivate_request_id(group, request)


async def test_restart_recovers_preparation_but_does_not_retry_uncertain_send():
    from datetime import timedelta

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob, OrderedDeliveryJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import utcnow
    from nonebot_plugin_ai_groupmate.ordered_delivery_jobs import recover_expired

    now = utcnow()
    group = uuid4().hex
    preparing, delivering = uuid4().hex, uuid4().hex
    async with get_session() as db:
        for batch_id, status, fingerprint in ((preparing, "preparing", "a"), (delivering, "delivering", "b")):
            db.add(OrderedDeliveryJob(job_id=batch_id, platform="qq", session_id=group, is_private=False, bot_id="bot",
                bot_name="miyuki", user_id="alice", request_id="request", fingerprint=fingerprint * 64,
                parts=[{"type": "text", "text": "hello"}], status=status, claim_token="old-worker",
                created_at=now - timedelta(minutes=11), updated_at=now - timedelta(minutes=11)))
        job_id = uuid4().hex
        db.add(DrawingJob(job_id=job_id, batch_id=delivering, platform="qq", session_id=group, is_private=False,
            bot_id="bot", bot_name="miyuki", user_id="alice", request_id="request", fingerprint="c" * 64,
            prompt="cat", reference_file_ids=[], size="1024x1024", transparent_background=False, model="model",
            quality="medium", endpoint_hash="d" * 64, status="generated", result_file_id="e" * 32,
            created_at=now, updated_at=now))
        await db.commit()
        await recover_expired(db)
        await db.commit()
        first = await db.get(OrderedDeliveryJob, preparing, populate_existing=True)
        second = await db.get(OrderedDeliveryJob, delivering, populate_existing=True)
        assert first is not None and second is not None
        assert first.status == "waiting"
        assert first.claim_token is None
        assert second.status == "delivery_unknown"
        assert second.claim_token is None
        drawing = await db.get(DrawingJob, job_id, populate_existing=True)
        assert drawing is not None and drawing.status == "delivery_unknown"


def test_ordered_delivery_migration_preserves_existing_drawings():
    from sqlalchemy import text, inspect, create_engine
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine = create_engine("sqlite://")
    migration = import_module("nonebot_plugin_ai_groupmate.migrations.e9f2a3b4c615_add_ordered_delivery")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE nonebot_plugin_ai_groupmate_drawingjob (job_id VARCHAR(32) PRIMARY KEY)"))
        connection.execute(text("INSERT INTO nonebot_plugin_ai_groupmate_drawingjob VALUES ('old-job')"))
        old_op = getattr(migration, "op")
        setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        try:
            migration.upgrade()
        finally:
            setattr(migration, "op", old_op)
        assert "nonebot_plugin_ai_groupmate_ordereddeliveryjob" in inspect(connection).get_table_names()
        row = connection.execute(text("SELECT job_id, batch_id FROM nonebot_plugin_ai_groupmate_drawingjob")).one()
        assert tuple(row) == ("old-job", None)
