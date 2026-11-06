from uuid import uuid4
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Select


async def test_real_drawing_adapters_preserve_reference_and_reply(monkeypatch, tmp_path):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate import drawing_service as service
    from nonebot_plugin_ai_groupmate.model import DrawingJob, ChatHistory
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_delivery, run_generation
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage, resolve_image_credentials

    cfg = image_config()
    group = f"draw-service-{uuid4()}"
    workspace = FileWorkspace(tmp_path, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    reference = workspace.save(png("red"), "reference.png")
    req = replace(request(), session_id=group, reference_file_ids=(reference.file_id,),
                  model=cfg.image_generation_model, quality=cfg.image_generation_quality,
                  endpoint_hash=service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    provider = AsyncMock(return_value=GeneratedImage(png("blue"), {}))
    monkeypatch.setattr(service, "request_image", provider)
    captured = []
    async def send(message, *, target, fallback):
        captured.append((message, target, fallback))
        return SimpleNamespace(msg_ids=[{"message_id": "sent-drawing"}])
    monkeypatch.setattr(UniMessage, "send", send)
    async with get_session() as db:
        job = await enqueue_drawing(db, req)
        await db.commit()
    assert await run_generation(job.job_id, lambda task: service.generate_job(task, root=tmp_path, config=cfg))
    assert provider.await_count == 1
    assert provider.await_args is not None
    assert provider.await_args.args[2][0][0] == png("red")
    assert await run_delivery(job.job_id, lambda task: service.deliver_job(task, root=tmp_path))
    assert not await run_delivery(job.job_id, lambda task: service.deliver_job(task, root=tmp_path))
    assert len(captured) == 1
    message, target, fallback = captured[0]
    assert target.id == group
    assert target.self_id == "bot"
    assert not fallback
    assert any(segment.type == "reply" and str(segment.id) == "message-1" for segment in message)
    async with get_session() as db:
        saved = await db.get(DrawingJob, job.job_id)
        assert saved is not None
        assert saved.status == "completed"
        assert saved.result_file_id is not None
        rows = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == group))).all()
        assert len(rows) == 1
        assert "sent-drawing" in rows[0].content
        viewer = FileWorkspace(tmp_path, session_id=group, user_id="bob", bot_id="other", is_private=False, peer_bot_ids=())
        assert viewer.read_published_image(saved.result_file_id, "sent-drawing") == png("blue")


async def test_changed_provider_is_not_called(monkeypatch, tmp_path):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate import drawing_service as service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation

    cfg = image_config()
    provider = AsyncMock()
    monkeypatch.setattr(service, "request_image", provider)
    async with get_session() as db:
        job = await enqueue_drawing(db, replace(request(), reference_file_ids=(), endpoint_hash="0" * 64))
        await db.commit()
    assert not await run_generation(job.job_id, lambda task: service.generate_job(task, root=tmp_path, config=cfg))
    provider.assert_not_awaited()
    async with get_session() as db:
        saved = await db.get(DrawingJob, job.job_id)
        assert saved is not None
        assert saved.status == "failed_notice_pending"
        assert saved.error_code == "provider_changed"


@pytest.mark.parametrize(("code", "message"), [
    ("image_request_rejected", "没被服务接受"),
    ("image_auth_failed", "绘图暂时用不了"),
    ("image_access_denied", "绘图暂时用不了"),
    ("image_quota_exhausted", "限流或额度不足"),
])
async def test_explicit_provider_rejection_is_reported_without_retry(monkeypatch, tmp_path, code, message):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation, run_failure_notice
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError, resolve_image_credentials

    cfg = image_config()
    item = replace(request(), reference_file_ids=(), endpoint_hash=drawing_service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    provider = AsyncMock(side_effect=ImageGenerationError(code, "private provider detail"))
    sent = []
    async def send(outgoing, **_kwargs):
        sent.append(outgoing)
        return SimpleNamespace(msg_ids=[{"message_id": "rejected-notice"}])
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(drawing_service, "request_image", provider)
    monkeypatch.setattr(UniMessage, "send", send)
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        await db.commit()
    assert not await run_generation(job.job_id, lambda task: drawing_service.generate_job(task, root=tmp_path, config=cfg))
    assert await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert not await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert provider.await_count == 1
    assert len(sent) == 1
    assert message in str(sent[0])
    assert "private provider detail" not in str(sent[0])
    async with get_session() as db:
        saved = await db.get(DrawingJob, job.job_id)
        assert saved is not None
        assert saved.status == "failed"
        assert saved.error_code == code


@pytest.mark.parametrize("code", ["image_timeout", "image_provider_failed", "invalid_image_response"])
async def test_uncertain_provider_error_remains_unknown(monkeypatch, tmp_path, code):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate import drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError, resolve_image_credentials

    cfg = image_config()
    item = replace(request(), reference_file_ids=(), endpoint_hash=drawing_service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    provider = AsyncMock(side_effect=ImageGenerationError(code, "private provider detail"))
    monkeypatch.setattr(drawing_service, "request_image", provider)
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        await db.commit()
    assert not await run_generation(job.job_id, lambda task: drawing_service.generate_job(task, root=tmp_path, config=cfg))
    assert provider.await_count == 1
    async with get_session() as db:
        saved = await db.get(DrawingJob, job.job_id)
        assert saved is not None
        assert saved.status == "interrupted_notice_pending"
        assert saved.error_code == "generation_outcome_unknown"


async def test_missing_reference_fails_before_provider_call(monkeypatch, tmp_path):
    import json

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate import drawing_service as service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation
    from nonebot_plugin_ai_groupmate.agent.image_generation import resolve_image_credentials
    from nonebot_plugin_ai_groupmate.agent.drawing_management import create_drawing_management_tools

    cfg = image_config()
    provider = AsyncMock()
    monkeypatch.setattr(service, "request_image", provider)
    item = replace(request(), reference_file_ids=("a" * 32,), endpoint_hash=service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        await db.commit()
    assert not await run_generation(job.job_id, lambda task: service.generate_job(task, root=tmp_path, config=cfg))
    provider.assert_not_awaited()
    async with get_session() as db:
        saved = await db.get(DrawingJob, job.job_id)
        assert saved is not None
        assert saved.status == "failed_notice_pending"
        assert saved.error_code == "reference_unavailable"
        tools = create_drawing_management_tools(db, platform=item.platform, session_id=item.session_id,
                                               is_private=item.is_private, user_id=item.user_id, request_id=None)
        result = json.loads(await tools[0].ainvoke({}))
        details = next(entry for entry in result["data"]["jobs"] if entry["job_id"] == job.job_id)
        assert details["status_message"] == "绘图未开始，等待通知"
        assert details["error_message"] == "参考图已失效或无法读取"


async def test_failed_drawing_notifies_original_conversation_once(monkeypatch, tmp_path):
    import asyncio

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import drawing_poll, drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob, ChatHistory
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation

    item = replace(request(), reference_file_ids=(), endpoint_hash="0" * 64, bot_id=f"notice-bot-{uuid4()}")
    cfg = image_config()
    sent = []
    async def send(message, **kwargs):
        sent.append((message, kwargs))
        return SimpleNamespace(msg_ids=[{"message_id": "failure-notice"}])
    provider = AsyncMock()
    monkeypatch.setattr(UniMessage, "send", send)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(drawing_service, "request_image", provider)
    monkeypatch.setattr(drawing_poll, "get_bots", lambda: {item.bot_id: SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))})
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        await db.commit()
    assert not await run_generation(job.job_id, lambda task: drawing_service.generate_job(task, root=tmp_path, config=cfg))
    provider.assert_not_awaited()
    drawing_poll.start_drawing_jobs()
    try:
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        await asyncio.sleep(0)
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        assert len(sent) == 1
        message, kwargs = sent[0]
        assert kwargs["target"].id == item.session_id
        assert kwargs["target"].self_id == item.bot_id
        assert any(segment.type == "reply" and str(segment.id) == item.reply_to_id for segment in message)
        assert "绘图暂时用不了" in str(message)
        async with get_session() as db:
            stored = await db.get(DrawingJob, job.job_id)
            assert stored is not None
            assert stored.status == "failed"
            assert stored.result_message_id == "failure-notice"
            history = (await db.scalars(Select(ChatHistory).where(ChatHistory.session_id == item.session_id))).all()
            assert len(history) == 1
            assert "failure-notice" in history[0].content
    finally:
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()


async def test_failure_notice_unknown_never_resends(monkeypatch, tmp_path):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate import drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import DrawingPreflightFailed, run_generation, run_failure_notice

    sender = AsyncMock(side_effect=TimeoutError("unknown platform outcome"))
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        job = await enqueue_drawing(db, replace(request(), reference_file_ids=()))
        await db.commit()
    async def fail_before_provider(_job):
        raise DrawingPreflightFailed("reference_unavailable")
    assert not await run_generation(job.job_id, fail_before_provider)
    assert not await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert not await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert sender.await_count == 1
    async with get_session() as db:
        stored = await db.get(DrawingJob, job.job_id)
        assert stored is not None
        assert stored.status == "failed_notice_unknown"
        assert stored.error_code == "reference_unavailable"


async def test_unknown_provider_outcome_notifies_without_retry(monkeypatch, tmp_path):
    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation, run_failure_notice

    sent = []
    async def send(message, **_kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "uncertain-notice"}])
    provider = AsyncMock(side_effect=TimeoutError("provider result unknown"))
    monkeypatch.setattr(UniMessage, "send", send)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    async with get_session() as db:
        job = await enqueue_drawing(db, replace(request(), reference_file_ids=()))
        await db.commit()
    assert not await run_generation(job.job_id, provider)
    assert await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert not await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert provider.await_count == 1
    assert len(sent) == 1
    assert "没法确认服务有没有收到" in str(sent[0])
    async with get_session() as db:
        stored = await db.get(DrawingJob, job.job_id)
        assert stored is not None
        assert stored.status == "interrupted"
        assert stored.error_code == "generation_outcome_unknown"


async def test_missing_generated_image_notifies_without_retrying_provider(monkeypatch, tmp_path):
    import json

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate import drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_delivery, run_generation, run_failure_notice
    from nonebot_plugin_ai_groupmate.agent.drawing_management import create_drawing_management_tools

    sent = []
    async def send(message, **_kwargs):
        sent.append(message)
        return SimpleNamespace(msg_ids=[{"message_id": "missing-image-notice"}])
    provider = AsyncMock(return_value="c" * 32)
    monkeypatch.setattr(UniMessage, "send", send)
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    item = replace(request(), reference_file_ids=())
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        await db.commit()
    assert await run_generation(job.job_id, provider)
    assert not await run_delivery(job.job_id, lambda task: drawing_service.deliver_job(task, root=tmp_path))
    assert not sent
    async with get_session() as db:
        stored = await db.get(DrawingJob, job.job_id)
        assert stored is not None
        assert stored.status == "failed_notice_pending"
        assert stored.error_code == "generated_image_unavailable"
        tools = create_drawing_management_tools(db, platform=item.platform, session_id=item.session_id,
                                               is_private=item.is_private, user_id=item.user_id, request_id=None)
        details = json.loads(await tools[0].ainvoke({}))["data"]["jobs"][0]
        assert details["status_message"] == "图片已生成但无法读取，等待通知"
    assert await run_failure_notice(job.job_id, drawing_service.deliver_failure_notice)
    assert len(sent) == 1
    assert "图已经画好了" in str(sent[0])
    assert provider.await_count == 1


async def test_ready_delivery_is_not_hidden_by_a_large_generation_queue(monkeypatch, tmp_path):
    import asyncio

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate import drawing_poll
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing, claim_generation, complete_generation

    item = replace(request(), bot_id=f"backlog-bot-{uuid4()}", reference_file_ids=())
    async with get_session() as db:
        for index in range(101):
            await enqueue_drawing(db, replace(item, request_id=f"queued-{index}"))
        ready = await enqueue_drawing(db, replace(item, request_id="ready"))
        assert await claim_generation(db, ready.job_id, "owner")
        assert await complete_generation(db, ready.job_id, "owner", "c" * 32) == "generated"
        await db.commit()
    generate = AsyncMock(return_value=True)
    deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(drawing_poll, "run_generation", generate)
    monkeypatch.setattr(drawing_poll, "run_delivery", deliver)
    monkeypatch.setattr(drawing_poll, "get_bots", lambda: {item.bot_id: SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))})
    drawing_poll.start_drawing_jobs()
    try:
        await drawing_poll.poll_drawing_jobs(tmp_path, image_config())
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        assert deliver.await_count == 1
        assert deliver.await_args is not None
        assert deliver.await_args.args[0] == ready.job_id
        assert generate.await_count == 1
    finally:
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()


async def test_poll_runs_queued_and_delivers_after_worker_restart(monkeypatch, tmp_path):
    import asyncio

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate import drawing_poll, drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage, resolve_image_credentials

    cfg = image_config()
    group = f"draw-poll-{uuid4()}"
    bot_id = f"bot-{uuid4()}"
    req = replace(request(), session_id=group, bot_id=bot_id, reference_file_ids=(), model=cfg.image_generation_model,
                  quality=cfg.image_generation_quality,
                  endpoint_hash=drawing_service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    bot = SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))
    monkeypatch.setattr(drawing_poll, "get_bots", lambda: {bot_id: bot})
    provider = AsyncMock(return_value=GeneratedImage(png(), {}))
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "sent-after-restart"}]))
    monkeypatch.setattr(drawing_service, "request_image", provider)
    monkeypatch.setattr(UniMessage, "send", sender)
    async with get_session() as db:
        job = await enqueue_drawing(db, req)
        await db.commit()
    drawing_poll.start_drawing_jobs()
    try:
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        await asyncio.sleep(0)
        async with get_session() as db:
            saved = await db.get(DrawingJob, job.job_id)
            assert saved is not None
            assert saved.status == "generated"
        # No task is held by the original chat; a later poll can deliver it.
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        await asyncio.sleep(0)
        async with get_session() as db:
            saved = await db.get(DrawingJob, job.job_id)
            assert saved is not None
            assert saved.status == "completed"
        assert provider.await_count == 1
        assert sender.await_count == 1
    finally:
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()


async def test_background_tool_queues_then_delivers_without_waiting(monkeypatch, tmp_path):
    import json
    import asyncio

    from nonebot_plugin_orm import get_session
    from test_image_generation import png, image_config
    from nonebot_plugin_alconna import UniMessage

    from nonebot_plugin_ai_groupmate import drawing_poll, drawing_runner, drawing_service
    from nonebot_plugin_ai_groupmate.agent import delivery, image_generation_tools
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.concurrency import agent_run_gate
    from nonebot_plugin_ai_groupmate.message_trace import trace_id
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    cfg = image_config()
    group = f"draw-tool-{uuid4()}"
    bot_id = f"bot-{uuid4()}"
    workspace = FileWorkspace(tmp_path, session_id=group, user_id="alice", bot_id=bot_id, is_private=False, peer_bot_ids=())
    reference = workspace.save(png("red"), "source.png")
    monkeypatch.setattr(image_generation_tools, "is_request_active", AsyncMock(return_value=True))
    traces = []
    trace_logger = SimpleNamespace(info=traces.append, warning=lambda *args, **kwargs: None)
    monkeypatch.setattr(image_generation_tools, "logger", trace_logger)
    monkeypatch.setattr(drawing_runner, "logger", trace_logger)
    monkeypatch.setattr(delivery, "logger", trace_logger)
    started, finish = asyncio.Event(), asyncio.Event()
    async def generate(*args, **kwargs):
        started.set()
        await finish.wait()
        return GeneratedImage(png("blue"), {})
    provider = AsyncMock(side_effect=generate)
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "generated-image"}]))
    monkeypatch.setattr(drawing_service, "request_image", provider)
    monkeypatch.setattr(UniMessage, "send", sender)
    monkeypatch.setattr(drawing_poll, "get_bots", lambda: {bot_id: SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))})
    async with get_session() as db:
        tool = image_generation_tools.create_image_generation_tool(
            db, group, "request-1", config=cfg, workspace=workspace, pic_dir=tmp_path,
            current_message_id="user-message", replied_message_id=None, current_user_id="alice",
            background=True, bot_id=bot_id, bot_name="miyuki", platform="qq", is_private=False,
        )
        arguments = {"prompt": "把这张图改成蓝色", "reference_file_ids": [reference.file_id]}
        result = json.loads(await tool.ainvoke(arguments))
        assert result["reason_code"] == "drawing_queued"
        assert result["data"]["job_status"] == "queued"
        assert json.loads(await tool.ainvoke(arguments))["data"]["job_id"] == result["data"]["job_id"]
        provider.assert_not_awaited()
        sender.assert_not_awaited()
    drawing_poll.start_drawing_jobs()
    try:
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        await asyncio.wait_for(started.wait(), timeout=2)
        async with agent_run_gate.slot() as acquired:
            assert acquired
        finish.set()
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        await asyncio.sleep(0)
        await drawing_poll.poll_drawing_jobs(tmp_path, cfg)
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        await asyncio.sleep(0)
        assert provider.await_count == 1
        assert provider.await_args is not None
        assert provider.await_args.args[2][0][0] == png("red")
        assert sender.await_count == 1
        trace_lines = [line for line in traces if "[MessageTrace]" in line]
        for stage in ("drawing_queued", "drawing_generation_started", "drawing_generation_finished",
                      "drawing_delivery_started", "delivery_ack", "drawing_delivery_finished"):
            assert any(f"trace={trace_id(group, 'request-1')} stage={stage}" in line for line in trace_lines)
        assert all("把这张图改成蓝色" not in line and "request-1" not in line for line in trace_lines)
        async with get_session() as db:
            job = await db.get(DrawingJob, result["data"]["job_id"])
            assert job is not None
            assert job.status == "completed"
            assert job.reply_to_id == "user-message"
    finally:
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()


async def test_job_management_only_shows_and_cancels_own_jobs():
    import json

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing
    from nonebot_plugin_ai_groupmate.agent.drawing_management import create_drawing_management_tools

    item = request()
    async with get_session() as db:
        own = await enqueue_drawing(db, item)
        other = await enqueue_drawing(db, replace(item, user_id="bob"))
        await db.commit()
        tools = create_drawing_management_tools(db, platform="qq", session_id=item.session_id,
                                               is_private=False, user_id="alice", request_id=None)
        status, cancel = (next(tool for tool in tools if tool.name == name)
                          for name in ("get_drawing_jobs", "cancel_drawing_job"))
        result = json.loads(await status.ainvoke({}))
        assert [job["job_id"] for job in result["data"]["jobs"]] == [own.job_id]
        forbidden = json.loads(await cancel.ainvoke({"job_id": other.job_id}))
        assert forbidden["reason_code"] == "drawing_not_cancellable"
        done = json.loads(await cancel.ainvoke({}))
        assert done["data"]["job_status"] == "cancelled"
        other_row = await db.get(DrawingJob, other.job_id, populate_existing=True)
        assert other_row is not None
        assert other_row.status == "queued"


def test_background_groups_accept_webui_input():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    cfg = ScopedConfig.model_validate({"image_generation_background_groups": "123456， 789012,123456"})
    assert cfg.image_generation_background_groups == ["123456", "789012"]
    assert ScopedConfig.model_validate({"image_generation_background_groups": ""}).image_generation_background_groups == []


async def test_cancel_while_waiting_for_provider_slot_does_not_generate(monkeypatch, tmp_path):
    import asyncio

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate import drawing_service
    from nonebot_plugin_ai_groupmate.model import DrawingJob
    from nonebot_plugin_ai_groupmate.drawing_jobs import cancel_drawing, enqueue_drawing
    from nonebot_plugin_ai_groupmate.drawing_runner import run_generation
    from nonebot_plugin_ai_groupmate.agent.image_generation import resolve_image_credentials
    from nonebot_plugin_ai_groupmate.agent.image_generation_tools import _generation_gate

    cfg = image_config()
    item = replace(request(), reference_file_ids=(), model=cfg.image_generation_model,
                   quality=cfg.image_generation_quality,
                   endpoint_hash=drawing_service.endpoint_hash(resolve_image_credentials(cfg)[1]))
    provider = AsyncMock()
    monkeypatch.setattr(drawing_service, "request_image", provider)
    async with get_session() as db:
        job = await enqueue_drawing(db, item)
        await db.commit()
    claimed = asyncio.Event()
    async def generation_after_claim(row):
        claimed.set()
        return await drawing_service.generate_job(row, root=tmp_path, config=cfg)
    async with _generation_gate:
        task = asyncio.create_task(run_generation(job.job_id, generation_after_claim))
        await asyncio.wait_for(claimed.wait(), timeout=5)
        async with get_session() as db:
            assert await cancel_drawing(db, job.job_id, platform=item.platform, session_id=item.session_id,
                                        is_private=False, user_id=item.user_id) == "cancel_requested"
            await db.commit()
    assert await task
    provider.assert_not_awaited()
    async with get_session() as db:
        row = await db.get(DrawingJob, job.job_id)
        assert row is not None
        assert row.status == "cancelled"


async def test_same_group_deliveries_are_serialized(monkeypatch, tmp_path):
    import asyncio
    from datetime import timedelta

    from test_drawing_jobs import request
    from nonebot_plugin_orm import get_session
    from test_image_generation import image_config

    from nonebot_plugin_ai_groupmate import drawing_poll
    from nonebot_plugin_ai_groupmate.drawing_jobs import enqueue_drawing, claim_generation, complete_generation

    item = replace(request(), bot_id=f"bot-{uuid4()}")
    async with get_session() as db:
        first = await enqueue_drawing(db, item)
        second = await enqueue_drawing(db, replace(item, request_id="second"))
        for job in (first, second):
            assert await claim_generation(db, job.job_id, "owner")
            assert await complete_generation(db, job.job_id, "owner", "c" * 32) == "generated"
        await db.commit()
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def deliver(task, *, root):
        calls.append(task.job_id)
        started.set()
        await finish.wait()
        return "sent"
    monkeypatch.setattr(drawing_poll, "deliver_job", deliver)
    monkeypatch.setattr(drawing_poll, "get_bots", lambda: {item.bot_id: SimpleNamespace(adapter=SimpleNamespace(get_name=lambda: "qq"))})
    clock = [drawing_poll.utcnow()]
    monkeypatch.setattr(drawing_poll, "utcnow", lambda: clock[0])
    drawing_poll.start_drawing_jobs()
    try:
        await drawing_poll.poll_drawing_jobs(tmp_path, image_config())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert len(calls) == 1
        assert len(drawing_poll._running) == 1
        finish.set()
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        await asyncio.sleep(0)
        await drawing_poll.poll_drawing_jobs(tmp_path, image_config())
        assert calls == [first.job_id]
        clock[0] += timedelta(seconds=drawing_poll.GROUP_DELIVERY_INTERVAL_SECONDS + 60)
        await drawing_poll.poll_drawing_jobs(tmp_path, image_config())
        await asyncio.gather(*(task for _, task in list(drawing_poll._running.values())))
        assert set(calls) == {first.job_id, second.job_id}
    finally:
        await drawing_poll.stop_drawing_jobs()
        drawing_poll.start_drawing_jobs()
