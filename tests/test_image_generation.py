import io
import json
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image


def png(color="blue"):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_config(**kwargs):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    return ScopedConfig(image_generation_enabled=True, chat_api_key="main-secret", chat_base_url="https://api.openai.com/v1", **kwargs)


@pytest.mark.parametrize("edit", [False, True])
async def test_images_api_wire_contract(monkeypatch, edit):
    from nonebot_plugin_ai_groupmate.agent import image_generation as module

    captured = []

    async def handle(request):
        captured.append(request)
        await request.aread()
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}], "usage": {"input_tokens": 5, "total_tokens": 9, "secret": "discard"}})

    client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    result = await module.request_image(image_config(), "draw a robot", [module.prepare_reference(png())] if edit else [], size="1024x1024", transparent_background=True)
    assert result.data == png()
    assert result.usage == {"input_tokens": 5, "total_tokens": 9}
    request = captured[0]
    assert request.url.path == ("/v1/images/edits" if edit else "/v1/images/generations")
    assert request.headers["Authorization"] == "Bearer main-secret"
    if edit:
        assert 'name="image[]"; filename="reference-1.png"' in request.content.decode("latin1")
        assert png() in request.content
    else:
        payload = json.loads(request.content)
        assert payload == {"model": "gpt-image-2.5-sunburst", "prompt": "draw a robot", "n": 1, "size": "1024x1024", "quality": "medium", "output_format": "png", "background": "transparent"}


@pytest.mark.parametrize(("status", "code"), [(400, "image_request_rejected"), (401, "image_auth_failed"), (403, "image_access_denied"), (429, "image_quota_exhausted"), (500, "image_provider_failed"), (302, "image_provider_failed")])
async def test_provider_errors_do_not_expose_body_or_credentials(monkeypatch, status, code):
    from nonebot_plugin_ai_groupmate.agent import image_generation as module

    client = httpx.AsyncClient
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, text="main-secret PRIVATE PROVIDER BODY", headers={"location": "https://other.example"})

    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    with pytest.raises(module.ImageGenerationError) as caught:
        await module.request_image(image_config(), "cat", [], size="1024x1024", transparent_background=False)
    assert caught.value.code == code
    assert "secret" not in str(caught.value)
    assert "PRIVATE" not in str(caught.value)
    assert len(calls) == 1


async def test_provider_timeout_and_response_bound(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import image_generation as module

    client = httpx.AsyncClient

    def timeout(request):
        raise httpx.ReadTimeout("main-secret", request=request)

    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(timeout), **kw))
    with pytest.raises(module.ImageGenerationError, match="超时"):
        await module.request_image(image_config(), "cat", [], size="auto", transparent_background=False)
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 10)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 11)), **kw))
    with pytest.raises(module.ImageGenerationError) as caught:
        await module.request_image(image_config(), "cat", [], size="auto", transparent_background=False)
    assert caught.value.code == "image_response_too_large"


def test_credential_pairing_and_timeout_configuration():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, agent_run_timeout_seconds
    from nonebot_plugin_ai_groupmate.runtime_config import SECRET_FIELDS
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError, resolve_image_credentials

    config = image_config(image_generation_base_url="https://other.example/v1")
    with pytest.raises(ImageGenerationError):
        resolve_image_credentials(config)
    config.image_generation_api_key = "dedicated-secret"
    assert resolve_image_credentials(config) == ("dedicated-secret", "https://other.example/v1")
    config.image_generation_base_url = ""
    assert resolve_image_credentials(config) == ("dedicated-secret", "https://api.openai.com/v1")
    config.image_generation_base_url = "https://other.example/v1?key=secret"
    with pytest.raises(ImageGenerationError):
        resolve_image_credentials(config)
    assert "image_generation_api_key" in SECRET_FIELDS
    assert agent_run_timeout_seconds(ScopedConfig()) == 180
    assert agent_run_timeout_seconds(image_config(image_generation_timeout_seconds=300)) == 390


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": [{"url": "https://other.example/private"}]}, {"data": [{"b64_json": "!!"}]}, {"data": [{"b64_json": base64.b64encode(b"not an image").decode()}]}])
def test_invalid_outputs_are_not_saved_or_downloaded(payload):
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError, decode_generated_image

    with pytest.raises(ImageGenerationError):
        decode_generated_image(payload)


def tool_fixture(monkeypatch, tmp_path, *, request_id=None, **kwargs):
    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as module
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    workspace = FileWorkspace(tmp_path / "files", session_id="group", user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    added = []
    session = SimpleNamespace(commit=AsyncMock(), add=added.append, get=AsyncMock())
    provider = AsyncMock(return_value=GeneratedImage(png(), {"total_tokens": 9}))
    monkeypatch.setattr(module, "request_image", provider)
    tool = module.create_image_generation_tool(session, "group", request_id, config=image_config(**kwargs), workspace=workspace, pic_dir=tmp_path / "pics", current_message_id="123", replied_message_id="456")
    return module, workspace, session, provider, tool, added


@pytest.mark.parametrize("private", [False, True])
async def test_generate_edit_and_native_delivery_without_paid_duplicates(monkeypatch, tmp_path, private):
    from nonebot_plugin_alconna import Target, UniMessage

    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools

    module, workspace, session, provider, generate, added = tool_fixture(monkeypatch, tmp_path)
    # Exercise the same production workspace scope in group and private messages.
    if private:
        workspace = module.FileWorkspace(tmp_path / "files", session_id="group", user_id="alice", bot_id="bot", is_private=True)
        generate = module.create_image_generation_tool(session, "group", None, config=image_config(), workspace=workspace, pic_dir=tmp_path / "pics", current_message_id="123", replied_message_id="456")
    generated = json.loads(await generate.ainvoke({"prompt": "robot"}))
    file_id = generated["data"]["file_id"]
    assert workspace.read(file_id) == png()
    assert json.loads(await generate.ainvoke({"prompt": "robot"})) == generated
    provider.assert_awaited_once()
    edited = json.loads(await generate.ainvoke({"prompt": "change to red", "reference_file_ids": [file_id]}))
    assert edited["status"] == "succeeded"
    assert provider.call_args.args[2][0][0] == png()
    assert session.commit.await_count == 2
    sent = []

    async def send(message, **kwargs):
        sent.append((message, kwargs))
        return SimpleNamespace(msg_ids=[{"message_id": 789}])

    monkeypatch.setattr(UniMessage, "send", send)
    target = Target("group", private=private)
    file_tools = create_file_tools(session, "group", None, root=tmp_path / "files", user_id="alice", bot_id="bot", is_private=private, bot_name="bot", send_target=target, current_message_id="123")
    send_file = next(tool for tool in file_tools if tool.name == "send_file")
    delivery = json.loads(await send_file.ainvoke({"file_id": file_id, "as_image": True, "text": "画好了", "reply_to": "current"}))
    assert delivery["delivery_state"] == "completed"
    assert sent[0][1]["target"] is target
    assert [segment.type for segment in sent[0][0]] == ["reply", "text", "image"]
    assert sent[0][0][-1].raw == png()
    assert len(added) == 2


async def test_failed_generation_is_not_automatically_retried(monkeypatch, tmp_path):
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError

    _, workspace, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path)
    provider.side_effect = ImageGenerationError("image_timeout", "timeout")
    assert json.loads(await generate.ainvoke({"prompt": "robot"}))["reason_code"] == "image_timeout"
    assert json.loads(await generate.ainvoke({"prompt": "robot"}))["reason_code"] == "image_generation_limit"
    provider.assert_awaited_once()
    assert not workspace.list()


async def test_reference_scope_and_missing_quote_never_fall_back(monkeypatch, tmp_path):
    module, workspace, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path)
    foreign = module.FileWorkspace(tmp_path / "files", session_id="other", user_id="bob", bot_id="bot", is_private=False, peer_bot_ids=())
    artifact = foreign.save(png(), "other.png")
    assert json.loads(await generate.ainvoke({"prompt": "edit", "reference_file_ids": [artifact.file_id]}))["reason_code"] == "file_not_found"
    lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(module, "load_replied_message_histories", lookup)
    assert json.loads(await generate.ainvoke({"prompt": "edit", "reference_message_ids": ["replied"]}))["reason_code"] == "reference_not_found"
    assert lookup.call_args.args[1:] == ("group", "456")
    provider.assert_not_awaited()
    assert not workspace.list()


async def test_expired_request_does_not_spend(monkeypatch, tmp_path):
    module, _, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path, request_id="request")
    monkeypatch.setattr(module, "is_request_active", AsyncMock(return_value=False))
    assert json.loads(await generate.ainvoke({"prompt": "robot"}))["reason_code"] == "request_expired"
    provider.assert_not_awaited()


async def test_quoted_generated_image_recovers_original_artifact(monkeypatch, tmp_path):
    module, workspace, _, provider, generate, _ = tool_fixture(monkeypatch, tmp_path)
    artifact = workspace.save(png("red"), "gpt-image.png", content_type="image/png")
    monkeypatch.setattr(module, "load_replied_message_histories", AsyncMock(return_value=[
        SimpleNamespace(content_type="bot", media_id=None, content=f"id: 456\n[图片文件] gpt-image.png file_id={artifact.file_id}"),
    ]))
    result = json.loads(await generate.ainvoke({"prompt": "remove background", "reference_message_ids": ["replied"], "transparent_background": True}))
    assert result["status"] == "succeeded"
    assert provider.call_args.args[2][0][0] == png("red")
    assert provider.call_args.kwargs["transparent_background"] is True


async def test_actual_message_lookup_respects_group_and_reads_all_reference_images(monkeypatch, tmp_path):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as module
    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "a.png").write_bytes(png())
    (pic_dir / "b.png").write_bytes(png("red"))
    provider = AsyncMock(return_value=GeneratedImage(png(), {}))
    monkeypatch.setattr(module, "request_image", provider)
    async with get_session() as session:
        for group, name in [("gpt-image-scope", "a.png"), ("gpt-image-other", "b.png")]:
            media = MediaStorage(file_path=name, file_hash=("gpt-image-" + group).ljust(64, "a"), description="test reference")
            session.add(media)
            await session.flush()
            session.add(ChatHistory(session_id=group, user_id="alice", user_name="alice", content_type="image", content="id: 123\n" + name, media_id=media.media_id))
        await session.commit()
        workspace = FileWorkspace(tmp_path / "files", session_id="gpt-image-scope", user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
        generate = module.create_image_generation_tool(session, "gpt-image-scope", None, config=image_config(), workspace=workspace, pic_dir=pic_dir, current_message_id="123", replied_message_id=None)
        result = json.loads(await generate.ainvoke({"prompt": "edit robot", "reference_message_ids": ["current"]}))
        assert result["status"] == "succeeded"
        assert [ref[0] for ref in provider.call_args.args[2]] == [png()]


def test_reference_paths_cannot_escape_picture_directory(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.image_generation import ImageGenerationError
    from nonebot_plugin_ai_groupmate.agent.image_generation_tools import _read_reference_path

    (tmp_path / "secret.png").write_bytes(png())
    with pytest.raises(ImageGenerationError):
        _read_reference_path(tmp_path / "pics", "../secret.png")


@pytest.mark.parametrize(("editing_user", "editing_bot"), [("alice", "bot"), ("bob", "bot"), ("bob", "peer")])
async def test_group_member_can_edit_image_previously_published_by_bot(monkeypatch, tmp_path, editing_user, editing_bot):
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import Target, UniMessage

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as module
    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.agent.image_generation import GeneratedImage

    group = "published-image-" + editing_user + editing_bot
    root = tmp_path / "files"
    original = module.FileWorkspace(root, session_id=group, user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    artifact = original.save(png("red"), "gpt-image.png", content_type="image/png")
    monkeypatch.setattr(UniMessage, "send", AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": -51807704}])))
    provider = AsyncMock(return_value=GeneratedImage(png(), {}))
    monkeypatch.setattr(module, "request_image", provider)
    async with get_session() as session:
        sender = next(tool for tool in create_file_tools(session, group, None, root=root, user_id="alice", bot_id="bot", is_private=False, bot_name="bot", send_target=Target(group)) if tool.name == "send_file")
        delivered = json.loads(await sender.ainvoke({"file_id": artifact.file_id, "as_image": True}))
        assert delivered["delivery_state"] == "completed"
        await session.commit()
        editor = module.FileWorkspace(root, session_id=group, user_id=editing_user, bot_id=editing_bot, is_private=False, peer_bot_ids=())
        generate = module.create_image_generation_tool(session, group, None, config=image_config(), workspace=editor, pic_dir=tmp_path / "pics", current_message_id="999", replied_message_id="-51807704")
        result = json.loads(await generate.ainvoke({"prompt": "换成圆框眼镜", "reference_message_ids": ["replied"]}))
        assert result["status"] == "succeeded", result
        assert provider.call_args.args[2][0][0] == png("red")
        followup = json.loads(await generate.ainvoke({"prompt": "再加一个红色帽子", "reference_file_ids": [artifact.file_id]}))
        assert followup["status"] == "succeeded", followup
        assert provider.call_args.args[2][0][0] == png("red")
        if editing_user != "alice":
            from nonebot_plugin_ai_groupmate.agent.file_workspace import FileTaskError

            with pytest.raises(FileTaskError):
                editor.read(artifact.file_id)


@pytest.mark.parametrize(("private", "other_scope"), [
    (False, {"session_id": "other"}),
    (False, {"is_private": True}),
    (True, {"user_id": "bob"}),
    (True, {"bot_id": "peer"}),
    (True, {"is_private": False}),
])
def test_published_images_do_not_cross_conversations_or_private_scopes(tmp_path, private, other_scope):
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileTaskError, FileWorkspace

    scope = {"session_id": "group", "user_id": "alice", "bot_id": "bot", "is_private": private, "peer_bot_ids": ()}
    original = FileWorkspace(tmp_path, **scope)
    artifact = original.save(png(), "image.png")
    original.publish_image(artifact.file_id, "123")
    assert original.read_published_image(artifact.file_id, "123") == png()
    foreign = FileWorkspace(tmp_path, **{**scope, **other_scope})
    for message_id in ("123", None):
        with pytest.raises(FileTaskError):
            foreign.read_published_image(artifact.file_id, message_id)


def test_unpublished_files_wrong_quotes_and_expired_images_are_not_shared(monkeypatch, tmp_path):
    from nonebot_plugin_ai_groupmate.agent import file_workspace as module

    original = module.FileWorkspace(tmp_path, session_id="group", user_id="alice", bot_id="bot", is_private=False, peer_bot_ids=())
    peer = module.FileWorkspace(tmp_path, session_id="group", user_id="bob", bot_id="peer", is_private=False, peer_bot_ids=())
    artifact = original.save(png(), "image.png")
    with pytest.raises(module.FileTaskError):
        peer.read_published_image(artifact.file_id)
    original.publish_image(artifact.file_id, "123")
    with pytest.raises(module.FileTaskError):
        peer.read_published_image(artifact.file_id, "124")
    assert peer.read_published_image(artifact.file_id, "123") == png()
    monkeypatch.setattr(module.time, "time", lambda: artifact.created_at + module.FILE_TTL_SECONDS + 1)
    with pytest.raises(module.FileTaskError):
        peer.read_published_image(artifact.file_id, "123")
