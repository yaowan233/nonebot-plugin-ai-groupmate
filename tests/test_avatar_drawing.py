import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


async def test_member_avatar_is_passed_as_image_reference_not_just_text(monkeypatch, tmp_path):
    from test_image_generation import png, image_config, tool_fixture

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as module
    from nonebot_plugin_ai_groupmate.agent.web_fetch import Download

    _, workspace, session, provider, old_tool, _ = tool_fixture(monkeypatch, tmp_path)
    assert "reference_user_ids" in old_tool.args, "The drawing tool has no way to receive a member avatar"
    avatar = png("green")
    download = AsyncMock(return_value=Download(avatar, "https://avatars.example/bob.png", "avatar.png", "image/png"))
    monkeypatch.setattr(module, "download_public_url", download)
    generate = module.create_image_generation_tool(session, "group", None, config=image_config(), workspace=workspace, pic_dir=tmp_path, current_message_id="123", replied_message_id=None,
                                                   current_user_id="100", group_members=[SimpleNamespace(id="200", user=SimpleNamespace(avatar="https://avatars.example/bob.png"))])
    result = json.loads(await generate.ainvoke({"prompt": "按他的头像画一张水彩画", "reference_user_ids": ["200"]}))
    assert result["status"] == "succeeded"
    assert provider.call_args.args[2][0][0] == avatar
    assert len(provider.call_args.args[2]) == 1
    download.assert_awaited_once_with("https://avatars.example/bob.png")


@pytest.mark.parametrize("failure", ["unknown_member", "download_failed", "invalid_image", "too_many"])
async def test_unavailable_avatar_never_falls_back_to_unreferenced_generation(monkeypatch, tmp_path, failure):
    from test_image_generation import png, image_config, tool_fixture

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as module
    from nonebot_plugin_ai_groupmate.agent.web_fetch import Download
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileTaskError

    _, workspace, session, provider, _, _ = tool_fixture(monkeypatch, tmp_path)
    download = AsyncMock(return_value=Download(b"invalid" if failure == "invalid_image" else png(), "https://example.org/avatar", "avatar.png", "image/png"))
    if failure == "download_failed":
        download.side_effect = FileTaskError("download_failed", "download failed")
    monkeypatch.setattr(module, "download_public_url", download)
    generate = module.create_image_generation_tool(session, "group", None, config=image_config(), workspace=workspace, pic_dir=tmp_path, current_message_id=None, replied_message_id=None, current_user_id="100")
    ids = ["200"] if failure == "unknown_member" else ["1", "2", "3", "4", "5"] if failure == "too_many" else ["current"]
    result = json.loads(await generate.ainvoke({"prompt": "draw an avatar", "reference_user_ids": ids}))
    assert result["status"] == "failed"
    provider.assert_not_awaited()
    if failure in {"unknown_member", "too_many"}:
        download.assert_not_awaited()


async def test_current_avatar_and_member_avatar_without_url_use_confirmed_ids(monkeypatch, tmp_path):
    from test_image_generation import png, image_config, tool_fixture

    from nonebot_plugin_ai_groupmate.agent import image_generation_tools as module
    from nonebot_plugin_ai_groupmate.agent.web_fetch import Download

    _, workspace, session, provider, _, _ = tool_fixture(monkeypatch, tmp_path)
    download = AsyncMock(return_value=Download(png(), "", "avatar.png", "image/png"))
    monkeypatch.setattr(module, "download_public_url", download)
    generate = module.create_image_generation_tool(session, "group", None, config=image_config(), workspace=workspace, pic_dir=tmp_path, current_message_id=None, replied_message_id=None,
                                                   current_user_id="100", group_members=[SimpleNamespace(id="200", user=None)])
    result = json.loads(await generate.ainvoke({"prompt": "draw us together", "reference_user_ids": ["current", "200", "100"]}))
    assert result["status"] == "succeeded"
    assert [call.args[0] for call in download.await_args_list] == ["https://q1.qlogo.cn/g?b=qq&nk=100&s=100", "https://q1.qlogo.cn/g?b=qq&nk=200&s=100"]
    assert len(provider.call_args.args[2]) == 2


async def test_avatar_bytes_reach_the_actual_edits_multipart_transport(monkeypatch, tmp_path):
    import base64

    import httpx
    from test_image_generation import png, image_config, tool_fixture

    from nonebot_plugin_ai_groupmate.agent import image_generation as transport, image_generation_tools as module
    from nonebot_plugin_ai_groupmate.agent.web_fetch import Download

    _, workspace, session, _, _, _ = tool_fixture(monkeypatch, tmp_path)
    avatar = png("purple")
    monkeypatch.setattr(module, "download_public_url", AsyncMock(return_value=Download(avatar, "", "avatar.png", "image/png")))
    monkeypatch.setattr(module, "request_image", transport.request_image)
    requests = []

    async def respond(request):
        requests.append(request)
        await request.aread()
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(transport.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    generate = module.create_image_generation_tool(session, "group", None, config=image_config(), workspace=workspace, pic_dir=tmp_path, current_message_id=None, replied_message_id=None, current_user_id="100")
    result = json.loads(await generate.ainvoke({"prompt": "avatar watercolor", "reference_user_ids": ["current"]}))
    assert result["status"] == "succeeded"
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/images/edits"
    assert 'name="image[]"; filename="reference-1.png"' in requests[0].content.decode("latin1")
    assert avatar in requests[0].content
