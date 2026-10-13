import json
import asyncio
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_file_tools import toolset, workspace, xlsx_bytes
from test_inbound_conversation import deliver_message as deliver_message


@pytest.fixture
def incoming(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import chat_files

    bot = SimpleNamespace(self_id="bot", call_api=AsyncMock(return_value={"url": "https://cdn.example.org/table.csv"}))
    event = SimpleNamespace()
    download = AsyncMock(return_value=SimpleNamespace(data=b"region,amount\nA,2\nA,3\nB,4", name="table.csv", content_type="text/csv"))
    monkeypatch.setattr(chat_files, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(chat_files, "download_public_url", download)
    return chat_files, bot, event, download


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["csv", "xlsx"])
async def test_received_file_can_be_read_and_summarized(tmp_path, incoming, kind):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    data = b"region,amount\nA,2\nA,3\nB,4" if kind == "csv" else xlsx_bytes([["region", "amount"], ["A", 2], ["A", 3], ["B", 4]])
    file = File(id="qq-attachment", name=f"table.{kind}", raw=data)
    store = workspace(tmp_path)
    imported = await module.import_chat_file(file, store, bot=bot, event=event, message_id="1")
    assert "source_key" not in imported.public()
    tools = toolset(tmp_path)
    listed = json.loads(await tools["list_files"].ainvoke({}))
    assert listed["data"]["files"][0]["file_id"] == imported.file_id
    read = json.loads(await tools["read_file"].ainvoke({"file_id": imported.file_id}))
    assert "region" in read["data"]["text"]
    summary = json.loads(await tools["summarize_table"].ainvoke({"file_id": imported.file_id, "group_by": ["region"], "value_columns": ["amount"]}))
    assert summary["data"]["input_rows"] == 3
    assert summary["data"]["preview"] == [["A", 5], ["B", 4]]
    download.assert_not_awaited()
    bot.call_api.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("group_id", "params", "api"), [("123", {"group_id": "123", "file_id": "native", "busid": 102}, "get_group_file_url"), (None, {"file_id": "native"}, "get_private_file_url")])
async def test_qq_file_id_resolves_download_url(tmp_path, incoming, group_id, params, api):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    store = workspace(tmp_path)
    artifact = await module.import_chat_file(File(id="native", name="报表.csv"), store, bot=bot, event=event,
                                             message_id="1", group_id=group_id, metadata={"busid": 102, "size": 40})
    bot.call_api.assert_awaited_once_with(api, **params)
    download.assert_awaited_once_with("https://cdn.example.org/table.csv")
    assert artifact.name == "报表.csv"
    assert artifact.source_url == ""
    assert store.read(artifact.file_id).startswith(b"region")


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [RuntimeError("unsupported"), {"url": None}])
async def test_qq_get_file_fallback_accepts_only_url(tmp_path, incoming, first):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    bot.call_api.side_effect = [first, {"url": "https://cdn.example.org/table.csv", "file": "/protocol/private.txt"}]
    await module.import_chat_file(File(id="native"), workspace(tmp_path), bot=bot, event=event, message_id="1")
    assert [call.args[0] for call in bot.call_api.await_args_list] == ["get_private_file_url", "get_file"]
    download.assert_awaited_once()


@pytest.mark.asyncio
async def test_file_message_with_url_does_not_call_onebot_api(tmp_path, incoming):
    from nonebot_plugin_alconna import File, UniMessage
    from nonebot.adapters.onebot.v11 import Message, MessageSegment

    module, bot, event, download = incoming
    msg = UniMessage.of(Message(MessageSegment("file", {"file_id": "native", "file_name": "报表.csv", "url": "https://cdn.example.org/table.csv"})), adapter="OneBot V11")
    files = list(msg.include(File))
    assert len(files) == 1
    await module.import_chat_file(files[0], workspace(tmp_path), bot=bot, event=event, message_id="1")
    bot.call_api.assert_not_awaited()
    download.assert_awaited_once()


@pytest.mark.asyncio
async def test_protocol_and_received_paths_are_never_read(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    local = tmp_path / "secret.txt"
    local.write_text("secret")
    bot.call_api.return_value = {"file": str(local)}
    with pytest.raises(module.FileTaskError, match="本地路径"):
        await module.import_chat_file(File(id="native", path=local, url=f"file://{local}"), workspace(tmp_path), bot=bot, event=event, message_id="1")
    assert workspace(tmp_path).list() == []
    download.assert_not_awaited()


@pytest.mark.asyncio
async def test_qq_download_url_keeps_public_network_checks(tmp_path, incoming, monkeypatch):
    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent.web_fetch import download_public_url

    module, bot, event, _ = incoming
    monkeypatch.setattr(module, "download_public_url", download_public_url)
    bot.call_api.return_value = {"url": "http://127.0.0.1/secret"}
    with pytest.raises(module.FileTaskError, match="公开网络"):
        await module.import_chat_file(File(id="native"), workspace(tmp_path), bot=bot, event=event, message_id="1")
    assert workspace(tmp_path).list() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [False, True])
async def test_attachment_size_limits(tmp_path, incoming, raw):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    file = File(id="large", raw=b"x" * (module.MAX_FILE_BYTES + 1) if raw else None)
    with pytest.raises(module.FileTaskError, match="10 MB"):
        await module.import_chat_file(file, workspace(tmp_path), bot=bot, event=event, message_id="1",
                                     metadata={} if raw else {"size": module.MAX_FILE_BYTES + 1})
    download.assert_not_awaited()
    bot.call_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_upload_deduplicates_without_redownload(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    file = File(id="native", name="table.csv")
    store = workspace(tmp_path)
    first = await module.import_chat_file(file, store, bot=bot, event=event, message_id="1")
    second = await module.import_chat_file(file, workspace(tmp_path), bot=bot, event=event, message_id="different-notice")
    assert first.file_id == second.file_id
    assert len(store.list()) == 1
    download.assert_awaited_once()
    for scope in ({"user_id": "other"}, {"bot_id": "other"}, {"session_id": "other"}, {"is_private": True}):
        assert workspace(tmp_path, **scope).list() == []


@pytest.mark.asyncio
async def test_list_files_waits_for_attachment_import(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    entered, release = asyncio.Event(), asyncio.Event()
    original = download.return_value

    async def delayed(_):
        entered.set()
        await release.wait()
        return original

    download.side_effect = delayed
    receive = asyncio.create_task(module.import_chat_file(File(id="native"), workspace(tmp_path), bot=bot, event=event, message_id="1"))
    try:
        await entered.wait()
        listing = asyncio.create_task(toolset(tmp_path)["list_files"].ainvoke({}))
        await asyncio.sleep(0.01)
        assert not listing.done()
        release.set()
        artifact = await receive
        result = json.loads(await listing)
        assert result["data"]["files"][0]["file_id"] == artifact.file_id
        assert module._pending == {}
    finally:
        release.set()
        await receive


@pytest.mark.asyncio
async def test_import_markers_limit_and_record_failure(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    module, bot, event, download = incoming
    download.side_effect = module.FileTaskError("download_failed", "下载失败")
    files = [File(id=str(i), name=f"{i}.csv", url="https://cdn.example.org/file") for i in range(4)]
    marker = await module.import_chat_file_markers(files, workspace(tmp_path), bot=bot, event=event, message_id="1")
    assert marker.count("导入失败") == 3
    assert "只导入前 3 个" in marker
    assert "https://" not in marker
    assert module._pending == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(("is_private", "addressed", "expected"), [(False, False, False), (False, True, True), (True, True, True)])
async def test_file_only_message_is_saved_and_replies_when_addressed(tmp_path, monkeypatch, deliver_message, is_private, addressed, expected):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_alconna import File, UniMessage

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin.plugin_config, "agent_file_tools_enabled", True)
    deliver, queued, _ = deliver_message
    session_id = f"file-message-{uuid4()}"
    await deliver(session_id, "1", "", message=UniMessage(File(name="input.csv", raw=b"a,b\n1,2")), addressed=addressed, is_private=is_private)
    assert queued.await_count == int(expected)
    store = FileWorkspace(tmp_path / "agent_files", session_id=session_id, user_id="member-1", bot_id="bot-1", is_private=is_private)
    assert len(store.list()) == 1
    async with get_session() as db:
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == session_id))).scalars().all()
        assert len(rows) == 1
        assert "[用户上传文件]" in rows[0].content
        assert "input.csv" in rows[0].content


@pytest.mark.asyncio
async def test_file_upload_in_continuing_group_conversation(tmp_path, monkeypatch, deliver_message):
    from nonebot_plugin_alconna import File, UniMessage

    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin.plugin_config, "agent_file_tools_enabled", True)
    deliver, queued, _ = deliver_message
    session_id = f"file-followup-{uuid4()}"
    await deliver(session_id, "1", "我稍后上传表格，请汇总", addressed=True)
    queued.reset_mock()
    await deliver(session_id, "2", "", message=UniMessage(File(name="table.csv", raw=b"a,b\n1,2")))
    queued.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("notice_type", ["group_upload", "offline_file", "friend_upload"])
async def test_qq_upload_notice_imports_and_records_once(tmp_path, incoming, monkeypatch, notice_type):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    _, bot, _, download = incoming
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(plugin, "get_bots", lambda: {})
    monkeypatch.setattr(plugin.plugin_config, "agent_file_tools_enabled", True)
    user_id = f"u-{uuid4()}"
    group_id = f"g-{uuid4()}" if notice_type == "group_upload" else None
    data = {"id": "native", "name": "上传.csv", "busid": 102, "size": 50}
    if group_id:
        from nonebot.adapters.onebot.v11.event import File as NativeFile

        native_file = NativeFile(**data)
    else:
        native_file = data
    event = SimpleNamespace(notice_type=notice_type, user_id=user_id, group_id=group_id, file=native_file)
    async with get_session() as db:
        await plugin.handle_file_upload(db, bot, event)
        await plugin.handle_file_upload(db, bot, event)
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == (group_id or user_id)))).scalars().all()
        assert len(rows) == 1
        assert "上传.csv" in rows[0].content
        assert "已保存" in rows[0].content
    store = FileWorkspace(tmp_path / "agent_files", session_id=group_id or user_id, user_id=user_id, bot_id="bot", is_private=group_id is None)
    assert len(store.list()) == 1
    download.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("disabled", "self_upload"), [(True, False), (False, True)])
async def test_disabled_tools_and_self_upload_are_ignored(tmp_path, incoming, monkeypatch, disabled, self_upload):
    import nonebot_plugin_ai_groupmate as plugin

    _, bot, _, download = incoming
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(plugin, "get_bots", lambda: {})
    monkeypatch.setattr(plugin.plugin_config, "agent_file_tools_enabled", not disabled)
    event = SimpleNamespace(notice_type="group_upload", user_id="bot" if self_upload else "user", group_id="group", file={"id": "native"})
    await plugin.handle_file_upload(SimpleNamespace(), bot, event)
    download.assert_not_awaited()
    bot.call_api.assert_not_awaited()
    assert not (tmp_path / "agent_files").exists()


@pytest.mark.asyncio
async def test_notice_retry_updates_failed_import_record(tmp_path, incoming, monkeypatch):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    module, bot, _, download = incoming
    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin, "is_onebot_context", lambda *_: True)
    monkeypatch.setattr(plugin, "get_bots", lambda: {})
    monkeypatch.setattr(plugin.plugin_config, "agent_file_tools_enabled", True)
    user_id = f"retry-{uuid4()}"
    event = SimpleNamespace(notice_type="offline_file", user_id=user_id, file={"id": "native", "name": "table.csv"})
    download.side_effect = [module.FileTaskError("download_failed", "下载失败"), download.return_value]
    async with get_session() as db:
        await plugin.handle_file_upload(db, bot, event)
        failed = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == user_id))).scalar_one()
        assert "导入失败" in failed.content
        await plugin.handle_file_upload(db, bot, event)
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == user_id))).scalars().all()
        assert len(rows) == 1
        assert "已保存" in rows[0].content


@pytest.mark.asyncio
async def test_disabled_file_message_does_not_save_attachment(tmp_path, monkeypatch, deliver_message):
    from nonebot_plugin_alconna import File, UniMessage

    import nonebot_plugin_ai_groupmate as plugin

    monkeypatch.setattr(plugin, "plugin_data_dir", tmp_path)
    monkeypatch.setattr(plugin.plugin_config, "agent_file_tools_enabled", False)
    deliver, queued, _ = deliver_message
    await deliver(f"disabled-{uuid4()}", "1", "", message=UniMessage(File(name="input.csv", raw=b"a,b\n1,2")), addressed=True)
    queued.assert_not_awaited()
    assert not (tmp_path / "agent_files").exists()


@pytest.mark.asyncio
async def test_attachment_timeout_is_recorded_as_import_failure(tmp_path, incoming, monkeypatch):
    from nonebot_plugin_alconna import File

    module, bot, event, _ = incoming
    monkeypatch.setattr(module, "_import", AsyncMock(side_effect=asyncio.TimeoutError))
    marker = await module.import_chat_file_markers([File(id="native")], workspace(tmp_path), bot=bot, event=event, message_id="1")
    assert "download_timeout" in marker
    assert "超时" in marker
    assert workspace(tmp_path).list() == []
    assert module._pending == {}


@pytest.mark.asyncio
async def test_pending_attachment_timeout_returns_retryable_status(tmp_path, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import file_tools

    monkeypatch.setattr(file_tools, "wait_for_chat_files", AsyncMock(side_effect=asyncio.TimeoutError))
    result = json.loads(await toolset(tmp_path)["list_files"].ainvoke({}))
    assert result["reason_code"] == "chat_file_pending"
    assert result["retryable"] is True
