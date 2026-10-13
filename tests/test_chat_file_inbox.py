import json

import pytest
from test_chat_files import incoming as incoming
from test_file_tools import toolset, workspace


@pytest.mark.asyncio
async def test_listing_does_not_download_and_read_fetches_only_selected_file(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent.chat_file_inbox import ChatFileInbox

    _, bot, event, download = incoming
    store = workspace(tmp_path)
    inbox = ChatFileInbox(store)
    selected = inbox.register(File(id="requested", name="requested.csv"), message_id="1", index=0, group_id="group")
    other = inbox.register(File(id="unrelated", name="unrelated.csv"), message_id="2", index=0, group_id="group")
    tools = toolset(tmp_path, bot=bot, event=event)
    result = json.loads(await tools["list_files"].ainvoke({}))
    assert {item["file_id"] for item in result["data"]["files"]} == {selected.file_id, other.file_id}
    assert all(item["state"] == "received" for item in result["data"]["files"])
    assert store.list() == []
    download.assert_not_awaited()
    bot.call_api.assert_not_awaited()

    read = json.loads(await tools["read_file"].ainvoke({"file_id": selected.file_id}))
    assert read["reason_code"] == "file_read"
    assert len(store.list()) == 1
    assert store.list()[0].name == "requested.csv"
    bot.call_api.assert_awaited_once_with("get_group_file_url", file_id="requested", group_id="group")
    download.assert_awaited_once()
    assert read["data"]["file_id"] != selected.file_id

    # Both the returned saved ID and the received ID remain valid, without
    # downloading again; listing also suppresses the duplicate descriptor.
    await tools["read_file"].ainvoke({"file_id": selected.file_id})
    await tools["read_file"].ainvoke({"file_id": read["data"]["file_id"]})
    listed = json.loads(await tools["list_files"].ainvoke({}))["data"]["files"]
    assert len(listed) == 2
    assert {item["state"] for item in listed} == {"received", "saved"}
    download.assert_awaited_once()


@pytest.mark.asyncio
async def test_requested_download_failure_can_be_retried_without_new_upload(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent.chat_file_inbox import ChatFileInbox

    module, bot, event, download = incoming
    item = ChatFileInbox(workspace(tmp_path)).register(File(id="native", name="table.csv"), message_id="1", index=0, group_id=None)
    tools = toolset(tmp_path, bot=bot, event=event)
    download.side_effect = [module.FileTaskError("download_failed", "下载失败"), download.return_value]
    failed = json.loads(await tools["read_file"].ainvoke({"file_id": item.file_id}))
    assert failed["reason_code"] == "download_failed"
    assert workspace(tmp_path).list() == []
    retried = json.loads(await tools["read_file"].ainvoke({"file_id": item.file_id}))
    assert retried["reason_code"] == "file_read"
    assert len(workspace(tmp_path).list()) == 1


@pytest.mark.asyncio
async def test_expired_request_does_not_download_registered_file(tmp_path, incoming, monkeypatch):
    from unittest.mock import AsyncMock

    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent import file_tools
    from nonebot_plugin_ai_groupmate.agent.chat_file_inbox import ChatFileInbox

    _, bot, event, download = incoming
    item = ChatFileInbox(workspace(tmp_path)).register(File(id="native"), message_id="1", index=0, group_id=None)
    monkeypatch.setattr(file_tools, "is_request_active", AsyncMock(return_value=False))
    tools = toolset(tmp_path, bot=bot, event=event, request_id="expired")
    for name, args in (("read_file", {"file_id": item.file_id}), ("send_file", {"file_id": item.file_id}), ("list_files", {}),
                       ("summarize_table", {"file_id": item.file_id, "group_by": [], "value_columns": []})):
        result = json.loads(await tools[name].ainvoke(args))
        assert result["reason_code"] == "request_expired"
    download.assert_not_awaited()
    bot.call_api.assert_not_awaited()


def test_descriptor_persistence_isolation_and_expiry(tmp_path, monkeypatch):
    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent import chat_file_inbox as module

    store = workspace(tmp_path)
    inbox = module.ChatFileInbox(store)
    item = inbox.register(File(id="native", name="../../table.csv"), message_id="1", index=0, group_id="group")
    assert store.list() == []
    assert module.ChatFileInbox(workspace(tmp_path)).get(item.file_id).name == "table.csv"
    assert not any(path.name == "data" for path in tmp_path.rglob("*"))
    for scope in ({"user_id": "other"}, {"bot_id": "other"}, {"session_id": "other"}, {"is_private": True}):
        other = module.ChatFileInbox(workspace(tmp_path, **scope))
        assert other.list() == []
        with pytest.raises(module.FileTaskError):
            other.get(item.file_id)
    monkeypatch.setattr(module.time, "time", lambda: item.received_at + module.FILE_TTL_SECONDS + 1)
    assert inbox.list() == []
    module.cleanup_chat_file_inbox(tmp_path)
    assert list(inbox.directory.glob("*.json")) == []


def test_descriptor_limits_and_large_files_do_not_store_originals(tmp_path, monkeypatch):
    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent import chat_file_inbox as module

    monkeypatch.setattr(module, "MAX_INBOX_FILES", 2)
    monkeypatch.setattr(module, "MAX_INBOX_ENTRIES", 3)
    inbox = module.ChatFileInbox(workspace(tmp_path))
    for i in range(3):
        inbox.register(File(id=str(i)), message_id=str(i), index=0, group_id=None)
    assert {item.platform_id for item in inbox.list()} == {"1", "2"}
    other = module.ChatFileInbox(workspace(tmp_path, user_id="other"))
    for i in range(2):
        other.register(File(id=str(i)), message_id=str(i), index=0, group_id=None)
    assert len(inbox.list()) + len(other.list()) == 3
    with pytest.raises(module.FileTaskError, match="10 MiB"):
        inbox.register(File(id="large"), message_id="large", index=0, group_id=None, metadata={"size": module.MAX_FILE_BYTES + 1})
    assert workspace(tmp_path).list() == []


@pytest.mark.asyncio
async def test_summary_materializes_only_selected_registered_attachment(tmp_path, incoming):
    from nonebot_plugin_alconna import File

    from nonebot_plugin_ai_groupmate.agent.chat_file_inbox import ChatFileInbox

    _, bot, event, download = incoming
    inbox = ChatFileInbox(workspace(tmp_path))
    selected = inbox.register(File(id="requested", name="table.csv"), message_id="1", index=0, group_id=None)
    inbox.register(File(id="unrelated", name="other.csv"), message_id="2", index=0, group_id=None)
    result = json.loads(await toolset(tmp_path, bot=bot, event=event)["summarize_table"].ainvoke({"file_id": selected.file_id, "group_by": ["region"], "value_columns": ["amount"]}))
    assert result["data"]["preview"] == [["A", 5], ["B", 4]]
    assert {item.name for item in workspace(tmp_path).list()} == {"table.csv", "table_summary.xlsx"}
    download.assert_awaited_once()
