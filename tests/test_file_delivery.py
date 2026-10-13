import io
import json
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openpyxl import load_workbook
from test_file_tools import toolset, workspace, xlsx_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("private", [True, False])
async def test_onebot_excel_delivery_without_shared_filesystem(tmp_path, monkeypatch, private):
    import nonebot
    from nonebot_plugin_alconna import Target, UniMessage
    from nonebot.adapters.onebot.v11 import Bot, Adapter, ActionFailed
    from nonebot_plugin_alconna.uniseg.adapters.onebot11.exporter import Onebot11MessageExporter

    bot = Bot(nonebot.get_adapter(Adapter), "10001")
    exporter = Onebot11MessageExporter()
    payloads = []

    async def protocol(api, **params):
        payloads.append((api, params))
        # Replay the deployed LLBot failure: the protocol container cannot see
        # /app/data/... on the NoneBot host. It can consume inline file data.
        if not params["file"].startswith("base64://"):
            raise ActionFailed(status="failed", retcode=1200, message="未知文件类型或路径不存在")
        raw = base64.b64decode(params["file"][len("base64://"):], validate=True)
        book = load_workbook(io.BytesIO(raw))
        assert book.active is not None
        assert list(book.active.values) == [("项目", "值"), ("发送测试", 1)]
        book.close()
        return {"message_id": "receipt-1"}

    monkeypatch.setattr(bot, "call_api", AsyncMock(side_effect=protocol))

    async def transmit(message, *, target, fallback, **kwargs):
        assert fallback is False
        native = await exporter.export(message, bot, fallback)
        result = await exporter.send_to(target, bot, native)
        return SimpleNamespace(msg_ids=[result])

    monkeypatch.setattr(UniMessage, "send", transmit)
    tools = toolset(tmp_path, is_private=private, bot=bot, send_target=Target("10002", private=private, self_id=bot.self_id))
    artifact = workspace(tmp_path, is_private=private).save(xlsx_bytes([["项目", "值"], ["发送测试", 1]]), "测试表.xlsx")
    result = json.loads(await tools["send_file"].ainvoke({"file_id": artifact.file_id}))
    assert result["reason_code"] == "file_sent"
    assert result["delivery_state"] == "completed"
    expected = "upload_private_file" if private else "upload_group_file"
    assert payloads[0][0] == expected
    assert payloads[0][1]["name"] == "测试表.xlsx"
    assert payloads[0][1]["user_id" if private else "group_id"] == 10002
    again = json.loads(await tools["send_file"].ainvoke({"file_id": artifact.file_id}))
    assert again["reason_code"] == "duplicate_file"
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_non_onebot11_file_uses_original_alconna_exporter(tmp_path, monkeypatch):
    import nonebot
    from nonebot_plugin_alconna import File, UniMessage
    from nonebot.adapters.onebot.v12 import Bot, Adapter
    from nonebot_plugin_alconna.uniseg.adapters.onebot12.exporter import Onebot12MessageExporter

    from nonebot_plugin_ai_groupmate.agent.file_delivery import file_message

    path = tmp_path / "report.xlsx"
    path.write_bytes(xlsx_bytes([["value"], [1]]))
    uploader = AsyncMock(return_value={"file_id": "v12-upload"})
    bot = Bot(Adapter(nonebot.get_driver()), "10001", "test", "qq")
    monkeypatch.setattr(bot, "call_api", uploader)
    exporter = Onebot12MessageExporter()
    portable = await exporter.export(file_message(path, "report.xlsx"), bot, False)
    ordinary = await exporter.export(UniMessage.file(path=path, name="report.xlsx"), bot, False)
    assert portable == ordinary
    assert portable[0].type == "file"
    assert isinstance(file_message(path, "report.xlsx")[0], File)
    assert uploader.await_args_list[0] == uploader.await_args_list[1]


@pytest.mark.asyncio
async def test_inline_onebot_payload_is_bounded_before_api(tmp_path, monkeypatch):
    from nonebot_plugin_alconna.uniseg.adapters.onebot11.exporter import Onebot11MessageExporter

    from nonebot_plugin_ai_groupmate.agent import file_delivery

    path = tmp_path / "large.xlsx"
    path.write_bytes(b"x" * 11)
    monkeypatch.setattr(file_delivery, "MAX_FILE_BYTES", 10)
    with pytest.raises(file_delivery.FileTaskError, match="10 MiB"):
        await Onebot11MessageExporter().export(file_delivery.file_message(path, "large.xlsx"), None, False)
