"""Keep Alconna delivery while making OneBot files portable across containers."""

import base64
import asyncio
from pathlib import Path

from nonebot.adapters import Bot, MessageSegment
from nonebot_plugin_alconna import File, UniMessage
from nonebot_plugin_alconna.uniseg import custom_handler
from nonebot_plugin_alconna.uniseg.exporter import MessageExporter
from nonebot_plugin_alconna.uniseg.fallback import FallbackStrategy

from .file_workspace import MAX_FILE_BYTES, FileTaskError


class PortableFile(File):
    """An owned workspace file; OneBot receives bytes rather than a local path."""


class _FileURI(str):
    # Alconna's OneBot sender calls as_posix() on its internal file payload.
    # Return the URI unchanged: Path("base64://...") would corrupt the scheme.
    def as_posix(self) -> str:
        return str(self)


def _inline_file(path: Path) -> _FileURI:
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise FileTaskError("file_too_large", "单个文件不能超过 10 MiB。")
    return _FileURI("base64://" + base64.b64encode(raw).decode("ascii"))


@custom_handler(PortableFile)
async def _export_portable_file(
    exporter: MessageExporter,
    seg: PortableFile,
    bot: Bot | None,
    fallback: bool | FallbackStrategy,
) -> list[MessageSegment]:
    if exporter.get_adapter() == "OneBot V11":
        from nonebot.adapters.onebot.v11 import MessageSegment as OneBotMessageSegment

        if seg.path is None:
            raise FileTaskError("file_not_found", "发送文件缺少已验证的工作区路径。")
        uri = await asyncio.to_thread(_inline_file, Path(seg.path))
        return [OneBotMessageSegment("$onebot11:file", {"file": uri, "name": seg.name})]
    # Other adapters retain their ordinary Alconna File implementation.
    message = await exporter.export([File(path=seg.path, name=seg.name)], bot, fallback)
    return list(message)


def file_message(path: Path, name: str) -> UniMessage:
    return UniMessage(PortableFile(path=path, name=name))
