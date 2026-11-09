"""Deliver a command image with scoped history and publication recovery."""

import asyncio

from nonebot.log import logger
from nonebot_plugin_alconna import UniMessage

from ..model import ChatHistory
from .delivery import PublishedImage, OutboundIdentity, send_and_finalize
from .file_workspace import FileWorkspace
from ..message_metadata import is_reliable_platform_id


async def send_command_image(db, data: bytes, *, workspace: FileWorkspace, identity: OutboundIdentity,
                             reply_to_id: str | None, name: str, recovery_enabled: bool = True) -> None:
    """Keep the adapter's existing reply behavior and record acknowledged sends."""
    try:
        artifact = await asyncio.to_thread(workspace.save, data, name)
    except Exception as error:
        logger.warning(f"保存命令图片失败，继续发送 error_type={type(error).__name__}")
        artifact = None
    message = UniMessage.image(raw=data)
    quoted_id = reply_to_id if is_reliable_platform_id(reply_to_id) else None
    def record(message_id: str) -> list[ChatHistory]:
        header = f"id: {message_id}\n" + (f"回复id: {quoted_id}\n" if quoted_id else "")
        image = f"[图片文件] {artifact.name} file_id={artifact.file_id}" if artifact else "发送了图片"
        return [ChatHistory(session_id=workspace.session_id, user_id=identity.bot_name, user_name=identity.bot_name,
                            content_type="bot", content=header + image)]
    publication = PublishedImage(workspace, artifact.file_id) if artifact else None
    receipt = await send_and_finalize(db, workspace.session_id, message, send_options={"reply_to": True}, record=record,
                                      publish=publication.publish if publication else None,
                                      publication=publication.describe if publication else None,
                                      recovery_enabled=recovery_enabled, outbound_identity=identity, reply_to_id=quoted_id)
    if receipt.note:
        logger.warning("命令图片已发送，但本地记录不完整")
