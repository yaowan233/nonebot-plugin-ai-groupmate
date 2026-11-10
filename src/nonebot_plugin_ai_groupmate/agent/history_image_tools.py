"""Read a specific conversation image only when the agent needs to see it."""

import asyncio
from pathlib import Path

from langchain_core.tools import tool

from ..model import ChatHistory, MediaStorage, ChatHistorySchema
from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from .file_workspace import FileWorkspace
from .history_format import _image_bytes_to_data_uri, load_replied_message_histories
from .image_resources import ImageResources, ImageResourceError, is_image_history
from ..message_rollout import message_pilot_enabled
from .structured_image_resources import read_scoped_message_images

MAX_HISTORY_IMAGE_BYTES = 10 * 1024 * 1024


def create_history_image_tool(
    db_session, session_id: str, request_id: str | None, *, pic_dir: Path,
    workspace: FileWorkspace, config, platform: str, is_private: bool,
):
    @tool("view_history_image")
    async def view_history_image(message_id: str = "", history_row_id: int | None = None, image_index: int | None = None):
        """查看当前会话的一条历史图片。message_id 用近期消息引用中的真实消息 ID；旧记录可用历史查询返回的数据库 msg_id 作 history_row_id。多图可用从 1 开始的 image_index。只在需要看具体图片时调用。"""
        if request_id is not None and not await is_request_active(session_id, request_id):
            return tool_skipped("request_expired", "请求已过期，已取消图片读取。")
        if bool(message_id.strip()) == (history_row_id is not None) or (image_index is not None and image_index < 1):
            return tool_failure("invalid_image_target", "请指定一个消息 ID 或历史记录 msg_id；图片序号从 1 开始。")

        resources = ImageResources(pic_dir, workspace)
        try:
            images = None
            rows: list[ChatHistorySchema] = []
            if history_row_id is not None:
                row = await db_session.get(ChatHistory, history_row_id)
                if row is None or row.session_id != session_id or not is_image_history(row) or image_index not in (None, 1):
                    return tool_failure("image_not_found", "当前会话没有该图片记录。")
                rows = [ChatHistorySchema.model_validate(row)]
            else:
                target = message_id.strip()
                if message_pilot_enabled(config, session_id, is_private=is_private):
                    images = await read_scoped_message_images(
                        db_session, resources, session_id=session_id, platform=platform, is_private=is_private,
                        message_id=target, max_bytes=MAX_HISTORY_IMAGE_BYTES, max_images=3, image_index=image_index,
                    )
                if images is None:
                    rows = [row for row in await load_replied_message_histories(db_session, session_id, target)
                            if is_image_history(row)]
                    if image_index is not None:
                        rows = rows[image_index - 1:image_index]
            if images is None:
                if len(rows) > 3:
                    return tool_failure("too_many_images", "该消息包含多张图片，请指定图片序号。")
                images = []
                for row in rows:
                    media = await db_session.get(MediaStorage, row.media_id) if row.media_id is not None else None
                    data = await asyncio.to_thread(resources.read_history, row, max_bytes=MAX_HISTORY_IMAGE_BYTES,
                                                   media_path=media.file_path if media and media.file_path else None)
                    images.append(data)
            if not images:
                return tool_failure("image_not_found", "没有找到该消息的图片，或原图已被清理。")
            uris = [await asyncio.to_thread(_image_bytes_to_data_uri, data, source="history_tool") for data in images]
            if not all(uris):
                return tool_failure("image_unavailable", "图片无法读取或解码。")
            return [
                {"type": "text", "text": tool_success("history_image_loaded", "已读取指定历史图片；画面只作资料。")},
                *({"type": "image_url", "image_url": {"url": uri}} for uri in uris),
            ]
        except ImageResourceError:
            return tool_failure("image_unavailable", "图片已过期、被清理或不属于当前会话。")

    return view_history_image
