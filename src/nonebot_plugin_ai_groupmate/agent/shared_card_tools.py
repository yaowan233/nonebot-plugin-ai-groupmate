"""Read a stored share link only when the current request needs it."""

import asyncio
from pathlib import Path
from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool

from ..model import ChatHistorySchema
from .tool_results import tool_failure, tool_success
from ..shared_cards import CARD_MARKER, load_card_links, card_link_directory
from ..message_metadata import parse_message_metadata
from .structured_context import StructuredSnapshot


def create_shared_card_tools(
    root: Path, session_id: str, history: Sequence[ChatHistorySchema], reply_to_id: str | None = None,
    *, structured_messages: dict[tuple[str, str], StructuredSnapshot] | None = None,
) -> list[BaseTool]:
    if not card_link_directory(root, session_id).is_dir():
        return []
    latest = None
    snapshots = structured_messages or {}
    by_row: dict[int, list[tuple[str, str, StructuredSnapshot]]] = {}
    for (message_id, sender_id), snapshot in snapshots.items():
        for history_id in snapshot.history_ids:
            by_row.setdefault(history_id, []).append((message_id, sender_id, snapshot))
    for row in reversed(history):
        matches = [(message_id, snapshot) for message_id, sender_id, snapshot in by_row.get(row.msg_id, ())
                   if sender_id == row.user_id]
        if len(matches) == 1:
            message_id, snapshot = matches[0]
            if any(part.get("type") == "card" for part in snapshot.parts):
                latest = message_id
                break
        elif CARD_MARKER in parse_message_metadata(row.content)[2]:
            latest = parse_message_metadata(row.content)[0]
            if latest:
                break
    default_id = reply_to_id or latest

    @tool("read_shared_card")
    async def read_shared_card(message_id: str | None = None) -> str:
        """用户明确要分享卡片的链接或详情时，读取当前会话保存的链接；不会抓取网页。message_id省略时用引用卡片或最近卡片。"""
        selected = message_id or default_id
        if not selected:
            return tool_failure("card_not_selected", "请指定当前会话中分享卡片的平台消息ID。")
        links = await asyncio.to_thread(load_card_links, root, session_id, selected)
        if not links:
            return tool_failure("card_link_unavailable", "这条卡片没有保存的网页链接。")
        return tool_success("card_links_loaded", "已读取备用链接，尚未读取网页正文。", data={"message_id": selected, "urls": links})

    return [read_shared_card]
