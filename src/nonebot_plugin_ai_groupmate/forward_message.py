from __future__ import annotations

import json
import datetime
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable, Iterable

from nonebot.log import logger
from nonebot.adapters import Bot
from nonebot_plugin_alconna.uniseg import Reference, CustomNode

from .media_message import MediaKind

MAX_FORWARD_DEPTH = 2
MAX_FORWARD_NODES = 50
MAX_FORWARD_CHARS = 12_000
FORWARD_MARKER_HEADER = "【合并转发消息（内容未展开）】"
FORWARD_ID_PREFIX = "forward_id: "


@dataclass
class _ForwardBudget:
    nodes: int = 0
    chars: int = 0
    truncated: bool = False

    def add(self, text: str) -> str:
        remaining = MAX_FORWARD_CHARS - self.chars
        if remaining <= 0:
            self.truncated = True
            return ""
        if len(text) > remaining:
            self.truncated = True
            text = text[:remaining]
        self.chars += len(text)
        return text


def _segment_type(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("type", ""))
    return str(getattr(segment, "type", ""))


def _segment_data(segment: Any) -> dict[str, Any]:
    if isinstance(segment, dict):
        data = segment.get("data", {})
    else:
        data = getattr(segment, "data", {})
    return data if isinstance(data, dict) else {}


def _format_segments(
    segments: Any,
    register_media: Callable[[MediaKind, dict[str, Any]], str] | None = None,
) -> tuple[str, list[str]]:
    if isinstance(segments, str):
        return segments, []
    if not isinstance(segments, Iterable) or isinstance(segments, (dict, bytes)):
        return str(segments or ""), []

    parts: list[str] = []
    nested_ids: list[str] = []
    placeholders = {
        "image": "[图片]",
        "mface": "[表情]",
        "face": "[表情]",
        "file": "[文件]",
        "json": "[JSON 卡片]",
        "xml": "[XML 卡片]",
    }
    for segment in segments:
        segment_type = _segment_type(segment)
        data = _segment_data(segment)
        if segment_type == "text":
            parts.append(str(data.get("text", getattr(segment, "text", ""))))
        elif segment_type == "at":
            target = (
                data.get("qq")
                or data.get("target")
                or getattr(segment, "target", "")
            )
            parts.append(f"@{target}")
        elif segment_type == "reply":
            reply_id = data.get("id", "")
            parts.append(f"[回复消息 {reply_id}]" if reply_id else "[回复消息]")
        elif segment_type in {"forward", "reference"}:
            forward_id = data.get("id") or getattr(segment, "id", None)
            if forward_id:
                nested_ids.append(str(forward_id))
            parts.append("[嵌套合并转发]")
        elif segment_type in {"record", "audio", "video"}:
            kind: MediaKind = (
                "video" if segment_type == "video" else "audio"
            )
            label = "视频" if kind == "video" else "语音"
            if register_media is None:
                parts.append(f"[{label}，内容未读取]")
            else:
                media_ref = register_media(kind, data)
                parts.append(
                    f"[{label}，内容未读取，media_ref: {media_ref}；"
                    f"需要时调用 read_{kind}_message]"
                )
        elif segment_type in placeholders:
            parts.append(placeholders[segment_type])
        elif segment_type:
            parts.append(f"[{segment_type}]")
    return "".join(parts).strip(), nested_ids


def _response_nodes(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        return []
    for key in ("message", "messages", "nodes"):
        nodes = response.get(key)
        if isinstance(nodes, list):
            return nodes
    data = response.get("data")
    if isinstance(data, dict):
        return _response_nodes(data)
    return []


def _node_fields(node: Any) -> tuple[str, str, Any, Any]:
    if isinstance(node, CustomNode):
        return node.name, node.uid, node.content, node.time
    if not isinstance(node, dict):
        return "未知用户", "", str(node), None

    data = node.get("data") if node.get("type") == "node" else node
    if not isinstance(data, dict):
        data = node
    sender = data.get("sender", {})
    if not isinstance(sender, dict):
        sender = {}
    name = (
        sender.get("card")
        or sender.get("nickname")
        or data.get("name")
        or "未知用户"
    )
    uid = sender.get("user_id") or data.get("uin") or data.get("user_id") or ""
    content = data.get("message", data.get("content", ""))
    timestamp = data.get("time")
    return str(name), str(uid), content, timestamp


def _format_time(value: Any) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError, ValueError):
            pass
    return ""


async def _expand_forward_id(
    bot: Bot,
    forward_id: str,
    *,
    depth: int,
    budget: _ForwardBudget,
    register_media: Callable[[MediaKind, dict[str, Any]], str] | None = None,
) -> list[str]:
    if depth > MAX_FORWARD_DEPTH:
        budget.truncated = True
        return ["[嵌套转发层级过深，已省略]"]
    try:
        response = await bot.call_api("get_forward_msg", id=forward_id)
    except Exception as error:
        logger.warning(
            f"读取合并转发消息失败 id={forward_id} error_type={type(error).__name__}"
        )
        return ["[合并转发内容读取失败]"]

    nodes = _response_nodes(response)
    if not nodes:
        return ["[合并转发记录为空或当前适配器未返回内容]"]

    lines: list[str] = []
    for node in nodes:
        if budget.nodes >= MAX_FORWARD_NODES:
            budget.truncated = True
            break
        budget.nodes += 1
        name, uid, content, timestamp = _node_fields(node)
        body, nested_ids = _format_segments(content, register_media)
        identity = f"{name}({uid})" if uid else name
        time_text = _format_time(timestamp)
        prefix = f"[{time_text}] " if time_text else ""
        line = budget.add(f"{prefix}{identity}: {body or '[空消息]'}")
        if line:
            lines.append(line)
        for nested_id in nested_ids:
            nested_lines = await _expand_forward_id(
                bot,
                nested_id,
                depth=depth + 1,
                budget=budget,
                register_media=register_media,
            )
            lines.extend(f"  {item}" for item in nested_lines)
    return lines


def format_forward_reference_markers(references: Iterable[Reference]) -> str:
    """Keep forward messages lazy while exposing stable IDs to the agent."""
    markers: list[str] = []
    for reference in references:
        if reference.id:
            encoded_id = json.dumps(str(reference.id), ensure_ascii=False)
            markers.append(
                f"{FORWARD_MARKER_HEADER}\n"
                f"{FORWARD_ID_PREFIX}{encoded_id}\n"
                "需要查看聊天记录时调用 read_forward_message。"
            )
        else:
            markers.append("【合并转发消息（缺少可读取的 forward_id）】")
    return "\n".join(markers)


def extract_forward_message_ids(contents: Iterable[str]) -> set[str]:
    """Extract only IDs from markers generated by this module."""
    forward_ids: set[str] = set()
    for content in contents:
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if (
                not line.startswith(FORWARD_ID_PREFIX)
                or index == 0
                or lines[index - 1] != FORWARD_MARKER_HEADER
            ):
                continue
            try:
                forward_id = json.loads(line.removeprefix(FORWARD_ID_PREFIX))
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(forward_id, str) and forward_id:
                forward_ids.add(forward_id)
    return forward_ids


async def expand_forward_message(
    bot: Bot,
    forward_id: str,
    *,
    register_media: Callable[[MediaKind, dict[str, Any]], str] | None = None,
) -> str:
    """Resolve one merged-forward ID into bounded, model-readable text."""
    budget = _ForwardBudget()
    lines = await _expand_forward_id(
        bot,
        forward_id,
        depth=1,
        budget=budget,
        register_media=register_media,
    )
    sections = ["【合并转发聊天记录】\n" + "\n".join(lines)]
    if budget.truncated:
        sections.append("[合并转发内容过长，后续部分已省略]")
    return "\n".join(sections)
