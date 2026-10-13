"""Small attachment descriptors; original bytes are fetched only on demand."""

import re
import json
import time
import threading
from typing import Any
from dataclasses import asdict, dataclass

from nonebot_plugin_alconna import File

from .chat_files import chat_file_source_key, chat_file_download_url
from .file_workspace import MAX_FILE_BYTES, FILE_TTL_SECONDS, FileTaskError, FileWorkspace, clean_filename

MAX_INBOX_FILES = 30
MAX_INBOX_ENTRIES = 1000
_ID = re.compile(r"^[0-9a-f]{32}$")
_SCOPE = re.compile(r"^[0-9a-f]{64}$")
_LOCK = threading.RLock()


@dataclass(frozen=True)
class ReceivedFile:
    file_id: str
    source_key: str
    name: str
    platform_id: str
    url: str
    mimetype: str
    size: int | None
    received_at: float
    message_id: str
    index: int
    group_id: str | None
    busid: int | None
    url_from_message: bool = False


def cleanup_chat_file_inbox(root) -> None:
    """Bound descriptors independently of the quota for downloaded originals."""
    with _LOCK:
        inbox = root / "_incoming"
        if not inbox.exists() or inbox.is_symlink():
            return
        remaining = []
        for scope in inbox.iterdir():
            if scope.is_symlink() or not scope.is_dir() or not _SCOPE.fullmatch(scope.name):
                continue
            for path in scope.iterdir():
                if path.is_symlink() or not path.is_file() or path.suffix != ".json" or not _ID.fullmatch(path.stem):
                    continue
                if time.time() - path.stat().st_mtime >= FILE_TTL_SECONDS:
                    path.unlink(missing_ok=True)
                else:
                    remaining.append(path)
        for path in sorted(remaining, key=lambda item: item.stat().st_mtime)[:max(0, len(remaining) - MAX_INBOX_ENTRIES)]:
            path.unlink(missing_ok=True)


class ChatFileInbox:
    def __init__(self, workspace: FileWorkspace):
        self.workspace = workspace
        self.directory = workspace.root / "_incoming" / workspace.directory.name

    def _path(self, file_id: str):
        if not _ID.fullmatch(file_id):
            raise FileTaskError("file_not_found", "请先 list_files 选择当前用户的 file_id。")
        path = self.directory / f"{file_id}.json"
        if any(parent.is_symlink() for parent in (self.directory.parent, self.directory, path)) or not path.resolve().is_relative_to(self.workspace.root):
            raise FileTaskError("file_not_found", "附件记录不可访问。")
        return path

    def get(self, file_id: str) -> ReceivedFile:
        with _LOCK:
            return self._get(file_id)

    def _get(self, file_id: str) -> ReceivedFile:
        path = self._path(file_id)
        try:
            if path.stat().st_size > 16384:
                raise ValueError("oversized descriptor")
            item = ReceivedFile(**json.loads(path.read_text(encoding="utf-8")))
            if item.file_id != file_id or time.time() >= item.received_at + FILE_TTL_SECONDS:
                raise ValueError("expired")
            return item
        except (OSError, ValueError, TypeError):
            raise FileTaskError("file_not_found", "附件记录不存在、已过期或不属于当前用户。请重新发送文件。") from None

    def list(self) -> list[ReceivedFile]:
        with _LOCK:
            if not self.directory.exists() or self.directory.is_symlink() or self.directory.parent.is_symlink():
                return []
            items = []
            for path in self.directory.iterdir():
                if path.suffix != ".json":
                    continue
                try:
                    items.append(self.get(path.stem))
                except FileTaskError:
                    continue
            return sorted(items, key=lambda item: item.received_at, reverse=True)

    def register(self, file: File, *, message_id: str, index: int, group_id: str | None, metadata: dict[str, Any] | None = None) -> ReceivedFile:
        data = dict(getattr(file.origin, "data", {}) or {})
        data.update(metadata or {})
        size = data.get("size", data.get("file_size"))
        try:
            size = int(size) if size is not None else None
            busid = int(data["busid"]) if data.get("busid") is not None else None
        except (ValueError, TypeError):
            raise FileTaskError("invalid_file", "文件大小或协议信息无效。") from None
        if size is not None and size > MAX_FILE_BYTES:
            raise FileTaskError("file_too_large", "文件超过 10 MiB，不下载或保存原件。")
        if size is not None and size < 0:
            raise FileTaskError("invalid_file", "文件大小不能为负数。")
        url = chat_file_download_url(file)
        if not file.id and not url:
            raise FileTaskError("chat_file_unavailable", "附件没有可供稍后下载的文件 ID 或链接，请提供公开下载链接。")
        if len(file.id or "") > 4096 or len(url) > 4096:
            raise FileTaskError("invalid_file", "附件链接或平台 ID 过长。")
        source_key = chat_file_source_key(file, message_id, index)
        item = ReceivedFile(source_key[:32], source_key, clean_filename(file.name), file.id or "", url,
                            (file.mimetype or "")[:200], size, time.time(), message_id, index, group_id, busid,
                            url_from_message=bool(url))
        with _LOCK:
            path = self._path(item.file_id)
            descriptor = json.dumps(asdict(item), ensure_ascii=False)
            if len(descriptor.encode("utf-8")) > 16384:
                raise FileTaskError("invalid_file", "附件元信息过长。")
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text(descriptor, encoding="utf-8")
            for old in self.list()[MAX_INBOX_FILES:]:
                self._path(old.file_id).unlink(missing_ok=True)
            cleanup_chat_file_inbox(self.workspace.root)
        return item


def register_chat_file_markers(files: list[File], workspace: FileWorkspace, *, message_id: str, group_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
    inbox = ChatFileInbox(workspace)
    markers = []
    for index, file in enumerate(files[:3]):
        name = json.dumps(clean_filename(file.name), ensure_ascii=False)
        try:
            inbox.register(file, message_id=message_id, index=index, group_id=group_id, metadata=metadata)
            markers.append(f"[用户上传文件] {name} 已登记，尚未下载原件。用户要求处理时先 list_files 找到 file_id，再 read_file 下载并读取。")
        except FileTaskError as error:
            markers.append(f"[用户上传文件] {name} 未登记：{error}（{error.code}）")
    if len(files) > 3:
        markers.append("单条消息只登记前 3 个文件，其余文件请分开发送。")
    return "\n".join(markers)
