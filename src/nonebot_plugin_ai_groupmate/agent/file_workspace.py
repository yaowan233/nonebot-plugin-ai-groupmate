"""Bounded, persistent artifacts addressed by opaque IDs, never model-supplied paths."""

import re
import json
import time
import uuid
import hashlib
import threading
from copy import copy
from pathlib import Path
from dataclasses import asdict, dataclass
from collections.abc import Iterable

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SCOPE_BYTES = 50 * 1024 * 1024
MAX_STORE_BYTES = 512 * 1024 * 1024
MAX_SCOPE_FILES = 30
FILE_TTL_SECONDS = 24 * 3600
_STORE_LOCK = threading.RLock()
_FILE_ID = re.compile(r"^[0-9a-f]{32}$")
_SCOPE_ID = re.compile(r"^[0-9a-f]{64}$")


class FileTaskError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def clean_filename(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[\x00-\x1f<>:"|?*]', "_", name).strip(" .")
    if not name:
        name = "download.bin"
    suffix = Path(name).suffix[:15]
    return name if len(name) <= 100 else name[:100 - len(suffix)] + suffix


@dataclass(frozen=True)
class Artifact:
    file_id: str
    name: str
    size: int
    created_at: float
    content_type: str = ""
    source_url: str = ""
    source_key: str = ""

    def public(self) -> dict:
        data = asdict(self)
        data.pop("source_key", None)
        return {**data, "expires_at": self.created_at + FILE_TTL_SECONDS}


def _artifact_dirs(root: Path):
    if not root.exists():
        return
    for scope in root.iterdir():
        if scope.is_symlink() or not scope.is_dir() or not _SCOPE_ID.fullmatch(scope.name):
            continue
        for folder in scope.iterdir():
            if folder.is_dir() and not folder.is_symlink() and _FILE_ID.fullmatch(folder.name):
                yield folder


def cleanup_workspace(root: Path) -> None:
    """Remove expired artifacts only; unrelated paths and symlinks are untouched."""
    with _STORE_LOCK:
        now = time.time()
        for folder in _artifact_dirs(root):
            if now - folder.stat().st_mtime <= FILE_TTL_SECONDS:
                continue
            for name in ("data", "metadata.json"):
                path = folder / name
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)
            try:
                folder.rmdir()
            except OSError:
                pass


class FileWorkspace:
    def __init__(self, root: Path, *, session_id: str, user_id: str, bot_id: str, is_private: bool,
                 peer_bot_ids: Iterable[str] | None = None):
        self.root = root.resolve()
        self.session_id = session_id
        self.user_id = user_id
        self.bot_id = bot_id
        self.is_private = is_private
        if peer_bot_ids is None:
            from nonebot import get_bots

            connected = get_bots()
            peer_bot_ids = connected if bot_id in connected else ()
        bot_ids = [bot_id, *sorted({str(peer) for peer in peer_bot_ids} - {bot_id})] if not is_private else [bot_id]
        self.directories = tuple(
            self.root / hashlib.sha256(json.dumps([peer, session_id, user_id, is_private], ensure_ascii=False).encode()).hexdigest()
            for peer in bot_ids
        )
        # New files retain their owner; group follow-ups may read connected peers.
        self.directory = self.directories[0]
        # Only explicitly sent images are shared. Private chats retain their
        # user/Bot scope; a group publication can be quoted by other members.
        publication_scope = [True, session_id, user_id, bot_id] if is_private else [False, session_id]
        self.publication_directory = self.root / "published_images" / hashlib.sha256(json.dumps(publication_scope, ensure_ascii=False).encode()).hexdigest()

    def _folder(self, file_id: str) -> Path:
        if not _FILE_ID.fullmatch(file_id):
            raise FileTaskError("file_not_found", "请使用当前用户文件列表中的 file_id，不能传文件路径。")
        for directory in self.directories:
            folder = directory / file_id
            if directory.is_symlink() or folder.is_symlink() or not folder.resolve().is_relative_to(self.root):
                continue
            if folder.is_dir():
                return folder
        folder = self.directory / file_id
        if self.directory.is_symlink() or folder.is_symlink() or not folder.resolve().is_relative_to(self.root):
            raise FileTaskError("file_not_found", "该文件不可访问。")
        return folder

    def get(self, file_id: str) -> Artifact:
        folder = self._folder(file_id)
        try:
            metadata, data = folder / "metadata.json", folder / "data"
            if metadata.is_symlink() or data.is_symlink():
                raise OSError("symlink")
            if metadata.stat().st_size > 16384:
                raise OSError("metadata too large")
            artifact = Artifact(**json.loads(metadata.read_text(encoding="utf-8")))
            if artifact.file_id != file_id:
                raise OSError("invalid file id")
            if time.time() >= artifact.created_at + FILE_TTL_SECONDS:
                raise FileTaskError("file_expired", "文件已过期，请重新发送或生成。")
            if data.stat().st_size != artifact.size or artifact.size > MAX_FILE_BYTES:
                raise OSError("size mismatch")
            return artifact
        except FileTaskError:
            raise
        except (OSError, ValueError, TypeError):
            raise FileTaskError("file_not_found", "文件不存在、已过期，或不属于当前用户和会话。请先 list_files 或重新下载。") from None

    def path(self, file_id: str) -> Path:
        self.get(file_id)
        return self._folder(file_id) / "data"

    def read(self, file_id: str) -> bytes:
        return self.path(file_id).read_bytes()

    def _publication_path(self, file_id: str, message_id: str | None = None) -> Path:
        if not _FILE_ID.fullmatch(file_id) or message_id == "" or message_id == "unknown":
            raise FileTaskError("published_image_not_found", "没有这条消息对应的已发送图片。")
        name = hashlib.sha256(json.dumps([message_id, file_id]).encode()).hexdigest() + ".json"
        path = self.publication_directory / name
        if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise FileTaskError("published_image_not_found", "已发送图片的关联不可访问。")
        return path

    def publish_image(self, file_id: str, message_id: str) -> None:
        """Called only after a validated native image was actually delivered."""
        with _STORE_LOCK:
            self.get(file_id)
            path = self._publication_path(file_id, message_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            for old in path.parent.glob("*.json"):
                if not old.is_symlink() and time.time() - old.stat().st_mtime > FILE_TTL_SECONDS:
                    old.unlink(missing_ok=True)
            record = {"file_id": file_id, "message_id": message_id, "directory": self._folder(file_id).parent.name}
            for index in (path, self._publication_path(file_id)):
                temporary = index.with_suffix("." + uuid.uuid4().hex + ".tmp")
                temporary.write_text(json.dumps(record), encoding="utf-8")
                temporary.replace(index)

    def read_published_image(self, file_id: str, message_id: str | None = None) -> bytes:
        """Read only an artifact published under this exact conversation/message."""
        try:
            path = self._publication_path(file_id, message_id)
            if path.stat().st_size > 1024:
                raise ValueError("invalid publication")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("file_id") != file_id or (message_id is not None and record.get("message_id") != message_id) or not isinstance(record.get("directory"), str) or not _SCOPE_ID.fullmatch(record["directory"]):
                raise ValueError("invalid publication")
            published = copy(self)
            published.directory = self.root / record["directory"]
            published.directories = (published.directory,)
            return published.read(file_id)  # Includes the original expiry/path checks.
        except FileTaskError as error:
            if error.code == "file_expired":
                raise
            raise FileTaskError("published_image_not_found", "这条消息的图片未登记或不属于当前会话。") from None
        except (OSError, ValueError, TypeError, AttributeError):
            raise FileTaskError("published_image_not_found", "这条消息的图片未登记、已过期或不属于当前会话。") from None

    def list(self) -> list[Artifact]:
        with _STORE_LOCK:
            artifacts = {}
            for directory in self.directories:
                if not directory.exists() or directory.is_symlink():
                    continue
                for folder in directory.iterdir():
                    try:
                        artifacts[folder.name] = self.get(folder.name)
                    except FileTaskError:
                        continue
            return sorted(artifacts.values(), key=lambda item: item.created_at, reverse=True)

    def find_source(self, source_key: str) -> Artifact | None:
        return next((item for item in self.list() if item.source_key == source_key), None)

    def save(self, data: bytes, name: str, *, content_type: str = "", source_url: str = "", source_key: str = "") -> Artifact:
        if len(data) > MAX_FILE_BYTES:
            raise FileTaskError("file_too_large", "单个文件不能超过 10 MB。")
        with _STORE_LOCK:
            cleanup_workspace(self.root)
            current = self.list()
            if source_key and (existing := next((item for item in current if item.source_key == source_key), None)):
                return existing
            if len(current) >= MAX_SCOPE_FILES or sum(item.size for item in current) + len(data) > MAX_SCOPE_BYTES:
                raise FileTaskError("workspace_full", "当前用户文件工作区已满（30 个文件 / 50 MB），请等待旧文件在 24 小时后过期。")
            total = sum((folder / "data").stat().st_size for folder in _artifact_dirs(self.root) if (folder / "data").is_file())
            if total + len(data) > MAX_STORE_BYTES:
                raise FileTaskError("workspace_full", "Bot 文件存储额度已满，请稍后再试。")
            artifact = Artifact(uuid.uuid4().hex, clean_filename(name), len(data), time.time(), content_type[:200], source_url[:4096], source_key)
            folder = self._folder(artifact.file_id)
            folder.mkdir(parents=True, exist_ok=False)
            try:
                (folder / "data").write_bytes(data)
                (folder / "metadata.json").write_text(json.dumps(asdict(artifact), ensure_ascii=False), encoding="utf-8")
            except BaseException:
                (folder / "data").unlink(missing_ok=True)
                (folder / "metadata.json").unlink(missing_ok=True)
                folder.rmdir()
                raise
            return artifact
