"""Bounded, persistent artifacts addressed by opaque IDs, never model-supplied paths."""

import re
import json
import time
import uuid
import hashlib
import threading
from pathlib import Path
from dataclasses import asdict, dataclass

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

    def public(self) -> dict:
        return {**asdict(self), "expires_at": self.created_at + FILE_TTL_SECONDS}


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
    def __init__(self, root: Path, *, session_id: str, user_id: str, bot_id: str, is_private: bool):
        self.root = root.resolve()
        scope = json.dumps([bot_id, session_id, user_id, is_private], ensure_ascii=False)
        self.directory = self.root / hashlib.sha256(scope.encode()).hexdigest()

    def _folder(self, file_id: str) -> Path:
        if not _FILE_ID.fullmatch(file_id):
            raise FileTaskError("file_not_found", "请使用当前用户文件列表中的 file_id，不能传文件路径。")
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
            if artifact.file_id != file_id or time.time() >= artifact.created_at + FILE_TTL_SECONDS:
                raise OSError("expired")
            if data.stat().st_size != artifact.size or artifact.size > MAX_FILE_BYTES:
                raise OSError("size mismatch")
            return artifact
        except (OSError, ValueError, TypeError):
            raise FileTaskError("file_not_found", "文件不存在、已过期，或不属于当前用户和会话。请先 list_files 或重新下载。") from None

    def path(self, file_id: str) -> Path:
        self.get(file_id)
        return self._folder(file_id) / "data"

    def read(self, file_id: str) -> bytes:
        return self.path(file_id).read_bytes()

    def list(self) -> list[Artifact]:
        with _STORE_LOCK:
            if not self.directory.exists() or self.directory.is_symlink():
                return []
            artifacts = []
            for folder in self.directory.iterdir():
                try:
                    artifacts.append(self.get(folder.name))
                except FileTaskError:
                    continue
            return sorted(artifacts, key=lambda item: item.created_at, reverse=True)

    def save(self, data: bytes, name: str, *, content_type: str = "", source_url: str = "") -> Artifact:
        if len(data) > MAX_FILE_BYTES:
            raise FileTaskError("file_too_large", "单个文件不能超过 10 MB。")
        with _STORE_LOCK:
            cleanup_workspace(self.root)
            current = self.list()
            if len(current) >= MAX_SCOPE_FILES or sum(item.size for item in current) + len(data) > MAX_SCOPE_BYTES:
                raise FileTaskError("workspace_full", "当前用户文件工作区已满（30 个文件 / 50 MB），请等待旧文件在 24 小时后过期。")
            total = sum((folder / "data").stat().st_size for folder in _artifact_dirs(self.root) if (folder / "data").is_file())
            if total + len(data) > MAX_STORE_BYTES:
                raise FileTaskError("workspace_full", "Bot 文件存储额度已满，请稍后再试。")
            artifact = Artifact(uuid.uuid4().hex, clean_filename(name), len(data), time.time(), content_type[:200], source_url[:4096])
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
