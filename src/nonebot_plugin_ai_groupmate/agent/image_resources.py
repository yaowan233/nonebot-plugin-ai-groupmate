"""Resolve image bytes from history, scoped publications, and owned artifacts."""

import re
from typing import Any
from pathlib import Path

from .file_workspace import FileTaskError, FileWorkspace
from ..message_metadata import parse_message_metadata


class ImageResourceError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def published_image_file_id(message: Any) -> str | None:
    if message.content_type != "bot":
        return None
    _, _, body = parse_message_metadata(message.content)
    match = re.search(r"(?m)^\[图片文件\] [^\n]+ file_id=([0-9a-f]{32})$", body)
    return match[1] if match else None


def image_file_name_from_history(message: Any) -> str:
    for line in reversed(message.content.strip().splitlines()):
        line = line.strip()
        if line.startswith("图片文件:"):
            return line.split(":", 1)[1].strip()
    return message.content.strip().split("\n")[-1].strip()


def is_image_history(message: Any) -> bool:
    return (message.content_type == "image" or
            (message.content_type == "bot" and message.media_id is not None) or
            published_image_file_id(message) is not None)


class ImageResources:
    """One scoped reader; callers decide how to present errors to their users."""

    def __init__(self, pic_dir: Path, workspace: FileWorkspace | None = None):
        self.pic_dir = pic_dir.resolve()
        self.workspace = workspace

    @staticmethod
    def _check_size(data: bytes, max_bytes: int | None) -> bytes:
        if not data:
            raise ImageResourceError("empty")
        if max_bytes is not None and len(data) > max_bytes:
            raise ImageResourceError("too_large")
        return data

    def read_media(self, filename: str, *, max_bytes: int | None = None) -> bytes:
        try:
            path = (self.pic_dir / filename).resolve()
            if not path.is_relative_to(self.pic_dir) or not path.is_file():
                raise ImageResourceError("not_found")
            size = path.stat().st_size
            if not size:
                raise ImageResourceError("empty")
            if max_bytes is not None and size > max_bytes:
                raise ImageResourceError("too_large")
            return self._check_size(path.read_bytes(), max_bytes)
        except OSError:
            raise ImageResourceError("not_found") from None

    def read_history(self, message: Any, *, max_bytes: int | None = None,
                     allow_owned_legacy: bool = False, media_path: str | None = None) -> bytes:
        if not is_image_history(message):
            raise ImageResourceError("not_image")
        if file_id := published_image_file_id(message):
            own_id, _, _ = parse_message_metadata(message.content)
            if self.workspace is None or not own_id:
                raise ImageResourceError("not_found")
            try:
                data = self.workspace.read_published_image(file_id, own_id)
            except FileTaskError as error:
                if error.code == "file_expired":
                    raise ImageResourceError("expired") from None
                if not allow_owned_legacy:
                    raise ImageResourceError("not_found") from None
                try:
                    data = self.workspace.read(file_id)
                except FileTaskError as owned_error:
                    raise ImageResourceError("expired" if owned_error.code == "file_expired" else "not_found") from None
            return self._check_size(data, max_bytes)
        return self.read_media(media_path if media_path is not None else image_file_name_from_history(message), max_bytes=max_bytes)

    def read_file_reference(self, file_id: str, *, max_bytes: int | None = None) -> bytes:
        if self.workspace is None:
            raise ImageResourceError("not_found")
        try:
            data = self.workspace.read(file_id)
        except FileTaskError as error:
            if error.code == "file_expired":
                raise ImageResourceError("expired") from None
            try:
                data = self.workspace.read_published_image(file_id)
            except FileTaskError as published_error:
                raise ImageResourceError("expired" if published_error.code == "file_expired" else "not_found") from None
        return self._check_size(data, max_bytes)

    def read_publication(self, file_id: str, message_id: str, *, max_bytes: int | None = None) -> bytes:
        """Read a published image bound to this conversation and transport message."""
        if self.workspace is None:
            raise ImageResourceError("not_found")
        try:
            data = self.workspace.read_published_image(file_id, message_id)
        except FileTaskError as error:
            raise ImageResourceError("expired" if error.code == "file_expired" else "not_found") from None
        return self._check_size(data, max_bytes)
