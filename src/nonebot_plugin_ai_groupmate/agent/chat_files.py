"""Import adapter-provided Alconna files into the current sender's workspace."""
import json
import asyncio
import hashlib
from io import BytesIO
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna import File

from .reaction import is_onebot_context
from .web_fetch import download_public_url
from .file_workspace import MAX_FILE_BYTES, Artifact, FileTaskError, FileWorkspace, clean_filename

MAX_CHAT_FILES = 3
_download_gate = asyncio.Semaphore(2)
_pending: dict[str, set[asyncio.Future[None]]] = {}


def _metadata(file: File, extra: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(getattr(file.origin, "data", {}) or {})
    data.update(extra or {})
    return data


def chat_file_source_key(file: File, message_id: str, index: int = 0) -> str:
    identity = ["platform-file", file.id] if file.id else ["message-file", message_id, index]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def chat_file_download_url(file: File) -> str:
    """OneBot's `file` field may be a filename which Alconna prefixes with HTTPS."""
    origin = file.origin
    if origin is not None and "onebot" in type(origin).__module__.lower():
        candidates = [origin.data.get("url"), origin.data.get("file")]
    else:
        candidates = [file.url]
    return next((value for value in candidates if isinstance(value, str) and value.startswith(("http://", "https://"))), "")


async def wait_for_chat_files(workspace: FileWorkspace) -> list[BaseException]:
    """A follow-up can arrive while the attachment message is still downloading."""
    pending = tuple(_pending.get(str(workspace.directory), ()))
    results = await asyncio.gather(*(asyncio.shield(item) for item in pending), return_exceptions=True)
    return [item for item in results if isinstance(item, BaseException)]


async def import_chat_file(
    file: File, workspace: FileWorkspace, *, bot: Bot | None, event: Event | None,
    message_id: str, index: int = 0, group_id: str | None = None, metadata: dict[str, Any] | None = None,
) -> Artifact:
    source_key = chat_file_source_key(file, message_id, index)
    data = _metadata(file, metadata)
    size = data.get("size", data.get("file_size"))
    if size is not None:
        try:
            if int(size) > MAX_FILE_BYTES:
                raise FileTaskError("file_too_large", "接收文件超过 10 MB 限制。")
        except (TypeError, ValueError) as error:
            if isinstance(error, FileTaskError):
                raise
            raise FileTaskError("invalid_file", "文件大小信息无效。") from None
    key = str(workspace.directory)
    completed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    completed.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
    _pending.setdefault(key, set()).add(completed)
    try:
        async def receive():
            if existing := await asyncio.to_thread(workspace.find_source, source_key):
                return existing
            async with _download_gate:
                return await _import(file, workspace, bot=bot, event=event, group_id=group_id, data=data, source_key=source_key)

        try:
            artifact = await asyncio.wait_for(
                receive(), timeout=30,
            )
        except asyncio.TimeoutError:
            raise FileTaskError("download_timeout", "聊天附件下载超时，请稍后重新发送。") from None
        completed.set_result(None)
        return artifact
    except Exception as error:
        completed.set_exception(error)
        raise
    finally:
        if not completed.done():
            completed.cancel()
        _pending[key].discard(completed)
        if not _pending[key]:
            _pending.pop(key, None)


async def _import(file: File, workspace: FileWorkspace, *, bot: Bot | None, event: Event | None, group_id: str | None, data: dict[str, Any], source_key: str) -> Artifact:
    if file.raw is not None:
        raw = file.raw.getvalue() if isinstance(file.raw, BytesIO) else file.raw
        return await asyncio.to_thread(workspace.save, raw, file.name, content_type=file.mimetype or "", source_key=source_key)
    url = chat_file_download_url(file)
    if not url and file.id and bot is not None and is_onebot_context(bot, event):
        api = "get_group_file_url" if group_id is not None else "get_private_file_url"
        params: dict[str, Any] = {"file_id": file.id}
        if group_id is not None:
            params["group_id"] = group_id
            if data.get("busid") is not None:
                params["busid"] = data["busid"]
        try:
            response = await bot.call_api(api, **params)
            url = str(response.get("url") or "") if isinstance(response, dict) else ""
        except Exception:
            url = ""
        if not url:
            # Some OneBot implementations return a URL only from get_file.
            try:
                response = await bot.call_api("get_file", file_id=file.id)
                url = str(response.get("url") or "") if isinstance(response, dict) else ""
            except Exception:
                raise FileTaskError("chat_file_unavailable", "协议端无法提供文件下载链接，请重新发送文件或提供公开下载链接。") from None
    if not url:
        # Never interpret received CQ data or protocol-server paths as local files.
        raise FileTaskError("chat_file_unavailable", "附件没有可用下载链接；不能读取协议端的本地路径。请重新发送或提供公开链接。")
    downloaded = await download_public_url(url)
    name = file.name if file.name not in {"file.bin", "media"} else downloaded.name
    return await asyncio.to_thread(workspace.save, downloaded.data, name, content_type=file.mimetype or downloaded.content_type, source_key=source_key)


async def import_chat_file_markers(files: list[File], workspace: FileWorkspace, *, bot: Bot, event: Event, message_id: str, group_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
    markers = []
    for index, file in enumerate(files[:MAX_CHAT_FILES]):
        name = json.dumps(clean_filename(file.name), ensure_ascii=False)
        try:
            await import_chat_file(file, workspace, bot=bot, event=event, message_id=message_id, index=index, group_id=group_id, metadata=metadata)
            markers.append(f"[用户上传文件] {name} 已保存到当前用户文件区；用 list_files 找到 file_id 后 read_file 读取。")
        except FileTaskError as error:
            markers.append(f"[用户上传文件] {name} 导入失败：{error}（{error.code}）")
        except Exception:
            markers.append(f"[用户上传文件] {name} 导入失败：下载或保存异常，请重新发送或提供公开链接。")
    if len(files) > MAX_CHAT_FILES:
        markers.append("单条消息只导入前 3 个文件，其余文件请分开发送。")
    return "\n".join(markers)
