"""Public HTTP downloads with pinned DNS, bounded bodies and checked redirects."""

import socket
import asyncio
import ipaddress
from dataclasses import dataclass
from urllib.parse import unquote
from email.message import Message

import httpx

from .file_workspace import MAX_FILE_BYTES, FileTaskError, clean_filename


@dataclass(frozen=True)
class Download:
    data: bytes
    url: str
    name: str
    content_type: str


def public_url(value: str) -> httpx.URL:
    try:
        if len(value) > 4096 or any(ord(char) < 32 for char in value):
            raise ValueError
        url = httpx.URL(value)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.port not in {None, 80, 443}:
            raise ValueError
        if "%" in url.host:
            raise ValueError
        return url.copy_with(fragment=None)
    except (ValueError, httpx.InvalidURL):
        raise FileTaskError("invalid_url", "请提供不含用户名密码的公开 HTTP/HTTPS 链接（80/443 端口）。") from None


async def _resolve_public_ip(url: httpx.URL) -> str:
    addresses = await asyncio.to_thread(socket.getaddrinfo, url.host, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM)
    ips = list(dict.fromkeys(item[4][0] for item in addresses))
    if not ips or any(not ipaddress.ip_address(address).is_global for address in ips):
        raise FileTaskError("non_public_url", "只允许访问公开网络地址，不能访问内网或本机服务。")
    return ips[0]


async def download_public_url(value: str) -> Download:
    try:
        return await asyncio.wait_for(_download(value), timeout=25)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        raise FileTaskError("download_timeout", "下载超时，可以稍后重试或改用其他来源。") from None
    except (httpx.HTTPError, OSError):
        raise FileTaskError("download_failed", "无法连接或下载该公开链接；请检查链接，或改用其他来源。") from None


async def _download(value: str) -> Download:
    url = public_url(value)
    for _ in range(5):
        ip = await _resolve_public_ip(url)
        # Connect to the validated address, while preserving HTTP Host and TLS SNI.
        # Resolving once then connecting to the hostname would allow DNS rebinding.
        pinned = url.copy_with(host=ip)
        headers = {"Host": url.netloc.decode("ascii"), "User-Agent": "AIGroupmate/2 (+public-document-fetch)", "Accept-Encoding": "identity"}
        async with httpx.AsyncClient(timeout=12, trust_env=False, follow_redirects=False) as client:
            async with client.stream("GET", pinned, headers=headers, extensions={"sni_hostname": url.host}) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FileTaskError("invalid_redirect", "网页返回了没有目标地址的跳转。")
                    url = public_url(str(url.join(location)))
                    continue
                if response.status_code >= 400:
                    raise FileTaskError("http_error", f"网页返回 HTTP {response.status_code}，未取得文件；可尝试其他公开来源。")
                if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                    raise FileTaskError("unsupported_encoding", "服务器未按请求返回未压缩内容，无法执行有界下载。请改用文件直链或其他来源。")
                length = response.headers.get("content-length", "")
                if length.isdigit() and int(length) > MAX_FILE_BYTES:
                    raise FileTaskError("file_too_large", "远程文件超过 10 MB 限制。")
                body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > MAX_FILE_BYTES:
                        raise FileTaskError("file_too_large", "远程文件超过 10 MB 限制。")
                    body.extend(chunk)
                disposition = Message()
                disposition["content-disposition"] = response.headers.get("content-disposition", "")
                name = disposition.get_filename() or unquote(url.path.rsplit("/", 1)[-1]) or "download"
                content_type = response.headers.get("content-type", "application/octet-stream")
                extension = {"text/html": ".html", "text/plain": ".txt", "text/csv": ".csv", "application/pdf": ".pdf", "application/json": ".json",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}.get(content_type.split(";")[0].strip(), ".bin")
                if "." not in name or (extension in {".pdf", ".xlsx", ".docx", ".csv"} and not name.lower().endswith(extension)):
                    name += extension
                return Download(bytes(body), str(url), clean_filename(name), content_type)
    raise FileTaskError("too_many_redirects", "跳转次数过多，未下载文件。")
