"""Turn share cards into bounded visible text; retain links outside chat context."""

import re
import json
import hashlib
from html import unescape
from pathlib import Path
from itertools import islice
from xml.etree import ElementTree
from dataclasses import dataclass
from urllib.parse import urlsplit
from collections.abc import Iterable

CARD_MARKER = "【分享卡片文字】"
MAX_CARD_TEXT = 300
MAX_CARD_PAYLOAD = 65536
TEXT_KEYS = ("title", "desc", "summary", "description", "tag", "source", "author")
LINK_KEYS = {"url", "jumpurl", "qqdocurl", "share_url", "target_url", "web_url"}


@dataclass(frozen=True)
class SharedCard:
    text: str
    urls: tuple[str, ...] = ()


def _visible(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", unescape(value[:2048])).replace("\x00", "")).strip()[:220]


def _url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() for c in value):
        return None
    try:
        parsed = urlsplit(unescape(value))
        return unescape(value) if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username else None
    except ValueError:
        return None


def parse_shared_card(format_name: str, payload: object) -> SharedCard:
    texts: list[str] = []
    urls: list[str] = []

    def add_text(value: object) -> None:
        if text := _visible(value):
            if text not in texts and not _url(text):
                texts.append(text)

    def add_link(value: object) -> None:
        if (link := _url(value)) and link not in urls and len(urls) < 3:
            urls.append(link)

    try:
        if isinstance(payload, str) and len(payload) > MAX_CARD_PAYLOAD:
            raise ValueError("oversized card")
        if format_name == "xml":
            if not isinstance(payload, str) or re.search(r"<!\s*(?:DOCTYPE|ENTITY)", payload, re.I):
                raise ValueError("unsupported XML")
            root = ElementTree.fromstring(payload)
            for index, element in enumerate(root.iter()):
                if index >= 128:
                    break
                tag = element.tag.rsplit("}", 1)[-1].lower()
                if tag in {"title", "summary", "desc"}:
                    add_text("".join(element.itertext()))
                if tag == "source":
                    add_text(element.get("name"))
                if tag in {"msg", "item", "link"}:
                    add_link(element.get("url"))
                if tag in {"url", "link"}:
                    add_link(element.text)
        else:
            data = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(data, dict):
                raise ValueError("card is not an object")
            # Presentation metadata only: never traverse config, credentials,
            # preview images, layouts or arbitrary backend response payloads.
            pending = [(data.get("meta", data), 0)]
            traverse = isinstance(data.get("meta"), (dict, list))
            visited = 0
            while pending and visited < 128:
                node, depth = pending.pop(0)
                visited += 1
                if depth > 5:
                    continue
                if isinstance(node, list):
                    pending.extend((item, depth + 1) for item in node[:16])
                elif isinstance(node, dict):
                    for key in TEXT_KEYS:
                        add_text(node.get(key))
                    for key, value in islice(node.items(), 128):
                        if isinstance(key, str) and key.lower() in LINK_KEYS:
                            add_link(value)
                        elif traverse and key not in {"config", "preview", "icon", "layout"} and isinstance(value, (dict, list)):
                            pending.append((value, depth + 1))
            if not texts:
                add_text(data.get("prompt"))
            if format_name in {"share", "music"}:
                add_text(data.get("content"))
    except (ValueError, TypeError, RecursionError, ElementTree.ParseError):
        texts.clear()
        urls.clear()
    text = CARD_MARKER + "\n" + ("\n".join(texts) or "（无可见文字）")
    if len(text) > MAX_CARD_TEXT:
        text = text[:MAX_CARD_TEXT - 1] + "…"
    return SharedCard(text, tuple(urls))


def parse_shared_segment(segment: object) -> SharedCard | None:
    origin = getattr(segment, "origin", None) or segment
    kind = origin.get("type") if isinstance(origin, dict) else getattr(origin, "type", None)
    if kind == "hyper":
        kind = getattr(origin, "format", None)
        payload = getattr(origin, "content", None)
        if payload is None:
            payload = getattr(origin, "raw", None)
    elif kind in {"json", "xml", "share", "music"}:
        data = origin.get("data", {}) if isinstance(origin, dict) else getattr(origin, "data", {})
        payload = data.get("data") if kind in {"json", "xml"} else data
    else:
        return None
    return parse_shared_card(str(kind), payload)


def card_link_directory(root: Path, session_id: str) -> Path:
    return root / hashlib.sha256(session_id.encode()).hexdigest()


def _link_file(root: Path, session_id: str, message_id: str) -> Path:
    return card_link_directory(root, session_id) / (hashlib.sha256(message_id.encode()).hexdigest() + ".json")


def save_card_links(root: Path, session_id: str, message_id: str, urls: Iterable[str]) -> None:
    urls = list(dict.fromkeys(url for value in urls if (url := _url(value))))[:3]
    if not urls:
        return
    from uuid import uuid4
    target = _link_file(root, session_id, message_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix("." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps({"urls": urls}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_card_links(root: Path, session_id: str, message_id: str) -> list[str]:
    target = _link_file(root, session_id, message_id)
    try:
        if target.stat().st_size > 16384:
            return []
        data = json.loads(target.read_text(encoding="utf-8"))
        return [url for value in data.get("urls", [])[:3] if (url := _url(value))]
    except (OSError, ValueError, TypeError, AttributeError):
        return []
