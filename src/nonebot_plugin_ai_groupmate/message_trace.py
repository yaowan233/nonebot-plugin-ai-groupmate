"""Stable, content-free correlation IDs for one conversation request."""

import json
import hashlib


def trace_id(session_id: str, request_id: str | None) -> str:
    if not request_id:
        return "-"
    value = json.dumps([str(session_id), str(request_id)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
