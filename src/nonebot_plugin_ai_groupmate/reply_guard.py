import asyncio

_lock = asyncio.Lock()
_active_request_ids: dict[str, set[str]] = {}


async def set_latest_request_id(session_id: str, request_id: str) -> None:
    """Replace active requests, used by latest-only background work."""
    async with _lock:
        _active_request_ids[session_id] = {request_id}


async def activate_request_id(session_id: str, request_id: str) -> None:
    """Allow an independent addressed request to deliver its result."""
    async with _lock:
        _active_request_ids.setdefault(session_id, set()).add(request_id)


async def deactivate_request_id(session_id: str, request_id: str) -> None:
    async with _lock:
        active_ids = _active_request_ids.get(session_id)
        if active_ids is None:
            return
        active_ids.discard(request_id)
        if not active_ids:
            _active_request_ids.pop(session_id, None)


async def is_request_active(session_id: str, request_id: str) -> bool:
    async with _lock:
        return request_id in _active_request_ids.get(session_id, set())
