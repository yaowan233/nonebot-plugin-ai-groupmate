import asyncio


_lock = asyncio.Lock()
_latest_request_ids: dict[str, str] = {}


async def set_latest_request_id(session_id: str, request_id: str) -> None:
    async with _lock:
        _latest_request_ids[session_id] = request_id


async def is_request_active(session_id: str, request_id: str) -> bool:
    async with _lock:
        return _latest_request_ids.get(session_id) == request_id
