import pytest


@pytest.mark.asyncio
async def test_multiple_addressed_requests_can_remain_active_together():
    from nonebot_plugin_ai_groupmate.reply_guard import (
        is_request_active,
        activate_request_id,
        deactivate_request_id,
        set_latest_request_id,
    )

    session_id = "concurrent-addressed-test"
    request_ids = ("background", "direct-1", "direct-2", "replacement")
    try:
        await set_latest_request_id(session_id, "background")
        await activate_request_id(session_id, "direct-1")
        await activate_request_id(session_id, "direct-2")

        assert await is_request_active(session_id, "background") is True
        assert await is_request_active(session_id, "direct-1") is True
        assert await is_request_active(session_id, "direct-2") is True

        await deactivate_request_id(session_id, "background")
        assert await is_request_active(session_id, "direct-1") is True
        assert await is_request_active(session_id, "direct-2") is True

        await set_latest_request_id(session_id, "replacement")
        assert await is_request_active(session_id, "direct-1") is False
        assert await is_request_active(session_id, "direct-2") is False
        assert await is_request_active(session_id, "replacement") is True
    finally:
        for request_id in request_ids:
            await deactivate_request_id(session_id, request_id)
