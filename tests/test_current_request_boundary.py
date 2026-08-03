import datetime
from types import SimpleNamespace
from typing import Any, cast

from nonebot.adapters import Event


def _history_message(
    msg_id: int,
    user_id: str,
    user_name: str,
    content: str,
) -> Any:
    from nonebot_plugin_ai_groupmate.model import ChatHistorySchema

    return ChatHistorySchema(
        msg_id=msg_id,
        session_id="group-1",
        user_id=user_id,
        content_type="text",
        content=f"id: {msg_id}\n{content}",
        created_at=datetime.datetime(2026, 8, 3, 12, msg_id),
        user_name=user_name,
        media_id=None,
        vectorized=False,
    )


def test_current_request_boundary_only_makes_latest_request_actionable():
    from nonebot_plugin_ai_groupmate.agent import _build_current_request_boundary

    history = [
        _history_message(1, "alice-id", "Alice", "查一下我的 BP 列表"),
        _history_message(2, "bob-id", "Bob", "也查一下我的 BP 列表"),
    ]
    event = cast(
        Event,
        SimpleNamespace(
            get_user_id=lambda: "bob-id",
            get_plaintext=lambda: "也查一下我的 BP 列表",
        ),
    )

    boundary = _build_current_request_boundary(
        history,
        "bob-id",
        "Bob",
        event,
    )

    assert "触发用户：Bob" in boundary
    assert "也查一下我的 BP 列表" in boundary
    assert "历史中其他成员更早的询问" in boundary
    assert "省略目标用户时的默认对象，只能指本轮触发用户" in boundary


def test_current_request_boundary_can_fall_back_to_latest_matching_history():
    from nonebot_plugin_ai_groupmate.agent import _build_current_request_boundary

    history = [
        _history_message(1, "alice-id", "Alice", "旧消息"),
        _history_message(2, "bob-id", "Bob", "当前 BP 请求"),
    ]

    boundary = _build_current_request_boundary(
        history,
        "bob-id",
        None,
        None,
    )

    assert "触发用户：Bob" in boundary
    assert "当前 BP 请求" in boundary
    assert "旧消息" not in boundary
