import json
from types import SimpleNamespace
from typing import Any

import pytest


class FakeSession:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, value: Any) -> None:
        self.added.append(value)


class FakeBot:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, int]] = []
        self.error = error

    async def set_group_ban(self, **kwargs: int) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


def _member(user_id: str, name: str, role: str = "member") -> Any:
    return SimpleNamespace(
        id=user_id,
        name=name,
        nick=name,
        user=SimpleNamespace(name=name, nick=name),
        role=SimpleNamespace(name=role),
    )


def _create_tool(
    moderation_tools: Any,
    *,
    session: FakeSession,
    bot: Any | None,
    members: list[Any],
) -> Any:
    return moderation_tools.create_mute_tool(
        session,
        "12345",
        None,
        None,
        "9000",
        bot_name="群友Bot",
        bot=bot,
        group_members=members,
    )


@pytest.mark.asyncio
async def test_mute_user_prefers_id_and_uses_current_bot(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    session = FakeSession()
    bot = FakeBot()
    members = [
        _member("9000", "群友Bot", "admin"),
        _member("1001", "同名用户"),
        _member("1002", "同名用户"),
    ]

    def forbidden_get_bot(_bot_id: str):
        raise AssertionError("已有当前 bot 时不应查询全局 bot")

    monkeypatch.setattr(moderation_tools, "get_bot", forbidden_get_bot)
    tool = _create_tool(
        moderation_tools,
        session=session,
        bot=bot,
        members=members,
    )
    assert {
        "duration_seconds",
        "reason",
        "target_user_id",
        "target_user_name",
    } <= set(tool.args)

    result = json.loads(await tool.ainvoke({
        "duration_seconds": 120,
        "reason": "刷屏",
        "target_user_id": "1002",
        "target_user_name": "同名用户",
    }))

    assert result["ok"] is True
    assert result["status"] == "succeeded"
    assert result["reason_code"] == "mute_applied"
    assert result["data"]["target_user_id"] == "1002"
    assert result["delivery_state"] == "completed"
    assert bot.calls == [{"group_id": 12345, "user_id": 1002, "duration": 120}]
    assert len(session.added) == 1


@pytest.mark.asyncio
async def test_mute_user_returns_candidates_for_ambiguous_name():
    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    session = FakeSession()
    bot = FakeBot()
    tool = _create_tool(
        moderation_tools,
        session=session,
        bot=bot,
        members=[
            _member("9000", "群友Bot", "owner"),
            _member("1001", "同名用户"),
            _member("1002", "同名用户"),
        ],
    )

    result = json.loads(await tool.ainvoke({
        "duration_seconds": 60,
        "reason": "测试重名",
        "target_user_name": "同名用户",
    }))

    assert result["status"] == "failed"
    assert result["reason_code"] == "target_ambiguous"
    assert result["retryable"] is True
    assert {item["user_id"] for item in result["data"]["candidates"]} == {
        "1001",
        "1002",
    }
    assert bot.calls == []
    assert session.added == []


@pytest.mark.asyncio
async def test_mute_user_fallback_resolves_exact_bot_id(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    session = FakeSession()
    bot = FakeBot()
    requested_bot_ids: list[str] = []

    def fake_get_bot(bot_id: str):
        requested_bot_ids.append(bot_id)
        return bot

    monkeypatch.setattr(moderation_tools, "get_bot", fake_get_bot)
    tool = _create_tool(
        moderation_tools,
        session=session,
        bot=None,
        members=[
            _member("9000", "群友Bot", "admin"),
            _member("1001", "目标用户"),
        ],
    )

    result = json.loads(await tool.ainvoke({
        "duration_seconds": 0,
        "reason": "解除禁言",
        "target_user_name": "目标用户",
    }))

    assert result["status"] == "succeeded"
    assert requested_bot_ids == ["9000"]
    assert bot.calls == [{"group_id": 12345, "user_id": 1001, "duration": 0}]


@pytest.mark.asyncio
async def test_mute_user_rejects_admin_target():
    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    session = FakeSession()
    bot = FakeBot()
    tool = _create_tool(
        moderation_tools,
        session=session,
        bot=bot,
        members=[
            _member("9000", "群友Bot", "admin"),
            _member("1001", "群主", "owner"),
        ],
    )

    result = json.loads(await tool.ainvoke({
        "duration_seconds": 60,
        "reason": "不会执行",
        "target_user_id": "1001",
    }))

    assert result["status"] == "failed"
    assert result["reason_code"] == "protected_target"
    assert bot.calls == []


@pytest.mark.asyncio
async def test_mute_user_sanitizes_adapter_error():
    from nonebot_plugin_ai_groupmate.agent import moderation_tools

    session = FakeSession()
    bot = FakeBot(error=RuntimeError("adapter secret detail"))
    tool = _create_tool(
        moderation_tools,
        session=session,
        bot=bot,
        members=[
            _member("9000", "群友Bot", "admin"),
            _member("1001", "目标用户"),
        ],
    )

    raw_result = await tool.ainvoke({
        "duration_seconds": 60,
        "reason": "测试异常",
        "target_user_id": "1001",
    })
    result = json.loads(raw_result)

    assert result["status"] == "failed"
    assert result["reason_code"] == "provider_error"
    assert result["retryable"] is False
    assert result["delivery_state"] == "unknown"
    assert "adapter secret detail" not in raw_result


def test_moderation_prompt_allows_requests_from_regular_members():
    from nonebot_plugin_ai_groupmate.agent.prompts import build_permission_prompt_parts

    _, instruction = build_permission_prompt_parts(True)

    assert "发起者不需要是管理员" in instruction
    assert "target_user_id" in instruction


def test_tool_status_accepts_succeeded():
    from nonebot_plugin_ai_groupmate.agent.graph import _tool_result_status

    assert _tool_result_status('{"status":"succeeded"}') == "succeeded"
