import asyncio
import datetime
from types import SimpleNamespace
from typing import cast
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import delete


@pytest.mark.asyncio
async def test_group_daily_quota_stops_at_limit_and_resets_next_day():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GlobalModelGroupUsage
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        consume_group_daily_quota,
        get_group_daily_quota_status,
    )

    group_id = "daily-quota-limit-test"
    timezone = datetime.timezone(datetime.timedelta(hours=8))
    today = datetime.datetime(2026, 8, 25, 20, 30, tzinfo=timezone)
    tomorrow = today + datetime.timedelta(days=1)

    async with get_session() as session:
        await session.execute(
            delete(GlobalModelGroupUsage).where(
                GlobalModelGroupUsage.group_id == group_id
            )
        )
        await session.commit()

        first = await consume_group_daily_quota(
            session, group_id, 2, now=today
        )
        second = await consume_group_daily_quota(
            session, group_id, 2, now=today
        )
        exhausted = await consume_group_daily_quota(
            session, group_id, 2, now=today
        )

        assert first.allowed is True
        assert second.allowed is True
        assert exhausted.allowed is False
        assert exhausted.used == 2
        assert exhausted.remaining == 0
        assert exhausted.resets_at == datetime.datetime(
            2026, 8, 26, tzinfo=timezone
        )

        reset = await consume_group_daily_quota(
            session, group_id, 2, now=tomorrow
        )
        assert reset.allowed is True
        assert reset.used == 1
        assert (
            await get_group_daily_quota_status(
                session, group_id, 2, now=tomorrow
            )
        ).remaining == 1

        await session.execute(
            delete(GlobalModelGroupUsage).where(
                GlobalModelGroupUsage.group_id == group_id
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_private_user_daily_quota_stops_at_ten_and_is_per_user():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GlobalModelPrivateUserUsage
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        consume_private_user_daily_quota,
    )

    user_ids = ("private-quota-user-a", "private-quota-user-b")
    now = datetime.datetime(
        2026,
        8,
        26,
        12,
        tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
    )
    async with get_session() as session:
        await session.execute(
            delete(GlobalModelPrivateUserUsage).where(
                GlobalModelPrivateUserUsage.user_id.in_(user_ids)
            )
        )
        await session.commit()

        user_a_results = [
            await consume_private_user_daily_quota(
                session,
                user_ids[0],
                10,
                now=now,
            )
            for _ in range(11)
        ]
        user_b_first = await consume_private_user_daily_quota(
            session,
            user_ids[1],
            10,
            now=now,
        )

        assert sum(status.allowed for status in user_a_results) == 10
        assert user_a_results[-1].allowed is False
        assert user_a_results[-1].used == 10
        assert user_b_first.allowed is True
        assert user_b_first.used == 1

        await session.execute(
            delete(GlobalModelPrivateUserUsage).where(
                GlobalModelPrivateUserUsage.user_id.in_(user_ids)
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_zero_daily_quota_limit_is_unlimited_and_not_persisted():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GlobalModelGroupUsage
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        consume_group_daily_quota,
    )

    group_id = "daily-quota-unlimited-test"
    async with get_session() as session:
        await session.execute(
            delete(GlobalModelGroupUsage).where(
                GlobalModelGroupUsage.group_id == group_id
            )
        )
        await session.commit()

        status = await consume_group_daily_quota(session, group_id, 0)

        assert status.allowed is True
        assert await session.get(GlobalModelGroupUsage, group_id) is None


@pytest.mark.asyncio
async def test_exhausted_quota_does_not_read_expired_row_after_commit():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GlobalModelGroupUsage
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        consume_group_daily_quota,
    )

    group_id = "daily-quota-expired-row-test"
    now = datetime.datetime(
        2026,
        8,
        25,
        20,
        tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
    )
    async with get_session() as session:
        session.sync_session.expire_on_commit = True
        await session.execute(
            delete(GlobalModelGroupUsage).where(
                GlobalModelGroupUsage.group_id == group_id
            )
        )
        await session.commit()

        allowed = await consume_group_daily_quota(
            session, group_id, 1, now=now
        )
        exhausted = await consume_group_daily_quota(
            session, group_id, 1, now=now
        )

        assert allowed.allowed is True
        assert exhausted.allowed is False
        assert exhausted.used == 1

        await session.execute(
            delete(GlobalModelGroupUsage).where(
                GlobalModelGroupUsage.group_id == group_id
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_concurrent_daily_quota_reservations_do_not_exceed_limit():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GlobalModelGroupUsage
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        consume_group_daily_quota,
    )

    group_id = "daily-quota-concurrency-test"
    now = datetime.datetime(
        2026,
        8,
        25,
        20,
        tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
    )
    async with get_session() as session:
        await session.execute(
            delete(GlobalModelGroupUsage).where(
                GlobalModelGroupUsage.group_id == group_id
            )
        )
        await session.commit()

    async def reserve_once():
        async with get_session() as session:
            return await consume_group_daily_quota(
                session,
                group_id,
                3,
                now=now,
            )

    results = await asyncio.gather(*(reserve_once() for _ in range(12)))

    assert sum(status.allowed for status in results) == 3
    async with get_session() as session:
        row = await session.get(GlobalModelGroupUsage, group_id)
        assert row is not None
        assert row.used_count == 3
        await session.delete(row)
        await session.commit()


def test_quota_message_contains_limit_configuration_and_wait_time():
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        GroupDailyQuotaStatus,
        build_quota_exhausted_message,
    )

    timezone = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime(2026, 8, 25, 20, 30, tzinfo=timezone)
    status = GroupDailyQuotaStatus(
        allowed=False,
        used=50,
        limit=50,
        resets_at=datetime.datetime(2026, 8, 26, tzinfo=timezone),
    )

    message = build_quota_exhausted_message(status, now=now)

    assert "50 次" in message
    assert "/配置群API" in message
    assert "约 3 小时 30 分钟后恢复" in message


def test_daily_quota_config_rejects_negative_values():
    from pydantic import ValidationError

    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    assert ScopedConfig().global_model_daily_group_limit_enabled is True
    assert ScopedConfig().global_model_daily_group_limit == 50
    assert ScopedConfig().global_model_daily_private_user_limit_enabled is True
    assert ScopedConfig().global_model_daily_private_user_limit == 10
    with pytest.raises(ValidationError):
        ScopedConfig(global_model_daily_group_limit=-1)
    with pytest.raises(ValidationError):
        ScopedConfig(global_model_daily_private_user_limit=-1)


def test_private_quota_message_contains_limit_and_wait_time():
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        GroupDailyQuotaStatus,
        build_private_quota_exhausted_message,
    )

    timezone = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime(2026, 8, 26, 20, 30, tzinfo=timezone)
    status = GroupDailyQuotaStatus(
        allowed=False,
        used=10,
        limit=10,
        resets_at=datetime.datetime(2026, 8, 27, tzinfo=timezone),
    )

    message = build_private_quota_exhausted_message(status, now=now)

    assert "私聊公共模型额度（10 次）已用完" in message
    assert "/配置个人API" in message
    assert "约 3 小时 30 分钟后恢复" in message


@pytest.mark.asyncio
async def test_exhausted_private_user_quota_notifies_and_skips_agent(monkeypatch):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        GroupDailyQuotaStatus,
    )

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return [
                SimpleNamespace(
                    msg_id=1,
                    session_id="private-user-1",
                    user_id="private-user-1",
                    content_type="text",
                    content="你好",
                    created_at=datetime.datetime.now(),
                    user_name="tester",
                    media_id=None,
                    vectorized=False,
                    vectorized_version=0,
                )
            ]

    class _Session:
        async def execute(self, _statement):
            return _Result()

        async def commit(self):
            return None

    opened_sessions = 0

    @asynccontextmanager
    async def fake_get_session():
        nonlocal opened_sessions
        opened_sessions += 1
        yield _Session()

    async def fake_status(*_args, **_kwargs):
        return GroupDailyQuotaStatus(
            allowed=False,
            used=10,
            limit=10,
            resets_at=datetime.datetime.now().astimezone()
            + datetime.timedelta(hours=2),
        )

    notices: list[GroupDailyQuotaStatus] = []

    async def fake_notice(_db_session, *, status, **_kwargs):
        notices.append(status)

    async def unexpected_agent(*_args, **_kwargs):
        raise AssertionError("私聊额度耗尽时不应调用主 Agent")

    monkeypatch.setattr(
        plugin.plugin_config,
        "global_model_daily_private_user_limit",
        10,
    )
    monkeypatch.setattr(plugin, "get_session", fake_get_session)
    monkeypatch.setattr(
        plugin,
        "get_private_user_daily_quota_status",
        fake_status,
    )
    monkeypatch.setattr(plugin, "_send_quota_notice", fake_notice)
    monkeypatch.setattr(plugin, "choice_response_strategy", unexpected_agent)

    fake_session = SimpleNamespace(
        scene=SimpleNamespace(id="private-user-1", type=SceneType.PRIVATE),
        self_id="bot-1",
    )
    await plugin.handle_reply_logic(
        "private-quota-request",
        cast(Uninfo, fake_session),
        cast(QryItrface, SimpleNamespace()),
        cast(Bot, SimpleNamespace()),
        cast(Event, SimpleNamespace()),
        "bot",
        "private-user-1",
        "tester",
        True,
        False,
        None,
    )

    assert len(notices) == 1
    assert notices[0].limit == 10
    assert opened_sessions == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_tome", "expected_notices"),
    [(True, 1), (False, 0)],
)
async def test_exhausted_group_only_notifies_explicit_requests(
    monkeypatch,
    is_tome: bool,
    expected_notices: int,
):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.group_daily_quota import (
        GroupDailyQuotaStatus,
    )

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return [
                SimpleNamespace(
                    msg_id=1,
                    session_id="quota-notice-group",
                    user_id="user-1",
                    content_type="text",
                    content="bot 在吗",
                    created_at=datetime.datetime.now(),
                    user_name="tester",
                    media_id=None,
                    vectorized=False,
                    vectorized_version=0,
                )
            ]

    class _Session:
        async def execute(self, _statement):
            return _Result()

        async def commit(self):
            return None

    opened_sessions = 0

    @asynccontextmanager
    async def fake_get_session():
        nonlocal opened_sessions
        opened_sessions += 1
        yield _Session()

    async def fake_status(*_args, **_kwargs):
        return GroupDailyQuotaStatus(
            allowed=False,
            used=2,
            limit=2,
            resets_at=datetime.datetime.now().astimezone()
            + datetime.timedelta(hours=2),
        )

    notices: list[GroupDailyQuotaStatus] = []

    async def fake_notice(_db_session, *, status, **_kwargs):
        notices.append(status)

    async def unexpected_agent(*_args, **_kwargs):
        raise AssertionError("额度耗尽时不应调用主 Agent")

    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit", 2)
    monkeypatch.setattr(plugin, "has_group_model_config", lambda _group_id: False)
    monkeypatch.setattr(plugin, "get_session", fake_get_session)
    monkeypatch.setattr(plugin, "get_group_daily_quota_status", fake_status)
    monkeypatch.setattr(plugin, "_send_quota_notice", fake_notice)
    monkeypatch.setattr(plugin, "choice_response_strategy", unexpected_agent)

    fake_session = SimpleNamespace(
        scene=SimpleNamespace(id="quota-notice-group", type=SceneType.GROUP),
        self_id="bot-1",
    )
    await plugin.handle_reply_logic(
        "quota-request",
        cast(Uninfo, fake_session),
        cast(QryItrface, SimpleNamespace()),
        cast(Bot, SimpleNamespace()),
        cast(Event, SimpleNamespace()),
        "bot",
        "user-1",
        "tester",
        is_tome,
        False,
        None,
    )

    assert len(notices) == expected_notices
    assert opened_sessions == 1 + expected_notices


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quota_enabled", "has_custom_api"),
    [(True, True), (False, False)],
)
async def test_custom_api_or_disabled_switch_bypasses_public_quota(
    monkeypatch,
    quota_enabled: bool,
    has_custom_api: bool,
):
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface

    import nonebot_plugin_ai_groupmate as plugin

    history_row = SimpleNamespace(
        msg_id=1,
        session_id="custom-api-quota-group",
        user_id="user-1",
        content_type="text",
        content="bot 在吗",
        created_at=datetime.datetime.now(),
        user_name="tester",
        media_id=None,
        vectorized=False,
        vectorized_version=0,
    )

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return [history_row]

    class _Session:
        async def execute(self, _statement):
            return _Result()

        async def commit(self):
            return None

    @asynccontextmanager
    async def fake_get_session():
        yield _Session()

    async def quota_must_not_run(*_args, **_kwargs):
        raise AssertionError("已配置群 API 时不应读取或消耗公共额度")

    async def fake_load_history(_db_session, _session_id):
        return [history_row]

    agent_calls = 0

    async def fake_agent(*_args, **_kwargs):
        nonlocal agent_calls
        agent_calls += 1
        return "done"

    class _Interface:
        async def get_members(self, _scene_type, _scene_id):
            return []

    monkeypatch.setattr(plugin.plugin_config, "global_model_daily_group_limit", 2)
    monkeypatch.setattr(
        plugin.plugin_config,
        "global_model_daily_group_limit_enabled",
        quota_enabled,
    )
    monkeypatch.setattr(
        plugin,
        "has_group_model_config",
        lambda _group_id: has_custom_api,
    )
    monkeypatch.setattr(plugin, "get_session", fake_get_session)
    monkeypatch.setattr(plugin, "get_group_daily_quota_status", quota_must_not_run)
    monkeypatch.setattr(plugin, "consume_group_daily_quota", quota_must_not_run)
    monkeypatch.setattr(plugin, "_load_agent_history", fake_load_history)
    monkeypatch.setattr(plugin, "choice_response_strategy", fake_agent)

    fake_session = SimpleNamespace(
        scene=SimpleNamespace(id="custom-api-quota-group", type=SceneType.GROUP),
        self_id="bot-1",
    )
    await plugin.handle_reply_logic(
        "custom-api-request",
        cast(Uninfo, fake_session),
        cast(QryItrface, _Interface()),
        cast(Bot, SimpleNamespace()),
        cast(Event, SimpleNamespace()),
        "bot",
        "user-1",
        "tester",
        True,
        False,
        None,
    )

    assert agent_calls == 1
