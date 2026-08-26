import datetime

import pytest
from sqlalchemy import delete


def _payload(
    *,
    ticket_id: str = "ticket-1",
    reply_probability: float | None = 0.075,
):
    from nonebot_plugin_ai_groupmate.group_model_config import GroupModelPayload

    return GroupModelPayload(
        ticket_id=ticket_id,
        api_format="openai",
        base_url="https://group-provider.example/v1/",
        api_key="sk-group-secret",
        chat_model="group-chat-model",
        chat_multimodal=False,
        reply_probability=reply_probability,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )


def test_local_secret_cipher_round_trip_is_purpose_bound():
    from nonebot_plugin_ai_groupmate.group_model_config import (
        LocalSecretCipher,
        LocalEncryptionKeyError,
    )

    cipher = LocalSecretCipher(LocalSecretCipher.generate_key())
    ciphertext = cipher.encrypt("sk-sensitive", purpose="group:1")

    assert "sk-sensitive" not in ciphertext
    assert cipher.decrypt(ciphertext, purpose="group:1") == "sk-sensitive"
    with pytest.raises(LocalEncryptionKeyError, match="无法解密"):
        cipher.decrypt(ciphertext, purpose="group:2")
    with pytest.raises(LocalEncryptionKeyError, match="格式无效"):
        LocalSecretCipher("!" + LocalSecretCipher.generate_key())


def test_local_encryption_key_is_generated_once_in_plugin_data(tmp_path):
    from nonebot_plugin_ai_groupmate.group_model_config import (
        LocalSecretCipher,
        load_or_create_local_encryption_key,
    )

    key_path = tmp_path / "group_api_local_encryption.key"
    first = load_or_create_local_encryption_key(key_path)
    second = load_or_create_local_encryption_key(key_path)

    assert first == second
    assert key_path.read_text(encoding="utf-8").strip() == first
    LocalSecretCipher(first)


def test_damaged_generated_local_key_is_not_silently_replaced(tmp_path):
    from nonebot_plugin_ai_groupmate.group_model_config import (
        LocalEncryptionKeyError,
        load_or_create_local_encryption_key,
    )

    key_path = tmp_path / "group_api_local_encryption.key"
    key_path.write_text("damaged", encoding="utf-8")
    with pytest.raises(LocalEncryptionKeyError, match="32 字节"):
        load_or_create_local_encryption_key(key_path)
    assert key_path.read_text(encoding="utf-8") == "damaged"


def test_group_payload_normalizes_base_url_and_rejects_non_http():
    from pydantic import ValidationError

    from nonebot_plugin_ai_groupmate.group_model_config import GroupModelPayload

    assert _payload().base_url == "https://group-provider.example/v1"
    with pytest.raises(ValidationError, match="Base URL"):
        GroupModelPayload(
            ticket_id="ticket",
            base_url="ftp://provider.example",
            api_key="secret",
            chat_model="model",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
    with pytest.raises(ValidationError, match="主机名"):
        GroupModelPayload(
            ticket_id="ticket",
            base_url="https:provider.example",
            api_key="secret",
            chat_model="model",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
    with pytest.raises(ValidationError, match="暂不支持"):
        GroupModelPayload(
            ticket_id="ticket",
            base_url="https://provider.example/v1",
            api_key="secret",
            chat_model="model",
            allow_global_fallback=True,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
    assert _payload(reply_probability=0.1).reply_probability == 0.1
    with pytest.raises(ValidationError, match="less than or equal to 0.1"):
        _payload(reply_probability=0.1001)


def test_group_provider_policy_blocks_internal_targets_unless_allowlisted():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.group_model_config import (
        GroupModelPayload,
        GroupModelConfigError,
        build_candidate_chat_config,
    )

    internal_payload = GroupModelPayload(
        ticket_id="ticket",
        base_url="https://model.service.internal/v1",
        api_key="secret",
        chat_model="model",
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    with pytest.raises(GroupModelConfigError, match="内网"):
        build_candidate_chat_config(internal_payload, ScopedConfig())

    allowed = ScopedConfig(
        group_api_allowed_provider_hosts=["*.service.internal"],
    )
    assert build_candidate_chat_config(internal_payload, allowed).chat_model == "model"


def test_group_model_connection_requires_non_empty_response():
    from types import SimpleNamespace

    from nonebot_plugin_ai_groupmate.group_model_config import (
        GroupModelConfigError,
        validate_group_model_test_response,
    )

    validate_group_model_test_response(SimpleNamespace(content="OK"))
    validate_group_model_test_response(
        SimpleNamespace(content=[{"type": "text", "text": "OK"}])
    )
    with pytest.raises(GroupModelConfigError, match="空响应"):
        validate_group_model_test_response(SimpleNamespace(content="  "))
    with pytest.raises(GroupModelConfigError, match="空响应"):
        validate_group_model_test_response(SimpleNamespace(content=[]))


@pytest.mark.asyncio
async def test_group_provider_resolution_rejects_private_dns_result(monkeypatch):
    import socket

    from nonebot_plugin_ai_groupmate.group_model_config import (
        GroupModelConfigError,
        validate_group_provider_resolution,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.20", 443))],
    )
    with pytest.raises(GroupModelConfigError, match="非公网"):
        await validate_group_provider_resolution("https://provider.example/v1", [])


def test_agent_builds_group_model_with_resolved_credentials(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.group_model_config import (
        ActiveGroupModelConfig,
        set_active_group_model_config,
        clear_active_group_model_configs,
    )

    captured = []
    group_id = "group-model-factory"
    clear_active_group_model_configs()
    set_active_group_model_config(
        ActiveGroupModelConfig(
            group_id=group_id,
            api_format="openai",
            base_url="https://group-factory.example/v1",
            api_key="sk-factory-secret",
            chat_model="factory-model",
            chat_multimodal=True,
        )
    )
    monkeypatch.setattr(
        agent,
        "create_chat_llm",
        lambda config: captured.append(config) or object(),
    )
    agent.clear_group_chat_model_cache(group_id)
    try:
        agent.get_group_chat_model(group_id)
        assert captured[0].chat_model == "factory-model"
        assert captured[0].chat_api_key == "sk-factory-secret"
        assert captured[0].chat_base_url == "https://group-factory.example/v1"
    finally:
        agent.clear_group_chat_model_cache(group_id)
        clear_active_group_model_configs()


def test_group_config_permission_accepts_admin_role(monkeypatch):
    from types import SimpleNamespace
    from typing import cast

    from nonebot.adapters import Event
    from nonebot_plugin_uninfo import Uninfo

    from nonebot_plugin_ai_groupmate import group_api_commands

    monkeypatch.setattr(group_api_commands, "_is_superuser", lambda _user_id: False)
    session = SimpleNamespace(
        user=SimpleNamespace(id="user-1"),
        member=SimpleNamespace(role=SimpleNamespace(name="admin")),
    )
    typed_session = cast(Uninfo, session)
    event = cast(Event, SimpleNamespace())
    assert group_api_commands.can_manage_group_config(typed_session, event)
    session.member.role.name = "member"
    assert not group_api_commands.can_manage_group_config(typed_session, event)


def test_group_config_private_message_shows_remaining_minutes():
    from nonebot_plugin_ai_groupmate import group_api_commands
    from nonebot_plugin_ai_groupmate.group_api_relay import ConfigTicket

    now = datetime.datetime(2026, 8, 25, 10, 0, tzinfo=datetime.timezone.utc)
    ticket = ConfigTicket(
        ticket_id="ticket-private-message",
        config_url="https://mayumi.xyz/config/ticket-private-message#token=secret",
        expires_at=now + datetime.timedelta(minutes=15),
    )

    message, validity_minutes = group_api_commands._build_ticket_private_message(
        ticket,
        now=now,
    )

    assert validity_minutes == 15
    assert "约 15 分钟内有效" in message
    assert "10:15:00" not in message
    assert "配置码同样需要在有效期内提交" in message


@pytest.mark.asyncio
async def test_group_model_config_is_encrypted_persisted_and_resolved():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GroupModelConfig
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.group_model_config import (
        LocalSecretCipher,
        resolve_chat_config,
        save_group_model_config,
        list_group_model_configs,
        delete_group_model_config,
        resolve_group_reply_probability,
        clear_active_group_model_configs,
        get_decrypted_group_model_config,
    )

    group_id = "group-config-test"
    clear_active_group_model_configs()
    cipher = LocalSecretCipher(LocalSecretCipher.generate_key())
    global_config = ScopedConfig(
        chat_model="global-model",
        chat_api_key="global-key",
        chat_base_url="https://global.example/v1",
        reply_probability=0.02,
    )

    async with get_session() as session:
        await session.execute(delete(GroupModelConfig).where(GroupModelConfig.group_id == group_id))
        await session.commit()
        active = await save_group_model_config(
            session,
            group_id=group_id,
            operator_id="admin-1",
            payload=_payload(),
            cipher=cipher,
        )
        row = await session.get(GroupModelConfig, group_id)
        assert row is not None
        assert "sk-group-secret" not in row.api_key_ciphertext
        assert row.reply_probability == 0.075
        assert active.version == 1
        assert active.reply_probability == 0.075

        admin_rows = await list_group_model_configs(session)
        admin_row = next(item for item in admin_rows if item.group_id == group_id)
        assert admin_row.base_url == "https://group-provider.example/v1"
        assert admin_row.api_key_configured is True
        assert admin_row.reply_probability == 0.075
        assert "api_key" not in admin_row.model_dump()
        decrypted = await get_decrypted_group_model_config(
            session,
            group_id,
            cipher,
        )
        assert decrypted is not None
        assert decrypted.api_key == "sk-group-secret"

        resolved = resolve_chat_config(group_id, global_config)
        assert resolved.chat_model == "group-chat-model"
        assert resolved.chat_api_key == "sk-group-secret"
        assert resolved.chat_base_url == "https://group-provider.example/v1"
        assert resolved.chat_multimodal is False
        assert resolve_group_reply_probability(group_id, global_config) == 0.075

        untouched = resolve_chat_config("another-group", global_config)
        assert untouched is global_config
        assert resolve_group_reply_probability("another-group", global_config) == 0.02

        assert await delete_group_model_config(session, group_id) is True
        assert resolve_chat_config(group_id, global_config) is global_config
        assert resolve_group_reply_probability(group_id, global_config) == 0.02


@pytest.mark.asyncio
async def test_loading_with_wrong_local_key_does_not_activate_group():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GroupModelConfig
    from nonebot_plugin_ai_groupmate.group_model_config import (
        LocalSecretCipher,
        has_group_model_config,
        save_group_model_config,
        load_group_model_configs,
        clear_active_group_model_configs,
    )

    group_id = "group-config-wrong-key"
    first_cipher = LocalSecretCipher(LocalSecretCipher.generate_key())
    second_cipher = LocalSecretCipher(LocalSecretCipher.generate_key())
    clear_active_group_model_configs()
    async with get_session() as session:
        await session.execute(delete(GroupModelConfig).where(GroupModelConfig.group_id == group_id))
        await session.commit()
        await save_group_model_config(
            session,
            group_id=group_id,
            operator_id="admin-1",
            payload=_payload(),
            cipher=first_cipher,
        )
        clear_active_group_model_configs()
        assert await load_group_model_configs(session, second_cipher) == 0
        assert has_group_model_config(group_id) is False
        await session.execute(delete(GroupModelConfig).where(GroupModelConfig.group_id == group_id))
        await session.commit()


@pytest.mark.asyncio
async def test_private_model_config_is_encrypted_resolved_and_deleted():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import PrivateModelConfig
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.group_model_config import (
        LocalSecretCipher,
        has_private_model_config,
        save_private_model_config,
        delete_private_model_config,
        resolve_session_chat_config,
        get_private_model_config_summary,
        clear_active_private_model_configs,
    )

    user_id = "private-config-test"
    cipher = LocalSecretCipher(LocalSecretCipher.generate_key())
    global_config = ScopedConfig(chat_model="global-model", chat_api_key="global-key")
    clear_active_private_model_configs()
    async with get_session() as session:
        await session.execute(
            delete(PrivateModelConfig).where(PrivateModelConfig.user_id == user_id)
        )
        await session.commit()
        active = await save_private_model_config(
            session,
            user_id=user_id,
            payload=_payload(),
            cipher=cipher,
        )
        row = await session.get(PrivateModelConfig, user_id)
        assert row is not None
        assert "sk-group-secret" not in row.api_key_ciphertext
        assert active.chat_model == "group-chat-model"
        assert has_private_model_config(user_id)

        resolved = resolve_session_chat_config(
            session_id=user_id,
            user_id=user_id,
            is_private=True,
            global_config=global_config,
        )
        assert resolved.chat_model == "group-chat-model"
        assert resolved.chat_api_key == "sk-group-secret"
        summary = await get_private_model_config_summary(session, user_id)
        assert summary is not None
        assert summary.provider_host == "group-provider.example"

        assert await delete_private_model_config(session, user_id)
        assert not has_private_model_config(user_id)
        assert (
            resolve_session_chat_config(
                session_id=user_id,
                user_id=user_id,
                is_private=True,
                global_config=global_config,
            )
            is global_config
        )
