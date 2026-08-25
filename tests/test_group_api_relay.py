import os
import datetime
from typing import Any

import pytest
from sqlalchemy import delete
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class FakeRelayTransport:
    def __init__(self, *, registration_authorization: str | None = "Bearer registration-token"):
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self.public_key_jwk: dict[str, Any] | None = None
        self.instance_token = "instance-secret-token"
        self.ticket_id = "ticket-relay-test"
        self.registration_authorization = registration_authorization

    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        authorization: str | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((path, payload, authorization))
        if path == "/v1/instances/register":
            self.public_key_jwk = payload["public_key_jwk"]
            assert authorization == self.registration_authorization
            assert idempotency_key
            return {
                "instance_id": "instance-relay-test",
                "instance_token": self.instance_token,
                "key_id": "key-relay-test",
                "created_at": "2026-08-24T08:00:00Z",
            }
        if path == "/v1/config-tickets":
            assert authorization == f"Bearer {self.instance_token}"
            return {
                "ticket_id": self.ticket_id,
                "config_url": "https://relay.example/config/ticket#token=submit",
                "expires_at": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)).isoformat(),
            }
        if path == "/v1/config-payloads/redeem":
            assert payload["code"] == "AGC-2345-67AB-ABCD-EFGH"
            return self._encrypted_redeem_response()
        if path.endswith("/ack"):
            assert payload["outcome"] == "applied"
            return {"ok": True}
        raise AssertionError(f"unexpected relay path: {path}")

    def _encrypted_redeem_response(self) -> dict[str, Any]:
        from nonebot_plugin_ai_groupmate.group_api_relay import (
            _b64url_decode,
            _b64url_encode,
            relay_payload_aad,
        )

        assert self.public_key_jwk is not None
        public_key = rsa.RSAPublicNumbers(
            e=int.from_bytes(_b64url_decode(str(self.public_key_jwk["e"])), "big"),
            n=int.from_bytes(_b64url_decode(str(self.public_key_jwk["n"])), "big"),
        ).public_key()
        aes_key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        plaintext = (
            "{"
            f'"schema_version":1,"ticket_id":"{self.ticket_id}",'
            '"api_format":"openai",'
            '"base_url":"https://group.example/v1",'
            '"api_key":"sk-relay-secret",'
            '"chat_model":"relay-model",'
            '"chat_multimodal":true,'
            '"allow_global_fallback":false,'
            f'"created_at":"{datetime.datetime.now(datetime.timezone.utc).isoformat()}"'
            "}"
        ).encode()
        ciphertext = AESGCM(aes_key).encrypt(
            nonce,
            plaintext,
            relay_payload_aad(
                ticket_id=self.ticket_id,
                instance_id="instance-relay-test",
                key_id="key-relay-test",
            ),
        )
        wrapped_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return {
            "delivery_id": "delivery-relay-test",
            "ticket_id": self.ticket_id,
            "envelope": {
                "protocol_version": 1,
                "key_id": "key-relay-test",
                "wrapped_key": _b64url_encode(wrapped_key),
                "nonce": _b64url_encode(nonce),
                "ciphertext": _b64url_encode(ciphertext),
            },
        }


def _relay_config(*, registration_token: str = "registration-token"):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.group_model_config import LocalSecretCipher

    return ScopedConfig(
        group_api_relay_url="https://relay.example",
        group_api_relay_registration_token=registration_token,
        group_api_local_encryption_key=LocalSecretCipher.generate_key(),
    )


def test_config_code_normalization_rejects_short_or_ambiguous_codes():
    from nonebot_plugin_ai_groupmate.group_api_relay import (
        RelayTicketError,
        normalize_config_code,
    )

    assert normalize_config_code("agc 2345 a7ab cdef ghjk") == "AGC-2345-A7AB-CDEF-GHJK"
    with pytest.raises(RelayTicketError, match="格式"):
        normalize_config_code("123456")
    with pytest.raises(RelayTicketError, match="格式"):
        normalize_config_code("AGC-0000-0000-0000-0000")


@pytest.mark.asyncio
async def test_relay_register_ticket_redeem_and_ack_round_trip():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import (
        PendingGroupConfig,
        RelayInstanceIdentity,
    )
    from nonebot_plugin_ai_groupmate.group_api_relay import GroupModelRelay

    transport = FakeRelayTransport()
    relay = GroupModelRelay(_relay_config(), transport=transport)
    async with get_session() as session:
        await session.execute(delete(PendingGroupConfig))
        await session.execute(delete(RelayInstanceIdentity))
        await session.commit()
        first_identity = await relay.ensure_registered(session)
        second_identity = await relay.ensure_registered(session)
        assert first_identity.instance_id == second_identity.instance_id
        assert [call[0] for call in transport.calls].count("/v1/instances/register") == 1

        ticket = await relay.create_ticket(
            session,
            group_id="group-relay-test",
            operator_id="admin-relay-test",
        )
        assert ticket.ticket_id == transport.ticket_id
        pending = await session.get(PendingGroupConfig, transport.ticket_id)
        assert pending is not None
        assert pending.group_id == "group-relay-test"

        # Base32 deliberately excludes 0, 1, 8 and 9.
        code = "AGC-2345-67AB-ABCD-EFGH"
        redeemed = await relay.redeem(
            session,
            code=code,
            operator_id="admin-relay-test",
        )
        assert redeemed.group_id == "group-relay-test"
        assert redeemed.payload.api_key == "sk-relay-secret"
        assert redeemed.payload.chat_model == "relay-model"

        await relay.acknowledge(session, redeemed, outcome="applied")
        assert await session.get(PendingGroupConfig, transport.ticket_id) is None

        identity_row = await session.get(RelayInstanceIdentity, 1)
        assert identity_row is not None
        assert "instance-secret-token" not in identity_row.instance_token_ciphertext
        assert "PRIVATE KEY" not in identity_row.private_key_ciphertext

        await session.execute(delete(RelayInstanceIdentity))
        await session.commit()


@pytest.mark.asyncio
async def test_relay_can_register_without_token_when_server_is_public():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import RelayInstanceIdentity
    from nonebot_plugin_ai_groupmate.group_api_relay import GroupModelRelay

    transport = FakeRelayTransport(registration_authorization=None)
    relay = GroupModelRelay(
        _relay_config(registration_token=""),
        transport=transport,
    )
    async with get_session() as session:
        await session.execute(delete(RelayInstanceIdentity))
        await session.commit()
        identity = await relay.ensure_registered(session)
        assert identity.instance_id == "instance-relay-test"
        assert transport.calls[0][2] is None
        await session.execute(delete(RelayInstanceIdentity))
        await session.commit()


@pytest.mark.asyncio
async def test_relay_rejects_code_from_another_operator():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import (
        PendingGroupConfig,
        RelayInstanceIdentity,
    )
    from nonebot_plugin_ai_groupmate.group_api_relay import (
        GroupModelRelay,
        RelayTicketError,
    )

    transport = FakeRelayTransport()
    relay = GroupModelRelay(_relay_config(), transport=transport)
    async with get_session() as session:
        await session.execute(delete(PendingGroupConfig))
        await session.execute(delete(RelayInstanceIdentity))
        await session.commit()
        await relay.create_ticket(
            session,
            group_id="group-relay-test",
            operator_id="original-admin",
        )
        with pytest.raises(RelayTicketError, match="本人"):
            await relay.redeem(
                session,
                code="AGC-2345-67AB-ABCD-EFGH",
                operator_id="other-admin",
            )
        await session.execute(delete(PendingGroupConfig))
        await session.execute(delete(RelayInstanceIdentity))
        await session.commit()
