import re
import uuid
import asyncio
import datetime
from typing import Any, Literal, Protocol, cast
from importlib import metadata
from dataclasses import field, dataclass
from urllib.parse import urlsplit

import httpx
from pydantic import Field, BaseModel, ValidationError
from sqlalchemy import Select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .model import PendingGroupConfig, RelayInstanceIdentity
from .config import ScopedConfig
from .group_model_config import (
    GroupModelPayload,
    LocalSecretCipher,
    _b64url_decode,
    _b64url_encode,
)

PROTOCOL_VERSION = 1
INSTANCE_TOKEN_PURPOSE = "ai-groupmate:relay-instance-token"
PRIVATE_KEY_PURPOSE = "ai-groupmate:relay-private-key"
CONFIG_CODE_PATTERN = re.compile(
    r"^AGC(?:-[A-Z2-7]{4}){4}$",
    re.IGNORECASE,
)

try:
    PLUGIN_VERSION = metadata.version("nonebot-plugin-ai-groupmate")
except metadata.PackageNotFoundError:
    PLUGIN_VERSION = "unknown"


class RelayError(RuntimeError):
    code = "relay_error"


class RelayDisabledError(RelayError):
    code = "relay_disabled"


class RelayAuthenticationError(RelayError):
    code = "relay_authentication_failed"


class RelayConnectionError(RelayError):
    code = "relay_connection_failed"


class RelayProtocolError(RelayError):
    code = "relay_protocol_error"


class RelayTicketError(RelayError):
    code = "relay_ticket_invalid"


class RelayPayloadError(RelayError):
    code = "relay_payload_invalid"


class RelayTransport(Protocol):
    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        authorization: str | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]: ...


class HttpRelayTransport:
    """HTTPS adapter for the relay protocol."""

    def __init__(self, base_url: str, *, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        authorization: str | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if authorization:
            headers["Authorization"] = authorization
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            ) as client:
                response = await client.post(path, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise RelayConnectionError("无法连接群 API 中转服务") from error
        except httpx.HTTPError as error:
            raise RelayConnectionError("群 API 中转请求失败") from error

        if response.status_code in {401, 403}:
            raise RelayAuthenticationError("群 API 中转身份验证失败")
        if response.status_code == 404:
            raise RelayTicketError("配置单不存在或已过期")
        if response.status_code == 409:
            raise RelayTicketError("配置单状态不允许当前操作")
        if response.status_code >= 400:
            raise RelayProtocolError(f"群 API 中转返回异常状态 {response.status_code}")
        try:
            body = response.json()
        except ValueError as error:
            raise RelayProtocolError("群 API 中转返回了无效 JSON") from error
        if not isinstance(body, dict):
            raise RelayProtocolError("群 API 中转返回格式无效")
        return body


class _RegistrationResponse(BaseModel):
    instance_id: str = Field(min_length=1, max_length=160)
    instance_token: str = Field(min_length=1, max_length=8192, repr=False)
    key_id: str = Field(min_length=1, max_length=160)
    created_at: datetime.datetime


class _TicketResponse(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=160)
    config_url: str = Field(min_length=1, max_length=4096)
    expires_at: datetime.datetime


class _Envelope(BaseModel):
    protocol_version: Literal[1]
    key_id: str = Field(min_length=1, max_length=160)
    wrapped_key: str = Field(min_length=1, max_length=8192, repr=False)
    nonce: str = Field(min_length=1, max_length=128, repr=False)
    ciphertext: str = Field(min_length=1, max_length=65536, repr=False)


class _RedeemResponse(BaseModel):
    delivery_id: str = Field(min_length=1, max_length=160)
    ticket_id: str = Field(min_length=1, max_length=160)
    envelope: _Envelope


@dataclass(frozen=True)
class InstanceIdentity:
    instance_id: str
    instance_token: str = field(repr=False)
    key_id: str
    public_key_jwk: dict[str, object]
    private_key_pem: bytes = field(repr=False)


@dataclass(frozen=True)
class ConfigTicket:
    ticket_id: str
    config_url: str
    expires_at: datetime.datetime


@dataclass(frozen=True)
class ConfigTarget:
    scope: Literal["group", "private"]
    subject_id: str
    operator_id: str


@dataclass(frozen=True)
class RedeemedModelConfig:
    delivery_id: str
    ticket_id: str
    target: ConfigTarget
    payload: GroupModelPayload = field(repr=False)


def normalize_config_code(value: str) -> str:
    compact = re.sub(r"[\s-]+", "", str(value or "")).upper()
    if compact.startswith("AGC"):
        raw = compact[3:]
    else:
        raw = compact
    if len(raw) != 16 or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for char in raw):
        raise RelayTicketError("配置码格式无效")
    normalized = "AGC-" + "-".join(raw[index : index + 4] for index in range(0, 16, 4))
    if not CONFIG_CODE_PATTERN.fullmatch(normalized):
        raise RelayTicketError("配置码格式无效")
    return normalized


def _utc_naive(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def _utc_now_naive() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _public_key_jwk(public_key: rsa.RSAPublicKey) -> dict[str, object]:
    numbers = public_key.public_numbers()
    modulus = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")
    exponent = numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")
    return {
        "kty": "RSA",
        "alg": "RSA-OAEP-256",
        "use": "enc",
        "n": _b64url_encode(modulus),
        "e": _b64url_encode(exponent),
    }


def _generate_rsa_identity() -> tuple[dict[str, object], bytes]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return _public_key_jwk(private_key.public_key()), private_pem


def relay_payload_aad(
    *,
    ticket_id: str,
    instance_id: str,
    key_id: str,
) -> bytes:
    return (f"ai-groupmate-config:v1:{ticket_id}:{instance_id}:{key_id}").encode()


def decrypt_relay_payload(
    *,
    private_key_pem: bytes,
    instance_id: str,
    ticket_id: str,
    envelope: _Envelope,
) -> GroupModelPayload:
    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
        )
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise RelayPayloadError("中转身份私钥类型无效")
        aes_key = private_key.decrypt(
            _b64url_decode(envelope.wrapped_key),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        plaintext = AESGCM(aes_key).decrypt(
            _b64url_decode(envelope.nonce),
            _b64url_decode(envelope.ciphertext),
            relay_payload_aad(
                ticket_id=ticket_id,
                instance_id=instance_id,
                key_id=envelope.key_id,
            ),
        )
        payload = GroupModelPayload.model_validate_json(plaintext)
    except RelayPayloadError:
        raise
    except (ValueError, TypeError, ValidationError) as error:
        raise RelayPayloadError("配置密文无法解密、已被篡改或内容无效") from error
    except Exception as error:
        raise RelayPayloadError("配置密文无法解密、已被篡改或内容无效") from error
    if payload.ticket_id != ticket_id:
        raise RelayPayloadError("配置内容与配置单不匹配")
    return payload


class GroupModelRelay:
    """Deep plugin-side module for relay identity, tickets and redemption."""

    def __init__(
        self,
        config: ScopedConfig,
        *,
        transport: RelayTransport | None = None,
    ):
        self._config = config
        self._relay_url = config.group_api_relay_url.rstrip("/")
        self._registration_lock = asyncio.Lock()
        self._cipher = LocalSecretCipher(config.group_api_local_encryption_key) if config.group_api_local_encryption_key else None
        self._transport = transport or (
            HttpRelayTransport(
                self._relay_url,
                timeout_seconds=config.group_api_relay_timeout_seconds,
            )
            if self._relay_url
            else None
        )

    @property
    def enabled(self) -> bool:
        return bool(self._relay_url and self._cipher and self._transport)

    @property
    def cipher(self) -> LocalSecretCipher:
        if self._cipher is None:
            raise RelayDisabledError("尚未配置群 API 本地加密密钥")
        return self._cipher

    def _require_transport(self) -> RelayTransport:
        if not self._relay_url:
            raise RelayDisabledError("尚未配置群 API 中转地址")
        parsed_url = urlsplit(self._relay_url)
        local_development = parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
        if not parsed_url.hostname:
            raise RelayDisabledError("群 API 中转地址缺少有效主机名")
        if parsed_url.scheme != "https" and not (parsed_url.scheme == "http" and local_development):
            raise RelayDisabledError("群 API 中转地址必须使用 HTTPS")
        if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise RelayDisabledError("群 API 中转地址不能包含凭据、查询参数或 fragment")
        if parsed_url.path not in {"", "/"}:
            raise RelayDisabledError("群 API 中转地址必须填写服务根地址")
        if self._transport is None:
            raise RelayDisabledError("群 API 中转客户端未启用")
        if self._cipher is None:
            raise RelayDisabledError("尚未配置群 API 本地加密密钥")
        return self._transport

    async def ensure_registered(
        self,
        db_session: AsyncSession,
    ) -> InstanceIdentity:
        async with self._registration_lock:
            return await self._ensure_registered(db_session)

    async def _ensure_registered(
        self,
        db_session: AsyncSession,
    ) -> InstanceIdentity:
        transport = self._require_transport()
        row = await db_session.get(RelayInstanceIdentity, 1)
        if row is not None:
            if row.relay_url.rstrip("/") != self._relay_url:
                raise RelayProtocolError("中转地址已变更；请先注销旧实例身份再重新注册")
            return InstanceIdentity(
                instance_id=row.instance_id,
                instance_token=self.cipher.decrypt(
                    row.instance_token_ciphertext,
                    purpose=INSTANCE_TOKEN_PURPOSE,
                ),
                key_id=row.key_id,
                public_key_jwk=dict(row.public_key_jwk),
                private_key_pem=self.cipher.decrypt_bytes(
                    row.private_key_ciphertext,
                    purpose=PRIVATE_KEY_PURPOSE,
                ),
            )

        registration_token = self._config.group_api_relay_registration_token.strip()
        public_jwk, private_pem = await asyncio.to_thread(_generate_rsa_identity)
        response = await transport.post(
            "/v1/instances/register",
            {
                "protocol_version": PROTOCOL_VERSION,
                "public_key_jwk": public_jwk,
                "plugin_version": PLUGIN_VERSION,
            },
            authorization=(f"Bearer {registration_token}" if registration_token else None),
            idempotency_key=str(uuid.uuid4()),
        )
        try:
            registration = _RegistrationResponse.model_validate(response)
        except ValidationError as error:
            raise RelayProtocolError("中转服务注册响应格式无效") from error
        row = RelayInstanceIdentity(
            id=1,
            relay_url=self._relay_url,
            instance_id=registration.instance_id,
            instance_token_ciphertext=self.cipher.encrypt(
                registration.instance_token,
                purpose=INSTANCE_TOKEN_PURPOSE,
            ),
            public_key_jwk=public_jwk,
            private_key_ciphertext=self.cipher.encrypt(
                private_pem,
                purpose=PRIVATE_KEY_PURPOSE,
            ),
            key_id=registration.key_id,
            registered_at=_utc_naive(registration.created_at),
        )
        db_session.add(row)
        await db_session.commit()
        return InstanceIdentity(
            instance_id=registration.instance_id,
            instance_token=registration.instance_token,
            key_id=registration.key_id,
            public_key_jwk=public_jwk,
            private_key_pem=private_pem,
        )

    async def create_ticket(
        self,
        db_session: AsyncSession,
        *,
        target: ConfigTarget | None = None,
        group_id: str | None = None,
        operator_id: str | None = None,
    ) -> ConfigTicket:
        if target is None:
            if group_id is None or operator_id is None:
                raise TypeError("target or group_id/operator_id is required")
            target = ConfigTarget(
                scope="group",
                subject_id=str(group_id),
                operator_id=str(operator_id),
            )
        transport = self._require_transport()
        identity = await self.ensure_registered(db_session)
        ticket_request: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "expires_in": self._config.group_api_ticket_ttl_seconds,
        }
        # Preserve compatibility with protocol-v1 relays that predate personal
        # configuration: group tickets keep the original request shape.
        if target.scope != "group":
            ticket_request["scope"] = target.scope
        response = await transport.post(
            "/v1/config-tickets",
            ticket_request,
            authorization=f"Bearer {identity.instance_token}",
            idempotency_key=str(uuid.uuid4()),
        )
        try:
            ticket = _TicketResponse.model_validate(response)
        except ValidationError as error:
            raise RelayProtocolError("中转服务配置单响应格式无效") from error
        expires_at = _utc_naive(ticket.expires_at)
        if expires_at <= _utc_now_naive():
            raise RelayProtocolError("中转服务返回了已过期的配置单")
        config_url = urlsplit(ticket.config_url)
        if config_url.scheme != "https" or not config_url.hostname or config_url.username or config_url.password:
            raise RelayProtocolError("中转服务返回了不安全的配置链接")

        await db_session.execute(
            delete(PendingGroupConfig).where(
                PendingGroupConfig.scope == target.scope,
                PendingGroupConfig.group_id == target.subject_id,
                PendingGroupConfig.operator_id == target.operator_id,
            )
        )
        db_session.add(
            PendingGroupConfig(
                ticket_id=ticket.ticket_id,
                scope=target.scope,
                group_id=target.subject_id,
                operator_id=target.operator_id,
                expires_at=expires_at,
                created_at=_utc_now_naive(),
            )
        )
        await db_session.commit()
        return ConfigTicket(
            ticket_id=ticket.ticket_id,
            config_url=ticket.config_url,
            expires_at=expires_at,
        )

    async def redeem(
        self,
        db_session: AsyncSession,
        *,
        code: str,
        operator_id: str,
    ) -> RedeemedModelConfig:
        transport = self._require_transport()
        normalized_code = normalize_config_code(code)
        identity = await self.ensure_registered(db_session)
        response = await transport.post(
            "/v1/config-payloads/redeem",
            {"code": normalized_code},
            authorization=f"Bearer {identity.instance_token}",
            idempotency_key=str(uuid.uuid4()),
        )
        try:
            redeemed = _RedeemResponse.model_validate(response)
        except ValidationError as error:
            raise RelayProtocolError("中转服务兑换响应格式无效") from error
        pending = await db_session.get(PendingGroupConfig, redeemed.ticket_id)
        if pending is None:
            raise RelayTicketError("本地找不到对应的待处理配置单")
        if pending.operator_id != str(operator_id):
            if pending.scope == "private":
                raise RelayTicketError("请由发起配置的用户本人提交配置码")
            raise RelayTicketError("请由发起配置的管理员本人提交配置码")
        if pending.expires_at <= _utc_now_naive():
            raise RelayTicketError("配置单已过期，请重新生成")
        if redeemed.envelope.key_id != identity.key_id:
            raise RelayPayloadError("配置密文使用了未知的实例公钥")
        payload = await asyncio.to_thread(
            decrypt_relay_payload,
            private_key_pem=identity.private_key_pem,
            instance_id=identity.instance_id,
            ticket_id=redeemed.ticket_id,
            envelope=redeemed.envelope,
        )
        payload_created_at = _utc_naive(payload.created_at)
        if not (pending.created_at - datetime.timedelta(minutes=5) <= payload_created_at <= pending.expires_at):
            raise RelayPayloadError("配置内容的创建时间不在配置单有效期内")
        return RedeemedModelConfig(
            delivery_id=redeemed.delivery_id,
            ticket_id=redeemed.ticket_id,
            target=ConfigTarget(
                scope=cast(Literal["group", "private"], pending.scope),
                subject_id=pending.group_id,
                operator_id=pending.operator_id,
            ),
            payload=payload,
        )

    async def acknowledge(
        self,
        db_session: AsyncSession,
        redeemed: RedeemedModelConfig,
        *,
        outcome: str,
    ) -> None:
        transport = self._require_transport()
        identity = await self.ensure_registered(db_session)
        try:
            await transport.post(
                f"/v1/config-payloads/{redeemed.delivery_id}/ack",
                {"outcome": outcome},
                authorization=f"Bearer {identity.instance_token}",
                idempotency_key=str(uuid.uuid4()),
            )
        finally:
            pending = await db_session.get(PendingGroupConfig, redeemed.ticket_id)
            if pending is not None:
                await db_session.delete(pending)
                await db_session.commit()

    async def delete_expired_pending(self, db_session: AsyncSession) -> int:
        return await delete_expired_pending_group_configs(db_session)


async def delete_expired_pending_group_configs(db_session: AsyncSession) -> int:
    """Remove stale local ticket bindings without contacting the relay."""
    result = await db_session.execute(Select(PendingGroupConfig.ticket_id).where(PendingGroupConfig.expires_at <= _utc_now_naive()))
    ticket_ids = list(result.scalars().all())
    if ticket_ids:
        await db_session.execute(delete(PendingGroupConfig).where(PendingGroupConfig.ticket_id.in_(ticket_ids)))
        await db_session.commit()
    return len(ticket_ids)
