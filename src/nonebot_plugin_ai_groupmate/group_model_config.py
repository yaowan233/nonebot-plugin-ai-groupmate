import os
import base64
import socket
import asyncio
import datetime
import ipaddress
from typing import Literal, cast
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, BaseModel, ConfigDict, field_validator
from sqlalchemy import Select
from nonebot.log import logger
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .model import GroupModelConfig
from .config import ScopedConfig

LOCAL_CIPHERTEXT_VERSION = "v1"
GROUP_API_KEY_PURPOSE_PREFIX = "ai-groupmate:group-api-key:"
LOCAL_ENCRYPTION_KEY_FILENAME = "group_api_local_encryption.key"
MAX_GROUP_REPLY_PROBABILITY = 0.1


class GroupModelConfigError(RuntimeError):
    """Base error for safe, user-actionable group model configuration failures."""


class LocalEncryptionKeyError(GroupModelConfigError):
    pass


class GroupModelPayload(BaseModel):
    """Decrypted group configuration accepted from a relay payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    ticket_id: str = Field(min_length=1, max_length=160)
    api_format: Literal["openai", "anthropic", "vertex"] = "openai"
    base_url: str = Field(min_length=1, max_length=2048)
    api_key: str = Field(min_length=1, max_length=8192, repr=False)
    chat_model: str = Field(min_length=1, max_length=256)
    chat_multimodal: bool = True
    reply_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=MAX_GROUP_REPLY_PROBABILITY,
    )
    allow_global_fallback: bool = False
    created_at: datetime.datetime

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not parsed.hostname:
            raise ValueError("Base URL 缺少有效主机名")
        if parsed.scheme != "https":
            raise ValueError("Base URL 必须使用 HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Base URL 不能包含凭据、查询参数或 fragment")
        return value.rstrip("/")

    @field_validator("allow_global_fallback")
    @classmethod
    def _global_fallback_is_not_available_yet(cls, value: bool) -> bool:
        if value:
            raise ValueError("当前版本暂不支持失败后回退全局主模型")
        return value


class ActiveGroupModelConfig(BaseModel):
    """In-memory decrypted representation used by the model resolver."""

    model_config = ConfigDict(str_strip_whitespace=True)

    group_id: str
    api_format: Literal["openai", "anthropic", "vertex"]
    base_url: str
    api_key: str = Field(repr=False)
    chat_model: str
    chat_multimodal: bool
    reply_probability: float | None = None
    allow_global_fallback: bool = False
    version: int = 1


class GroupModelConfigSummary(BaseModel):
    group_id: str
    enabled: bool
    api_format: str
    provider_host: str
    chat_model: str
    chat_multimodal: bool
    reply_probability: float | None = None
    allow_global_fallback: bool
    updated_by: str
    updated_at: datetime.datetime
    last_test_status: str
    version: int


class GroupModelConfigDetail(GroupModelConfigSummary):
    """Admin-facing group configuration without the decrypted API key."""

    base_url: str
    api_key_configured: bool = True


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except Exception as error:
        raise LocalEncryptionKeyError("本地加密数据格式无效") from error


class LocalSecretCipher:
    """Encrypt secrets at rest with a deployment-owned 256-bit AES key."""

    def __init__(self, encoded_key: str):
        try:
            key = _b64url_decode(encoded_key.strip())
        except LocalEncryptionKeyError:
            raise
        if len(key) != 32:
            raise LocalEncryptionKeyError("GROUP_API_LOCAL_ENCRYPTION_KEY 必须是 32 字节 Base64URL 密钥")
        self._cipher = AESGCM(key)

    @staticmethod
    def generate_key() -> str:
        """Generate a 256-bit Base64URL key."""
        return _b64url_encode(os.urandom(32))

    def encrypt(self, plaintext: str | bytes, *, purpose: str) -> str:
        raw = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        nonce = os.urandom(12)
        ciphertext = self._cipher.encrypt(
            nonce,
            raw,
            purpose.encode("utf-8"),
        )
        return ".".join(
            (
                LOCAL_CIPHERTEXT_VERSION,
                _b64url_encode(nonce),
                _b64url_encode(ciphertext),
            )
        )

    def decrypt_bytes(self, encoded: str, *, purpose: str) -> bytes:
        try:
            version, nonce_value, ciphertext_value = encoded.split(".", 2)
            if version != LOCAL_CIPHERTEXT_VERSION:
                raise LocalEncryptionKeyError("不支持的本地密文版本")
            return self._cipher.decrypt(
                _b64url_decode(nonce_value),
                _b64url_decode(ciphertext_value),
                purpose.encode("utf-8"),
            )
        except LocalEncryptionKeyError:
            raise
        except Exception as error:
            raise LocalEncryptionKeyError("本地密文无法解密或已被篡改") from error

    def decrypt(self, encoded: str, *, purpose: str) -> str:
        try:
            return self.decrypt_bytes(encoded, purpose=purpose).decode("utf-8")
        except UnicodeDecodeError as error:
            raise LocalEncryptionKeyError("本地密文不是有效的 UTF-8 文本") from error


def load_or_create_local_encryption_key(path: Path) -> str:
    """Load the relay key from plugin data, creating it atomically once."""

    key_path = Path(path)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    def read_existing() -> str:
        try:
            value = key_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise LocalEncryptionKeyError("无法读取自动生成的群 API 本地密钥") from error
        # Validate before returning. A damaged key must never be silently
        # replaced because existing encrypted data depends on it.
        LocalSecretCipher(value)
        return value

    if key_path.exists():
        return read_existing()

    generated = LocalSecretCipher.generate_key()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(key_path, flags, 0o600)
    except FileExistsError:
        return read_existing()
    except OSError as error:
        raise LocalEncryptionKeyError("无法创建群 API 本地密钥文件") from error

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as key_file:
            key_file.write(generated + "\n")
            key_file.flush()
            os.fsync(key_file.fileno())
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            # Windows and some mounted filesystems do not implement POSIX
            # permission bits. The plugin data directory remains the boundary.
            pass
    except Exception:
        try:
            key_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return generated


_active_configs: dict[str, ActiveGroupModelConfig] = {}


def _api_key_purpose(group_id: str) -> str:
    return GROUP_API_KEY_PURPOSE_PREFIX + str(group_id)


def _host_matches_allowlist(hostname: str, allowed_hosts: list[str]) -> bool:
    hostname = hostname.rstrip(".").lower()
    for raw_pattern in allowed_hosts:
        pattern = str(raw_pattern).strip().rstrip(".").lower()
        if not pattern:
            continue
        if pattern.startswith("*."):
            suffix = pattern[1:]
            if hostname.endswith(suffix) and hostname != suffix[1:]:
                return True
        elif hostname == pattern:
            return True
    return False


def validate_group_provider_policy(
    base_url: str,
    allowed_hosts: list[str],
) -> None:
    """Reject internal-network targets unless the deployer explicitly allows one."""
    parsed = urlsplit(base_url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if allowed_hosts:
        if not _host_matches_allowlist(hostname, allowed_hosts):
            raise GroupModelConfigError("模型服务地址不在部署者允许的主机白名单中")
        return

    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise GroupModelConfigError("模型服务地址不能指向 Bot 内网")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        if "." not in hostname:
            raise GroupModelConfigError("模型服务地址不能使用内网短主机名") from None
    else:
        if not address.is_global:
            raise GroupModelConfigError("模型服务地址不能指向非公网 IP")


async def validate_group_provider_resolution(
    base_url: str,
    allowed_hosts: list[str],
) -> None:
    """Resolve an unlisted provider once before connecting to reduce SSRF risk."""
    validate_group_provider_policy(base_url, allowed_hosts)
    if allowed_hosts:
        return

    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    try:
        port = parsed.port or 443
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            port,
            0,
            socket.SOCK_STREAM,
        )
    except (OSError, ValueError) as error:
        raise GroupModelConfigError("无法解析模型服务地址") from error
    if not addresses:
        raise GroupModelConfigError("无法解析模型服务地址")
    for item in addresses:
        try:
            address = ipaddress.ip_address(item[4][0])
        except ValueError as error:
            raise GroupModelConfigError("模型服务地址解析结果无效") from error
        if not address.is_global:
            raise GroupModelConfigError("模型服务地址解析到了非公网 IP")


def _active_from_row(
    row: GroupModelConfig,
    cipher: LocalSecretCipher,
) -> ActiveGroupModelConfig:
    return ActiveGroupModelConfig(
        group_id=row.group_id,
        api_format=cast(Literal["openai", "anthropic", "vertex"], row.api_format),
        base_url=row.base_url,
        api_key=cipher.decrypt(
            row.api_key_ciphertext,
            purpose=_api_key_purpose(row.group_id),
        ),
        chat_model=row.chat_model,
        chat_multimodal=row.chat_multimodal,
        reply_probability=row.reply_probability,
        allow_global_fallback=row.allow_global_fallback,
        version=row.version,
    )


def _detail_from_row(row: GroupModelConfig) -> GroupModelConfigDetail:
    return GroupModelConfigDetail(
        group_id=row.group_id,
        enabled=row.enabled,
        api_format=row.api_format,
        provider_host=urlsplit(row.base_url).hostname or "未知",
        base_url=row.base_url,
        api_key_configured=bool(row.api_key_ciphertext),
        chat_model=row.chat_model,
        chat_multimodal=row.chat_multimodal,
        reply_probability=row.reply_probability,
        allow_global_fallback=row.allow_global_fallback,
        updated_by=row.updated_by,
        updated_at=row.updated_at,
        last_test_status=row.last_test_status,
        version=row.version,
    )


async def load_group_model_configs(
    db_session: AsyncSession,
    cipher: LocalSecretCipher,
    global_config: ScopedConfig | None = None,
) -> int:
    result = await db_session.execute(Select(GroupModelConfig).where(GroupModelConfig.enabled.is_(True)))
    loaded: dict[str, ActiveGroupModelConfig] = {}
    for row in result.scalars().all():
        try:
            if global_config is not None:
                await validate_group_provider_resolution(
                    row.base_url,
                    global_config.group_api_allowed_provider_hosts,
                )
            loaded[row.group_id] = _active_from_row(row, cipher)
        except (GroupModelConfigError, LocalEncryptionKeyError, ValueError):
            logger.warning(f"群 {row.group_id} 的模型配置无法加载，本群将使用全局配置")
    _active_configs.clear()
    _active_configs.update(loaded)
    return len(loaded)


def set_active_group_model_config(config: ActiveGroupModelConfig) -> None:
    """Install one validated config in memory; useful for startup and tests."""
    _active_configs[config.group_id] = config


def clear_active_group_model_configs() -> None:
    _active_configs.clear()


def has_group_model_config(group_id: str) -> bool:
    return str(group_id) in _active_configs


def resolve_group_reply_probability(
    group_id: str | None,
    global_config: ScopedConfig | None = None,
) -> float:
    """Resolve the proactive reply probability for one group.

    A group override can only exist as part of an active group-owned model
    configuration. Missing/disabled group configs continue to use the Bot
    owner's global probability.
    """

    if global_config is None:
        from .runtime_config import get_runtime_config

        global_config = get_runtime_config()
    if group_id:
        group_config = _active_configs.get(str(group_id))
        if group_config is not None and group_config.reply_probability is not None:
            return group_config.reply_probability
    return float(global_config.reply_probability)


def resolve_chat_config(
    group_id: str | None,
    global_config: ScopedConfig | None = None,
) -> ScopedConfig:
    if global_config is None:
        from .runtime_config import get_runtime_config

        global_config = get_runtime_config()
    if not group_id:
        return global_config
    group_config = _active_configs.get(str(group_id))
    if group_config is None:
        return global_config

    return _resolve_active_chat_config(group_config, global_config)


def _resolve_active_chat_config(
    group_config: ActiveGroupModelConfig,
    global_config: ScopedConfig,
) -> ScopedConfig:

    resolved = global_config.model_copy(deep=True)
    resolved.chat_api_format = group_config.api_format
    resolved.chat_api_key = group_config.api_key
    resolved.chat_base_url = group_config.base_url
    resolved.chat_model = group_config.chat_model
    resolved.chat_multimodal = group_config.chat_multimodal
    if group_config.api_format == "vertex":
        # Group Vertex credentials support Express Mode only in phase one.
        resolved.vertex_api_key = group_config.api_key
        resolved.vertex_credentials_path = ""
        resolved.vertex_project = ""
    return resolved


def build_candidate_chat_config(
    payload: GroupModelPayload,
    global_config: ScopedConfig,
) -> ScopedConfig:
    validate_group_provider_policy(
        payload.base_url,
        global_config.group_api_allowed_provider_hosts,
    )
    return _resolve_active_chat_config(
        ActiveGroupModelConfig(
            group_id="candidate",
            api_format=payload.api_format,
            base_url=payload.base_url,
            api_key=payload.api_key,
            chat_model=payload.chat_model,
            chat_multimodal=payload.chat_multimodal,
            reply_probability=payload.reply_probability,
            allow_global_fallback=payload.allow_global_fallback,
        ),
        global_config,
    )


def validate_group_model_test_response(response: object) -> None:
    """Require a real, non-empty model response before accepting credentials."""
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str) and block.strip():
                return
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str) and text.strip():
                    return
    raise GroupModelConfigError("模型连接成功但返回了空响应")


async def save_group_model_config(
    db_session: AsyncSession,
    *,
    group_id: str,
    operator_id: str,
    payload: GroupModelPayload,
    cipher: LocalSecretCipher,
) -> ActiveGroupModelConfig:
    group_id = str(group_id)
    now = datetime.datetime.now()
    row = await db_session.get(GroupModelConfig, group_id)
    next_version = (row.version + 1) if row else 1
    ciphertext = cipher.encrypt(
        payload.api_key,
        purpose=_api_key_purpose(group_id),
    )
    if row is None:
        row = GroupModelConfig(
            group_id=group_id,
            enabled=True,
            api_format=payload.api_format,
            base_url=payload.base_url,
            api_key_ciphertext=ciphertext,
            chat_model=payload.chat_model,
            chat_multimodal=payload.chat_multimodal,
            reply_probability=payload.reply_probability,
            allow_global_fallback=payload.allow_global_fallback,
            updated_by=str(operator_id),
            updated_at=now,
            last_tested_at=now,
            last_test_status="success",
            version=next_version,
        )
        db_session.add(row)
    else:
        row.enabled = True
        row.api_format = payload.api_format
        row.base_url = payload.base_url
        row.api_key_ciphertext = ciphertext
        row.chat_model = payload.chat_model
        row.chat_multimodal = payload.chat_multimodal
        row.reply_probability = payload.reply_probability
        row.allow_global_fallback = payload.allow_global_fallback
        row.updated_by = str(operator_id)
        row.updated_at = now
        row.last_tested_at = now
        row.last_test_status = "success"
        row.version = next_version
    await db_session.commit()

    active = ActiveGroupModelConfig(
        group_id=group_id,
        api_format=payload.api_format,
        base_url=payload.base_url,
        api_key=payload.api_key,
        chat_model=payload.chat_model,
        chat_multimodal=payload.chat_multimodal,
        reply_probability=payload.reply_probability,
        allow_global_fallback=payload.allow_global_fallback,
        version=next_version,
    )
    _active_configs[group_id] = active
    return active


async def delete_group_model_config(
    db_session: AsyncSession,
    group_id: str,
) -> bool:
    group_id = str(group_id)
    row = await db_session.get(GroupModelConfig, group_id)
    if row is None:
        _active_configs.pop(group_id, None)
        return False
    await db_session.delete(row)
    await db_session.commit()
    _active_configs.pop(group_id, None)
    return True


async def list_group_model_configs(
    db_session: AsyncSession,
) -> list[GroupModelConfigDetail]:
    result = await db_session.execute(
        Select(GroupModelConfig).order_by(
            GroupModelConfig.updated_at.desc(),
            GroupModelConfig.group_id.asc(),
        )
    )
    return [_detail_from_row(row) for row in result.scalars().all()]


async def get_group_model_config_detail(
    db_session: AsyncSession,
    group_id: str,
) -> GroupModelConfigDetail | None:
    row = await db_session.get(GroupModelConfig, str(group_id))
    return _detail_from_row(row) if row is not None else None


async def get_decrypted_group_model_config(
    db_session: AsyncSession,
    group_id: str,
    cipher: LocalSecretCipher,
) -> ActiveGroupModelConfig | None:
    """Load one config for an authenticated local admin update."""

    row = await db_session.get(GroupModelConfig, str(group_id))
    return _active_from_row(row, cipher) if row is not None else None


async def get_group_model_config_summary(
    db_session: AsyncSession,
    group_id: str,
) -> GroupModelConfigSummary | None:
    row = await db_session.get(GroupModelConfig, str(group_id))
    if row is None:
        return None
    return GroupModelConfigSummary.model_validate(
        _detail_from_row(row).model_dump()
    )
