from typing import Any, Literal, cast
from collections.abc import Callable, Awaitable

from nonebot import get_bot
from nonebot.log import logger
from langchain.tools import tool
from nonebot.adapters import Bot
from nonebot_plugin_uninfo import SceneType, QryItrface

from ..model import ChatHistory
from ..reply_guard import is_request_active
from .tool_results import tool_result

MAX_MUTE_DURATION_SECONDS = 2_592_000
MAX_MUTE_REASON_CHARS = 300


def _result(
    status: Literal["succeeded", "skipped", "failed"],
    reason_code: str,
    message: str,
    *,
    retryable: bool = False,
    **extra: Any,
) -> str:
    delivery_state = extra.pop(
        "delivery_state",
        "completed" if status == "succeeded" else "not_attempted",
    )
    return tool_result(
        status,
        reason_code,
        message,
        retryable=retryable,
        data=extra or None,
        delivery_state=delivery_state,
    )


def _member_aliases(member: Any) -> set[str]:
    aliases = {
        getattr(member, "name", None),
        getattr(member, "nick", None),
        getattr(getattr(member, "user", None), "name", None),
        getattr(getattr(member, "user", None), "nick", None),
    }
    return {
        str(alias).strip()
        for alias in aliases
        if alias is not None and str(alias).strip()
    }


def _member_display_name(member: Any) -> str:
    aliases = _member_aliases(member)
    if aliases:
        return sorted(aliases, key=lambda value: (len(value), value))[0]
    return f"用户{getattr(member, 'id', 'unknown')}"


def _member_role_name(member: Any) -> str:
    role = getattr(member, "role", None)
    for candidate in (
        getattr(role, "name", None),
        getattr(role, "value", None),
        role,
    ):
        if candidate is not None:
            return str(candidate).strip().lower()
    return ""


def _find_members_by_name(members: list[Any], target_name: str) -> list[Any]:
    normalized_name = target_name.strip().casefold()
    matched_by_id: dict[str, Any] = {}
    for member in members:
        if any(alias.casefold() == normalized_name for alias in _member_aliases(member)):
            matched_by_id[str(member.id)] = member
    return list(matched_by_id.values())


def create_mute_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    interface: QryItrface | None,
    bot_id: str | None,
    *,
    bot_name: str,
    bot: Bot | None = None,
    group_members: list[Any] | None = None,
):
    """创建禁言工具（仅在 bot 是管理员时可用）。"""

    @tool("mute_user")
    async def mute_user(
        duration_seconds: int,
        reason: str,
        target_user_id: str | None = None,
        target_user_name: str | None = None,
    ) -> str:
        """
        禁言指定用户。任何群成员都可以请求此操作，但 bot 必须是管理员或群主。

        Args:
            duration_seconds: 禁言时长（秒），最多 2592000 秒（30 天），0 表示解除禁言。
            reason: 禁言原因，用于日志和历史记录。
            target_user_id: 目标用户 ID；已知 ID 时优先使用，能够避免重名。
            target_user_name: 目标用户昵称；不知道 ID 时使用。重名时工具会返回候选 ID。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return _result(
                "skipped",
                "request_expired",
                "请求已过期，已取消禁言操作。",
            )

        if not bot_id:
            return _result(
                "failed",
                "missing_bot_id",
                "无法确认当前 bot，禁言失败。",
            )
        if group_members is None and interface is None:
            return _result(
                "failed",
                "missing_member_context",
                "无法获取群成员信息，禁言失败。",
            )
        if duration_seconds < 0 or duration_seconds > MAX_MUTE_DURATION_SECONDS:
            return _result(
                "failed",
                "invalid_duration",
                "禁言时长必须在 0～2592000 秒（30 天）之间。",
            )

        normalized_reason = " ".join(str(reason or "").split())
        if not normalized_reason or len(normalized_reason) > MAX_MUTE_REASON_CHARS:
            return _result(
                "failed",
                "invalid_reason",
                f"禁言原因必须为 1～{MAX_MUTE_REASON_CHARS} 个字符。",
            )

        normalized_target_id = str(target_user_id or "").strip()
        normalized_target_name = str(target_user_name or "").strip()
        if not normalized_target_id and not normalized_target_name:
            return _result(
                "failed",
                "missing_target",
                "请提供 target_user_id 或 target_user_name。",
                retryable=True,
            )

        try:
            if group_members is not None:
                members = list(group_members)
            else:
                assert interface is not None
                members = list(
                    await interface.get_members(SceneType.GROUP, session_id)
                )
        except Exception as error:
            logger.warning(
                "获取群成员失败，无法执行禁言: "
                f"error_type={type(error).__name__}"
            )
            return _result(
                "failed",
                "member_lookup_failed",
                "获取群成员失败，暂时无法执行禁言。",
                retryable=True,
            )

        bot_member = next(
            (member for member in members if str(member.id) == str(bot_id)),
            None,
        )
        if bot_member is None:
            return _result(
                "failed",
                "bot_member_not_found",
                "无法在当前群成员中确认 bot 身份。",
            )
        if _member_role_name(bot_member) not in {"owner", "admin"}:
            return _result(
                "failed",
                "bot_permission_denied",
                "bot 不是管理员或群主，无法执行禁言操作。",
            )

        target_member: Any | None = None
        if normalized_target_id:
            target_member = next(
                (
                    member
                    for member in members
                    if str(member.id) == normalized_target_id
                ),
                None,
            )
            if target_member is None:
                return _result(
                    "failed",
                    "target_not_found",
                    f"当前群内没有 ID 为 {normalized_target_id!r} 的成员。",
                    retryable=True,
                )
        else:
            matched_members = _find_members_by_name(members, normalized_target_name)
            if not matched_members:
                return _result(
                    "failed",
                    "target_not_found",
                    f"未找到群成员 {normalized_target_name!r}。",
                    retryable=True,
                )
            if len(matched_members) > 1:
                candidates = [
                    {
                        "user_id": str(member.id),
                        "display_name": _member_display_name(member),
                    }
                    for member in matched_members[:10]
                ]
                return _result(
                    "failed",
                    "target_ambiguous",
                    "存在多个同名群成员，请根据候选 user_id 重新调用。",
                    retryable=True,
                    candidates=candidates,
                )
            target_member = matched_members[0]

        assert target_member is not None
        resolved_target_id = str(target_member.id)
        resolved_target_name = _member_display_name(target_member)
        if resolved_target_id == str(bot_id):
            return _result(
                "failed",
                "cannot_mute_self",
                "bot 不能禁言自己。",
            )
        if _member_role_name(target_member) in {"owner", "admin"}:
            return _result(
                "failed",
                "protected_target",
                f"无法禁言管理员或群主 {resolved_target_name!r}。",
            )

        current_bot = bot
        if current_bot is None:
            try:
                current_bot = get_bot(str(bot_id))
            except Exception as error:
                logger.warning(
                    "获取当前 bot 实例失败: "
                    f"bot_id={bot_id}, error_type={type(error).__name__}"
                )
                return _result(
                    "failed",
                    "bot_unavailable",
                    "当前 bot 实例不可用，无法执行禁言。",
                )

        set_group_ban = getattr(current_bot, "set_group_ban", None)
        if not callable(set_group_ban):
            return _result(
                "failed",
                "adapter_unsupported",
                "当前适配器不支持禁言功能。",
            )

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return _result(
                "skipped",
                "request_expired",
                "请求已过期，已取消禁言操作。",
            )

        try:
            ban_call = cast(Callable[..., Awaitable[Any]], set_group_ban)
            await ban_call(
                group_id=int(session_id),
                user_id=int(resolved_target_id),
                duration=duration_seconds,
            )
        except Exception as error:
            logger.warning(
                "调用禁言 API 失败: "
                f"bot_id={bot_id}, target_id={resolved_target_id}, "
                f"error_type={type(error).__name__}"
            )
            return _result(
                "failed",
                "provider_error",
                "禁言接口调用失败；为避免重复操作，请不要立即重试。",
                delivery_state="unknown",
            )

        action = (
            "解除禁言"
            if duration_seconds == 0
            else f"禁言 {duration_seconds} 秒"
        )
        logger.info(
            f"已{action}用户 {resolved_target_name}（{resolved_target_id}），"
            f"原因: {normalized_reason}"
        )
        db_session.add(ChatHistory(
            session_id=session_id,
            user_id=bot_name,
            content_type="bot",
            content=(
                "id: system\n"
                f"已执行禁言操作: {action}用户 {resolved_target_name!r}"
                f"（{resolved_target_id}）。原因: {normalized_reason}"
            ),
            user_name=bot_name,
        ))
        return _result(
            "succeeded",
            "mute_applied",
            f"已成功{action}用户 {resolved_target_name!r}。",
            target_user_id=resolved_target_id,
            target_user_name=resolved_target_name,
            duration_seconds=duration_seconds,
        )

    return mute_user
