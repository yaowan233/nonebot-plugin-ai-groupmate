import re
import math
import asyncio
import datetime
from typing import Literal, cast
from functools import lru_cache

from nonebot import logger, on_regex, get_driver
from nonebot.params import RegexMatched
from nonebot.matcher import Matcher
from nonebot.adapters import Bot, Event
from nonebot_plugin_orm import async_scoped_session
from nonebot_plugin_uninfo import Uninfo, SceneType
from nonebot_plugin_alconna import Target, UniMessage
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage

from .config import create_chat_llm
from .runtime_config import get_runtime_config
from .group_api_relay import (
    RelayError,
    ConfigTarget,
    ConfigTicket,
    GroupModelRelay,
    RelayTicketError,
    RelayPayloadError,
    RelayDisabledError,
    RelayConnectionError,
    RelayAuthenticationError,
)
from .group_model_config import (
    LocalEncryptionKeyError,
    save_group_model_config,
    delete_group_model_config,
    save_private_model_config,
    build_candidate_chat_config,
    delete_private_model_config,
    get_group_model_config_summary,
    get_private_model_config_summary,
    validate_group_model_test_response,
    validate_group_provider_resolution,
)


def _is_superuser(user_id: str) -> bool:
    return str(user_id) in {str(item) for item in get_driver().config.superusers}


def _member_role_name(session: Uninfo, event: Event) -> str:
    role = getattr(getattr(session.member, "role", None), "name", None)
    if role:
        return str(role).strip().lower()
    sender = getattr(event, "sender", None)
    return str(getattr(sender, "role", "") or "").strip().lower()


def can_manage_group_config(session: Uninfo, event: Event) -> bool:
    if _is_superuser(session.user.id):
        return True
    return _member_role_name(session, event) in {
        "owner",
        "admin",
        "administrator",
        "群主",
        "管理员",
        "群管理员",
    }


@lru_cache
def _relay() -> GroupModelRelay:
    return GroupModelRelay(get_runtime_config())


ConfigScope = Literal["group", "private"]


def _feature_name(scope: ConfigScope) -> str:
    return "个人 API" if scope == "private" else "群 API"


def _relay_error_message(error: RelayError, scope: ConfigScope = "group") -> str:
    feature_name = _feature_name(scope)
    if isinstance(error, RelayDisabledError):
        return f"{feature_name} 配置功能尚未启用：{error}"
    if isinstance(error, RelayAuthenticationError):
        return "中转服务身份验证失败，请联系 Bot 管理员检查注册配置。"
    if isinstance(error, RelayConnectionError):
        return f"暂时无法连接{feature_name} 配置服务，请稍后重试。"
    if isinstance(error, RelayTicketError):
        return str(error)
    if isinstance(error, RelayPayloadError):
        return "配置内容无法验证，请重新生成配置链接。"
    return f"{feature_name} 配置服务返回异常，请稍后重试。"


def _build_ticket_private_message(
    ticket: ConfigTicket,
    *,
    scope: ConfigScope = "group",
    now: datetime.datetime | None = None,
) -> tuple[str, int]:
    expires_at = ticket.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=datetime.timezone.utc)
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    remaining_seconds = (
        expires_at.astimezone(datetime.timezone.utc)
        - now.astimezone(datetime.timezone.utc)
    ).total_seconds()
    remaining_minutes = max(1, math.ceil(remaining_seconds / 60))
    scope_name = "私聊" if scope == "private" else "群聊"
    submit_command = "/提交个人API" if scope == "private" else "/提交群API"
    message = (
        f"下面的{scope_name}模型配置链接约 {remaining_minutes} 分钟内有效，请尽快打开：\n"
        f"{ticket.config_url}\n\n"
        "网页提交后会生成一次性配置码，请在当前私聊发送：\n"
        f"{submit_command} <配置码>\n"
        "配置码同样需要在有效期内提交，请勿直接发送 API Key。"
    )
    return message, remaining_minutes


def _clear_group_chat_model_cache(group_id: str) -> None:
    from .agent import clear_group_chat_model_cache

    clear_group_chat_model_cache(group_id)


def _clear_private_chat_model_cache(user_id: str) -> None:
    from .agent import clear_private_chat_model_cache

    clear_private_chat_model_cache(user_id)


async def _test_candidate_connection(payload) -> None:
    config = build_candidate_chat_config(payload, get_runtime_config())
    await validate_group_provider_resolution(
        payload.base_url,
        config.group_api_allowed_provider_hosts,
    )
    model = create_chat_llm(config)
    response = await asyncio.wait_for(
        model.ainvoke([HumanMessage(content="请只回复 OK")]),
        timeout=min(config.agent_llm_timeout_seconds, 20.0),
    )
    validate_group_model_test_response(response)


def _command_pattern(*names: str) -> str:
    """Mirror on_command prefix behavior; aliases fold into the alternation."""

    starts = sorted(
        {str(item) for item in get_driver().config.command_start if str(item)},
        key=len,
        reverse=True,
    )
    commands = "|".join(re.escape(name) for name in names)
    if starts:
        prefix = "|".join(re.escape(item) for item in starts)
        return rf"^(?:{prefix})(?:{commands})\s*(?P<arg>[\s\S]*)$"
    return rf"^(?:{commands})\s*(?P<arg>[\s\S]*)$"


def _api_command_matcher(*names: str) -> type[Matcher]:
    # on_command is case-sensitive and offers no ignore-case option, so the
    # API 命令统一用 on_regex + IGNORECASE 实现大小写不敏感匹配。
    return on_regex(
        _command_pattern(*names),
        flags=re.IGNORECASE,
        priority=1,
        block=True,
    )


configure_group_api = _api_command_matcher("配置群API", "群API配置")
submit_group_api = _api_command_matcher("提交群API")
show_group_api = _api_command_matcher("查看群API")
delete_group_api = _api_command_matcher("删除群API")
configure_private_api = _api_command_matcher("配置个人API", "个人API配置", "配置私聊API")
submit_private_api = _api_command_matcher("提交个人API", "提交API")
show_private_api = _api_command_matcher("查看个人API", "查看私聊API")
delete_private_api = _api_command_matcher("删除个人API", "删除私聊API")


@configure_group_api.handle()
async def _configure_group_api(
    db_session: async_scoped_session,
    session: Uninfo,
    event: Event,
    bot: Bot,
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.GROUP:
        await configure_group_api.finish("请在需要配置的群聊中使用此命令。")
    if not can_manage_group_config(session, event):
        await configure_group_api.finish(
            "只有群主、群管理员或 Bot 超级用户可以配置群 API。"
        )
    try:
        ticket = await _relay().create_ticket(
            database,
            target=ConfigTarget(
                scope="group",
                subject_id=str(session.scene.id),
                operator_id=str(session.user.id),
            ),
        )
    except (RelayError, LocalEncryptionKeyError) as error:
        if isinstance(error, RelayError):
            await configure_group_api.finish(_relay_error_message(error))
        await configure_group_api.finish(f"群 API 配置功能尚未启用：{error}")
        return

    message, validity_minutes = _build_ticket_private_message(ticket)
    try:
        await UniMessage.text(message).send(
            target=Target(
                id=session.user.id,
                private=True,
                self_id=session.self_id,
            )
        )
    except Exception as error:
        logger.warning(f"发送群 API 配置私聊失败: error_type={type(error).__name__}")
        await configure_group_api.finish(
            "配置单已创建，但无法向你发送私聊。请先添加 Bot 好友后重新执行命令。"
        )
    await configure_group_api.finish(
        f"配置链接已通过私聊发送，约 {validity_minutes} 分钟内有效，请勿在群内发送 API Key。"
    )


@submit_group_api.handle()
async def _submit_group_api(
    db_session: async_scoped_session,
    session: Uninfo,
    matched: re.Match[str] = RegexMatched(),
) -> None:
    await _do_submit(
        db_session,
        session,
        scope="group",
        code=matched.group("arg").strip(),
        matcher=submit_group_api,
    )


@submit_private_api.handle()
async def _submit_private_api(
    db_session: async_scoped_session,
    session: Uninfo,
    matched: re.Match[str] = RegexMatched(),
) -> None:
    await _do_submit(
        db_session,
        session,
        scope="private",
        code=matched.group("arg").strip(),
        matcher=submit_private_api,
    )


async def _do_submit(
    db_session: async_scoped_session,
    session: Uninfo,
    *,
    scope: ConfigScope,
    code: str,
    matcher: type[Matcher],
) -> None:
    database = cast(AsyncSession, db_session)
    submit_command = "/提交群API" if scope == "group" else "/提交个人API"
    if session.scene.type != SceneType.PRIVATE:
        await matcher.finish("请私聊 Bot 提交配置码。")
    if not code:
        await matcher.finish(f"请发送：{submit_command} <配置码>")

    try:
        relay = _relay()
        redeemed = await relay.redeem(
            database,
            code=code,
            operator_id=session.user.id,
        )
    except (RelayError, LocalEncryptionKeyError) as error:
        if isinstance(error, RelayError):
            await matcher.finish(_relay_error_message(error, scope))
        await matcher.finish(f"{_feature_name(scope)} 配置功能尚未启用：{error}")
        return

    try:
        await _test_candidate_connection(redeemed.payload)
    except Exception as error:
        logger.warning(
            f"模型 API 连接测试失败: scope={redeemed.target.scope}, subject={redeemed.target.subject_id}, error_type={type(error).__name__}"
        )
        try:
            await relay.acknowledge(database, redeemed, outcome="rejected")
        except RelayError:
            logger.warning(
                f"确认删除失败的群 API 配置密文失败: ticket={redeemed.ticket_id[-6:]}"
            )
        await matcher.finish(
            "模型连接测试失败，原有配置未变更。请检查 Base URL、API Key 和模型名称后重新配置。"
        )

    try:
        if redeemed.target.scope == "private":
            active = await save_private_model_config(
                database,
                user_id=redeemed.target.subject_id,
                payload=redeemed.payload,
                cipher=relay.cipher,
            )
            _clear_private_chat_model_cache(redeemed.target.subject_id)
        else:
            active = await save_group_model_config(
                database,
                group_id=redeemed.target.subject_id,
                operator_id=redeemed.target.operator_id,
                payload=redeemed.payload,
                cipher=relay.cipher,
            )
            _clear_group_chat_model_cache(redeemed.target.subject_id)
    except Exception as error:
        await database.rollback()
        logger.warning(
            f"保存模型 API 配置失败: scope={redeemed.target.scope}, subject={redeemed.target.subject_id}, error_type={type(error).__name__}"
        )
        try:
            await relay.acknowledge(database, redeemed, outcome="rejected")
        except RelayError:
            logger.warning(
                f"确认删除未保存的群 API 配置密文失败: ticket={redeemed.ticket_id[-6:]}"
            )
        await matcher.finish("配置未能保存，原有配置未变更，请稍后重试。")

    try:
        await relay.acknowledge(database, redeemed, outcome="applied")
    except RelayError:
        logger.warning(
            f"群 API 配置已应用但中转确认失败: ticket={redeemed.ticket_id[-6:]}"
        )
    if redeemed.target.scope == "private":
        await matcher.finish(
            f"模型连接测试通过。你的私聊主模型已更新为 {active.chat_model}，之后不再占用每日公共模型额度。"
        )
    await matcher.finish(
        f"模型连接测试通过。群 {active.group_id} 的主聊天模型已更新为 {active.chat_model}。\n"
        + (
            "主动发言概率继续跟随 Bot 全局配置。"
            if active.reply_probability is None
            else f"主动发言概率已设为 {active.reply_probability:.3g}（{active.reply_probability:.1%}）。"
        )
    )


@show_group_api.handle()
async def _show_group_api(
    db_session: async_scoped_session,
    session: Uninfo,
    event: Event,
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.GROUP:
        await show_group_api.finish("请在群聊中查看当前群 API 配置。")
    if not can_manage_group_config(session, event):
        await show_group_api.finish(
            "只有群主、群管理员或 Bot 超级用户可以查看群 API 配置。"
        )
    summary = await get_group_model_config_summary(database, session.scene.id)
    await database.commit()
    if summary is None:
        await show_group_api.finish("当前群未配置独立主模型，将使用 Bot 全局配置。")
    fallback = "允许" if summary.allow_global_fallback else "禁止"
    reply_probability = (
        "跟随 Bot 全局配置"
        if summary.reply_probability is None
        else f"{summary.reply_probability:.3g}（{summary.reply_probability:.1%}）"
    )
    await show_group_api.finish(
        f"当前群主模型配置：\n接口格式：{summary.api_format}\n服务地址：{summary.provider_host}\n模型：{summary.chat_model}\n图片输入：{'开启' if summary.chat_multimodal else '关闭'}\n主动发言概率：{reply_probability}\n全局回退：{fallback}\n配置版本：{summary.version}\nAPI Key 已隐藏。"
    )


@delete_group_api.handle()
async def _delete_group_api(
    db_session: async_scoped_session,
    session: Uninfo,
    event: Event,
    matched: re.Match[str] = RegexMatched(),
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.GROUP:
        await delete_group_api.finish("请在需要删除配置的群聊中使用此命令。")
    if not can_manage_group_config(session, event):
        await delete_group_api.finish(
            "只有群主、群管理员或 Bot 超级用户可以删除群 API 配置。"
        )
    if matched.group("arg").strip() != "确认":
        await delete_group_api.finish(
            "此操作将恢复全局主模型。确认删除请发送：/删除群API 确认"
        )
    deleted = await delete_group_model_config(database, session.scene.id)
    _clear_group_chat_model_cache(session.scene.id)
    await delete_group_api.finish(
        "已删除当前群的独立主模型配置，将使用 Bot 全局配置。"
        if deleted
        else "当前群没有独立主模型配置。"
    )


@configure_private_api.handle()
async def _configure_private_api(
    db_session: async_scoped_session,
    session: Uninfo,
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.PRIVATE:
        await configure_private_api.finish("请私聊 Bot 使用此命令。")
    try:
        ticket = await _relay().create_ticket(
            database,
            target=ConfigTarget(
                scope="private",
                subject_id=str(session.user.id),
                operator_id=str(session.user.id),
            ),
        )
    except (RelayError, LocalEncryptionKeyError) as error:
        if isinstance(error, RelayError):
            await configure_private_api.finish(_relay_error_message(error, "private"))
        await configure_private_api.finish(f"个人 API 配置功能尚未启用：{error}")
        return
    message, _ = _build_ticket_private_message(ticket, scope="private")
    await configure_private_api.finish(message)


@show_private_api.handle()
async def _show_private_api(
    db_session: async_scoped_session,
    session: Uninfo,
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.PRIVATE:
        await show_private_api.finish("请私聊 Bot 查看个人 API 配置。")
    summary = await get_private_model_config_summary(database, session.user.id)
    await database.commit()
    if summary is None:
        await show_private_api.finish("你尚未配置个人主模型，将使用 Bot 公共模型和每日额度。")
    await show_private_api.finish(
        f"你的私聊主模型配置：\n接口格式：{summary.api_format}\n服务地址：{summary.provider_host}\n模型：{summary.chat_model}\n图片输入：{'开启' if summary.chat_multimodal else '关闭'}\n配置版本：{summary.version}\nAPI Key 已隐藏。"
    )


@delete_private_api.handle()
async def _delete_private_api(
    db_session: async_scoped_session,
    session: Uninfo,
    matched: re.Match[str] = RegexMatched(),
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.PRIVATE:
        await delete_private_api.finish("请私聊 Bot 删除个人 API 配置。")
    if matched.group("arg").strip() != "确认":
        await delete_private_api.finish(
            "删除后将恢复 Bot 公共模型和每日额度。确认删除请发送：/删除个人API 确认"
        )
    deleted = await delete_private_model_config(database, session.user.id)
    _clear_private_chat_model_cache(session.user.id)
    await delete_private_api.finish(
        "已删除你的个人主模型配置，将恢复 Bot 公共模型和每日额度。"
        if deleted
        else "你尚未配置个人主模型。"
    )
