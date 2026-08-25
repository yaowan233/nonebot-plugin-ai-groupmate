import math
import asyncio
import datetime
from typing import cast
from functools import lru_cache

from nonebot import logger, get_driver, on_command
from nonebot.params import CommandArg
from nonebot.adapters import Bot, Event, Message
from nonebot_plugin_orm import async_scoped_session
from nonebot_plugin_uninfo import Uninfo, SceneType
from nonebot_plugin_alconna import Target, UniMessage
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage

from .config import create_chat_llm
from .runtime_config import get_runtime_config
from .group_api_relay import (
    RelayError,
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
    build_candidate_chat_config,
    get_group_model_config_summary,
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


def _relay_error_message(error: RelayError) -> str:
    if isinstance(error, RelayDisabledError):
        return f"群 API 配置功能尚未启用：{error}"
    if isinstance(error, RelayAuthenticationError):
        return "中转服务身份验证失败，请联系 Bot 管理员检查注册配置。"
    if isinstance(error, RelayConnectionError):
        return "暂时无法连接群 API 配置服务，请稍后重试。"
    if isinstance(error, RelayTicketError):
        return str(error)
    if isinstance(error, RelayPayloadError):
        return "配置内容无法验证，请重新生成配置链接。"
    return "群 API 配置服务返回异常，请稍后重试。"


def _build_ticket_private_message(
    ticket: ConfigTicket,
    *,
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
    message = (
        f"下面的群模型配置链接约 {remaining_minutes} 分钟内有效，请尽快打开：\n"
        f"{ticket.config_url}\n\n"
        "网页提交后会生成一次性配置码，请在当前私聊发送：\n"
        "/提交群API <配置码>\n"
        "配置码同样需要在有效期内提交，请勿直接发送 API Key。"
    )
    return message, remaining_minutes


def _clear_group_chat_model_cache(group_id: str) -> None:
    from .agent import clear_group_chat_model_cache

    clear_group_chat_model_cache(group_id)


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


configure_group_api = on_command(
    "配置群API",
    aliases={"群API配置"},
    priority=1,
    block=True,
)
submit_group_api = on_command(
    "提交群API",
    priority=1,
    block=True,
)
show_group_api = on_command(
    "查看群API",
    priority=1,
    block=True,
)
delete_group_api = on_command(
    "删除群API",
    priority=1,
    block=True,
)


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
            group_id=session.scene.id,
            operator_id=session.user.id,
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
    arg: Message = CommandArg(),
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.PRIVATE:
        await submit_group_api.finish("请私聊 Bot 提交配置码。")
    code = arg.extract_plain_text().strip()
    if not code:
        await submit_group_api.finish("请发送：/提交群API <配置码>")

    try:
        relay = _relay()
        redeemed = await relay.redeem(
            database,
            code=code,
            operator_id=session.user.id,
        )
    except (RelayError, LocalEncryptionKeyError) as error:
        if isinstance(error, RelayError):
            await submit_group_api.finish(_relay_error_message(error))
        await submit_group_api.finish(f"群 API 配置功能尚未启用：{error}")
        return

    try:
        await _test_candidate_connection(redeemed.payload)
    except Exception as error:
        logger.warning(
            f"群 API 模型连接测试失败: group={redeemed.group_id}, error_type={type(error).__name__}"
        )
        try:
            await relay.acknowledge(database, redeemed, outcome="rejected")
        except RelayError:
            logger.warning(
                f"确认删除失败的群 API 配置密文失败: ticket={redeemed.ticket_id[-6:]}"
            )
        await submit_group_api.finish(
            "模型连接测试失败，原有配置未变更。请检查 Base URL、API Key 和模型名称后重新配置。"
        )

    try:
        active = await save_group_model_config(
            database,
            group_id=redeemed.group_id,
            operator_id=redeemed.operator_id,
            payload=redeemed.payload,
            cipher=relay.cipher,
        )
        _clear_group_chat_model_cache(redeemed.group_id)
    except Exception as error:
        await database.rollback()
        logger.warning(
            f"保存群 API 配置失败: group={redeemed.group_id}, error_type={type(error).__name__}"
        )
        try:
            await relay.acknowledge(database, redeemed, outcome="rejected")
        except RelayError:
            logger.warning(
                f"确认删除未保存的群 API 配置密文失败: ticket={redeemed.ticket_id[-6:]}"
            )
        await submit_group_api.finish("配置未能保存，原有配置未变更，请稍后重试。")

    try:
        await relay.acknowledge(database, redeemed, outcome="applied")
    except RelayError:
        logger.warning(
            f"群 API 配置已应用但中转确认失败: ticket={redeemed.ticket_id[-6:]}"
        )
    await submit_group_api.finish(
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
    arg: Message = CommandArg(),
) -> None:
    database = cast(AsyncSession, db_session)
    if session.scene.type != SceneType.GROUP:
        await delete_group_api.finish("请在需要删除配置的群聊中使用此命令。")
    if not can_manage_group_config(session, event):
        await delete_group_api.finish(
            "只有群主、群管理员或 Bot 超级用户可以删除群 API 配置。"
        )
    if arg.extract_plain_text().strip() != "确认":
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
