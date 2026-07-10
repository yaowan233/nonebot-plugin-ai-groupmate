import traceback
from typing import Any

from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_uninfo import SceneType, QryItrface

from ..model import ChatHistory
from ..reply_guard import is_request_active


def create_mute_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    interface: QryItrface | None,
    bot_id: str | None,
    *,
    bot_name: str,
    group_members: list[Any] | None = None,
):
    """
    创建禁言工具（仅在bot是管理员时可用）
    """

    @tool("mute_user")
    async def mute_user(target_user_name: str, duration_seconds: int, reason: str) -> str:
        """
        禁言指定用户。仅在bot是管理员或群主时可用。

        参数:
        - target_user_name: 要禁言的用户昵称（从聊天记录中获取）
        - duration_seconds: 禁言时长（秒），最多2592000秒(30天)，0表示解除禁言
        - reason: 禁言原因（必填，用于记录和说明）

        返回: 操作结果描述
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消操作。"

        if not interface:
            return "无法获取群成员信息接口，禁言失败。"

        if not bot_id:
            return "无法获取bot ID，禁言失败。"

        try:
            if duration_seconds < 0 or duration_seconds > 2592000:
                return "禁言时长必须在0-2592000秒(30天)之间。"

            members = group_members
            if members is None:
                members = await interface.get_members(SceneType.GROUP, session_id)

            bot_member = None
            target_member = None

            for member in members:
                if str(member.id) == str(bot_id):
                    bot_member = member

                member_name = member.nick or (member.user.name if member.user else None)
                if member_name == target_user_name:
                    target_member = member

            if not bot_member:
                return "无法获取bot的群成员信息。"

            bot_role = getattr(getattr(bot_member, "role", None), "name", None)
            if bot_role not in {"owner", "admin"}:
                return "bot不是管理员或群主，无法执行禁言操作。"

            if not target_member:
                return f"未找到用户 '{target_user_name}'，请确认昵称是否正确。"

            target_role = getattr(getattr(target_member, "role", None), "name", None)
            if target_role in {"owner", "admin"}:
                return f"无法禁言管理员或群主 '{target_user_name}'。"

            try:
                from nonebot import get_bot

                bot = get_bot()
                target_user_id = str(target_member.id)
                if hasattr(bot, "set_group_ban"):
                    await bot.set_group_ban(
                        group_id=int(session_id),
                        user_id=int(target_user_id),
                        duration=duration_seconds,
                    )
                else:
                    return "当前适配器不支持禁言功能。"

                action = "解除禁言" if duration_seconds == 0 else f"禁言 {duration_seconds} 秒"
                logger.info(
                    f"已{action}用户 {target_user_name}（{target_user_id}），原因: {reason}"
                )

                chat_history = ChatHistory(
                    session_id=session_id,
                    user_id=bot_name,
                    content_type="bot",
                    content=f"id: system\n已执行禁言操作: {action}用户 '{target_user_name}'。原因: {reason}",
                    user_name=bot_name,
                )
                db_session.add(chat_history)

                return f"已成功{action}用户 '{target_user_name}'。原因: {reason}"

            except Exception as api_err:
                logger.error(f"调用禁言API失败: {api_err}")
                return f"禁言操作失败: {str(api_err)}"

        except Exception as e:
            logger.error(f"禁言工具执行失败: {e}")
            print(traceback.format_exc())
            return f"禁言失败: {str(e)}"

    return mute_user
