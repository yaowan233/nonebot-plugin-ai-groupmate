from typing import Any

from sqlalchemy import Select
from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_alconna import Target, UniMessage

from ..model import ChatHistory
from ..reply_guard import is_request_active


def create_private_message_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    interface: QryItrface | None,
    *,
    bot_id: str | None,
    bot_name: str,
    group_members: list[Any] | None = None,
):
    async def _resolve_group_member(
        target_user_id: str | None, target_name: str | None
    ) -> tuple[str | None, str | None]:
        if interface is None:
            return None, "缺少群成员接口，无法确认私聊目标。"

        target_user_id = str(target_user_id or "").strip()
        target_name = str(target_name or "").strip()
        if not target_user_id and not target_name:
            return None, "缺少私聊目标，请提供 target_user_id 或 target_name。"

        members = group_members
        if members is None:
            try:
                members = await interface.get_members(SceneType.GROUP, session_id)
            except Exception as e:
                logger.warning(f"获取群成员失败，无法主动私聊: {e}")
                return None, "获取群成员失败，无法确认私聊目标。"

        name_to_id: dict[str, str] = {}
        member_ids: set[str] = set()
        for member in members:
            member_id = str(member.id)
            member_ids.add(member_id)
            aliases = {
                getattr(member, "name", None),
                getattr(member, "nick", None),
                getattr(getattr(member, "user", None), "name", None),
                getattr(getattr(member, "user", None), "nick", None),
            }
            for alias in aliases:
                if alias:
                    name_to_id[str(alias).strip()] = member_id

        if target_user_id:
            if target_user_id not in member_ids:
                return None, "目标用户不在当前群内，已拒绝主动私聊。"
            if bot_id is not None and target_user_id == str(bot_id):
                return None, "不能给自己发送私聊。"
            return target_user_id, None

        resolved_id = name_to_id.get(target_name)
        if not resolved_id:
            return None, f"找不到群成员 {target_name!r}。"
        if bot_id is not None and resolved_id == str(bot_id):
            return None, "不能给自己发送私聊。"
        return resolved_id, None

    @tool("send_private_message")
    async def send_private_message(
        content: str,
        target_user_id: str | None = None,
        target_name: str | None = None,
        reason: str | None = None,
    ) -> str:
        """
        主动给当前群内某个成员发送私聊消息。

        只在确实不适合群内公开回复时使用，例如提醒隐私信息、避免当众尴尬、继续一段只和对方相关的话题。
        不要群发，不要骚扰，不要绕过用户明确拒绝。

        Args:
            content: 要私聊发送的文本。
            target_user_id: 目标用户 ID，必须是当前群成员。
            target_name: 目标用户昵称/名称；不知道 ID 时使用。
            reason: 简短说明为什么需要私聊，用于日志和历史记录。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消主动私聊。"

        content = str(content or "").strip()
        if not content:
            return "主动私聊内容为空，未发送。"

        target_id, error = await _resolve_group_member(target_user_id, target_name)
        if error or not target_id:
            return f"主动私聊失败: {error}"

        latest_private_msg = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(
                        ChatHistory.session_id == target_id,
                        ChatHistory.content_type == "bot",
                    )
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if latest_private_msg and latest_private_msg.content.endswith(content):
            return "检测到重复私聊内容，已跳过发送。"

        try:
            target = Target(id=target_id, private=True, self_id=bot_id)
            result = await UniMessage.text(content).send(target=target)
            msg_id = "unknown"
            if result.msg_ids:
                raw_msg_id = result.msg_ids[-1].get("message_id") or result.msg_ids[
                    -1
                ].get("msg_id")
                if raw_msg_id is not None:
                    msg_id = str(raw_msg_id)

            private_history = ChatHistory(
                session_id=target_id,
                user_id=bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n{content}",
                user_name=bot_name,
            )
            group_history = ChatHistory(
                session_id=session_id,
                user_id=bot_name,
                content_type="bot",
                content=(
                    "id: system\n"
                    f"已主动私聊用户 {target_id}: {reason or '未填写原因'}"
                ),
                user_name=bot_name,
            )
            db_session.add(private_history)
            db_session.add(group_history)
            logger.info(
                f"已主动私聊用户 {target_id}，reason={reason or '未填写原因'}"
            )
            return f"已主动私聊用户 {target_id}。"
        except Exception as e:
            logger.error(f"主动私聊发送失败: {e}")
            await db_session.rollback()
            return f"主动私聊发送失败: {e}"

    return send_private_message
