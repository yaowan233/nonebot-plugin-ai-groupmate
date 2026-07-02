import datetime

from sqlalchemy import Select
from nonebot.log import logger
from langchain.tools import tool
from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna import message_recall

from ..model import ChatHistory
from ..reply_guard import is_request_active

SELF_RECALL_WINDOW = datetime.timedelta(minutes=5)


def _extract_stored_message_id(content: str) -> str | None:
    first_line = content.splitlines()[0] if content else ""
    if not first_line.startswith("id:"):
        return None
    message_id = first_line.split(":", 1)[1].strip()
    if not message_id or message_id == "system":
        return None
    return message_id


def create_recall_message_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    *,
    bot_name: str,
    has_admin_permission: bool,
    bot: Bot | None,
    event: Event | None,
):
    async def _find_history_by_message_id(message_id: str) -> ChatHistory | None:
        rows = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(ChatHistory.session_id == session_id)
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(300)
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            if _extract_stored_message_id(row.content) == message_id:
                return row
        return None

    @tool("recall_message")
    async def recall_message(target_msg_id: str, reason: str | None = None) -> str:
        """
        撤回当前群历史中的一条消息。

        管理员/群主权限下可以撤回他人消息；普通权限下只能撤回 bot 自己发送且 5 分钟内的消息。

        Args:
            target_msg_id: 聊天历史里 `id: xxx` 的平台消息 ID。
            reason: 撤回原因，用于日志和历史记录。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消撤回。"

        if bot is None or event is None:
            return "撤回失败: 缺少 bot/event 上下文。"

        target_msg_id = str(target_msg_id or "").strip()
        if not target_msg_id or target_msg_id == "system":
            return "撤回失败: 缺少有效的目标消息 ID。"

        history = await _find_history_by_message_id(target_msg_id)
        if history is None:
            return "撤回失败: 没有在当前群历史中找到这条消息。"

        is_bot_message = (
            history.content_type == "bot" and str(history.user_id) == str(bot_name)
        )
        if not has_admin_permission:
            if not is_bot_message:
                return "撤回失败: bot 不是管理员，只能撤回自己发送的消息。"
            if datetime.datetime.now() - history.created_at > SELF_RECALL_WINDOW:
                return "撤回失败: bot 不是管理员，只能撤回自己 5 分钟内发送的消息。"

        try:
            await message_recall(message_id=target_msg_id, event=event, bot=bot)
        except ValueError as e:
            logger.warning(f"撤回消息失败，消息 ID 不被当前适配器支持: {target_msg_id} {e}")
            return "撤回失败: 当前适配器不支持这个消息 ID 格式。"
        except Exception as e:
            logger.error(f"撤回消息失败: {e}")
            return f"撤回失败: {e}"

        action_scope = "管理员撤回" if has_admin_permission else "撤回自己消息"
        chat_history = ChatHistory(
            session_id=session_id,
            user_id=bot_name,
            content_type="bot",
            content=(
                "id: system\n"
                f"已执行{action_scope}: message_id={target_msg_id}, "
                f"reason={reason or '未填写原因'}"
            ),
            user_name=bot_name,
        )
        db_session.add(chat_history)
        logger.info(
            f"已执行{action_scope}: message_id={target_msg_id}, reason={reason or '未填写原因'}"
        )
        return f"已撤回消息 {target_msg_id}。"

    return recall_message
