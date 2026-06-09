import re
import json
import traceback
from pathlib import Path

from sqlalchemy import Select, desc
from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_alconna import Target, UniMessage

from ..model import ChatHistory, MediaStorage
from ..memory import DB
from ..reply_guard import is_request_active


def create_similar_meme_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str | None,
    *,
    pic_dir: Path,
):
    """
    创建基于消息ID搜索相似表情包的工具
    """

    @tool("search_similar_meme_by_id")
    async def search_similar_meme_by_pic(target_msg_id: str | None = None) -> str:
        """
        根据指定的历史图片，搜索与之相似的表情包。
        当用户说"找一张跟这张差不多的"或引用某张图片求相似图时使用。
        参数：
        - target_msg_id: 聊天记录中图片消息的 id（从聊天记录的 "id: xxxxx" 中获取）。
          如果不传，则自动使用**当前发消息的用户**最近发送的一张图片（而非群内最新图片）。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消搜索。"

        logger.info("正在搜索相似图片...")

        try:
            base_stmt = (
                Select(ChatHistory)
                .where(
                    ChatHistory.session_id == session_id,
                    ChatHistory.content_type == "image",
                )
                .order_by(desc(ChatHistory.created_at))
            )
            if target_msg_id:
                stmt = base_stmt.where(
                    ChatHistory.content.contains(f"id: {target_msg_id}\n")
                ).limit(1)
            elif user_id:
                stmt = base_stmt.where(ChatHistory.user_id == user_id).limit(1)
            else:
                stmt = base_stmt.limit(1)
            result = await db_session.execute(stmt)
            msg = result.scalar_one_or_none()

            if not msg:
                return "本群近期没有发送过图片，无法进行相似搜索。"

            stmt = Select(MediaStorage).where(MediaStorage.media_id == msg.media_id)
            media_obj = (await db_session.execute(stmt)).scalar_one_or_none()

            if not media_obj or not media_obj.file_path:
                return "无法找到原图文件，无法进行分析。"

            pic_ids = await DB.search_similar_meme(str(pic_dir / media_obj.file_path))

            if not pic_ids:
                logger.info(f"未找到相似图片, source_id: {msg.media_id}")
                return "没有搜索到相似图片"

            images_info = []
            stmt = Select(MediaStorage).where(MediaStorage.media_id.in_(pic_ids))
            rows = (await db_session.execute(stmt)).scalars().all()
            media_map = {m.media_id: m for m in rows}

            for pid in pic_ids:
                if pid in media_map:
                    media_obj = media_map[pid]
                    images_info.append(
                        {
                            "pic_id": str(pid),
                            "description": media_obj.description or "未知描述",
                        }
                    )

            return json.dumps(
                {
                    "success": True,
                    "source_media_id": msg.media_id,
                    "images": images_info,
                    "count": len(images_info),
                    "note": "请根据 pic_id 调用 send_meme_image 发送",
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"相似图片搜索失败: {e}")
            return f"搜索出错: {e}"

    return search_similar_meme_by_pic


def create_search_meme_tool(db_session, session_id: str, request_id: str | None):
    """
    创建一个带数据库会话的表情包搜索工具
    """

    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """
        根据描述搜索合适的表情包图片。

        这个工具只负责搜索，不会发送图片。搜索后会返回匹配的图片列表及其详细描述。
        你可以查看这些图片的描述，判断是否合适，然后使用 send_meme_image 工具发送。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消搜索。"

        try:
            pic_ids = await DB.search_meme(description)

            if not pic_ids:
                logger.info(f"未找到匹配的表情包: {description}")
                return json.dumps({"success": False, "images": []}, ensure_ascii=False)

            images_info = []
            for pic_id in pic_ids[:5]:
                pic = (
                    await db_session.execute(
                        Select(MediaStorage).where(MediaStorage.media_id == int(pic_id))
                    )
                ).scalar()

                if pic:
                    images_info.append(
                        {
                            "pic_id": pic_id,
                            "description": pic.description,
                        }
                    )

            if not images_info:
                return json.dumps(
                    {
                        "success": False,
                        "images": [],
                    },
                    ensure_ascii=False,
                )

            logger.info(f"找到 {len(images_info)} 张匹配的表情包: {description}")
            return json.dumps(
                {
                    "success": True,
                    "images": images_info,
                    "count": len(images_info),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"表情包搜索失败: {repr(e)}")
            logger.error(traceback.format_exc())
            return json.dumps(
                {"success": False, "images": [], "error": str(e) or "未知错误"},
                ensure_ascii=False,
            )

    return search_meme_image


def create_send_meme_tool(
    db_session,
    session_id: str,
    request_id: str | None = None,
    *,
    send_target: Target | None = None,
    pic_dir: Path,
    bot_name: str,
):
    """
    创建一个带上下文的表情包发送工具
    """

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """
        发送表情包图片到聊天中。

        你需要先使用 search_meme_image 搜索图片，然后决定是否发送。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消发送。"

        try:
            match = re.search(r"\d+", pic_id)
            if not match:
                return f"发送表情包失败: 无法从 pic_id 中提取有效数字: {pic_id!r}"
            selected_pic_id = int(match.group())
            logger.info(f"使用指定的图片ID: {selected_pic_id}")

            pic = (
                await db_session.execute(
                    Select(MediaStorage).where(MediaStorage.media_id == selected_pic_id)
                )
            ).scalar()

            if not pic:
                logger.warning(f"图片记录不存在: {selected_pic_id}")
                return "图片记录不存在"

            pic_path = pic_dir / pic.file_path

            if not pic_path.exists():
                logger.warning(f"图片文件不存在: {pic_path}")
                return "图片文件不存在"

            pic_data = pic_path.read_bytes()
            description = pic.description

            if request_id is not None and not await is_request_active(
                session_id, request_id
            ):
                return "请求已过期，已取消发送。"

            message = UniMessage.image(raw=pic_data)
            res = await (
                message.send(target=send_target)
                if send_target is not None
                else message.send()
            )
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=bot_name,
                content_type="bot",
                content=(
                    f"id: {res.msg_ids[-1]['message_id']}\n"
                    f"发送了图片，图片描述是: {description}\n"
                    f"图片文件: {pic.file_path}"
                ),
                user_name=bot_name,
                media_id=pic.media_id,
            )
            db_session.add(chat_history)
            logger.info(f"id:{res.msg_ids}\n发送表情包: {description}")
            return f"已成功发送表情包: {description}"

        except Exception as e:
            logger.error(f"发送表情包失败: {e}")
            return f"发送表情包失败: {str(e)}"

    return send_meme_image
