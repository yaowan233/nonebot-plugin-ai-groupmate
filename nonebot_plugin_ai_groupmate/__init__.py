import asyncio
import datetime
import random
import jieba
import aiofiles
from io import BytesIO
from pathlib import Path

from nonebot import on_message, require, Bot, logger, get_plugin_config, on_command
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.internal.adapter import Event, Message
from nonebot.internal.rule import Rule
from nonebot.params import CommandArg
from nonebot.typing import T_State
from wordcloud import WordCloud
from PIL import Image as PILImage
require("nonebot_plugin_alconna")
require("nonebot_plugin_orm")
from nonebot_plugin_alconna import Image, image_fetch, UniMessage
from nonebot_plugin_orm import async_scoped_session, get_session
from nonebot_plugin_alconna.uniseg import UniMsg
from nonebot_plugin_uninfo import Uninfo
from sqlalchemy import Select
from sqlalchemy.exc import IntegrityError

from .agent import choice_response_strategy
from .milvus import MilvusOP, milvus_async
from .model import ChatHistory, MediaStorage, ChatHistorySchema, MediaStorageSchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    bytes_to_base64,
    combine_messages_into_context,
    process_and_vectorize_session_chats,
)
from .vlm import image_vl
from .config import Config

require("nonebot_plugin_localstore")
require("nonebot_plugin_apscheduler")

from nonebot_plugin_apscheduler import scheduler
import nonebot_plugin_localstore as store

plugin_data_dir: Path = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
pic_dir.mkdir(parents=True, exist_ok=True)
plugin_config = get_plugin_config(Config)
with open(Path(__file__).parent / "stop_words.txt", "r", encoding="utf-8") as f:
    stop_words = f.read().splitlines()


async def check_group_permission(event: GroupMessageEvent):
    # 检查是否为群聊
    return True


record = on_message(
    priority=999,
    rule=Rule(check_group_permission),
    block=True,
)


@record.handle()
async def handle_message(
        db_session: async_scoped_session,
        msg: UniMsg,
        session: Uninfo,
        event: Event,
        bot: Bot,
        state: T_State,
):
    """处理消息的主函数"""
    bot_name = plugin_config.bot_name
    imgs = msg.include(Image)
    content = f"id: {msg.get_message_id(event)}\n"
    to_me = False
    is_text = False
    if event.is_tome():
        to_me = True
        content += f"@{plugin_config.bot_name} "
    for i in msg:
        if i.type == "at":
            qq = i.target
            res = await bot.get_group_member_info(group_id=session.scene.id, user_id=qq, no_cache=False)
            name = res["nickname"]
            content += "@" + name + " "
            is_text = True
        if i.type == "reply":
            content += "回复id:" + i.id
        if i.type == "text":
            content += i.text
            is_text = True


    # 构建用户名（包含昵称和职位）
    user_name = session.user.name
    if session.member.nick:
        user_name = f"({session.member.nick}){user_name}"
    if session.member.role.name == "owner":
        user_name = f"群主-{user_name}"
    elif session.member.role.name == "admin":
        user_name = f"管理员-{user_name}"

    # ========== 步骤1: 处理文本消息（快速） ==========
    if is_text:
        chat_history = ChatHistory(
            session_id=session.scene.id,
            user_id=session.user.id,
            content_type="text",
            content=content,
            user_name=user_name,
        )
        db_session.add(chat_history)

    # 立即提交文本消息
    try:
        await db_session.commit()
    except Exception as e:
        logger.error(f"保存文本消息失败: {e}")
        await db_session.rollback()

    # ========== 步骤2: 处理图片消息（耗时） ==========
    for img in imgs:
        await process_image_message(
            db_session, img, event, bot, state,
            session, user_name, f"id: {msg.get_message_id(event)}\n"
        )

    # ========== 步骤3: 决定是否回复 ==========
    if msg.extract_plain_text().strip().startswith(plugin_config.bot_name):
        to_me = True
    should_reply = to_me or (random.random() < plugin_config.reply_probability)
    if not event.get_plaintext() and not imgs:
        should_reply = False
    if event.get_plaintext().startswith(("!", "！", "/", "#", "?", "\\")):
        should_reply = False
    if not event.get_plaintext() and not event.is_tome():
        should_reply = False
    if should_reply:
        await handle_reply_logic(db_session, session, bot_name, event)

    await db_session.commit()


async def process_image_message(
        db_session,
        img: Image,
        event: Event,
        bot: Bot,
        state: T_State,
        session: Uninfo,
        user_name: str,
        content: str,
):
    """处理单张图片消息"""
    try:
        content_type = "image"
        image_format = img.id.split(".")[-1]

        # 获取和压缩图片
        pic = await image_fetch(event, bot, state, img)
        pic = await asyncio.to_thread(
            check_and_compress_image_bytes, pic, image_format=image_format.upper()
        )
        file_hash = generate_file_hash(pic)
        file_path = pic_dir / f"{file_hash}.{image_format}"

        # 保存文件
        if not file_path.exists():
            file_path.write_bytes(pic)

        # 查询或创建媒体记录
        existing_media = (
            await db_session.execute(
                Select(MediaStorage).where(MediaStorage.file_hash == file_hash)
            )
        ).scalar()

        image_description = None

        if existing_media:
            # 已存在，直接使用描述
            image_description = existing_media.description
            existing_media.references += 1
            db_session.add(existing_media)
        else:
            # 新图片，调用VLM获取描述
            image_description = await image_vl(file_path)

            if image_description:
                media_storage = MediaStorage(
                    file_hash=file_hash,
                    file_path=f"{file_hash}.{image_format}",
                    references=1,
                    description=image_description,
                )
                db_session.add(media_storage)
                await db_session.flush()  # 确保获取media_id
                existing_media = media_storage

        # 添加聊天历史记录
        if existing_media and image_description:
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=content + image_description,
                user_name=user_name,
                media_id=existing_media.media_id,
            )
            db_session.add(chat_history)

        await db_session.commit()

    except IntegrityError:
        # 处理并发插入冲突
        await db_session.rollback()
        existing_media = (
            await db_session.execute(
                Select(MediaStorage).where(MediaStorage.file_hash == file_hash)
            )
        ).scalar()

        if existing_media:
            existing_media.references += 1
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=existing_media.description,
                user_name=user_name,
                media_id=existing_media.media_id,
            )
            db_session.add(chat_history)
        await db_session.commit()

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        await db_session.rollback()


async def handle_reply_logic(
        db_session,
        session: Uninfo,
        bot_name: str,
        event: Event,
):
    """处理回复逻辑"""
    try:
        # 如果是@机器人，稍微延迟一下显得更自然
        if event.is_tome():
            await asyncio.sleep(random.uniform(1, 3))

        # 获取最近1小时内的消息历史
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        last_msg = (
            await db_session.execute(
                Select(ChatHistory)
                .where(ChatHistory.session_id == session.scene.id)
                .where(ChatHistory.created_at >= cutoff_time)
                .order_by(ChatHistory.msg_id.desc())
                .limit(20)
            )
        ).scalars().all()

        if not last_msg:
            logger.info("没有历史消息，跳过回复")
            return

        # 转换为模型对象并反转顺序（从旧到新）
        last_msg = [ChatHistorySchema.model_validate(m) for m in last_msg]
        last_msg = last_msg[::-1]

        # 构建上下文用于搜索相似消息
        # context, _ = combine_messages_into_context(last_msg)
        # search_context, _ = combine_messages_into_context(last_msg[-5:])
        #
        # # 搜索相似的历史对话
        # _, similar_msgs = await milvus_async.search([search_context])
        # contexts = [similar_msgs] if similar_msgs else []

        # 使用Agent决定回复策略
        logger.info("开始调用Agent决策...")
        strategy = await choice_response_strategy([], last_msg, "")

        logger.info(f"Agent决策结果: {strategy}")

        # 检查是否需要回复
        if not strategy.text and not strategy.image_desc:
            logger.info("Agent决定不回复")
            return

        # 处理文本回复
        if strategy.text:
            text = strategy.text
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=bot_name,
                content_type="bot",
                content=text,
                user_name=bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()

            await record.send(text)
            logger.info(f"发送文本回复: {text}")

        # 处理图片回复（表情包）
        if strategy.image_desc:
            await send_meme_image(
                db_session, strategy.image_desc,
                session.scene.id, bot_name
            )
    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
        await db_session.rollback()


async def send_meme_image(
        db_session,
        image_desc: str,
        session_id: str,
        bot_name: str,
):
    """发送表情包图片"""
    try:
        # 从向量数据库搜索匹配的表情包
        pic_ids = await milvus_async.search_media([image_desc])

        if not pic_ids:
            logger.warning(f"未找到匹配的表情包: {image_desc}")
            return

        # 随机选择一个
        pic_id = random.choice(pic_ids)

        # 从数据库获取图片信息
        pic = (
            await db_session.execute(
                Select(MediaStorage).where(MediaStorage.media_id == pic_id)
            )
        ).scalar()

        if not pic:
            logger.warning(f"图片记录不存在: {pic_id}")
            return

        pic_path = pic_dir / pic.file_path

        if not pic_path.exists():
            logger.warning(f"图片文件不存在: {pic_path}")
            return

        # 读取图片并发送
        pic_data = pic_path.read_bytes()

        # 记录发送历史
        chat_history = ChatHistory(
            session_id=session_id,
            user_id=bot_name,
            content_type="bot",
            content=f"发送了图片，图片描述是: {pic.description}",
            user_name=bot_name,
        )
        db_session.add(chat_history)
        logger.info(f"发送表情包: {pic.description}")
        await db_session.commit()

        await UniMessage.image(raw=pic_data).send()

    except Exception as e:
        logger.error(f"发送表情包失败: {e}")
        await db_session.rollback()



def _build_wordcloud_image(words: str) -> BytesIO:
    """Generate a PNG image bytes object from words using WordCloud."""
    wc = WordCloud(font_path=Path(__file__).parent / "SourceHanSans.otf", width=1000, height=500).generate(words).to_image()
    image_bytes = BytesIO()
    wc.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


async def _collect_words_from_db(db_session, session_id: str, days: int = 1, user_id: str | None = None) -> str:
    """Query chat history and return a cleaned space-joined word string for wordcloud."""
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    where = [ChatHistory.session_id == session_id, ChatHistory.content_type == "text", ChatHistory.created_at >= cutoff]
    if user_id:
        where.append(ChatHistory.user_id == user_id)

    res = await db_session.execute(Select(ChatHistory.content).where(*where))
    ans = res.scalars().all()
    # tokenize and join
    ans = [" ".join([j.strip() for j in jieba.lcut(i)]) for i in ans]
    words = " ".join(ans)
    for sw in stop_words:
        words = words.replace(sw, "")
    return words


frequency = on_command("词频")


@frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    if not arg_text:
        arg_text = "1"
    if not arg_text.isdigit():
        await frequency.finish("统计范围应为纯数字")
    days = int(arg_text)

    words = await _collect_words_from_db(db_session, session_id, days=days, user_id=session.user.id)
    if not words:
        await frequency.finish("在指定时间内，没有说过话呢")

    image_bytes = _build_wordcloud_image(words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


group_frequency = on_command("群词频")


@group_frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    if not arg_text:
        arg_text = "1"
    if not arg_text.isdigit():
        await group_frequency.finish("统计范围应为纯数字")
    days = int(arg_text)

    words = await _collect_words_from_db(db_session, session_id, days=days, user_id=None)
    # Even if no words, return an empty wordcloud (original group_frequency didn't check emptiness)
    if not words:
        await group_frequency.finish("在指定时间内，没有消息可统计")

    image_bytes = _build_wordcloud_image(words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


@scheduler.scheduled_job("interval", minutes=20)
async def vectorize_message_history():
    async with get_session() as db_session:
        session_ids = await db_session.execute(Select(ChatHistory.session_id.distinct()))
        session_ids = session_ids.scalars().all()
        logger.info("开始向量化会话")
        for session_id in session_ids:
            try:
                res = await process_and_vectorize_session_chats(db_session, session_id)
                if res:
                    logger.info(f"向量化会话 {res['session_id']} 成功，共处理 {res['processed_groups']}/{res['total_groups']} 组")
                else:
                    logger.info(f"{session_id} 无需向量化")
            except Exception as e:
                logger.error(f"向量化会话 {session_id} 失败: {e}")
                continue


@scheduler.scheduled_job("interval", minutes=30)
async def vectorize_media():
    async with get_session() as db_session:
        medias_res = await db_session.execute(
            Select(MediaStorage).where(MediaStorage.references >= 3, MediaStorage.vectorized == False)
        )
        medias = medias_res.scalars().all()
        logger.info(f"待向量化媒体数量: {len(medias)}")

        for media in medias:
            try:
                file_path = pic_dir / media.file_path
                if not file_path.exists():
                    logger.warning(f"文件不存在: {file_path}")
                    media.vectorized = True
                    db_session.add(media)
                    continue

                # 判断是否适合作为表情包
                vlm_res = await image_vl(file_path, "请判断这张图适不适合作为表情包，只回答是或否")
                if not vlm_res or vlm_res != "是":
                    media.vectorized = True
                    db_session.add(media)
                    continue

                # 插入向量数据库 (MilvusOP is synchronous in original use)
                try:
                    MilvusOP.insert_media(media.media_id, [PILImage.open(file_path)])
                    media.vectorized = True
                    db_session.add(media)
                    logger.info("向量化成功")
                except Exception as e:
                    logger.error(f"向量化插入失败 {media.media_id}: {e}")
                    # don't mark as vectorized so it can retry later
                    continue

            except Exception as e:
                logger.error(f"处理媒体 {getattr(media, 'media_id', 'unknown')} 失败: {e}")
                continue

        await db_session.commit()
        logger.info("向量化媒体完成")


@scheduler.scheduled_job("interval", minutes=35)
async def clear_cache_pic():
    async with get_session() as db_session:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=30)
        result = await db_session.execute(
            Select(MediaStorage).where(MediaStorage.references < 3, datetime.datetime.now() - MediaStorage.created_at > datetime.timedelta(days=30))
        )
        medias = result.scalars().all()

        if not medias:
            logger.info("没有需要清理的媒体文件")
            return

        records_to_delete = []
        for media in medias:
            try:
                file_path = pic_dir / media.file_path
                # use pathlib unlink with missing_ok=True to avoid raising if missing
                await asyncio.to_thread(Path.unlink, file_path, True)
                records_to_delete.append(media)
                logger.debug(f"删除文件: {file_path}")
            except Exception as e:
                logger.error(f"删除文件失败 {getattr(media, 'file_path', 'unknown')}: {e}")
                records_to_delete.append(media)

        for media in records_to_delete:
            try:
                await db_session.delete(media)
            except Exception as e:
                logger.error(f"删除数据库记录失败 {getattr(media, 'media_id', 'unknown')}: {e}")

        await db_session.commit()
        logger.info(f"成功清理 {len(records_to_delete)} 个媒体记录")
