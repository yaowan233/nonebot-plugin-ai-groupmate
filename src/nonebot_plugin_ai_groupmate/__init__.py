import random
import asyncio
import datetime
import traceback
from io import BytesIO
from pathlib import Path

import jieba
from PIL import Image as PILImage
from nonebot import logger, require, on_command, on_message, get_plugin_config
from wordcloud import WordCloud
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event, Message

require("nonebot_plugin_alconna")
require("nonebot_plugin_orm")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_localstore")
require("nonebot_plugin_apscheduler")
import nonebot_plugin_localstore as store
from sqlalchemy import Select
from sqlalchemy.exc import IntegrityError
from nonebot_plugin_orm import get_session, async_scoped_session
from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface
from nonebot_plugin_alconna import Image, UniMessage, image_fetch, get_message_id
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import UniMsg

from .vlm import image_vl
from .agent import choice_response_strategy
from .model import ChatHistory, MediaStorage, ChatHistorySchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .config import Config
from .milvus import MilvusOP

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-ai-groupmate",
    description="AI虚拟群友",
    usage="@bot 让bot进行回复\n/词频 <统计天数>\n/群词频<统计天数>",
    type="application",
    homepage="https://github.com/yaowan233/nonebot-plugin-ai-groupmate",
    config=Config,
    supported_adapters=inherit_supported_adapters("nonebot_plugin_alconna", "nonebot_plugin_uninfo"),
    extra={"author": "yaowan233 <572473053@qq.com>"},
)
plugin_data_dir: Path = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
pic_dir.mkdir(parents=True, exist_ok=True)
plugin_config = get_plugin_config(Config).ai_groupmate
with open(Path(__file__).parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]


record = on_message(
    priority=999,
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
        interface: QryItrface
):
    """处理消息的主函数"""
    bot_name = plugin_config.bot_name
    imgs = msg.include(Image)
    content = f"id: {get_message_id()}\n"
    to_me = False
    is_text = False
    if event.is_tome():
        to_me = True
        content += f"@{plugin_config.bot_name} "
    for i in msg:
        if i.type == "at":
            members = await interface.get_members(SceneType.GROUP, session.scene.id)
            for member in members:
                if member.id == i.target:
                    name = member.user.name
                    break
            else:
                continue
            content += "@" + name + " "
            is_text = True
        if i.type == "reply":
            content += "回复id:" + i.id
        if i.type == "text":
            content += i.text
            is_text = True


    # 构建用户名（包含昵称和职位）
    user_name = session.user.name
    if session.member:
        if session.member.nick:
            user_name = f"({session.member.nick}){user_name}"
        if session.member.role:
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
            session, user_name, f"id: {get_message_id()}\n"
        )

    # ========== 步骤3: 决定是否回复 ==========
    if msg.extract_plain_text().strip().lower().startswith(plugin_config.bot_name):
        to_me = True
    should_reply = to_me or (random.random() < plugin_config.reply_probability)
    if not event.get_plaintext() and not imgs:
        should_reply = False
    if event.get_plaintext().startswith(("!", "！", "/", "#", "?", "\\")):
        should_reply = False
    if not event.get_plaintext() and not event.is_tome():
        should_reply = False
    if to_me:
        user_id = session.user.id
        user_name = session.user.name or session.user.nick
    else:
        user_id = ""
        user_name = ""
    if should_reply:
        await handle_reply_logic(db_session, session, bot_name, user_id, user_name)

    await db_session.commit()


async def process_image_message(
        db_session,
        img: Image,
        event: Event,
        bot: Bot,
        state: T_State,
        session: Uninfo,
        user_name: str | None,
        content: str,
):
    """处理单张图片消息"""
    content_type = "image"
    if not img.id:
        return
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
    try:
        # 查询或创建媒体记录
        existing_media = (
            await db_session.execute(
                Select(MediaStorage).where(MediaStorage.file_hash == file_hash)
            )
        ).scalar()

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
            else:
                file_path.unlink()

        # 添加聊天历史记录
        if existing_media and image_description:
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=content + image_description,
                user_name=user_name or "",
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
                content=content + existing_media.description,
                user_name=user_name or "",
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
        user_id: str,
        user_name: str | None,
):
    """处理回复逻辑"""
    try:

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

        # 使用Agent决定回复策略
        logger.info("开始调用Agent决策...")
        strategy = await choice_response_strategy(db_session, session.scene.id, last_msg, user_id, user_name, "")

        logger.info(f"Agent决策结果: {strategy}")

        # 检查是否需要回复
        if not strategy.text:
            logger.info("Agent决定不回复")
            return

        # 处理文本回复
        if strategy.text:
            text = strategy.text
            res = await record.send(text)
            logger.info(f"发送文本回复: {text}")
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=bot_name,
                content_type="bot",
                content= f"id:{res['message_id']}\n" +text,
                user_name=bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()
    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
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


@scheduler.scheduled_job("interval", minutes=60)
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
                print(traceback.format_exc())
                logger.error(f"向量化会话 {session_id} 失败: {e}")
                continue


@scheduler.scheduled_job("interval", minutes=30)
async def vectorize_media():
    async with get_session() as db_session:
        medias_res = await db_session.execute(
            Select(MediaStorage).where(MediaStorage.references >= 3, MediaStorage.vectorized.is_(False))
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

                try:
                    await MilvusOP.insert_media(media.media_id, [PILImage.open(file_path)])
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
                file_path = Path(pic_dir / media.file_path)
                # use pathlib unlink with missing_ok=True to avoid raising if missing
                await asyncio.to_thread(file_path.unlink, True)
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
