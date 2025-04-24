import asyncio
import datetime
import json
import random
import jieba
from io import BytesIO
from pathlib import Path

from nonebot import on_message, require, Bot, logger, get_plugin_config, on_command
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.internal.adapter import Event, Message
from nonebot.params import CommandArg
from nonebot.typing import T_State
from wordcloud import WordCloud

require("nonebot_plugin_apscheduler")
require("nonebot_plugin_localstore")
require("nonebot_plugin_alconna")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_orm")

from nonebot_plugin_alconna import Image, Text, image_fetch, UniMessage
from nonebot_plugin_orm import async_scoped_session, get_scoped_session
from nonebot_plugin_alconna.uniseg import UniMsg
from nonebot_plugin_apscheduler import scheduler
import nonebot_plugin_localstore as store
from nonebot_plugin_uninfo import Uninfo

from sqlalchemy import Select
from sqlalchemy.exc import IntegrityError

from .llm import choice_response_strategy
from .milvus import MilvusOP
from .model import ChatHistory, MediaStorage
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    bytes_to_base64,
    combine_messages_into_context,
    process_and_vectorize_session_chats,
)
from .vlm import image_vl
from .config import Config


__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-ai-groupmate",
    description="描述",
    usage="用法",
    type="application",
    homepage="https://github.com/yaowan233/nonebot-plugin-ai-groupmate",
    config=Config,
    supported_adapters=inherit_supported_adapters("nonebot_plugin_alconna", "nonebot_plugin_uninfo"),
    # supported_adapters={"~onebot.v11"}, # 仅 onebot 应取消注释
    extra={"author": "yaowan233 <572473053@qq.com>"},
)
plugin_data_dir: Path = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
pic_dir.mkdir(parents=True, exist_ok=True)
plugin_config = get_plugin_config(Config)
stop_words = ["的", "了", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们", "边", "种", "只", "能", "用",
              "在", "有", "没有", "不是", "还是", "怎么", "这", "那", "这个", "那个", "这些", "那些", "下", "只",
              "就", "和", "与", "或", "以及", "及", "等", "等等", "感觉", "样", "之", "之一", "想", "啊", "人",
              "一", "二", "三", "四", "现", "会", "么", "什么", "点", "还", "没", "个", "不", "是", "为",
              "一些", "一种", "一会儿", "一样", "一起", "一直", "一般", "一部分", "一方面", "一点", "一次", "一定",
              "很", "非常", "特别", "更", "最", "太", "比较", "吗", "越", "好", "什", "喜欢", "然后", "应该", "知道",
              "可以", "能够", "可能", "也许", "也", "并且", "而且", "同时", "此外", "另外", "然而", "但是", "但",
              "因为", "所以", "由于", "即使", "尽管", "虽然", "不过", "只是", "而是", "因此", "所以", "行", "便",
              "如何", "怎样", "怎么样", "如", "例如", "比如", "像", "像是", "真", "们", "要", "呢", "吧", "都", ]


record = on_message(
    priority=999,
    block=True,
)


@record.handle()
async def _(
    db_session: async_scoped_session,
    msg: UniMsg,
    session: Uninfo,
    event: Event,
    bot: Bot,
    state: T_State,
):
    bot_name = plugin_config.bot_name
    texts = msg.include(Text)
    imgs = msg.include(Image)
    user_name = session.user.name
    if session.member.nick:
        user_name += f"({session.member.nick})"
    if session.member.role.name == "owner":
        user_name += "[群主]"
    if session.member.role.name == "admin":
        user_name += "[管理员]"

    for i in texts:
        if event.is_tome():
            i.text = f"@{bot_name} " + i.text
        if not i.text:
            continue
        content_type = "text"
        content = i.text
        chat_history = ChatHistory(
            session_id=session.scene.id,
            user_id=session.user.id,
            content_type=content_type,
            content=content,
            user_name=user_name,
        )
        db_session.add(chat_history)

    for i in imgs:
        content_type = "image"
        image_format = i.id.split(".")[-1]
        pic = await image_fetch(event, bot, state, i)
        pic = await asyncio.to_thread(
            check_and_compress_image_bytes, pic, image_format=image_format.upper()
        )
        file_hash = generate_file_hash(pic)
        file_path = pic_dir / f"{file_hash}.{image_format}"
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(pic)
        try:
            # 2. 尝试查询现有记录
            media_storage = (
                await db_session.execute(
                    Select(MediaStorage)
                    .where(MediaStorage.file_hash == file_hash)
                    .with_for_update()
                )
            ).scalar()

            if media_storage:
                # 存在则引用计数+1
                media_storage.references += 1
                image_description = media_storage.description
            else:
                if image_description := await image_vl(bytes_to_base64(pic)):
                    media_storage = MediaStorage(
                        file_hash=file_hash,
                        file_path=str(file_path),
                        references=1,
                        description=image_description,
                    )
                    # 必须先刷新或提交才能获取自增ID
                    db_session.add(media_storage)
                    await db_session.flush()  # 关键步骤：立即生成ID不提交整体事务
        except IntegrityError:
            # 并发情况下可能出现的哈希冲突回滚
            await db_session.rollback()
            media_storage = (
                await db_session.execute(
                    Select(MediaStorage)
                    .where(MediaStorage.file_hash == file_hash)
                    .with_for_update()
                )
            ).scalar()
            image_description = media_storage.description
        if media_storage:
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=image_description,
                user_name=user_name,
                media_id=media_storage.media_id,
            )
            db_session.add(chat_history)
    if event.is_tome() or (random.random() < plugin_config.reply_probability):
        # 构造消息
        last_msg = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(ChatHistory.session_id == session.scene.id)
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(20)
                )
            )
            .scalars()
            .all()
        )
        last_msg = last_msg[::-1]
        context, _ = combine_messages_into_context(last_msg)
        search_context, _ = combine_messages_into_context(last_msg[-5:])
        # if random.random() > 0.5:
        #     similar_msgs = MilvusOP.search([context], session.scene.id)[0]
        # else:
        similar_msgs = MilvusOP.search([search_context])[0]
        texts = MilvusOP.query_ids([i["id"] for i in similar_msgs])
        context1, context2 = texts[0]["text"], texts[1]["text"]
        contexts = [context1, context2]
        strategy = await choice_response_strategy(contexts, last_msg, "")
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            strategy = {
                "need_reply": False,
                "reply_type": "none",
                "text": "",
                "image_desc": "",
                "image_emotion": "",
            }
        logger.info(strategy)
        if not strategy.get("need_reply"):
            await db_session.commit()
            return
        if strategy.get("text"):
            text = strategy.get("text")
            # 插入数据库
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=bot_name,
                content_type="bot",
                content=text,
                user_name=bot_name,
            )
            db_session.add(chat_history)
            await record.send(text)
            logger.info(f"大模型回复: {text}")
        if strategy.get("image_desc") and strategy.get("image_emotion"):
            if strategy.get("image_emotion") not in [
                "搞笑",
                "讽刺",
                "愤怒",
                "无奈",
                "喜爱",
                "惊讶",
                "中立",
            ]:
                # 防止错误的情感
                strategy["image_emotion"] = "搞笑"
            similar_pics = MilvusOP.search(
                [strategy.get("image_desc")],
                f'emotion == "{strategy.get("image_emotion")}"',
                "media_collection",
            )[0]
            similar_pic = random.choice(similar_pics)
            print(similar_pic)
            pic_id = similar_pic["id"]
            pic = (
                await db_session.execute(
                    Select(MediaStorage).where(MediaStorage.media_id == pic_id)
                )
            ).scalar()
            if pic:
                pic_path = pic.file_path
                # 发送图片
                with open(pic_path, "rb") as f:
                    pic_data = f.read()
                chat_history = ChatHistory(
                    session_id=session.scene.id,
                    user_id=bot_name,
                    content_type="bot",
                    content=f"发送了图片，图片描述是：{pic.description}",
                    user_name=bot_name,
                )
                db_session.add(chat_history)
                logger.info(f"大模型回复图片: {pic.description}\n{pic_path}")
                await UniMessage.image(raw=pic_data).send()
    await db_session.commit()



frequency = on_command('词频')


@frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    session_id = session.scene.id
    arg = arg.extract_plain_text().strip()
    if not arg:
        arg = '1'
    if not arg.isdigit():
        await frequency.finish('统计范围应为纯数字')
    ans = (await (db_session.execute(Select(ChatHistory.content).where(ChatHistory.session_id == session_id,ChatHistory.content_type == "text", ChatHistory.user_id == session.user.id, ChatHistory.created_at >= (datetime.datetime.now() - datetime.timedelta(days=int(arg))))))).scalars()
    ans = [' '.join([j.strip() for j in jieba.lcut(i)]) for i in ans]
    words = ' '.join(ans)
    for i in stop_words:
        words = words.replace(i, '')
    if not words:
        await frequency.finish('在指定时间内，没有说过话呢')
    wc = WordCloud(font_path=Path(__file__).parent / 'SourceHanSans.otf', width=1000, height=500).generate(
        words).to_image()
    image_bytes = BytesIO()
    wc.save(image_bytes, format="PNG")
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


group_frequency = on_command('群词频')


@group_frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    session_id = session.scene.id
    arg = arg.extract_plain_text().strip()
    if not arg:
        arg = '1'
    if not arg.isdigit():
        await frequency.finish('统计范围应为纯数字')
    ans = (await (db_session.execute(Select(ChatHistory.content).where(ChatHistory.session_id == session_id,ChatHistory.content_type == "text", ChatHistory.created_at >= (datetime.datetime.now() - datetime.timedelta(days=int(arg))))))).scalars()
    ans = [' '.join([j.strip() for j in jieba.lcut(i)]) for i in ans]
    words = ' '.join(ans)
    for i in stop_words:
        words = words.replace(i, '')
    wc = WordCloud(font_path=Path(__file__).parent / 'SourceHanSans.otf', width=1000, height=500).generate(
        words).to_image()
    image_bytes = BytesIO()
    wc.save(image_bytes, format="PNG")
    await UniMessage.image(raw=image_bytes).send(reply_to=True)

@scheduler.scheduled_job("interval", hours=1)
async def vectorize_message_history():
    db_session = get_scoped_session()
    # 查询出不同的 session_id
    session_ids = await db_session.execute(Select(ChatHistory.session_id.distinct()))
    session_ids = session_ids.scalars().all()
    # 可以加上黑名单过滤
    logger.info("开始向量化会话")
    for session_id in session_ids:
        res = await process_and_vectorize_session_chats(db_session, session_id)
        if res:
            logger.info(
                f"向量化会话 {res['session_id']} 成功，共处理 {res['processed_groups']}/{res['total_groups']} 组"
            )
        else:
            logger.info(f"{session_id} 无需向量化")


@scheduler.scheduled_job("interval", minutes=35)
async def vectorize_media():
    db_session = get_scoped_session()
    # 查询出不同的 session_id
    try:
        medias = await db_session.execute(
            Select(MediaStorage).where(
                MediaStorage.references >= 3, MediaStorage.vectorized == False
            )
        )
    except Exception as e:
        await db_session.rollback()  # 出错时回滚
        print(f"Error occurred: {e}")
        return
    medias = medias.scalars().all()
    logger.info(f"待向量化媒体数量: {len(medias)}")
    # 可以加上黑名单过滤
    logger.info("开始向量化媒体")
    for media in medias:
        # 使用大模型判断是不是表情包
        try:
            with open(media.file_path, "rb") as f:
                pic = f.read()
        except Exception as e:
            logger.error(f"读取图片失败: {e}")
            media.file_path.unlink(missing_ok=True)
        b64_pic = bytes_to_base64(pic)
        vlm_res = await image_vl(
            b64_pic, "请判断这张图适不适合作为表情包，只回答是或否"
        )
        if vlm_res != "是":
            # 设置为已向量化
            media.vectorized = True
            continue
        vlm_res = await image_vl(
            b64_pic,
            "请判断这张图所表达的情感，请从以下情感中选择一个，只回答情感：搞笑,讽刺,愤怒,无奈,喜爱,惊讶,中立",
        )
        media_id, description = media.media_id, media.description
        MilvusOP.insert_media(media_id, description, vlm_res)
        media.vectorized = True
    await db_session.commit()
    logger.info("向量化媒体完成")
