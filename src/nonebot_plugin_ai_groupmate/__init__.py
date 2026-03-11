import base64
import json
import random
import asyncio
import datetime
import traceback
from io import BytesIO
from pathlib import Path

import jieba
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from nonebot import logger, require, on_command, on_message, get_plugin_config
from pydantic import SecretStr
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

from .agent import choice_response_strategy, check_if_should_reply
from .model import ChatHistory, MediaStorage, ChatHistorySchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .config import Config
from .memory import DB


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
                    name = member.user.name if member.user.name else ""
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
        await handle_reply_logic(db_session, session, bot_name, user_id, user_name, to_me)

    await db_session.commit()


async def process_image_message(
        db_session,
        img: Image,
        event: Event,
        bot: Bot,
        state: T_State,
        session: Uninfo,
        user_name: str | None,
        content_prefix: str,
):
    """处理单张图片消息 (修复并发插入报错)"""
    try:
        content_type = "image"
        if not img.id:
            return
        # 简单判断后缀，默认为 jpg
        image_format = img.id.split(".")[-1] if "." in img.id else "jpg"

        # 1. 获取和压缩图片
        try:
            pic = await asyncio.wait_for(
                image_fetch(event, bot, state, img),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning("下载图片超时，跳过")
            return

        pic = await asyncio.to_thread(
            check_and_compress_image_bytes, pic, image_format=image_format.upper()
        )
        file_hash = generate_file_hash(pic)
        file_name = f"{file_hash}.{image_format}"
        file_path = pic_dir / file_name

        # 2. 保存文件到本地
        if not file_path.exists():
            file_path.write_bytes(pic)

        # 3. 数据库操作 (MediaStorage)

        # 第一步：先查一次
        stmt = Select(MediaStorage).where(MediaStorage.file_hash == file_hash)
        media_obj = (await db_session.execute(stmt)).scalar_one_or_none()

        if media_obj:
            # A. 如果已存在，引用计数+1
            media_obj.references += 1
            db_session.add(media_obj)
        else:
            # B. 如果不存在，尝试插入
            try:
                # 使用嵌套事务 (Savepoint)，防止插入失败导致整个 Session 报废
                async with db_session.begin_nested():
                    new_media = MediaStorage(
                        file_hash=file_hash,
                        file_path=file_name,
                        references=1,
                        description="[图片]",  # 占位符
                    )
                    db_session.add(new_media)
                    # 必须 flush 以触发可能的 UniqueViolation 错误
                    await db_session.flush()
                    media_obj = new_media

            except IntegrityError:
                # C. 如果捕获到“唯一约束冲突”，说明刚才那瞬间有人插进去了
                # 不需要手动 rollback，begin_nested 会自动回滚这个子事务
                logger.info(f"图片并发插入冲突 {file_hash}，转为更新模式")

                # 重新查询 (这时候一定有了)
                media_obj = (await db_session.execute(stmt)).scalar_one()
                media_obj.references += 1
                db_session.add(media_obj)

        # 4. 添加聊天历史 (ChatHistory)
        # 此时 media_obj 一定是有效的 (无论是新插的还是查出来的)
        if media_obj:
            # 确保 flush 拿到 media_id (如果是新插入的对象)
            await db_session.flush()

            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=f"{content_prefix}{file_name}",
                user_name=user_name,
                media_id=media_obj.media_id,
            )
            db_session.add(chat_history)

        # 5. 最终提交
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
        is_tome: bool,
):
    """处理回复逻辑"""
    try:
        # 获取最近几条用于 Flash 快速判断
        # 注意：Flash 模型是纯文本模型，它看不懂图片，所以这里我们只喂文本内容
        recent_msgs = (
            await db_session.execute(
                Select(ChatHistory)
                .where(ChatHistory.session_id == session.scene.id)
                .order_by(ChatHistory.msg_id.desc())
                .limit(3)
            )
        ).scalars().all()
        recent_msgs = recent_msgs[::-1]

        if not recent_msgs:
            return

        # 简单的文本摘要用于 Gatekeeper
        history_summary = ""
        for m in recent_msgs:
            if m.content_type == "image":
                history_summary += f"{m.user_name}: [发送了一张图片]\n"
            else:
                history_summary += f"{m.user_name}: {m.content}\n"

        current_msg_text = recent_msgs[-1].content if recent_msgs[-1].content_type == "text" else "[图片]"

        # === Gatekeeper 判断 ===
        if not is_tome:
            should_reply = await check_if_should_reply(history_summary, current_msg_text, bot_name)
            if not should_reply:
                return

        # === 获取详细历史给 Agent ===
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

        last_msg = [ChatHistorySchema.model_validate(m) for m in last_msg]
        last_msg = last_msg[::-1]

        logger.info("开始调用Agent决策...")
        try:
            strategy = await asyncio.wait_for(
                choice_response_strategy(db_session, session.scene.id, last_msg, user_id, user_name, ""),
                timeout=240.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent 思考超时 - session: {session.scene.id}")
            return

        logger.info(f"Agent决策结果: {strategy}")

    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
        print(traceback.format_exc())
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


@scheduler.scheduled_job("interval", minutes=60, max_instances=1, coalesce=True, id="vectorize_chat")
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


tagging_model = ChatOpenAI(
    model="qwen-vl-max",
    api_key=SecretStr(plugin_config.qwen_token),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.01,
)


@scheduler.scheduled_job("interval", minutes=30, max_instances=1, coalesce=True, id="vectorize_media")
async def vectorize_media():
    """
    定期处理图片：
    1. 筛选高频图片
    2. 使用 qwen-vl-max 判断是否为表情包 + 生成描述
    3. 写入 SQL (描述) 和 Qdrant (向量)
    """
    async with get_session() as db_session:
        # 只处理引用次数 >= 3 且未向量化的图片
        medias_res = await db_session.execute(
            Select(MediaStorage).where(MediaStorage.references >= 3, MediaStorage.vectorized == False)
        )
        medias = medias_res.scalars().all()
        logger.info(f"待处理高频图片数量: {len(medias)}")

        for media in medias:
            try:
                file_path = pic_dir / media.file_path
                if not file_path.exists():
                    logger.warning(f"文件不存在: {file_path}")
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    continue

                # 1. 读取文件并转 Base64 (Qwen VL 需要)
                try:
                    with open(file_path, "rb") as image_file:
                        file_data = image_file.read()
                        encoded_string = base64.b64encode(file_data).decode('utf-8')

                        # 构造 Data URI
                        ext = media.file_path.split('.')[-1].lower()
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        if ext == "gif": mime = "image/gif"  # Qwen-VL 支持 GIF

                        img_data_uri = f"data:{mime};base64,{encoded_string}"
                except Exception as e:
                    logger.error(f"读取图片失败: {e}")
                    continue

                # 2. 调用 qwen-vl-max 进行【鉴别】和【描述】
                prompt = """
你是一个专业的表情包分析员。请分析这张图片：

任务 A：判断这是否是一张“表情包”(Meme)。
- 是：带文字的梗图、熊猫头、二次元表情、明显的搞笑图片。
- 否：普通的聊天截图、风景照、自拍、证件照、长篇文字截图。

任务 B：如果是表情包，请提取画面中的【所有文字内容】，并结合画面描述其表达的【情绪或含义】。
描述要简练，方便用户搜索。例如：“熊猫头流泪，配文‘我太难了’，表达悲伤和无奈”。

请务必只返回合法的 JSON 格式，不要使用 Markdown 代码块：
{
    "is_meme": true,
    "description": "熊猫头流泪，配文'我太难了'"
}
"""
                try:
                    # 调用模型
                    response = await tagging_model.ainvoke([
                        HumanMessage(content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_data_uri}}
                        ])
                    ])

                    if isinstance(response.content, list):
                        # 如果模型返回了一个列表，跳过
                        continue
                    # 解析 JSON
                    else:
                        content = response.content.strip()
                    if content.startswith("```"):
                        content = content.replace("```json", "").replace("```", "")

                    res_json = json.loads(content)
                    is_meme = res_json.get("is_meme", False)
                    description = res_json.get("description", "")

                except Exception as e:
                    logger.error(f"模型识别图片失败 {media.media_id}: {e}")
                    continue

                # 3. 结果处理
                if not is_meme:
                    logger.info(f"图片 {media.media_id} 被判定为非表情包(杂图)，跳过入库")
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    continue

                # 4. 是表情包 -> 入库
                try:
                    # A. 存描述到 SQL
                    media.description = description

                    # B. 存向量到 Qdrant
                    await DB.insert_media(media.media_id, encoded_string, description)

                    # C. 标记完成
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    logger.info(f"表情包入库成功 {media.media_id}: {description}")

                except Exception as e:
                    logger.error(f"向量化插入失败 {media.media_id}: {e}")
                    await db_session.rollback()
                    continue

            except Exception as e:
                logger.error(f"处理媒体循环异常 {getattr(media, 'media_id', 'unknown')}: {e}")
                await db_session.rollback()
                continue

        await db_session.commit()
        if len(medias) > 0:
            logger.info("本轮图片处理完成")


@scheduler.scheduled_job("interval", minutes=35, max_instances=1, coalesce=True, id="clear_cache")
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
