import re
import json
import base64
import random
import asyncio
import datetime
import traceback
from io import BytesIO
from typing import Any
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import jieba
from nonebot import logger, require, on_command, on_message, get_plugin_config
from wordcloud import WordCloud
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event, Message
from langchain_core.messages import HumanMessage

require("nonebot_plugin_alconna")
require("nonebot_plugin_orm")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_localstore")
require("nonebot_plugin_apscheduler")
import nonebot_plugin_localstore as store
from sqlalchemy import Select
from nonebot_plugin_orm import get_session, async_scoped_session
from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface
from nonebot_plugin_alconna import Image, UniMessage, image_fetch, get_message_id
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import UniMsg

from .agent import check_if_should_reply, choice_response_strategy
from .model import ChatHistory, GroupMemory, MediaStorage, ChatHistorySchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .webui import register_usage_webui
from .config import Config, create_chat_openai, create_tagging_llm
from .memory import DB
from .reply_guard import set_latest_request_id


async def _safe_rollback(db_session) -> None:
    try:
        await db_session.rollback()
    except Exception:
        logger.exception("数据库回滚失败")


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
register_usage_webui(plugin_config)
MAX_WORDCLOUD_DAYS = 3650
with open(Path(__file__).parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]

@lru_cache
def get_summary_model():
    return create_chat_openai(plugin_config, "summary")


@lru_cache
def get_tagging_model():
    return create_tagging_llm(plugin_config)


@dataclass
class ReplyRequest:
    request_id: str
    session: Uninfo
    interface: QryItrface
    bot: Bot
    event: Event
    bot_name: str
    user_id: str
    user_name: str | None
    is_tome: bool
    is_continuous: bool
    reply_to_id: str | None


@dataclass
class GroupReplyState:
    running: bool = False
    latest: ReplyRequest | None = None
    task: asyncio.Task | None = None


# 每个群只保留"最新一条"待处理回复请求，避免高峰期堆积后刷屏。
_group_reply_states: dict[str, GroupReplyState] = {}
_group_reply_state_lock = asyncio.Lock()
_continuous_conversation_until: dict[tuple[str, str], datetime.datetime] = {}

# 多bot去重锁: 每个群串行化消息记录,防止并发SELECT查不到对方未提交数据
_dedup_locks: dict[str, asyncio.Lock] = {}

AGENT_HISTORY_LIMIT = 20
AGENT_RECENT_HISTORY_HOURS = 1
AGENT_EXTENDED_HISTORY_HOURS = 6
AGENT_MIN_RECENT_HISTORY = 6


def _continuous_conversation_ttl() -> datetime.timedelta:
    minutes = max(float(plugin_config.continuous_conversation_minutes or 0), 0)
    return datetime.timedelta(minutes=minutes)


def _is_continuous_conversation(session_id: str, user_id: str) -> bool:
    expires_at = _continuous_conversation_until.get((session_id, user_id))
    if not expires_at:
        return False
    if datetime.datetime.now() > expires_at:
        _continuous_conversation_until.pop((session_id, user_id), None)
        return False
    return True


def _refresh_continuous_conversation(session_id: str, user_id: str) -> None:
    ttl = _continuous_conversation_ttl()
    if ttl <= datetime.timedelta(0):
        return
    _continuous_conversation_until[(session_id, user_id)] = (
        datetime.datetime.now() + ttl
    )


def _get_dedup_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _dedup_locks:
        _dedup_locks[session_id] = asyncio.Lock()
    return _dedup_locks[session_id]


async def _load_agent_history(db_session, session_id: str) -> list[ChatHistorySchema]:
    now = datetime.datetime.now()

    async def query_since(hours: int) -> list[ChatHistory]:
        cutoff_time = now - datetime.timedelta(hours=hours)
        rows = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(ChatHistory.session_id == session_id)
                    .where(ChatHistory.created_at >= cutoff_time)
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(AGENT_HISTORY_LIMIT)
                )
            )
            .scalars()
            .all()
        )
        return list(rows)

    rows = await query_since(AGENT_RECENT_HISTORY_HOURS)
    if len(rows) < AGENT_MIN_RECENT_HISTORY:
        extended_rows = await query_since(AGENT_EXTENDED_HISTORY_HOURS)
        if len(extended_rows) > len(rows):
            rows = extended_rows

    return [ChatHistorySchema.model_validate(m) for m in reversed(rows)]


def _extract_reply_message_id_from_event(event: Event) -> str | None:
    event_reply = getattr(event, "reply", None)
    if event_reply is not None:
        if isinstance(event_reply, dict):
            reply_id = event_reply.get("message_id") or event_reply.get("id")
            if reply_id is not None and str(reply_id).strip():
                return str(reply_id).strip()
        else:
            for attr in ("message_id", "id"):
                reply_id = getattr(event_reply, attr, None)
                if reply_id is not None and str(reply_id).strip():
                    return str(reply_id).strip()

    try:
        message_text = str(event.get_message())
    except Exception:
        message_text = str(event)

    patterns = (
        r"\[reply:id=([^\],]+)[^\]]*\]",
        r"\[CQ:reply,(?:[^\]]*,)?id=([^,\]]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, message_text)
        if match and match.group(1).strip():
            return match.group(1).strip()

    return None


def _start_group_reply_worker_locked(group_id: str, state: GroupReplyState):
    """在已持有状态锁时启动群回复 worker。"""
    state.running = True
    state.task = asyncio.create_task(_run_group_reply_worker(group_id))


async def _run_group_reply_worker(group_id: str):
    """按群串行处理回复，只消费最新请求。"""
    try:
        while True:
            async with _group_reply_state_lock:
                state = _group_reply_states.get(group_id)
                if not state:
                    return
                request = state.latest
                state.latest = None

            if request is None:
                break

            async with get_session() as reply_session:
                await handle_reply_logic(
                    reply_session,
                    request.request_id,
                    request.session,
                    request.interface,
                    request.bot,
                    request.event,
                    request.bot_name,
                    request.user_id,
                    request.user_name,
                    request.is_tome,
                    request.is_continuous,
                    request.reply_to_id,
                )
    finally:
        async with _group_reply_state_lock:
            state = _group_reply_states.get(group_id)
            if state:
                state.running = False
                state.task = None
                if state.latest is not None:
                    _start_group_reply_worker_locked(group_id, state)


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
    interface: QryItrface,
):
    """处理消息的主函数"""
    if str(session.user.id) == str(bot.self_id):
        logger.debug(f"忽略机器人自身消息 - session: {session.scene.id}")
        return

    bot_name = plugin_config.bot_name
    imgs = msg.include(Image)
    # 第1行固定是本条消息的平台 ID 元数据，格式 "id: {id}"
    content = f"id: {get_message_id()}\n"
    to_me = False
    is_text = False
    reply_id: str | None = None  # 记录回复 ID，稍后单独成行插入
    body = ""  # 正文部分单独拼接
    has_at_mention = False
    if event.is_tome():
        to_me = True
    for i in msg:
        if i.type == "at":
            has_at_mention = True
            name = ""
            if session.scene.type == SceneType.GROUP:
                try:
                    members = await interface.get_members(SceneType.GROUP, session.scene.id)
                    for member in members:
                        if member.id == i.target:
                            name = member.user.name if member.user.name else ""
                            break
                except Exception:
                    pass
            if not name:
                name = i.target or ""
            body += "@" + name + " "
            is_text = True
        if i.type == "reply":
            reply_id = i.id
        if i.type == "text":
            body += i.text
            is_text = True
        if i.type == "image":
            body += "[图片]"
        if i.type == "mface":
            body += "[表情]"

    if to_me and not has_at_mention:
        reply_to_bot = False
        if reply := getattr(event, "reply", None):
            try:
                sender = getattr(reply, "sender", None)
                if sender and str(getattr(sender, "user_id", "")) == str(bot.self_id):
                    reply_to_bot = True
            except Exception:
                pass
        if not reply_to_bot:
            body = f"{plugin_config.bot_name} {body}"

    if not reply_id:
        reply_id = _extract_reply_message_id_from_event(event)

    # 第2行（可选）：回复元数据，格式 "回复id: {id}"
    if reply_id:
        content += f"回复id: {reply_id}\n"
    # 第3行起：正文
    content += body

    # 构建用户名：仅保留用户真实显示名，不混入群身份标签（群主/管理员）
    # 避免模型误把“群主-”等前缀当成用户名的一部分。
    user_name = session.user.name or session.user.nick or session.user.id
    if session.member and session.member.nick:
        user_name = session.member.nick

    # ========== 步骤1: 处理文本消息（快速） ==========
    # 用锁保证多bot并发安全: SELECT + INSERT + COMMIT 原子化
    is_new_text_message = True
    async with _get_dedup_lock(session.scene.id):
        if is_text:
            do_insert = True
            if body:
                time_window = datetime.datetime.now() - datetime.timedelta(seconds=3)
                existing = await db_session.execute(
                    Select(ChatHistory).where(
                        ChatHistory.session_id == session.scene.id,
                        ChatHistory.user_id == session.user.id,
                        ChatHistory.created_at >= time_window,
                    )
                )
                if any(
                    history.content.endswith(body)
                    for history in existing.scalars().all()
                ):
                    logger.debug("消息已存在，跳过重复记录")
                    do_insert = False

            is_new_text_message = do_insert

            if do_insert:
                chat_history = ChatHistory(
                    session_id=session.scene.id,
                    user_id=session.user.id,
                    content_type="text",
                    content=content,
                    user_name=user_name,
                )
                db_session.add(chat_history)

        # 在锁内提交,确保第二个bot的SELECT能看到第一个bot的写入
        try:
            await db_session.commit()
        except Exception as e:
            logger.error(f"保存文本消息失败: {e}")
            await db_session.rollback()

    if is_text and not is_new_text_message:
        logger.info(f"检测到重复入站消息，跳过回复 - session: {session.scene.id}")
        return

    # ========== 步骤2: 决定是否回复（在图片处理前判断） ==========
    plain_text = event.get_plaintext()
    stripped_plain_text = msg.extract_plain_text().strip()
    command_like = plain_text.startswith(("!", "！", "/", "#", "?", "\\"))
    if stripped_plain_text.lower().startswith(plugin_config.bot_name):
        to_me = True
    explicit_to_me = to_me
    continuous_to_me = (
        not explicit_to_me
        and not command_like
        and bool(stripped_plain_text)
        and session.scene.type == SceneType.GROUP
        and _is_continuous_conversation(session.scene.id, session.user.id)
    )
    if continuous_to_me:
        logger.debug(
            f"群 {session.scene.id} 用户 {session.user.id} 命中连续对话窗口"
        )
    should_reply = (
        to_me
        or continuous_to_me
        or (random.random() < plugin_config.reply_probability)
    )
    if explicit_to_me or continuous_to_me:
        _refresh_continuous_conversation(session.scene.id, session.user.id)
    if not plain_text and not imgs:
        should_reply = False
    if command_like:
        should_reply = False
    if not plain_text and not (to_me or continuous_to_me):
        should_reply = False

    # ========== 步骤3: 处理图片消息 ==========
    # 如果要回复则同步等待图片处理完成,否则后台异步
    content_prefix = f"id: {get_message_id()}\n"
    if imgs:
        if should_reply:
            for img in imgs:
                await process_image_message(
                    db_session, img, event, bot, state, session, user_name, content_prefix
                )
        else:
            for img in imgs:
                asyncio.create_task(
                    _process_image_task(
                        img, event, bot, state, session, user_name, content_prefix
                    )
                )

    # ========== 步骤4: 处理回复 ==========
    if to_me or continuous_to_me:
        user_id = session.user.id
        user_name = session.user.name or session.user.nick
    else:
        user_id = ""
        user_name = ""
    if should_reply:
        group_id = session.scene.id
        request = ReplyRequest(
            request_id=f"{group_id}:{datetime.datetime.now().timestamp()}:{random.random()}",
            session=session,
            interface=interface,
            bot=bot,
            event=event,
            bot_name=bot_name,
            user_id=user_id,
            user_name=user_name,
            is_tome=to_me,
            is_continuous=continuous_to_me,
            reply_to_id=reply_id,
        )
        await set_latest_request_id(group_id, request.request_id)
        async with _group_reply_state_lock:
            reply_state = _group_reply_states.setdefault(group_id, GroupReplyState())
            reply_state.latest = request
            if reply_state.running:
                if reply_state.task and not reply_state.task.done():
                    reply_state.task.cancel()
                    logger.info(f"群 {group_id} 收到更新请求，已取消旧回复并切换到最新")
                else:
                    # 兜底：若 running=True 但 worker 已结束（或异常丢失），立即拉起新 worker，
                    # 避免最新请求长期卡在 latest 槽位里无人消费。
                    logger.warning(
                        f"群 {group_id} 回复状态异常（running=True 但 worker 不可用），已重启并切换到最新请求"
                    )
                    _start_group_reply_worker_locked(group_id, reply_state)
            else:
                _start_group_reply_worker_locked(group_id, reply_state)

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
                image_fetch(event, bot, state, img), timeout=15.0
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
            new_media = MediaStorage(
                file_hash=file_hash,
                file_path=file_name,
                references=1,
                description="[图片]",  # 占位符
            )
            db_session.add(new_media)
            try:
                # 必须 flush 以触发可能的 UniqueViolation 错误
                await db_session.flush()
                media_obj = new_media

            except Exception as e:
                # C. 插入失败，从 session 中移除失败的对象，重新查询判断是否为唯一约束冲突
                await db_session.rollback()  # 先回滚，清理 session 状态
                media_obj = (await db_session.execute(stmt)).scalar_one_or_none()
                if media_obj is None:
                    # 非唯一约束冲突，记录错误并重新抛出
                    logger.error(f"插入图片记录失败（非并发冲突）: {e}")
                    raise
                # 唯一约束冲突，说明是并发插入
                logger.info(f"图片并发插入冲突 {file_hash}，转为更新模式")
                media_obj.references += 1
                db_session.add(media_obj)

        # 4. 添加聊天历史 (ChatHistory)
        # 此时 media_obj 一定是有效的 (无论是新插的还是查出来的)
        if media_obj:
            async with _get_dedup_lock(session.scene.id):
                # 确保 flush 拿到 media_id (如果是新插入的对象)
                await db_session.flush()

                # 刷新对象以确保它在当前 session 中
                await db_session.refresh(media_obj)

                # 检查是否已存在相同的图片记录 (多bot去重)
                time_window = datetime.datetime.now() - datetime.timedelta(seconds=3)
                existing_img = await db_session.execute(
                    Select(ChatHistory).where(
                        ChatHistory.session_id == session.scene.id,
                        ChatHistory.media_id == media_obj.media_id,
                        ChatHistory.created_at >= time_window,
                    )
                )
                if existing_img.scalar_one_or_none():
                    logger.debug("图片记录已存在，跳过重复")
                else:
                    chat_history = ChatHistory(
                        session_id=session.scene.id,
                        user_id=session.user.id,
                        content_type=content_type,
                        content=f"{content_prefix}{file_name}",
                        user_name=user_name,
                        media_id=media_obj.media_id,
                    )
                    db_session.add(chat_history)

                # 5. 在锁内提交
                await db_session.commit()

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        await db_session.rollback()


async def _process_image_task(
    img, event, bot, state, session, user_name, content_prefix
):
    """后台图片处理任务，使用独立的数据库会话，不阻塞主消息流程"""
    async with get_session() as db_session:
        await process_image_message(
            db_session, img, event, bot, state, session, user_name, content_prefix
        )


async def handle_reply_logic(
    db_session,
    request_id: str,
    session: Uninfo,
    interface: QryItrface,
    bot: Bot,
    event: Event,
    bot_name: str,
    user_id: str,
    user_name: str | None,
    is_tome: bool,
    is_continuous: bool,
    reply_to_id: str | None,
):
    """处理回复逻辑"""
    is_private = session.scene.type == SceneType.PRIVATE
    try:
        # 获取最近几条用于 Flash 快速判断
        # 注意：Flash 模型是纯文本模型，它看不懂图片，所以这里我们只喂文本内容
        recent_msgs = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(
                        ChatHistory.session_id == session.scene.id,
                        ChatHistory.content_type != "bot",
                    )
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(3)
                )
            )
            .scalars()
            .all()
        )
        recent_msgs = recent_msgs[::-1]

        if not recent_msgs:
            return

        # 简单的文本摘要用于 Gatekeeper
        history_summary = ""
        for m in recent_msgs:
            if m.content_type == "image":
                history_summary += f"{m.user_name}: [发送了一张图片/表情包，可能只是随手发的]\n"
            else:
                history_summary += f"{m.user_name}: {m.content}\n"

        current_msg_text = (
            recent_msgs[-1].content
            if recent_msgs[-1].content_type == "text"
            else "[图片/表情包。除非用户明确在问这张图、@bot、回复bot或正在延续图片话题，否则通常不需要回应]"
        )
        gatekeeper_msg_text = current_msg_text
        if is_continuous:
            gatekeeper_msg_text = (
                "这是用户在刚才主动呼叫 bot 后的连续对话消息。"
                "如果像追问、补充、回应 bot 或继续话题，应倾向回复；"
                "如果只是“嗯”“哈哈”“行”等无需回应的短反馈，可以不回复。\n"
                f"{current_msg_text}"
            )

        # === Gatekeeper 判断 ===
        if not is_tome:
            should_reply = await check_if_should_reply(
                history_summary, gatekeeper_msg_text, bot_name, is_private=is_private
            )
            if not should_reply:
                return

        # === 获取详细历史给 Agent ===
        last_msg = await _load_agent_history(db_session, session.scene.id)

        if not last_msg:
            logger.info("没有历史消息，跳过回复")
            return

        role_map: dict[str, str] = {}
        group_members: list[Any] | None = None
        if not is_private:
            try:
                group_members = list(
                    await interface.get_members(SceneType.GROUP, session.scene.id)
                )
                for member in group_members:
                    role_name = getattr(getattr(member, "role", None), "name", None)
                    if role_name in {"owner", "admin"}:
                        role_map[str(member.id)] = role_name
            except Exception as e:
                logger.warning(f"获取群成员身份信息失败，降级为无身份标注: {e}")

        logger.info("开始调用Agent决策...")
        try:
            strategy = await asyncio.wait_for(
                choice_response_strategy(
                    db_session,
                    session.scene.id,
                    request_id,
                    last_msg,
                    user_id,
                    user_name,
                    "",
                    interface,
                    role_map,
                    session.self_id,  # 传递bot的ID
                    reply_to_id,
                    bot,
                    event,
                    is_private=is_private,
                    group_members=group_members,
                ),
                timeout=plugin_config.agent_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent 思考超时 - session: {session.scene.id}")
            return

        except asyncio.CancelledError:
            logger.info(f"群 {session.scene.id} 回复任务被取消（切换到更新请求）")
            raise

        logger.info(f"Agent决策结果: {strategy}")

    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
        print(traceback.format_exc())
        await _safe_rollback(db_session)


def _build_wordcloud_image(words: str) -> BytesIO:
    """Generate a PNG image bytes object from words using WordCloud."""
    wc = (
        WordCloud(
            font_path=Path(__file__).parent / "SourceHanSans.otf",
            width=1000,
            height=500,
        )
        .generate(words)
        .to_image()
    )
    image_bytes = BytesIO()
    wc.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


async def _collect_words_from_db(
    db_session, session_id: str, days: int = 1, user_id: str | None = None
) -> str:
    """Query chat history and return a cleaned space-joined word string for wordcloud."""
    if not 1 <= days <= MAX_WORDCLOUD_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_WORDCLOUD_DAYS}")
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    where = [
        ChatHistory.session_id == session_id,
        ChatHistory.content_type == "text",
        ChatHistory.created_at >= cutoff,
    ]
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


def _parse_wordcloud_days(arg_text: str) -> int:
    if not arg_text:
        return 1
    if not arg_text.isdigit():
        raise ValueError("统计范围应为纯数字")
    days = int(arg_text)
    if not 1 <= days <= MAX_WORDCLOUD_DAYS:
        raise ValueError(f"统计范围应为 1-{MAX_WORDCLOUD_DAYS} 天")
    return days


frequency = on_command("词频")


@frequency.handle()
async def _(
    db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()
):
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    try:
        days = _parse_wordcloud_days(arg_text)
    except ValueError as e:
        await frequency.finish(str(e))

    words = await _collect_words_from_db(
        db_session, session_id, days=days, user_id=session.user.id
    )
    if not words:
        await frequency.finish("在指定时间内，没有说过话呢")

    image_bytes = await asyncio.to_thread(_build_wordcloud_image, words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


group_frequency = on_command("群词频")


@group_frequency.handle()
async def _(
    db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()
):
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    try:
        days = _parse_wordcloud_days(arg_text)
    except ValueError as e:
        await group_frequency.finish(str(e))

    words = await _collect_words_from_db(
        db_session, session_id, days=days, user_id=None
    )
    # Even if no words, return an empty wordcloud (original group_frequency didn't check emptiness)
    if not words:
        await group_frequency.finish("在指定时间内，没有消息可统计")

    image_bytes = await asyncio.to_thread(_build_wordcloud_image, words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


@scheduler.scheduled_job(
    "interval", minutes=60, max_instances=1, coalesce=True, id="vectorize_chat"
)
async def vectorize_message_history():
    async with get_session() as db_session:
        session_ids = await db_session.execute(
            Select(ChatHistory.session_id.distinct())
        )
        session_ids = session_ids.scalars().all()
        logger.info("开始向量化会话")
        for session_id in session_ids:
            try:
                res = await process_and_vectorize_session_chats(db_session, session_id)
                if res:
                    logger.info(
                        f"向量化会话 {res['session_id']} 成功，共处理 {res['processed_groups']}/{res['total_groups']} 组"
                    )
                else:
                    logger.info(f"{session_id} 无需向量化")
            except Exception as e:
                print(traceback.format_exc())
                logger.error(f"向量化会话 {session_id} 失败: {e}")
                continue


@scheduler.scheduled_job(
    "interval", minutes=30, max_instances=1, coalesce=True, id="vectorize_media"
)
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
            Select(MediaStorage).where(
                MediaStorage.references >= 3, MediaStorage.vectorized.is_(False)
            )
        )
        medias = medias_res.scalars().all()
        media_ids = [m.media_id for m in medias]
        logger.info(f"待处理高频图片数量: {len(media_ids)}")

        for media_id in media_ids:
            media = await db_session.get(MediaStorage, media_id)
            if media is None:
                continue
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
                        encoded_string = base64.b64encode(file_data).decode("utf-8")

                        # 构造 Data URI
                        ext = media.file_path.split(".")[-1].lower()
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        if ext == "gif":
                            mime = "image/gif"  # Qwen-VL 支持 GIF

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
                    response = await get_tagging_model().ainvoke(
                        [
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": img_data_uri},
                                    },
                                ]
                            )
                        ]
                    )

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
                    err_str = str(e)
                    # 400 错误（图片尺寸/格式非法、内容违规）不可重试，标记跳过
                    if "Error code: 400" in err_str:
                        if "data_inspection_failed" in err_str:
                            logger.warning(f"图片 {media_id} 内容违规，跳过向量化")
                        else:
                            logger.warning(
                                f"图片 {media_id} 请求非法（400），跳过向量化: {e}"
                            )
                        media.vectorized = True
                        db_session.add(media)
                        await db_session.commit()
                    else:
                        logger.error(f"模型识别图片失败 {media_id}: {e}")
                    continue

                # 3. 结果处理
                if not is_meme:
                    logger.info(f"图片 {media_id} 被判定为非表情包(杂图)，跳过入库")
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    continue

                # 4. 是表情包 -> 入库
                try:
                    # A. 存描述到 SQL
                    media.description = description

                    # B. 存向量到 Qdrant (传带 MIME 头的 data URI，避免 PNG/GIF 被误判为 JPEG)
                    await DB.insert_media(media_id, img_data_uri, description)

                    # C. 标记完成
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    logger.info(f"表情包入库成功 {media_id}: {description}")

                except Exception as e:
                    logger.error(f"向量化插入失败 {media_id}: {e}")
                    await db_session.rollback()
                    continue

            except Exception as e:
                logger.error(f"处理媒体循环异常 {media_id}: {e}")
                await db_session.rollback()
                continue

        await db_session.commit()
        if len(medias) > 0:
            logger.info("本轮图片处理完成")


@scheduler.scheduled_job(
    "interval", minutes=35, max_instances=1, coalesce=True, id="clear_cache"
)
async def clear_cache_pic():
    async with get_session() as db_session:
        # ── 1. 删除低引用且过期的数据库记录及对应文件 ──
        result = await db_session.execute(
            Select(MediaStorage).where(
                MediaStorage.references < 3,
                MediaStorage.created_at
                < datetime.datetime.now() - datetime.timedelta(days=30),
            )
        )
        medias = result.scalars().all()

        if medias:
            # 批量删除文件，减少线程切换
            media_files = [Path(pic_dir / media.file_path) for media in medias]

            def _batch_delete_media_files(files: list):
                deleted = 0
                for file_path in files:
                    try:
                        file_path.unlink(missing_ok=True)
                        deleted += 1
                        logger.debug(f"删除文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
                return deleted

            deleted_files = await asyncio.to_thread(_batch_delete_media_files, media_files)

            # 批量删除数据库记录
            for media in medias:
                await db_session.delete(media)

            try:
                await db_session.commit()
                logger.info(f"成功清理 {len(medias)} 个过期媒体记录（{deleted_files} 个文件）")
            except Exception as e:
                logger.error(f"批量删除数据库记录失败: {e}")
                await db_session.rollback()

        # ── 2. 删除磁盘上有但数据库里没有的孤立文件 ──
        known_files_result = await db_session.execute(Select(MediaStorage.file_path))
        known_files = {row[0] for row in known_files_result.all()}

        disk_files = await asyncio.to_thread(lambda: list(pic_dir.iterdir()))
        orphaned = [f for f in disk_files if f.is_file() and f.name not in known_files]

        if orphaned:
            # 批量删除文件，减少异步切换开销
            def _batch_delete_orphaned_files(files: list):
                deleted = 0
                for f in files:
                    try:
                        f.unlink(missing_ok=True)
                        deleted += 1
                        logger.debug(f"删除孤立文件: {f.name}")
                    except Exception as e:
                        logger.error(f"删除孤立文件失败 {f.name}: {e}")
                return deleted

            deleted_count = await asyncio.to_thread(_batch_delete_orphaned_files, orphaned)
            logger.info(f"成功清理 {deleted_count}/{len(orphaned)} 个孤立文件")


async def _call_summary_model(existing_summary: str, chat_text: str) -> str | None:
    """调用 LLM 更新群体认知档案。
    若触发内容违规（data_inspection_failed），会对聊天记录做二分截断后最多重试 3 次。
    """
    from langchain_core.messages import HumanMessage as LCHumanMessage, SystemMessage

    system = """你是一个群文化分析师。你的任务是维护一份关于QQ群的认知档案。
档案包含：群内常见话题、活跃成员特征、内部梗/黑话、群文化氛围。

【核心原则】标注为[BOT]的消息是机器人自身的回复。档案只记录真实用户的群文化，绝对不记录机器人的行为模式或回复话术。

规则：
1. 只能基于提供的聊天记录总结，不要凭空发明内容
2. 保留档案中仍然有效的内容，用新聊天补充或修正旧内容
3. 如果某个内容长期（超过30天）无聊天印证，可删除
4. 输出完整更新后的档案，不超过500字，不要输出任何其他内容
5. 【必须执行】如果现有档案中包含BOT的回复话术（如"别得寸进尺"、"？你又来"、"行了行了"等BOT说的话），必须将其删除。内部梗/黑话只能来自真实用户的发言，不能来自BOT。
6. 【必须执行】"活跃成员特征"部分只描述真实用户，不要把BOT列为活跃成员或描述BOT的行为模式。
7. 【必须执行】"群文化氛围"只描述用户之间的互动氛围，不要描述用户与BOT之间的互动循环。"""
    history_intro = (
        "（无，这是首次建档）" if not existing_summary.strip() else existing_summary
    )

    lines = chat_text.splitlines()
    max_retries = 3

    for attempt in range(max_retries + 1):
        current_text = "\n".join(lines)
        if not current_text.strip():
            logger.warning("档案更新：聊天记录经截断后已为空，放弃本次更新")
            return None

        user_msg = f"【现有档案】\n{history_intro}\n\n【最新聊天记录】\n{current_text}\n\n请输出更新后的档案："
        try:
            resp = await get_summary_model().ainvoke(
                [
                    SystemMessage(content=system),
                    LCHumanMessage(content=user_msg),
                ]
            )
            if not isinstance(resp.content, str) or not resp.content.strip():
                return None
            if attempt > 0:
                logger.info(
                    f"档案更新：截断后第 {attempt} 次重试成功（剩余 {len(lines)} 条消息）"
                )
            return resp.content.strip()
        except Exception as e:
            err_str = str(e)
            if "data_inspection_failed" in err_str or (
                "Error code: 400" in err_str and "inappropriate" in err_str
            ):
                if attempt < max_retries:
                    # 去掉后半段消息，逐步缩小范围
                    lines = lines[: max(1, len(lines) // 2)]
                    logger.warning(
                        f"档案更新：内容违规，截断至 {len(lines)} 条消息后重试（第 {attempt + 1}/{max_retries} 次）"
                    )
                else:
                    logger.warning(
                        f"档案更新：内容违规，已重试 {max_retries} 次仍失败，放弃本次更新"
                    )
                    return None
            else:
                logger.error(f"档案更新 LLM 调用失败: {e}")
                return None

    return None


async def _update_single_group_memory(db_session, session_id: str):
    """更新单个群的认知档案（内部函数）"""
    from sqlalchemy import func as sqlfunc

    stmt = Select(GroupMemory).where(GroupMemory.session_id == session_id)
    record = (await db_session.execute(stmt)).scalar_one_or_none()

    # 获取当前消息总量
    total_count = (
        await db_session.execute(
            Select(sqlfunc.count(ChatHistory.msg_id)).where(
                ChatHistory.session_id == session_id
            )
        )
    ).scalar_one()

    last_count = record.msg_count_at_last_update if record else 0
    new_msg_count = total_count - last_count

    # 双重触发条件：新增消息 >= 100 或 距上次更新 >= 6 小时
    if record and new_msg_count < 100:
        time_since = datetime.datetime.now() - record.updated_at
        if time_since.total_seconds() < 6 * 3600:
            logger.info(
                f"群 {session_id} 无需更新档案（新增 {new_msg_count} 条，距上次更新 {time_since}）"
            )
            return

    # 拉取自上次更新后的文本消息，最多 200 条
    cutoff = record.updated_at if record else datetime.datetime.min
    recent_msgs = (
        (
            await db_session.execute(
                Select(ChatHistory)
                .where(
                    ChatHistory.session_id == session_id,
                    ChatHistory.created_at > cutoff,
                    ChatHistory.content_type.in_(["text", "bot"]),
                )
                .order_by(ChatHistory.created_at)
                .limit(200)
            )
        )
        .scalars()
        .all()
    )

    if not recent_msgs:
        return

    def _format_msg_for_summary(m):
        content = m.content
        if m.content_type == "bot":
            # 去掉 "id: XXXXX\n" 前缀，只保留实际回复内容
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1 :]
        return f"[{m.created_at.strftime('%m-%d %H:%M')}] {'[BOT] ' if m.content_type == 'bot' else ''}{m.user_name}: {content[:100]}"

    chat_text = "\n".join(_format_msg_for_summary(m) for m in recent_msgs)

    existing_summary = record.summary if record else ""
    new_summary = await _call_summary_model(existing_summary, chat_text)
    if not new_summary:
        return

    # 后处理：过滤掉描述BOT自身行为/话术的条目
    bot_name = plugin_config.bot_name
    filtered_lines = []
    for line in new_summary.splitlines():
        stripped = line.strip().lstrip("-•* ").strip()
        # 跳过描述bot行为模式的行
        if bot_name and (
            f"{bot_name}为" in stripped
            or f"{bot_name}是" in stripped
            or f"{bot_name}主导" in stripped
            or f"{bot_name}维持" in stripped
            or f"{bot_name}以" in stripped
        ):
            logger.info(f"档案过滤：移除BOT行为描述行: {stripped[:50]}")
            continue
        # 跳过将bot回复标注为"标准回应"/"模板"的行
        if "标准回应" in stripped or "回应模板" in stripped:
            logger.info(f"档案过滤：移除BOT模板描述行: {stripped[:50]}")
            continue
        filtered_lines.append(line)
    new_summary = "\n".join(filtered_lines)
    if not new_summary.strip():
        return

    if not record:
        record = GroupMemory(
            session_id=session_id,
            summary=new_summary,
            msg_count_at_last_update=total_count,
        )
        db_session.add(record)
    else:
        record.summary = new_summary
        record.msg_count_at_last_update = total_count

    await db_session.commit()
    logger.info(f"群 {session_id} 档案更新成功（{len(new_summary)} 字）")


@scheduler.scheduled_job(
    "interval", hours=6, max_instances=1, coalesce=True, id="update_group_memory"
)
async def update_group_memory():
    async with get_session() as db_session:
        # 只查询最近 24 小时内有新消息的群
        time_threshold = datetime.datetime.now() - datetime.timedelta(days=1)
        stmt = Select(ChatHistory.session_id.distinct()).where(
            ChatHistory.created_at > time_threshold
        )
        session_ids = (await db_session.execute(stmt)).scalars().all()

    # 如果最近没人说话，直接返回
    if not session_ids:
        return

    sem = asyncio.Semaphore(5)  # 最多同时处理 5 个群

    async def _update_one(session_id: str):
        async with sem:
            async with get_session() as db_session:
                try:
                    await _update_single_group_memory(db_session, session_id)
                except Exception as e:
                    logger.error(f"更新群档案失败 {session_id}: {e}")

    await asyncio.gather(*[_update_one(sid) for sid in session_ids])
