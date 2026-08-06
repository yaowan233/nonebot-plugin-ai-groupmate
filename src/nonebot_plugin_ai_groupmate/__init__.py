import re
import json
import time
import base64
import random
import asyncio
import datetime
import traceback
from io import BytesIO
from typing import Any
from pathlib import Path
from functools import lru_cache
from collections import deque
from dataclasses import field, dataclass

import jieba
from nonebot import logger, require, get_bots, get_driver, on_command, on_message
from wordcloud import WordCloud
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event, Message
from nonebot.permission import SUPERUSER
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
from nonebot_plugin_alconna import (
    Image,
    Target,
    UniMessage,
    image_fetch,
    get_message_id,
)
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import UniMsg

from .agent import (
    _parse_msg_meta,
    _detect_repeat_chain,
    check_if_should_reply,
    choice_response_strategy,
)
from .model import ChatHistory, MediaStorage, ChatHistorySchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .webui import register_usage_webui
from .config import Config, create_tagging_llm
from .memory import DB, MEDIA_EMBEDDING_VERSION
from .concurrency import agent_run_gate, maintenance_gate, background_image_gate
from .reply_guard import set_latest_request_id
from .agent.reaction import is_onebot_context
from .runtime_config import (
    RESTART_REQUIRED_FIELDS,
    get_runtime_config,
    mark_restart_fields_applied,
    load_runtime_config_overrides,
)
from .agent.reply_tools import create_reply_tool
from .relation_maintenance import count_negative_relations, reset_negative_relations


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
relation_backup_dir = plugin_data_dir / "relation_backups"
plugin_config = get_runtime_config()
MAX_WORDCLOUD_DAYS = 3650
with open(Path(__file__).parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]

@lru_cache
def get_tagging_model():
    return create_tagging_llm(plugin_config)


def _refresh_runtime_resources(_changed_fields: set[str]) -> None:
    get_tagging_model.cache_clear()
    from .agent import refresh_runtime_resources as refresh_agent_resources
    from .group_memory import get_summary_model

    refresh_agent_resources()
    get_summary_model.cache_clear()


register_usage_webui(
    plugin_config,
    on_config_change=_refresh_runtime_resources,
)


@get_driver().on_startup
async def _load_webui_runtime_config() -> None:
    try:
        async with get_session() as db_session:
            changed_fields = await load_runtime_config_overrides(db_session)
        _refresh_runtime_resources(changed_fields)
        if changed_fields & RESTART_REQUIRED_FIELDS:
            await DB.reconfigure()
            mark_restart_fields_applied()
        if changed_fields:
            logger.info(
                f"已加载 WebUI 配置覆盖项，变更字段数={len(changed_fields)}"
            )
    except Exception:
        logger.exception(
            "加载 WebUI 配置失败，继续使用环境变量；请确认已执行 nb orm upgrade"
        )


@get_driver().on_shutdown
async def _close_vector_resources() -> None:
    await DB.close()


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
    proactive_meme_only: bool = False
    meme_required: bool = False
    proactive_reaction_only: bool = False
    reaction_required: bool = False
    repeat_text: str | None = None


@dataclass
class GroupReplyState:
    running: bool = False
    latest: ReplyRequest | None = None
    addressed: deque[ReplyRequest] = field(default_factory=deque)
    active: ReplyRequest | None = None
    task: asyncio.Task | None = None


# 每个群串行处理少量明确 @；非定向回复只保留最新一条，避免高峰期刷屏。
_group_reply_states: dict[str, GroupReplyState] = {}
_group_reply_state_lock = asyncio.Lock()
_continuous_conversation_until: dict[tuple[str, str], datetime.datetime] = {}
MAX_GROUP_ADDRESSED_REQUESTS = 3

# 多bot去重锁: 每个群串行化消息记录,防止并发SELECT查不到对方未提交数据
_dedup_locks: dict[str, asyncio.Lock] = {}
_background_image_tasks: set[asyncio.Task[None]] = set()

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


def _sample_proactive_reply_modes(
    *,
    addressed: bool,
    continuous: bool,
    command_like: bool,
    has_text: bool,
    is_group: bool,
    reaction_supported: bool,
) -> tuple[bool, bool, bool]:
    """返回（普通回复、reaction 专用、图片表情专用）；定向消息不采样。"""
    if addressed or continuous:
        return False, False, False
    random_reply = random.random() < plugin_config.reply_probability
    eligible = not command_like and has_text and is_group
    proactive_reaction_only = (
        not random_reply
        and eligible
        and reaction_supported
        and random.random() < plugin_config.proactive_reaction_probability
    )
    proactive_meme_only = (
        not random_reply
        and not proactive_reaction_only
        and eligible
        and random.random() < plugin_config.proactive_meme_probability
    )
    return random_reply, proactive_reaction_only, proactive_meme_only


def _is_explicit_reaction_request(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if _is_explicit_meme_request(normalized):
        return False
    return bool(
        re.search(
            r"(?:回应|回复|点|加|来|贴|上)(?:一个|个|一下|点)?(?:表情|reaction)"
            r"|(?:表情|reaction)(?:回应|回复|一下)"
            r"|点(?:个|一下)?(?:赞|爱心)",
            normalized,
        )
    )


def _is_explicit_meme_request(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if "表情包" in normalized:
        return True
    return bool(
        re.search(
            r"(?:发|来|整|找|搜|给)(?:一个|个|点|些|一下|几张|张)?(?:图片|图|表情)"
            r"|(?:图片|图|表情)(?:发|来|整|找|搜)(?:一个|个|点|些|一下)?"
            r"|(?:图片|图|表情包)(?:呢|在哪)[？?]?($|\s)",
            normalized,
        )
    )


def _sample_repeat_reply(
    *,
    repeat_text: str | None,
    addressed: bool,
    continuous: bool,
    command_like: bool,
    is_group: bool,
) -> bool:
    """对已识别的群聊复读队形独立采样。"""
    if (
        repeat_text is None
        or addressed
        or continuous
        or command_like
        or not is_group
    ):
        return False
    return random.random() < plugin_config.repeat_probability


async def _load_repeat_chain_text(
    db_session: Any,
    session_id: str,
) -> str | None:
    """读取刚入库的近期消息，并检测当前是否仍是连续复读队形。"""
    context_minutes = plugin_config.continuous_conversation_minutes
    if context_minutes <= 0:
        return None
    context_since = datetime.datetime.now() - datetime.timedelta(
        minutes=context_minutes
    )
    rows = (
        (
            await db_session.execute(
                Select(ChatHistory)
                .where(
                    ChatHistory.session_id == session_id,
                    ChatHistory.created_at >= context_since,
                )
                .order_by(ChatHistory.msg_id.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    history = [ChatHistorySchema.model_validate(row) for row in rows[::-1]]
    return _detect_repeat_chain(history)


def _get_dedup_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _dedup_locks:
        _dedup_locks[session_id] = asyncio.Lock()
    return _dedup_locks[session_id]


def _select_addressed_bot_id(
    event: Any,
    connected_bot_ids: set[str],
) -> str | None:
    """Select the one connected Bot explicitly addressed by the raw message."""
    original_message = getattr(event, "original_message", None)
    if original_message is not None:
        for segment in original_message:
            if getattr(segment, "type", None) != "at":
                continue
            data = getattr(segment, "data", {})
            if not isinstance(data, dict):
                continue
            target = data.get("qq") or data.get("target") or data.get("user_id")
            if target is not None and str(target) in connected_bot_ids:
                return str(target)

    reply = getattr(event, "reply", None)
    sender = getattr(reply, "sender", None)
    reply_user_id = getattr(sender, "user_id", None)
    if reply_user_id is not None and str(reply_user_id) in connected_bot_ids:
        return str(reply_user_id)
    return None


def _matches_inbound_message(
    history_content: str,
    content_prefix: str,
    body: str,
    *,
    exact_id_only: bool = False,
) -> bool:
    if history_content.startswith(content_prefix):
        return True
    return not exact_id_only and bool(body) and history_content.endswith(body)


def _is_connected_bot_sender(user_id: Any, connected_bot_ids: set[str]) -> bool:
    return str(user_id) in connected_bot_ids


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


async def _queue_group_reply_request(
    group_id: str,
    request: ReplyRequest,
) -> bool:
    """明确 @ 串行排队；非定向消息不能打断或挤占定向请求。"""
    async with _group_reply_state_lock:
        reply_state = _group_reply_states.setdefault(group_id, GroupReplyState())
        active_is_addressed = bool(
            reply_state.active is not None and reply_state.active.is_tome
        )

        if request.is_tome:
            addressed_count = len(reply_state.addressed) + int(active_is_addressed)
            if addressed_count >= MAX_GROUP_ADDRESSED_REQUESTS:
                logger.warning(
                    f"群 {group_id} 待处理的 @Bot 请求已达上限，忽略额外请求"
                )
                return False
            reply_state.addressed.append(request)
            # 明确 @ 到来时丢弃尚未开始的随机/表情/复读请求。
            reply_state.latest = None
            if not reply_state.running:
                _start_group_reply_worker_locked(group_id, reply_state)
            elif not reply_state.task or reply_state.task.done():
                logger.warning(f"群 {group_id} 回复 worker 不可用，已重新启动")
                _start_group_reply_worker_locked(group_id, reply_state)
            elif reply_state.active is not None and not active_is_addressed:
                # 抢占正在运行的低优先级 Agent，并提前使其发送工具失效。
                await set_latest_request_id(group_id, request.request_id)
                reply_state.task.cancel()
                logger.info(f"群 {group_id} 收到明确 @，已取消后台回复")
            return True

        if active_is_addressed or reply_state.addressed:
            logger.debug(f"群 {group_id} 存在待处理的 @Bot 请求，忽略后台回复采样")
            return False

        # 非定向请求仍只保留最新一个，避免随机插话、主动表情或复读积压刷屏。
        reply_state.latest = request
        if not reply_state.running:
            _start_group_reply_worker_locked(group_id, reply_state)
        elif not reply_state.task or reply_state.task.done():
            logger.warning(f"群 {group_id} 回复 worker 不可用，已重新启动")
            _start_group_reply_worker_locked(group_id, reply_state)
        elif reply_state.active is not None:
            await set_latest_request_id(group_id, request.request_id)
            reply_state.task.cancel()
            logger.info(f"群 {group_id} 已将后台回复切换到最新消息")
        return True


async def _run_group_reply_worker(group_id: str):
    """按群串行处理明确 @ 队列，并在队列为空时消费最新后台请求。"""
    try:
        while True:
            async with _group_reply_state_lock:
                state = _group_reply_states.get(group_id)
                if not state:
                    return
                if state.addressed:
                    request = state.addressed.popleft()
                else:
                    request = state.latest
                    state.latest = None
                state.active = request

            if request is None:
                break

            # 排队中的 @ 请求不能提前覆盖当前请求的发送许可；轮到它执行时
            # 再切换 request_id，保证前一个 Agent 能正常完成发送。
            await set_latest_request_id(group_id, request.request_id)

            # Wait for an Agent slot before opening a database session.  This
            # bounds cross-group concurrency without consuming pool capacity
            # while queued.
            async with agent_run_gate.slot():
                await handle_reply_logic(
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
                    getattr(request, "proactive_meme_only", False),
                    getattr(request, "meme_required", False),
                    getattr(request, "proactive_reaction_only", False),
                    getattr(request, "reaction_required", False),
                    getattr(request, "repeat_text", None),
                )
    finally:
        async with _group_reply_state_lock:
            state = _group_reply_states.get(group_id)
            if state:
                state.running = False
                state.active = None
                state.task = None
                if state.addressed or state.latest is not None:
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
    connected_bot_ids = {str(bot_id) for bot_id in get_bots()}
    connected_bot_ids.add(str(bot.self_id))
    sender_is_connected_bot = _is_connected_bot_sender(
        session.user.id,
        connected_bot_ids,
    )
    addressed_bot_id = _select_addressed_bot_id(event, connected_bot_ids)
    if addressed_bot_id is not None and addressed_bot_id != str(bot.self_id):
        logger.debug(
            f"消息明确发给其他 Bot {addressed_bot_id}，当前 Bot {bot.self_id} 跳过处理"
        )
        return

    bot_name = plugin_config.bot_name
    imgs = msg.include(Image)
    # 第1行固定是本条消息的平台 ID 元数据，格式 "id: {id}"
    incoming_message_id = str(get_message_id())
    content_prefix = f"id: {incoming_message_id}\n"
    content = content_prefix
    to_me = addressed_bot_id == str(bot.self_id) or (
        addressed_bot_id is None and event.is_tome()
    )
    is_text = False
    reply_id: str | None = None  # 记录回复 ID，稍后单独成行插入
    body = ""  # 正文部分单独拼接
    has_at_mention = False
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
    if is_text and sender_is_connected_bot:
        # 本插件的发送工具会主动写入 bot 消息。给发送侧事务一个很短的提交窗口，
        # 再按平台消息 ID 判断是否已经入库；其他插件发出的消息仍会继续记录。
        await asyncio.sleep(0.25)
    is_new_text_message = True
    async with _get_dedup_lock(session.scene.id):
        if is_text:
            do_insert = True
            time_window = datetime.datetime.now() - datetime.timedelta(seconds=3)
            existing_query = Select(ChatHistory).where(
                ChatHistory.session_id == session.scene.id,
                ChatHistory.created_at >= time_window,
            )
            if not sender_is_connected_bot:
                existing_query = existing_query.where(
                    ChatHistory.user_id == session.user.id
                )
            existing = await db_session.execute(existing_query)
            if any(
                _matches_inbound_message(
                    history.content,
                    content_prefix,
                    body,
                    exact_id_only=sender_is_connected_bot,
                )
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
        and not has_at_mention
        and not reply_id
        and bool(stripped_plain_text)
        and session.scene.type == SceneType.GROUP
        and _is_continuous_conversation(session.scene.id, session.user.id)
    )
    if continuous_to_me:
        logger.debug(
            f"群 {session.scene.id} 用户 {session.user.id} 命中连续对话窗口"
        )
    is_group = session.scene.type == SceneType.GROUP
    repeat_text = None
    if (
        is_group
        and not to_me
        and not continuous_to_me
        and not command_like
        and bool(stripped_plain_text)
    ):
        repeat_text = await _load_repeat_chain_text(db_session, session.scene.id)
    repeat_reply_sample = _sample_repeat_reply(
        repeat_text=repeat_text,
        addressed=to_me,
        continuous=continuous_to_me,
        command_like=command_like,
        is_group=is_group,
    )
    if repeat_text is not None:
        # 队形中的消息只走复读采样，不再触发普通插话或主动表情回应。
        random_reply_sample = False
        proactive_reaction_only = False
        proactive_meme_only = False
    else:
        (
            random_reply_sample,
            proactive_reaction_only,
            proactive_meme_only,
        ) = _sample_proactive_reply_modes(
            addressed=to_me,
            continuous=continuous_to_me,
            command_like=command_like,
            has_text=bool(stripped_plain_text),
            is_group=is_group,
            reaction_supported=is_onebot_context(bot, event),
        )
    meme_required = (
        (to_me or continuous_to_me)
        and _is_explicit_meme_request(stripped_plain_text)
    )
    reaction_required = (
        not meme_required
        and (to_me or continuous_to_me)
        and is_onebot_context(bot, event)
        and _is_explicit_reaction_request(stripped_plain_text)
    )
    if meme_required:
        random_reply_sample = False
        proactive_reaction_only = False
        proactive_meme_only = True
    elif reaction_required:
        random_reply_sample = False
        proactive_reaction_only = True
        proactive_meme_only = False
    should_reply = (
        to_me
        or continuous_to_me
        or random_reply_sample
        or proactive_reaction_only
        or proactive_meme_only
        or repeat_reply_sample
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
    if imgs:
        if should_reply:
            for img in imgs:
                await process_image_message(
                    db_session, img, event, bot, state, session, user_name, content_prefix
                )
        else:
            for img in imgs:
                _start_background_image_task(
                    img,
                    event,
                    bot,
                    state,
                    session,
                    user_name,
                    content_prefix,
                )

    # ========== 步骤4: 处理回复 ==========
    # Whether to reply is independent from who sent the triggering message.
    # Always keep the real sender identity so bound custom tools can resolve
    # requests such as "查询我的成绩" without asking the model for an ID.
    user_id = session.user.id
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
            proactive_meme_only=proactive_meme_only,
            meme_required=meme_required,
            proactive_reaction_only=proactive_reaction_only,
            reaction_required=reaction_required,
            repeat_text=repeat_text if repeat_reply_sample else None,
        )
        await _queue_group_reply_request(group_id, request)

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
    # Queueing happens before a session is opened, so an image burst cannot
    # reserve every pooled connection while waiting for CPU/network work.
    async with background_image_gate.slot():
        async with get_session() as db_session:
            await process_image_message(
                db_session, img, event, bot, state, session, user_name, content_prefix
            )


def _consume_background_image_task(task: asyncio.Task[None]) -> None:
    _background_image_tasks.discard(task)
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def _start_background_image_task(
    img,
    event: Event,
    bot: Bot,
    state: T_State,
    session: Uninfo,
    user_name: str | None,
    content_prefix: str,
) -> bool:
    if len(_background_image_tasks) >= plugin_config.background_image_max_pending:
        logger.warning(
            "后台图片任务已达上限 "
            f"{plugin_config.background_image_max_pending}，跳过本张图片"
        )
        return False

    task = asyncio.create_task(
        _process_image_task(
            img, event, bot, state, session, user_name, content_prefix
        ),
        name=f"ai-groupmate-image:{session.scene.id}",
    )
    _background_image_tasks.add(task)
    task.add_done_callback(_consume_background_image_task)
    return True


async def handle_reply_logic(
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
    proactive_meme_only: bool = False,
    meme_required: bool = False,
    proactive_reaction_only: bool = False,
    reaction_required: bool = False,
    repeat_text: str | None = None,
):
    """处理回复逻辑"""
    is_private = session.scene.type == SceneType.PRIVATE
    try:
        if repeat_text is not None and not is_private:
            # 复读采样已经代表“加入队形”的决定，直接发送，避免模型二次犹豫或点评。
            async with get_session() as repeat_session:
                reply_tool = create_reply_tool(
                    repeat_session,
                    session.scene.id,
                    request_id,
                    interface=interface,
                    send_target=Target(
                        id=session.scene.id,
                        private=False,
                        self_id=session.self_id,
                    ),
                    bot_name=bot_name,
                    parse_msg_meta=_parse_msg_meta,
                    group_members=[],
                    repeat_text=repeat_text,
                )
                raw_result = await reply_tool.ainvoke(
                    {"content": repeat_text, "next_step": "end"}
                )
                result = json.loads(raw_result)
                if result.get("status") != "sent":
                    logger.info(f"复读消息未发送: {result.get('message', raw_result)}")
                await repeat_session.commit()
            return

        # 获取最近几条用于 Flash 快速判断
        # 注意：Flash 模型是纯文本模型，它看不懂图片，所以这里我们只喂文本内容
        async with get_session() as history_session:
            recent_msgs = (
                (
                    await history_session.execute(
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
            recent_msgs = [
                ChatHistorySchema.model_validate(row) for row in recent_msgs[::-1]
            ]
            await history_session.commit()

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
        elif proactive_reaction_only:
            gatekeeper_msg_text = (
                "这条群消息命中了低概率主动 reaction 采样。"
                "仅当用一个消息表情回应就能自然表达赞同、好笑、惊讶、安慰、感谢或轻量态度时才回复；"
                "提问、求助、敏感/沉重话题、真实冲突、他人之间的定向对话，"
                "以及没有明显反应价值的普通消息都应保持沉默。\n"
                f"{current_msg_text}"
            )
        elif proactive_meme_only:
            gatekeeper_msg_text = (
                "这条群消息命中了低概率主动表情包采样。"
                "仅当一张表情包能作为群友式的自然接梗、吐槽或情绪反应时才回复；"
                "认真求助、事实问题、敏感/沉重话题、真实冲突、他人之间的定向对话，"
                "以及普通到没有反应价值的消息都应保持沉默。\n"
                f"{current_msg_text}"
            )

        # === Gatekeeper 判断 ===
        if not is_tome and repeat_text is None:
            should_reply = await check_if_should_reply(
                history_summary,
                gatekeeper_msg_text,
                bot_name,
                is_private=is_private,
                proactive_meme_only=proactive_meme_only,
                proactive_reaction_only=proactive_reaction_only,
            )
            if not should_reply:
                return

        # === 获取详细历史给 Agent ===
        async with get_session() as history_session:
            last_msg = await _load_agent_history(history_session, session.scene.id)
            await history_session.commit()

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
            async with get_session() as tool_session:
                strategy = await asyncio.wait_for(
                    choice_response_strategy(
                        tool_session,
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
                        proactive_meme_only=proactive_meme_only,
                        meme_required=meme_required,
                        proactive_reaction_only=proactive_reaction_only,
                        reaction_required=reaction_required,
                        repeat_text=repeat_text,
                    ),
                    timeout=plugin_config.agent_timeout_seconds,
                )
        except asyncio.TimeoutError:
            logger.warning(f"Agent 思考超时 - session: {session.scene.id}")
            return

        except asyncio.CancelledError:
            logger.debug(f"群 {session.scene.id} 回复任务被取消（切换到更新请求）")
            raise

        logger.info(f"Agent决策结果: {strategy}")

    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
        print(traceback.format_exc())


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


reset_negative_relation = on_command(
    "重置负面关系",
    aliases={"清理负面关系"},
    permission=SUPERUSER,
    priority=1,
    block=True,
)


@reset_negative_relation.handle()
async def _(db_session: async_scoped_session, arg: Message = CommandArg()):
    confirmation = arg.extract_plain_text().strip()
    try:
        affected_count = await count_negative_relations(db_session)
    except Exception:
        await _safe_rollback(db_session)
        logger.exception("读取负面关系数量失败")
        await reset_negative_relation.finish("读取失败，数据库未修改，请查看日志。")

    if confirmation != "确认":
        await _safe_rollback(db_session)
        if affected_count == 0:
            await reset_negative_relation.finish("没有需要重置的负好感度关系。")
        await reset_negative_relation.finish(
            f"检测到 {affected_count} 条负好感度关系。\n"
            "本操作会先备份，再将这些用户的好感度归零并清空旧标签。\n"
            "确认执行请发送：/重置负面关系 确认"
        )

    try:
        result = await reset_negative_relations(db_session, relation_backup_dir)
    except Exception:
        await _safe_rollback(db_session)
        logger.exception("重置负面关系失败")
        await reset_negative_relation.finish("重置失败，数据库未修改，请查看日志。")

    if result.affected_count == 0:
        await reset_negative_relation.finish("没有需要重置的负好感度关系。")
    await reset_negative_relation.finish(
        f"已重置 {result.affected_count} 条负好感度关系。\n"
        f"备份文件：{result.backup_path}"
    )


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
    await db_session.commit()
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
    await db_session.commit()
    # Even if no words, return an empty wordcloud (original group_frequency didn't check emptiness)
    if not words:
        await group_frequency.finish("在指定时间内，没有消息可统计")

    image_bytes = await asyncio.to_thread(_build_wordcloud_image, words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


@scheduler.scheduled_job(
    "interval", minutes=60, max_instances=1, coalesce=True, id="vectorize_chat"
)
async def vectorize_message_history():
    async with maintenance_gate.slot(wait=False) as admitted:
        if not admitted:
            logger.info("其他维护任务正在运行，跳过本轮会话向量化")
            return

        started_at = time.perf_counter()
        try:
            async with get_session() as discovery_session:
                result = await discovery_session.execute(
                    Select(ChatHistory.session_id.distinct())
                )
                session_ids = list(result.scalars().all())
                await discovery_session.commit()

            logger.info("开始向量化会话")
            for session_id in session_ids:
                try:
                    # A failed or cancelled session cannot poison subsequent
                    # sessions, and no session exists while this job waits for
                    # its maintenance slot.
                    async with get_session() as vector_session:
                        res = await process_and_vectorize_session_chats(
                            vector_session, session_id
                        )
                    if res:
                        logger.info(
                            f"向量化会话 {res['session_id']} 成功，共处理 {res['processed_groups']}/{res['total_groups']} 组"
                        )
                    else:
                        logger.info(f"{session_id} 无需向量化")
                except Exception as e:
                    print(traceback.format_exc())
                    logger.error(f"向量化会话 {session_id} 失败: {e}")
        finally:
            elapsed = time.perf_counter() - started_at
            logger.info(f"会话向量化任务结束，耗时 {elapsed:.2f}s")


def _read_image_data_uri(file_path: Path, media_file_path: str) -> str | None:
    if not file_path.is_file():
        return None
    file_data = file_path.read_bytes()
    encoded_string = base64.b64encode(file_data).decode("utf-8")
    ext = media_file_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    if ext == "gif":
        mime = "image/gif"
    return f"data:{mime};base64,{encoded_string}"


async def _mark_media_vectorized(
    media_id: int,
    description: str | None = None,
    *,
    embedding_version: int | None = None,
) -> bool:
    async with get_session() as db_session:
        media = await db_session.get(MediaStorage, media_id)
        if media is None:
            await db_session.commit()
            return False
        if description is not None:
            media.description = description
        if embedding_version is not None:
            media.embedding_version = embedding_version
        media.vectorized = True
        db_session.add(media)
        await db_session.commit()
        return True


MEDIA_TAGGING_PROMPT = """
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


async def _process_media_vectorization(media_id: int) -> str:
    """处理单张图片；网络请求期间不占用 SQL 连接。"""
    async with get_session() as db_session:
        media = await db_session.get(MediaStorage, media_id)
        if media is None:
            await db_session.commit()
            return "skipped"
        media_file_path = media.file_path
        existing_description = media.description
        needs_reindex = (
            media.vectorized
            and media.embedding_version < MEDIA_EMBEDDING_VERSION
            and existing_description != "[图片]"
        )
        await db_session.commit()

    try:
        file_path = pic_dir / media_file_path
        try:
            img_data_uri = await asyncio.to_thread(
                _read_image_data_uri, file_path, media_file_path
            )
        except Exception as e:
            logger.error(f"读取图片失败 {media_id}: {e}")
            return "failed"

        if img_data_uri is None:
            logger.warning(f"文件不存在: {file_path}")
            # 新图片不再反复尝试；旧向量则保留旧版本，方便文件恢复后重试。
            if not needs_reindex:
                await _mark_media_vectorized(media_id)
                return "skipped"
            return "failed"

        if needs_reindex:
            try:
                inserted = await DB.insert_media(
                    media_id,
                    img_data_uri,
                    existing_description,
                )
                if not inserted:
                    logger.warning(f"旧表情包向量重建失败，等待下轮重试: {media_id}")
                    return "failed"
                await _mark_media_vectorized(
                    media_id,
                    embedding_version=MEDIA_EMBEDDING_VERSION,
                )
                logger.info(f"旧表情包向量重建成功 {media_id}: {existing_description}")
                return "indexed"
            except Exception as e:
                logger.error(f"旧表情包向量重建异常 {media_id}: {e}")
                return "failed"

        try:
            response = await get_tagging_model().ainvoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": MEDIA_TAGGING_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_data_uri},
                            },
                        ]
                    )
                ]
            )

            if isinstance(response.content, list):
                logger.warning(f"图片标注返回了非文本内容，等待下轮重试: {media_id}")
                return "failed"
            content = response.content.strip()
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")

            res_json = json.loads(content)
            is_meme = res_json.get("is_meme", False)
            description = str(res_json.get("description", "")).strip()

        except Exception as e:
            err_str = str(e)
            # 400 错误（图片尺寸/格式非法、内容违规）不可重试，标记跳过。
            if "Error code: 400" in err_str:
                if "data_inspection_failed" in err_str:
                    logger.warning(f"图片 {media_id} 内容违规，跳过向量化")
                else:
                    logger.warning(f"图片 {media_id} 请求非法（400），跳过向量化: {e}")
                await _mark_media_vectorized(media_id)
                return "skipped"
            logger.error(f"模型识别图片失败 {media_id}: {e}")
            return "failed"

        if not is_meme:
            logger.info(f"图片 {media_id} 被判定为非表情包(杂图)，跳过入库")
            await _mark_media_vectorized(media_id)
            return "skipped"
        if not description:
            logger.warning(f"图片 {media_id} 缺少描述，等待下轮重试")
            return "failed"

        try:
            inserted = await DB.insert_media(media_id, img_data_uri, description)
            if not inserted:
                logger.warning(f"表情包向量生成失败，保留待重试状态: {media_id}")
                return "failed"

            await _mark_media_vectorized(
                media_id,
                description,
                embedding_version=MEDIA_EMBEDDING_VERSION,
            )
            logger.info(f"表情包入库成功 {media_id}: {description}")
            return "indexed"

        except Exception as e:
            logger.error(f"向量化插入失败 {media_id}: {e}")
            return "failed"

    except Exception as e:
        logger.error(f"处理媒体异常 {media_id}: {e}")
        return "failed"


async def _vectorize_media_impl():
    """
    并发处理新图片与旧向量。每个 worker 使用独立 SQL 会话，且网络请求
    期间不持有数据库连接。
    """
    batch_size = plugin_config.media_vectorize_batch_size
    min_references = plugin_config.media_vectorize_min_references
    concurrency = plugin_config.media_vectorize_concurrency

    async with get_session() as db_session:
        # 新图片和旧向量分别取一批，避免大量新图导致历史重建一直饥饿。
        pending_res = await db_session.execute(
            Select(MediaStorage.media_id)
            .where(
                MediaStorage.references >= min_references,
                MediaStorage.vectorized.is_(False),
            )
            .order_by(MediaStorage.media_id)
            .limit(batch_size)
        )
        pending_ids = list(pending_res.scalars().all())

        outdated_res = await db_session.execute(
            Select(MediaStorage.media_id)
            .where(
                MediaStorage.references >= min_references,
                MediaStorage.vectorized.is_(True),
                MediaStorage.description != "[图片]",
                MediaStorage.embedding_version < MEDIA_EMBEDDING_VERSION,
            )
            .order_by(MediaStorage.media_id)
            .limit(batch_size)
        )
        outdated_ids = list(outdated_res.scalars().all())
        await db_session.commit()

    media_ids = list(dict.fromkeys(pending_ids + outdated_ids))
    logger.info(
        f"本轮待处理新图片: {len(pending_ids)}，"
        f"待重建旧表情包: {len(outdated_ids)}，并发数: {concurrency}"
    )
    if not media_ids:
        return

    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(media_id: int) -> str:
        async with semaphore:
            return await _process_media_vectorization(media_id)

    results = await asyncio.gather(*(process_one(media_id) for media_id in media_ids))
    logger.info(
        f"本轮图片处理完成：成功 {results.count('indexed')}，"
        f"跳过 {results.count('skipped')}，失败待重试 {results.count('failed')}"
    )


@scheduler.scheduled_job(
    "interval", minutes=10, max_instances=1, coalesce=True, id="vectorize_media"
)
async def vectorize_media():
    async with maintenance_gate.slot(wait=False) as admitted:
        if not admitted:
            logger.info("其他维护任务正在运行，跳过本轮媒体向量化")
            return

        started_at = time.perf_counter()
        try:
            await _vectorize_media_impl()
        finally:
            elapsed = time.perf_counter() - started_at
            logger.info(f"媒体向量化任务结束，耗时 {elapsed:.2f}s")


def _batch_delete_files(files: list[Path], label: str) -> int:
    deleted = 0
    for file_path in files:
        try:
            file_path.unlink(missing_ok=True)
            deleted += 1
            logger.debug(f"删除{label}文件: {file_path}")
        except Exception as e:
            logger.error(f"删除{label}文件失败 {file_path}: {e}")
    return deleted


def _delete_orphaned_files(
    directory: Path,
    known_files: set[str],
    grace_period: datetime.timedelta,
) -> tuple[int, int]:
    """Scan, stat and delete orphaned files entirely outside the event loop."""
    orphaned_count = 0
    deleted_count = 0
    cutoff_timestamp = time.time() - max(grace_period.total_seconds(), 0)

    for file_path in directory.iterdir():
        try:
            if not file_path.is_file() or file_path.name in known_files:
                continue
            # A file is written before its MediaStorage row is committed.  The
            # grace period prevents cleanup from racing that in-flight insert.
            if file_path.stat().st_mtime > cutoff_timestamp:
                continue
            orphaned_count += 1
            file_path.unlink(missing_ok=True)
            deleted_count += 1
            logger.debug(f"删除孤立文件: {file_path.name}")
        except Exception as e:
            logger.error(f"删除孤立文件失败 {file_path.name}: {e}")

    return orphaned_count, deleted_count


@scheduler.scheduled_job(
    "interval", minutes=35, max_instances=1, coalesce=True, id="clear_cache"
)
async def clear_cache_pic():
    async with maintenance_gate.slot(wait=False) as admitted:
        if not admitted:
            logger.info("其他维护任务正在运行，跳过本轮媒体清理")
            return

        started_at = time.perf_counter()
        logger.info("开始清理过期媒体和孤立文件")
        try:
            # Delete database rows in a short transaction.  Files are removed
            # only after the connection has been returned to the pool.
            media_files: list[Path] = []
            async with get_session() as cleanup_session:
                result = await cleanup_session.execute(
                    Select(MediaStorage).where(
                        MediaStorage.references < 3,
                        MediaStorage.created_at
                        < datetime.datetime.now() - datetime.timedelta(days=30),
                    )
                )
                medias = list(result.scalars().all())
                media_files = [pic_dir / media.file_path for media in medias]
                for media in medias:
                    await cleanup_session.delete(media)
                await cleanup_session.commit()

            if media_files:
                deleted_files = await asyncio.to_thread(
                    _batch_delete_files, media_files, "过期媒体"
                )
                logger.info(
                    f"成功清理 {len(media_files)} 个过期媒体记录"
                    f"（{deleted_files} 个文件）"
                )

            # Materialize the known filenames and close the database session
            # before scanning/stat'ing the directory.
            async with get_session() as discovery_session:
                known_result = await discovery_session.execute(
                    Select(MediaStorage.file_path)
                )
                known_files = {str(row[0]) for row in known_result.all()}
                await discovery_session.commit()

            orphaned_count, deleted_count = await asyncio.to_thread(
                _delete_orphaned_files,
                pic_dir,
                known_files,
                datetime.timedelta(minutes=10),
            )
            if orphaned_count:
                logger.info(
                    f"成功清理 {deleted_count}/{orphaned_count} 个孤立文件"
                )
        finally:
            elapsed = time.perf_counter() - started_at
            logger.info(f"媒体清理任务结束，耗时 {elapsed:.2f}s")
# 群档案由 agent 工具按需维护，不再注册定时任务。
