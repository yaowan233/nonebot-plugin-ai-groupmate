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
from dataclasses import replace
from urllib.parse import urlsplit

import jieba
from nonebot import logger, require, get_bots, on_notice, get_driver, on_command, on_message, get_plugin_config
from wordcloud import WordCloud
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, get_plugin, inherit_supported_adapters
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
    File,
    Image,
    Target,
    UniMessage,  # noqa: F401 - public compatibility import
    image_fetch,
    get_message_id,
)
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import Audio, Video, UniMsg, Reference

from . import message_index as _message_index  # noqa: F401 - registers ORM listeners
from .agent import (
    _parse_msg_meta,
    _detect_repeat_chain,
    check_if_should_reply,
    get_jargon_review_model,
    choice_response_strategy,
    get_jargon_discovery_model,
)
from .model import ChatHistory, MediaStorage, ChatHistorySchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .webui import register_usage_webui
from .config import Config, create_tagging_llm, agent_run_timeout_seconds
from .memory import (
    DB,
    MEDIA_EMBEDDING_VERSION as MEDIA_EMBEDDING_VERSION,
    EmbeddingProviderUnavailableError,
    CollectionEmbeddingConfigMismatchError,
)
from .concurrency import agent_run_gate, maintenance_gate, background_image_gate
from .observation import RECENT_PARTICIPATION_SECONDS, ObservationSchedule
from .drawing_poll import poll_drawing_jobs, stop_drawing_jobs, start_drawing_jobs
from .shared_cards import MAX_CARD_TEXT, save_card_links, parse_shared_segment
from .media_message import format_media_markers
from .message_trace import trace_id
from .participation import participation, delivery_context, observation_guard, confirmed_turn_delivery
from .reply_request import ReplyRequest, GroupReplyState, merge_reply_requests, current_transport_message_id
from .agent.delivery import OutboundIdentity, send_and_finalize
from .agent.reaction import is_onebot_context, requested_reaction_emoji
from .group_affinity import note_group_activity, load_group_affinities
from .runtime_config import (
    RESTART_REQUIRED_FIELDS,
    get_runtime_config,
    mark_restart_fields_applied,
    load_runtime_config_overrides,
)
from .forward_message import format_forward_reference_markers
from .group_api_relay import delete_expired_pending_group_configs
from .message_rollout import message_pilot_enabled
from .scheduled_tasks import poll_scheduled_tasks, stop_scheduled_tasks
from .agent.chat_files import chat_file_source_key
from .message_batching import InputBatcher, current_input_batch
from .message_mentions import mentioned_user_ids, format_member_mention, event_mentioned_user_ids, current_mentioned_user_ids, current_normalized_message
from .message_metadata import BOT_ECHO_LOOKBACK, platform_message_ids, is_reliable_platform_id
from .agent.reply_tools import create_reply_tool
from .delivery_recovery import prune_receipts, recover_pending
from .group_daily_quota import (
    GroupDailyQuotaStatus,
    consume_group_daily_quota,
    get_group_daily_quota_status,
    build_quota_exhausted_message,
    consume_private_user_daily_quota,
    get_private_user_daily_quota_status,
    build_private_quota_exhausted_message,
)
from .agent.turn_context import TurnHistoryResolver, bind_turn_history, build_turn_context, is_one_to_one_exchange
from .brain.observations import record_message, configured_space, poll_observations, prune_observations, record_interaction
from .group_model_config import (
    LOCAL_ENCRYPTION_KEY_FILENAME,
    LocalSecretCipher,
    LocalEncryptionKeyError,
    has_group_model_config,
    has_private_model_config,
    load_group_model_configs,
    load_private_model_configs,
    resolve_group_reply_probability,
    load_or_create_local_encryption_key,
)
from .structured_message import adapter_name, image_source_id, normalize_parts, record_inbound_safely, bind_image_resource_safely
from .agent.file_workspace import FileWorkspace, clean_filename
from .relation_maintenance import count_negative_relations, reset_negative_relations
from .agent.chat_file_inbox import cleanup_chat_file_inbox, register_chat_file_markers, register_chat_file_resources
from .agent.explicit_memory import cleanup_explicit_memories
from .group_reply_scheduler import GroupReplyScheduler
from .agent.command_delivery import send_command_image
from .agent.web_image_search import is_explicit_web_image_search_request
from .history_index_backfill import backfill_one, has_pending_backfill
from .agent.structured_context import load_structured_context_safely


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
RAG_IDLE_VECTORIZE_SECONDS = 90.0
RAG_MAINTENANCE_RETRY_SECONDS = 30.0
RAG_BACKFILL_START_DELAY_SECONDS = 30.0
_rag_vectorization_tasks: dict[str, asyncio.Task[None]] = {}
_rag_last_activity: dict[str, float] = {}
_rag_backfill_task: asyncio.Task[None] | None = None
with open(Path(__file__).parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]


@lru_cache
def get_tagging_model():
    return create_tagging_llm(plugin_config)


def _refresh_runtime_resources(_changed_fields: set[str]) -> None:
    if "proactive_observation_enabled" in _changed_fields and not plugin_config.proactive_observation_enabled:
        _observation_schedule.clear()
    get_tagging_model.cache_clear()
    from .agent import refresh_runtime_resources as refresh_agent_resources
    from .group_memory import get_summary_model

    refresh_agent_resources()
    get_summary_model.cache_clear()


register_usage_webui(
    plugin_config,
    on_config_change=_refresh_runtime_resources,
)

# Register the group-owned model configuration commands. The public Mayumi
# relay is the default; deployments can still override it with their own URL.
from . import group_api_commands as group_api_commands


@get_driver().on_startup
async def _load_group_affinity_state() -> None:
    try:
        async with get_session() as db_session:
            loaded = await load_group_affinities(db_session)
        logger.info(f"已加载 {loaded} 个群相处感受")
    except Exception:
        logger.exception("加载群相处感受失败，暂按中性参与；请确认已执行 nb orm upgrade")


@get_driver().on_startup
async def _load_webui_runtime_config() -> None:
    try:
        async with get_session() as db_session:
            changed_fields = await load_runtime_config_overrides(db_session)
            group_api_management_enabled = bool(
                plugin_config.group_api_relay_url
                or plugin_config.usage_webui_token
            )
            if group_api_management_enabled and not plugin_config.group_api_local_encryption_key:
                try:
                    plugin_config.group_api_local_encryption_key = load_or_create_local_encryption_key(plugin_data_dir / LOCAL_ENCRYPTION_KEY_FILENAME)
                    logger.info("已生成或加载群 API 本地加密密钥")
                except LocalEncryptionKeyError:
                    logger.exception("无法初始化群 API 本地加密密钥；群 API 自助配置和管理页已禁用")
            try:
                expired_tickets = await delete_expired_pending_group_configs(db_session)
                if expired_tickets:
                    logger.info(f"已清理 {expired_tickets} 个过期的群 API 配置单")
            except Exception:
                # Ticket cleanup is maintenance only and must never prevent the
                # plugin's existing runtime configuration from loading.
                logger.warning("清理过期群 API 配置单失败，将在下次启动时重试")
            if plugin_config.group_api_local_encryption_key:
                try:
                    loaded_group_configs = await load_group_model_configs(
                        db_session,
                        LocalSecretCipher(plugin_config.group_api_local_encryption_key),
                        plugin_config,
                    )
                    if loaded_group_configs:
                        logger.info(f"已加载 {loaded_group_configs} 个群的独立主模型配置")
                    loaded_private_configs = await load_private_model_configs(
                        db_session,
                        LocalSecretCipher(plugin_config.group_api_local_encryption_key),
                        plugin_config,
                    )
                    if loaded_private_configs:
                        logger.info(f"已加载 {loaded_private_configs} 个用户的私聊独立主模型配置")
                except Exception:
                    logger.exception("加载群级主模型配置失败；相关群暂时使用全局配置")
        _refresh_runtime_resources(changed_fields)
        has_restart_required_changes = bool(changed_fields & RESTART_REQUIRED_FIELDS)
        if has_restart_required_changes:
            await DB.reconfigure()
            # reconfigure 已经让连接资源使用当前运行配置。集合兼容性是独立
            # 状态，不能继续借用 restart baseline 表示，否则校验失败后把
            # WebUI 配置恢复为旧值会错误清除“等待重启”。
            mark_restart_fields_applied()
        if DB.enabled:
            try:
                await DB.check_collections()
            except CollectionEmbeddingConfigMismatchError:
                logger.error("启动时 Qdrant 集合校验失败，相关向量操作已拒绝；请根据上方日志中的当前配置和 collection 配置人工重建向量")
            except EmbeddingProviderUnavailableError:
                logger.warning("启动时 Embedding API 暂时不可用，文本向量操作将在首次使用时重试")
            except Exception:
                logger.exception("启动时 Qdrant 集合校验暂时失败，将在首次向量操作时重试")
            _start_rag_backfill_task()
        if changed_fields:
            logger.info(f"已加载 WebUI 配置覆盖项，变更字段数={len(changed_fields)}")
    except Exception:
        logger.exception("加载 WebUI 配置失败，继续使用环境变量；请确认已执行 nb orm upgrade")


@scheduler.scheduled_job("interval", seconds=1, max_instances=1, coalesce=True, id="ai_groupmate_poll_scheduled_tasks")
async def _poll_scheduled_tasks() -> None:
    try:
        await poll_scheduled_tasks()
    except Exception:
        logger.exception("扫描数据库定时任务失败，将在下次扫描重试；请确认已执行 nb orm upgrade")


@get_driver().on_startup
async def _start_drawing_jobs() -> None:
    start_drawing_jobs()


@scheduler.scheduled_job("interval", seconds=2, max_instances=1, coalesce=True, id="ai_groupmate_poll_drawing_jobs")
async def _poll_drawing_jobs() -> None:
    try:
        await poll_drawing_jobs(plugin_data_dir / "agent_files", get_runtime_config())
    except Exception:
        logger.exception("扫描绘图任务失败，将在下次扫描重试")


@scheduler.scheduled_job("interval", seconds=30, max_instances=1, coalesce=True, id="ai_groupmate_recover_deliveries")
async def _recover_deliveries() -> None:
    try:
        await recover_pending(plugin_data_dir / "delivery_recovery", plugin_data_dir / "agent_files")
    except Exception:
        logger.exception("恢复发送记录失败，将在下次扫描重试")


@scheduler.scheduled_job("interval", hours=24, max_instances=1, coalesce=True, id="ai_groupmate_prune_delivery_receipts")
async def _prune_delivery_receipts() -> None:
    try:
        await prune_receipts(plugin_data_dir / "delivery_recovery")
    except Exception:
        logger.exception("清理发送记录恢复标记失败")


@scheduler.scheduled_job("interval", seconds=5, max_instances=1, coalesce=True, id="ai_groupmate_backfill_history_index")
async def _backfill_history_index() -> None:
    try:
        groups = plugin_config.message_refactor_pilot_groups
        if not groups and not plugin_config.message_refactor_pilot_private:
            return
        allowed = None if "*" in groups or plugin_config.message_refactor_pilot_private else list(groups)
        # Do not occupy the shared maintenance slot when there is no work.
        async with get_session() as db_session:
            pending = await has_pending_backfill(db_session, allowed_sessions=allowed)
        if not pending:
            return
        async with maintenance_gate.slot(wait=False) as granted:
            if granted:
                async with get_session() as db_session:
                    await backfill_one(db_session, allowed_sessions=allowed)
                    await db_session.commit()
    except Exception:
        logger.exception("历史消息索引回填失败，将稍后继续")


@scheduler.scheduled_job("interval", minutes=5, max_instances=1, coalesce=True, id="ai_groupmate_expire_explicit_memories")
async def _expire_explicit_memories() -> None:
    try:
        async with get_session() as db_session:
            await cleanup_explicit_memories(db_session)
            await db_session.commit()
    except Exception:
        logger.exception("清理到期明确记忆失败，请确认已执行 nb orm upgrade")


@scheduler.scheduled_job("interval", minutes=30, max_instances=1, coalesce=True, id="ai_groupmate_expire_agent_files")
async def _expire_agent_files() -> None:
    from .agent.file_workspace import cleanup_workspace

    try:
        await asyncio.to_thread(cleanup_workspace, plugin_data_dir / "agent_files")
        await asyncio.to_thread(cleanup_chat_file_inbox, plugin_data_dir / "agent_files")
    except Exception:
        logger.exception("清理到期 Agent 文件失败")


@get_driver().on_shutdown
async def _close_scheduled_tasks() -> None:
    await stop_scheduled_tasks()
    await stop_drawing_jobs()


@get_driver().on_shutdown
async def _close_vector_resources() -> None:
    tasks = list(_rag_vectorization_tasks.values())
    if _rag_backfill_task is not None:
        tasks.append(_rag_backfill_task)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await DB.close()


_observation_schedule: ObservationSchedule[ReplyRequest] = ObservationSchedule()


@scheduler.scheduled_job("interval", minutes=1, max_instances=1, coalesce=True, id="ai_groupmate_cross_group_observe")
async def _poll_cross_group_observations() -> None:
    if not plugin_config.cross_group_spaces:
        return
    from .group_memory import get_summary_model

    connected = {str(bot_id): adapter_name(bot) for bot_id, bot in get_bots().items()}
    try:
        async with maintenance_gate.slot():
            await poll_observations(
                spaces=plugin_config.cross_group_spaces, connected=connected,
                paused=lambda bot_id, scene: time.monotonic() < participation.state((bot_id, scene)).quiet_until,
                model_factory=get_summary_model,
                now=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            )
    except Exception:
        logger.exception("跨群观察调度失败")


@scheduler.scheduled_job("interval", seconds=5, max_instances=1, coalesce=True, id="ai_groupmate_cross_group_tasks")
async def _poll_cross_group_tasks() -> None:
    if not plugin_config.cross_group_spaces:
        return
    from .agent import get_chat_model
    from .brain.poll import poll
    from .agent.outbound_plan import compose_segments

    connected_bots = get_bots()
    connected = {str(bot_id): adapter_name(bot) for bot_id, bot in connected_bots.items()}

    async def send(platform: str, bot_id: str, scene_id: str, text: str, reply_to: str | None) -> str:
        bot = get_bots().get(bot_id)
        if bot is None or adapter_name(bot) != platform:
            raise RuntimeError("Bot is unavailable")
        message = compose_segments([UniMessage.text(text)], reply_to)
        target = Target(id=scene_id, private=False, self_id=bot_id)
        identity = OutboundIdentity(platform, "group", bot_id, plugin_config.bot_name)

        def record(message_id: str) -> list[ChatHistory]:
            return [ChatHistory(session_id=scene_id, user_id=plugin_config.bot_name, user_name=plugin_config.bot_name,
                                content_type="bot", content=f"id: {message_id}\n{text}")]

        receipt = await send_and_finalize(None, scene_id, message, send_options={"target": target, "bot": bot},
                                           record=record, outbound_identity=identity, session_factory=get_session)
        return receipt.message_id

    try:
        await poll(spaces=lambda: plugin_config.cross_group_spaces, connected=connected, model_factory=get_chat_model,
                   sender=send, target_names=plugin_config.cross_group_scene_names)
    except Exception:
        logger.exception("跨群任务调度失败")


@scheduler.scheduled_job("interval", hours=1, max_instances=1, coalesce=True, id="ai_groupmate_prune_cross_group_observations")
async def _prune_cross_group_observations() -> None:
    try:
        async with get_session() as db:
            await prune_observations(db, now=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None))
            await db.commit()
    except Exception:
        logger.exception("跨群观察数据清理失败")


def _observation_enabled(key: tuple[str, str]) -> bool:
    watch = _observation_schedule.watches.get(key)
    return (
        plugin_config.proactive_observation_enabled
        and (watch is None or not watch.paused)
        and key[0] in get_bots()
        and resolve_group_reply_probability(key[1]) > 0
    )


def _observation_allowed(key: tuple[str, str]) -> bool:
    return (
        _observation_enabled(key)
        and participation.clock() - participation.state(key).last_sent >= RECENT_PARTICIPATION_SECONDS
        and participation.clock() - participation.state(key).last_check >= 300
        and participation.can_reply(key)
    )


@scheduler.scheduled_job("interval", minutes=1, max_instances=1, coalesce=True, id="ai_groupmate_observe_groups")
async def _occasionally_observe_groups() -> None:
    if not plugin_config.proactive_observation_enabled:
        _observation_schedule.clear()
        return
    request = _observation_schedule.take_due(_observation_allowed)
    if request is not None:
        try:
            await _queue_group_reply_request(request.session.scene.id, replace(
                request, request_id=f"observation:{request.session.scene.id}:{time.monotonic()}",
                is_tome=False, is_continuous=False, reply_to_id=None,
                proactive_meme_only=False, meme_required=False, meme_send_count=1,
                proactive_reaction_only=False, reaction_required=False, repeat_text=None,
                proactive_observation=True, created_monotonic=time.monotonic(),
                message_id=None, input_batch=(), mentioned_user_ids=(), normalized_parts=None,
            ))
        except Exception:
            logger.exception("主动查看群聊未能入队，等待下次机会")


# 每个群有限并发处理明确 @；非定向回复只保留最新一条，避免高峰期刷屏。
_reply_scheduler = GroupReplyScheduler(lambda request: _execute_reply_request(request),
                                       lambda: plugin_config.agent_max_concurrency_per_group)
_group_reply_states: dict[str, GroupReplyState] = _reply_scheduler.states
_input_batcher: InputBatcher[ReplyRequest] = InputBatcher()
_continuous_conversation_until: dict[tuple[str, str, str], datetime.datetime] = {}
_explicit_conversation_until: dict[tuple[str, str, str], datetime.datetime] = {}


@get_driver().on_shutdown
async def _stop_input_learning_tasks() -> None:
    from .agent.style_learning import stop_style_learning
    from .agent.jargon_learning import stop_jargon_learning

    await _input_batcher.close()
    await stop_jargon_learning()
    await stop_style_learning()


def _input_batch_key(bot_id: str, group_id: str, user_id: str, reply_id: str | None) -> tuple[str, ...]:
    key = bot_id, group_id, user_id, reply_id or ""
    if not reply_id and key not in _input_batcher.pending:
        candidates = [item for item in _input_batcher.pending if item[:3] == key[:3]]
        if len(candidates) == 1:
            return candidates[0]
    return key


async def _submit_reply_input(group_id: str, request: ReplyRequest, message_id: str, text: str) -> None:
    if (request.session.scene.type != SceneType.GROUP or request.proactive_meme_only
            or request.proactive_reaction_only or request.repeat_text is not None or request.proactive_observation):
        await _queue_group_reply_request(group_id, request)
        return

    async def deliver(ready: ReplyRequest, messages: tuple[tuple[str, str], ...]) -> None:
        logger.info(f"[InputBatch] group={group_id} messages={len(messages)} ids={[mid for mid, _ in messages]}")
        await _queue_group_reply_request(group_id, replace(ready, input_batch=messages))

    await _input_batcher.submit(
        _input_batch_key(str(request.bot.self_id), group_id, request.user_id, request.reply_to_id), request,
        message_id=message_id if is_reliable_platform_id(message_id) else "", text=text, delay=plugin_config.group_message_debounce_seconds,
        max_wait=plugin_config.group_message_debounce_max_seconds, merge=merge_reply_requests, deliver=deliver,
    )

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


def _shared_balancer_enabled() -> bool:
    """Only share a conversation when the optional bot pool actually routes sends."""
    balancer = get_plugin("nonebot_plugin_bot_load_balancer")
    if balancer is None:
        return False
    try:
        return bool(get_plugin_config(balancer.module.Config).bot_load_balancer.enabled)
    except (AttributeError, TypeError, ValueError):
        return False


def _continuous_key(session_id: str, user_id: str, bot_id: str) -> tuple[str, str, str]:
    return (session_id, user_id, "__balanced_pool__" if _shared_balancer_enabled() else bot_id)


def _is_continuous_conversation(session_id: str, user_id: str, bot_id: str) -> bool:
    key = _continuous_key(session_id, user_id, bot_id)
    expires_at = _continuous_conversation_until.get(key)
    if not expires_at:
        return False
    if datetime.datetime.now() > expires_at:
        _continuous_conversation_until.pop(key, None)
        _explicit_conversation_until.pop(key, None)
        return False
    return True


def _refresh_continuous_conversation(session_id: str, user_id: str, bot_id: str) -> None:
    ttl = _continuous_conversation_ttl()
    if ttl <= datetime.timedelta(0):
        return
    key = _continuous_key(session_id, user_id, bot_id)
    expires_at = datetime.datetime.now() + ttl
    _continuous_conversation_until[key] = expires_at
    if key in _explicit_conversation_until:
        _explicit_conversation_until[key] = expires_at


def _is_explicit_conversation_request(text: str) -> bool:
    return bool(re.search(r"(?:连续|一直|持续|接下来).{0,16}(?:聊|对话)", text))


def _is_explicit_conversation(session_id: str, user_id: str, bot_id: str) -> bool:
    key = _continuous_key(session_id, user_id, bot_id)
    until = _explicit_conversation_until.get(key)
    if until is None:
        return False
    if datetime.datetime.now() > until:
        _explicit_conversation_until.pop(key, None)
        return False
    return True


def _is_explicit_disengagement(text: str, bot_name: str) -> bool:
    """Recognize direct, complete requests to leave this conversation alone."""
    normalized = re.sub(r"\s+", "", text).casefold().strip("，。！？,!?;；")
    name = re.sub(r"\s+", "", bot_name).casefold()
    for prefix in (f"@{name}", name):
        if name and normalized.startswith(prefix):
            normalized = normalized[len(prefix):].lstrip("，,：:")
            break
    # Full matches deliberately leave quotes, stories and playful remarks alone.
    return any(re.fullmatch(pattern, normalized) for pattern in (
        r"(?:我)?(?:不是|没(?:有)?)(?:在)?(?:和|跟|对)你(?:说话|聊天|交流)(?:呢|呐|啊|了|啦)?",
        r"(?:我)?不是(?:在)?问你(?:呢|呐|啊|了|啦)?",
        r"(?:你|bot|ai|机器人)?(?:现在|先|暂时)?闭嘴(?:了|啦|啊|好吗|可以吗)?",
        r"(?:我)?(?:正在|在)?(?:和|跟)(?:另一个|另个|其他|别的)(?:bot|ai|机器人)(?:说话|聊天|交流)(?:呢|呐|啊|了|啦)?(?:[，,](?:别搞混了|别弄混了|别插话|不要插话))?",
        r"(?:你|bot|ai|机器人)?(?:现在|先|暂时)?(?:别|不要|不用)(?:插话|讲话|说话|回复|回话|接话|出声)(?:了|啦|啊|好吗|可以吗)?",
    ))


def _sample_proactive_reply_modes(
    *,
    addressed: bool,
    continuous: bool,
    command_like: bool,
    has_text: bool,
    is_group: bool,
    reaction_supported: bool,
    reply_probability: float,
    participation_key: tuple[str, str] | None = None,
) -> tuple[bool, bool, bool]:
    """返回（普通回复、reaction 专用、图片表情专用）。

    群聊先在本地以低概率独立抽样；不单独采样 reaction 或图片表情，
    避免对每条普通消息额外调用 Gatekeeper 和主模型。
    """
    if addressed or continuous:
        return False, False, False
    if participation_key is not None:
        if command_like or not has_text or not is_group:
            return False, False, False
        return participation.candidate(participation_key, reply_probability), False, False
    random_reply = random.random() < reply_probability
    return random_reply, False, False


def _is_explicit_reaction_request(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if _is_explicit_meme_request(normalized):
        return False
    if re.search(r"(?:不要|别|不用|停止)[^，。！？?\n]*(?:表情|reaction)", normalized):
        return False
    if requested_reaction_emoji(normalized) is not None:
        return True
    return bool(
        re.search(
            r"(?:回应|回复|回|点|加|来|贴|上|给)(?:一个|个|一下|点)?(?:(?:消息)?小?表情(?!包)|reaction)"
            r"|(?:(?:消息)?小?表情(?!包)|reaction)(?:回应|回复|一下)"
            r"|点(?:个|一下)?(?:赞|爱心)",
            normalized,
        )
    )


def _is_explicit_meme_request(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if re.search(r"(?:小|消息)表情(?!包)|reaction", normalized):
        return False
    # A generic image request leaves the image source to the agent. Only an
    # explicit meme request may force delivery from the local meme collection.
    if not re.search(r"表情|梗图|斗图", normalized):
        return False
    if is_explicit_web_image_search_request(normalized):
        return False
    # Online image requests must retain web search/send tools instead of
    # forcing the local meme retrieval workflow.
    if re.search(r"网上|联网|网络|搜图|搜[^，。！？?\n]*(?:图|照片|壁纸)", normalized):
        return False
    # Only explicit image-delivery requests may remove the normal reply tools.
    # Search requests and questions about images need the full conversation flow.
    if re.search(r"什么|为何|怎么|结果|来源|出处|不要|别|不用", normalized):
        return False
    return bool(
        re.search(
            r"(?:发|来|整|给)"
            r"(?:(?:\d{1,3}|[一二两三四五六七八九十]{1,3})(?:张|个)?"
            r"|一个|个|点|些|一下|几张|张)?(?:图片|图|表情)"
            r"|(?:图片|图|表情)(?:发|来|整)"
            r"(?:(?:\d{1,3}|[一二两三四五六七八九十]{1,3})(?:张|个)?"
            r"|一个|个|点|些|一下)?"
            r"|(?:图片|图|表情包)(?:呢|在哪)[？?]?($|\s)",
            normalized,
        )
        or re.search(
            r"(?:发|来|整|给)"
            r"(?:(?:\d{1,3}|[一二两三四五六七八九十]{1,3})(?:张|个)?"
            r"|一个|个|点|些|一下|几张|张)"
            r"[^，。！？?\n]{1,20}?(?:图片|图|表情)",
            normalized,
        )
    )


MAX_EXPLICIT_MEME_SEND_COUNT = 5
DEFAULT_MULTIPLE_MEME_SEND_COUNT = 3


def _parse_chinese_number(token: str) -> int | None:
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if token in digits:
        return digits[token]
    if token == "十":
        return 10
    if "十" not in token:
        return None
    tens, ones = token.split("十", 1)
    tens_value = digits.get(tens, 1) if tens else 1
    ones_value = digits.get(ones, 0) if ones else 0
    return tens_value * 10 + ones_value


def _get_explicit_meme_send_count(text: str) -> int:
    """解析用户明确要求的发图数量；模糊复数按 3 张处理，最终硬限制为 5。"""
    normalized = " ".join((text or "").strip().lower().split())
    if not _is_explicit_meme_request(normalized):
        return 1

    match = re.search(
        r"(\d{1,3}|[一二两三四五六七八九十]{1,3})\s*(?:张|个)(?=[^\d]|$)",
        normalized,
    )
    if match:
        token = match.group(1)
        requested = int(token) if token.isdigit() else _parse_chinese_number(token)
        if requested is not None:
            return max(1, min(requested, MAX_EXPLICIT_MEME_SEND_COUNT))

    if re.search(
        r"多发|多来|多整|发点|来点|整点|发些|来些|整些|几张|几个|多张|一组|一套",
        normalized,
    ):
        return DEFAULT_MULTIPLE_MEME_SEND_COUNT
    return 1


def _sample_repeat_reply(
    *,
    repeat_text: str | None,
    addressed: bool,
    continuous: bool,
    command_like: bool,
    is_group: bool,
) -> bool:
    """对已识别的群聊复读队形独立采样。"""
    if repeat_text is None or addressed or continuous or command_like or not is_group:
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
    context_since = datetime.datetime.now() - datetime.timedelta(minutes=context_minutes)
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
        rows = (await db_session.execute(Select(ChatHistory).where(ChatHistory.session_id == session_id).where(ChatHistory.created_at >= cutoff_time).order_by(ChatHistory.msg_id.desc()).limit(AGENT_HISTORY_LIMIT))).scalars().all()
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


async def _execute_reply_request(request: ReplyRequest) -> None:
    # Wait for a global Agent slot before opening a database session. This
    # bounds cross-group concurrency without consuming pool capacity while queued.
    slot_started = time.monotonic()
    async with agent_run_gate.slot():
        created = getattr(request, "created_monotonic", None)
        scene = getattr(getattr(request, "session", None), "scene", None)
        scene_id = getattr(scene, "id", None)
        if created is not None and scene_id is not None and message_pilot_enabled(
            plugin_config, str(scene_id),
            is_private=getattr(scene, "type", None) == SceneType.PRIVATE,
        ):
            logger.info(f"[MessageTrace] trace={trace_id(str(scene_id), request.request_id)} "
                        f"stage=worker_admitted wait_ms={(time.monotonic() - created) * 1000:.0f} "
                        f"global_slot_wait_ms={(time.monotonic() - slot_started) * 1000:.0f}")
        is_observation = bool(getattr(request, "proactive_observation", False))
        token = current_input_batch.set(() if is_observation else getattr(request, "input_batch", ()))
        message_token = current_transport_message_id.set(None if is_observation else getattr(request, "message_id", None))
        normalized = getattr(request, "normalized_parts", None)
        normalized_token = current_normalized_message.set(
            (request.event, normalized) if not is_observation and normalized is not None else None
        )
        try:
            mentions = () if is_observation else (getattr(request, "mentioned_user_ids", ()) or event_mentioned_user_ids(request.event))
            mentions_token = current_mentioned_user_ids.set(mentions)
            try:
                await handle_reply_request(request)
            finally:
                current_mentioned_user_ids.reset(mentions_token)
        finally:
            current_input_batch.reset(token)
            current_transport_message_id.reset(message_token)
            current_normalized_message.reset(normalized_token)


async def _queue_group_reply_request(group_id: str, request: ReplyRequest) -> bool:
    return await _reply_scheduler.enqueue(group_id, request)


async def _run_group_reply_worker(group_id: str) -> None:
    await _reply_scheduler.run_background(group_id)


def _is_file_upload_notice(bot: Bot, event: Event) -> bool:
    return (
        plugin_config.agent_file_tools_enabled
        and is_onebot_context(bot, event)
        and getattr(event, "notice_type", None) in {"group_upload", "offline_file", "friend_upload"}
    )


file_upload = on_notice(rule=_is_file_upload_notice, priority=999, block=False)


@file_upload.handle()
async def handle_file_upload(db_session: async_scoped_session, bot: Bot, event: Event):
    """OneBot group uploads arrive as notices rather than normal file messages."""
    if not _is_file_upload_notice(bot, event):
        return
    user_id = str(getattr(event, "user_id", "") or "")
    if not user_id or user_id in {str(bot.self_id), *map(str, get_bots())}:
        return
    native_file = getattr(event, "file", None)
    data = native_file if isinstance(native_file, dict) else getattr(native_file, "model_dump", lambda: {})()
    if not isinstance(data, dict):
        return
    file_id = data.get("id") or data.get("file_id")
    if not file_id:
        return
    group_id = str(getattr(event, "group_id", "") or "") or None
    session_id = group_id or user_id
    file = File(id=str(file_id), name=str(data.get("name") or data.get("file_name") or "file.bin"),
                url=str(data["url"]) if data.get("url") else None)
    source_key = chat_file_source_key(file, "")
    content_prefix = f"id: qq-file:{source_key}\n"
    workspace = FileWorkspace(plugin_data_dir / "agent_files", session_id=session_id, user_id=user_id,
                              bot_id=str(bot.self_id), is_private=group_id is None)
    marker = await asyncio.to_thread(register_chat_file_markers, [file], workspace,
                                     message_id=source_key, group_id=group_id, metadata=data)
    # No model call on an unsolicited upload notice. A later @/private request
    # sees the attachment marker and retrieves the current sender's workspace.
    async with _get_dedup_lock(session_id):
        existing = (await db_session.execute(Select(ChatHistory).where(
            ChatHistory.session_id == session_id, ChatHistory.user_id == user_id,
            ChatHistory.created_at >= datetime.datetime.now() - datetime.timedelta(hours=24),
            ChatHistory.content.startswith(content_prefix, autoescape=True),
        ).limit(1))).scalar_one_or_none()
        if existing is None:
            db_session.add(ChatHistory(session_id=session_id, user_id=user_id, user_name=user_id,
                                       content_type="text", content=content_prefix + marker))
        elif existing.content != content_prefix + marker:
            existing.content = content_prefix + marker
        await db_session.commit()


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
        logger.debug(f"消息明确发给其他 Bot {addressed_bot_id}，当前 Bot {bot.self_id} 跳过处理")
        return

    try:
        try:
            native_segments = event.get_message()
        except Exception:
            native_segments = msg
        if not native_segments or not any(getattr(segment, "type", "") not in {"reply", ""}
                                          for segment in native_segments):
            native_segments = msg
        normalized_parts = tuple(normalize_parts(event, native_segments))
    except Exception as error:
        logger.warning(f"解析结构化消息失败，继续使用适配器事件 error_type={type(error).__name__}")
        normalized_parts = None
        native_segments = msg

    bot_name = plugin_config.bot_name
    imgs = msg.include(Image)
    audios = list(msg.include(Audio))
    videos = list(msg.include(Video))
    forward_references = list(msg.include(Reference))
    files = list(msg.include(File)) if plugin_config.agent_file_tools_enabled and not sender_is_connected_bot else []
    # 第1行固定是本条消息的平台 ID 元数据，格式 "id: {id}"
    incoming_message_id = str(get_message_id()).strip()
    content_prefix = f"id: {incoming_message_id}\n"
    content = content_prefix
    to_me = addressed_bot_id == str(bot.self_id) or (addressed_bot_id is None and event.is_tome())
    is_text = False
    reply_id: str | None = None  # 记录回复 ID，稍后单独成行插入
    body = ""  # 正文部分单独拼接
    has_at_mention = False
    mention_names: dict[str, str] = {}
    shared_cards = []
    for i in msg:
        if len(shared_cards) < 3 and (card := parse_shared_segment(i)) is not None:
            shared_cards.append(card)
            is_text = True
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
            if name and str(i.target or "") and name != str(i.target):
                mention_names[str(i.target)] = str(name)
            body += format_member_mention(str(i.target or ""), name)
            is_text = True
        if i.type == "reply":
            reply_id = i.id
        if i.type == "text":
            body += i.text
            is_text = True
        if i.type == "image":
            body += "[图片]"
        if i.type in {"mface", "face"}:
            body += "[表情]"
            is_text = True

    # Some adapters retain native faces only on the original event, not in the
    # unified message. Keep a history anchor even when the message is face-only.
    if not is_text and not imgs:
        original_segments = getattr(event, "original_message", None) or ()
        if any(getattr(segment, "type", "") in {"face", "mface"} for segment in original_segments):
            body += "[表情]"
            is_text = True

    card_text = "\n".join(card.text for card in shared_cards)[:MAX_CARD_TEXT]
    if card_text:
        body += ("\n" if body and not body.endswith("\n") else "") + card_text
        try:
            scope = f"{'private' if session.scene.type == SceneType.PRIVATE else 'group'}:{session.scene.id}"
            await asyncio.to_thread(save_card_links, plugin_data_dir / "shared_card_links", scope, incoming_message_id,
                                    (url for card in shared_cards for url in card.urls))
        except OSError as error:
            logger.warning(f"保存分享卡片备用链接失败: error_type={type(error).__name__}")

    if forward_references:
        forward_text = format_forward_reference_markers(forward_references)
        if body and not body.endswith("\n"):
            body += "\n"
        body += forward_text
        is_text = True

    if audios or videos:
        media_text = format_media_markers(
            incoming_message_id,
            audio_count=len(audios),
            video_count=len(videos),
        )
        if body and not body.endswith("\n"):
            body += "\n"
        body += media_text
        is_text = True

    file_resource_ids: tuple[str | None, ...] = ()
    if files:
        workspace = FileWorkspace(plugin_data_dir / "agent_files", session_id=session.scene.id,
                                  user_id=session.user.id, bot_id=str(bot.self_id),
                                  is_private=session.scene.type == SceneType.PRIVATE)
        marker, file_resource_ids = await asyncio.to_thread(register_chat_file_resources, files, workspace,
                                                            message_id=incoming_message_id,
                                                            group_id=session.scene.id if session.scene.type == SceneType.GROUP else None)
        body += ("\n" if body and not body.endswith("\n") else "") + marker
        is_text = True

    # 私聊默认 to_me，不代表用户在正文中提及了机器人。
    if to_me and not has_at_mention and session.scene.type != SceneType.PRIVATE:
        reply_to_bot = False
        if reply := getattr(event, "reply", None):
            try:
                sender = getattr(reply, "sender", None)
                if sender and str(getattr(sender, "user_id", "")) == str(bot.self_id):
                    reply_to_bot = True
            except Exception:
                pass
        if not reply_to_bot:
            separator = "\n" if shared_cards or forward_references or audios or videos or files else " "
            body = f"{plugin_config.bot_name}{separator}{body}"

    if not reply_id:
        reply_id = _extract_reply_message_id_from_event(event)

    # 第2行（可选）：回复元数据，格式 "回复id: {id}"
    if reply_id:
        content += f"回复id: {reply_id}\n"
    # 第3行起：正文
    # Match the stored text when deduplicating messages whose bodies contain NUL.
    body = body.replace("\x00", "")
    content += body

    # 构建用户名：仅保留用户真实显示名，不混入群身份标签（群主/管理员）
    # 避免模型误把“群主-”等前缀当成用户名的一部分。
    user_name = session.user.name or session.user.nick or session.user.id
    if session.member and session.member.nick:
        user_name = session.member.nick

    # ========== 步骤1: 处理文本消息（快速） ==========
    # 用锁保证多bot并发安全: SELECT + INSERT + COMMIT 原子化
    if (is_text or imgs) and sender_is_connected_bot:
        # 本插件的发送工具会主动写入 bot 消息。给发送侧事务一个很短的提交窗口，
        # 再按平台消息 ID 判断是否已经入库；其他插件发出的消息仍会继续记录。
        await asyncio.sleep(0.25)
    is_new_text_message = True
    async with _get_dedup_lock(session.scene.id):
        if is_reliable_platform_id(incoming_message_id):
            # Outbound records use bot_name rather than the QQ account. Match
            # the transport ID before filtering by user_id, also for delayed
            # echoes and pure images received by another Bot process.
            recorded_bot_id = (await db_session.execute(
                Select(ChatHistory.msg_id).where(
                    ChatHistory.session_id == session.scene.id,
                    ChatHistory.content_type == "bot",
                    ChatHistory.created_at >= datetime.datetime.now() - BOT_ECHO_LOOKBACK,
                    ChatHistory.content.startswith(content_prefix, autoescape=True)
                    | ChatHistory.content.startswith(f"id:{incoming_message_id}\n", autoescape=True),
                ).limit(1)
            )).scalar_one_or_none()
            if recorded_bot_id is not None:
                await db_session.commit()
                logger.debug("检测到已记录的 bot 消息回声，跳过入站记录和回复")
                return
        if is_text:
            do_insert = True
            time_window = datetime.datetime.now() - datetime.timedelta(seconds=3)
            existing_query = Select(ChatHistory).where(
                ChatHistory.session_id == session.scene.id,
                ChatHistory.created_at >= time_window,
            )
            if not sender_is_connected_bot:
                existing_query = existing_query.where(ChatHistory.user_id == session.user.id)
            existing = await db_session.execute(existing_query)
            if any(
                _matches_inbound_message(
                    history.content,
                    content_prefix,
                    body,
                    # Distinct platform IDs are distinct messages even when a
                    # user repeats the exact same text within a few seconds.
                    exact_id_only=sender_is_connected_bot or is_reliable_platform_id(incoming_message_id),
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
            if (is_new_text_message and (is_text or imgs) and message_pilot_enabled(
                plugin_config, str(session.scene.id), is_private=session.scene.type == SceneType.PRIVATE
            )):
                await record_inbound_safely(
                    db_session,
                    event=event,
                    fallback_segments=native_segments,
                    normalized_parts=normalized_parts,
                    mention_names=mention_names,
                    file_ids=file_resource_ids,
                    file_source_ids=tuple(str(file.id or "") for file in files),
                    file_names=tuple(clean_filename(str(file.name or "")) for file in files),
                    image_source_ids=tuple(image_source_id(image) for image in imgs),
                    platform=adapter_name(bot),
                    scope="private" if session.scene.type == SceneType.PRIVATE else "group",
                    session_id=str(session.scene.id),
                    platform_id=incoming_message_id,
                    sender_id=str(session.user.id),
                    sender_name=str(user_name),
                    sender_kind="bot" if sender_is_connected_bot else "user",
                    bot_id=str(bot.self_id),
                    reply_to_id=reply_id,
                )
            await db_session.commit()
        except Exception as e:
            logger.error(f"保存文本消息失败: {e}")
            await db_session.rollback()

    if is_text and not is_new_text_message:
        logger.info(f"检测到重复入站消息，跳过回复 - session: {session.scene.id}")
        return
    if is_new_text_message:
        _schedule_rag_vectorization(session.scene.id)

    # ========== 步骤2: 决定是否回复（在图片处理前判断） ==========
    plain_text = event.get_plaintext()
    if card_text:
        plain_text = (plain_text + "\n" + card_text).strip()
    stripped_plain_text = msg.extract_plain_text().strip()
    command_like = plain_text.startswith(("!", "！", "/", "#", "?", "\\"))
    if stripped_plain_text.lower().startswith(plugin_config.bot_name):
        to_me = True
    explicit_to_me = to_me
    is_group = session.scene.type == SceneType.GROUP
    bot_id = str(bot.self_id)
    if (is_group and is_text and is_new_text_message and not sender_is_connected_bot and not command_like
            and (bool(stripped_plain_text) or "[表情]" not in body)
            and plugin_config.group_jargon_learning_enabled
            and (plugin_config.chat_api_key or plugin_config.llm_api_key or plugin_config.qwen_token
                 or plugin_config.vertex_credentials_path or plugin_config.vertex_api_key or plugin_config.chat_api_format == "vertex")):
        from .agent.jargon_learning import schedule_jargon_learning

        schedule_jargon_learning(plugin_data_dir / "group_jargon", session.scene.id, get_jargon_discovery_model,
                                review_model_factory=get_jargon_review_model, excluded_terms=(plugin_config.bot_name,),
                                excluded_users=tuple(connected_bot_ids | {bot_id}),
                                structured_platform=adapter_name(bot) if message_pilot_enabled(
                                    plugin_config, str(session.scene.id), is_private=False) else None)
    active_followup = is_group and _is_continuous_conversation(session.scene.id, session.user.id, bot_id)
    group_followup = is_group and time.monotonic() - participation.state((bot_id, session.scene.id)).last_sent < 300
    disengaged = is_group and (explicit_to_me or ((active_followup or group_followup) and not has_at_mention and not reply_id)) and _is_explicit_disengagement(stripped_plain_text, plugin_config.bot_name)
    if disengaged:
        _input_batcher.discard_speaker((bot_id, session.scene.id, session.user.id))
        _continuous_conversation_until.pop(_continuous_key(session.scene.id, session.user.id, bot_id), None)
        _explicit_conversation_until.pop(_continuous_key(session.scene.id, session.user.id, bot_id), None)
        logger.debug("用户明确退出当前 bot 对话，本条保持沉默并结束跟进窗口")
    continuous_to_me = not disengaged and not explicit_to_me and not command_like and not has_at_mention and not reply_id and bool(stripped_plain_text) and active_followup
    if continuous_to_me:
        logger.debug(f"群 {session.scene.id} 用户 {session.user.id} 命中连续对话窗口")
    if is_group and stripped_plain_text and not command_like:
        participation.observe(
            (bot_id, session.scene.id),
            addressed=explicit_to_me,
            disengaged=disengaged,
        )
        if is_new_text_message and resolve_group_reply_probability(session.scene.id) > 0:
            note_group_activity((bot_id, session.scene.id), interaction=explicit_to_me or disengaged)
    repeat_text = None
    if not disengaged and is_group and not to_me and not continuous_to_me and not command_like and bool(stripped_plain_text):
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
            command_like=command_like or has_at_mention or bool(reply_id),
            has_text=bool(stripped_plain_text),
            is_group=is_group,
            reaction_supported=is_onebot_context(bot, event),
            reply_probability=resolve_group_reply_probability(
                session.scene.id if is_group else None
            ),
            participation_key=(bot_id, session.scene.id) if is_group else None,
        )
    meme_required = (to_me or continuous_to_me) and _is_explicit_meme_request(stripped_plain_text)
    meme_send_count = _get_explicit_meme_send_count(stripped_plain_text) if meme_required else 1
    reaction_required = is_group and not meme_required and (to_me or continuous_to_me) and is_onebot_context(bot, event) and _is_explicit_reaction_request(stripped_plain_text)
    if meme_required:
        random_reply_sample = False
        proactive_reaction_only = False
        proactive_meme_only = False
    elif reaction_required:
        random_reply_sample = False
        proactive_reaction_only = False
        proactive_meme_only = False
    should_reply = not disengaged and (to_me or continuous_to_me or random_reply_sample or proactive_reaction_only or proactive_meme_only or repeat_reply_sample)
    if (is_group and is_new_text_message and not sender_is_connected_bot and not command_like and stripped_plain_text
            and configured_space(plugin_config.cross_group_spaces, str(session.scene.id))):
        try:
            async with get_session() as observation_session:
                await record_message(observation_session, platform=adapter_name(bot), bot_id=str(bot.self_id),
                                     scene_id=str(session.scene.id), message_id=incoming_message_id,
                                     user_id=str(session.user.id), text=stripped_plain_text, reply_to=reply_id,
                                     now=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None))
                await observation_session.commit()
        except Exception:
            logger.exception("记录跨群观察消息失败，请确认已执行 nb orm upgrade")
    if explicit_to_me and not disengaged:
        _refresh_continuous_conversation(session.scene.id, session.user.id, bot_id)
        if _is_explicit_conversation_request(stripped_plain_text):
            key = _continuous_key(session.scene.id, session.user.id, bot_id)
            if key in _continuous_conversation_until:
                _explicit_conversation_until[key] = _continuous_conversation_until[key]
    if not plain_text and not imgs and not forward_references and not audios and not videos and not files:
        should_reply = False
    if files and not stripped_plain_text and not shared_cards and not imgs and not forward_references and not audios and not videos:
        # Receiving a file is not a request to analyze it or download it.
        should_reply = False
    if shared_cards and not stripped_plain_text and not (to_me or continuous_to_me):
        # A standalone share supplies context, not a request to inspect a page.
        should_reply = False
    if command_like:
        should_reply = False
    if (
        not plain_text
        and not forward_references
        and not audios
        and not videos
        and not files
        and not (to_me or continuous_to_me)
    ):
        should_reply = False

    # ========== 步骤3: 处理图片消息 ==========
    # 如果要回复则同步等待图片处理完成,否则后台异步
    if imgs:
        if should_reply:
            for image_index, img in enumerate(imgs):
                await process_image_message(
                    db_session,
                    img,
                    event,
                    bot,
                    state,
                    session,
                    user_name,
                    content_prefix,
                    image_index=image_index,
                    image_count=len(imgs),
                )
        else:
            for image_index, img in enumerate(imgs):
                _start_background_image_task(
                    img,
                    event,
                    bot,
                    state,
                    session,
                    user_name,
                    content_prefix,
                    image_index=image_index,
                    image_count=len(imgs),
                )

    # ========== 步骤4: 处理回复 ==========
    # Whether to reply is independent from who sent the triggering message.
    # Always keep the real sender identity so bound custom tools can resolve
    # requests such as "查询我的成绩" without asking the model for an ID.
    user_id = session.user.id
    track_observation = (
        plugin_config.proactive_observation_enabled and is_group and is_new_text_message
        and bool(stripped_plain_text) and not command_like and not sender_is_connected_bot
        and (explicit_to_me or disengaged or (not has_at_mention and not reply_id))
        and resolve_group_reply_probability(session.scene.id) > 0
    )
    pending_input = (is_group and not command_like and not disengaged and (not has_at_mention or explicit_to_me)
                     and _input_batch_key(bot_id, session.scene.id, user_id, reply_id) in _input_batcher.pending)
    if should_reply or track_observation or pending_input:
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
            message_id=incoming_message_id if is_reliable_platform_id(incoming_message_id) else None,
            proactive_meme_only=proactive_meme_only,
            meme_required=meme_required,
            meme_send_count=meme_send_count,
            proactive_reaction_only=proactive_reaction_only,
            reaction_required=reaction_required,
            repeat_text=repeat_text if repeat_reply_sample else None,
            mentioned_user_ids=(mentioned_user_ids(normalized_parts) if normalized_parts is not None
                                else event_mentioned_user_ids(event)),
            normalized_parts=normalized_parts,
        )
        if track_observation:
            _observation_schedule.note((bot_id, group_id), request, addressed=explicit_to_me, disengaged=disengaged)
        # Commit all text/image rows before the batch can wake a reader.
        await db_session.commit()
        if should_reply or pending_input:
            await _submit_reply_input(group_id, request, str(incoming_message_id), (body or plain_text) if has_at_mention else plain_text)
    else:
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
    *,
    image_index: int | None = None,
    image_count: int | None = None,
):
    """处理单张图片消息 (修复并发插入报错)"""
    try:
        content_type = "image"
        raw = getattr(img, "raw", None)
        url = getattr(img, "url", None)
        if not img.id and not (raw or url):
            return
        if not img.id and not raw and getattr(img, "path", None):
            # An inbound path is not a local file authorization.
            return
        source_name = str(img.id or url or getattr(img, "name", ""))
        try:
            suffix = Path(urlsplit(source_name).path).suffix.lower().lstrip(".")
        except ValueError:
            suffix = ""
        image_format = suffix if suffix in {"jpg", "jpeg", "png", "gif", "webp", "bmp"} else "jpg"

        # 1. 获取和压缩图片
        try:
            pic = await asyncio.wait_for(image_fetch(event, bot, state, img), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("下载图片超时，跳过")
            return
        if not pic:
            logger.warning("图片来源未返回数据，跳过保存")
            return

        pic = await asyncio.to_thread(check_and_compress_image_bytes, pic, image_format=image_format.upper())
        file_hash = generate_file_hash(pic)
        file_name = f"{file_hash}.{image_format}"
        file_path = pic_dir / file_name

        # 2. 保存文件到本地
        if not file_path.exists():
            file_path.write_bytes(pic)

        # Match text ingestion: acquire the group lock BEFORE the first SQL
        # operation. Holding a pooled connection/write transaction while waiting
        # for this lock can deadlock with text ingestion waiting for that resource.
        async with _get_dedup_lock(session.scene.id):
            # 3. 数据库操作 (MediaStorage)
            stmt = Select(MediaStorage).where(MediaStorage.file_hash == file_hash)
            media_obj = (await db_session.execute(stmt)).scalar_one_or_none()

            if media_obj:
                media_obj.references += 1
                db_session.add(media_obj)
            else:
                new_media = MediaStorage(
                    file_hash=file_hash,
                    file_path=file_name,
                    references=1,
                    description="[图片]",
                )
                db_session.add(new_media)
                try:
                    await db_session.flush()
                    media_obj = new_media
                except Exception as e:
                    await db_session.rollback()
                    media_obj = (await db_session.execute(stmt)).scalar_one_or_none()
                    if media_obj is None:
                        logger.error(f"插入图片记录失败（非并发冲突）: {e}")
                        raise
                    logger.info(f"图片并发插入冲突 {file_hash}，转为更新模式")
                    media_obj.references += 1
                    db_session.add(media_obj)

            # 4. 添加聊天历史 (ChatHistory)
            if media_obj:
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
                        ChatHistory.user_id == session.user.id,
                        ChatHistory.created_at >= time_window,
                    )
                )
                candidates = existing_img.scalars().all()
                incoming_id = content_prefix.removeprefix("id: ").splitlines()[0]
                if image_index is not None and image_count is not None and is_reliable_platform_id(incoming_id):
                    existing_image = next((row for row in candidates if incoming_id in platform_message_ids(row.content)), None)
                else:
                    existing_image = candidates[0] if candidates else None
                if existing_image:
                    # Preserve aliases used by the legacy no-index path while
                    # keeping the filename as the last line.
                    message_marker = content_prefix.splitlines()[0]
                    if message_marker.startswith("id: ") and f"{message_marker}\n" not in existing_image.content:
                        header, separator, filename = existing_image.content.rpartition("\n")
                        existing_image.content = f"{header}{separator}alias_{message_marker}\n{filename}"
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

                if (image_index is not None and image_count is not None and message_pilot_enabled(
                    plugin_config, str(session.scene.id), is_private=session.scene.type == SceneType.PRIVATE
                )):
                    await bind_image_resource_safely(
                        db_session,
                        platform=adapter_name(bot),
                        scope="private" if session.scene.type == SceneType.PRIVATE else "group",
                        session_id=str(session.scene.id),
                        platform_id=content_prefix.removeprefix("id: ").splitlines()[0],
                        image_index=image_index,
                        image_count=image_count,
                        media_id=media_obj.media_id,
                        sender_id=str(session.user.id),
                        source_id=image_source_id(img),
                        require_source_for_multi=event is not None,
                    )

                # 5. 在锁内提交
                await db_session.commit()

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        await db_session.rollback()


async def _process_image_task(img, event, bot, state, session, user_name, content_prefix, *, image_index=None, image_count=None):
    """后台图片处理任务，使用独立的数据库会话，不阻塞主消息流程"""
    # Queueing happens before a session is opened, so an image burst cannot
    # reserve every pooled connection while waiting for CPU/network work.
    async with background_image_gate.slot():
        async with get_session() as db_session:
            await process_image_message(db_session, img, event, bot, state, session, user_name, content_prefix,
                                        image_index=image_index, image_count=image_count)


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
    *,
    image_index: int | None = None,
    image_count: int | None = None,
) -> bool:
    if len(_background_image_tasks) >= plugin_config.background_image_max_pending:
        logger.warning(f"后台图片任务已达上限 {plugin_config.background_image_max_pending}，跳过本张图片")
        return False

    task = asyncio.create_task(
        _process_image_task(img, event, bot, state, session, user_name, content_prefix,
                            image_index=image_index, image_count=image_count),
        name=f"ai-groupmate-image:{session.scene.id}",
    )
    _background_image_tasks.add(task)
    task.add_done_callback(_consume_background_image_task)
    return True


async def _send_quota_notice(
    db_session,
    *,
    status: GroupDailyQuotaStatus,
    request_id: str,
    session: Uninfo,
    interface: QryItrface,
    bot_name: str,
    is_private: bool = False,
    user_id: str | None = None,
) -> None:
    if is_private:
        message = build_private_quota_exhausted_message(status)
        send_target = Target(
            id=user_id or session.scene.id,
            private=True,
            self_id=session.self_id,
        )
        log_label = "私聊公共模型额度提示未发送"
    else:
        message = build_quota_exhausted_message(status)
        send_target = Target(
            id=session.scene.id,
            private=False,
            self_id=session.self_id,
        )
        log_label = "公共模型额度提示未发送"
    pilot = message_pilot_enabled(plugin_config, str(session.scene.id), is_private=is_private)
    identity = (OutboundIdentity(adapter_name(get_bots().get(str(session.self_id))), "private" if is_private else "group",
                                 str(session.self_id), bot_name) if pilot else None)
    reply_tool = create_reply_tool(
        db_session,
        session.scene.id,
        request_id,
        interface=interface,
        send_target=send_target,
        bot_name=bot_name,
        parse_msg_meta=_parse_msg_meta,
        group_members=[],
        recovery_enabled=pilot,
        outbound_identity=identity,
    )
    raw_result = await reply_tool.ainvoke(
        {"content": message, "next_step": "end"}
    )
    result = json.loads(raw_result)
    if not result.get("ok"):
        logger.info(f"{log_label}: {result.get('message', raw_result)}")
    await db_session.commit()


async def _send_agent_error_notice(
    db_session,
    *,
    content: str,
    request_id: str,
    session: Uninfo,
    interface: QryItrface,
    bot_name: str,
    is_private: bool,
    group_members: list[Any] | None,
) -> None:
    pilot = message_pilot_enabled(plugin_config, str(session.scene.id), is_private=is_private)
    identity = (OutboundIdentity(adapter_name(get_bots().get(str(session.self_id))), "private" if is_private else "group",
                                 str(session.self_id), bot_name) if pilot else None)
    reply_tool = create_reply_tool(
        db_session,
        session.scene.id,
        request_id,
        interface=interface,
        send_target=Target(
            id=session.scene.id,
            private=is_private,
            self_id=session.self_id,
        ),
        bot_name=bot_name,
        parse_msg_meta=_parse_msg_meta,
        group_members=group_members or [],
        recovery_enabled=pilot,
        outbound_identity=identity,
    )
    raw_result = await reply_tool.ainvoke(
        {"content": content, "next_step": "end"}
    )
    result = json.loads(raw_result)
    if not result.get("ok"):
        logger.info(f"模型 API 错误提示未发送: {result.get('message', raw_result)}")
    await db_session.commit()


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
    meme_send_count: int = 1,
    proactive_reaction_only: bool = False,
    reaction_required: bool = False,
    repeat_text: str | None = None,
    proactive_observation: bool = False,
):
    """Compatibility entry point for callers still using individual arguments."""
    await handle_reply_request(ReplyRequest(
        request_id=request_id, session=session, interface=interface, bot=bot, event=event,
        bot_name=bot_name, user_id=user_id, user_name=user_name, is_tome=is_tome,
        is_continuous=is_continuous, reply_to_id=reply_to_id, proactive_meme_only=proactive_meme_only,
        meme_required=meme_required, meme_send_count=meme_send_count, proactive_reaction_only=proactive_reaction_only,
        reaction_required=reaction_required, repeat_text=repeat_text, proactive_observation=proactive_observation,
        input_batch=current_input_batch.get(), mentioned_user_ids=current_mentioned_user_ids.get() or (),
    ))


async def handle_reply_request(request: ReplyRequest) -> None:
    """Handle one admitted request with its conversation context intact."""
    request_id, session, interface, bot, event = request.request_id, request.session, request.interface, request.bot, request.event
    bot_name, user_id, user_name = request.bot_name, request.user_id, request.user_name
    is_tome, is_continuous, reply_to_id = request.is_tome, request.is_continuous, request.reply_to_id
    proactive_meme_only, meme_required, meme_send_count = request.proactive_meme_only, request.meme_required, request.meme_send_count
    proactive_reaction_only, reaction_required = request.proactive_reaction_only, request.reaction_required
    repeat_text, proactive_observation = request.repeat_text, request.proactive_observation
    is_private = session.scene.type == SceneType.PRIVATE
    participation_key = (str(session.self_id), session.scene.id)
    autonomous = not is_private and not is_tome and not is_continuous and repeat_text is None
    delivery_token = delivery_context.set(
        None if is_private else (participation_key, autonomous)
    )
    observation_token = observation_guard.set((lambda: _observation_enabled(participation_key)) if proactive_observation else None)
    delivered = [False]
    delivered_token = confirmed_turn_delivery.set(delivered)
    previous_sent = participation.state(participation_key).last_sent
    group_quota_limit = plugin_config.global_model_daily_group_limit
    private_quota_limit = plugin_config.global_model_daily_private_user_limit
    uses_global_model_quota = (
        not is_private
        and plugin_config.global_model_daily_group_limit_enabled
        and group_quota_limit > 0
        and not has_group_model_config(session.scene.id)
    )
    uses_private_model_quota = (
        is_private
        and plugin_config.global_model_daily_private_user_limit_enabled
        and private_quota_limit > 0
        and not has_private_model_config(user_id)
    )
    try:
        if proactive_observation and (is_private or not _observation_allowed(participation_key)):
            return
        if autonomous and (
            resolve_group_reply_probability(session.scene.id) <= 0
            or not participation.can_reply(participation_key)
        ):
            return
        if repeat_text is not None and not is_private:
            # 复读采样已经代表“加入队形”的决定，直接发送，避免模型二次犹豫或点评。
            async with get_session() as repeat_session:
                # 请求可能在群队列里等待；执行前重新检查队形，避免 bot 已经
                # 跟过一次后，旧的待处理请求再次复读同一条连续队形。
                current_repeat_text = await _load_repeat_chain_text(
                    repeat_session,
                    session.scene.id,
                )
                current_normalized = " ".join(current_repeat_text.split()).casefold() if current_repeat_text is not None else None
                requested_normalized = " ".join(repeat_text.split()).casefold()
                if current_normalized != requested_normalized:
                    logger.info(f"复读队形已变化或 bot 已参与，跳过二次复读: {repeat_text}")
                    await repeat_session.commit()
                    return
                # 跟随执行时最新一条的原始格式，检测比较仍使用标准化文本。
                assert current_repeat_text is not None
                repeat_text = current_repeat_text
                pilot = message_pilot_enabled(plugin_config, str(session.scene.id), is_private=False)
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
                    recovery_enabled=pilot,
                    outbound_identity=(OutboundIdentity(adapter_name(bot), "group", str(session.self_id),
                                                        bot_name, request_id) if pilot else None),
                )
                raw_result = await reply_tool.ainvoke({"content": repeat_text, "next_step": "end"})
                result = json.loads(raw_result)
                if not result.get("ok"):
                    logger.info(f"复读消息未发送: {result.get('message', raw_result)}")
                await repeat_session.commit()
            return

        # 获取最近几条用于 Flash 快速判断
        # 注意：Flash 模型是纯文本模型，它看不懂图片，所以这里我们只喂文本内容
        async with get_session() as history_session:
            quota_status = (
                await get_group_daily_quota_status(
                    history_session,
                    session.scene.id,
                    group_quota_limit,
                )
                if uses_global_model_quota
                else (
                    await get_private_user_daily_quota_status(
                        history_session,
                        user_id,
                        private_quota_limit,
                    )
                    if uses_private_model_quota
                    else None
                )
            )
            recent_query = Select(ChatHistory).where(ChatHistory.session_id == session.scene.id)
            if proactive_observation:
                recent_query = recent_query.where(
                    ChatHistory.created_at >= datetime.datetime.now() - datetime.timedelta(hours=1),
                    ChatHistory.content_type.in_(("text", "bot")),
                    ChatHistory.media_id.is_(None),
                )
            recent_msgs = (
                (
                    await history_session.execute(
                        recent_query
                        .order_by(ChatHistory.msg_id.desc())
                        .limit(20 if proactive_observation else 12)
                    )
                )
                .scalars()
                .all()
            )
            recent_msgs = [ChatHistorySchema.model_validate(row) for row in recent_msgs[::-1]]
            if proactive_observation:
                recent_msgs = [row.model_copy(update={"content": row.content[:600]}) for row in recent_msgs]
            recent_snapshots = (await load_structured_context_safely(
                history_session, recent_msgs, session_id=str(session.scene.id),
                scope="private" if is_private else "group", platform=adapter_name(bot), include_media=False,
            )) if message_pilot_enabled(plugin_config, str(session.scene.id), is_private=is_private) else None
            await history_session.commit()

        recent_resolver = TurnHistoryResolver(recent_snapshots)

        if quota_status is not None and not quota_status.allowed:
            if uses_private_model_quota:
                async with get_session() as notice_session:
                    await _send_quota_notice(
                        notice_session,
                        status=quota_status,
                        request_id=request_id,
                        session=session,
                        interface=interface,
                        bot_name=bot_name,
                        is_private=True,
                        user_id=user_id,
                    )
            elif is_tome:
                async with get_session() as notice_session:
                    await _send_quota_notice(
                        notice_session,
                        status=quota_status,
                        request_id=request_id,
                        session=session,
                        interface=interface,
                        bot_name=bot_name,
                    )
            return

        if not recent_msgs:
            return

        trigger_message_id = None if proactive_observation else (
            request.message_id if is_reliable_platform_id(request.message_id) else
            str(getattr(event, "message_id", "") or "") or None
        )
        if not proactive_observation and not is_reliable_platform_id(trigger_message_id) and request.input_batch:
            trigger_message_id = request.input_batch[-1][0]
        pilot_trace = (trace_id(str(session.scene.id), request_id)
                       if message_pilot_enabled(plugin_config, str(session.scene.id), is_private=is_private) else None)
        if pilot_trace:
            logger.info(f"[MessageTrace] trace={pilot_trace} stage=turn_started "
                        f"source={trace_id(str(session.scene.id), trigger_message_id)}")
        if proactive_observation:
            latest_human = next((m for m in reversed(recent_msgs) if m.content_type != "bot"), None)
        else:
            recent_msgs, latest_human = bind_turn_history(recent_msgs, user_id, trigger_message_id, recent_resolver)
        event_text = ""
        if not proactive_observation:
            try:
                event_text = event.get_plaintext().strip()
            except Exception:
                pass
        if latest_human is None and not event_text:
            return

        one_to_one = bool(is_continuous and latest_human is not None and not proactive_observation
                          and is_one_to_one_exchange(recent_msgs, user_id, bot_ids={str(item) for item in get_bots()}))
        direct_conversation = bool(one_to_one and _is_explicit_conversation(session.scene.id, user_id, str(bot.self_id)))

        # 简单的文本摘要用于 Gatekeeper
        history_summary = ""
        for m in recent_msgs:
            role = "Bot" if m.content_type == "bot" else f"群友 user_id={m.user_id}"
            if m.content_type == "image":
                history_summary += f"[{role}] {m.user_name}: [发送了一张图片/表情包，可能只是随手发的]\n"
            else:
                history_summary += f"[{role}] {m.user_name}: {recent_resolver.gate_content(m)[:1200]}\n"

        if proactive_observation and latest_human is not None and latest_human.created_at < datetime.datetime.now() - datetime.timedelta(minutes=15):
            return
        current_msg_text = event_text or (
            recent_resolver.metadata(latest_human)[2][:1200] if latest_human is not None and latest_human.content_type == "text"
            else "[图片/表情包。除非用户明确在问这张图、@bot、回复bot或正在延续图片话题，否则通常不需要回应]"
        )
        gatekeeper_msg_text = current_msg_text
        if is_continuous:
            gatekeeper_msg_text = f"这是用户在刚才主动呼叫 bot 后的连续对话消息。如果像追问、补充、回应 bot 或继续话题，应倾向回复；如果只是“嗯”“哈哈”“行”等无需回应的短反馈，可以不回复。\n{current_msg_text}"
        elif proactive_reaction_only:
            gatekeeper_msg_text = f"这条群消息命中了低概率主动 reaction 采样。仅当用一个消息表情回应就能自然表达赞同、好笑、惊讶、安慰、感谢或轻量态度时才回复；提问、求助、敏感/沉重话题、真实冲突、他人之间的定向对话，以及没有明显反应价值的普通消息都应保持沉默。\n{current_msg_text}"
        elif proactive_meme_only:
            gatekeeper_msg_text = f"这条群消息命中了低概率主动表情包采样。只要一张表情包能大致作为群友式接梗、吐槽或情绪反应就可以回复，允许轻微偏题，不要求完全精确；认真求助、事实问题、敏感/沉重话题、真实冲突、他人之间的定向对话，以及普通到没有反应价值的消息都应保持沉默。\n{current_msg_text}"

        # === Gatekeeper 判断 ===
        if not is_tome and repeat_text is None and not direct_conversation:
            observed_sent = participation.state(participation_key).last_sent
            gate_started = time.perf_counter()
            gate_failure: list[tuple[str, int | None]] = []
            should_reply = await check_if_should_reply(
                history_summary,
                gatekeeper_msg_text,
                bot_name,
                is_private=is_private,
                proactive_meme_only=proactive_meme_only,
                proactive_reaction_only=proactive_reaction_only,
                participation_context=participation.context(participation_key) if autonomous else "",
                on_continuation=(
                    lambda: participation.engaged(participation_key, expected_sent=observed_sent)
                ) if autonomous and not proactive_observation else None,
                turn_context=build_turn_context(
                    recent_msgs,
                    user_id=latest_human.user_id if proactive_observation and latest_human is not None else user_id,
                    user_name=latest_human.user_name if proactive_observation and latest_human is not None else user_name,
                    current_text=current_msg_text,
                    message_id=None if proactive_observation else trigger_message_id,
                    continuous=is_continuous, observation=proactive_observation,
                    resolver=recent_resolver,
                ) if not is_private else "",
                is_continuous=is_continuous,
                on_failure=lambda error_type, status: gate_failure.append((error_type, status)),
            )
            if pilot_trace:
                logger.info(f"[MessageTrace] trace={pilot_trace} stage=gatekeeper_decided reply={should_reply} "
                            f"duration_ms={(time.perf_counter() - gate_started) * 1000:.0f}")
            if not should_reply:
                transient_types = {"ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError", "RemoteProtocolError", "TimeoutError"}
                transient = bool(gate_failure and (gate_failure[0][0] in transient_types or
                                 gate_failure[0][1] == 429 or (gate_failure[0][1] is not None and gate_failure[0][1] >= 500)))
                if not (one_to_one and transient):
                    return
                direct_conversation = True
                logger.info("[TurnAddress] decision_unavailable fallback=one_to_one")
            if autonomous and (
                resolve_group_reply_probability(session.scene.id) <= 0
                or not participation.can_reply(participation_key)
            ):
                return
        elif pilot_trace:
            logger.info(f"[MessageTrace] trace={pilot_trace} stage=gatekeeper_bypassed")

        # === 获取详细历史给 Agent ===
        if proactive_observation:
            last_msg = recent_msgs
            last_resolver = recent_resolver
        else:
            async with get_session() as history_session:
                last_msg = await _load_agent_history(history_session, session.scene.id)
                last_snapshots = (await load_structured_context_safely(
                    history_session, last_msg, session_id=str(session.scene.id),
                    scope="private" if is_private else "group", platform=adapter_name(bot), include_media=False,
                )) if message_pilot_enabled(plugin_config, str(session.scene.id), is_private=is_private) else None
                await history_session.commit()
            last_resolver = TurnHistoryResolver(last_snapshots)

        if not last_msg:
            logger.info("没有历史消息，跳过回复")
            return

        if not proactive_observation:
            last_msg, _ = bind_turn_history(last_msg, user_id, trigger_message_id, last_resolver)

        if proactive_observation and not _observation_allowed(participation_key):
            return

        # A matching window is only a candidate. Refresh it after Gatekeeper
        # accepts the follow-up, so short ignored feedback cannot extend it.
        if is_continuous:
            if not _is_continuous_conversation(session.scene.id, user_id, str(bot.self_id)):
                return
            _refresh_continuous_conversation(session.scene.id, user_id, str(bot.self_id))

        # A group-level API bypasses the public quota. Re-check the active
        # configuration here in case an administrator changed it while this
        # request was waiting in the per-group queue.
        if uses_global_model_quota and not has_group_model_config(
            session.scene.id
        ):
            async with get_session() as quota_session:
                quota_status = await consume_group_daily_quota(
                    quota_session,
                    session.scene.id,
                    group_quota_limit,
                )
                if not quota_status.allowed:
                    if is_tome:
                        await _send_quota_notice(
                            quota_session,
                            status=quota_status,
                            request_id=request_id,
                            session=session,
                            interface=interface,
                            bot_name=bot_name,
                        )
                    return

        # A private-level API bypasses the public quota. Re-check the active
        # configuration here in case the user changed it while this request
        # was waiting in the queue.
        if uses_private_model_quota and not has_private_model_config(user_id):
            async with get_session() as quota_session:
                quota_status = await consume_private_user_daily_quota(
                    quota_session,
                    user_id,
                    private_quota_limit,
                )
                if not quota_status.allowed:
                    await _send_quota_notice(
                        quota_session,
                        status=quota_status,
                        request_id=request_id,
                        session=session,
                        interface=interface,
                        bot_name=bot_name,
                        is_private=True,
                        user_id=user_id,
                    )
                    return

        role_map: dict[str, str] = {}
        group_members: list[Any] | None = [] if proactive_observation else None
        if not is_private and not proactive_observation:
            try:
                group_members = list(await interface.get_members(SceneType.GROUP, session.scene.id))
                for member in group_members:
                    role_name = getattr(getattr(member, "role", None), "name", None)
                    if role_name in {"owner", "admin"}:
                        role_map[str(member.id)] = role_name
            except Exception as e:
                logger.warning(f"获取群成员身份信息失败，降级为无身份标注: {e}")

        if proactive_observation:
            if not _observation_allowed(participation_key):
                return
            participation.state(participation_key).last_check = participation.clock()
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
                        None if proactive_observation else event,
                        is_private=is_private,
                        group_members=group_members,
                        proactive_meme_only=proactive_meme_only,
                        meme_required=meme_required,
                        meme_send_count=meme_send_count,
                        proactive_reaction_only=proactive_reaction_only,
                        reaction_required=reaction_required,
                        repeat_text=repeat_text,
                        proactive_observation=proactive_observation,
                        direct_conversation=direct_conversation,
                    ),
                    timeout=agent_run_timeout_seconds(plugin_config),
                )
        except asyncio.TimeoutError:
            logger.warning(f"Agent 思考超时 - session: {session.scene.id}")
            return

        except asyncio.CancelledError:
            logger.debug(f"群 {session.scene.id} 回复任务被取消（切换到更新请求）")
            raise

        if (
            strategy.need_reply
            and strategy.text
            and (is_private or is_tome)
        ):
            async with get_session() as notice_session:
                await _send_agent_error_notice(
                    notice_session,
                    content=strategy.text,
                    request_id=request_id,
                    session=session,
                    interface=interface,
                    bot_name=bot_name,
                    is_private=is_private,
                    group_members=group_members,
                )

        logger.info(f"Agent决策结果: {strategy}")

    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
        print(traceback.format_exc())
    finally:
        confirmed_turn_delivery.reset(delivered_token)
        if (not is_private and delivered[0] and (is_tome or is_continuous) and not proactive_observation
                and not proactive_meme_only and not proactive_reaction_only and repeat_text is None
                and configured_space(plugin_config.cross_group_spaces, str(session.scene.id))):
            try:
                async with get_session() as observation_session:
                    await record_interaction(observation_session, platform=adapter_name(bot), bot_id=str(bot.self_id),
                                             scene_id=str(session.scene.id), turn_id=request_id, explicitly_addressed=is_tome,
                                             now=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None))
                    await observation_session.commit()
            except Exception:
                logger.exception("记录跨群观察交互失败，请确认已执行 nb orm upgrade")
        if not is_private and participation.state(participation_key).last_sent > previous_sent:
            _observation_schedule.participated(participation_key)
        observation_guard.reset(observation_token)
        delivery_context.reset(delivery_token)


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


async def _collect_words_from_db(db_session, session_id: str, days: int = 1, user_id: str | None = None) -> str:
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
        await reset_negative_relation.finish(f"检测到 {affected_count} 条负好感度关系。\n本操作会先备份，再将这些用户的好感度归零并清空旧标签。\n确认执行请发送：/重置负面关系 确认")

    try:
        result = await reset_negative_relations(db_session, relation_backup_dir)
    except Exception:
        await _safe_rollback(db_session)
        logger.exception("重置负面关系失败")
        await reset_negative_relation.finish("重置失败，数据库未修改，请查看日志。")

    if result.affected_count == 0:
        await reset_negative_relation.finish("没有需要重置的负好感度关系。")
    await reset_negative_relation.finish(f"已重置 {result.affected_count} 条负好感度关系。\n备份文件：{result.backup_path}")


frequency = on_command("词频")


async def _send_wordcloud_image(db_session, session: Uninfo, image_bytes: bytes) -> None:
    session_id = str(session.scene.id)
    is_private = session.scene.type == SceneType.PRIVATE
    pilot = message_pilot_enabled(plugin_config, session_id, is_private=is_private)
    bot_id = str(session.self_id)
    workspace = FileWorkspace(plugin_data_dir / "agent_files", session_id=session_id, user_id=str(session.user.id),
                              bot_id=bot_id, is_private=is_private)
    identity = OutboundIdentity(adapter_name(get_bots().get(bot_id)), "private" if is_private else "group", bot_id, plugin_config.bot_name)
    await send_command_image(db_session, image_bytes, workspace=workspace, identity=identity,
                             reply_to_id=str(get_message_id()), name="词云.png", recovery_enabled=pilot)


@frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    try:
        days = _parse_wordcloud_days(arg_text)
    except ValueError as e:
        await frequency.finish(str(e))

    words = await _collect_words_from_db(db_session, session_id, days=days, user_id=session.user.id)
    await db_session.commit()
    if not words:
        await frequency.finish("在指定时间内，没有说过话呢")

    image_bytes = (await asyncio.to_thread(_build_wordcloud_image, words)).getvalue()
    await _send_wordcloud_image(db_session, session, image_bytes)


group_frequency = on_command("群词频")


@group_frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    try:
        days = _parse_wordcloud_days(arg_text)
    except ValueError as e:
        await group_frequency.finish(str(e))

    words = await _collect_words_from_db(db_session, session_id, days=days, user_id=None)
    await db_session.commit()
    # Even if no words, return an empty wordcloud (original group_frequency didn't check emptiness)
    if not words:
        await group_frequency.finish("在指定时间内，没有消息可统计")

    image_bytes = (await asyncio.to_thread(_build_wordcloud_image, words)).getvalue()
    await _send_wordcloud_image(db_session, session, image_bytes)


def _consume_rag_task(session_id: str, task: asyncio.Task[None]) -> None:
    if _rag_vectorization_tasks.get(session_id) is task:
        _rag_vectorization_tasks.pop(session_id, None)
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception(f"会话 {session_id} 自主向量化任务异常")


async def _run_idle_rag_vectorization(session_id: str) -> None:
    while DB.enabled:
        last_activity = _rag_last_activity.get(session_id, time.monotonic())
        delay = RAG_IDLE_VECTORIZE_SECONDS - (time.monotonic() - last_activity)
        if delay > 0:
            await asyncio.sleep(delay)
            continue

        handled_activity = _rag_last_activity.get(session_id, last_activity)
        async with maintenance_gate.slot(wait=False) as admitted:
            if not admitted:
                await asyncio.sleep(RAG_MAINTENANCE_RETRY_SECONDS)
                continue
            try:
                async with get_session() as vector_session:
                    result = await process_and_vectorize_session_chats(
                        vector_session,
                        session_id,
                    )
                if result:
                    logger.info(f"自主向量化会话 {session_id} 完成，处理 {result['processed_groups']}/{result['total_groups']} 组")
                    if result["failed_groups"]:
                        logger.warning(f"自主向量化会话 {session_id} 尚有 {result['failed_groups']} 组失败，稍后重试")
                        await asyncio.sleep(RAG_MAINTENANCE_RETRY_SECONDS)
                        continue
            except Exception:
                logger.exception(f"自主向量化会话 {session_id} 失败，稍后重试")
                await asyncio.sleep(RAG_MAINTENANCE_RETRY_SECONDS)
                continue

        if _rag_last_activity.get(session_id, handled_activity) > handled_activity:
            continue
        _rag_last_activity.pop(session_id, None)
        return


def _schedule_rag_vectorization(session_id: str) -> None:
    if not DB.enabled:
        return
    _rag_last_activity[session_id] = time.monotonic()
    existing = _rag_vectorization_tasks.get(session_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _run_idle_rag_vectorization(session_id),
        name=f"ai-groupmate-rag:{session_id}",
    )
    _rag_vectorization_tasks[session_id] = task
    task.add_done_callback(
        lambda completed, current_session=session_id: _consume_rag_task(
            current_session,
            completed,
        )
    )


def _consume_rag_backfill_task(task: asyncio.Task[None]) -> None:
    global _rag_backfill_task
    if _rag_backfill_task is task:
        _rag_backfill_task = None
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("RAG 历史后台回填任务异常")


def _start_rag_backfill_task() -> None:
    global _rag_backfill_task
    if not DB.enabled or (_rag_backfill_task is not None and not _rag_backfill_task.done()):
        return
    _rag_backfill_task = asyncio.create_task(
        _run_delayed_rag_backfill(),
        name="ai-groupmate-rag-backfill",
    )
    _rag_backfill_task.add_done_callback(_consume_rag_backfill_task)


async def _run_delayed_rag_backfill() -> None:
    # 避开 Bot 启动、适配器连接和数据库迁移的高峰，防止争抢小连接池。
    await asyncio.sleep(RAG_BACKFILL_START_DELAY_SECONDS)
    await _run_rag_backfill()


@scheduler.scheduled_job("interval", hours=6, max_instances=1, coalesce=True, id="vectorize_chat")
async def vectorize_message_history():
    await _run_rag_backfill()


async def _run_rag_backfill() -> None:
    if not DB.enabled:
        logger.debug("Qdrant 未启用，跳过会话向量化任务")
        return
    async with maintenance_gate.slot(wait=False) as admitted:
        if not admitted:
            logger.info("其他维护任务正在运行，跳过本轮会话向量化")
            return

        started_at = time.perf_counter()
        try:
            async with get_session() as discovery_session:
                result = await discovery_session.execute(Select(ChatHistory.session_id.distinct()))
                session_ids = list(result.scalars().all())
                await discovery_session.commit()

            logger.info("开始向量化会话")
            for session_id in session_ids:
                try:
                    # A failed or cancelled session cannot poison subsequent
                    # sessions, and no session exists while this job waits for
                    # its maintenance slot.
                    async with get_session() as vector_session:
                        res = await process_and_vectorize_session_chats(vector_session, session_id)
                    if res:
                        logger.info(f"向量化会话 {res['session_id']} 成功，共处理 {res['processed_groups']}/{res['total_groups']} 组")
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

任务 B：如果是表情包，请生成可供多维检索的描述，必须尽量包含：
1. 【所有文字内容】：保持原文，不要只概括意思；
2. 【主体与形象】：可确认的角色/IP/人物/动物/梗模板名称；不确定身份时描述发色、服装、物种等可见特征，不要猜名字；
3. 【画面元素】：表情、动作、姿势、物体、场景、构图与画风；
4. 【梗与语用】：可确认的梗名、典故、反差或笑点，以及适合表达的态度、情绪和使用场景。

描述应简练但信息完整，方便按台词、形象、动作、画面或梗搜索。例如：“熊猫头流泪，双手捂脸，配文‘我太难了’，经典熊猫头诉苦梗，表达崩溃和无奈”。

请务必只返回合法的 JSON 格式，不要使用 Markdown 代码块：
{
    "is_meme": true,
    "description": "熊猫头流泪，配文'我太难了'"
}
"""


async def _process_media_vectorization(media_id: int) -> str:
    """处理单张图片；网络请求期间不占用 SQL 连接。"""
    if not DB.enabled:
        return "disabled"
    active_embedding_version = DB.media_embedding_version
    async with get_session() as db_session:
        media = await db_session.get(MediaStorage, media_id)
        if media is None:
            await db_session.commit()
            return "skipped"
        media_file_path = media.file_path
        existing_description = media.description
        needs_reindex = media.vectorized and media.embedding_version != active_embedding_version and existing_description != "[图片]"
        await db_session.commit()

    try:
        file_path = pic_dir / media_file_path
        try:
            img_data_uri = await asyncio.to_thread(_read_image_data_uri, file_path, media_file_path)
        except Exception as e:
            logger.error(f"读取图片失败 {media_id}: {e}")
            return "failed"

        if img_data_uri is None:
            if needs_reindex:
                # 旧集合已不再参与检索，原图缺失时也无法生成当前格式的向量。
                # 将版本推进到当前值，避免每轮重复扫描并逐条刷 Warning。
                await _mark_media_vectorized(
                    media_id,
                    embedding_version=active_embedding_version,
                )
                return "missing"
            logger.warning(f"文件不存在: {file_path}")
            await _mark_media_vectorized(media_id)
            return "skipped"

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
                    embedding_version=active_embedding_version,
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
                embedding_version=active_embedding_version,
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
    if not DB.enabled:
        logger.debug("Qdrant 未启用，跳过媒体向量化任务")
        return
    batch_size = plugin_config.media_vectorize_batch_size
    min_references = plugin_config.media_vectorize_min_references
    concurrency = plugin_config.media_vectorize_concurrency
    active_embedding_version = DB.media_embedding_version

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
                MediaStorage.embedding_version != active_embedding_version,
            )
            .order_by(MediaStorage.media_id)
            .limit(batch_size)
        )
        outdated_ids = list(outdated_res.scalars().all())
        await db_session.commit()

    media_ids = list(dict.fromkeys(pending_ids + outdated_ids))
    logger.info(f"本轮待处理新图片: {len(pending_ids)}，待重建旧表情包: {len(outdated_ids)}，并发数: {concurrency}")
    if not media_ids:
        return

    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(media_id: int) -> str:
        async with semaphore:
            return await _process_media_vectorization(media_id)

    results = await asyncio.gather(*(process_one(media_id) for media_id in media_ids))
    missing_count = results.count("missing")
    if missing_count:
        logger.warning(f"本轮跳过 {missing_count} 个文件缺失的历史表情包，已停止后续重建重试")
    logger.info(f"本轮图片处理完成：成功 {results.count('indexed')}，跳过 {results.count('skipped') + missing_count}，失败待重试 {results.count('failed')}")


@scheduler.scheduled_job("interval", minutes=10, max_instances=1, coalesce=True, id="vectorize_media")
async def vectorize_media():
    if not DB.enabled:
        logger.debug("Qdrant 未启用，跳过媒体向量化调度")
        return
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


@scheduler.scheduled_job("interval", minutes=35, max_instances=1, coalesce=True, id="clear_cache")
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
                        MediaStorage.created_at < datetime.datetime.now() - datetime.timedelta(days=30),
                    )
                )
                medias = list(result.scalars().all())
                media_files = [pic_dir / media.file_path for media in medias]
                for media in medias:
                    await cleanup_session.delete(media)
                await cleanup_session.commit()

            if media_files:
                deleted_files = await asyncio.to_thread(_batch_delete_files, media_files, "过期媒体")
                logger.info(f"成功清理 {len(media_files)} 个过期媒体记录（{deleted_files} 个文件）")

            # Materialize the known filenames and close the database session
            # before scanning/stat'ing the directory.
            async with get_session() as discovery_session:
                known_result = await discovery_session.execute(Select(MediaStorage.file_path))
                known_files = {str(row[0]) for row in known_result.all()}
                await discovery_session.commit()

            orphaned_count, deleted_count = await asyncio.to_thread(
                _delete_orphaned_files,
                pic_dir,
                known_files,
                datetime.timedelta(minutes=10),
            )
            if orphaned_count:
                logger.info(f"成功清理 {deleted_count}/{orphaned_count} 个孤立文件")
        finally:
            elapsed = time.perf_counter() - started_at
            logger.info(f"媒体清理任务结束，耗时 {elapsed:.2f}s")


# 群档案由 agent 工具按需维护，不再注册定时任务。
