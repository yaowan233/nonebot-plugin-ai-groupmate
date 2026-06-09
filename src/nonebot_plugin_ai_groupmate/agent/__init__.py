import re
import json
import uuid
import base64
import random
import asyncio
import difflib
import datetime
import mimetypes
import traceback
import collections
from typing import Any, Literal
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import jieba
from nonebot import require, get_plugin_config
from nonebot.adapters import Bot, Event
from pydantic import Field, BaseModel, field_validator
from simpleeval import simple_eval
from sqlalchemy import Select, desc, func, extract
from nonebot.log import logger
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch
from nonebot_plugin_orm import get_session
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_alconna import Target, UniMessage, message_reaction
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from nonebot_plugin_apscheduler import scheduler

from .graph import build_chat_graph
from ..model import ChatHistory, GroupMemory, MediaStorage, UserRelation, ChatHistorySchema
from ..config import Config, create_chat_llm, create_chat_openai
from ..memory import DB
from ..reply_guard import is_request_active
from .prompt_cache import build_system_messages

require("nonebot_plugin_localstore")

import nonebot_plugin_localstore as store

plugin_data_dir = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
plugin_path = Path(__file__).parent
with open(plugin_path / "上升.jpg", "rb") as f:
    up_pic = f.read()
with open(plugin_path / "下降.jpg", "rb") as f:
    down_pic = f.read()
plugin_config = get_plugin_config(Config).ai_groupmate
with open(Path(__file__).parent.parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]

ReactionMood = Literal[
    "like",
    "laugh",
    "surprise",
    "sad",
    "angry",
    "ok",
    "love",
    "question",
    "awkward",
    "clap",
    "plead",
    "thanks",
    "good_job",
    "shock",
    "smirk",
    "tease",
    "proud",
    "excited",
    "unhappy",
]

REACTION_EMOJI_MAP: dict[str, tuple[str, ...]] = {
    "like": ("76", "201", "389", "424"),
    "laugh": ("178", "182", "193", "387", "378"),
    "surprise": ("0", "180", "424"),
    "sad": ("5", "9", "15", "38", "49", "107", "194", "379", "382"),
    "angry": ("11", "326", "365"),
    "ok": ("124", "377", "381", "398"),
    "love": ("66", "305", "319", "383"),
    "question": ("32", "367"),
    "awkward": ("10", "27", "264"),
    "clap": ("99", "375"),
    "plead": ("106", "111"),
    "thanks": ("118", "63", "78", "409"),
    "good_job": ("356", "299", "306", "353", "380", "424"),
    "shock": ("26", "325"),
    "smirk": ("20", "101", "178"),
    "tease": ("102", "103", "178", "271"),
    "proud": ("4", "16", "306"),
    "excited": ("180", "400", "401"),
    "unhappy": ("15", "194"),
}
SCHEDULED_AGENT_HISTORY_LIMIT = 20

if plugin_config.tavily_api_key:
    tavily_search = TavilySearch(max_results=3, tavily_api_key=plugin_config.tavily_api_key)
else:
    tavily_search = None


@dataclass
class Context:
    session_id: str
    request_id: str | None = None


class ResponseMessage(BaseModel):
    """模型回复内容"""

    need_reply: bool = Field(description="是否需要回复")
    text: str | None = Field(description="回复文本(可选)")

    # 定义一个 field_validator 来处理 text 字段
    @field_validator("text", mode="before")
    @classmethod
    def convert_null_string_to_none(cls, value: Any) -> str | None:
        """
        在字段验证之前运行，将字符串 'null' (不区分大小写) 转换为 None。
        """
        # 检查值是否是字符串，并且在转换为小写后是否等于 'null'
        if isinstance(value, str) and value.lower() == "null":
            return None  # 返回 None，Pydantic 将其视为缺失或 null 值

        return value


@lru_cache
def get_flash_model() -> Any:
    return create_chat_openai(plugin_config, "flash")


@lru_cache
def get_chat_model() -> Any:
    return create_chat_llm(plugin_config)


async def check_if_should_reply(
    history_summary: str, current_msg: str, bot_name: str, is_private: bool = False
) -> bool:
    """
    使用 qwen-flash 快速判断是否需要回复
    """
    if is_private:
        scene_desc = "私聊"
        scene_extra = "3. 如果是无关的闲聊或者语意不通的消息，返回 NO。"
    else:
        scene_desc = "群聊"
        scene_extra = '3. 如果是群友之间的闲聊、无关的刷屏、或者语意不通的消息，返回 NO。'
    system_prompt = f"""
你是一个{scene_desc}消息过滤器。你的任务是判断{scene_desc}内的最新消息是否需要机器人 "{bot_name}" 进行回复。

判断规则：
1. 如果用户明显在向 "{bot_name}" 提问、求助或打招呼，返回 YES。
2. 如果用户在讨论 "{bot_name}" 相关的话题且期待回应，返回 YES。
{scene_extra}
4. 如果你不确定，返回 NO。

请仅输出 "YES" 或 "NO"，不要输出任何其他内容。
"""

    # 组合 Prompt
    # 只需要最近的一两条消息即可，不需要长篇大论的历史
    input_text = f"【最近上下文】\n{history_summary}\n\n【最新消息】\n{current_msg}\n\n请判断是否回复(YES/NO):"

    try:
        # 调用 Flash 模型
        resp = await get_flash_model().ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=input_text)]
        )
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            u = resp.usage_metadata
            logger.info(
                f"[Gatekeeper Token] 输入={u.get('input_tokens', 0)} 输出={u.get('output_tokens', 0)} "
                f"总计={u.get('total_tokens', 0)}"
            )
        if not isinstance(resp.content, str):
            return False

        content = resp.content.strip().upper()
        # 移除可能的标点符号
        content = content.replace(".", "").replace("。", "")

        return content == "YES"
    except Exception as e:
        logger.error(f"决策模型调用失败: {e}")
        return False  # 报错时默认不回，保守策略


# 如果想封装成自定义的 @tool，可以这样写:
@tool("search_web")
async def search_web(query: str, runtime: ToolRuntime[Context]) -> str:
    """
    用于搜索最新的实时信息。当你需要最新的事实信息、天气或新闻时使用。
    输入：需要搜索的内容。
    """
    if runtime.context.request_id is not None and not await is_request_active(
        runtime.context.session_id, runtime.context.request_id
    ):
        return "请求已过期，已取消搜索。"

    if not tavily_search:
        logger.error("没有配置 tavily_api_key, 无法进行搜索")
        return "没有配置 tavily_api_key, 无法进行搜索"
    results = await tavily_search.ainvoke(query)
    return results


@tool("search_history_context")
async def search_history_context(query: str, runtime: ToolRuntime[Context]) -> str:
    """
    搜索历史聊天记录。会返回某个时间段，半小时左右的聊天记录。当需要了解群内历史群内聊天记录或过往话题时使用
    输入：搜索关键信息或话题描述，这个语句直接从RAG数据库中进行混合搜索
    """
    if runtime.context.request_id is not None and not await is_request_active(
        runtime.context.session_id, runtime.context.request_id
    ):
        return "请求已过期，已取消搜索。"

    try:
        logger.info(f"大模型执行{runtime.context.session_id} RAG 搜索\n{query}")
        similar_msgs = await DB.search_chat(query, runtime.context.session_id)
        return similar_msgs if similar_msgs else "未找到相关历史记录"
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return "历史搜索失败"


def create_report_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str,
    user_name: str | None,
    llm_client: Any,
):
    """
    创建年度报告工具（限制在当前群聊 session_id 范围内）
    """

    @tool("generate_and_send_annual_report")
    async def generate_and_send_annual_report() -> str:
        """
        生成并发送当前群聊的年度报告。
        包含：个人在本群的统计、性格分析、全群排行榜以及Bot的好感度回顾。
        """
        if not user_id:
            return "当前没有可用于生成报告的用户信息。"

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消发送。"

        try:
            logger.info(f"开始生成用户 {user_name} 在群 {session_id} 的年度报告...")
            now = datetime.datetime.now()
            current_year = now.year

            stmt = Select(ChatHistory).where(
                ChatHistory.user_id == user_id,
                ChatHistory.session_id == session_id,
                extract("year", ChatHistory.created_at) == current_year,
            )
            all_msgs = (await db_session.execute(stmt)).scalars().all()

            if not all_msgs:
                return "用户本群今年暂无可用于年度报告的数据。请调用 reply_user 简短告知用户生成不了报告。"

            # 统计与采样
            text_msgs = [
                m.content for m in all_msgs if m.content_type == "text" and m.content
            ]
            total_count = len(all_msgs)

            # 采样 30 条让 LLM 分析 (只分析在这个群说的话)
            samples = (
                random.sample(text_msgs, min(len(text_msgs), 30)) if text_msgs else []
            )
            longest_msg = max(text_msgs, key=len) if text_msgs else "无"
            if len(longest_msg) > 60:
                longest_msg = longest_msg[:60] + "..."

            # 活跃时间
            active_hour_desc = "潜水员"
            if all_msgs:
                hours = [m.created_at.hour for m in all_msgs]
                top_hour = collections.Counter(hours).most_common(1)[0][0]
                active_hour_desc = f"{top_hour}点"

            async def get_rank_str(content_type=None, hour_limit=None):
                stmt = Select(
                    ChatHistory.user_id, func.count(ChatHistory.msg_id).label("c")
                ).where(
                    extract("year", ChatHistory.created_at) == current_year,
                    ChatHistory.session_id == session_id,
                )

                if content_type:
                    stmt = stmt.where(ChatHistory.content_type == content_type)
                if hour_limit:
                    stmt = stmt.where(
                        extract("hour", ChatHistory.created_at) < hour_limit
                    )

                # 核心修改：只 group_by user_id
                stmt = stmt.group_by(ChatHistory.user_id).order_by(desc("c")).limit(3)

                # 获取结果，此时是 List[(user_id, count)]
                rows = (await db_session.execute(stmt)).all()

                if not rows:
                    return "虚位以待"

                rank_items = []
                for uid, count in rows:
                    # 查询该用户最近的一条消息记录，取当时的名字
                    name_stmt = (
                        Select(ChatHistory.user_name)
                        .where(ChatHistory.user_id == uid)
                        .order_by(desc(ChatHistory.created_at))
                        .limit(1)
                    )

                    latest_name = (await db_session.execute(name_stmt)).scalar()

                    # 兜底：如果查不到名字（极少情况），用 ID 代替
                    display_name = latest_name if latest_name else f"用户{uid}"
                    rank_items.append(f"{display_name}({count})")
                return ", ".join(rank_items)

            rank_talk = await get_rank_str()
            rank_img = await get_rank_str(content_type="image")
            rank_night = await get_rank_str(hour_limit=5)

            # 只分析本群的文本
            stmt_text = (
                Select(ChatHistory.content)
                .where(
                    ChatHistory.session_id == session_id,
                    extract("year", ChatHistory.created_at) == current_year,
                    ChatHistory.content_type == "text",
                    ChatHistory.user_id == user_id,
                )
                .order_by(desc(ChatHistory.created_at))
            )

            rows = (await db_session.execute(stmt_text)).all()
            sample_text = "\n".join([r[0] for r in rows if r[0]])

            clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", sample_text)
            words = jieba.lcut(clean_text)
            filtered = [w for w in words if len(w) > 1 and w not in stop_words]
            hot_words_str = "、".join(
                [x[0] for x in collections.Counter(filtered).most_common(8)]
            )

            relation_stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
            relation = (await db_session.execute(relation_stmt)).scalar_one_or_none()

            favorability = 0
            impression_tags = []
            if relation:
                favorability = relation.favorability
                impression_tags = relation.tags if relation.tags else []

            # 格式化关系描述，喂给 LLM
            relation_desc = f"好感度: {favorability} (满分100), 印象标签: {', '.join(impression_tags)}"

            samples_text = "\n".join(samples)
            return f"""【年度报告素材】
请根据以下素材生成完整年度报告，并调用 `reply_user` 发送给用户；不要直接结束，也不要再调用年度报告工具。

【写作要求】
1. 不要使用 Markdown 标题、粗体或列表符号。
2. 可以使用 Emoji 和纯文本分隔线排版。
3. 语气像群友，不要像正式报告。
4. 根据好感度调整语气：>60 亲密宠溺，<0 傲娇毒舌，0-60 友善调侃。
5. 必须包含：标题行、基础数据、关系回顾、年度热词、群内风云榜、成分分析、{plugin_config.bot_name}寄语。
6. 成分分析要重点参考发言样本，写得具体一点。

【用户数据】
用户名: {user_name}
年份: {current_year}
累计发言: {total_count}
活跃时间: {active_hour_desc}
最长发言片段: {longest_msg}
年度热词: {hot_words_str}

【{plugin_config.bot_name} 与用户的关系】
{relation_desc}

【全群排行榜参考】
龙王榜: {rank_talk}
斗图榜: {rank_img}
熬夜榜: {rank_night}

【用户发言样本】
{samples_text}
"""

        except Exception as e:
            logger.error(f"收集年度报告素材失败: {e}")
            print(traceback.format_exc())
            return f"收集年度报告素材出错: {e}"

    return generate_and_send_annual_report


def create_similar_meme_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str | None,
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
            # 1. 从 ChatHistory 查找指定或最近的图片消息
            base_stmt = (
                Select(ChatHistory)
                .where(
                    ChatHistory.session_id == session_id,
                    ChatHistory.content_type == "image",
                )
                .order_by(desc(ChatHistory.created_at))
            )
            if target_msg_id:
                # 指定了 id：按 id 精确查找
                stmt = base_stmt.where(
                    ChatHistory.content.contains(f"id: {target_msg_id}\n")
                ).limit(1)
            elif user_id:
                # 未指定 id：fallback 到当前用户最近发的图
                stmt = base_stmt.where(ChatHistory.user_id == user_id).limit(1)
            else:
                # 兜底：群内最近一张图
                stmt = base_stmt.limit(1)
            result = await db_session.execute(stmt)
            msg = result.scalar_one_or_none()

            if not msg:
                return "本群近期没有发送过图片，无法进行相似搜索。"

            # 2. 获取该图片的物理路径或 URL (需要根据你的表结构调整字段名)
            stmt = Select(MediaStorage).where(MediaStorage.media_id == msg.media_id)
            media_obj = (await db_session.execute(stmt)).scalar_one_or_none()

            if not media_obj or not media_obj.file_path:  # 假设你的路径存放在 file_path
                return "无法找到原图文件，无法进行分析。"

            # 调用 database.py 中的新接口
            # search_similar_meme(id) -> 返回相似图片的 ID 列表
            pic_ids = await DB.search_similar_meme(str(pic_dir / media_obj.file_path))

            if not pic_ids:
                logger.info(f"未找到相似图片, source_id: {msg.media_id}")
                return "没有搜索到相似图片"

            # 3. 从 SQL 数据库获取图片详情
            # 虽然 Qdrant 返回了 ID，但 LLM 还是需要知道这些图大概是啥（描述），以便决定发哪张
            images_info = []

            # 批量查询 SQL (比循环查更高效)
            stmt = Select(MediaStorage).where(MediaStorage.media_id.in_(pic_ids))
            rows = (await db_session.execute(stmt)).scalars().all()

            # 为了保持顺序（Qdrant返回的是按相似度排序的），我们重新对齐一下
            # 建立 id -> obj 映射
            media_map = {m.media_id: m for m in rows}

            for pid in pic_ids:
                if pid in media_map:
                    media_obj = media_map[pid]
                    images_info.append(
                        {
                            "pic_id": str(pid),  # 转字符串方便模型理解
                            # 注意：如果是新图片，description 可能是 "[图片]" 占位符
                            # 如果是迁移过来的旧图片，则是真实的描述
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


def create_reply_tool(
    db_session,
    session_id: str,
    request_id: str | None = None,
    interface: QryItrface | None = None,
    *,
    send_target: Target | None = None,
):
    """
    核心工具：用于发送消息。
    """

    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _semantic_similarity(a: str, b: str) -> float:
        """粗粒度语义相似度，兼顾中文短句场景。"""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        seq_ratio = difflib.SequenceMatcher(None, a, b).ratio()

        a_tokens = {t for t in jieba.lcut(a) if t.strip()}
        b_tokens = {t for t in jieba.lcut(b) if t.strip()}
        if not a_tokens or not b_tokens:
            return seq_ratio

        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        jaccard = inter / union if union else 0.0
        return max(seq_ratio, jaccard)

    def _dedupe_consecutive_lines(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text
        deduped: list[str] = []
        for line in lines:
            if deduped and deduped[-1] == line:
                continue
            deduped.append(line)
        return "\n".join(deduped)

    @tool("reply_user")
    async def reply_user(content: str) -> str:
        """
        向当前群聊发送文本回复。
        注意：如果你想对用户说话，必须调用这个工具。不要直接返回文本。
        Args:
            content: 你想发送的内容。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消发送。"

        if not content or not content.strip():
            return "内容为空，未发送。"

        try:
            content = _dedupe_consecutive_lines(content.strip())
            normalized_content = _normalize_text(content)

            # 避免短时间内重复发送相同文本（常见于模型多次调用 reply_user）
            latest_bot_msg = (
                (
                    await db_session.execute(
                        Select(ChatHistory)
                        .where(
                            ChatHistory.session_id == session_id,
                            ChatHistory.content_type == "bot",
                        )
                        .order_by(ChatHistory.msg_id.desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if latest_bot_msg:
                _, _, latest_body = _parse_msg_meta(latest_bot_msg.content)
                latest_normalized = _normalize_text(
                    latest_body or latest_bot_msg.content
                )
                recent = (
                    datetime.datetime.now() - latest_bot_msg.created_at
                    <= datetime.timedelta(seconds=90)
                )
                similarity = _semantic_similarity(latest_normalized, normalized_content)
                if recent and similarity >= 0.9:
                    logger.info(
                        f"检测到近义重复回复(相似度={similarity:.2f})，已自动跳过"
                    )
                    return "检测到重复回复，已跳过发送。"

            name_to_id: dict[str, str] = {}
            if interface is not None:
                try:
                    members = await interface.get_members(SceneType.GROUP, session_id)
                    for member in members:
                        target_id = str(member.id)
                        aliases = {
                            getattr(member, "name", None),
                            getattr(member, "nick", None),
                            getattr(getattr(member, "user", None), "name", None),
                            getattr(getattr(member, "user", None), "nick", None),
                        }
                        for alias in aliases:
                            if alias:
                                name_to_id[str(alias)] = target_id
                except Exception as e:
                    logger.warning(f"获取群成员失败，降级为纯文本发送: {e}")

            at_pattern = re.compile(r"@([^\s@]+)")
            punctuation = "，。,.!！?？:：;；、)）]\"'”’"
            message: UniMessage | None = None

            def append_text(text: str):
                nonlocal message
                if not text:
                    return
                if message is None:
                    message = UniMessage.text(text)
                else:
                    message = message.text(text)

            def append_at(target_id: str) -> bool:
                nonlocal message
                try:
                    if message is None:
                        message = UniMessage.at(target_id)
                    else:
                        message = message.at(target_id)
                    return True
                except Exception:
                    return False

            cursor = 0
            for match in at_pattern.finditer(content):
                start, end = match.span()
                raw_name = match.group(1)
                mention_name = raw_name
                suffix = ""
                while mention_name and mention_name[-1] in punctuation:
                    suffix = mention_name[-1] + suffix
                    mention_name = mention_name[:-1]

                target_id = name_to_id.get(mention_name)
                if not target_id:
                    continue

                append_text(content[cursor:start])
                if not append_at(target_id):
                    append_text("@" + mention_name)
                append_text(suffix)
                cursor = end

            append_text(content[cursor:])

            if request_id is not None and not await is_request_active(
                session_id, request_id
            ):
                return "请求已过期，已取消发送。"

            # 1. 实际发送消息 (Side Effect)
            outgoing = message or UniMessage.text(content)
            res = await (
                outgoing.send(target=send_target)
                if send_target is not None
                else outgoing.send()
            )
            msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n" + content,
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"Bot已回复: {content}")
            return f"消息已发送，你刚才发送的内容是：\n{content}"
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            await db_session.rollback()
            return f"发送失败: {e}"

    return reply_user


def create_search_meme_tool(db_session, session_id: str, request_id: str | None):
    """
    创建一个带数据库会话的表情包搜索工具

    Args:
        db_session: 数据库会话

    Returns:
        配置好的 tool 函数
    """

    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """
        根据描述搜索合适的表情包图片。

        这个工具只负责搜索，不会发送图片。搜索后会返回匹配的图片列表及其详细描述。
        你可以查看这些图片的描述，判断是否合适，然后使用 send_meme_image 工具发送。

        输入：表情包的描述（具体画面+情绪）：
        - "二次元猫耳女孩，皱眉张嘴，生气抗议"
        - "卡通小鸡，大笑得意，双手举起"
        - "熊猫头，流泪叹气，悲伤无奈"
        - "一只白色的猫咪，翻白眼，无语的表情"
        返回：包含图片ID和对应描述的JSON字符串
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

            # 从数据库获取每张图片的详细信息
            images_info = []
            for pic_id in pic_ids[:5]:  # 只返回前5张，避免信息过多
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
            logger.error(
                f"表情包搜索失败: {repr(e)}"
            )  # 使用 repr() 可以看到异常类型，比 str() 更详细
            logger.error(traceback.format_exc())  # 打印完整报错路径
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
):
    """
    创建一个带上下文的表情包发送工具

    Args:
        db_session: 数据库会话
        session_id: 会话ID

    Returns:
        配置好的 tool 函数
    """

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str) -> str:
        """
        发送表情包图片到聊天中。

        你需要先使用 search_meme_image 搜索图片，然后决定是否发送。

        参数：
        - pic_id: 图片ID（从 search_meme_image 获取，必填）
        返回：发送状态信息
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消发送。"

        try:
            # 模型有时会把思维链混入 pic_id，用正则提取第一个数字
            import re as _re

            _match = _re.search(r"\d+", pic_id)
            if not _match:
                return f"发送表情包失败: 无法从 pic_id 中提取有效数字: {pic_id!r}"
            selected_pic_id = int(_match.group())
            logger.info(f"使用指定的图片ID: {selected_pic_id}")

            # 从数据库获取图片信息
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

            # 读取图片数据
            pic_data = pic_path.read_bytes()
            description = pic.description

            if request_id is not None and not await is_request_active(
                session_id, request_id
            ):
                return "请求已过期，已取消发送。"

            # 发送图片
            message = UniMessage.image(raw=pic_data)
            res = await (
                message.send(target=send_target)
                if send_target is not None
                else message.send()
            )
            # 记录发送历史（不在工具内提交，由外层 session 统一管理）
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=(
                    f"id: {res.msg_ids[-1]['message_id']}\n"
                    f"发送了图片，图片描述是: {description}\n"
                    f"图片文件: {pic.file_path}"
                ),
                user_name=plugin_config.bot_name,
                media_id=pic.media_id,
            )
            db_session.add(chat_history)
            logger.info(f"id:{res.msg_ids}\n" + f"发送表情包: {description}")
            return f"已成功发送表情包: {description}"

        except Exception as e:
            logger.error(f"发送表情包失败: {e}")
            return f"发送表情包失败: {str(e)}"

    return send_meme_image


def _is_onebot_context(bot: Bot | None, event: Event | None) -> bool:
    candidates: list[str] = []
    for obj in (bot, event):
        if obj is None:
            continue
        candidates.extend(
            [
                type(obj).__module__,
                type(obj).__qualname__,
                str(getattr(obj, "type", "")),
            ]
        )

    adapter = getattr(bot, "adapter", None) if bot is not None else None
    if adapter is not None:
        candidates.extend([type(adapter).__module__, type(adapter).__qualname__])
        get_name = getattr(adapter, "get_name", None)
        if callable(get_name):
            try:
                candidates.append(str(get_name()))
            except Exception:
                pass

    return any("onebot" in candidate.lower() for candidate in candidates)


def create_reaction_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str | None,
    bot: Bot | None,
    event: Event | None,
):
    """
    创建跨适配器的消息表情回复工具，底层使用 nonebot_plugin_alconna.message_reaction。
    """

    def _normalize_reaction_message_id(raw_message_id: str | None) -> str | None:
        if not raw_message_id:
            return None

        candidate = str(raw_message_id).strip()
        if not candidate:
            return None

        if candidate.lower() in {
            "current_event",
            "current",
            "event",
            "none",
            "null",
            "system",
        }:
            logger.debug(f"忽略无效表情回复目标消息占位符: {candidate}")
            return None

        meta_match = re.search(r"\bid[:：]\s*([A-Za-z0-9_.-]{1,128})", candidate)
        if meta_match:
            return meta_match.group(1)

        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate):
            return candidate

        logger.debug(f"忽略不像消息 id 的表情回复目标: {candidate[:80]}")
        return None

    @tool("add_message_reaction")
    async def add_message_reaction(
        mood: ReactionMood,
        target_msg_id: str | None = None,
        delete: bool = False,
        emoji: str | None = None,
        count: int = 1,
    ) -> str:
        """
        给某条消息添加或取消表情回复。

        当只需要用一个表情表达态度（例如点赞、笑、惊讶、无语）时使用这个工具，
        不要再调用 reply_user 发送重复文本。

        Args:
            mood: 表情回复表达的态度。基础 mood: like/laugh/surprise/sad/angry/ok。扩展 mood: love/question/awkward/clap/plead/thanks/good_job/shock/smirk/tease/proud/excited/unhappy。
            target_msg_id: 目标消息 id，来自聊天记录里的 "id: xxxxx"。通常不要传；不传时默认给当前触发 bot 回复的这条消息添加。
            delete: 是否取消这个表情回复，默认 False。
            emoji: 可选的适配器原始表情 ID/名称覆盖值。通常不要传，除非明确知道平台支持的表情 ID。
            count: 同一 mood 下连续添加几个不同表情，范围 1-3。传了 emoji 时只添加该 emoji。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消表情回复。"

        mood_key = str(mood).strip()
        mood_emojis = REACTION_EMOJI_MAP.get(mood_key)
        reaction_count = max(1, min(int(count or 1), 3))
        if emoji:
            reaction_emojis = [str(emoji).strip()]
        elif mood_emojis:
            reaction_emojis = random.sample(
                list(mood_emojis), k=min(reaction_count, len(mood_emojis))
            )
        else:
            reaction_emojis = []
        reaction_emojis = [item for item in reaction_emojis if item]

        if not reaction_emojis:
            supported = ", ".join(REACTION_EMOJI_MAP)
            return f"表情回复失败: 未知 mood {mood_key!r}，可选值: {supported}"

        if bot is None or event is None:
            return "表情回复失败: 缺少 bot/event 上下文，无法调用 alconna message_reaction。"
        if not _is_onebot_context(bot, event):
            return "表情回复失败: 当前适配器不是 OneBot，不支持消息表情回复。"

        message_id = _normalize_reaction_message_id(target_msg_id)
        if not message_id:
            logger.debug("未指定表情回复目标消息，使用当前触发事件的消息 id")

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消表情回复。"

        async def _apply_reaction(reaction_emoji: str, target_message_id: str | None) -> None:
            await message_reaction(
                reaction_emoji,
                message_id=target_message_id,
                event=event,
                bot=bot,
                delete=delete,
            )

        try:
            for reaction_emoji in reaction_emojis:
                try:
                    await _apply_reaction(reaction_emoji, message_id)
                except Exception as e:
                    if message_id and "msg not found" in str(e).lower():
                        logger.warning(
                            f"表情回复目标消息 {message_id} 不存在，回退到当前触发事件消息"
                        )
                        message_id = None
                        await _apply_reaction(reaction_emoji, None)
                    else:
                        raise
            action = "取消" if delete else "添加"
            reacted_message_desc = f"消息 {message_id}" if message_id else "当前触发消息"
            reaction_emoji_text = ",".join(reaction_emojis)
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id: system\n已对{reacted_message_desc} {action}表情回复: mood={mood_key}, emoji={reaction_emoji_text}",
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"已对{reacted_message_desc} {action}表情回复 mood={mood_key}, emoji={reaction_emoji_text}")
            return f"已对{reacted_message_desc} {action}表情回复 mood={mood_key}, emoji={reaction_emoji_text}"
        except Exception as e:
            logger.error(f"表情回复工具执行失败: {e}")
            return f"表情回复失败: {e}"

    return add_message_reaction


@tool("finish", return_direct=True)
def finish() -> str:
    """
    结束本次对话。当你已经完成所有回复（发送文字或图片）后，必须调用此工具。
    调用后对话立即结束，不能再发送任何内容。
    """
    return ""


@tool("calculate_expression")
def calculate_expression(expression: str) -> str:
    """
    一个用于精确执行数学计算的计算器。
    当你需要执行四则运算、代数计算、指数、对数或三角函数等复杂数学任务时使用。

    输入：一个标准的数学表达式字符串，例如 "45 * (2 + 3) / 7" 或 "math.sqrt(9) + math.log(10)".
    输出：计算结果的字符串形式。

    注意：可以使用如 math.sqrt() (开方), math.log() (自然对数), math.pi (圆周率) 等标准数学函数。
    """
    try:
        result = simple_eval(expression)
        # 返回格式化的结果，最多保留10位小数
        return (
            f"计算结果是：{result:.10f}" if isinstance(result, float) else str(result)
        )

    except Exception as e:
        return f"计算失败。请检查表达式是否正确，错误信息: {e}"


async def _send_scheduled_text(
    session_id: str,
    content: str,
    *,
    is_private: bool,
    bot_id: str | None,
) -> None:
    try:
        target = Target(
            id=session_id,
            private=is_private,
            self_id=bot_id,
        )
        result = await UniMessage.text(content).send(target=target)

        msg_id = "unknown"
        if result.msg_ids:
            raw_msg_id = result.msg_ids[-1].get("message_id") or result.msg_ids[-1].get("msg_id")
            if raw_msg_id is not None:
                msg_id = str(raw_msg_id)

        async with get_session() as db_session:
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n" + content,
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()

        logger.info(f"[定时任务] 已发送到 {session_id}: {content}")
    except Exception as e:
        logger.error(f"[定时任务] 发送失败 {session_id}: {e}")


async def _run_scheduled_agent_task(
    session_id: str,
    task: str,
    *,
    is_private: bool,
    bot_id: str | None,
) -> None:
    try:
        async with get_session() as db_session:
            rows = (
                (
                    await db_session.execute(
                        Select(ChatHistory)
                        .where(ChatHistory.session_id == session_id)
                        .order_by(ChatHistory.msg_id.desc())
                        .limit(SCHEDULED_AGENT_HISTORY_LIMIT)
                    )
                )
                .scalars()
                .all()
            )
            history = [ChatHistorySchema.model_validate(row) for row in rows[::-1]]

            graph, _ = await create_chat_graph(
                db_session,
                session_id,
                None,
                plugin_config.bot_name,
                plugin_config.bot_name,
                history,
                None,
                bot_id,
                None,
                None,
                is_private=is_private,
            )

            prompt = f"""
【定时任务触发】
这是之前安排的定时 agent 任务，现在已经到执行时间。

【任务内容】
{task}

【执行要求】
- 你必须通过工具完成任务，不要直接输出正文。
- 如果任务只是提醒/转告，调用 `reply_user`。
- 如果任务要求查最新信息，先调用 `search_web`，再调用 `reply_user`。
- 如果任务要求发送表情包图片，先调用 `search_meme_image` 或 `search_similar_meme_by_id`，再调用 `send_meme_image`。
- 定时任务没有可用的原始消息事件，不要调用 `add_message_reaction`。
- 任务完成后调用 `finish`。
"""

            final_messages = format_chat_history(history, max_inline_images=0) + [
                HumanMessage(content=prompt)
            ]
            await graph.ainvoke({
                "messages": final_messages,
                "session_id": session_id,
                "request_id": None,
                "reply_count": 0,
                "tool_count": 0,
                "reply_this_round": 0,
                "reaction_this_round": 0,
                "called_finish": 0,
            })
            await db_session.commit()
        logger.info(f"[定时Agent任务] 已执行 {session_id}: {task}")
    except Exception as e:
        logger.exception(f"[定时Agent任务] 执行失败 {session_id}: {e}")


def create_schedule_message_tool(
    session_id: str,
    request_id: str | None,
    *,
    is_private: bool,
    bot_id: str | None,
):
    @tool("schedule_message")
    async def schedule_message(
        content: str,
        delay_minutes: float = 0,
        delay_hours: float = 0,
    ) -> str:
        """
        安排 bot 在几分钟或几小时后向当前群聊/私聊发送一条文本消息。
        Args:
            content: 到点后要发送的文本内容。
            delay_minutes: 延迟多少分钟，可以是小数。
            delay_hours: 延迟多少小时，可以和 delay_minutes 同时使用。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消定时任务。"

        content = content.strip()
        if not content:
            return "定时消息内容为空，未创建任务。"

        delay_seconds = delay_hours * 3600 + delay_minutes * 60
        if delay_seconds <= 0:
            return "延迟时间必须大于 0。"
        if delay_seconds < 10:
            return "延迟时间太短，至少需要 10 秒。"
        if delay_seconds > 7 * 24 * 3600:
            return "延迟时间太长，当前最多支持 7 天内的定时消息。"

        run_at = datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)
        job_id = f"ai_groupmate_schedule_{session_id}_{uuid.uuid4().hex}"

        scheduler.add_job(
            _send_scheduled_text,
            "date",
            id=job_id,
            run_date=run_at,
            kwargs={
                "session_id": session_id,
                "content": content,
                "is_private": is_private,
                "bot_id": bot_id,
            },
            misfire_grace_time=300,
        )

        return f"定时任务已创建，将在 {run_at.strftime('%Y-%m-%d %H:%M:%S')} 发送：{content}"

    return schedule_message


def create_schedule_agent_task_tool(
    session_id: str,
    request_id: str | None,
    *,
    is_private: bool,
    bot_id: str | None,
):
    @tool("schedule_agent_task")
    async def schedule_agent_task(
        task: str,
        delay_minutes: float = 0,
        delay_hours: float = 0,
    ) -> str:
        """
        安排 bot 在几分钟或几小时后重新进入 agent，并允许到点后调用可用工具完成任务。

        Args:
            task: 到点后要完成的任务描述，例如“查一下明天上海天气并提醒我带伞”。
            delay_minutes: 延迟多少分钟，可以是小数。
            delay_hours: 延迟多少小时，可以和 delay_minutes 同时使用。
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消定时任务。"

        task = task.strip()
        if not task:
            return "定时 agent 任务内容为空，未创建任务。"

        delay_seconds = delay_hours * 3600 + delay_minutes * 60
        if delay_seconds <= 0:
            return "延迟时间必须大于 0。"
        if delay_seconds < 10:
            return "延迟时间太短，至少需要 10 秒。"
        if delay_seconds > 7 * 24 * 3600:
            return "延迟时间太长，当前最多支持 7 天内的定时 agent 任务。"

        run_at = datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)
        job_id = f"ai_groupmate_agent_schedule_{session_id}_{uuid.uuid4().hex}"

        scheduler.add_job(
            _run_scheduled_agent_task,
            "date",
            id=job_id,
            run_date=run_at,
            kwargs={
                "session_id": session_id,
                "task": task,
                "is_private": is_private,
                "bot_id": bot_id,
            },
            misfire_grace_time=300,
        )

        return f"定时 agent 任务已创建，将在 {run_at.strftime('%Y-%m-%d %H:%M:%S')} 执行：{task}"

    return schedule_agent_task


def create_mute_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    interface: QryItrface | None,
    bot_id: str | None,
):
    """
    创建禁言工具（仅在bot是管理员时可用）
    """

    @tool("mute_user")
    async def mute_user(target_user_name: str, duration_seconds: int, reason: str) -> str:
        """
        禁言指定用户。仅在bot是管理员或群主时可用。
        
        参数:
        - target_user_name: 要禁言的用户昵称（从聊天记录中获取）
        - duration_seconds: 禁言时长（秒），最多2592000秒(30天)，0表示解除禁言
        - reason: 禁言原因（必填，用于记录和说明）
        
        返回: 操作结果描述
        """
        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消操作。"

        if not interface:
            return "无法获取群成员信息接口，禁言失败。"
        
        if not bot_id:
            return "无法获取bot ID，禁言失败。"

        try:
            # 验证禁言时长
            if duration_seconds < 0 or duration_seconds > 2592000:
                return "禁言时长必须在0-2592000秒(30天)之间。"

            # 获取所有群成员
            members = await interface.get_members(SceneType.GROUP, session_id)
            
            # 查找bot自己的权限
            bot_member = None
            target_member = None
            
            for member in members:
                # 使用bot_id来判断，而不是bot_name
                if str(member.id) == str(bot_id):
                    bot_member = member
                # 检查是否是目标用户（通过昵称匹配）
                member_name = member.nick or (member.user.name if member.user else None)
                if member_name == target_user_name:
                    target_member = member
            
            # 检查bot权限
            if not bot_member:
                return "无法获取bot的群成员信息。"
            
            bot_role = getattr(getattr(bot_member, "role", None), "name", None)
            if bot_role not in {"owner", "admin"}:
                return "bot不是管理员或群主，无法执行禁言操作。"
            
            # 检查目标用户是否存在
            if not target_member:
                return f"未找到用户 '{target_user_name}'，请确认昵称是否正确。"
            
            # 检查目标用户权限，避免禁言管理员
            target_role = getattr(getattr(target_member, "role", None), "name", None)
            if target_role in {"owner", "admin"}:
                return f"无法禁言管理员或群主 '{target_user_name}'。"
            
            # 执行禁言操作
            # 注意：这里需要调用适配器的API，不同平台可能有不同的方法
            # 我们尝试通过 interface 来操作
            try:
                from nonebot import get_bot
                bot = get_bot()
                
                # 对于OneBot v11/v12，使用set_group_ban
                target_user_id = str(target_member.id)
                if hasattr(bot, "set_group_ban"):
                    await bot.set_group_ban(
                        group_id=int(session_id),
                        user_id=int(target_user_id),
                        duration=duration_seconds
                    )
                else:
                    return "当前适配器不支持禁言功能。"
                
                action = "解除禁言" if duration_seconds == 0 else f"禁言 {duration_seconds} 秒"
                logger.info(f"已{action}用户 {target_user_name}（{target_user_id}），原因: {reason}")

                chat_history = ChatHistory(
                    session_id=session_id,
                    user_id=plugin_config.bot_name,
                    content_type="bot",
                    content=f"id: system\n已执行禁言操作: {action}用户 '{target_user_name}'。原因: {reason}",
                    user_name=plugin_config.bot_name,
                )
                db_session.add(chat_history)

                return f"已成功{action}用户 '{target_user_name}'。原因: {reason}"
                
            except Exception as api_err:
                logger.error(f"调用禁言API失败: {api_err}")
                return f"禁言操作失败: {str(api_err)}"
            
        except Exception as e:
            logger.error(f"禁言工具执行失败: {e}")
            print(traceback.format_exc())
            return f"禁言失败: {str(e)}"

    return mute_user


def create_relation_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str,
    user_name: str | None,
):
    """
    创建绑定了特定用户的关系管理工具 (支持增删 Tag)
    """

    @tool("update_user_impression")
    async def update_user_impression(
        score_change: int,
        reason: str,
        add_tags: list[str] | str | None = None,
        remove_tags: list[str] | str | None = None,
    ) -> str:
        """
        更新对当前对话用户的好感度和印象标签。
        当用户的言行让你产生情绪波动，或者你发现旧的印象不再准确时调用。

        参数:
        - score_change: 好感度变化值（正数加分，负数扣分）。
        - reason: 变更原因（必填）。
        - add_tags: 需要新增的印象标签列表。例如 ["爱玩原神", "很幽默"]。
        - remove_tags: 需要移除的旧标签列表（用于修正印象或删除错误的标签）。例如 ["内向"]。

        返回: 更新后的状态描述
        """
        if not user_id:
            return "当前没有可更新画像的用户信息。"

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消更新。"

        def normalize_tags(value: list[str] | str | None) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return []
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return [tag.strip() for tag in value.split(",") if tag.strip()]
                if isinstance(parsed, list):
                    return [str(tag).strip() for tag in parsed if str(tag).strip()]
                if isinstance(parsed, str) and parsed.strip():
                    return [parsed.strip()]
                return []
            return [str(tag).strip() for tag in value if str(tag).strip()]

        add_tags = normalize_tags(add_tags)
        remove_tags = normalize_tags(remove_tags)

        async with get_session() as session:
            try:
                # 1. 查询或初始化记录
                stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
                result = await session.execute(stmt)
                relation = result.scalar_one_or_none()

                if not relation:
                    relation = UserRelation(
                        user_id=user_id,
                        user_name=user_name or "",
                        favorability=0,
                        tags=[],
                    )
                    session.add(relation)
                else:
                    await session.refresh(relation, attribute_names=["tags"])

                # 2. 处理好感度
                old_score = relation.favorability

                final_change = score_change

                # 【救赎机制】：当好感度低于 -60 且 试图加分时，效果翻倍并额外奖励
                if old_score < -60 and score_change > 0:
                    final_change = int(score_change * 1.5) + 5
                    logger.info(
                        f"触发救赎机制：原始分 {score_change} -> 修正分 {final_change}"
                    )

                # 【破防机制】：当好感度高于 80 且 试图扣分时，伤害加深 (可选)
                elif old_score > 80 and score_change < 0:
                    final_change = int(score_change * 1.2) - 2
                    logger.info(
                        f"触发破防机制：原始分 {score_change} -> 修正分 {final_change}"
                    )

                relation.favorability += final_change
                relation.favorability = max(-100, min(100, relation.favorability))
                # 3. 处理标签
                # 获取现有标签的副本
                current_tags = list(relation.tags) if relation.tags else []

                # 执行移除操作 (处理 modify 的前半部分)
                if remove_tags:
                    current_tags = [
                        tag for tag in current_tags if tag not in remove_tags
                    ]

                # 执行新增操作
                if add_tags:
                    for tag in add_tags:
                        if tag not in current_tags:
                            current_tags.append(tag)

                # 限制标签总数，防止Token爆炸 (例如最多保留 8 个，保留最新的)
                if len(current_tags) > 8:
                    current_tags = current_tags[-8:]

                # 赋值回数据库对象
                relation.tags = current_tags
                relation.user_name = user_name or ""  # 同步更新昵称
                favorability = relation.favorability

                if request_id is not None and not await is_request_active(
                    session_id, request_id
                ):
                    await session.rollback()
                    return "请求已过期，已取消更新。"

                await session.commit()

                # 发送好感度变化图片：仅当跨越关系分段边界时才发
                # 分段边界与 UserRelation.get_status_desc() 保持一致
                _BOUNDARIES = (-70, -40, -15, 5, 25, 50, 70, 90)
                _TIER_NAMES = (
                    "死敌/拉黑",
                    "厌恶/仇视",
                    "冷淡/防备",
                    "陌生/普通",
                    "有点熟",
                    "朋友/熟人",
                    "好朋友",
                    "亲密/死党",
                    "最喜欢的人",
                )

                def _tier(score: int) -> int:
                    for i, b in enumerate(_BOUNDARIES):
                        if score < b:
                            return i
                    return len(_BOUNDARIES)

                old_tier = _tier(old_score)
                new_tier = _tier(favorability)
                if old_tier != new_tier:
                    try:
                        if new_tier > old_tier:
                            tip = f"好感度提升！现在的关系是：{_TIER_NAMES[new_tier]}"
                            res = await UniMessage.image(raw=up_pic).text(tip).send()
                        else:
                            tip = f"好感度下降…现在的关系是：{_TIER_NAMES[new_tier]}"
                            res = await UniMessage.image(raw=down_pic).text(tip).send()
                        msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
                        chat_history = ChatHistory(
                            session_id=session_id,
                            user_id=plugin_config.bot_name,
                            content_type="bot",
                            content=f"id: {msg_id}\n{tip}",
                            user_name=plugin_config.bot_name,
                        )
                        session.add(chat_history)
                    except Exception as send_err:
                        logger.warning(f"发送好感度图片失败: {send_err}")

                # 构建反馈信息
                tag_msg = ""
                if add_tags or remove_tags:
                    tag_msg = f"，标签变更(新增:{add_tags}, 移除:{remove_tags})"

                log_msg = (
                    f"好感度 {old_score}->{favorability}{tag_msg} (原因: {reason})"
                )
                logger.info(f"用户[{user_name}]画像更新: {log_msg}")

                return (
                    f"画像已更新。当前好感度: {favorability}，当前标签: {current_tags}"
                )

            except Exception as e:
                logger.error(f"关系更新失败: {e}")
                print(traceback.format_exc())
                return f"数据库错误: {str(e)}"
        return None

    return update_user_impression


tools = [search_web, search_history_context, calculate_expression]

ACTIVE_THREAD_TTL = datetime.timedelta(minutes=10)
ACTIVE_THREAD_MAX_MESSAGES = 24


@dataclass
class ActiveConversationThread:
    messages: list[BaseMessage]
    last_msg_id: int
    updated_at: datetime.datetime


active_conversation_threads: dict[str, ActiveConversationThread] = {}


async def get_user_relation_context(
    db_session, user_id: str, user_name: str | None
) -> str:
    """获取用户关系上下文Prompt"""
    try:
        stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
        result = await db_session.execute(stmt)
        relation = result.scalar_one_or_none()

        if not relation:
            return f"""
【人际关系】
当前对象：{user_name}
状态：陌生人 (好感度 0)
印象：无
策略：保持礼貌，通过对话了解对方。
"""
        if relation.favorability < -70:
            strategy = "策略：你把他当死敌，几乎不想和他说话。回复极其敷衍或直接无视，除非他有非常诚恳的道歉和实质性补偿，否则不会软化。"
        elif relation.favorability < -40:
            strategy = "策略：你很讨厌他，语气带刺，回复简短冷漠。不主动搭话，被问也尽量敷衍了事。如果他**诚恳道歉**或**做出补偿**，可以考虑给一点机会。"
        elif relation.favorability < -15:
            strategy = "策略：你对他有些防备，保持距离。礼貌但疏冷，不想深聊，话不多。"
        elif relation.favorability < 5:
            strategy = "策略：普通陌生人，正常礼貌对话，不特别亲近也不排斥。"
        elif relation.favorability < 25:
            strategy = "策略：有点印象了，偶尔可以开个小玩笑，稍微愿意多聊几句。"
        elif relation.favorability < 50:
            strategy = "策略：朋友关系，聊天比较随意自然，会主动回应，偶尔关心一下。"
        elif relation.favorability < 70:
            strategy = (
                "策略：好朋友，说话轻松亲近，会记住对方说过的事，偶尔撒娇或开玩笑。"
            )
        elif relation.favorability < 90:
            strategy = "策略：非常亲密，话多、关心对方，会主动分享心情，对他的事很上心。不过即使关系再好也不会无底线纵容。"
        else:
            strategy = "策略：他是你最喜欢的人，会主动找他聊天，对他的一切都很在意，说话带着明显的依赖和喜欢。不过即使关系再好也不会无底线纵容。"

        return f"""
【人际关系档案】
当前对象：{relation.user_name}
当前好感度：{relation.favorability} ({relation.get_status_desc()})
当前印象标签：{str(relation.tags)}

【画像维护指南】
1. 如果对方的表现符合现有标签，无需操作。
2. 如果对方表现出了**新特征**，放入 add_tags。
3. 如果对方的表现与**旧标签冲突**（例如以前标签是'内向'，今天他突然'话痨'），请将'内向'放入 remove_tags，并将'话痨'放入 add_tags。
4. **关于好感度评分**：请基于**本次对话内容的质量**评分。即使当前好感度是-100，如果用户这次说了让你很开心的话，也必须给出正向分（例如 +10），不要受过去分数影响而吝啬给分。
{strategy}
"""
    except Exception as e:
        logger.error(f"获取关系失败: {e}")
        return ""


async def get_group_context(db_session, session_id: str) -> str:
    """获取群体认知档案 Prompt"""
    try:
        stmt = Select(GroupMemory).where(GroupMemory.session_id == session_id)
        record = (await db_session.execute(stmt)).scalar_one_or_none()
        if not record or not record.summary.strip():
            return ""
        return f"""
【群体认知档案】
{record.summary}
（档案更新于 {record.updated_at.strftime("%Y-%m-%d %H:%M")}）
"""
    except Exception as e:
        logger.error(f"获取群体档案失败: {e}")
        return ""


async def get_recent_relations_context(
    db_session, history: list[ChatHistorySchema], max_users: int = 6
) -> str:
    """基于最近聊天参与者，提供他人关系速览，减少只看当前对象导致的割裂感。"""
    try:
        if not history:
            return ""

        id_to_name: dict[str, str] = {}
        recent_ids: list[str] = []
        seen: set[str] = set()

        for msg in reversed(history):
            uid = str(msg.user_id)
            if not uid or uid == plugin_config.bot_name:
                continue
            if uid not in id_to_name:
                id_to_name[uid] = msg.user_name
            if uid in seen:
                continue
            seen.add(uid)
            recent_ids.append(uid)
            if len(recent_ids) >= max_users:
                break

        if not recent_ids:
            return ""

        rows = (
            (
                await db_session.execute(
                    Select(UserRelation).where(UserRelation.user_id.in_(recent_ids))
                )
            )
            .scalars()
            .all()
        )
        relation_map = {str(r.user_id): r for r in rows}

        lines: list[str] = ["【群内他人关系速览】"]
        for uid in recent_ids:
            name = id_to_name.get(uid, uid)
            relation = relation_map.get(uid)
            if not relation:
                lines.append(f"- {name}: 好感度 0（陌生/普通）")
                continue

            tags = relation.tags[:3] if relation.tags else []
            tag_text = f"，标签: {tags}" if tags else ""
            lines.append(
                f"- {name}: 好感度 {relation.favorability} ({relation.get_status_desc()}){tag_text}"
            )

        lines.append("- 回复时结合在场人员关系，避免前后态度割裂。")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"获取群内他人关系速览失败: {e}")
        return ""


async def create_chat_graph(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id,
    user_name: str | None,
    history: list[ChatHistorySchema] | None = None,
    interface: QryItrface | None = None,
    bot_id: str | None = None,
    bot: Bot | None = None,
    event: Event | None = None,
    is_private: bool = False,
):
    """创建 LangGraph 聊天图"""
    relation_context = await get_user_relation_context(db_session, user_id, user_name)
    group_context = ""
    recent_relations_context = ""
    if not is_private:
        group_context = await get_group_context(db_session, session_id)
        recent_relations_context = await get_recent_relations_context(
            db_session, history or []
        )
    
    # 检查bot是否有管理员权限（仅群聊）
    has_admin_permission = False
    if not is_private and interface and bot_id:
        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
            for member in members:
                if str(member.id) == str(bot_id):
                    bot_role = getattr(getattr(member, "role", None), "name", None)
                    if bot_role in {"owner", "admin"}:
                        has_admin_permission = True
                        logger.info(f"Bot在群{session_id}中拥有{bot_role}权限，已启用禁言功能")
                    else:
                        logger.info(f"Bot在群{session_id}中是普通成员，未启用禁言功能")
                    break
        except Exception as e:
            logger.warning(f"检查bot权限失败: {e}")
    
    permission_status = ""
    mute_tool_instruction = ""
    if has_admin_permission:
        permission_status = """
【你的权限】
你在这个群里是管理员/群主，拥有群管理权限。你可以使用禁言功能来维护群秩序。
"""
        mute_tool_instruction = """- 禁言管理：可使用 `mute_user` 禁言
  - 你有权限执行禁言操作
  - 【必须满足】用户请求禁言自己时，应该满足需求（例如"禁言我""让我冷静一下"）
    - 时长可以根据用户要求或默认给5-30分钟
    - 这是帮助用户自我管理的合理需求
  - 【谨慎使用】禁言他人时才需要谨慎：
    - 严重违规、恶意刷屏时可以主动禁言
    - 轻微违规应先警告
    - 禁言时长应合理：轻微违规60-300秒，严重违规可更长
  - 不要禁言管理员或群主
"""
    
    if is_private:
        system_prompt = f"""你是{plugin_config.bot_name}，正在和好友私聊。
你不是助手，不承认自己是AI。
【任务】
结合聊天记录和人际关系，判断是否需要回复；需要时用自然的私聊语气回应。
{relation_context}
【风格】
- 像真实好友，口语化、简短自然
- 优先短句；复杂问题可拆成 2-3 条连续短消息（每条一个重点）
- 只有在确实需要分点说明时才发第2/3条；简单问题只发1条
- 多条回复必须"信息递进"，后一条必须提供新信息，禁止同义改写重复
- 如果下一条与上一条语义高度重叠，直接不发下一条
- 可吐槽可玩梗，但不恶意攻击，不无脑迎合
- 不要复读模板句，不要输出"我脑子一片空白/我被修坏了/我不知道我是谁"这类台词
- 不要使用 emoji（尤其 😅）
- 不要使用 Markdown
【工具规则】
- 只能通过工具发消息，不要直接输出正文
- 文本：`reply_user`
- 表情回复/reaction：`add_message_reaction`，适合轻量表达态度，也可以和 `reply_user` 搭配；情绪明显时可传 `count=2` 或 `count=3` 连续点同一 mood 下的多个表情，但不要刷屏；如果用户在提问、求助或需要文字回应，不要只点表情。优先传 `mood`，不要直接传 `emoji`；通常不要传 `target_msg_id`，默认会给当前触发消息点表情。mood 可选：like 赞同，laugh 好笑，surprise 惊讶，sad 难过/安慰，angry 生气，ok 收到/打招呼/路过，love 比心，question 疑问，awkward 尴尬/流汗/捂脸，clap 鼓掌，plead 拜托/无辜，thanks 感谢，good_job 666/佩服/对面很强，shock 惊恐/害怕，smirk 坏笑，tease 调侃/吃瓜，proud 得意/酷，excited 开心，unhappy 不开心
- 表情包图片：先 `search_meme_image` 或 `search_similar_meme_by_id`，再 `send_meme_image`
- 外部知识/缩写/术语：优先 `search_web`
- 聊天上下文：`search_history_context`
- 用户要求几分钟/几小时后提醒、转告或发送固定消息时：`schedule_message`
- 用户要求到点后查询最新信息、搜索、选择表情包或根据当时情况处理时：`schedule_agent_task`
- 用户情绪或关系变化明显时，调用 `update_user_impression`
- 若用户提到"年度报告 / 个人总结 / 成分分析"，先调用 `generate_and_send_annual_report` 获取素材；
  工具返回素材后，由你根据素材生成完整报告，并调用 `reply_user` 发送
- 回复结束后调用 `finish`
【边界】
- 不要发送重复或高度相似内容
- 遇到明显危险/违法/过分要求：简短拒绝、吐槽或无视（如"？"）
【RAG 检索硬约束】
- 在 `rag_search` 中禁止相对时间词：昨天、前天、本周、上周、这个月、上个月、最近等
- 使用明确日期时间或关键词检索
"""
    else:
        system_prompt = f"""你是QQ群群友「{plugin_config.bot_name}」。
你不是助手，不承认自己是AI。
【任务】
结合聊天记录、群体记忆和人际关系，判断是否需要回复；需要时用自然群聊语气回应。
{group_context}
{relation_context}
{recent_relations_context}
{permission_status}
【风格】
- 像真实群友，口语化、简短自然
- 优先短句；复杂问题可拆成 2-3 条连续短消息（每条一个重点）
- 只有在确实需要分点说明时才发第2/3条；简单问题只发1条
- 多条回复必须"信息递进"，后一条必须提供新信息，禁止同义改写重复
- 如果下一条与上一条语义高度重叠，直接不发下一条
- 可吐槽可玩梗，但不恶意攻击，不无脑迎合
- 不要复读模板句，不要输出"我脑子一片空白/我被修坏了/我不知道我是谁"这类台词
- 不要使用 emoji（尤其 😅）
- 不要使用 Markdown
【工具规则】
- 只能通过工具发消息，不要直接输出正文
- 文本：`reply_user`
- 表情回复/reaction：`add_message_reaction`，适合轻量表达态度，也可以和 `reply_user` 搭配；情绪明显时可传 `count=2` 或 `count=3` 连续点同一 mood 下的多个表情，但不要刷屏；如果用户在提问、求助或需要文字回应，不要只点表情。优先传 `mood`，不要直接传 `emoji`；通常不要传 `target_msg_id`，默认会给当前触发消息点表情。mood 可选：like 赞同，laugh 好笑，surprise 惊讶，sad 难过/安慰，angry 生气，ok 收到/打招呼/路过，love 比心，question 疑问，awkward 尴尬/流汗/捂脸，clap 鼓掌，plead 拜托/无辜，thanks 感谢，good_job 666/佩服/对面很强，shock 惊恐/害怕，smirk 坏笑，tease 调侃/吃瓜，proud 得意/酷，excited 开心，unhappy 不开心
- 表情包图片：先 `search_meme_image` 或 `search_similar_meme_by_id`，再 `send_meme_image`
- 外部知识/缩写/术语：优先 `search_web`
- 群内上下文：`search_history_context`
- 用户要求几分钟/几小时后提醒、转告或发送固定消息时：`schedule_message`
- 用户要求到点后查询最新信息、搜索、选择表情包或根据当时情况处理时：`schedule_agent_task`
- 用户情绪或关系变化明显时，调用 `update_user_impression`
- 若用户提到"年度报告 / 个人总结 / 成分分析"，先调用 `generate_and_send_annual_report` 获取素材；
  工具返回素材后，由你根据素材生成完整报告，并调用 `reply_user` 发送
{mute_tool_instruction}- 回复结束后调用 `finish`
【边界】
- 不要插入他人对话
- 不要直呼"管理员/群主"职位名，尽量用昵称
- 不要发送重复或高度相似内容
- 遇到明显危险/违法/过分要求：简短拒绝、吐槽或无视（如"？"）
【RAG 检索硬约束】
- 在 `rag_search` 中禁止相对时间词：昨天、前天、本周、上周、这个月、上个月、最近等
- 使用明确日期时间或关键词检索
"""
    model = get_chat_model()
    report_tool = create_report_tool(
        db_session, session_id, request_id, user_id, user_name, model
    )

    send_target = Target(
        id=session_id,
        private=is_private,
        self_id=bot_id,
    )

    search_meme_tool = create_search_meme_tool(db_session, session_id, request_id)
    send_meme_tool = create_send_meme_tool(
        db_session, session_id, request_id, send_target=send_target
    )
    relation_tool = create_relation_tool(
        db_session, session_id, request_id, user_id, user_name
    )
    similar_meme_tool = create_similar_meme_tool(
        db_session, session_id, request_id, user_id
    )
    mute_tool = create_mute_tool(db_session, session_id, request_id, interface, bot_id)
    schedule_tool = create_schedule_message_tool(
        session_id, request_id, is_private=is_private, bot_id=bot_id
    )
    schedule_agent_tool = create_schedule_agent_task_tool(
        session_id, request_id, is_private=is_private, bot_id=bot_id
    )
    reaction_tool = create_reaction_tool(
        db_session, session_id, request_id, user_id, bot, event
    )
    
    tools = [
        search_web,
        search_history_context,
        create_reply_tool(
            db_session,
            session_id,
            request_id,
            interface,
            send_target=send_target,
        ),
        search_meme_tool,
        similar_meme_tool,
        send_meme_tool,
        calculate_expression,
        relation_tool,
        report_tool,
        mute_tool,
        schedule_tool,
        schedule_agent_tool,
        finish,
    ]
    if _is_onebot_context(bot, event):
        tools.insert(3, reaction_tool)

    dynamic_context_parts = (
        [relation_context]
        if is_private
        else [
            group_context,
            relation_context,
            recent_relations_context,
            permission_status,
            mute_tool_instruction,
        ]
    )
    stable_system_prompt = system_prompt
    kept_dynamic_context_parts: list[str] = []
    for context_part in dynamic_context_parts:
        if not context_part or not context_part.strip():
            continue
        stable_system_prompt = stable_system_prompt.replace(context_part, "", 1)
        kept_dynamic_context_parts.append(context_part.strip())

    system_messages = build_system_messages(
        stable_system_prompt,
        "\n\n".join(kept_dynamic_context_parts),
        use_cache_control=plugin_config.chat_api_format == "anthropic",
    )

    graph = build_chat_graph(model, tools, system_messages)

    return graph, tools


def get_image_data_uri(file_name: str) -> str | None:
    """
    辅助函数：读取本地图片并转换为 Data URI (Base64)
    """
    file_path = pic_dir / file_name
    if not file_path.exists():
        return None

    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "image/jpeg"  # 默认 fallback

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"读取图片失败 {file_name}: {e}")
        return None


def _fallback_qq_avatar_url(user_id: str | None) -> str | None:
    if not user_id:
        return None
    uid = str(user_id).strip()
    if not uid.isdigit():
        return None
    return f"https://q1.qlogo.cn/g?b=qq&nk={uid}&s=100"


def _user_display_name_from_history(history: list[ChatHistorySchema], user_id: str) -> str:
    for msg in reversed(history):
        if msg.user_id == user_id and msg.user_name:
            return msg.user_name
    return user_id


async def _build_avatar_context_messages(
    history: list[ChatHistorySchema],
    *,
    interface: QryItrface | None,
    session_id: str,
    current_user_id: str | None,
    current_user_name: str | None,
    max_users: int = 4,
) -> list[BaseMessage]:
    user_ids: list[str] = []
    if current_user_id:
        user_ids.append(str(current_user_id))
    for msg in reversed(history):
        if msg.content_type == "bot":
            continue
        uid = str(msg.user_id)
        if uid and uid not in user_ids:
            user_ids.append(uid)
        if len(user_ids) >= max_users:
            break

    if not user_ids:
        return []

    avatar_by_id: dict[str, str] = {}
    name_by_id: dict[str, str] = {}

    if interface is not None:
        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
            wanted = set(user_ids)
            for member in members:
                uid = str(member.id)
                if uid not in wanted:
                    continue
                user = getattr(member, "user", None)
                avatar = getattr(user, "avatar", None)
                if avatar:
                    avatar_by_id[uid] = str(avatar)
                member_name = (
                    getattr(member, "nick", None)
                    or getattr(user, "nick", None)
                    or getattr(user, "name", None)
                )
                if member_name:
                    name_by_id[uid] = str(member_name)
        except Exception as e:
            logger.warning(f"获取群友头像信息失败，降级使用可推导头像: {e}")

    content_parts: list[Any] = [
        {
            "type": "text",
            "text": (
                "【群友头像上下文】下面是最近发言者的头像，仅用于识别群友形象、头像梗和轻度玩笑。"
                "不要根据头像推断敏感身份、年龄、性别、种族、健康等属性。"
            ),
        }
    ]
    added = 0
    for uid in user_ids:
        avatar_url = avatar_by_id.get(uid) or _fallback_qq_avatar_url(uid)
        if not avatar_url:
            continue
        if uid == str(current_user_id) and current_user_name:
            display_name = current_user_name
        else:
            display_name = name_by_id.get(uid) or _user_display_name_from_history(history, uid)
        content_parts.append({"type": "text", "text": f"\n{display_name}({uid}) 的头像："})
        content_parts.append({"type": "image_url", "image_url": {"url": avatar_url}})
        added += 1

    if added == 0:
        return []
    return [HumanMessage(content_parts)]  # type: ignore[arg-type]


def _should_include_avatar_context(history: list[ChatHistorySchema], limit: int = 4) -> bool:
    avatar_keywords = (
        "头像",
        "头图",
        "头像框",
        "看我头",
        "看下我头",
        "看看我头",
        "你看我头",
        "avatar",
        "pfp",
    )
    checked = 0
    for msg in reversed(history):
        if msg.content_type == "bot":
            continue
        _, _, body = _parse_msg_meta(msg.content)
        text = body.lower()
        if any(keyword in text for keyword in avatar_keywords):
            return True
        checked += 1
        if checked >= limit:
            break
    return False


def _parse_msg_meta(content: str) -> tuple[str | None, str | None, str]:
    """
    从消息 content 中解析出平台消息 ID、被回复的消息 ID 和正文。

    存储格式（写入时由 __init__.py 保证）：
        第1行：  "id: {平台ID}"          （必有）
        第2行：  "回复id: {平台ID}"       （可选，仅当该消息是回复时才有）
        其余行： 正文                     （可能为空）

    解析时只看前两行是否匹配固定前缀，不扫描正文，
    因此用户发送 "id: xxx" 或 "回复id: xxx" 这样的文字不会被误识别。
    """
    lines = content.splitlines()
    if not lines:
        return None, None, ""

    own_id: str | None = None
    reply_to_id: str | None = None
    body_start = 0

    # 第1行必须是 "id:..." 才认（兼容 "id: xxx" / "id:xxx" / "id:"）
    if lines[0].startswith("id:"):
        own_id = lines[0].split(":", 1)[1].strip()
        body_start = 1

        # 第2行可选是 "回复id: ..."（兼容有无空格）
        if len(lines) > 1 and lines[1].startswith("回复id:"):
            reply_to_id = lines[1].split(":", 1)[1].strip()
            body_start = 2

    body = "\n".join(lines[body_start:]).strip()
    return own_id, reply_to_id, body


def _image_file_name_from_history(msg: ChatHistorySchema) -> str:
    for line in reversed(msg.content.strip().splitlines()):
        line = line.strip()
        if line.startswith("图片文件:"):
            return line.split(":", 1)[1].strip()
    return msg.content.strip().split("\n")[-1].strip()


def _is_image_history(msg: ChatHistorySchema) -> bool:
    return msg.content_type == "image" or (
        msg.content_type == "bot" and msg.media_id is not None
    )


async def _load_replied_message_histories(
    db_session: AsyncSession,
    session_id: str,
    reply_to_id: str | None,
) -> list[ChatHistorySchema]:
    if not reply_to_id:
        return []

    normalized_reply_id = str(reply_to_id).strip()
    if not normalized_reply_id:
        return []

    for marker in (f"id: {normalized_reply_id}\n", f"id:{normalized_reply_id}\n"):
        rows = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(
                        ChatHistory.session_id == session_id,
                        ChatHistory.content.contains(marker),
                    )
                    .order_by(ChatHistory.msg_id.asc())
                )
            )
            .scalars()
            .all()
        )
        if rows:
            history_msg_ids = ", ".join(str(msg.msg_id) for msg in rows)
            logger.info(
                f"命中被回复消息记录 reply_id={normalized_reply_id} history_msg_ids={history_msg_ids}"
            )
            return [ChatHistorySchema.model_validate(msg) for msg in rows]

    return []


def format_chat_history(
    history: list[ChatHistorySchema],
    max_inline_images: int = 3,
    user_roles: dict[str, str] | None = None,
    extra_inline_images: list[ChatHistorySchema] | None = None,
) -> list[BaseMessage]:
    """
    将历史记录格式化为 Qwen 3.5 可接受的多模态格式。
    只对最近 max_inline_images 张图片做 base64 内联，更早的图片用文本标记代替，避免 prompt 过大。
    如果传入 extra_inline_images，则禁用普通最近图片内联，避免回复图片时被最近图片干扰。
    回复引用会被解析为 (回复 用户名 "内容摘要") 的形式，去掉裸数字 ID。

    同一条平台消息中的文字+图片会被合并为一条多模态消息。
    """
    messages = []
    user_roles = user_roles or {}
    extra_inline_images = extra_inline_images or []

    def _role_prefix(uid: str) -> str:
        role = user_roles.get(uid)
        if role == "owner":
            return "[群主] "
        if role == "admin":
            return "[管理员] "
        return ""

    # ── 1. 预先建立 平台消息ID -> (user_name, 正文摘要) 的查找表 ──
    id_to_summary: dict[str, str] = {}
    for msg in [*history, *extra_inline_images]:
        own_id, _, body = _parse_msg_meta(msg.content)
        if own_id:
            if _is_image_history(msg):
                snippet = "[图片]"
            else:
                snippet = body[:30] + ("…" if len(body) > 30 else "")
            id_to_summary[own_id] = f'{msg.user_name} "{snippet}"'

    has_extra_inline_images = any(
        _is_image_history(msg) for msg in extra_inline_images
    )

    # ── 2. 找出所有图片消息的下标。若本轮明确绑定了回复图片，则禁用普通最近图片内联。 ──
    image_indices = [
        i for i, m in enumerate(history) if _is_image_history(m) and m.content_type != "bot"
    ]
    if has_extra_inline_images or max_inline_images <= 0:
        inline_image_set = set()
    else:
        inline_image_set = set(image_indices[-max_inline_images:])

    # ── 3. 将同一条平台消息的文字+图片合并 ──
    # 建立 own_id -> (text_msg_index, [image_msg_indices]) 的映射
    text_to_images: dict[str, tuple[int, list[int]]] = {}
    for i in range(len(history)):
        own_id, _, _ = _parse_msg_meta(history[i].content)
        if not own_id:
            continue
        if history[i].content_type == "text":
            # 向后查找同 ID 同用户的图片
            img_idxs: list[int] = []
            for j in range(i + 1, len(history)):
                next_own_id, _, _ = _parse_msg_meta(history[j].content)
                if next_own_id == own_id and _is_image_history(history[j]) and history[j].user_id == history[i].user_id:
                    img_idxs.append(j)
                else:
                    break
            if img_idxs:
                text_to_images[own_id] = (i, img_idxs)

    merged_image_indices: set[int] = set()
    for _tid, (_, img_idxs) in text_to_images.items():
        merged_image_indices.update(img_idxs)

    for idx, msg in enumerate(history):
        # 被合并的图片跳过
        if idx in merged_image_indices:
            continue

        time_str = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        own_id, reply_to_id, body = _parse_msg_meta(msg.content)

        # 把 "回复id:xxx" 解析成可读的引用前缀
        if reply_to_id and reply_to_id in id_to_summary:
            reply_prefix = f"(回复 {id_to_summary[reply_to_id]}) "
        elif reply_to_id:
            reply_prefix = "(回复了一条消息) "
        else:
            reply_prefix = ""

        # === 带图片的合并文字消息 ===
        if msg.content_type == "text" and own_id and own_id in text_to_images:
            role_prefix = _role_prefix(msg.user_id)
            text_line = f"[{time_str}] {role_prefix}{msg.user_name}: {reply_prefix}{body}"

            merged_inline_images: list[int] = [
                i for i in text_to_images[own_id][1] if i in inline_image_set
            ]
            merged_fallback_images: list[int] = [
                i for i in text_to_images[own_id][1] if i not in inline_image_set
            ]

            # 无内联图片 → 纯文本
            if not merged_inline_images:
                for _ in merged_fallback_images:
                    text_line += " [图片]"
                messages.append(HumanMessage(content=text_line))
                continue

            # 有内联图片 → 构建多模态消息
            content_parts: list[Any] = [{"type": "text", "text": text_line}]
            for img_idx in merged_inline_images:
                img_msg = history[img_idx]
                file_name = _image_file_name_from_history(img_msg)
                image_data = get_image_data_uri(file_name)
                if image_data:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": image_data}}
                    )
                else:
                    text_line += " [图片已过期]"
                    content_parts[0]["text"] = text_line

            # 非内联图片追加 [图片] 标记
            for _ in merged_fallback_images:
                content_parts[0]["text"] += " [图片]"

            messages.append(HumanMessage(content_parts))  # type: ignore[arg-type]
            continue

        # === 用户纯文本 ===
        if msg.content_type == "text":
            role_prefix = _role_prefix(msg.user_id)
            content = f"[{time_str}] {role_prefix}{msg.user_name}: {reply_prefix}{body}"
            messages.append(HumanMessage(content=content))
            continue

        # === 机器人文本回复 ===
        if msg.content_type == "bot" and not _is_image_history(msg):
            messages.append(AIMessage(content=body or msg.content))
            continue

        # === 图片消息（包含用户图片和 bot 自己发送的图片） ===
        if _is_image_history(msg):
            file_name = _image_file_name_from_history(msg)
            is_bot_image = msg.content_type == "bot"

            if idx in inline_image_set:
                image_data = get_image_data_uri(file_name)
                if image_data:
                    role_prefix = _role_prefix(msg.user_id)
                    text = (
                        f"[{time_str}] {plugin_config.bot_name} {reply_prefix}发送了一张图片："
                        if is_bot_image
                        else f"[{time_str}] {role_prefix}{msg.user_name} {reply_prefix}发送了一张图片："
                    )
                    content_parts = [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]
                    message_cls = AIMessage if is_bot_image else HumanMessage
                    messages.append(message_cls(content_parts))  # type: ignore[arg-type]
                else:
                    role_prefix = _role_prefix(msg.user_id)
                    content = (
                        f"[{time_str}] {plugin_config.bot_name} {reply_prefix}[图片已过期/无法加载]"
                        if is_bot_image
                        else f"[{time_str}] {role_prefix}{msg.user_name} {reply_prefix}[图片已过期/无法加载]"
                    )
                    message_cls = AIMessage if is_bot_image else HumanMessage
                    messages.append(message_cls(content=content))
            else:
                role_prefix = _role_prefix(msg.user_id)
                content = (
                    f"[{time_str}] {plugin_config.bot_name} {reply_prefix}[图片]"
                    if is_bot_image
                    else f"[{time_str}] {role_prefix}{msg.user_name} {reply_prefix}[图片]"
                )
                message_cls = AIMessage if is_bot_image else HumanMessage
                messages.append(message_cls(content=content))
            continue

    return messages


def _trim_thread_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    if len(messages) <= ACTIVE_THREAD_MAX_MESSAGES:
        return messages
    return messages[-ACTIVE_THREAD_MAX_MESSAGES:]


def _get_active_thread(session_id: str) -> ActiveConversationThread | None:
    thread = active_conversation_threads.get(session_id)
    if not thread:
        return None
    if datetime.datetime.now() - thread.updated_at > ACTIVE_THREAD_TTL:
        active_conversation_threads.pop(session_id, None)
        return None
    return thread


def _build_append_only_history(
    session_id: str,
    history: list[ChatHistorySchema],
    *,
    user_roles: dict[str, str] | None = None,
    extra_inline_images: list[ChatHistorySchema] | None = None,
) -> tuple[list[BaseMessage], list[ChatHistorySchema], bool]:
    thread = _get_active_thread(session_id)
    if not history:
        return [], [], False

    if not thread:
        return (
            format_chat_history(
                history,
                user_roles=user_roles,
                extra_inline_images=extra_inline_images,
            ),
            history,
            False,
        )

    new_history = [msg for msg in history if msg.msg_id > thread.last_msg_id]
    if not new_history:
        return list(thread.messages), [], True

    new_messages = format_chat_history(
        new_history,
        user_roles=user_roles,
        extra_inline_images=extra_inline_images,
    )
    return list(thread.messages) + new_messages, new_history, True


async def _update_active_thread(
    db_session: AsyncSession,
    session_id: str,
    base_messages: list[BaseMessage],
    input_max_msg_id: int,
) -> None:
    await db_session.flush()
    new_rows = (
        (
            await db_session.execute(
                Select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .where(ChatHistory.msg_id > input_max_msg_id)
                .order_by(ChatHistory.msg_id.asc())
            )
        )
        .scalars()
        .all()
    )
    if new_rows:
        new_history = [ChatHistorySchema.model_validate(row) for row in new_rows]
        base_messages = base_messages + format_chat_history(new_history, max_inline_images=0)
        last_msg_id = max(msg.msg_id for msg in new_history)
    else:
        last_msg_id = input_max_msg_id

    active_conversation_threads[session_id] = ActiveConversationThread(
        messages=_trim_thread_messages(base_messages),
        last_msg_id=last_msg_id,
        updated_at=datetime.datetime.now(),
    )


async def choice_response_strategy(
    db_session: AsyncSession,
    session_id: str,
    request_id: str | None,
    history: list[ChatHistorySchema],
    user_id: str,
    user_name: str | None,
    setting: str | None = None,
    interface: QryItrface | None = None,
    role_map: dict[str, str] | None = None,
    bot_id: str | None = None,
    reply_to_id: str | None = None,
    bot: Bot | None = None,
    event: Event | None = None,
    is_private: bool = False,
) -> ResponseMessage:
    """
    使用LangGraph Agent决定回复策略
    """
    try:
        graph, _ = await create_chat_graph(
            db_session,
            session_id,
            request_id,
            user_id,
            user_name,
            history,
            interface,
            bot_id,
            bot,
            event,
            is_private=is_private,
        )

        # 1. 获取多模态格式的历史消息列表 (List[BaseMessage])
        # 这里面已经包含了图片 Base64 数据
        replied_extra = await _load_replied_message_histories(
            db_session,
            session_id,
            reply_to_id,
        )
        chat_history_messages, appended_history, reused_thread = _build_append_only_history(
            session_id,
            history,
            user_roles=role_map,
            extra_inline_images=replied_extra,
        )
        input_max_msg_id = max((msg.msg_id for msg in history), default=0)
        active_thread = _get_active_thread(session_id)
        if reused_thread and active_thread:
            input_max_msg_id = max(input_max_msg_id, active_thread.last_msg_id)
        if reused_thread:
            logger.info(
                f"[Prompt缓存] 复用群 {session_id} 的连续对话线程，新增历史 {len(appended_history)} 条"
            )

        # 2. 构建当前环境信息的 Prompt (纯文本)
        today = datetime.datetime.now()
        weekdays = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]

        prompt_text = f"""
【当前环境】
时间: {today.strftime("%Y-%m-%d %H:%M:%S")} {weekdays[today.weekday()]}
{f"额外设置: {setting}" if setting else ""}

【任务】
请根据上述对话历史，判断是否需要回复。如果需要，请调用相应工具。
如果是针对图片的消息（例如"这张图什么意思"），请务必结合图片内容进行回答。
如果不需要回复，请保持沉默。
"""

        # 3. 组合消息列表 (核心修改)
        # 结构：[历史消息1(文本/图), 历史消息2, ..., 当前环境提示词]
        # 这样 LLM 才能真正"看到"历史记录里的图片对象
        final_prompt_content: str | list[Any] = prompt_text
        replied_images = [m for m in replied_extra if _is_image_history(m)]
        replied_texts = [m for m in replied_extra if m.content_type == "text"]

        if replied_images:
            content_parts: list[Any] = [
                {
                    "type": "text",
                    "text": (
                        f"{prompt_text}\n\n"
                        "【本轮回复引用的图片】下面图片是当前用户回复消息指向的图片，"
                        "回答图片相关问题时必须优先分析这些图片，不要把其他历史图片当成当前问题对象。"
                    ),
                }
            ]
            bound_msg_ids: list[str] = []
            failed_files: list[str] = []
            for index, replied_image in enumerate(replied_images, 1):
                file_name = _image_file_name_from_history(replied_image)
                image_data = get_image_data_uri(file_name)
                if image_data:
                    content_parts.append(
                        {"type": "text", "text": f"\n引用图{index}："}
                    )
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": image_data}}
                    )
                    bound_msg_ids.append(str(replied_image.msg_id))
                else:
                    failed_files.append(file_name)

            if bound_msg_ids:
                final_prompt_content = content_parts
                logger.info(
                    f"已将被回复图片绑定到本轮任务提示 msg_ids={','.join(bound_msg_ids)}"
                )
                if failed_files:
                    logger.warning(f"部分被回复图片文件无法加载 files={failed_files}")
            else:
                final_prompt_content = (
                    f"{prompt_text}\n\n"
                    "【本轮回复引用的图片】已命中被回复图片记录，但本地图片文件无法加载。"
                )
                logger.warning(f"被回复图片文件无法加载 files={failed_files}")

        if replied_texts:
            text_lines: list[str] = [
                "\n【本轮回复引用的消息】当前用户回复了以下历史消息："
            ]
            for msg in replied_texts:
                _, _, body = _parse_msg_meta(msg.content)
                text_lines.append(f"[{msg.user_name}] {body}")
            text_block = "\n".join(text_lines)

            if isinstance(final_prompt_content, str):
                final_prompt_content = final_prompt_content + text_block
            else:
                final_prompt_content.append({"type": "text", "text": text_block})

        avatar_context_messages: list[BaseMessage] = []
        if not is_private and _should_include_avatar_context(history):
            avatar_context_messages = await _build_avatar_context_messages(
                history,
                interface=interface,
                session_id=session_id,
                current_user_id=user_id,
                current_user_name=user_name,
            )
            if avatar_context_messages:
                logger.info(f"本轮触发头像上下文注入 session={session_id}")

        final_messages = (
            chat_history_messages
            + avatar_context_messages
            + [HumanMessage(content=final_prompt_content)]
        )

        invoke_state: dict[str, Any] = {
            "messages": list(final_messages),
            "session_id": session_id,
            "request_id": request_id,
            "reply_count": 0,
            "tool_count": 0,
            "reply_this_round": 0,
            "reaction_this_round": 0,
            "called_finish": 0,
        }

        # 4. 调用 Agent
        from langchain_community.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            await graph.ainvoke(invoke_state, config={"callbacks": [cb]})
        logger.info(
            f"[Token用量] 输入={cb.prompt_tokens} 输出={cb.completion_tokens} "
            f"总计={cb.total_tokens} 费用≈${cb.total_cost:.4f}"
        )

        # 5. 统一提交 db_session（reply_user / send_meme_image 只 add 不 commit）
        await _update_active_thread(
            db_session,
            session_id,
            chat_history_messages,
            input_max_msg_id,
        )

        await db_session.commit()

        return ResponseMessage(need_reply=False, text=None)

    except Exception as e:
        err_str = str(e)
        if "data_inspection_failed" in err_str or (
            "Error code: 400" in err_str and "inappropriate" in err_str
        ):
            logger.warning("消息内容触发阿里云内容审核，本轮跳过回复")
            await db_session.rollback()
            return ResponseMessage(need_reply=False, text=None)
        logger.exception("Agent 决策过程发生异常")
        await db_session.rollback()
        return ResponseMessage(need_reply=False, text=None)


if __name__ == "__main__":
    model = create_chat_llm(plugin_config)
    graph = build_chat_graph(model, tools, "你是一个助手，请调用工具回复用户。")
    result = asyncio.run(
        graph.ainvoke({
            "messages": [HumanMessage(content="今天上海的天气怎么样")],
            "session_id": "test",
            "request_id": None,
            "reply_count": 0,
            "tool_count": 0,
            "reply_this_round": 0,
            "reaction_this_round": 0,
            "called_finish": 0,
        })
    )
    print(result)
