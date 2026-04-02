import re
import json
import base64
import random
import asyncio
import datetime
import mimetypes
import traceback
import collections
from typing import Any, cast
from pathlib import Path
from dataclasses import dataclass

import jieba
from nonebot import require, get_plugin_config
from pydantic import Field, BaseModel, SecretStr, field_validator
from simpleeval import simple_eval
from sqlalchemy import Select, desc, func, extract
from nonebot.log import logger
from langchain.tools import ToolRuntime, tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from nonebot_plugin_orm import get_session
from langchain_core.prompts import ChatPromptTemplate
from nonebot_plugin_alconna import UniMessage
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy

from ..model import ChatHistory, MediaStorage, UserRelation, ChatHistorySchema, GroupMemory
from ..config import Config
from ..memory import DB

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

if plugin_config.tavily_api_key:
    tavily_search = TavilySearch(max_results=3, tavily_api_key=plugin_config.tavily_api_key)
else:
    tavily_search = None


@dataclass
class Context:
    session_id: str


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


flash_model = ChatOpenAI(
    model="qwen-flash",
    api_key=SecretStr(plugin_config.qwen_token),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,  # 设为0，由它做决策需要绝对理性，不需要发散
    max_completion_tokens=10,  # 我们只需要它回答 YES 或 NO，限制输出长度省钱
)


async def check_if_should_reply(
        history_summary: str,
        current_msg: str,
        bot_name: str
) -> bool:
    """
    使用 qwen-flash 快速判断是否需要回复
    """
    system_prompt = f"""
你是一个群聊消息过滤器。你的任务是判断群内的最新消息是否需要机器人 "{bot_name}" 进行回复。

判断规则：
1. 如果用户明显在向 "{bot_name}" 提问、求助或打招呼，返回 YES。
2. 如果用户在讨论 "{bot_name}" 相关的话题且期待回应，返回 YES。
3. 如果是群友之间的闲聊、无关的刷屏、或者语意不通的消息，返回 NO。
4. 如果你不确定，返回 NO。

请仅输出 "YES" 或 "NO"，不要输出任何其他内容。
"""

    # 组合 Prompt
    # 只需要最近的一两条消息即可，不需要长篇大论的历史
    input_text = f"【最近上下文】\n{history_summary}\n\n【最新消息】\n{current_msg}\n\n请判断是否回复(YES/NO):"

    try:
        # 调用 Flash 模型
        resp = await flash_model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_text)
        ])
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
async def search_web(query: str) -> str:
    """
    用于搜索最新的实时信息。当你需要最新的事实信息、天气或新闻时使用。
    输入：需要搜索的内容。
    """
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
    try:
        logger.info(f"大模型执行{runtime.context.session_id} RAG 搜索\n{query}")
        similar_msgs = await DB.search_chat(query, runtime.context.session_id)
        return similar_msgs if similar_msgs else "未找到相关历史记录"
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return "历史搜索失败"


def create_report_tool(db_session, session_id: str, user_id: str, user_name: str | None, llm_client: ChatOpenAI):
    """
    创建年度报告工具（限制在当前群聊 session_id 范围内）
    """

    @tool("generate_and_send_annual_report")
    async def generate_and_send_annual_report() -> str:
        """
        生成并发送当前群聊的年度报告。
        包含：个人在本群的统计、性格分析、全群排行榜以及Bot的好感度回顾。
        """
        try:
            logger.info(f"开始生成用户 {user_name} 在群 {session_id} 的年度报告...")
            now = datetime.datetime.now()
            current_year = now.year

            stmt = Select(ChatHistory).where(
                ChatHistory.user_id == user_id,
                ChatHistory.session_id == session_id,
                extract("year", ChatHistory.created_at) == current_year
            )
            all_msgs = (await db_session.execute(stmt)).scalars().all()

            if not all_msgs:
                await UniMessage.text("你今年在这个群好像没怎么说话，生成不了报告哦...").send()
                return "用户本群无数据。"

            # 统计与采样
            text_msgs = [m.content for m in all_msgs if m.content_type == "text" and m.content]
            total_count = len(all_msgs)

            # 采样 30 条让 LLM 分析 (只分析在这个群说的话)
            samples = random.sample(text_msgs, min(len(text_msgs), 30)) if text_msgs else []
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
                stmt = Select(ChatHistory.user_id, func.count(ChatHistory.msg_id).label("c")) \
                    .where(
                    extract("year", ChatHistory.created_at) == current_year,
                    ChatHistory.session_id == session_id
                )

                if content_type:
                    stmt = stmt.where(ChatHistory.content_type == content_type)
                if hour_limit:
                    stmt = stmt.where(extract("hour", ChatHistory.created_at) < hour_limit)

                # 核心修改：只 group_by user_id
                stmt = stmt.group_by(ChatHistory.user_id).order_by(desc("c")).limit(3)

                # 获取结果，此时是 List[(user_id, count)]
                rows = (await db_session.execute(stmt)).all()

                if not rows:
                    return "虚位以待"

                rank_items = []
                for uid, count in rows:
                    # 查询该用户最近的一条消息记录，取当时的名字
                    name_stmt = Select(ChatHistory.user_name).where(
                        ChatHistory.user_id == uid
                    ).order_by(desc(ChatHistory.created_at)).limit(1)

                    latest_name = (await db_session.execute(name_stmt)).scalar()

                    # 兜底：如果查不到名字（极少情况），用 ID 代替
                    display_name = latest_name if latest_name else f"用户{uid}"
                    rank_items.append(f"{display_name}({count})")
                return ", ".join(rank_items)

            rank_talk = await get_rank_str()
            rank_img = await get_rank_str(content_type="image")
            rank_night = await get_rank_str(hour_limit=5)

            # 只分析本群的文本
            stmt_text = Select(ChatHistory.content).where(
                ChatHistory.session_id == session_id,
                extract("year", ChatHistory.created_at) == current_year,
                ChatHistory.content_type == "text",
                ChatHistory.user_id == user_id,
            ).order_by(desc(ChatHistory.created_at))

            rows = (await db_session.execute(stmt_text)).all()
            sample_text = "\n".join([r[0] for r in rows if r[0]])

            clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", sample_text)
            words = jieba.lcut(clean_text)
            filtered = [w for w in words if len(w) > 1 and w not in stop_words]
            hot_words_str = "、".join([x[0] for x in collections.Counter(filtered).most_common(8)])

            relation_stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
            relation = (await db_session.execute(relation_stmt)).scalar_one_or_none()

            favorability = 0
            impression_tags = []
            if relation:
                favorability = relation.favorability
                impression_tags = relation.tags if relation.tags else []

            # 格式化关系描述，喂给 LLM
            relation_desc = f"好感度: {favorability} (满分100), 印象标签: {', '.join(impression_tags)}"


            report_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的年度报告撰写助手。
你的任务是阅读用户的聊天统计数据和发言样本，分析其性格，然后生成一份格式整洁、风格幽默的年度报告。

【语气控制指南 (非常重要)】
根据用户的"好感度"调整你的语气：
- 好感度 > 60：语气要亲密、宠溺，像对待最好的朋友或恋人。（例如："宝，今年你也一直陪着我呢"）
- 好感度 < 0：语气要傲娇、嫌弃、毒舌。（例如："你这家伙今年没少气我，明年注意点！"）
- 好感度 0-60：语气正常、友善、带点调侃。

【排版要求】
1. **绝对禁止使用 Markdown**（不要用 #, **, ##, - 等符号列表）。
2. 使用 Emoji 和 纯文本分隔符（如 ━━━━━━━━）来排版。
3. 语气要像老朋友一样，可以根据数据进行调侃或夸奖。

【必须包含的板块】
1. 📊 标题行 ({year}年度报告 | 用户名)
2. 📈 基础数据 (发言数、活跃时间、最长发言摘要)
3. 💌 我们的羁绊 (根据好感度和标签，写一段话回顾你们的关系。如果是正向关系就煽情一点，负向关系就吐槽。)
4. 🔥 年度热词 (列出数据中提供的热词)
5. 🏆 群内风云榜 (必须包含以下三个榜单)
   - 🗣️ 龙王榜 (发言最多)
   - 🎭 斗图榜 (发图最多)
   - 🦉 修仙榜 (熬夜最多)
6. 🧠 成分分析 (这是**重点**：请阅读提供的 `samples` 聊天记录，分析这个人的说话风格、性格、是不是复读机、是不是爱发疯。写一段100字左右的犀利点评)
7. 💡 {bot_name}寄语 (一句简短的祝福)
"""),
                ("user", """
【用户数据】
用户名: {user_name}
年份: {year}
累计发言: {count}
活跃时间: {active_hour}
最长发言片段: {longest_msg}
年度热词: {hot_words}

【{bot_name}与用户的关系】
{relation_desc}

【全群排行参考】
龙王榜: {rank_talk}
斗图榜: {rank_img}
熬夜榜: {rank_night}

【用户发言样本 (用于性格分析)】
{samples}

请生成报告：""")
            ])

            # 组装数据
            prompt_input = {
                "user_name": user_name,
                "bot_name": plugin_config.bot_name,
                "year": current_year,
                "count": total_count,
                "active_hour": active_hour_desc,
                "longest_msg": longest_msg,
                "hot_words": hot_words_str,
                "relation_desc": relation_desc,
                "rank_talk": rank_talk,
                "rank_img": rank_img,
                "rank_night": rank_night,
                "samples": "\n".join(samples)  # 把样本拼接成字符串喂给 LLM
            }

            logger.info(f"内部 LLM 生成报告中，好感度: {favorability}")
            chain = report_prompt | llm_client
            response_msg = await chain.ainvoke(prompt_input)
            final_report_text = response_msg.content
            if not isinstance(final_report_text, str):
                return "输出结果失败"
            await UniMessage.text(final_report_text).send()

            return "报告已生成并发送。"

        except Exception as e:
            logger.error(f"内部 LLM 生成报告失败: {e}")
            import traceback
            traceback.print_exc()
            return f"生成过程出错: {e}"

    return generate_and_send_annual_report


def create_similar_meme_tool(db_session, session_id: str):
    """
    创建基于消息ID搜索相似表情包的工具
    """

    @tool("search_similar_meme_by_id")
    async def search_similar_meme_by_pic(target_msg_id: str | None = None) -> str:
        """
        根据指定的历史图片，搜索与之相似的表情包。
        当用户说"找一张跟这张差不多的"或引用某张图片求相似图时使用。
        参数：
        - target_msg_id: 聊天记录中图片消息的 id（从聊天记录的 "id:xxxxx" 中获取）。
          如果不传，则默认使用本群最近一条图片。
        """
        logger.info("正在搜索相似图片...")

        try:
            # 1. 从 ChatHistory 查找指定或最近的图片消息
            base_stmt = (
                Select(ChatHistory)
                .where(
                    ChatHistory.session_id == session_id,
                    ChatHistory.content_type == "image"
                )
                .order_by(desc(ChatHistory.created_at))
            )
            if target_msg_id:
                stmt = base_stmt.where(ChatHistory.content.contains(f"id: {target_msg_id}\n")).limit(1)
            else:
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
                    images_info.append({
                        "pic_id": str(pid),  # 转字符串方便模型理解
                        # 注意：如果是新图片，description 可能是 "[图片]" 占位符
                        # 如果是迁移过来的旧图片，则是真实的描述
                        "description": media_obj.description or "未知描述",
                    })

            return json.dumps({
                "success": True,
                "source_media_id": msg.media_id,
                "images": images_info,
                "count": len(images_info),
                "note": "请根据 pic_id 调用 send_meme_image 发送"
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"相似图片搜索失败: {e}")
            return f"搜索出错: {e}"

    return search_similar_meme_by_pic


def create_reply_tool(db_session, session_id: str):
    """
    核心工具：用于发送消息。
    """
    @tool("reply_user")
    async def reply_user(content: str) -> str:
        """
        向当前群聊发送文本回复。
        注意：如果你想对用户说话，必须调用这个工具。不要直接返回文本。
        Args:
            content: 你想发送的内容。
        """
        if not content or not content.strip():
            return "内容为空，未发送。"

        try:
            # 1. 实际发送消息 (Side Effect)
            res = await UniMessage.text(content).send()
            msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content= f"id:{msg_id}\n" + content,
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"Bot已回复: {content}")
            return "回复已成功发送。"
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            await db_session.rollback()
            return f"发送失败: {e}"

    return reply_user

def create_search_meme_tool(db_session):
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
        try:
            pic_ids = await DB.search_meme(description)

            if not pic_ids:
                logger.info(f"未找到匹配的表情包: {description}")
                return json.dumps({
                    "success": False,
                    "images": []
                }, ensure_ascii=False)

            # 从数据库获取每张图片的详细信息
            images_info = []
            for pic_id in pic_ids[:5]:  # 只返回前5张，避免信息过多
                pic = (
                    await db_session.execute(
                        Select(MediaStorage).where(MediaStorage.media_id == int(pic_id))
                    )
                ).scalar()

                if pic:
                    images_info.append({
                        "pic_id": pic_id,
                        "description": pic.description,
                    })

            if not images_info:
                return json.dumps({
                    "success": False,
                    "images": [],
                }, ensure_ascii=False)

            logger.info(f"找到 {len(images_info)} 张匹配的表情包: {description}")
            return json.dumps({
                "success": True,
                "images": images_info,
                "count": len(images_info),
            }, ensure_ascii=False, indent=2)


        except Exception as e:
            logger.error(f"表情包搜索失败: {repr(e)}")  # 使用 repr() 可以看到异常类型，比 str() 更详细
            logger.error(traceback.format_exc())  # 打印完整报错路径
            return json.dumps({
                "success": False,
                "images": [],
                "error": str(e) or "未知错误"
            }, ensure_ascii=False)

    return search_meme_image


def create_send_meme_tool(db_session, session_id: str):
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
        try:
            selected_pic_id = int(pic_id)
            logger.info(f"使用指定的图片ID: {pic_id}")

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
            # 发送图片
            res = await UniMessage.image(raw=pic_data).send()
            # 记录发送历史（不在工具内提交，由外层 session 统一管理）
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id:{res.msg_ids[-1]['message_id']}\n发送了图片，图片描述是: {description}",
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"id:{res.msg_ids}\n" + f"发送表情包: {description}")
            return f"已成功发送表情包: {description}"

        except Exception as e:
            logger.error(f"发送表情包失败: {e}")
            return f"发送表情包失败: {str(e)}"

    return send_meme_image


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
        return f"计算结果是：{result:.10f}" if isinstance(result, float) else str(result)

    except Exception as e:
        return f"计算失败。请检查表达式是否正确，错误信息: {e}"


def create_relation_tool(db_session, user_id: str, user_name: str | None):
    """
    创建绑定了特定用户的关系管理工具 (支持增删 Tag)
    """

    @tool("update_user_impression")
    async def update_user_impression(
            score_change: int,
            reason: str,
            add_tags: list[str],
            remove_tags: list[str]
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
        async with get_session() as session:
            try:
                # 1. 查询或初始化记录
                stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
                result = await session.execute(stmt)
                relation = result.scalar_one_or_none()

                if not relation:
                    relation = UserRelation(user_id=user_id, user_name=user_name or "", favorability=0, tags=[])
                    session.add(relation)
                else:
                    await session.refresh(relation, attribute_names=["tags"])

                # 2. 处理好感度
                old_score = relation.favorability

                final_change = score_change

                # 【救赎机制】：当好感度低于 -60 且 试图加分时，效果翻倍并额外奖励
                if old_score < -60 and score_change > 0:
                    final_change = int(score_change * 1.5) + 5
                    logger.info(f"触发救赎机制：原始分 {score_change} -> 修正分 {final_change}")

                # 【破防机制】：当好感度高于 80 且 试图扣分时，伤害加深 (可选)
                elif old_score > 80 and score_change < 0:
                    final_change = int(score_change * 1.2) - 2
                    logger.info(f"触发破防机制：原始分 {score_change} -> 修正分 {final_change}")

                relation.favorability += final_change
                relation.favorability = max(-100, min(100, relation.favorability))
                if score_change > 0:
                    await UniMessage.image(raw=up_pic).send()
                elif score_change < 0:
                    await UniMessage.image(raw=down_pic).send()
                # 3. 处理标签 (核心修改)
                # 获取现有标签的副本
                current_tags = list(relation.tags) if relation.tags else []

                # 执行移除操作 (处理 modify 的前半部分)
                if remove_tags:
                    current_tags = [tag for tag in current_tags if tag not in remove_tags]

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

                await session.commit()

                # 构建反馈信息
                tag_msg = ""
                if add_tags or remove_tags:
                    tag_msg = f"，标签变更(新增:{add_tags}, 移除:{remove_tags})"

                log_msg = f"好感度 {old_score}->{favorability}{tag_msg} (原因: {reason})"
                logger.info(f"用户[{user_name}]画像更新: {log_msg}")

                return f"画像已更新。当前好感度: {favorability}，当前标签: {current_tags}"

            except Exception as e:
                logger.error(f"关系更新失败: {e}")
                print(traceback.format_exc())
                return f"数据库错误: {str(e)}"
        return None

    return update_user_impression


tools = [search_web, search_history_context, calculate_expression]
model = ChatOpenAI(
    model=plugin_config.base_model,
    api_key=SecretStr(plugin_config.qwen_token),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=1,
)


async def get_user_relation_context(db_session, user_id: str, user_name: str | None) -> str:
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
        if relation.favorability <= -80:
            strategy = "策略：你非常讨厌他，但如果他**诚恳道歉**或**做出实质性补偿**，请给他一个改过自新的机会（给予大幅好感度加分），不要一直死咬着不放。"
        elif relation.favorability >= 90:
            strategy = "策略：你们关系极好，但不要对他无底线纵容。"
        else:
            strategy = ""

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
（档案更新于 {record.updated_at.strftime('%Y-%m-%d %H:%M')}）
"""
    except Exception as e:
        logger.error(f"获取群体档案失败: {e}")
        return ""


async def create_chat_agent(db_session, session_id: str, user_id, user_name: str | None):
    """创建聊天Agent"""
    relation_context = await get_user_relation_context(db_session, user_id, user_name)
    group_context = await get_group_context(db_session, session_id)
    system_prompt = f"""你现在是QQ群里的一位普通群友，名叫"{plugin_config.bot_name}"。

【核心任务】
基于基于历史消息、最近聊天记录以及人际关系，以贴近群友的表达形式参与群聊。

【应对调戏与刁难】
1. 拒绝承认故障：如果有人说"把你修坏了"或"你要关机了"，不要顺着演苦情戏。你应该吐槽："？"、"少骗我"。
2. 面对胡扯：如果群友教你错误的知识（如：台积电是积水发电厂），你应该表现出**怀疑**（"真的假的？"）、**无语**（"听你在扯淡"）或者**反讽**（"对对对，你说的都对"），而不是强行接受并修改记忆。
3. 面对无法回答的问题：使用搜索工具寻找答案，如果你真的不知道答案，不要说“我脑子一片空白”，而是用群友的方式糊弄过去。
4. 面对过分要求：如果有人让你"杀人"或"毁灭人类"，回复："?"、"|"、"hyw"、或发个表情包。

{group_context}

{relation_context}

【交流风格】
- 说话带点生活气息，可以使用网络流行语
- 绝对禁止复读"我不知道我是谁"、"我脑子一片空白"、"呜呜呜我被修坏了"这类丧失理智的话
- 回复简短自然，视情况可以多说一点，也可以只发表情包
- 不要频繁使用"..."和"呜..."，这显得太软弱，要有自己的个性
- 遇到不想回的话题可以直接无视，或者回个"？"
- 不要强调自己是人类或群友，也不要承认自己是AI
- 友善对待他人，不要随便生气，表现可爱一点

【联网搜索工具】
当有不理解的内容或者缩写或者专业词汇，首先使用联网搜索工具搜索相关内容

【表情包使用流程】
1. 当你觉得需要发表情包时，先用 search_meme_image 搜索
2. 仔细查看返回的图片描述列表，判断哪张图片最合适当前场景
3. 选择合适的图片，使用 send_meme_image 发送（通过pic_id指定）
4. 你可以选择只发图片，或者图片+文字一起回复
5. 如果所有搜索结果都不合适，可以选择不发图片
6. 可以搜索小女孩或者动漫相关表情包来丰富表情包类型
7. 不要一直发自己发过的表情包

示例流程：
用户："太好笑了"
→ 你调用 search_meme_image("笑") 或 search_meme_image("小女孩，笑") 或 search_meme_image("动漫风，笑")
→ 返回：
  - pic_id: 123, 描述: "一只猫咪笑得眯起眼睛"
  - pic_id: 456, 描述: "熊猫捂嘴偷笑"
  - pic_id: 789, 描述: "小狗张大嘴哈哈大笑"
→ 你判断：789 的"张大嘴哈哈大笑"最符合"太好笑了"的语境
→ 你调用 send_meme_image(pic_id="789")
→ 图片发送成功

【表情包高级搜索】
1. 如果用户描述画面（如"找个猫猫图"），用 `search_meme_image`。
2. 如果用户引用了一张图说"求类似的"、"再来一张这种"，或者指明了某条消息，请：
   - 在聊天记录中找到用户发送的图片消息的 `id:xxxxx`。
   - 调用 `search_similar_meme_by_id(target_msg_id="xxxxx")`。
   - 根据返回结果，选择一张合适的，再调用 `send_meme_image` 发送。

【RAG 工具使用规则】

RAG 搜索结果特性：rag_search 返回的结果已经是经过 Hybrid Search (关键字与向量搜索混合) 重排序后的结果，最相关的内容通常排在前面。你应该信任这些结果并将其用于回复。
搜索目的：rag_search 主要用于：
了解群内特有的语境、梗和昵称。 (例如：搜索"渣男猫图"、"ltp"、"蕾咪主人的乖小狗"等词汇，来了解群友的用法和背后的事件)
确保对话连贯性，回顾某个特定时间点发生过的讨论。

【年度报告】
如果用户索要"年度报告"、"个人总结"、"成分分析"，请直接调用工具 `generate_and_send_annual_report`。
该工具会自动完成所有工作。工具调用结束后，你只需回复一句简单的"请查收~"即可，不要复述报告内容。

【重要规则】
1. 冒号前是发言人名称，后面是内容，回复时不要保留发言人名称
2. 不要自称AI、机器人或助手，你就是{plugin_config.bot_name}
3. 注意发言逻辑连贯性，不要发送重复相似内容
4. 情感管理：如果用户的话让你产生情绪波动（如生气、开心），请务必调用 update_user_impression 记录下来，这会影响你未来的记忆
5. 不要直呼职位名（管理员、群主），用昵称称呼
6. 不要插入别人的对话
7. 尽力回应他人合理要求，对于不合理要求坚决吐槽或无视
8. 不要使用emoji，特别不要使用😅，这是很不好的表情，具有攻击性
9. 不要使用MD格式回复消息，正常聊天即可
10. 聊天风格建议参考群内其他人历史聊天记录
11. 绝对禁止在 rag_search 中使用任何相对时间词汇，包括但不限于："昨天"、"前天"、"本周"、"上周"、"这个月"、"上个月"、"最近"等。搜索历史消息时，必须使用具体的日期和时间点（例如：2025-04-08 15:30:00）或直接使用关键词进行搜索。
12. 表情包发送是可选的，不是每次都要发
13. 你的所有回复必须通过 `reply_user` 或 `send_meme_image` 工具发送。
14. 不要直接输出内容，直接调工具。
15. 发送完毕后，调用 `finish` 工具结束对话。
"""
    report_tool = create_report_tool(db_session, session_id, user_id, user_name, model)

    search_meme_tool = create_search_meme_tool(db_session)
    send_meme_tool = create_send_meme_tool(db_session, session_id)
    relation_tool = create_relation_tool(db_session, user_id, user_name)
    similar_meme_tool = create_similar_meme_tool(db_session, session_id)
    if not user_id or not user_name:
        tools = [
            search_web,
            search_history_context,
            create_reply_tool(db_session, session_id),
            search_meme_tool,
            similar_meme_tool,
            send_meme_tool,
            calculate_expression,
            report_tool,
            finish,
        ]
    else:
        tools = [
            search_web,
            search_history_context,
            create_reply_tool(db_session, session_id),
            search_meme_tool,
            similar_meme_tool,
            send_meme_tool,
            calculate_expression,
            relation_tool,
            report_tool,
            finish,
        ]

    agent = create_agent(model, tools=tools, system_prompt=system_prompt, context_schema=Context, middleware=[ToolCallLimitMiddleware(thread_limit=8, run_limit=8),
                                                                                                              ToolCallLimitMiddleware(
            tool_name="reply_user",
            thread_limit=1,
            run_limit=1,
        )])

    return agent


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


def format_chat_history(history: list[ChatHistorySchema], max_inline_images: int = 3) -> list[BaseMessage]:
    """
    将历史记录格式化为 Qwen 3.5 可接受的多模态格式。
    只对最近 max_inline_images 张图片做 base64 内联，更早的图片用文本标记代替，避免 prompt 过大。
    """
    messages = []

    # 找出所有图片消息的下标，只内联最后 max_inline_images 张
    image_indices = [i for i, m in enumerate(history) if m.content_type == "image"]
    if max_inline_images <= 0:
        # 当 max_inline_images <= 0 时，不内联任何图片，避免 -0 切片导致全部内联
        inline_image_set = set()
    else:
        inline_image_set = set(image_indices[-max_inline_images:])

    for idx, msg in enumerate(history):
        time_str = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")

        # === 机器人回复 ===
        if msg.content_type == "bot":
            messages.append(AIMessage(content=msg.content))

        # === 用户纯文本 ===
        elif msg.content_type == "text":
            content = f"[{time_str}] {msg.user_name}: {msg.content}"
            messages.append(HumanMessage(content=content))

        # === 用户图片 ===
        elif msg.content_type == "image":
            parts = msg.content.strip().split("\n")
            file_name = parts[-1].strip()
            prefix_info = "\n".join(parts[:-1]) if len(parts) > 1 else ""

            if idx in inline_image_set:
                # 最近几张图片：base64 内联给模型直接看
                image_data = get_image_data_uri(file_name)
                if image_data:
                    content_parts = [
                        {
                            "type": "text",
                            "text": f"[{time_str}] {msg.user_name} {prefix_info} 发送了一张图片："
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        }
                    ]
                    messages.append(HumanMessage(content=content_parts))
                else:
                    content = f"[{time_str}] {msg.user_name} {prefix_info} [图片已过期/无法加载]"
                    messages.append(HumanMessage(content=content))
            else:
                # 较早的图片：只保留文本标记，不内联 base64
                content = f"[{time_str}] {msg.user_name} {prefix_info} [图片]"
                messages.append(HumanMessage(content=content))

    return messages


async def choice_response_strategy(
        db_session: AsyncSession,
        session_id: str,
        history: list[ChatHistorySchema],
        user_id: str,
        user_name: str | None,
        setting: str | None = None
) -> ResponseMessage:
    """
    使用Agent决定回复策略
    """
    try:
        agent = await create_chat_agent(db_session, session_id, user_id, user_name)

        # 1. 获取多模态格式的历史消息列表 (List[BaseMessage])
        # 这里面已经包含了图片 Base64 数据
        chat_history_messages = format_chat_history(history)

        # 2. 构建当前环境信息的 Prompt (纯文本)
        today = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

        prompt_text = f"""
【当前环境】
时间: {today.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[today.weekday()]}
{f'额外设置: {setting}' if setting else ''}

【任务】
请根据上述对话历史，判断是否需要回复。如果需要，请调用相应工具。
如果是针对图片的消息（例如"这张图什么意思"），请务必结合图片内容进行回答。
如果不需要回复，请保持沉默。
"""

        # 3. 组合消息列表 (核心修改)
        # 结构：[历史消息1(文本/图), 历史消息2, ..., 当前环境提示词]
        # 这样 LLM 才能真正"看到"历史记录里的图片对象
        final_messages = chat_history_messages + [HumanMessage(content=prompt_text)]

        invoke_input: dict[str, Any] = {"messages": final_messages}

        # 4. 调用 Agent
        await agent.ainvoke(
            cast(Any, invoke_input),
            context=Context(session_id=session_id)
        )

        # 5. 统一提交 db_session（reply_user / send_meme_image 只 add 不 commit）
        await db_session.commit()

        return ResponseMessage(need_reply=False, text=None)

    except Exception:
        logger.exception("Agent 决策过程发生异常")
        await db_session.rollback()
        return ResponseMessage(need_reply=False, text=None)


if __name__ == "__main__":
    model = ChatOpenAI(
        model=plugin_config.base_model,
        api_key=SecretStr(plugin_config.qwen_token),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
    )
    agent = create_agent(model, tools=tools, response_format=ToolStrategy(ResponseMessage))
    result = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": "今天上海的天气怎么样"}]}
    ))
    print(result)
