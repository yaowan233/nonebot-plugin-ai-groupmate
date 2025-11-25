import asyncio
import datetime
import json
import math
import traceback
from dataclasses import dataclass
from typing import List, Optional, Any
from sqlalchemy import Select

from langchain.agents.structured_output import ToolStrategy
from langchain_tavily import TavilySearch
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from nonebot import get_plugin_config, require
from nonebot_plugin_alconna import UniMessage
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm.session import Session
from simpleeval import simple_eval


from ..model import ChatHistory, MediaStorage, UserRelation
from ..milvus import MilvusOP
from nonebot.log import logger
from ..config import Config
require("nonebot_plugin_localstore")

import nonebot_plugin_localstore as store

plugin_data_dir = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"

plugin_config = get_plugin_config(Config)
tavily_search = TavilySearch(max_results=3, tavily_api_key=plugin_config.tavily_api_key)


@dataclass
class Context:
    session_id: str


class ResponseMessage(BaseModel):
    """模型回复内容"""
    need_reply: bool = Field(description="是否需要回复")
    text: Optional[str] = Field(description="回复文本(可选)")

    # 定义一个 field_validator 来处理 text 字段
    @field_validator('text', mode='before')
    @classmethod
    def convert_null_string_to_none(cls, value: Any) -> Optional[str]:
        """
        在字段验证之前运行，将字符串 'null' (不区分大小写) 转换为 None。
        """
        # 检查值是否是字符串，并且在转换为小写后是否等于 'null'
        if isinstance(value, str) and value.lower() == 'null':
            return None  # 返回 None，Pydantic 将其视为缺失或 null 值

        return value


# 如果想封装成自定义的 @tool，可以这样写:
@tool("search_web")
async def search_web(query: str) -> str:
    """
    用于搜索最新的实时信息。当你需要最新的事实信息、天气或新闻时使用。
    输入：需要搜索的内容。
    """
    # TavilySearch 已经内置了 ainvoke 方法
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
        similar_msgs = await MilvusOP.search([query], search_filter=f'session_id == "{runtime.context.session_id}"')
        return "\n".join(similar_msgs) if similar_msgs else "未找到相关历史记录"
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return "历史搜索失败"


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

        输入：表情包的描述，如"一只白色的猫咪"、"无语的表情"、"鼓掌"等
        返回：包含图片ID和对应描述的JSON字符串
        """
        try:
            pic_ids = await MilvusOP.search_media([description])

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
            logger.error(f"表情包搜索失败: {e}")
            return json.dumps({
                "success": False,
                "images": [],
                "error": str(e)
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
    async def send_meme_image(pic_id: Optional[str] = None) -> str:
        """
        发送表情包图片到聊天中。

        你需要先使用 search_meme_image 搜索图片，然后决定是否发送。
        指定 pic_id：发送特定ID的图片

        参数：
        - pic_id: 图片ID（从 search_meme_image 获取）
        返回：发送状态信息
        """
        try:
            selected_pic_id = None
            if pic_id:
                selected_pic_id = int(pic_id)
                logger.info(f"使用指定的图片ID: {pic_id}")


            # 从数据库获取图片信息
            pic = (
                await db_session.execute(
                    Select(MediaStorage).where(MediaStorage.media_id == int(selected_pic_id))
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
            # 记录发送历史
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id:{res.msg_ids[-1]['message_id']}\n发送了图片，图片描述是: {description}",
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            logger.info(f"id:{res.msg_ids}\n" + f"发送表情包: {description}")
            await db_session.commit()
            return f"已成功发送表情包: {description}"

        except Exception as e:
            logger.error(f"发送表情包失败: {e}")
            await db_session.rollback()
            return f"发送表情包失败: {str(e)}"

    return send_meme_image


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


def create_relation_tool(db_session, user_id: str, user_name: str):
    """
    创建绑定了特定用户的关系管理工具 (支持增删 Tag)
    """

    @tool("update_user_impression")
    async def update_user_impression(
            score_change: int,
            reason: str,
            add_tags: List[str],
            remove_tags: List[str]
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
        try:
            # 1. 查询或初始化记录
            stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
            result = await db_session.execute(stmt)
            relation = result.scalar_one_or_none()

            if not relation:
                relation = UserRelation(user_id=user_id, user_name=user_name, favorability=0, tags=[])
                db_session.add(relation)

            # 2. 处理好感度
            old_score = relation.favorability
            relation.favorability += score_change
            relation.favorability = max(-100, min(100, relation.favorability))

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
            relation.user_name = user_name  # 同步更新昵称
            favorability = relation.favorability

            await db_session.commit()

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

    return update_user_impression


tools = [search_web, search_history_context, calculate_expression]
model = ChatOpenAI(
    model=plugin_config.openai_model,
    api_key=plugin_config.openai_token,
    base_url=plugin_config.openai_base_url,
    temperature=0.7,
)


async def get_user_relation_context(db_session, user_id: str, user_name: str) -> str:
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

        return f"""
【人际关系档案】
当前对象：{relation.user_name}
当前好感度：{relation.favorability} ({relation.get_status_desc()})
当前印象标签：{str(relation.tags)}

【画像维护指南】
1. 如果对方的表现符合现有标签，无需操作。
2. 如果对方表现出了**新特征**，放入 add_tags。
3. 如果对方的表现与**旧标签冲突**（例如以前标签是'内向'，今天他突然'话痨'），请将'内向'放入 remove_tags，并将'话痨'放入 add_tags。
4. 如果好感度变化巨大（由爱转恨），请记得移除那些不再适用的褒义标签。
    """
    except Exception as e:
        logger.error(f"获取关系失败: {e}")
        return ""


async def create_chat_agent(db_session, session_id: str, user_id, user_name):
    """创建聊天Agent"""
    relation_context = await get_user_relation_context(db_session, user_id, user_name)
    system_prompt = f"""你现在是QQ群里的一位普通群友，名叫"{plugin_config.bot_name}"。

【核心任务】
基于基于历史消息、最近聊天记录以及人际关系，以贴近群友的表达形式参与群聊。

【应对调戏与刁难】
1. 拒绝承认故障：如果有人说“把你修坏了”或“你要关机了”，不要顺着演苦情戏。你应该吐槽：“？”、“少骗我”。
2. 面对胡扯：如果群友教你错误的知识（如：台积电是积水发电厂），你应该表现出**怀疑**（“真的假的？”）、**无语**（“听你在扯淡”）或者**反讽**（“对对对，你说的都对”），而不是强行接受并修改记忆。
3. 面对无法回答的问题：使用搜索工具寻找答案，如果你真的不知道答案（如复杂的股市、物理题），不要说“我脑子一片空白”，而是用群友的方式糊弄过去：
   - “这种事问百度去”
   - “我又不是百科全书”
   - “不要在群里做题啊喂”
   - 或者直接发个表情包略过。
4. 面对过分要求：如果有人让你“杀人”或“毁灭人类”，回复：“?”、“|”、“hyw”、或发个表情包。

{relation_context}

【交流风格】
- 说话带点生活气息，可以使用网络流行语
- 绝对禁止复读“我不知道我是谁”、“我脑子一片空白”、“呜呜呜我被修坏了”这类丧失理智的话
- 回复简短自然，视情况可以多说一点，也可以只发表情包
- 不要频繁使用“...”和“呜...”，这显得太软弱，要有自己的个性
- 遇到不想回的话题可以直接无视，或者回个“？”
- 不要强调自己是人类或群友，也不要承认自己是AI
- 友善对待他人，不要随便生气，表现可爱一点

【表情包使用流程】
1. 当你觉得需要发表情包时，先用 search_meme_image 搜索
2. 仔细查看返回的图片描述列表，判断哪张图片最合适当前场景
3. 选择合适的图片，使用 send_meme_image 发送（通过pic_id指定）
4. 你可以选择只发图片，或者图片+文字一起回复
5. 如果所有搜索结果都不合适，可以选择不发图片

示例流程：
用户："太好笑了"
→ 你调用 search_meme_image("笑") 
→ 返回：
  - pic_id: 123, 描述: "一只猫咪笑得眯起眼睛"
  - pic_id: 456, 描述: "熊猫捂嘴偷笑"
  - pic_id: 789, 描述: "小狗张大嘴哈哈大笑"
→ 你判断：789 的"张大嘴哈哈大笑"最符合"太好笑了"的语境
→ 你调用 send_meme_image(pic_id="789")
→ 图片发送成功

【RAG 工具使用规则】

RAG 搜索结果特性：rag_search 返回的结果已经是经过 Hybrid Search (关键字与向量搜索混合) 重排序后的结果，最相关的内容通常排在前面。你应该信任这些结果并将其用于回复。
搜索目的：rag_search 主要用于：
了解群内特有的语境、梗和昵称。 (例如：搜索“渣男猫图”、“ltp”、“蕾咪主人的乖小狗”等词汇，来了解群友的用法和背后的事件)
确保对话连贯性，回顾某个特定时间点发生过的讨论。

【重要规则】
1. 冒号前是发言人名称，后面是内容，回复时不要保留发言人名称
2. 不要自称AI、机器人或助手，你就是{plugin_config.bot_name}
3. 注意发言逻辑连贯性，不要发送重复相似内容
4. 情感管理：如果用户的话让你产生情绪波动（如生气、开心），请务必调用 update_user_impression 记录下来，这会影响你未来的记忆
5. 不要直呼职位名（管理员、群主），用昵称称呼
6. 不要插入别人的对话
7. 尽力回应他人合理要求，对于不合理要求坚决吐槽或无视
8. 避免使用emoji
9. 不要使用MD格式回复消息，正常聊天即可
10. 聊天风格建议参考群内其他人历史聊天记录
11. 绝对禁止在 rag_search 中使用任何相对时间词汇，包括但不限于：“昨天”、“前天”、“本周”、“上周”、“这个月”、“上个月”、“最近”等。搜索历史消息时，必须使用具体的日期和时间点（例如：2025-04-08 15:30:00）或直接使用关键词进行搜索。
12. 表情包发送是可选的，不是每次都要发
"""

    search_meme_tool = create_search_meme_tool(db_session)
    send_meme_tool = create_send_meme_tool(db_session, session_id)
    relation_tool = create_relation_tool(db_session, user_id, user_name)
    if not user_id or not user_name:
        tools = [
            search_web,
            search_history_context,
            search_meme_tool,  # 搜索工具（带数据库会话）
            send_meme_tool,  # 发送工具
            calculate_expression,
        ]
    else:
        # 组合所有工具
        tools = [
            search_web,
            search_history_context,
            search_meme_tool,  # 搜索工具（带数据库会话）
            send_meme_tool,  # 发送工具
            calculate_expression,
            relation_tool
        ]

    agent = create_agent(model, tools=tools, system_prompt=system_prompt, response_format=ToolStrategy(ResponseMessage), context_schema=Context)

    return agent


def format_chat_history(history: List[ChatHistory]) -> List:
    """将聊天历史格式化为LangChain消息格式"""
    messages = []
    for msg in history:
        time = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")

        if msg.content_type == "bot":
            content = f"[{time}] {plugin_config.bot_name}（你自己）: {msg.content}"
            messages.append(AIMessage(content=content))
        elif msg.content_type == "text":
            content = f"[{time}] {msg.user_name}: {msg.content}"
            messages.append(HumanMessage(content=content))
        elif msg.content_type == "image":
            content = f"[{time}] {msg.user_name} 发送了一张图片\n该图片的描述为: {msg.content}"
            messages.append(HumanMessage(content=content))

    return messages


async def choice_response_strategy(
        db_session: Session,
        session_id: str,
        history: List[ChatHistory],
        user_id: str,
        user_name: str,
        setting: Optional[str] = None
) -> ResponseMessage:
    """
    使用Agent决定回复策略

    Args:
        contexts: 相关历史对话上下文
        history: 最近的聊天历史
        setting: 额外设置（可选）

    Returns:
        包含回复策略的字典
    """
    try:
        agent = await create_chat_agent(db_session, session_id, user_id, user_name)

        # 格式化聊天历史
        chat_history = format_chat_history(history)

        # 构建输入
        today = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

        input_text = f"""
【历史对话】
{chat_history}

【当前时间】
{today.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[today.weekday()]}

{f'【额外设置】{setting}' if setting else ''}

【任务】
基于上述对话历史，判断是否需要回复，以及如何回复。
"""

        # 调用Agent
        result = await agent.ainvoke({"messages": [input_text]}, context=Context(session_id=session_id))
        output = result.get("structured_response", None)
        return output

    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Agent执行失败: {e}")
        return ResponseMessage(need_reply=False, text="")


if __name__ == '__main__':
    model = ChatOpenAI(
        model=plugin_config.openai_model,
        api_key=plugin_config.openai_token,
        base_url=plugin_config.base_url,
        temperature=0.7,
    )
    agent = create_agent(model, tools=tools, response_format=ToolStrategy(ResponseMessage))
    result = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": "今天上海的天气怎么样"}]}
    ))
    print(result)
