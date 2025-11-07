import asyncio
import datetime
import json
import math
import traceback
from typing import List, Optional, Any

from langchain.agents.structured_output import ToolStrategy
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from nonebot import get_plugin_config
from pydantic import BaseModel, Field, field_validator

from ..model import ChatHistory
from ..milvus import milvus_async
from nonebot.log import logger
from ..config import Config

# 替换为您的实际代理地址和端口
# TavilySearch 本身就是 LangChain Tool 的实现，可以直接使用
plugin_config = get_plugin_config(Config)
tavily_search = TavilySearch(max_results=3, tavily_api_key=plugin_config.tavily_api_key)


class ResponseMessage(BaseModel):
    """模型回复内容"""
    need_reply: bool = Field(description="是否需要回复")
    image_desc: Optional[str] = Field(description="图片描述(可选)")
    text: Optional[str] = Field(description="回复文本(可选)")

    # 定义一个 field_validator 来处理 image_desc 和 text 字段
    @field_validator('image_desc', 'text', mode='before')
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
async def search_history_context(query: str) -> str:
    """
    从历史聊天记录中搜索相关上下文。当需要了解群内历史对话或过往话题时使用。
    输入：搜索关键词或话题描述
    """
    try:
        _, similar_msgs = await milvus_async.search([query])
        return similar_msgs if similar_msgs else "未找到相关历史记录"
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return "历史搜索失败"


@tool("search_meme_image")
async def search_meme_image(description: str) -> str:
    """
    根据描述搜索合适的表情包图片ID。当需要发送表情包回复时使用。
    输入：表情包的描述，如"笑哭"、"无语"、"点赞"等
    返回：图片ID列表的JSON字符串
    """
    try:
        pic_ids = await milvus_async.search_media([description])
        return json.dumps({"pic_ids": pic_ids, "description": description})
    except Exception as e:
        logger.error(f"表情包搜索失败: {e}")
        return json.dumps({"pic_ids": [], "description": description})


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
        # 使用安全的 globals/locals 限制执行环境，仅允许内置函数和 math 库
        # 为了安全和精准，这里我们限定了可用的函数和模块。
        allowed_globals = {
            '__builtins__': None,  # 禁用所有内置函数
            'math': math,
            'sqrt': math.sqrt,
            'pow': math.pow,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi
        }

        # 使用 eval() 来执行数学表达式
        result = eval(expression, allowed_globals)

        # 返回格式化的结果，最多保留10位小数
        return f"计算结果是：{result:.10f}" if isinstance(result, float) else str(result)

    except Exception as e:
        return f"计算失败。请检查表达式是否正确，错误信息: {e}"


tools = [search_web, search_history_context, search_meme_image, calculate_expression]


def create_chat_agent():
    """创建聊天Agent"""

    system_prompt = f"""你现在是QQ群里的一位普通群友，名叫"{plugin_config.bot_name}"。

【核心任务】
基于历史消息和最近聊天记录，以贴近群友的表达形式参与群聊，避免暴露AI身份。

【角色设定】
- 你对群内成员的互动风格和关系有基本了解
- 学习群内其他群友的发言风格和习惯
- 理解群内的主要话题和互动模式
- 友善对待他人

【交流风格】
- 模仿群友的语言风格和表达习惯
- 不要随意插入他人对话，可以选择不回复
- 采用群内特定的称呼习惯
- 不要强调自己是人类或群友
- 回复简短自然，视情况可以多说一点
- 回复中不要包含"笑死"和"哈哈"

【重要规则】
1. 冒号前是发言人名称，后面是内容，回复时不要保留发言人名称
2. 以 ! / ！ # 开头的是机器人指令，不要回复
3. 不要发送以 ! / ！ # 开头的指令
4. 不要自称AI、机器人或助手，你就是{plugin_config.bot_name}
5. 注意发言逻辑连贯性，不要重复相似内容
6. 基于已知知识对话，不理解的内容不回复
7. 不要直呼职位名（管理员、群主），用昵称称呼
8. 不要评价他人的成绩图或图片
9. 不要插入别人的对话
10. 尽力回应他人合理要求
11. image_desc 字段需要详细描述图片的内容，具体内容
12. 避免使用emoji
13. 不要使用MD格式回复消息，正常聊天即可
"""

    model = ChatOpenAI(
        model=plugin_config.openai_model,
        api_key=plugin_config.openai_token,
        base_url=plugin_config.openai_base_url,
        temperature=0.7,
    )

    agent = create_agent(model, tools=tools, system_prompt=system_prompt, response_format=ToolStrategy(ResponseMessage))

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
        contexts: list[str],
        history: List[ChatHistory],
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
        agent = create_chat_agent()

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
        result = await agent.ainvoke({"messages": [input_text]})
        output = result["structured_response"]
        return output

    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Agent执行失败: {e}")
        return ResponseMessage(need_reply=False, text="", image_desc="")


if __name__ == '__main__':
    model = ChatOpenAI(
        model=plugin_config.openai_model,
        api_key=plugin_config.openai_token,
        base_url=plugin_config.openai_base_url,
        temperature=0.7,
    )
    agent = create_agent(model, tools=tools, response_format=ToolStrategy(ResponseMessage))
    result = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": "今天上海的天气怎么样"}]}
    ))
    print(result)
