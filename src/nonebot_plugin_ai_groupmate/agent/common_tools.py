from typing import Any
from dataclasses import dataclass

from simpleeval import simple_eval
from nonebot.log import logger
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch

from ..memory import DB
from ..reply_guard import is_request_active


@dataclass
class Context:
    session_id: str
    request_id: str | None = None


def create_search_web_tool(tavily_api_key: str | None):
    tavily_search = (
        TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
        if tavily_api_key
        else None
    )

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

    return search_web


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
    """
    try:
        result: Any = simple_eval(expression)
        return (
            f"计算结果是：{result:.10f}" if isinstance(result, float) else str(result)
        )
    except Exception as e:
        return f"计算失败。请检查表达式是否正确，错误信息: {e}"
