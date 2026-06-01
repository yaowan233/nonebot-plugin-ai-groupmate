"""LangGraph-based agent replacement for create_agent + middleware."""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from nonebot.log import logger

from ..reply_guard import is_request_active

MAX_REPLY_COUNT = 5
MAX_TOOL_COUNT = 20
MAX_REPLY_PER_ROUND = 1  # 每轮只发1条，强制模型逐条思考，上下文连续


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    request_id: str | None
    reply_count: int
    tool_count: int
    reply_this_round: int
    called_finish: int


@dataclass
class _AgentContext:
    session_id: str
    request_id: str | None


def _build_tool_runtime(ctx: _AgentContext, tool_call_id: str, args: dict) -> Any:
    return SimpleNamespace(
        state=ctx,
        context=ctx,
        config=RunnableConfig(),
        tool_call_id=tool_call_id,
        tool_input=args,
    )


def _make_agent_node(model: Any, tools: list[BaseTool], system_prompt: str):
    bound_model = model.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        full: list[BaseMessage] = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response: AIMessage = await bound_model.ainvoke(full)
        return {
            "messages": [response],
            "reply_this_round": 0,
            "called_finish": 0,
        }

    return agent_node


def _make_tool_node(tools_by_name: dict[str, BaseTool]):
    async def tool_node(state: AgentState) -> dict:
        messages = state["messages"]
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        tool_calls = last_message.tool_calls
        results: list[ToolMessage] = []
        reply_count = state.get("reply_count", 0)
        tool_count = state.get("tool_count", 0)
        reply_this_round = state.get("reply_this_round", 0)
        called_finish = 0
        session_id = state["session_id"]
        request_id = state["request_id"]

        agent_ctx = _AgentContext(session_id=session_id, request_id=request_id)

        for tc in tool_calls:
            name: str = tc["name"]
            tool_count += 1

            if name == "finish":
                called_finish += 1
                results.append(ToolMessage(content="", tool_call_id=tc["id"]))
                break

            if request_id and not await is_request_active(session_id, request_id):
                results.append(ToolMessage(content="请求已过期，已取消执行", tool_call_id=tc["id"]))
                continue

            if name == "reply_user":
                if reply_this_round >= MAX_REPLY_PER_ROUND:
                    results.append(ToolMessage(
                        content="本轮已经发送过消息了。如果你想发送更多，请等待下一轮。",
                        tool_call_id=tc["id"],
                    ))
                    continue
                reply_this_round += 1
                reply_count += 1

            tool = tools_by_name.get(name)
            if tool is None:
                results.append(ToolMessage(content=f"未知工具: {name}", tool_call_id=tc["id"]))
                continue

            try:
                args: dict = tc.get("args", {})
                runtime = _build_tool_runtime(agent_ctx, tc["id"], args)
                result = await tool.ainvoke(args, runtime=runtime)
                results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                logger.error(f"[Agent] 工具执行失败 {name}: {e}")
                results.append(ToolMessage(content=f"工具执行出错: {e}", tool_call_id=tc["id"]))

        return {
            "messages": results,
            "reply_count": reply_count,
            "tool_count": tool_count,
            "reply_this_round": reply_this_round,
            "called_finish": called_finish,
        }

    return tool_node


def _should_call_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def _should_continue(state: AgentState) -> str:
    if state.get("called_finish", 0) > 0:
        return "end"
    if state.get("reply_count", 0) >= MAX_REPLY_COUNT:
        logger.info("[Agent] 已达最大回复次数，结束本轮对话")
        return "end"
    if state.get("tool_count", 0) >= MAX_TOOL_COUNT:
        logger.info("[Agent] 已达最大工具调用次数，结束本轮对话")
        return "end"
    return "agent"


def build_chat_graph(model: Any, tools: list[BaseTool], system_prompt: str) -> StateGraph:
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    builder = StateGraph(AgentState)
    builder.add_node("agent", _make_agent_node(model, tools, system_prompt))
    builder.add_node("tools", _make_tool_node(tools_by_name))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_call_tools, {"tools": "tools", "end": END})
    builder.add_conditional_edges("tools", _should_continue, {"agent": "agent", "end": END})

    return builder.compile()
