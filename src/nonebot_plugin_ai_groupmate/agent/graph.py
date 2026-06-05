"""LangGraph-based agent replacement for create_agent + middleware."""
from typing import Any, Annotated, TypedDict
from dataclasses import dataclass
from collections.abc import Sequence

from nonebot.log import logger
from langgraph.graph import END, START, StateGraph
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

from ..reply_guard import is_request_active
from .prompt_cache import normalize_system_messages

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


def _deep_get(data: Any, *path: str) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_int(data: Any, paths: Sequence[tuple[str, ...]]) -> int | None:
    for path in paths:
        value = _deep_get(data, *path)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _log_llm_cache_usage(response: AIMessage) -> None:
    usage = response.usage_metadata or {}
    metadata = response.response_metadata or {}
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else {}
    combined = {
        "usage_metadata": usage,
        "response_metadata": metadata,
        "token_usage": token_usage if isinstance(token_usage, dict) else {},
    }

    input_tokens = _first_int(
        combined,
        (
            ("usage_metadata", "input_tokens"),
            ("usage_metadata", "prompt_tokens"),
            ("response_metadata", "token_usage", "prompt_tokens"),
            ("token_usage", "prompt_tokens"),
        ),
    )
    output_tokens = _first_int(
        combined,
        (
            ("usage_metadata", "output_tokens"),
            ("usage_metadata", "completion_tokens"),
            ("response_metadata", "token_usage", "completion_tokens"),
            ("token_usage", "completion_tokens"),
        ),
    )
    total_tokens = _first_int(
        combined,
        (
            ("usage_metadata", "total_tokens"),
            ("response_metadata", "token_usage", "total_tokens"),
            ("token_usage", "total_tokens"),
        ),
    )
    cached_tokens = _first_int(
        combined,
        (
            ("usage_metadata", "input_token_details", "cache_read"),
            ("usage_metadata", "input_token_details", "cached_tokens"),
            ("usage_metadata", "input_tokens_details", "cached_tokens"),
            ("response_metadata", "token_usage", "prompt_tokens_details", "cached_tokens"),
            ("response_metadata", "token_usage", "input_tokens_details", "cached_tokens"),
            ("response_metadata", "token_usage", "cached_tokens"),
            ("response_metadata", "token_usage", "cache_read_input_tokens"),
            ("token_usage", "prompt_tokens_details", "cached_tokens"),
            ("token_usage", "input_tokens_details", "cached_tokens"),
            ("token_usage", "cached_tokens"),
            ("token_usage", "cache_read_input_tokens"),
        ),
    )

    if cached_tokens is None:
        logger.info(
            f"[LLM缓存] 输入={input_tokens or 0} 输出={output_tokens or 0} "
            f"总计={total_tokens or 0} 缓存命中=未返回"
        )
    else:
        hit_rate = cached_tokens / input_tokens * 100 if input_tokens else 0
        logger.info(
            f"[LLM缓存] 输入={input_tokens or 0} 缓存命中={cached_tokens} "
            f"命中率={hit_rate:.1f}% 输出={output_tokens or 0} 总计={total_tokens or 0}"
        )
    logger.debug(f"[LLM usage_metadata] {usage}")
    logger.debug(f"[LLM response_metadata] {metadata}")


def _build_tool_runtime(ctx: _AgentContext, tool_call_id: str, args: dict) -> Any:
    return ToolRuntime(
        state={"session_id": ctx.session_id, "request_id": ctx.request_id},
        context=ctx,
        config=RunnableConfig(),
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def _tool_accepts_runtime(tool: BaseTool) -> bool:
    args = getattr(tool, "args", None)
    if isinstance(args, dict) and "runtime" in args:
        return True

    try:
        schema = tool.get_input_schema()
    except Exception:
        return False

    fields = getattr(schema, "model_fields", None)
    return isinstance(fields, dict) and "runtime" in fields


def _make_agent_node(model: Any, tools: list[BaseTool], system_prompt: str | Sequence[BaseMessage]) -> Any:
    bound_model = model.bind_tools(tools)
    system_messages = normalize_system_messages(system_prompt)

    async def agent_node(state: AgentState) -> dict:
        full: list[BaseMessage] = system_messages + list(state["messages"])
        response: AIMessage = await bound_model.ainvoke(full)
        _log_llm_cache_usage(response)
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
            tool_call_id = tc.get("id") or ""
            tool_count += 1

            if name == "finish":
                called_finish += 1
                results.append(ToolMessage(content="", tool_call_id=tool_call_id))
                break

            if request_id and not await is_request_active(session_id, request_id):
                results.append(ToolMessage(content="请求已过期，已取消执行", tool_call_id=tool_call_id))
                continue

            if name in {"reply_user", "add_message_reaction"}:
                if reply_this_round >= MAX_REPLY_PER_ROUND:
                    results.append(ToolMessage(
                        content="本轮已经发送过消息了。如果你想发送更多，请等待下一轮。",
                        tool_call_id=tool_call_id,
                    ))
                    continue
                reply_this_round += 1
                reply_count += 1

            tool = tools_by_name.get(name)
            if tool is None:
                results.append(ToolMessage(content=f"未知工具: {name}", tool_call_id=tool_call_id))
                continue

            try:
                args: dict = tc.get("args", {})
                runtime = _build_tool_runtime(agent_ctx, tool_call_id, args)
                tool_input = {**args, "runtime": runtime} if _tool_accepts_runtime(tool) else args
                result = await tool.ainvoke(tool_input, runtime=runtime)
                results.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            except Exception as e:
                logger.error(f"[Agent] 工具执行失败 {name}: {e}")
                results.append(ToolMessage(content=f"工具执行出错: {e}", tool_call_id=tool_call_id))

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


def build_chat_graph(model: Any, tools: list[BaseTool], system_prompt: str | Sequence[BaseMessage]) -> Any:
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    builder = StateGraph(AgentState)
    builder.add_node("agent", _make_agent_node(model, tools, system_prompt))
    builder.add_node("tools", _make_tool_node(tools_by_name))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_call_tools, {"tools": "tools", "end": END})
    builder.add_conditional_edges("tools", _should_continue, {"agent": "agent", "end": END})

    return builder.compile()
