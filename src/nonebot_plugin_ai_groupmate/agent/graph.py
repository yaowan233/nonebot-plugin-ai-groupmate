"""LangGraph-based agent replacement for create_agent + middleware."""
import json
import time
import asyncio
import hashlib
from typing import Any, Annotated, TypedDict
from dataclasses import dataclass
from collections.abc import Sequence

from nonebot.log import logger
from langchain.tools import ToolRuntime
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

from ..reply_guard import is_request_active
from .prompt_cache import normalize_system_messages

MAX_REPLY_COUNT = 5
MAX_TOOL_COUNT = 20
MAX_REPLY_PER_ROUND = 1  # 每轮只发1条，强制模型逐条思考，上下文连续
MAX_REACTION_PER_ROUND = 3
SIDE_EFFECT_TOOL_NAMES = frozenset({
    "add_message_reaction",
    "generate_and_send_annual_report",
    "mute_user",
    "recall_message",
    "reply_user",
    "schedule_agent_task",
    "schedule_message",
    "send_meme_image",
    "send_private_message",
    "update_user_impression",
})

ContentBlock = str | dict[str, Any]


@dataclass(frozen=True)
class AgentRunLimits:
    max_llm_calls: int = 8
    max_total_tokens: int = 64_000
    llm_timeout_seconds: float = 60.0
    tool_timeout_seconds: float = 30.0
    tool_result_max_chars: int = 6_000


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    request_id: str | None
    reply_count: int
    tool_count: int
    reply_this_round: int
    reply_requires_continuation: bool
    reaction_this_round: int
    called_finish: int
    llm_cached_tokens: int
    llm_cache_creation_tokens: int
    llm_call_count: int
    llm_total_tokens: int
    tool_timeout_count: int
    tool_result_truncation_count: int
    side_effect_duplicate_count: int
    completed_side_effect_keys: list[str]
    active_skills: list[str]


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


def _log_llm_cache_usage(response: AIMessage) -> dict[str, int]:
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
    cache_creation_tokens = _first_int(
        combined,
        (
            ("usage_metadata", "input_token_details", "cache_creation"),
            ("usage_metadata", "input_token_details", "cache_creation_input_tokens"),
            ("usage_metadata", "input_tokens_details", "cache_creation_input_tokens"),
            ("response_metadata", "token_usage", "prompt_tokens_details", "cache_creation_input_tokens"),
            ("response_metadata", "token_usage", "input_tokens_details", "cache_creation_input_tokens"),
            ("response_metadata", "token_usage", "cache_creation_input_tokens"),
            ("response_metadata", "token_usage", "cache_write_input_tokens"),
            ("token_usage", "prompt_tokens_details", "cache_creation_input_tokens"),
            ("token_usage", "input_tokens_details", "cache_creation_input_tokens"),
            ("token_usage", "cache_creation_input_tokens"),
            ("token_usage", "cache_write_input_tokens"),
        ),
    )

    if cached_tokens is None:
        logger.info(
            f"[LLM缓存] 输入={input_tokens or 0} 输出={output_tokens or 0} "
            f"总计={total_tokens or 0} 缓存命中=未返回 缓存创建={cache_creation_tokens or 0}"
        )
    else:
        hit_rate = cached_tokens / input_tokens * 100 if input_tokens else 0
        logger.info(
            f"[LLM缓存] 输入={input_tokens or 0} 缓存命中={cached_tokens} "
            f"命中率={hit_rate:.1f}% 缓存创建={cache_creation_tokens or 0} "
            f"输出={output_tokens or 0} 总计={total_tokens or 0}"
        )
    logger.debug(f"[LLM usage_metadata] {usage}")
    logger.debug(f"[LLM response_metadata] {metadata}")
    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens or 0,
        "cached_tokens": cached_tokens or 0,
        "cache_creation_tokens": cache_creation_tokens or 0,
    }


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


def _normalize_tool_result(result: Any) -> tuple[str, list[ContentBlock] | None]:
    if isinstance(result, str):
        return result, None
    if isinstance(result, list) and all(isinstance(item, dict) for item in result):
        content_blocks: list[ContentBlock] = list(result)
        text_parts = [
            item.get("text", "")
            for item in content_blocks
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        return "\n".join(text_parts) or "工具已返回多模态内容", content_blocks
    return str(result), None


def _tool_result_status(content: str) -> str | None:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    status = parsed.get("status") if isinstance(parsed, dict) else None
    return status if status in {"sent", "skipped", "failed"} else None


def _truncate_tool_content(content: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(content) <= max_chars:
        return content, False
    prefix_length = max(1, max_chars * 3 // 4)
    suffix_length = max(1, max_chars - prefix_length)
    return (
        f"{content[:prefix_length]}\n\n[工具结果过长，已截断]\n\n{content[-suffix_length:]}",
        True,
    )


def _side_effect_key(name: str, args: dict[str, Any]) -> str:
    stable_args = {key: value for key, value in args.items() if key != "next_step"}
    payload = json.dumps(stable_args, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{name}:{digest}"


def _estimate_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return max(1, len(content) // 4)
    if isinstance(content, dict):
        if content.get("type") == "image_url":
            return 1_024
        text = content.get("text")
        if isinstance(text, str):
            return max(1, len(text) // 4)
        return max(1, len(json.dumps(content, ensure_ascii=False, default=str)) // 4)
    if isinstance(content, list):
        return sum(_estimate_content_tokens(item) for item in content)
    return max(1, len(str(content)) // 4)


def _estimate_message_tokens(messages: Sequence[BaseMessage]) -> int:
    return sum(_estimate_content_tokens(message.content) for message in messages)


def _message_text_content(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


def _active_tools(
    base_tools: list[BaseTool],
    tools_by_skill: dict[str, list[BaseTool]],
    active_skills: Sequence[str],
) -> list[BaseTool]:
    """Return the currently visible tools, preserving their declared order."""
    tools = list(base_tools)
    known_names = {tool.name for tool in tools}
    for skill_name in active_skills:
        for tool in tools_by_skill.get(skill_name, []):
            if tool.name not in known_names:
                tools.append(tool)
                known_names.add(tool.name)
    return tools


def _make_agent_node(
    model: Any,
    base_tools: list[BaseTool],
    system_prompt: str | Sequence[BaseMessage],
    tools_by_skill: dict[str, list[BaseTool]],
    limits: AgentRunLimits,
) -> Any:
    system_messages = normalize_system_messages(system_prompt)
    bound_models: dict[tuple[str, ...], Any] = {}

    async def agent_node(state: AgentState) -> dict:
        visible_tools = _active_tools(
            base_tools,
            tools_by_skill,
            state.get("active_skills", []),
        )
        tool_names = tuple(tool.name for tool in visible_tools)
        bound_model = bound_models.get(tool_names)
        if bound_model is None:
            bound_model = model.bind_tools(visible_tools)
            bound_models[tool_names] = bound_model
        full: list[BaseMessage] = system_messages + list(state["messages"])
        call_number = state.get("llm_call_count", 0) + 1
        started_at = time.perf_counter()
        try:
            response: AIMessage = await asyncio.wait_for(
                bound_model.ainvoke(full),
                timeout=limits.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[AgentTrace] LLM 超时 session={state['session_id']} call={call_number} "
                f"timeout={limits.llm_timeout_seconds:.1f}s"
            )
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        usage = _log_llm_cache_usage(response)
        budget_tokens = usage["total_tokens"] or _estimate_message_tokens([*full, response])
        logger.info(
            f"[AgentTrace] LLM session={state['session_id']} call={call_number} "
            f"duration_ms={elapsed_ms:.0f} visible_tools={len(visible_tools)} "
            f"tokens={budget_tokens}{' (估算)' if not usage['total_tokens'] else ''}"
        )
        return {
            "messages": [response],
            "reply_this_round": 0,
            "reply_requires_continuation": False,
            "called_finish": 0,
            "llm_cached_tokens": state.get("llm_cached_tokens", 0) + usage["cached_tokens"],
            "llm_cache_creation_tokens": state.get("llm_cache_creation_tokens", 0)
            + usage["cache_creation_tokens"],
            "llm_call_count": call_number,
            "llm_total_tokens": state.get("llm_total_tokens", 0) + budget_tokens,
        }

    return agent_node


def _make_tool_node(
    tools_by_name: dict[str, BaseTool],
    base_tools: list[BaseTool],
    tools_by_skill: dict[str, list[BaseTool]],
    limits: AgentRunLimits,
):
    async def tool_node(state: AgentState) -> dict:
        messages = state["messages"]
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return {}

        tool_calls = last_message.tool_calls or []
        results: list[BaseMessage] = []
        reply_count = state.get("reply_count", 0)
        tool_count = state.get("tool_count", 0)
        reply_this_round = state.get("reply_this_round", 0)
        reply_requires_continuation = state.get("reply_requires_continuation", False)
        reaction_this_round = state.get("reaction_this_round", 0)
        called_finish = 0
        active_skills = list(state.get("active_skills", []))
        tool_timeout_count = state.get("tool_timeout_count", 0)
        tool_result_truncation_count = state.get("tool_result_truncation_count", 0)
        side_effect_duplicate_count = state.get("side_effect_duplicate_count", 0)
        completed_side_effect_keys = list(state.get("completed_side_effect_keys", []))
        session_id = state["session_id"]
        request_id = state["request_id"]

        agent_ctx = _AgentContext(session_id=session_id, request_id=request_id)

        if not tool_calls:
            direct_reply = _message_text_content(last_message)
            if direct_reply:
                reply_tool = tools_by_name.get("reply_user")
                if reply_tool is None:
                    logger.warning("[Agent] 模型直接返回文本，但 reply_user 工具不存在，无法发送")
                elif request_id and not await is_request_active(session_id, request_id):
                    logger.info("[Agent] 模型直接返回文本，但请求已过期，跳过发送")
                else:
                    try:
                        result = await reply_tool.ainvoke(
                            {"content": direct_reply, "next_step": "end"}
                        )
                        logger.info(f"[Agent] 已兜底发送模型直接回复: {result}")
                        reply_count += 1
                    except Exception as e:
                        logger.error(f"[Agent] 兜底发送模型直接回复失败: {e}")
            return {
                "messages": results,
                "reply_count": reply_count,
                "tool_count": tool_count,
                "reply_this_round": reply_this_round,
                "reply_requires_continuation": reply_requires_continuation,
                "reaction_this_round": reaction_this_round,
                "called_finish": 1,
                "tool_timeout_count": tool_timeout_count,
                "tool_result_truncation_count": tool_result_truncation_count,
                "side_effect_duplicate_count": side_effect_duplicate_count,
                "completed_side_effect_keys": completed_side_effect_keys,
            }

        for tc in tool_calls:
            name: str = tc["name"]
            tool_call_id = tc.get("id") or ""
            args: dict[str, Any] = tc.get("args", {})

            if tool_count >= MAX_TOOL_COUNT:
                results.append(ToolMessage(
                    content="工具调用已达本轮上限，未执行此调用。",
                    tool_call_id=tool_call_id,
                ))
                continue
            tool_count += 1

            if name == "finish":
                called_finish += 1
                results.append(ToolMessage(content="", tool_call_id=tool_call_id))
                break

            if request_id and not await is_request_active(session_id, request_id):
                results.append(ToolMessage(content="请求已过期，已取消执行", tool_call_id=tool_call_id))
                continue

            if name == "reply_user":
                if reply_this_round >= MAX_REPLY_PER_ROUND:
                    results.append(ToolMessage(
                        content="本轮已经发送过消息了。如果你想发送更多，请等待下一轮。",
                        tool_call_id=tool_call_id,
                    ))
                    continue
                reply_this_round += 1
                reply_count += 1
                # Let malformed or failed reply calls return to the model for repair.
                # A successful call below overwrites this with the explicit decision.
                reply_requires_continuation = True

            if name == "add_message_reaction":
                if reaction_this_round >= MAX_REACTION_PER_ROUND:
                    results.append(ToolMessage(
                        content="本轮表情回复已经够多了，避免刷屏。",
                        tool_call_id=tool_call_id,
                    ))
                    continue
                reaction_this_round += 1

            visible_tool_names = {
                tool.name
                for tool in _active_tools(base_tools, tools_by_skill, active_skills)
            }
            if name not in visible_tool_names:
                results.append(ToolMessage(
                    content=f"工具 {name} 当前未启用；请先调用 load_agent_skill 读取对应技能。",
                    tool_call_id=tool_call_id,
                ))
                continue

            tool = tools_by_name.get(name)
            if tool is None:
                results.append(ToolMessage(content=f"未知工具: {name}", tool_call_id=tool_call_id))
                continue

            if name in SIDE_EFFECT_TOOL_NAMES:
                effect_key = _side_effect_key(name, args)
                if effect_key in completed_side_effect_keys:
                    side_effect_duplicate_count += 1
                    results.append(ToolMessage(
                        content="相同的副作用请求已经执行过，已跳过重复执行。",
                        tool_call_id=tool_call_id,
                    ))
                    if name == "reply_user":
                        reply_requires_continuation = False
                    logger.info(
                        f"[AgentTrace] 工具去重 session={session_id} tool={name}"
                    )
                    continue
                completed_side_effect_keys.append(effect_key)

            try:
                runtime = _build_tool_runtime(agent_ctx, tool_call_id, args)
                tool_input = {**args, "runtime": runtime} if _tool_accepts_runtime(tool) else args
                started_at = time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        tool.ainvoke(tool_input, runtime=runtime),
                        timeout=limits.tool_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    tool_timeout_count += 1
                    logger.warning(
                        f"[AgentTrace] 工具超时 session={session_id} tool={name} "
                        f"timeout={limits.tool_timeout_seconds:.1f}s"
                    )
                    results.append(ToolMessage(
                        content="工具执行超时，请根据已有信息决定是否重试或换一种方式。",
                        tool_call_id=tool_call_id,
                    ))
                    continue
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                tool_content, extra_content = _normalize_tool_result(result)
                tool_status = _tool_result_status(tool_content)
                tool_content, truncated = _truncate_tool_content(
                    tool_content,
                    limits.tool_result_max_chars,
                )
                if truncated:
                    tool_result_truncation_count += 1
                results.append(ToolMessage(content=tool_content, tool_call_id=tool_call_id))
                if name == "reply_user":
                    reply_requires_continuation = (
                        args.get("next_step") == "continue"
                        if tool_status in {None, "sent"}
                        else False
                    )
                    if tool_status == "failed":
                        reply_requires_continuation = True
                if name == "load_agent_skill":
                    skill_name = str(args.get("skill_name", "")).strip()
                    if skill_name in tools_by_skill and skill_name not in active_skills:
                        active_skills.append(skill_name)
                if extra_content is not None:
                    results.append(HumanMessage(content=extra_content))
                logger.info(
                    f"[AgentTrace] 工具 session={session_id} tool={name} "
                    f"duration_ms={elapsed_ms:.0f} status={tool_status or 'unknown'} "
                    f"truncated={truncated}"
                )
            except Exception as e:
                logger.error(f"[Agent] 工具执行失败 {name}: {e}")
                results.append(ToolMessage(content=f"工具执行出错: {e}", tool_call_id=tool_call_id))

        return {
            "messages": results,
            "reply_count": reply_count,
            "tool_count": tool_count,
            "reply_this_round": reply_this_round,
            "reply_requires_continuation": reply_requires_continuation,
            "reaction_this_round": reaction_this_round,
            "called_finish": called_finish,
            "tool_timeout_count": tool_timeout_count,
            "tool_result_truncation_count": tool_result_truncation_count,
            "side_effect_duplicate_count": side_effect_duplicate_count,
            "completed_side_effect_keys": completed_side_effect_keys,
            "active_skills": active_skills,
        }

    return tool_node


def _should_call_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    if isinstance(last, AIMessage) and _message_text_content(last):
        return "tools"
    return "end"


def _should_continue(state: AgentState, limits: AgentRunLimits) -> str:
    if state.get("called_finish", 0) > 0:
        logger.info(f"[AgentTrace] 结束 session={state['session_id']} reason=finish")
        return "end"
    if state.get("reply_count", 0) >= MAX_REPLY_COUNT:
        logger.info("[Agent] 已达最大回复次数，结束本轮对话")
        return "end"
    if state.get("tool_count", 0) >= MAX_TOOL_COUNT:
        logger.info("[Agent] 已达最大工具调用次数，结束本轮对话")
        return "end"
    if state.get("llm_call_count", 0) >= limits.max_llm_calls:
        logger.info(
            f"[AgentTrace] 结束 session={state['session_id']} reason=max_llm_calls "
            f"limit={limits.max_llm_calls}"
        )
        return "end"
    if state.get("llm_total_tokens", 0) >= limits.max_total_tokens:
        logger.info(
            f"[AgentTrace] 结束 session={state['session_id']} reason=max_total_tokens "
            f"limit={limits.max_total_tokens}"
        )
        return "end"
    if state.get("reply_this_round", 0) > 0:
        if state.get("reply_requires_continuation", False):
            return "agent"
        logger.info(f"[AgentTrace] 结束 session={state['session_id']} reason=reply_end")
        return "end"
    return "agent"


def build_chat_graph(
    model: Any,
    tools: list[BaseTool],
    system_prompt: str | Sequence[BaseMessage],
    *,
    base_tools: list[BaseTool] | None = None,
    tools_by_skill: dict[str, list[BaseTool]] | None = None,
    limits: AgentRunLimits | None = None,
) -> Any:
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
    base_tools = list(base_tools) if base_tools is not None else list(tools)
    tools_by_skill = tools_by_skill or {}
    limits = limits or AgentRunLimits()

    builder = StateGraph(AgentState)
    builder.add_node("agent", _make_agent_node(model, base_tools, system_prompt, tools_by_skill, limits))
    builder.add_node("tools", _make_tool_node(tools_by_name, base_tools, tools_by_skill, limits))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_call_tools, {"tools": "tools", "end": END})
    builder.add_conditional_edges(
        "tools",
        lambda state: _should_continue(state, limits),
        {"agent": "agent", "end": END},
    )

    return builder.compile()
