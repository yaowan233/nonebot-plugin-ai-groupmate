import inspect
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Callable, Iterable, Awaitable

from nonebot.adapters import Bot, Event
from nonebot_plugin_uninfo import QryItrface
from nonebot_plugin_alconna import Target


@dataclass(frozen=True)
class AgentToolContext:
    db_session: Any
    session_id: str
    request_id: str | None
    user_id: str | None
    user_name: str | None
    interface: QryItrface | None
    send_target: Target
    is_private: bool
    bot_id: str | None = None
    bot: Bot | None = None
    event: Event | None = None
    model: Any | None = None


@dataclass(frozen=True)
class AgentToolBundle:
    tools: Iterable[Any] = field(default_factory=tuple)
    instructions: Iterable[str] = field(default_factory=tuple)


AgentToolFactory = Callable[
    [AgentToolContext],
    Any | Iterable[Any] | AgentToolBundle | Awaitable[Any | Iterable[Any] | AgentToolBundle],
]

_registered_agent_tool_factories: list[AgentToolFactory] = []


def register_agent_tool(factory: AgentToolFactory | None = None):
    """
    Register a factory that returns one or more LangChain tools for ai-groupmate.

    Example:
        @register_agent_tool
        def build_tools(ctx: AgentToolContext):
            @tool("my_tool")
            async def my_tool(...):
                ...
            return AgentToolBundle(
                tools=[my_tool],
                instructions=["- my_tool: when to use it"],
            )
    """

    def decorator(func: AgentToolFactory) -> AgentToolFactory:
        _registered_agent_tool_factories.append(func)
        return func

    if factory is None:
        return decorator
    return decorator(factory)


def clear_registered_agent_tools() -> None:
    _registered_agent_tool_factories.clear()


def _normalize_tool_result(result: Any) -> AgentToolBundle:
    if result is None:
        return AgentToolBundle()
    if isinstance(result, AgentToolBundle):
        return result
    if isinstance(result, (str, bytes)):
        return AgentToolBundle(tools=[result])
    if isinstance(result, Iterable):
        return AgentToolBundle(tools=list(result))
    return AgentToolBundle(tools=[result])


async def build_registered_agent_tools(
    context: AgentToolContext,
) -> tuple[list[Any], list[str]]:
    tools: list[Any] = []
    instructions: list[str] = []

    for factory in list(_registered_agent_tool_factories):
        result = factory(context)
        if inspect.isawaitable(result):
            result = await result
        bundle = _normalize_tool_result(result)
        tools.extend([tool for tool in bundle.tools if tool is not None])
        instructions.extend(
            instruction.strip()
            for instruction in bundle.instructions
            if instruction and instruction.strip()
        )

    return tools, instructions
