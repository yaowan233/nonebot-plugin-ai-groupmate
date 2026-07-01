import inspect
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Callable, Iterable, Awaitable

from langchain.tools import tool
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


AgentSkillPrompt = str | Callable[["AgentToolContext"], str | Awaitable[str]]


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    prompt: AgentSkillPrompt


@dataclass(frozen=True)
class AgentToolBundle:
    tools: Iterable[Any] = field(default_factory=tuple)
    instructions: Iterable[str] = field(default_factory=tuple)
    skills: Iterable[AgentSkill] = field(default_factory=tuple)


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
    tools, instructions, _ = await build_registered_agent_extensions(context)
    return tools, instructions


async def build_registered_agent_extensions(
    context: AgentToolContext,
) -> tuple[list[Any], list[str], list[AgentSkill]]:
    tools: list[Any] = []
    instructions: list[str] = []
    skills: list[AgentSkill] = []

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
        skills.extend(
            skill
            for skill in bundle.skills
            if skill and skill.name.strip() and skill.description.strip()
        )

    return tools, instructions, skills


def build_agent_skill_index(skills: Iterable[AgentSkill]) -> str:
    lines = [
        f"- {skill.name.strip()}: {skill.description.strip()}"
        for skill in skills
        if skill.name.strip() and skill.description.strip()
    ]
    if not lines:
        return ""
    return (
        "【可按需读取的技能】\n"
        "下面是可用技能的简短索引。只有当当前任务明显需要某个技能时，才调用 `load_agent_skill` 读取完整说明；不要预先读取无关技能。\n"
        + "\n".join(lines)
        + "\n"
    )


def create_agent_skill_loader_tool(
    skills: Iterable[AgentSkill],
    context: AgentToolContext,
) -> Any | None:
    skill_map = {skill.name.strip(): skill for skill in skills if skill.name.strip()}
    if not skill_map:
        return None

    @tool("load_agent_skill")
    async def load_agent_skill(skill_name: str) -> str:
        """
        按名称读取一个技能的完整 prompt。仅当用户请求明显需要该技能时调用。
        输入必须是技能索引中列出的 skill_name。
        """
        skill = skill_map.get(skill_name.strip())
        if skill is None:
            available = ", ".join(skill_map)
            return f"未找到技能: {skill_name}。可用技能: {available}"

        prompt = skill.prompt
        if callable(prompt):
            prompt = prompt(context)
        if inspect.isawaitable(prompt):
            prompt = await prompt
        return str(prompt).strip()

    return load_agent_skill
