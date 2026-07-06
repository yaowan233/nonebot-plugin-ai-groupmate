import asyncio
import datetime
from typing import Any
from pathlib import Path
from functools import lru_cache

from nonebot import require, get_plugin_config
from pydantic import Field, BaseModel, field_validator
from sqlalchemy import Select
from nonebot.log import logger
from nonebot.adapters import Bot, Event
from nonebot_plugin_orm import get_session
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_alconna import Target
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .graph import build_chat_graph
from ..model import ChatHistory, ChatHistorySchema
from ..usage import estimate_cost, record_token_usage, extract_cached_tokens
from ..config import Config, create_chat_llm, create_chat_openai
from .context import (
    get_group_context,
    get_user_relation_context,
    get_recent_relations_context,
)
from .prompts import (
    build_chat_system_prompt,
    build_permission_prompt_parts,
    build_reaction_tool_instruction,
)
from .reaction import is_onebot_context, create_reaction_tool
from .meme_tools import (
    create_send_meme_tool,
    create_search_meme_tool,
    create_similar_meme_tool,
)
from .reply_tools import create_reply_tool
from .common_tools import (
    finish,
    calculate_expression,
    create_search_web_tool,
    search_history_context,
)
from .conversation import (
    get_active_thread,
    update_active_thread,
    build_append_only_history,
)
from .custom_tools import (
    AgentSkill,
    AgentToolBundle,
    AgentToolContext,
    register_agent_tool,
    build_agent_skill_index,
    create_agent_skill_loader_tool,
    build_registered_agent_extensions,
)
from .prompt_cache import build_system_messages
from .recall_tools import create_recall_message_tool
from .private_tools import create_private_message_tool
from .profile_tools import create_report_tool, create_relation_tool
from .history_format import (
    parse_msg_meta,
    is_image_history,
    get_image_data_uri as _get_image_data_uri,
    format_chat_history as _format_chat_history,
    image_file_name_from_history,
    build_avatar_context_messages,
    should_include_avatar_context,
    load_replied_message_histories,
)
from .schedule_tools import (
    create_schedule_message_tool,
    create_schedule_agent_task_tool,
)
from .moderation_tools import create_mute_tool


async def _finish_db_operation(coro):
    task = asyncio.create_task(coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            logger.exception("取消期间数据库事务收尾失败")
        raise


async def _safe_rollback(db_session: AsyncSession) -> None:
    try:
        await _finish_db_operation(db_session.rollback())
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("数据库回滚失败")

__all__ = [
    "AgentToolBundle",
    "AgentToolContext",
    "AgentSkill",
    "register_agent_tool",
    "check_if_should_reply",
    "choice_response_strategy",
]

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

SCHEDULED_AGENT_HISTORY_LIMIT = 20
search_web = create_search_web_tool(plugin_config.tavily_api_key)


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
        scene_extra = "3. 如果是群友之间的闲聊、无关的刷屏、或者语意不通的消息，返回 NO。"
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
                "llm_cached_tokens": 0,
            })
            await db_session.commit()
        logger.info(f"[定时Agent任务] 已执行 {session_id}: {task}")
    except Exception as e:
        logger.exception(f"[定时Agent任务] 执行失败 {session_id}: {e}")


tools = [search_web, search_history_context, calculate_expression]


def _build_builtin_agent_skills(
    *,
    is_private: bool,
    has_admin_permission: bool,
    reaction_tool_instruction: str,
    mute_tool_instruction: str,
) -> list[AgentSkill]:
    context_name = "聊天上下文" if is_private else "群内上下文"
    skills = [
        AgentSkill(
            name="core_reply_tools",
            description="发送文字回复和结束本轮对话的基础规则。",
            prompt=(
                "基础回复规则：\n"
                "- 只能通过工具发消息，不要直接输出正文。\n"
                "- 文本回复使用 `reply_user`。\n"
                "- 回复完成后调用 `finish`。\n"
                "- 不要重复 bot 自己刚发过的内容；多条回复必须信息递进。"
                "- 群聊里可以偶尔复读群友短句、关键词或队形来参与，但不要刷屏。"
            ),
        ),
        AgentSkill(
            name="meme_tools",
            description="搜索、选择和发送表情包图片。",
            prompt=(
                "表情包工具规则：\n"
                "- 普通表情包需求：先调用 `search_meme_image(description)` 搜索合适图片。\n"
                "- 用户引用图片或要求“找一张类似这张的”：调用 `search_similar_meme_by_id(target_msg_id)`；没有明确 id 时可不传，工具会优先找当前用户最近图片。\n"
                "- 搜索工具只返回候选图片和 pic_id，不会发送。\n"
                "- 判断候选描述合适后，调用 `send_meme_image(pic_id)` 发送。\n"
                "- 发图完成后调用 `finish`；不要再发送同义文字。"
            ),
        ),
        AgentSkill(
            name="search_context_tools",
            description="联网搜索、历史聊天检索和数学计算。",
            prompt=(
                "搜索、上下文和计算工具规则：\n"
                "- 外部知识、缩写、术语、最新信息、新闻、天气等实时事实：调用 `search_web`。\n"
                f"- {context_name}：需要补充过去聊天记录、群内旧话题或用户提到“之前/上次/以前”时调用 `search_history_context`。\n"
                "- RAG 检索中禁止相对时间词：昨天、前天、本周、上周、这个月、上个月、最近等；使用明确日期时间或关键词检索。\n"
                "- 精确数学计算：调用 `calculate_expression`，不要心算复杂表达式。"
            ),
        ),
        AgentSkill(
            name="schedule_tools",
            description="安排延迟提醒、转告、固定消息或到点后执行复杂 agent 任务。",
            prompt=(
                "定时工具规则：\n"
                "- 用户要求几分钟/几小时后提醒、转告或发送固定消息时：调用 `schedule_message`。\n"
                "- 用户要求到点后查询最新信息、联网搜索、选择表情包或根据当时情况处理时：调用 `schedule_agent_task`。\n"
                "- 如果任务只是提醒/转告，优先固定消息；如果任务需要未来环境判断，使用 agent task。\n"
                "- 安排成功后简短告知用户，并调用 `finish`。"
            ),
        ),
        AgentSkill(
            name="profile_memory_tools",
            description="更新用户印象、好感度，以及生成年度报告/个人总结/成分分析。",
            prompt=(
                "用户画像和年度报告工具规则：\n"
                "- 用户情绪或你们的关系变化明显时，调用 `update_user_impression` 更新好感度和标签；普通闲聊不要频繁更新。\n"
                "- `score_change` 表示好感变化，正数增加、负数降低；`reason` 简短说明触发原因。\n"
                "- 标签用简短稳定的人设或印象词，不要加入一次性事件。\n"
                "- 若用户提到“年度报告 / 个人总结 / 成分分析”，先调用 `generate_and_send_annual_report` 获取素材。\n"
                "- 年度报告工具返回素材后，由你根据素材生成完整报告，并调用 `reply_user` 发送；不要直接结束，也不要重复调用年度报告工具。"
            ),
        ),
    ]
    if reaction_tool_instruction.strip():
        skills.append(
            AgentSkill(
                name="reaction_tools",
                description="给消息点 reaction，用表情轻量表达态度。",
                prompt=reaction_tool_instruction.strip(),
            )
        )
    if has_admin_permission and mute_tool_instruction.strip():
        skills.append(
            AgentSkill(
                name="moderation_tools",
                description="群管理禁言规则，仅在 bot 有管理员/群主权限时可用。",
                prompt=mute_tool_instruction.strip(),
            )
        )
    return skills


def get_image_data_uri(file_name: str) -> str | None:
    return _get_image_data_uri(file_name, pic_dir=pic_dir)


def _parse_msg_meta(content: str) -> tuple[str | None, str | None, str]:
    return parse_msg_meta(content)


def _image_file_name_from_history(msg: ChatHistorySchema) -> str:
    return image_file_name_from_history(msg)


def _is_image_history(msg: ChatHistorySchema) -> bool:
    return is_image_history(msg)


async def _load_replied_message_histories(
    db_session: AsyncSession,
    session_id: str,
    reply_to_id: str | None,
) -> list[ChatHistorySchema]:
    return await load_replied_message_histories(db_session, session_id, reply_to_id)


def format_chat_history(
    history: list[ChatHistorySchema],
    max_inline_images: int = 3,
    user_roles: dict[str, str] | None = None,
    extra_inline_images: list[ChatHistorySchema] | None = None,
) -> list[BaseMessage]:
    return _format_chat_history(
        history,
        pic_dir=pic_dir,
        bot_name=plugin_config.bot_name,
        max_inline_images=max_inline_images,
        user_roles=user_roles,
        extra_inline_images=extra_inline_images,
    )


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

    has_admin_permission = False
    if not is_private and interface and bot_id:
        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
            for member in members:
                if str(member.id) == str(bot_id):
                    bot_role = getattr(getattr(member, "role", None), "name", None)
                    if bot_role in {"owner", "admin"}:
                        has_admin_permission = True
                        logger.info(
                            f"Bot在群{session_id}中拥有{bot_role}权限，已启用禁言功能"
                        )
                    else:
                        logger.info(f"Bot在群{session_id}中是普通成员，未启用禁言功能")
                    break
        except Exception as e:
            logger.warning(f"检查bot权限失败: {e}")

    permission_status, mute_tool_instruction = build_permission_prompt_parts(
        has_admin_permission
    )
    reaction_tool_instruction = build_reaction_tool_instruction(
        is_onebot_context(bot, event)
    )
    private_message_enabled = (
        plugin_config.proactive_private_message
        and not is_private
        and interface is not None
    )
    recall_message_enabled = not is_private and bot is not None and event is not None
    prompt_result = build_chat_system_prompt(
        bot_name=plugin_config.bot_name,
        is_private=is_private,
        relation_context=relation_context,
        group_context=group_context,
        recent_relations_context=recent_relations_context,
        permission_status=permission_status,
        mute_tool_instruction=mute_tool_instruction,
        reaction_tool_instruction=reaction_tool_instruction,
    )
    system_prompt = prompt_result.system_prompt
    if private_message_enabled:
        system_prompt += """
【主动私聊】
- 可使用 `send_private_message` 主动私聊当前群内成员。
- 只在不适合公开说、涉及隐私、避免让对方尴尬、或用户明确希望私下继续时使用。
- 不要群发，不要骚扰，不要用私聊绕过对方拒绝，也不要发送营销/诱导内容。
- 能在群内自然说清的内容优先用 `reply_user`。
"""
    if recall_message_enabled:
        if has_admin_permission:
            system_prompt += """
【消息撤回】
- 你当前拥有管理员/群主权限，可使用 `recall_message` 撤回当前群历史中的消息，包括他人消息。
- 只在有明确原因时撤回，例如违规、刷屏、隐私泄露、用户明确要求撤回、误发敏感内容。
- 不要因为观点不同、普通玩笑或轻微跑题撤回他人消息。
- 对 bot 自己误发、重复发、格式错乱的消息，也可以用 `recall_message` 撤回。
- target_msg_id 必须来自聊天历史里的 `id: xxx`，不要编造消息 ID。
"""
        else:
            system_prompt += """
【消息撤回】
- 你当前没有管理员/群主权限，但可使用 `recall_message` 撤回 bot 自己发送且 5 分钟内的消息。
- 只能用于 bot 自己误发、重复发、格式错乱、发错对象、内容不合适等情况。
- 不能撤回用户或其他成员的消息；遇到他人违规消息时只能提醒、吐槽或请求管理员处理。
- target_msg_id 必须来自聊天历史里 bot 自己消息的 `id: xxx`，不要编造消息 ID。
"""
    model = get_chat_model()
    report_tool = create_report_tool(
        db_session,
        session_id,
        request_id,
        user_id,
        user_name,
        model,
        bot_name=plugin_config.bot_name,
        stop_words=stop_words,
    )

    send_target = Target(
        id=session_id,
        private=is_private,
        self_id=bot_id,
    )

    search_meme_tool = create_search_meme_tool(db_session, session_id, request_id)
    send_meme_tool = create_send_meme_tool(
        db_session,
        session_id,
        request_id,
        send_target=send_target,
        pic_dir=pic_dir,
        bot_name=plugin_config.bot_name,
    )
    relation_tool = create_relation_tool(
        db_session,
        session_id,
        request_id,
        user_id,
        user_name,
        bot_name=plugin_config.bot_name,
        up_pic=up_pic,
        down_pic=down_pic,
    )
    similar_meme_tool = create_similar_meme_tool(
        db_session, session_id, request_id, user_id, pic_dir=pic_dir
    )
    mute_tool = create_mute_tool(
        db_session,
        session_id,
        request_id,
        interface,
        bot_id,
        bot_name=plugin_config.bot_name,
    )
    schedule_tool = create_schedule_message_tool(
        session_id,
        request_id,
        is_private=is_private,
        bot_id=bot_id,
        bot_name=plugin_config.bot_name,
    )
    schedule_agent_tool = create_schedule_agent_task_tool(
        session_id,
        request_id,
        is_private=is_private,
        bot_id=bot_id,
        run_agent_task=_run_scheduled_agent_task,
    )
    reaction_tool = create_reaction_tool(
        db_session, session_id, request_id, plugin_config.bot_name, bot, event
    )
    recall_tool = (
        create_recall_message_tool(
            db_session,
            session_id,
            request_id,
            bot_name=plugin_config.bot_name,
            has_admin_permission=has_admin_permission,
            bot=bot,
            event=event,
        )
        if recall_message_enabled
        else None
    )
    private_message_tool = (
        create_private_message_tool(
            db_session,
            session_id,
            request_id,
            interface,
            bot_id=bot_id,
            bot_name=plugin_config.bot_name,
        )
        if private_message_enabled
        else None
    )
    custom_agent_tools, custom_tool_instructions, custom_agent_skills = await build_registered_agent_extensions(
        AgentToolContext(
            db_session=db_session,
            session_id=session_id,
            request_id=request_id,
            user_id=str(user_id) if user_id is not None else None,
            user_name=user_name,
            interface=interface,
            send_target=send_target,
            is_private=is_private,
            bot_id=bot_id,
            bot=bot,
            event=event,
            model=model,
        )
    )
    if custom_tool_instructions:
        system_prompt += "\n【自定义工具】\n" + "\n".join(custom_tool_instructions) + "\n"
    agent_skills = [
        *_build_builtin_agent_skills(
            is_private=is_private,
            has_admin_permission=has_admin_permission,
            reaction_tool_instruction=reaction_tool_instruction,
            mute_tool_instruction=mute_tool_instruction,
        ),
        *custom_agent_skills,
    ]
    skill_index = build_agent_skill_index(agent_skills)
    if skill_index:
        system_prompt += "\n" + skill_index
    skill_loader_tool = create_agent_skill_loader_tool(
        agent_skills,
        AgentToolContext(
            db_session=db_session,
            session_id=session_id,
            request_id=request_id,
            user_id=str(user_id) if user_id is not None else None,
            user_name=user_name,
            interface=interface,
            send_target=send_target,
            is_private=is_private,
            bot_id=bot_id,
            bot=bot,
            event=event,
            model=model,
        ),
    )

    agent_tools = [
        search_web,
        search_history_context,
        create_reply_tool(
            db_session,
            session_id,
            request_id,
            interface,
            send_target=send_target,
            bot_name=plugin_config.bot_name,
            parse_msg_meta=_parse_msg_meta,
        ),
        search_meme_tool,
        similar_meme_tool,
        send_meme_tool,
        calculate_expression,
        relation_tool,
        report_tool,
        mute_tool,
        *([recall_tool] if recall_tool is not None else []),
        schedule_tool,
        schedule_agent_tool,
        *([private_message_tool] if private_message_tool is not None else []),
        *([skill_loader_tool] if skill_loader_tool is not None else []),
        *custom_agent_tools,
        finish,
    ]
    if is_onebot_context(bot, event):
        agent_tools.insert(3, reaction_tool)

    stable_system_prompt = system_prompt
    kept_dynamic_context_parts: list[str] = []
    for context_part in prompt_result.dynamic_context_parts:
        if not context_part or not context_part.strip():
            continue
        stable_system_prompt = stable_system_prompt.replace(context_part, "", 1)
        kept_dynamic_context_parts.append(context_part.strip())

    system_messages = build_system_messages(
        stable_system_prompt,
        "\n\n".join(kept_dynamic_context_parts),
        use_cache_control=plugin_config.chat_api_format == "anthropic",
    )

    graph = build_chat_graph(model, agent_tools, system_messages)
    return graph, agent_tools


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
        chat_history_messages, appended_history, reused_thread = build_append_only_history(
            session_id,
            history,
            format_history=format_chat_history,
            user_roles=role_map,
            extra_inline_images=replied_extra,
        )
        input_max_msg_id = max((msg.msg_id for msg in history), default=0)
        active_thread = get_active_thread(session_id)
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
普通图片/表情包通常只是群聊氛围，不要主动解读、复述或围绕它展开回复。
只有当前用户明确询问图片内容、回复/引用图片、要求找图/发图，或上下文确实在讨论这张图时，才重点结合图片内容回答。
群友在质疑、反问、跟风或复读时，不要优先质疑这种行为本身；可以自然接一句、复读关键词、跟队形，或者保持沉默。
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
        if not is_private and should_include_avatar_context(history):
            avatar_context_messages = await build_avatar_context_messages(
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
            "llm_cached_tokens": 0,
        }

        # 4. 调用 Agent
        from langchain_community.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            graph_result = await graph.ainvoke(invoke_state, config={"callbacks": [cb]})
        logger.info(
            f"[Token用量] 输入={cb.prompt_tokens} 输出={cb.completion_tokens} "
            f"总计={cb.total_tokens} 费用≈${cb.total_cost:.4f}"
        )
        cached_tokens = max(
            extract_cached_tokens(cb),
            int(graph_result.get("llm_cached_tokens", 0) or 0),
        )
        estimated_cost = estimate_cost(
            prompt_tokens=int(cb.prompt_tokens or 0),
            completion_tokens=int(cb.completion_tokens or 0),
            cached_tokens=cached_tokens,
            callback_cost=float(cb.total_cost or 0.0),
            input_cost_per_million=plugin_config.chat_input_cost_per_million,
            output_cost_per_million=plugin_config.chat_output_cost_per_million,
            cached_input_cost_per_million=plugin_config.chat_cached_input_cost_per_million,
            long_context_threshold_tokens=plugin_config.chat_long_context_threshold_tokens,
            long_input_cost_per_million=plugin_config.chat_long_input_cost_per_million,
            long_output_cost_per_million=plugin_config.chat_long_output_cost_per_million,
            long_cached_input_cost_per_million=plugin_config.chat_long_cached_input_cost_per_million,
        )
        await record_token_usage(
            db_session,
            session_id=session_id,
            session_type="private" if is_private else "group",
            user_id=user_id,
            user_name=user_name,
            model=plugin_config.chat_model or plugin_config.base_model,
            request_id=request_id,
            prompt_tokens=int(cb.prompt_tokens or 0),
            completion_tokens=int(cb.completion_tokens or 0),
            cached_tokens=cached_tokens,
            total_tokens=int(cb.total_tokens or 0),
            estimated_cost=estimated_cost,
        )

        # 5. 统一提交 db_session（reply_user / send_meme_image 只 add 不 commit）
        await update_active_thread(
            db_session,
            session_id,
            chat_history_messages,
            input_max_msg_id,
            format_history=format_chat_history,
        )

        await _finish_db_operation(db_session.commit())

        return ResponseMessage(need_reply=False, text=None)

    except Exception as e:
        err_str = str(e)
        if "data_inspection_failed" in err_str or (
            "Error code: 400" in err_str and "inappropriate" in err_str
        ):
            logger.warning("消息内容触发阿里云内容审核，本轮跳过回复")
            await _safe_rollback(db_session)
            return ResponseMessage(need_reply=False, text=None)
        logger.exception("Agent 决策过程发生异常")
        await _safe_rollback(db_session)
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
