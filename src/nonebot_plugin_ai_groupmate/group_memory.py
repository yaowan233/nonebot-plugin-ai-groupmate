import asyncio
from functools import lru_cache

from sqlalchemy import Select, func
from nonebot.log import logger
from langchain_core.messages import HumanMessage, SystemMessage

from .model import ChatHistory, GroupMemory
from .config import create_chat_openai
from .runtime_config import get_runtime_config

MAX_SUMMARY_MESSAGES = 200
_update_locks: dict[str, asyncio.Lock] = {}


def _get_update_lock(session_id: str) -> asyncio.Lock:
    lock = _update_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _update_locks[session_id] = lock
    return lock


@lru_cache
def get_summary_model():
    return create_chat_openai(get_runtime_config(), "summary")


async def _call_summary_model(existing_summary: str, chat_text: str) -> str | None:
    """调用 LLM 更新群体认知档案。

    若触发内容违规（data_inspection_failed），会对聊天记录做二分截断后最多重试 3 次。
    """
    system = """你是一个群文化分析师。你的任务是维护一份关于QQ群的认知档案。
档案包含：群内常见话题、活跃成员特征、内部梗/黑话、群文化氛围。

【核心原则】标注为[BOT]的消息是机器人自身的回复。档案只记录真实用户的群文化，绝对不记录机器人的行为模式或回复话术。

规则：
1. 只能基于提供的聊天记录总结，不要凭空发明内容
2. 保留档案中仍然有效的内容，用新聊天补充或修正旧内容
3. 如果某个内容长期（超过30天）无聊天印证，可删除
4. 输出完整更新后的档案，不超过500字，不要输出任何其他内容
5. 【必须执行】如果现有档案中包含BOT的回复话术（如"别得寸进尺"、"？你又来"、"行了行了"等BOT说的话），必须将其删除。内部梗/黑话只能来自真实用户的发言，不能来自BOT。
6. 【必须执行】"活跃成员特征"部分只描述真实用户，不要把BOT列为活跃成员或描述BOT的行为模式。
7. 【必须执行】"群文化氛围"只描述用户之间的互动氛围，不要描述用户与BOT之间的互动循环。"""
    history_intro = (
        "（无，这是首次建档）" if not existing_summary.strip() else existing_summary
    )

    lines = chat_text.splitlines()
    max_retries = 3

    for attempt in range(max_retries + 1):
        current_text = "\n".join(lines)
        if not current_text.strip():
            logger.warning("档案更新：聊天记录经截断后已为空，放弃本次更新")
            return None

        user_msg = f"【现有档案】\n{history_intro}\n\n【最新聊天记录】\n{current_text}\n\n请输出更新后的档案："
        try:
            resp = await get_summary_model().ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=user_msg),
                ]
            )
            if not isinstance(resp.content, str) or not resp.content.strip():
                return None
            if attempt > 0:
                logger.info(
                    f"档案更新：截断后第 {attempt} 次重试成功（剩余 {len(lines)} 条消息）"
                )
            return resp.content.strip()
        except Exception as e:
            err_str = str(e)
            if "data_inspection_failed" in err_str or (
                "Error code: 400" in err_str and "inappropriate" in err_str
            ):
                if attempt < max_retries:
                    lines = lines[: max(1, len(lines) // 2)]
                    logger.warning(
                        f"档案更新：内容违规，截断至 {len(lines)} 条消息后重试（第 {attempt + 1}/{max_retries} 次）"
                    )
                else:
                    logger.warning(
                        f"档案更新：内容违规，已重试 {max_retries} 次仍失败，放弃本次更新"
                    )
                    return None
            else:
                logger.error(f"档案更新 LLM 调用失败: {e}")
                return None

    return None


def _format_message(message: ChatHistory) -> str:
    content = message.content
    if message.content_type == "bot":
        # 去掉 "id: XXXXX\n" 前缀，只保留实际回复内容。
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
    bot_marker = "[BOT] " if message.content_type == "bot" else ""
    return (
        f"[{message.created_at.strftime('%m-%d %H:%M')}] "
        f"{bot_marker}{message.user_name}: {content[:100]}"
    )


def _filter_bot_descriptions(summary: str, bot_name: str) -> str:
    filtered_lines: list[str] = []
    for line in summary.splitlines():
        stripped = line.strip().lstrip("-•* ").strip()
        if bot_name and (
            f"{bot_name}为" in stripped
            or f"{bot_name}是" in stripped
            or f"{bot_name}主导" in stripped
            or f"{bot_name}维持" in stripped
            or f"{bot_name}以" in stripped
        ):
            logger.info(f"档案过滤：移除BOT行为描述行: {stripped[:50]}")
            continue
        if "标准回应" in stripped or "回应模板" in stripped:
            logger.info(f"档案过滤：移除BOT模板描述行: {stripped[:50]}")
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


async def update_group_memory(
    db_session,
    session_id: str,
    *,
    bot_name: str,
) -> str:
    """按 agent 的决定更新一个群的认知档案，并返回适合工具消费的状态。"""
    async with _get_update_lock(session_id):
        stmt = Select(GroupMemory).where(GroupMemory.session_id == session_id)
        record = (await db_session.execute(stmt)).scalar_one_or_none()

        total_count = (
            await db_session.execute(
                Select(func.count(ChatHistory.msg_id)).where(
                    ChatHistory.session_id == session_id
                )
            )
        ).scalar_one()

        cutoff = record.updated_at if record else None
        messages_stmt = Select(ChatHistory).where(
            ChatHistory.session_id == session_id,
            ChatHistory.content_type.in_(["text", "bot"]),
        )
        if cutoff is not None:
            messages_stmt = messages_stmt.where(ChatHistory.created_at > cutoff)

        # 档案长时间未维护时优先保留最新内容，避免旧的 200 条消息挤掉近期变化。
        recent_messages = list(
            (
                (
                    await db_session.execute(
                        messages_stmt.order_by(ChatHistory.created_at.desc()).limit(
                            MAX_SUMMARY_MESSAGES
                        )
                    )
                )
                .scalars()
                .all()
            )[::-1]
        )
        if not recent_messages:
            return "没有新的文本聊天可归档，本次无需更新。"

        chat_text = "\n".join(_format_message(message) for message in recent_messages)
        existing_summary = record.summary if record else ""

        # 摘要生成是较慢的外部 I/O；先释放只读事务，生成完成后再重新读取目标行。
        await db_session.commit()
        new_summary = await _call_summary_model(existing_summary, chat_text)
        if not new_summary:
            return "摘要模型未生成有效档案，本次更新已跳过。"

        new_summary = _filter_bot_descriptions(new_summary, bot_name)
        if not new_summary:
            return "新档案经内容清理后为空，本次更新已跳过。"

        record = (await db_session.execute(stmt)).scalar_one_or_none()
        if record is None:
            record = GroupMemory(
                session_id=session_id,
                summary=new_summary,
                msg_count_at_last_update=total_count,
            )
            db_session.add(record)
        else:
            record.summary = new_summary
            record.msg_count_at_last_update = total_count

        await db_session.commit()
        logger.info(f"群 {session_id} 档案由 agent 自主更新成功（{len(new_summary)} 字）")
        return "群档案已在后台更新；如果用户没有明确询问档案内容，不要播报此结果。"
