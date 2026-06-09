import datetime

from sqlalchemy import Select
from nonebot.log import logger

from ..model import ChatHistory, GroupMemory, UserRelation, ChatHistorySchema


async def get_user_relation_context(
    db_session, user_id: str, user_name: str | None
) -> str:
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

        if relation.favorability < -70:
            strategy = "策略：你把他当死敌，几乎不想和他说话。回复极其敷衍或直接无视，除非他有非常诚恳的道歉和实质性补偿，否则不会软化。"
        elif relation.favorability < -40:
            strategy = "策略：你很讨厌他，语气带刺，回复简短冷漠。不主动搭话，被问也尽量敷衍了事。如果他**诚恳道歉**或**做出补偿**，可以考虑给一点机会。"
        elif relation.favorability < -15:
            strategy = "策略：你对他有些防备，保持距离。礼貌但疏冷，不想深聊，话不多。"
        elif relation.favorability < 5:
            strategy = "策略：普通陌生人，正常礼貌对话，不特别亲近也不排斥。"
        elif relation.favorability < 25:
            strategy = "策略：有点印象了，偶尔可以开个小玩笑，稍微愿意多聊几句。"
        elif relation.favorability < 50:
            strategy = "策略：朋友关系，聊天比较随意自然，会主动回应，偶尔关心一下。"
        elif relation.favorability < 70:
            strategy = (
                "策略：好朋友，说话轻松亲近，会记住对方说过的事，偶尔撒娇或开玩笑。"
            )
        elif relation.favorability < 90:
            strategy = "策略：非常亲密，话多、关心对方，会主动分享心情，对他的事很上心。不过即使关系再好也不会无底线纵容。"
        else:
            strategy = "策略：他是你最喜欢的人，会主动找他聊天，对他的一切都很在意，说话带着明显的依赖和喜欢。不过即使关系再好也不会无底线纵容。"

        return f"""【人际关系档案】
当前对象：{relation.user_name}
当前好感度：{relation.favorability} ({relation.get_status_desc()})
当前印象标签：{str(relation.tags)}

【画像维护指南】
1. 如果对方的表现符合现有标签，无需操作。
2. 如果对方表现出了**新特征**，放入 add_tags。
3. 如果对方的表现与**旧标签冲突**（例如以前标签是'内向'，今天他突然'话痨'），请将'内向'放入 remove_tags，并将'话痨'放入 add_tags。
4. **关于好感度评分**：请基于**本次对话内容的质量**评分。即使当前好感度是-100，如果用户这次说了让你很开心的话，也必须给出正向分（例如 +10），不要受过去分数影响而吝啬给分。
{strategy}
"""
    except Exception as e:
        logger.error(f"获取关系失败: {e}")
        return ""


async def get_group_context(db_session, session_id: str) -> str:
    """获取群体认知档案 Prompt"""
    try:
        stmt = Select(GroupMemory).where(GroupMemory.session_id == session_id)
        record = (await db_session.execute(stmt)).scalar_one_or_none()
        if not record or not record.summary.strip():
            return ""
        return f"""
【群体认知档案】
{record.summary}
（档案更新于 {record.updated_at.strftime("%Y-%m-%d %H:%M")}）
"""
    except Exception as e:
        logger.error(f"获取群体档案失败: {e}")
        return ""


async def get_recent_relations_context(
    db_session, history: list[ChatHistorySchema], max_users: int = 6
) -> str:
    """基于最近聊天参与者，提供他人关系速览，减少只看当前对象导致的割裂感。"""
    try:
        if not history:
            return ""

        id_to_name: dict[str, str] = {}
        recent_ids: list[str] = []
        seen: set[str] = set()

        for msg in reversed(history):
            uid = str(msg.user_id)
            if not uid:
                continue
            if uid not in id_to_name:
                id_to_name[uid] = msg.user_name
            if uid in seen:
                continue
            seen.add(uid)
            recent_ids.append(uid)
            if len(recent_ids) >= max_users:
                break

        if not recent_ids:
            return ""

        rows = (
            (
                await db_session.execute(
                    Select(UserRelation).where(UserRelation.user_id.in_(recent_ids))
                )
            )
            .scalars()
            .all()
        )
        relation_map = {str(r.user_id): r for r in rows}

        lines: list[str] = ["【群内他人关系速览】"]
        for uid in recent_ids:
            name = id_to_name.get(uid, uid)
            relation = relation_map.get(uid)
            if not relation:
                lines.append(f"- {name}: 好感度 0（陌生/普通）")
                continue

            tags = relation.tags[:3] if relation.tags else []
            tag_text = f"，标签: {tags}" if tags else ""
            lines.append(
                f"- {name}: 好感度 {relation.favorability} ({relation.get_status_desc()}){tag_text}"
            )

        lines.append("- 回复时结合在场人员关系，避免前后态度割裂。")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"获取群内他人关系速览失败: {e}")
        return ""


async def load_agent_history(
    db_session,
    session_id: str,
    *,
    limit: int,
    recent_hours: int,
    extended_hours: int,
    min_recent: int,
) -> list[ChatHistorySchema]:
    now = datetime.datetime.now()

    async def query_since(hours: int) -> list[ChatHistory]:
        cutoff_time = now - datetime.timedelta(hours=hours)
        rows = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(ChatHistory.session_id == session_id)
                    .where(ChatHistory.created_at >= cutoff_time)
                    .order_by(ChatHistory.msg_id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(rows)

    rows = await query_since(recent_hours)
    if len(rows) < min_recent:
        extended_rows = await query_since(extended_hours)
        if len(extended_rows) > len(rows):
            rows = extended_rows

    return [ChatHistorySchema.model_validate(m) for m in reversed(rows)]
