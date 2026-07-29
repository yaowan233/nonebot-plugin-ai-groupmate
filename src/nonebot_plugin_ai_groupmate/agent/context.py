import datetime

from sqlalchemy import Select
from nonebot.log import logger

from ..model import ChatHistory, GroupMemory, UserRelation, ChatHistorySchema


def _get_relation_response_strategy(favorability: int) -> str:
    """只用好感度调节亲疏感，不让历史关系降低正常回复质量。"""
    if favorability < -70:
        return (
            "策略：保持中性、简洁和清晰的边界，不主动亲近或延伸闲聊；"
            "对对方明确提出的正常请求仍要礼貌、公平并尽力完整回应。"
        )
    if favorability < -40:
        return (
            "策略：语气克制，少开玩笑，保持适当距离；"
            "对正常提问和任务仍要礼貌、认真回应。"
        )
    if favorability < -15:
        return (
            "策略：稍显克制，不主动表现亲密，但保持正常礼貌和可靠的回应质量。"
        )
    if favorability < 5:
        return "策略：普通陌生人，正常礼貌对话，不特别亲近也不排斥。"
    if favorability < 25:
        return "策略：有点印象了，偶尔可以开个小玩笑，稍微愿意多聊几句。"
    if favorability < 50:
        return "策略：朋友关系，聊天比较随意自然，会主动回应，偶尔关心一下。"
    if favorability < 70:
        return "策略：好朋友，说话轻松亲近，会记住对方说过的事，偶尔开玩笑。"
    if favorability < 90:
        return (
            "策略：非常亲近，愿意多聊、关心对方，对他的事比较上心；"
            "即使关系很好也要保持合理边界。"
        )
    return (
        "策略：这是你非常亲近和重视的人，会更主动地关心和回应；"
        "即使关系很好也要保持合理边界。"
    )


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

        strategy = _get_relation_response_strategy(relation.favorability)

        return f"""【人际关系档案】
当前对象：{relation.user_name}
当前好感度：{relation.favorability} ({relation.get_status_desc()})
画像维护专用历史标签（可能片面或过时，不得用于决定回复态度）：{str(relation.tags)}

【关系使用边界】
1. 当前消息的实际内容优先于历史好感度和标签。对方这次正常、友善或诚恳时，应按本次表现回应。
2. 好感度只影响语气亲疏、主动性和玩笑尺度，不得影响事实判断、正常请求的回答质量、工具使用或安全规则。
3. 不得因为负好感度或负面标签讽刺、羞辱、敌视、故意敷衍、无视对方，也不得要求对方道歉或补偿后才正常回应。
4. 标签不是确定事实，不得据此给用户定性；涉及具体事实时以当前消息和可靠证据为准。

【画像维护指南】
1. 只有多次表现稳定一致时才新增人格或印象标签，不要把一次性事件、短暂情绪或争执写成标签。
2. 如果当前表现与旧标签持续冲突，应移除或修正旧标签；不要为了维护旧印象而曲解当前消息。
3. **关于好感度评分**：只根据本次互动质量评分，不受过去分数和标签影响。即使当前好感度是 -100，本次互动良好也必须给出合理正向分。
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

        lines: list[str] = ["【群内他人关系速览（弱参考）】"]
        for uid in recent_ids:
            name = id_to_name.get(uid, uid)
            relation = relation_map.get(uid)
            if not relation:
                lines.append(f"- {name}: 陌生/普通")
                continue

            lines.append(f"- {name}: {relation.get_status_desc()}")

        lines.append(
            "- 仅用于调整亲疏感和主动性；当前发言优先，不得据此敌视、忽视或降低回应质量。"
        )
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
