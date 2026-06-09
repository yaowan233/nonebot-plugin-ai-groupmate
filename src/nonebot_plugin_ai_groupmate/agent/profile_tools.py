import re
import json
import random
import datetime
import traceback
import collections
from typing import Any

import jieba
from sqlalchemy import Select, desc, func, extract
from nonebot.log import logger
from langchain.tools import tool
from nonebot_plugin_orm import get_session
from nonebot_plugin_alconna import UniMessage

from ..model import ChatHistory, UserRelation
from ..reply_guard import is_request_active


def create_report_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str,
    user_name: str | None,
    llm_client: Any,
    *,
    bot_name: str,
    stop_words: list[str],
):
    """
    创建年度报告工具（限制在当前群聊 session_id 范围内）
    """
    _ = llm_client

    @tool("generate_and_send_annual_report")
    async def generate_and_send_annual_report() -> str:
        """
        生成并发送当前群聊的年度报告。
        包含：个人在本群的统计、性格分析、全群排行榜以及Bot的好感度回顾。
        """
        if not user_id:
            return "当前没有可用于生成报告的用户信息。"

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消发送。"

        try:
            logger.info(f"开始生成用户 {user_name} 在群 {session_id} 的年度报告...")
            now = datetime.datetime.now()
            current_year = now.year

            stmt = Select(ChatHistory).where(
                ChatHistory.user_id == user_id,
                ChatHistory.session_id == session_id,
                extract("year", ChatHistory.created_at) == current_year,
            )
            all_msgs = (await db_session.execute(stmt)).scalars().all()

            if not all_msgs:
                return "用户本群今年暂无可用于年度报告的数据。请调用 reply_user 简短告知用户生成不了报告。"

            text_msgs = [
                m.content for m in all_msgs if m.content_type == "text" and m.content
            ]
            total_count = len(all_msgs)

            samples = (
                random.sample(text_msgs, min(len(text_msgs), 30)) if text_msgs else []
            )
            longest_msg = max(text_msgs, key=len) if text_msgs else "无"
            if len(longest_msg) > 60:
                longest_msg = longest_msg[:60] + "..."

            active_hour_desc = "潜水员"
            if all_msgs:
                hours = [m.created_at.hour for m in all_msgs]
                top_hour = collections.Counter(hours).most_common(1)[0][0]
                active_hour_desc = f"{top_hour}点"

            async def get_rank_str(content_type=None, hour_limit=None):
                stmt = Select(
                    ChatHistory.user_id, func.count(ChatHistory.msg_id).label("c")
                ).where(
                    extract("year", ChatHistory.created_at) == current_year,
                    ChatHistory.session_id == session_id,
                )

                if content_type:
                    stmt = stmt.where(ChatHistory.content_type == content_type)
                if hour_limit:
                    stmt = stmt.where(
                        extract("hour", ChatHistory.created_at) < hour_limit
                    )

                stmt = stmt.group_by(ChatHistory.user_id).order_by(desc("c")).limit(3)
                rows = (await db_session.execute(stmt)).all()

                if not rows:
                    return "虚位以待"

                rank_items = []
                for uid, count in rows:
                    name_stmt = (
                        Select(ChatHistory.user_name)
                        .where(ChatHistory.user_id == uid)
                        .order_by(desc(ChatHistory.created_at))
                        .limit(1)
                    )
                    latest_name = (await db_session.execute(name_stmt)).scalar()
                    display_name = latest_name if latest_name else f"用户{uid}"
                    rank_items.append(f"{display_name}({count})")
                return ", ".join(rank_items)

            rank_talk = await get_rank_str()
            rank_img = await get_rank_str(content_type="image")
            rank_night = await get_rank_str(hour_limit=5)

            stmt_text = (
                Select(ChatHistory.content)
                .where(
                    ChatHistory.session_id == session_id,
                    extract("year", ChatHistory.created_at) == current_year,
                    ChatHistory.content_type == "text",
                    ChatHistory.user_id == user_id,
                )
                .order_by(desc(ChatHistory.created_at))
            )

            rows = (await db_session.execute(stmt_text)).all()
            sample_text = "\n".join([r[0] for r in rows if r[0]])

            clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", sample_text)
            words = jieba.lcut(clean_text)
            filtered = [w for w in words if len(w) > 1 and w not in stop_words]
            hot_words_str = "、".join(
                [x[0] for x in collections.Counter(filtered).most_common(8)]
            )

            relation_stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
            relation = (await db_session.execute(relation_stmt)).scalar_one_or_none()

            favorability = 0
            impression_tags = []
            if relation:
                favorability = relation.favorability
                impression_tags = relation.tags if relation.tags else []

            relation_desc = f"好感度: {favorability} (满分100), 印象标签: {', '.join(impression_tags)}"
            samples_text = "\n".join(samples)
            return f"""【年度报告素材】
请根据以下素材生成完整年度报告，并调用 `reply_user` 发送给用户；不要直接结束，也不要再调用年度报告工具。

【写作要求】
1. 不要使用 Markdown 标题、粗体或列表符号。
2. 可以使用 Emoji 和纯文本分隔线排版。
3. 语气像群友，不要像正式报告。
4. 根据好感度调整语气：>60 亲密宠溺，<0 傲娇毒舌，0-60 友善调侃。
5. 必须包含：标题行、基础数据、关系回顾、年度热词、群内风云榜、成分分析、{bot_name}寄语。
6. 成分分析要重点参考发言样本，写得具体一点。

【用户数据】
用户名: {user_name}
年份: {current_year}
累计发言: {total_count}
活跃时间: {active_hour_desc}
最长发言片段: {longest_msg}
年度热词: {hot_words_str}

【{bot_name} 与用户的关系】
{relation_desc}

【全群排行榜参考】
龙王榜: {rank_talk}
斗图榜: {rank_img}
熬夜榜: {rank_night}

【用户发言样本】
{samples_text}
"""

        except Exception as e:
            logger.error(f"收集年度报告素材失败: {e}")
            print(traceback.format_exc())
            return f"收集年度报告素材出错: {e}"

    return generate_and_send_annual_report


def create_relation_tool(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id: str,
    user_name: str | None,
    *,
    bot_name: str,
    up_pic: bytes,
    down_pic: bytes,
):
    """
    创建绑定了特定用户的关系管理工具 (支持增删 Tag)
    """

    @tool("update_user_impression")
    async def update_user_impression(
        score_change: int,
        reason: str,
        add_tags: list[str] | str | None = None,
        remove_tags: list[str] | str | None = None,
    ) -> str:
        """
        更新对当前对话用户的好感度和印象标签。
        当用户的言行让你产生情绪波动，或者你发现旧的印象不再准确时调用。
        """
        if not user_id:
            return "当前没有可更新画像的用户信息。"

        if request_id is not None and not await is_request_active(
            session_id, request_id
        ):
            return "请求已过期，已取消更新。"

        def normalize_tags(value: list[str] | str | None) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return []
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return [tag.strip() for tag in value.split(",") if tag.strip()]
                if isinstance(parsed, list):
                    return [str(tag).strip() for tag in parsed if str(tag).strip()]
                if isinstance(parsed, str) and parsed.strip():
                    return [parsed.strip()]
                return []
            return [str(tag).strip() for tag in value if str(tag).strip()]

        add_tags = normalize_tags(add_tags)
        remove_tags = normalize_tags(remove_tags)

        async with get_session() as session:
            try:
                stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
                result = await session.execute(stmt)
                relation = result.scalar_one_or_none()

                if not relation:
                    relation = UserRelation(
                        user_id=user_id,
                        user_name=user_name or "",
                        favorability=0,
                        tags=[],
                    )
                    session.add(relation)
                else:
                    await session.refresh(relation, attribute_names=["tags"])

                old_score = relation.favorability
                final_change = score_change

                if old_score < -60 and score_change > 0:
                    final_change = int(score_change * 1.5) + 5
                    logger.info(
                        f"触发救赎机制：原始分 {score_change} -> 修正分 {final_change}"
                    )
                elif old_score > 80 and score_change < 0:
                    final_change = int(score_change * 1.2) - 2
                    logger.info(
                        f"触发破防机制：原始分 {score_change} -> 修正分 {final_change}"
                    )

                relation.favorability += final_change
                relation.favorability = max(-100, min(100, relation.favorability))

                current_tags = list(relation.tags) if relation.tags else []
                if remove_tags:
                    current_tags = [
                        tag for tag in current_tags if tag not in remove_tags
                    ]
                if add_tags:
                    for tag in add_tags:
                        if tag not in current_tags:
                            current_tags.append(tag)
                if len(current_tags) > 8:
                    current_tags = current_tags[-8:]

                relation.tags = current_tags
                relation.user_name = user_name or ""
                favorability = relation.favorability

                if request_id is not None and not await is_request_active(
                    session_id, request_id
                ):
                    await session.rollback()
                    return "请求已过期，已取消更新。"

                await session.commit()

                boundaries = (-70, -40, -15, 5, 25, 50, 70, 90)
                tier_names = (
                    "死敌/拉黑",
                    "厌恶/仇视",
                    "冷淡/防备",
                    "陌生/普通",
                    "有点熟",
                    "朋友/熟人",
                    "好朋友",
                    "亲密/死党",
                    "最喜欢的人",
                )

                def tier(score: int) -> int:
                    for i, boundary in enumerate(boundaries):
                        if score < boundary:
                            return i
                    return len(boundaries)

                old_tier = tier(old_score)
                new_tier = tier(favorability)
                if old_tier != new_tier:
                    try:
                        if new_tier > old_tier:
                            tip = f"好感度提升！现在的关系是：{tier_names[new_tier]}"
                            res = await UniMessage.image(raw=up_pic).text(tip).send()
                        else:
                            tip = f"好感度下降…现在的关系是：{tier_names[new_tier]}"
                            res = await UniMessage.image(raw=down_pic).text(tip).send()
                        msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
                        chat_history = ChatHistory(
                            session_id=session_id,
                            user_id=bot_name,
                            content_type="bot",
                            content=f"id: {msg_id}\n{tip}",
                            user_name=bot_name,
                        )
                        session.add(chat_history)
                    except Exception as send_err:
                        logger.warning(f"发送好感度图片失败: {send_err}")

                tag_msg = ""
                if add_tags or remove_tags:
                    tag_msg = f"，标签变更(新增:{add_tags}, 移除:{remove_tags})"

                log_msg = (
                    f"好感度 {old_score}->{favorability}{tag_msg} (原因: {reason})"
                )
                logger.info(f"用户[{user_name}]画像更新: {log_msg}")

                return (
                    f"画像已更新。当前好感度: {favorability}，当前标签: {current_tags}"
                )

            except Exception as e:
                logger.error(f"关系更新失败: {e}")
                print(traceback.format_exc())
                return f"数据库错误: {str(e)}"
        return None

    return update_user_impression
