import asyncio
from uuid import uuid4

import pytest


def test_group_memory_skill_is_only_available_in_groups():
    from nonebot_plugin_ai_groupmate.agent import _build_builtin_agent_skills

    group_skills = _build_builtin_agent_skills(
        is_private=False,
        has_admin_permission=False,
        mute_tool_instruction="",
        meme_similar_enabled=True,
    )
    private_skills = _build_builtin_agent_skills(
        is_private=True,
        has_admin_permission=False,
        mute_tool_instruction="",
        meme_similar_enabled=True,
    )

    assert "group_memory_tools" in {skill.name for skill in group_skills}
    assert "group_memory_tools" not in {skill.name for skill in private_skills}
    # 群聊表情工具默认可见，不再要求额外加载；私聊仍按需加载。
    assert "meme_tools" not in {skill.name for skill in group_skills}
    assert "meme_tools" in {skill.name for skill in private_skills}
    assert "reaction_tools" not in {skill.name for skill in group_skills}


@pytest.mark.asyncio
async def test_group_memory_tool_queues_agent_requested_update(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import group_memory_tools

    calls: list[tuple[str, str, str, float]] = []

    def fake_start(
        session_id: str,
        *,
        bot_name: str,
        reason: str,
        timeout_seconds: float,
    ) -> bool:
        calls.append((session_id, bot_name, reason, timeout_seconds))
        return True

    monkeypatch.setattr(group_memory_tools, "start_group_memory_update", fake_start)
    memory_tool = group_memory_tools.create_group_memory_tool(
        "group-1",
        None,
        bot_name="小助手",
        timeout_seconds=45,
    )

    result = await memory_tool.ainvoke({"reason": "形成了新的群内梗"})

    assert calls == [("group-1", "小助手", "形成了新的群内梗", 45)]
    assert "无需等待" in result


@pytest.mark.asyncio
async def test_group_memory_background_tasks_are_deduplicated(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import group_memory_tools

    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_run(*args, **kwargs) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(group_memory_tools, "_run_group_memory_update", fake_run)

    first_started = group_memory_tools.start_group_memory_update(
        "deduplicate-group",
        bot_name="小助手",
        reason="第一次",
        timeout_seconds=30,
    )
    await started.wait()
    second_started = group_memory_tools.start_group_memory_update(
        "deduplicate-group",
        bot_name="小助手",
        reason="第二次",
        timeout_seconds=30,
    )

    assert first_started is True
    assert second_started is False

    release.set()
    task = group_memory_tools._background_update_tasks["deduplicate-group"]
    await task
    await asyncio.sleep(0)
    assert "deduplicate-group" not in group_memory_tools._background_update_tasks


@pytest.mark.asyncio
async def test_agent_update_persists_filtered_group_memory(monkeypatch):
    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import group_memory
    from nonebot_plugin_ai_groupmate.model import ChatHistory, GroupMemory

    session_id = f"group-memory-{uuid4()}"
    captured_chat: list[str] = []

    async def fake_summary(existing_summary: str, chat_text: str) -> str:
        assert existing_summary == ""
        captured_chat.append(chat_text)
        return "常见话题：最近开始讨论音游\n- 小助手是群里的气氛维护者"

    monkeypatch.setattr(group_memory, "_call_summary_model", fake_summary)

    async with get_session() as session:
        session.add(
            ChatHistory(
                session_id=session_id,
                user_id="user-1",
                content_type="text",
                content="最近大家开始一起打音游",
                user_name="Alice",
                media_id=None,
            )
        )
        await session.commit()

        result = await group_memory.update_group_memory(
            session,
            session_id,
            bot_name="小助手",
        )
        record = (
            await session.execute(
                Select(GroupMemory).where(GroupMemory.session_id == session_id)
            )
        ).scalar_one()

    assert captured_chat
    assert "Alice: 最近大家开始一起打音游" in captured_chat[0]
    assert record.summary == "常见话题：最近开始讨论音游"
    assert record.msg_count_at_last_update == 1
    assert "后台更新" in result


def test_group_memory_has_no_scheduled_job():
    from nonebot_plugin_apscheduler import scheduler

    assert scheduler.get_job("update_group_memory") is None
