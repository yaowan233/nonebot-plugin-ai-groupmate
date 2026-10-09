import json
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

    result = json.loads(
        await memory_tool.ainvoke({"reason": "形成了新的群内梗"})
    )

    assert calls == [("group-1", "小助手", "形成了新的群内梗", 45)]
    assert result["status"] == "succeeded"
    assert result["delivery_state"] == "completed"
    assert "无需等待" in result["message"]


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


@pytest.mark.asyncio
async def test_summary_input_excludes_bot_echoes_actions_and_transport_metadata(monkeypatch):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate import group_memory
    from nonebot_plugin_ai_groupmate.model import ChatHistory

    session_id = f"clean-group-memory-{uuid4()}"
    captured_chat: list[str] = []

    async def fake_summary(existing_summary: str, chat_text: str) -> str:
        captured_chat.append(chat_text)
        return "常见话题：音游"

    monkeypatch.setattr(group_memory, "_call_summary_model", fake_summary)
    records = [
        ("miyuki", "bot", "id: 700\n妈妈在这里", None),
        ("adapter-account", "text", "id: 700\n妈妈在这里", None),
        ("alice", "text", "id: 701\n回复id: 700\nalias_id: alternate-701\n一起打音游吧", None),
        ("miyuki", "bot", "id: system\n已执行禁言操作: 禁言30分钟", None),
        ("miyuki", "bot", "id: 702\n发送了网络图片，搜索描述: 水产\n看图状态: preview_provided\n图片来源: https://example.com/fish.png", 12),
        ("miyuki", "bot", "id: 703\n图片文件: transport.png", 13),
        ("miyuki", "bot", "id: 704\n图片文件: legacy-transport.png", None),
        ("alice", "text", "旧格式聊天第一行\n第二行", None),
    ]
    async with get_session() as session:
        for uid, kind, content, media in records:
            session.add(ChatHistory(session_id=session_id, user_id=uid, user_name=uid, content_type=kind, content=content, media_id=media))
        await session.commit()
        await group_memory.update_group_memory(session, session_id, bot_name="miyuki")

    assert len(captured_chat) == 1
    text = captured_chat[0]
    assert text.count("妈妈在这里") == 1
    assert "[BOT] miyuki: 妈妈在这里" in text
    assert "alice: 一起打音游吧" in text
    assert "旧格式聊天第一行" in text
    for metadata in ("id:", "alias_id:", "图片文件:", "transport.png", "已执行禁言", "发送了网络图片", "preview_provided"):
        assert metadata not in text


def test_summary_preserves_human_metadata_discussion_and_legacy_bot_first_line():
    from datetime import datetime

    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.group_memory import _format_message

    human = ChatHistory(
        session_id="group-1", user_id="alice", user_name="Alice", content_type="text",
        content="id: 101\n程序示例\nid: literal-code\n回复id: literal-reply\nalias_id: literal-alias\n图片文件: 这是我的字段名",
        created_at=datetime(2026, 10, 9), media_id=None,
    )
    formatted = _format_message(human)
    assert formatted is not None
    for literal in ("id: literal-code", "回复id: literal-reply", "alias_id: literal-alias", "图片文件: 这是我的字段名"):
        assert literal in formatted

    bot = ChatHistory(
        session_id="group-1", user_id="miyuki", user_name="miyuki", content_type="bot",
        content="旧格式第一行\n第二行", created_at=datetime(2026, 10, 9), media_id=None,
    )
    formatted_bot = _format_message(bot)
    assert formatted_bot is not None
    assert "旧格式第一行\n第二行" in formatted_bot
