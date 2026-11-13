from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("private", [False, True])
async def test_files_and_web_images_are_discoverable_but_not_initially_visible(monkeypatch, private):
    from nonebot_plugin_ai_groupmate import agent
    from nonebot_plugin_ai_groupmate.agent.graph import _active_tools

    monkeypatch.setattr(agent.plugin_config, "agent_file_tools_enabled", True)
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    captured = {}

    def build(_model, tools, prompt, **kwargs):
        captured.update(tools=tools, prompt=prompt, **kwargs)
        return object()

    monkeypatch.setattr(agent, "build_chat_graph", build)
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), "group", None, "user", "User",
                                  history=[], is_private=private, group_members=[])
    base = {tool.name for tool in captured["base_tools"]}
    assert "load_agent_skill" in base
    assert "send_file" in base
    assert not base.intersection({"fetch_url", "read_file", "list_files", "search_web_images", "preview_web_images", "send_web_image"})
    file_names = {tool.name for tool in _active_tools(captured["base_tools"], captured["tools_by_skill"], ["file_tools"])}
    image_names = {tool.name for tool in _active_tools(captured["base_tools"], captured["tools_by_skill"], ["web_image_tools"])}
    assert {"fetch_url", "read_file", "list_files", "create_file", "send_file"} <= file_names
    assert {"search_web_images", "preview_web_images", "send_web_image"} <= image_names
    assert "fetch_url" not in image_names
    assert "search_web_images" not in file_names
    prompt = str(captured["prompt"])
    assert "file_tools" in prompt
    assert "web_image_tools" in prompt
    assert "【网页与文件】" not in prompt
    assert "【联网搜图与发送】" not in prompt


@pytest.mark.asyncio
async def test_generated_images_can_still_be_sent_without_loading_file_tools(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    monkeypatch.setattr(agent.plugin_config, "agent_file_tools_enabled", True)
    monkeypatch.setattr(agent.plugin_config, "image_generation_enabled", True)
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    captured = {}

    def build(_model, _tools, _prompt, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent, "build_chat_graph", build)
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), "group", None, "user", "User",
                                  history=[], group_members=[])
    base = {tool.name for tool in captured["base_tools"]}
    assert {"generate_image", "send_file"} <= base


@pytest.mark.asyncio
async def test_execution_skill_can_inspect_and_read_its_files(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    monkeypatch.setattr(agent.plugin_config, "agent_file_tools_enabled", True)
    monkeypatch.setattr(agent.plugin_config, "agent_execution_backend", "local")
    for name in ("get_user_relation_context", "get_group_context", "get_recent_relations_context"):
        monkeypatch.setattr(agent, name, AsyncMock(return_value=""))
    monkeypatch.setattr(agent, "build_registered_agent_extensions", AsyncMock(return_value=([], [], [])))
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    captured = {}

    def build(_model, _tools, _prompt, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent, "build_chat_graph", build)
    await agent.create_chat_graph(SimpleNamespace(commit=AsyncMock()), "group", None, "user", "User",
                                  history=[], group_members=[])
    execution = {tool.name for tool in captured["tools_by_skill"]["execution_tools"]}
    assert {"run_code", "list_files", "read_file"} <= execution


@pytest.mark.asyncio
async def test_lazy_web_image_search_preview_and_delivery(monkeypatch):
    from nonebot_plugin_alconna import Target, UniMessage
    from langchain_core.messages import AIMessage, HumanMessage

    from nonebot_plugin_ai_groupmate.agent import image_search_tools
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.common_tools import finish
    from nonebot_plugin_ai_groupmate.agent.custom_tools import AgentSkill, AgentToolContext, create_agent_skill_loader_tool

    search = AsyncMock()
    search.ainvoke.return_value = {"images": [{"url": "https://example.com/cat.png", "description": "小猫"}]}
    monkeypatch.setattr(image_search_tools, "TavilySearch", lambda **_: search)
    sender = AsyncMock(return_value=SimpleNamespace(msg_ids=[{"message_id": "42"}]))
    monkeypatch.setattr(UniMessage, "send", sender)
    session = SimpleNamespace(commit=AsyncMock(), add=lambda _: None)
    target = Target("group")
    image_tools = image_search_tools.create_web_image_tools(session, "group", None, tavily_api_key="test",
                                                             bot_name="miyuki", send_target=target)
    ctx = AgentToolContext(db_session=None, session_id="group", request_id=None, user_id="user", user_name="User",
                           interface=None, send_target=target, is_private=False)
    loader = create_agent_skill_loader_tool(
        [AgentSkill("web_image_tools", "联网找图", image_search_tools.WEB_IMAGE_TOOLS_PROMPT)], ctx,
    )
    assert loader is not None

    class Model:
        def __init__(self):
            self.step = 0
            self.bound = []

        def bind_tools(self, tools):
            self.bound.append({item.name for item in tools})
            return self

        async def ainvoke(self, _messages, **_kwargs):
            steps = [
                ("load_agent_skill", {"skill_name": "web_image_tools"}),
                ("search_web_images", {"query": "小猫"}),
                ("preview_web_images", {"image_ids": ["web-image-1"]}),
                ("send_web_image", {"image_id": "web-image-1", "allow_unverified": True, "text": "我还没能看图。"}),
            ]
            name, args = steps[self.step]
            self.step += 1
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{self.step}"}])

    model = Model()
    graph = build_chat_graph(model, [loader, finish, *image_tools], "按需找图",
                             base_tools=[loader, finish], tools_by_skill={"web_image_tools": image_tools})
    result = await graph.ainvoke({"messages": [HumanMessage(content="找一张小猫图片")], "session_id": "group", "request_id": None})
    assert "search_web_images" not in model.bound[0]
    assert "search_web_images" in model.bound[1]
    assert result["active_skills"] == ["web_image_tools"]
    sender.assert_awaited_once()
