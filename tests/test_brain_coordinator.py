from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_planner_can_only_use_configured_other_group():
    from nonebot_plugin_ai_groupmate.brain.coordinator import plan_request

    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content='{"kind":"ask","target_scene":"B","text":"郊游几点出发？"}',
    )))
    plan = await plan_request(model, request="去 B 群问郊游时间", source_scene="A", target_scenes=("A", "B"))
    assert plan.kind == "ask"
    assert plan.target_scene == "B"
    model.ainvoke.return_value.content = '{"kind":"ask","target_scene":"C","text":"几点？"}'
    with pytest.raises(ValueError, match="unconfigured"):
        await plan_request(model, request="去 B 群问郊游时间", source_scene="A", target_scenes=("A", "B"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("response", "expected"), [
    ('{"result":"answer"}', "answer"),
    ('{"result":"unrelated"}', "unrelated"),
    ('{"result":"invented"}', "uncertain"),
    ("invalid json", "uncertain"),
])
async def test_quote_does_not_by_itself_prove_an_answer(response, expected):
    from nonebot_plugin_ai_groupmate.brain.coordinator import classify_reply

    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=response)))
    assert await classify_reply(model, question="几点出发？", candidate="哈哈") == expected
