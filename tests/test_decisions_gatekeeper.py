import json
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import httpx
import pytest


@pytest.mark.asyncio
async def test_decisions_sends_policy_as_question_instructions(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import decisions
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    def respond(request):
        assert str(request.url) == "https://api.openai.com/v1/decisions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-luna"
        assert body["input"] == '{"latest_message":"a question"}'
        question = body["questions"][0]
        assert "保持语义承接" in question["instructions"]
        assert "只输出JSON" not in question["instructions"]
        return httpx.Response(200, json={"answers": [{"type": "choice", "name": "admission", "choice": "bot_followup"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(decisions, "get_client", lambda: client)
        cfg = ScopedConfig(chat_api_key="test-key", chat_base_url="https://api.openai.com/v1")
        result = await decisions.classify(cfg, '保持语义承接。只输出JSON：{"recipient":"..."}。', '{"latest_message":"a question"}')
    assert result == {"recipient": "bot", "should_reply": True, "continues_bot_conversation": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    [],
    [{"type": "choice", "name": "admission", "choice": "invented"}],
    [{"type": "choice", "name": "different", "choice": "bot_followup"}],
    [{"type": "choice", "name": "admission", "choice": "other"}] * 2,
])
async def test_decisions_rejects_invalid_or_ambiguous_answer(monkeypatch, answer):
    from nonebot_plugin_ai_groupmate.agent import decisions
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": answer}))) as client:
        monkeypatch.setattr(decisions, "get_client", lambda: client)
        with pytest.raises(ValueError, match="Invalid Decisions"):
            await decisions.classify(ScopedConfig(chat_api_key="test-key", chat_base_url="https://api.openai.com/v1"), "policy", "evidence")


@pytest.mark.asyncio
async def test_decisions_does_not_reuse_a_different_providers_token():
    from nonebot_plugin_ai_groupmate.agent import decisions
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    with pytest.raises(ValueError, match="OpenAI"):
        await decisions.classify(ScopedConfig(chat_api_key="different-provider", chat_base_url="https://other.invalid/v1"), "policy", "evidence")


@pytest.mark.asyncio
@pytest.mark.parametrize(("recipient", "allowed"), [("bot", True), ("other", False)])
async def test_decisions_runs_through_the_existing_recipient_guards(monkeypatch, recipient, allowed):
    from nonebot_plugin_ai_groupmate import agent

    monkeypatch.setattr(agent.plugin_config, "gatekeeper_provider", "decisions")
    classify = AsyncMock(return_value={"recipient": recipient, "should_reply": True, "continues_bot_conversation": True})
    monkeypatch.setattr(agent, "classify_decisions", classify)
    flash = AsyncMock(side_effect=AssertionError("Flash should not be called"))
    monkeypatch.setattr(agent, "get_flash_model", lambda: SimpleNamespace(ainvoke=flash))
    context = json.dumps({"recent_messages": [{"role": "self_bot", "text": "original answer"}]})
    assert await agent.check_if_should_reply("", "correction", "miyuki", turn_context=context) is allowed
    assert classify.await_args is not None
    assert "60字内" in classify.await_args.args[1]
    flash.assert_not_awaited()


@pytest.mark.asyncio
async def test_decisions_failure_is_silent_without_a_flash_fallback(monkeypatch):
    from nonebot_plugin_ai_groupmate import agent

    monkeypatch.setattr(agent.plugin_config, "gatekeeper_provider", "decisions")
    monkeypatch.setattr(agent, "classify_decisions", AsyncMock(side_effect=TimeoutError()))
    flash = AsyncMock(side_effect=AssertionError("No second provider request"))
    monkeypatch.setattr(agent, "get_flash_model", lambda: SimpleNamespace(ainvoke=flash))
    assert not await agent.check_if_should_reply("", "question", "miyuki", turn_context="{}")
    flash.assert_not_awaited()


@pytest.mark.asyncio
async def test_settings_connection_test_uses_the_selected_decisions_service(monkeypatch):
    from nonebot_plugin_ai_groupmate import webui
    from nonebot_plugin_ai_groupmate.agent import decisions
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    classify = AsyncMock(return_value={"recipient": "bot", "should_reply": False, "continues_bot_conversation": False})
    monkeypatch.setattr(decisions, "classify", classify)
    monkeypatch.setattr(webui, "create_chat_openai", Mock(side_effect=AssertionError("Do not test a different provider")))
    cfg = ScopedConfig(gatekeeper_provider="decisions")
    await webui._test_model_connection("flash", cfg)
    classify.assert_awaited_once()
    assert classify.await_args is not None
    assert classify.await_args.args[0] is cfg
