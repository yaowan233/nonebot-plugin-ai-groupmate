import re
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _matcher_pattern(matcher) -> re.Pattern:
    from nonebot.rule import RegexRule

    for checker in matcher.rule.checkers:
        if isinstance(checker.call, RegexRule):
            rule = checker.call
            return re.compile(rule.regex, rule.flags)
    raise AssertionError("matcher has no regex rule")


def test_group_api_commands_match_case_insensitive_api_suffix():
    from nonebot_plugin_ai_groupmate import group_api_commands

    pattern = _matcher_pattern(group_api_commands.configure_group_api)
    for text in ("/配置群api", "/配置群Api", "/配置群API", "/群api配置"):
        assert pattern.fullmatch(text), text
    # The configured command_start prefix stays required, as with on_command.
    assert pattern.fullmatch("配置群API") is None

    pattern = _matcher_pattern(group_api_commands.show_group_api)
    assert pattern.fullmatch("/查看群api")

    pattern = _matcher_pattern(group_api_commands.delete_group_api)
    matched = pattern.fullmatch("/删除群Api 确认")
    assert matched is not None
    assert matched.group("arg").strip() == "确认"


def test_private_api_commands_match_case_insensitive_api_suffix():
    from nonebot_plugin_ai_groupmate import group_api_commands

    pattern = _matcher_pattern(group_api_commands.configure_private_api)
    for text in ("/配置个人api", "/个人Api配置", "/配置私聊aPI"):
        assert pattern.fullmatch(text), text

    pattern = _matcher_pattern(group_api_commands.show_private_api)
    assert pattern.fullmatch("/查看个人api")
    assert pattern.fullmatch("/查看私聊Api")

    pattern = _matcher_pattern(group_api_commands.delete_private_api)
    assert pattern.fullmatch("/删除私聊api 确认")


def test_submit_commands_capture_the_config_code_argument():
    from nonebot_plugin_ai_groupmate import group_api_commands

    pattern = _matcher_pattern(group_api_commands.submit_group_api)
    matched = pattern.fullmatch("/提交群api  AGC-2345-67AB-ABCD-EFGH ")
    assert matched is not None
    assert matched.group("arg").strip() == "AGC-2345-67AB-ABCD-EFGH"
    # An empty argument must still match so the handler shows the usage prompt.
    matched = pattern.fullmatch("/提交群API")
    assert matched is not None
    assert matched.group("arg") == ""

    pattern = _matcher_pattern(group_api_commands.submit_private_api)
    matched = pattern.fullmatch("/提交个人aPi AGC-2345-67AB-ABCD-EFGH")
    assert matched is not None
    assert matched.group("arg").strip() == "AGC-2345-67AB-ABCD-EFGH"
    assert pattern.fullmatch("/提交api")


@pytest.mark.parametrize("scope", ["group", "private"])
async def test_relay_submit_applies_without_second_model_call(monkeypatch, scope):
    from nonebot_plugin_ai_groupmate import group_api_commands as commands
    from nonebot_plugin_ai_groupmate.group_model_config import ActiveGroupModelConfig, ActivePrivateModelConfig

    saved = []
    acknowledged = []
    checked = []
    payload = SimpleNamespace(base_url="https://example.com/v1", chat_model="test-model")
    redeemed = SimpleNamespace(payload=payload, target=SimpleNamespace(scope=scope, subject_id="123", operator_id="123"), ticket_id="ticket")

    class Relay:
        cipher = object()

        async def redeem(self, *args, **kwargs):
            return redeemed

        async def acknowledge(self, db, item, *, outcome):
            acknowledged.append(outcome)

    class Finished(Exception):
        pass

    class Matcher:
        @staticmethod
        async def finish(message):
            raise Finished(message)

    async def check_provider(url, allowed):
        checked.append((url, allowed))

    async def save_group(db, **kwargs):
        saved.append(kwargs)
        return ActiveGroupModelConfig(group_id="123", api_format="openai", base_url=payload.base_url, api_key="secret", chat_model=payload.chat_model, chat_multimodal=True)

    async def save_private(db, **kwargs):
        saved.append(kwargs)
        return ActivePrivateModelConfig(user_id="123", api_format="openai", base_url=payload.base_url, api_key="secret", chat_model=payload.chat_model, chat_multimodal=True)

    async def no_model_call(*args, **kwargs):
        pytest.fail("relay submit must not perform another model request")

    monkeypatch.setattr(commands, "_relay", lambda: Relay())
    monkeypatch.setattr(commands, "get_runtime_config", lambda: SimpleNamespace(group_api_allowed_provider_hosts=[]))
    monkeypatch.setattr(commands, "validate_group_provider_resolution", check_provider)
    monkeypatch.setattr(commands, "_test_candidate_connection", no_model_call, raising=False)
    monkeypatch.setattr(commands, "save_group_model_config", save_group)
    monkeypatch.setattr(commands, "save_private_model_config", save_private)
    monkeypatch.setattr(commands, "_clear_group_chat_model_cache", lambda *_: None)
    monkeypatch.setattr(commands, "_clear_private_chat_model_cache", lambda *_: None)
    session = SimpleNamespace(scene=SimpleNamespace(type=commands.SceneType.PRIVATE), user=SimpleNamespace(id="123"))
    with pytest.raises(Finished) as result:
        await commands._do_submit(cast(Any, object()), cast(Any, session), scope=scope, code="valid", matcher=cast(Any, Matcher))
    assert "已更新" in str(result.value)
    assert "测试通过" not in str(result.value)
    assert checked == [(payload.base_url, [])]
    assert saved[0]["connection_tested"] is False
    assert acknowledged == ["applied"]
