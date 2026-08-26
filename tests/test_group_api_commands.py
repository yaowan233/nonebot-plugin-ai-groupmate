import re


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
