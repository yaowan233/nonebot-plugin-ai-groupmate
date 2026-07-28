from types import SimpleNamespace


def _event(*segments, reply_user_id: str | None = None):
    sender = (
        SimpleNamespace(user_id=reply_user_id)
        if reply_user_id is not None
        else None
    )
    reply = SimpleNamespace(sender=sender) if sender is not None else None
    return SimpleNamespace(original_message=list(segments), reply=reply)


def _at(target: str):
    return SimpleNamespace(type="at", data={"qq": target})


def _text(content: str):
    return SimpleNamespace(type="text", data={"text": content})


def test_raw_at_selects_only_the_targeted_connected_bot():
    from nonebot_plugin_ai_groupmate import _select_addressed_bot_id

    event = _event(_at("bot-2"), _text("hello"))

    assert _select_addressed_bot_id(event, {"bot-1", "bot-2"}) == "bot-2"


def test_at_to_group_member_does_not_select_a_bot():
    from nonebot_plugin_ai_groupmate import _select_addressed_bot_id

    event = _event(_at("member-1"), _text("hello"))

    assert _select_addressed_bot_id(event, {"bot-1", "bot-2"}) is None


def test_reply_selects_the_bot_that_sent_the_replied_message():
    from nonebot_plugin_ai_groupmate import _select_addressed_bot_id

    event = _event(_text("hello"), reply_user_id="bot-1")

    assert _select_addressed_bot_id(event, {"bot-1", "bot-2"}) == "bot-1"


def test_first_explicit_bot_mention_owns_a_multi_bot_message():
    from nonebot_plugin_ai_groupmate import _select_addressed_bot_id

    event = _event(_at("bot-2"), _at("bot-1"), _text("hello"))

    assert _select_addressed_bot_id(event, {"bot-1", "bot-2"}) == "bot-2"


def test_platform_message_id_deduplicates_different_processed_bodies():
    from nonebot_plugin_ai_groupmate import _matches_inbound_message

    assert _matches_inbound_message(
        "id: 42\n@Bot hello",
        "id: 42\n",
        "hello",
    )


def test_message_from_another_connected_bot_is_ignored():
    from nonebot_plugin_ai_groupmate import _is_connected_bot_sender

    connected_bot_ids = {"bot-1", "bot-2"}

    assert _is_connected_bot_sender("bot-1", connected_bot_ids)
    assert _is_connected_bot_sender("bot-2", connected_bot_ids)
    assert not _is_connected_bot_sender("member-1", connected_bot_ids)
