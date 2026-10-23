import json
import asyncio
from types import SimpleNamespace
from typing import Any, cast


async def test_burst_restarts_quiet_period_but_has_a_hard_cap():
    from nonebot_plugin_ai_groupmate.message_batching import InputBatcher, current_input_batch

    now = [0.0]
    batcher = InputBatcher[str](lambda: now[0])
    delivered = []

    async def deliver(request):
        delivered.append((request, current_input_batch.get()))

    key = ("bot", "group", "user", "")
    options: dict[str, Any] = {"delay": 1.2, "max_wait": 4, "merge": lambda a, b: a + "|" + b, "deliver": deliver}
    await batcher.submit(key, "question", message_id="1", text="图是什么", **options)
    for value in (1, 2, 3, 3.9):
        now[0] = value
        await batcher.submit(key, f"补充{value}", message_id=str(value + 2), text="补充条件", **options)
    assert batcher.pending[key].deadline == 4
    assert delivered == []
    now[0] = 4
    batcher.pending[key].changed.set()
    await asyncio.gather(*list(batcher.tasks))
    assert len(delivered) == 1
    assert delivered[0][0].startswith("question|")
    assert len(delivered[0][1]) == 5
    assert current_input_batch.get() == ()


async def test_batch_isolates_bot_group_sender_and_reply_target_and_deduplicates():
    from nonebot_plugin_ai_groupmate.message_batching import InputBatcher

    batcher = InputBatcher[str]()
    delivered = []

    async def deliver(request):
        delivered.append(request)

    keys = [("bot", "group", "u", "q1"), ("bot", "group", "u", "q2"),
            ("otherbot", "group", "u", "q1"), ("bot", "othergroup", "u", "q1"), ("bot", "group", "v", "q1")]
    options: dict[str, Any] = {"delay": 0.01, "max_wait": 0.1, "merge": lambda a, b: a + b, "deliver": deliver}
    for index, key in enumerate(keys):
        await batcher.submit(key, str(index), message_id="same-platform-id", text="hi", **options)
    await batcher.submit(keys[0], "DUPLICATE", message_id="same-platform-id", text="hi", **options)
    await asyncio.gather(*list(batcher.tasks))
    assert sorted(delivered) == ["0", "1", "2", "3", "4"]


async def test_stop_discards_pending_burst_and_new_burst_is_independent():
    from nonebot_plugin_ai_groupmate.message_batching import InputBatcher

    batcher = InputBatcher[str]()
    delivered = []

    async def deliver(request):
        delivered.append(request)

    options: dict[str, Any] = {"delay": 0.01, "max_wait": 0.1, "merge": lambda a, b: b, "deliver": deliver}
    key = ("bot", "group", "u", "")
    await batcher.submit(key, "cancelled", message_id="1", text="先说", **options)
    batcher.discard_speaker(("bot", "group", "u"))
    await batcher.submit(key, "new", message_id="2", text="重新说", **options)
    await asyncio.gather(*list(batcher.tasks))
    assert delivered == ["new"]
    assert batcher.pending == {}
    await batcher.close()


async def test_integration_keeps_first_address_and_both_inputs_for_worker(monkeypatch):
    from nonebot_plugin_uninfo import SceneType

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.agent import _build_current_request_boundary
    from nonebot_plugin_ai_groupmate.message_batching import InputBatcher, current_input_batch

    monkeypatch.setattr(plugin, "_input_batcher", InputBatcher())
    monkeypatch.setattr(plugin.plugin_config, "group_message_debounce_seconds", 0.01)
    requests = []

    async def queued(group, request):
        requests.append(request)
        return True

    monkeypatch.setattr(plugin, "_queue_group_reply_request", queued)
    def request(number, text, addressed, reply_id):
        return plugin.ReplyRequest(
            request_id=number, session=cast(Any, SimpleNamespace(scene=SimpleNamespace(type=SceneType.GROUP))),
            interface=cast(Any, None), bot=cast(Any, SimpleNamespace(self_id="bot")),
            event=cast(Any, SimpleNamespace(get_plaintext=lambda: text)), bot_name="miyuki",
            user_id="u", user_name="U", is_tome=addressed, is_continuous=False, reply_to_id=reply_id,
        )
    await plugin._submit_reply_input("group", request("a", "这个是什么厂的", True, "quoted-image"), "123", "这个是什么厂的")
    await plugin._submit_reply_input("group", request("b", "我说图片右边那个", False, None), "124", "我说图片右边那个")
    await asyncio.gather(*list(plugin._input_batcher.tasks))
    assert len(requests) == 1
    ready = requests[0]
    assert ready.is_tome
    assert ready.reply_to_id == "quoted-image"
    assert ready.input_batch == (("123", "这个是什么厂的"), ("124", "我说图片右边那个"))

    boundaries = []
    async def handle(*_args):
        boundaries.append(_build_current_request_boundary([], "u", "U", ready.event))
        assert current_input_batch.get() == ready.input_batch
    monkeypatch.setattr(plugin, "handle_reply_logic", handle)
    await plugin._execute_reply_request(ready)
    assert "这个是什么厂的" in boundaries[0]
    assert "我说图片右边那个" in boundaries[0]
    assert current_input_batch.get() == ()


async def test_batched_input_retains_two_independent_questions_in_boundary():
    from nonebot_plugin_ai_groupmate.agent import _build_current_request_boundary
    from nonebot_plugin_ai_groupmate.message_batching import current_input_batch

    inputs = (("1", "明天几点开服"), ("2", "还有这个文件也帮我整理一下"))
    token = current_input_batch.set(inputs)
    try:
        boundary = _build_current_request_boundary([], "u", "U", None)
    finally:
        current_input_batch.reset(token)
    assert all(text in boundary for _, text in inputs)
    assert "独立问题都保留" in boundary
    assert json.dumps("还有这个文件也帮我整理一下", ensure_ascii=False) in boundary
