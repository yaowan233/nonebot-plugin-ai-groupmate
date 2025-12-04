import pytest
from fake import fake_group_message_event_v11
from nonebug import App


@pytest.mark.asyncio
async def test_pip(app: App):
    import nonebot
    from nonebot.adapters.onebot.v11 import Bot, Adapter as OnebotV11Adapter

    event = fake_group_message_event_v11(message="hello")
    try:
        from nonebot_plugin_ai_groupmate import record  # type:ignore
    except ImportError:
        pytest.skip("nonebot_plugin_ai_groupmate.record not found")

    async with app.test_matcher(record) as ctx:
        adapter = nonebot.get_adapter(OnebotV11Adapter)
        bot = ctx.create_bot(base=Bot, adapter=adapter)
        ctx.receive_event(bot, event)
        ctx.should_call_api(
            "get_group_info",
            {"group_id": 87654321},  # 这里的 ID 必须和 fake.py 里定义的一致
            result={"group_id": 87654321, "group_name": "Test Group", "member_count": 10} # 模拟的返回值
        )
        ctx.should_call_api(
            "get_group_member_info",
            {
                "group_id": 87654321,
                "user_id": 12345678,
                "no_cache": True   # 注意：报错里显示 uninfo 传了 no_cache=True，这里必须完全一致
            },
            result={
                "group_id": 87654321,
                "user_id": 12345678,
                "nickname": "test_nick",
                "card": "",
                "role": "member",
            }
        )
        ctx.should_finished()
