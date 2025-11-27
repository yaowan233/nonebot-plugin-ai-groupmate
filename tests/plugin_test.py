from fake import fake_group_message_event_v11
from nonebug import App
import pytest


@pytest.mark.asyncio
async def test_pip(app: App):
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as OnebotV11Adapter
    from nonebot.adapters.onebot.v11 import Bot

    event = fake_group_message_event_v11(message="hello")
    try:
        from src.nonebot_plugin_ai_groupmate import record  # type:ignore
    except ImportError:
        pytest.skip("nonebot_plugin_ai_groupmate.record not found")

    async with app.test_matcher(record) as ctx:
        adapter = nonebot.get_adapter(OnebotV11Adapter)
        bot = ctx.create_bot(base=Bot, adapter=adapter)
        ctx.receive_event(bot, event)
        ctx.should_finished()
