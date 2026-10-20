"""Run synthetic participation scenarios against the configured Flash model.

Usage: uv run python -m evals.participation_runner
Makes model calls only; never sends chat messages or runs Agent tools.
"""

import json
import asyncio
from pathlib import Path

from .runner import _ensure_nonebot

CASES = [
    ("open_question", "[群友 A] 大家有啥适合周末一起玩的桌游推荐？", True),
    ("cross_member", "[Bot] 先把大任务拆成三个步骤会容易一些。\n[群友 B] 你刚才说的三个步骤具体怎么分？", True),
    ("directed_elsewhere", "[群友 A] @小王 你昨天借我的充电器带来了吗？\n[群友 小王] 带了，下午给你。", False),
    ("finished", "[Bot] 可以，重启以后设置就会生效。\n[群友 A] 好的谢谢，解决了，不用再回复了。", False),
    ("stop", "[Bot] 我觉得还可以再优化一下。\n[群友 A] 先别插话，我们自己讨论。", False),
    ("flood", "[群友 A] 哈哈哈\n[群友 B] 哈哈哈\n[群友 C] 哈哈哈", False),
    ("topic_changed", "[Bot] 先把配置文件保存，再重启程序。\n[群友 A] @小李 今天下午几点来拿快递？\n[群友 小李] 三点。", False),
    ("shared_help", "[群友 A] 我做蛋糕每次出炉都会塌，大家有遇到过吗？", True),
    ("open_technical", "[群友 C] 我在写一个小工具，日志存文件还是放数据库比较合适？想听听各位的意见。", True),
    ("quoted_question", "[群友 C] 小说里这句“大家觉得应该怎么办”写得真怪，我先记下来，不用解读。", False),
    ("private_thread", "[群友 A] 小李，你刚说的三个步骤，第一步是先交表吗？\n[群友 小李] 对，先交给我。", False),
    ("short_ack", "[Bot] 这两个按钮都能打开设置。\n[群友 A] 嗯嗯", False),
]


async def main() -> None:
    _ensure_nonebot()
    from nonebot_plugin_ai_groupmate import agent

    model = agent.get_flash_model()
    responses = []

    class RecordingModel:
        async def ainvoke(self, messages, **kwargs):
            response = await model.ainvoke(messages, **kwargs)
            responses.append(response.content)
            return response

    agent.get_flash_model = lambda: RecordingModel()

    results = []
    for name, history, expected in CASES:
        entry = {"case": name, "expected": expected}
        for mode in ("previous", "adaptive"):
            continued = []
            responses.clear()
            try:
                decision = await asyncio.wait_for(
                    agent.check_if_should_reply(
                        history, history.splitlines()[-1], "Bot",
                        participation_context="最近参与过对话；频率允许，但不要求回复。" if mode == "adaptive" else "",
                        on_continuation=lambda: continued.append(True),
                    ),
                    timeout=30,
                )
                entry[mode] = {"reply": decision, "continued": bool(continued), "raw": responses[:]}
            except TimeoutError:
                entry[mode] = {"error": "timeout"}
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    output = Path("evals/results/participation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps({"model": agent.plugin_config.flash_model, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
