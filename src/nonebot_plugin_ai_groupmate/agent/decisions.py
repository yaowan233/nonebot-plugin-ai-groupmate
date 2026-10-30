"""Typed admission decisions, isolated from auxiliary chat-model calls."""

import re
from typing import Any

import httpx

from ..config import ScopedConfig

_STATES = {
    "bot_followup": ("bot", True, True, "对 self_bot 具体回答的有效追问、补充或纠正，应回应。"),
    "bot_new_request": ("bot", True, False, "向 self_bot 提出的有效新请求，应回应。"),
    "bot_silent": ("bot", False, False, "对 self_bot 说，但只是叫停、无新增附和或无需回应的确认。"),
    "group_reply": ("group", True, False, "面向全群的开放话题，有具体自然的内容可以接。"),
    "group_silent": ("group", False, False, "面向全群，但没有具体参与点，只能泛泛附和或多余插话。"),
    "other": ("other", False, False, "在对其他群友或其他机器人说。"),
    "unclear": ("unclear", False, False, "无法确定对象，应保持沉默。"),
}
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=False)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def classify(cfg: ScopedConfig, policy: str, context: str) -> dict[str, Any]:
    base_url = (cfg.chat_base_url or cfg.llm_base_url).rstrip("/")
    api_key = cfg.chat_api_key or cfg.llm_api_key or cfg.qwen_token
    if base_url != "https://api.openai.com/v1" or not api_key:
        raise ValueError("Decisions 需要已配置的 OpenAI 官方主模型凭据")
    instructions = re.sub(r"只输出JSON：\{[^\n]*\}。", "", policy)
    response = await get_client().post(
        "https://api.openai.com/v1/decisions",
        headers={"Authorization": "Bearer " + api_key},
        json={
            "model": "gpt-6-luna",
            "input": context,
            "questions": [{
                "type": "choice", "name": "admission",
                "instructions": instructions + "\n选择最新发言对应的接话决定。",
                "choices": [{"value": name, "description": state[3]} for name, state in _STATES.items()],
            }],
        },
    )
    response.raise_for_status()
    result = response.json()
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict):
        raise ValueError("Invalid Decisions answer")
    answer = answers[0]
    if answer.get("type") != "choice" or answer.get("name") != "admission" or not isinstance(answer.get("choice"), str) or answer["choice"] not in _STATES:
        raise ValueError("Invalid Decisions choice")
    recipient, reply, continuation, _ = _STATES[answer["choice"]]
    return {"recipient": recipient, "should_reply": reply, "continues_bot_conversation": continuation}
