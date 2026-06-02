from typing import Any
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage


def _cacheable_text_block(text: str) -> list[str | dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_system_messages(
    stable_prompt: str,
    dynamic_context: str = "",
    *,
    use_cache_control: bool = False,
) -> list[BaseMessage]:
    stable_prompt = stable_prompt.strip()
    dynamic_context = dynamic_context.strip()

    stable_content: str | list[str | dict[str, Any]]
    if use_cache_control:
        stable_content = _cacheable_text_block(stable_prompt)
    else:
        stable_content = stable_prompt

    messages: list[BaseMessage] = [SystemMessage(content=stable_content)]
    if dynamic_context:
        messages.append(SystemMessage(content=dynamic_context))
    return messages


def normalize_system_messages(system_prompt: str | Sequence[BaseMessage]) -> list[BaseMessage]:
    if isinstance(system_prompt, str):
        return [SystemMessage(content=system_prompt)]
    return list(system_prompt)
