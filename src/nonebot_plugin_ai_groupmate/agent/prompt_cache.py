import hashlib
from typing import Any
from urllib.parse import urlparse
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage

CACHE_CONTROL = {"type": "ephemeral"}


def is_openrouter_base_url(base_url: str) -> bool:
    """Return whether the configured chat endpoint is OpenRouter."""
    hostname = (urlparse(base_url.strip()).hostname or "").lower()
    return hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai")


def should_use_explicit_prompt_cache(
    *,
    enabled: bool,
    api_format: str,
    base_url: str,
    model: str,
) -> bool:
    """Select providers where our block-level cache marker is supported."""
    if not enabled:
        return False
    if api_format.strip().lower() == "anthropic":
        return True

    hostname = (urlparse(base_url.strip()).hostname or "").lower()
    if hostname == "dashscope.aliyuncs.com":
        return True
    if not is_openrouter_base_url(base_url):
        return False

    # OpenRouter translates Anthropic-style cache_control blocks for Gemini and
    # also accepts them for Claude models exposed through Chat Completions.
    normalized_model = model.strip().lower().lstrip("~")
    return normalized_model.startswith(("google/gemini", "anthropic/"))


def build_openrouter_request_kwargs(
    base_url: str,
    session_id: str,
) -> dict[str, Any]:
    """Build a privacy-preserving sticky-routing key for an OpenRouter call."""
    if not is_openrouter_base_url(base_url):
        return {}
    digest = hashlib.blake2s(
        str(session_id).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    return {
        "extra_body": {
            "session_id": f"ai-groupmate-{digest}",
        }
    }


def _cacheable_text_block(text: str) -> list[str | dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": CACHE_CONTROL,
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


def _with_cache_control(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        if not content.strip():
            return content, False
        return _cacheable_text_block(content), True

    if not isinstance(content, list):
        return content, False

    new_content = list(content)
    for index in range(len(new_content) - 1, -1, -1):
        item = new_content[index]
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            if not item["text"].strip():
                continue
            new_item = dict(item)
            new_item["cache_control"] = CACHE_CONTROL
            new_content[index] = new_item
            return new_content, True
        if isinstance(item, str) and item.strip():
            new_content[index] = {
                "type": "text",
                "text": item,
                "cache_control": CACHE_CONTROL,
            }
            return new_content, True

    return content, False


def add_ephemeral_cache_marker(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Mark the last cacheable content block so providers can cache the prompt prefix."""
    marked_messages = list(messages)
    for index in range(len(marked_messages) - 1, -1, -1):
        message = marked_messages[index]
        content, changed = _with_cache_control(message.content)
        if not changed:
            continue
        marked_messages[index] = message.model_copy(update={"content": content})
        break
    return marked_messages
