"""DeepSeek Chat Completions transport: retain reasoning through tool turns."""

from typing import Any
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import InputTokenDetails


def is_deepseek_endpoint(base_url: str) -> bool:
    # Do not infer wire format from a model name served by an unrelated gateway.
    return (urlparse(base_url.strip()).hostname or "").lower() == "api.deepseek.com"


class DeepSeekChatOpenAI(ChatOpenAI):
    """Preserve the provider fields that generic ChatOpenAI deliberately drops.

    Kept local because both receiving AND replaying reasoning must work with
    our supported langchain-openai version. Reasoning never becomes reply text.
    """

    use_responses_api: bool | None = False

    def _get_request_payload(self, input_: Any, *, stop=None, **kwargs: Any) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        for original, wire in zip(messages, payload["messages"]):
            if isinstance(original, AIMessage):
                # Older/non-thinking assistant messages have no reasoning. An
                # empty field is valid; never synthesize a chain of thought.
                wire["reasoning_content"] = original.additional_kwargs.get("reasoning_content") or ""
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload

    @staticmethod
    def _preserve_cache_usage(message: BaseMessage, usage: dict) -> None:
        cached = usage.get("prompt_cache_hit_tokens")
        if isinstance(message, AIMessage) and isinstance(cached, int) and message.usage_metadata is not None:
            details: InputTokenDetails = {**(message.usage_metadata.get("input_token_details") or {})}
            details["cache_read"] = cached
            message.usage_metadata = {**message.usage_metadata, "input_token_details": details}

    def _create_chat_result(self, response: Any, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw.get("choices", [])):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
            self._preserve_cache_usage(generation.message, raw.get("usage") or {})
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if result is not None:
            choices = chunk.get("choices") or []
            if choices:
                reasoning = (choices[0].get("delta") or {}).get("reasoning_content")
                if reasoning is not None:
                    result.message.additional_kwargs["reasoning_content"] = reasoning
            self._preserve_cache_usage(result.message, chunk.get("usage") or {})
        return result
