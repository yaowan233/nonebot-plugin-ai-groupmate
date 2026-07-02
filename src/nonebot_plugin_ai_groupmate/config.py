from typing import Any

from pydantic import Field, BaseModel, SecretStr
from langchain_openai import ChatOpenAI


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    proactive_private_message: bool = True
    continuous_conversation_minutes: float = 5.0
    personality_setting: str = ""
    tavily_api_key: str = ""

    # === LLM 通用配置（作为各角色的默认值） ===
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # === 聊天主模型（对话、工具调用） ===
    chat_model: str = "qwen3.7-plus"
    chat_api_key: str = ""
    chat_base_url: str = ""
    chat_temperature: float = 0.7
    chat_api_format: str = "openai"  # "openai" 或 "anthropic"

    # === 快速决策模型（Gatekeeper） ===
    flash_model: str = "qwen-flash"
    flash_api_key: str = ""
    flash_base_url: str = ""
    flash_temperature: float = 0.0
    flash_max_tokens: int = 10

    # === 群摘要模型（更新群体认知档案） ===
    summary_model: str = "qwen-flash"
    summary_api_key: str = ""
    summary_base_url: str = ""
    summary_temperature: float = 0.3
    summary_max_tokens: int = 800

    # === 图片标注模型（表情包识别与描述） ===
    tagging_model: str = "qwen-vl-max"
    tagging_api_key: str = ""
    tagging_base_url: str = ""
    tagging_temperature: float = 0.01
    tagging_api_format: str = "openai"  # "openai" 或 "anthropic"

    # === 兼容旧配置 ===
    base_model: str = ""
    qwen_token: str = ""

    # === 向量数据库 & 其他 ===
    qdrant_uri: str = ""
    qdrant_api_key: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    rerank_api_url: str = ""
    rerank_api_key: str = ""


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)


def create_chat_openai(
    cfg: ScopedConfig,
    role: str = "chat",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    model = getattr(cfg, f"{role}_model") or cfg.base_model
    api_key = getattr(cfg, f"{role}_api_key") or cfg.llm_api_key or cfg.qwen_token
    base_url = getattr(cfg, f"{role}_base_url") or cfg.llm_base_url

    if temperature is None:
        temperature = getattr(cfg, f"{role}_temperature", 0.7)
    if max_tokens is None:
        max_tokens = getattr(cfg, f"{role}_max_tokens", None)

    kwargs: dict = {
        "model": model,
        "api_key": SecretStr(api_key),
        "base_url": base_url,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def create_chat_llm(cfg: ScopedConfig) -> Any:
    if cfg.chat_api_format == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = cfg.chat_api_key or cfg.llm_api_key or cfg.qwen_token
        base_url = cfg.chat_base_url or cfg.llm_base_url

        return ChatAnthropic(
            model_name=cfg.chat_model or cfg.base_model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=cfg.chat_temperature,
            max_tokens_to_sample=4096,
            timeout=None,
            stop=None,
        )
    return create_chat_openai(cfg, "chat")


def create_tagging_llm(cfg: ScopedConfig) -> Any:
    if cfg.tagging_api_format == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = cfg.tagging_api_key or cfg.llm_api_key or cfg.qwen_token
        base_url = cfg.tagging_base_url or cfg.llm_base_url

        return ChatAnthropic(
            model_name=cfg.tagging_model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=cfg.tagging_temperature,
            max_tokens_to_sample=1024,
            timeout=None,
            stop=None,
        )
    return create_chat_openai(cfg, "tagging")
