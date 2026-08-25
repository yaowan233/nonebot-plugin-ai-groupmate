from typing import Any, Literal

from pydantic import Field, BaseModel, SecretStr, field_validator
from langchain_openai import ChatOpenAI


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    global_model_daily_group_limit_enabled: bool = True
    global_model_daily_group_limit: int = Field(
        default=50,
        ge=0,
        description="每个群每天可使用公共主模型的回复次数，0 表示不限制",
    )
    repeat_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    proactive_reaction_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    proactive_meme_probability: float = Field(default=0.02, ge=0.0, le=1.0)
    proactive_private_message: bool = True
    continuous_conversation_minutes: float = 5.0
    usage_webui_enabled: bool = True
    usage_webui_path: str = "/ai-groupmate/usage"
    usage_webui_token: str = ""
    # === 分群模型 API 中转（启动配置） ===
    group_api_relay_url: str = "https://mayumi.xyz"
    group_api_relay_registration_token: str = ""
    group_api_local_encryption_key: str = ""
    group_api_relay_timeout_seconds: float = Field(default=15.0, gt=0)
    group_api_ticket_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    group_api_allowed_provider_hosts: list[str] = Field(default_factory=list)
    chat_input_cost_per_million: float = 2.0
    chat_output_cost_per_million: float = 8.0
    chat_cached_input_cost_per_million: float = 0.4
    chat_explicit_cached_input_cost_per_million: float = 0.2
    chat_cache_creation_input_cost_per_million: float = 2.5
    chat_explicit_prompt_cache: bool = True
    agent_timeout_seconds: float = Field(default=180.0, gt=0)
    agent_llm_timeout_seconds: float = Field(default=60.0, gt=0)
    agent_tool_timeout_seconds: float = Field(default=30.0, gt=0)
    agent_max_concurrency: int = Field(default=4, ge=1)
    background_image_max_concurrency: int = Field(default=2, ge=1)
    background_image_max_pending: int = Field(default=100, ge=1)
    maintenance_max_concurrency: int = Field(default=1, ge=1)
    media_vectorize_min_references: int = Field(default=3, ge=1)
    media_vectorize_batch_size: int = Field(default=1000, ge=1, le=10000)
    media_vectorize_concurrency: int = Field(default=8, ge=1, le=32)
    group_memory_update_timeout_seconds: float = Field(default=120.0, gt=0)
    agent_max_llm_calls: int = Field(default=8, ge=1)
    agent_max_total_tokens: int = Field(default=64000, ge=1)
    agent_tool_result_max_chars: int = Field(default=6000, ge=256)
    chat_long_context_threshold_tokens: int = 256000
    chat_long_input_cost_per_million: float = 6.0
    chat_long_output_cost_per_million: float = 24.0
    chat_long_cached_input_cost_per_million: float = 1.2
    chat_long_explicit_cached_input_cost_per_million: float = 0.6
    chat_long_cache_creation_input_cost_per_million: float = 7.5
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
    chat_api_format: Literal["openai", "anthropic", "vertex"] = "openai"
    chat_multimodal: bool = True  # 主聊天模型是否支持图片输入

    # === Google Vertex AI（chat/tagging/vision 共用） ===
    # 凭据优先级：vertex_credentials_path > vertex_api_key > ADC。
    # ADC 可通过 GOOGLE_APPLICATION_CREDENTIALS、Workload Identity 或
    # `gcloud auth application-default login` 提供。
    vertex_project: str = ""
    vertex_location: str = "global"
    vertex_api_key: str = ""
    vertex_credentials_path: str = ""

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
    tagging_api_format: Literal["openai", "anthropic", "vertex"] = "openai"

    # === 图片回读辅助模型（主模型不支持图片时，用它对工具返回的图片做内容总结） ===
    vision_model: str = ""
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_temperature: float = 0.01
    vision_api_format: Literal["openai", "anthropic", "vertex"] = "openai"
    vision_input_cost_per_million: float = 0.0
    vision_output_cost_per_million: float = 0.0

    # === 兼容旧配置 ===
    base_model: str = ""
    qwen_token: str = ""

    # === 向量数据库 & 其他 ===
    qdrant_uri: str = ""
    qdrant_api_key: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "BAAI/bge-m3"
    # 请求是否携带 dimensions 参数及文本向量维度。
    # 不填（None）：请求不带 dimensions，使用模型默认输出维度，兼容
    #   不支持 dimensions 参数的 provider（如硅基流动的 BAAI/bge-m3）。
    # 填写：请求携带 dimensions=<该值>，要求模型/provider 支持该参数
    #   （如硅基流动的 Qwen/Qwen3-Embedding-8B，可指定 1024 等维度）。
    #   若模型不支持 dimensions 参数，启动探测会报配置错误。
    embedding_dimension: int | None = Field(default=None, ge=1)

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def _blank_embedding_dimension_to_none(cls, value: Any) -> Any:
        # WebUI 提交空白输入时值为 ""，转换为 None 以保持"未配置"语义。
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    rerank_api_url: str = ""
    rerank_api_key: str = ""

    # 表情包向量化模式:
    #   "multimodal": 使用 qwen3-vl-embedding 生成描述文本 + 原图双向量，
    #                  需要配置 qwen_token (DashScope)，图找图可用。
    #                  若未配置 qwen_token，会自动降级为 "text"。
    #   "text":       仅用 embedding_api_key 的文本向量化描述，无需 qwen_token，
    #                  运维最轻；图找图 (search_similar_meme_by_id) 不可用。
    meme_embedding_mode: Literal["multimodal", "text"] = "multimodal"


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


def _vertex_model_name(model: str) -> str:
    """Accept an OpenRouter Gemini name when migrating to direct Vertex AI."""
    normalized = model.strip()
    if normalized.lower().startswith("google/"):
        return normalized.split("/", 1)[1]
    return normalized


def create_vertex_llm(
    cfg: ScopedConfig,
    role: str = "chat",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Create a Gemini chat model backed by Vertex AI with renewable auth."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = getattr(cfg, f"{role}_model") or cfg.base_model
    if not model:
        raise ValueError(f"{role}_model 未配置")
    if temperature is None:
        temperature = getattr(cfg, f"{role}_temperature", 0.7)
    if max_tokens is None:
        max_tokens = getattr(cfg, f"{role}_max_tokens", None)

    kwargs: dict[str, Any] = {
        "model": _vertex_model_name(model),
        "vertexai": True,
        "temperature": temperature,
    }

    if cfg.vertex_credentials_path:
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            cfg.vertex_credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        kwargs["credentials"] = credentials
        project = cfg.vertex_project or credentials.project_id
        if project:
            kwargs["project"] = project
        kwargs["location"] = cfg.vertex_location or "global"
    else:
        api_key = cfg.vertex_api_key
        if api_key:
            # Vertex AI Express Mode 的 API key 已绑定 Express 项目。这里不能
            # 同时传 project/location，否则 google-genai 会切换到标准 Vertex
            # 认证流程并尝试加载 ADC。
            kwargs["api_key"] = SecretStr(api_key)
        else:
            if cfg.vertex_project:
                kwargs["project"] = cfg.vertex_project
            kwargs["location"] = cfg.vertex_location or "global"

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatGoogleGenerativeAI(**kwargs)


def create_chat_llm(cfg: ScopedConfig) -> Any:
    if cfg.chat_api_format == "vertex":
        return create_vertex_llm(cfg, "chat")
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
    if cfg.tagging_api_format == "vertex":
        return create_vertex_llm(cfg, "tagging", max_tokens=1024)
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


def create_vision_llm(cfg: ScopedConfig) -> Any:
    if cfg.vision_api_format == "vertex":
        return create_vertex_llm(cfg, "vision", max_tokens=1024)
    if cfg.vision_api_format == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = cfg.vision_api_key or cfg.llm_api_key or cfg.qwen_token
        base_url = cfg.vision_base_url or cfg.llm_base_url

        return ChatAnthropic(
            model_name=cfg.vision_model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=cfg.vision_temperature,
            max_tokens_to_sample=1024,
            timeout=None,
            stop=None,
        )
    return create_chat_openai(cfg, "vision")
