from pydantic import BaseModel, Field


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    tavily_api_key: str = ""
    base_model: str = ""
    qwen_token: str = ""
    qdrant_uri: str = ""
    qdrant_api_key: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    rerank_api_url: str = ""
    rerank_api_key: str = ""


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)
