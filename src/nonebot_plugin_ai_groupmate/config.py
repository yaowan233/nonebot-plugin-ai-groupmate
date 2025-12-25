from pydantic import BaseModel, Field


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    milvus_uri: str = "milvus_demo.db"
    milvus_user: str = ""
    milvus_password: str = ""
    tavily_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""
    openai_token: str = ""
    vlm_ollama_base_url: str = ""
    vlm_model: str = ""
    vlm_provider: str = "ollama"  # 可选: "ollama", "openai"
    vlm_openai_base_url: str = ""
    vlm_openai_api_key: str = ""


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)
