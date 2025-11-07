from pydantic import BaseModel


class Config(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
    tavily_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""
    openai_token: str = ""
    vlm_ollama_base_url: str = ""
    vlm_model: str = ""

