from pydantic import BaseModel


class Config(BaseModel):
    bot_name: str = "bot"
    deepseek_bearer_token: str = ""
    siliconflow_bearer_token: str = ""
    reply_probability: float = 0.04
    personality_setting: str = ""
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
