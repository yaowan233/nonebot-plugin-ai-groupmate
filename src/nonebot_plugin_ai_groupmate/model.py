from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import JSON, Float, Index, String, Boolean, Integer
from sqlalchemy.orm import Mapped, mapped_column
from nonebot_plugin_orm import Model


class MediaStorage(Model):
    """媒体资源中心化存储"""

    media_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)  # SHA-256哈希
    file_path: Mapped[str]  # 实际存储路径或URL
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    references: Mapped[int] = mapped_column(default=1, index=True)  # 引用计数
    description: Mapped[str]
    vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    embedding_version: Mapped[int] = mapped_column(default=0, index=True)


class MediaStorageSchema(BaseModel):
    media_id: int
    file_hash: str
    file_path: str
    created_at: datetime
    references: int
    description: str
    vectorized: bool
    embedding_version: int = 0


class ChatHistory(Model):
    msg_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(index=True)
    user_id: Mapped[str] = mapped_column(index=True)
    content_type: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    user_name: Mapped[str]
    media_id: Mapped[int | None]  # 媒体消息专用
    vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    vectorized_version: Mapped[int] = mapped_column(Integer, default=0, index=True)

    __table_args__ = (
        # 覆盖 group_memory 更新查询: WHERE session_id=? AND created_at>? AND content_type IN (...)
        Index("ix_chat_session_time", "session_id", "created_at"),
        # 覆盖群内表情包热度聚合: session_id + content_type + media_id。
        Index(
            "ix_chat_session_type_media",
            "session_id",
            "content_type",
            "media_id",
        ),
    )


class UserRelation(Model):
    """用户关系/好感度表"""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(index=True)
    user_name: Mapped[str]
    favorability: Mapped[int] = mapped_column(default=0)  # 好感度，默认0
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now, onupdate=datetime.now)

    def get_status_desc(self) -> str:
        """根据分数返回关系描述"""
        score = self.favorability
        if score < -70:
            return "明显疏远"
        if score < -40:
            return "保持距离"
        if score < -15:
            return "稍显克制"
        if score < 5:
            return "陌生/普通"
        if score < 25:
            return "有点熟"
        if score < 50:
            return "朋友/熟人"
        if score < 70:
            return "亲近/好友"
        if score < 90:
            return "非常亲近"
        return "最亲近的人"


class GroupMemory(Model):
    """群体认知档案（每群一条记录）"""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(unique=True, index=True)
    summary: Mapped[str] = mapped_column(default="")
    msg_count_at_last_update: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now, onupdate=datetime.now, index=True)


class TokenUsage(Model):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(index=True)
    session_type: Mapped[str] = mapped_column(String(16), default="group", index=True)
    user_id: Mapped[str] = mapped_column(index=True)
    user_name: Mapped[str] = mapped_column(default="")
    model: Mapped[str] = mapped_column(default="", index=True)
    request_id: Mapped[str] = mapped_column(default="", index=True)
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    cached_tokens: Mapped[int] = mapped_column(default=0)
    cache_creation_tokens: Mapped[int] = mapped_column(default=0)
    total_tokens: Mapped[int] = mapped_column(default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    agent_llm_calls: Mapped[int] = mapped_column(default=0)
    agent_tool_calls: Mapped[int] = mapped_column(default=0)
    agent_duration_ms: Mapped[int] = mapped_column(default=0)
    agent_tool_timeouts: Mapped[int] = mapped_column(default=0)
    agent_tool_timeout_tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    agent_result_truncations: Mapped[int] = mapped_column(default=0)
    agent_side_effect_deduplications: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)

    __table_args__ = (
        Index("ix_token_usage_session_time", "session_id", "created_at"),
        Index("ix_token_usage_user_time", "user_id", "created_at"),
    )


class RuntimeConfigOverride(Model):
    """WebUI 保存的插件配置覆盖项。"""

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    overrides: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now,
        onupdate=datetime.now,
    )


class ChatHistorySchema(BaseModel):
    msg_id: int
    session_id: str
    user_id: str
    content_type: str
    content: str
    created_at: datetime
    user_name: str
    media_id: int | None = None
    vectorized: bool | None = False
    vectorized_version: int = 0

    class Config:
        from_attributes = True  # ✅ 允许从 ORM 对象创建
