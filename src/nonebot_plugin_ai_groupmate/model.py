from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import JSON, String, Boolean
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


class MediaStorageSchema(BaseModel):
    media_id: int
    file_hash: str
    file_path: str
    created_at: datetime
    references: int
    description: str
    vectorized: bool


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
        if score <= -30:
            return "厌恶/仇视"
        if score <= -10:
            return "冷淡/防备"
        if score <= 10:
            return "陌生/普通"
        if score <= 40:
            return "友善/熟人"
        if score <= 70:
            return "亲密/死党"
        return "恋人/依赖"


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

    class Config:
        from_attributes = True  # ✅ 允许从 ORM 对象创建
