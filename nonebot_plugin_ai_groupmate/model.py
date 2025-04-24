from datetime import datetime
from typing import Optional

from nonebot_plugin_orm import Model
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column


class MediaStorage(Model):
    """媒体资源中心化存储"""

    media_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)  # SHA-256哈希
    file_path: Mapped[str]  # 实际存储路径或URL
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    references: Mapped[int] = mapped_column(default=1, index=True)  # 引用计数
    description: Mapped[str]
    vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class ChatHistory(Model):
    msg_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(index=True)
    user_id: Mapped[str] = mapped_column(index=True)
    content_type: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    user_name: Mapped[str]
    media_id: Mapped[Optional[int]]  # 媒体消息专用
    vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
