import base64
import hashlib
import traceback
from datetime import timedelta
from typing import List, Optional, Dict, Tuple

from PIL import Image
import io

from nonebot_plugin_orm import AsyncSession
from sqlalchemy import Select, Update


from nonebot import logger, get_plugin_config
from .model import ChatHistory
from .milvus import MilvusOP
from .config import Config

plugin_config = get_plugin_config(Config)

def generate_file_hash(file_data: bytes) -> str:
    sha256 = hashlib.sha256()
    sha256.update(file_data)
    return sha256.hexdigest()


def check_and_compress_image_bytes(
    image_bytes, max_size_mb=2, quality_start=95, image_format="JPEG"
):
    """
    检查bytes格式的图片大小，如果大于指定大小则压缩

    参数:
    image_bytes (bytes): 图片的bytes数据
    max_size_mb (float): 最大文件大小（MB）
    quality_start (int): 初始压缩质量（1-100）
    image_format (str): 图像格式，如"JPEG", "PNG"等

    返回:
    bytes: 压缩后的图片bytes数据
    """
    if image_format.upper() == "JPG":
        image_format = "JPEG"
    # 将MB转为字节
    max_size_bytes = max_size_mb * 1024 * 1024

    # 检查bytes大小
    file_size = len(image_bytes)

    if file_size <= max_size_bytes:
        # print(f"图片大小为 {file_size / 1024 / 1024:.2f}MB，无需压缩")
        return image_bytes

    try:
        # 读取图片
        img = Image.open(io.BytesIO(image_bytes))

        # 尝试不同的质量等级直到文件大小小于目标大小
        quality = quality_start
        compressed_image = io.BytesIO()

        while quality > 10:
            compressed_image.seek(0)
            compressed_image.truncate(0)  # 清空BytesIO对象
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(
                compressed_image, format=image_format, quality=quality, optimize=True
            )

            compressed_size = compressed_image.tell()

            if compressed_size <= max_size_bytes:
                break

            quality -= 5

        # 如果压缩后的图像仍然大于目标大小，可以考虑调整图像尺寸
        if compressed_size > max_size_bytes:
            logger.info("通过调整质量无法达到目标大小，尝试调整图像尺寸...")
            width, height = img.size
            ratio = (max_size_bytes / compressed_size) ** 0.5  # 估算缩放比例

            new_width = int(width * ratio * 0.9)  # 稍微减小一点以确保大小达标
            new_height = int(height * ratio * 0.9)

            img = img.resize((new_width, new_height), Image.LANCZOS)

            compressed_image.seek(0)
            compressed_image.truncate(0)
            img.save(
                compressed_image, format=image_format, quality=quality, optimize=True
            )

        compressed_bytes = compressed_image.getvalue()
        final_size = len(compressed_bytes)
        logger.info(
            f"图片已压缩: {file_size / 1024 / 1024:.2f}MB -> {final_size / 1024 / 1024:.2f}MB (质量: {quality})"
        )
        return compressed_bytes

    except Exception as e:
        logger.error(f"压缩图片时出错: {e}")
        print(traceback.format_exc())
        return image_bytes


def bytes_to_base64(bytes_data):
    """
    将 bytes 数据转换为 base64 编码的字符串

    参数:
    bytes_data (bytes): 需要转换的字节数据

    返回:
    str: base64 编码的字符串
    """
    base64_bytes = base64.b64encode(bytes_data)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


async def split_chat_into_context_groups(
    db_session: AsyncSession,
    session_id: str,
    max_time_gap: timedelta = timedelta(hours=1),
    max_token_count: int = 1000,
    max_messages: int = 50,
) -> List[List[ChatHistory]]:
    """
    将一个会话内的聊天记录智能切分为多个上下文组

    参数:
        db_session: 数据库会话
        session_id: 对话会话ID
        max_time_gap: 最大时间间隔，超过此间隔视为新对话上下文
        max_token_count: 单个上下文组的最大token数
        max_messages: 单个上下文组的最大消息数

    返回:
        切分后的对话组列表，每组是ChatHistory对象列表
    """
    # 获取该会话的所有消息
    query = (
        Select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .where(ChatHistory.vectorized.is_(False))
        .order_by(ChatHistory.created_at)
    )

    all_messages = (await db_session.execute(query)).scalars().all()

    if not all_messages:
        return []

    # 初始化结果和当前组
    context_groups = []
    current_group = []
    current_token_count = 0
    last_message_time = None

    for message in all_messages:
        # 计算当前消息的token数量
        message_tokens = estimate_token_count(message.content)

        # 检查是否需要开始新的组
        start_new_group = False

        # 条件1: 时间间隔过大
        if (
            last_message_time
            and (message.created_at - last_message_time) > max_time_gap
        ):
            start_new_group = True

        # 条件2: token超限
        elif current_token_count + message_tokens > max_token_count:
            start_new_group = True

        # 条件3: 消息数超限
        elif len(current_group) >= max_messages:
            start_new_group = True

        # 如果满足任一条件，保存当前组并开始新组
        if start_new_group and current_group:
            context_groups.append(current_group)
            current_group = []
            current_token_count = 0

        # 添加当前消息到当前组
        current_group.append(message)
        current_token_count += message_tokens
        last_message_time = message.created_at

    # 添加最后一组(如果非空)
    if current_group:
        context_groups.append(current_group)

    return context_groups


def estimate_token_count(text: str) -> int:
    """
    估算文本的token数量，一个粗略的估计是每4个字符约1个token

    参数:
        text: 需要计算的文本

    返回:
        估计的token数量
    """
    # 使用简单的估算方法，实际情况应使用与向量模型匹配的tokenizer
    return len(text) // 3 + 1  # 简化估算，生产环境应使用模型的tokenizer


async def process_and_vectorize_session_chats(
    db_session: AsyncSession,
    session_id: str,
    max_time_gap: timedelta = timedelta(hours=1),
    max_token_count: int = 1000,
) -> Optional[Dict]:
    """
    处理并向量化一个会话内的聊天记录，按上下文智能切分

    参数:
        db_session: 数据库会话
        session_id: 会话ID
        max_time_gap: 时间间隔切分标准
        max_token_count: token数量切分标准
        max_workers: 并行向量化线程数
    """
    # 1. 获取并切分会话
    context_groups = await split_chat_into_context_groups(
        db_session,
        session_id,
        max_time_gap=max_time_gap,
        max_token_count=max_token_count,
    )

    if not context_groups:
        return None

    # 2. 处理每个上下文组
    processed_groups = 0

    for group in context_groups:
        # 3. 为每个组创建合并上下文
        context_text, msg_ids = combine_messages_into_context(group)

        # 4. 向量化合并后的上下文
        try:
            # 向量化上下文
            MilvusOP.insert(context_text, session_id)

            # 6. 更新消息向量化状态
            for msg_id in msg_ids:
                await mark_message_as_vectorized(db_session, msg_id)

            processed_groups += 1

        except Exception as e:
            logger.error(
                f"向量化会话组失败 {session_id}: {str(e)}\n{traceback.format_exc()}"
            )

    # 提交所有更改
    await db_session.commit()

    return {
        "session_id": session_id,
        "processed_groups": processed_groups,
        "total_groups": len(context_groups),
    }


def combine_messages_into_context(messages: List[ChatHistory]) -> Tuple[str, List[int]]:
    """
    将消息组合成一个上下文文本，并返回消息ID列表
    """
    combined_text = ""
    msg_ids = []

    for msg in messages:
        time = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        combined_text += f"[{time}] "
        if msg.content_type == "text":
            combined_text += f"{msg.user_name}: {msg.content}\n"
        elif msg.content_type == "image":
            combined_text += (
                f"{msg.user_name} 发送了一张图片\n该图片的描述为: {msg.content}\n"
            )
        elif msg.content_type == "bot":
            combined_text += f"[{plugin_config.bot_name}](你自己)发言: {msg.content}\n"
        msg_ids.append(msg.msg_id)

    return combined_text.strip(), msg_ids


async def mark_message_as_vectorized(db_session: AsyncSession, msg_id: int):
    """
    将消息标记为已向量化
    """
    await db_session.execute(
        Update(ChatHistory).where(ChatHistory.msg_id == msg_id).values(vectorized=True)
    )
