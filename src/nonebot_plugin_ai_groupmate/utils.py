import io
import base64
import asyncio
import hashlib
import traceback
from datetime import timedelta
from functools import lru_cache

import tiktoken
from PIL import Image
from tqdm import tqdm
from nonebot import logger
from sqlalchemy import Select, Update
from nonebot_plugin_orm import AsyncSession

from .model import ChatHistory, ChatHistorySchema
from .memory import DB


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
    # 1. 格式标准化
    if image_format.upper() in ["JPG", "JPEG"]:
        image_format = "JPEG"
    elif image_format.upper() == "PNG":
        image_format = "PNG"

    # 2. 初始检查
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    file_size = len(image_bytes)

    if file_size <= max_size_bytes:
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))

        # JPEG 不支持透明通道 (RGBA)，必须转 RGB
        if image_format == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        quality = quality_start
        compressed_image = io.BytesIO()
        compressed_size = file_size  # 初始默认为原大小，或者设为无限大也可以

        # 3. 尝试降低质量压缩 (Quality Loop)
        # 注意：PNG 格式忽略 quality 参数，会直接跳过或无效循环
        if image_format != "PNG":
            while quality >= 10:
                compressed_image.seek(0)
                compressed_image.truncate(0)

                img.save(
                    compressed_image,
                    format=image_format,
                    quality=quality,
                    optimize=True
                )

                compressed_size = compressed_image.tell()

                if compressed_size <= max_size_bytes:
                    logger.info(
                        f"图片通过降质压缩成功 (Q={quality}): {file_size / 1024 / 1024:.2f}MB -> {compressed_size / 1024 / 1024:.2f}MB")
                    return compressed_image.getvalue()

                quality -= 15


        # 4. 尝试缩小尺寸 (Resize)
        if compressed_size > max_size_bytes:
            logger.info("降质无法满足要求(或为PNG)，开始缩小尺寸...")
            current_size = compressed_size if compressed_size < file_size else file_size

            width, height = img.size
            # 计算缩放比例
            ratio = min((max_size_bytes / current_size) ** 0.5, 0.9)

            new_width = max(int(width * ratio), 1)
            new_height = max(int(height * ratio), 1)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            compressed_image.seek(0)
            compressed_image.truncate(0)

            # 使用最后一次有效的 quality (防止负数)
            final_quality = max(quality, 75)

            img.save(
                compressed_image,
                format=image_format,
                quality=final_quality,
                optimize=True
            )

        final_data = compressed_image.getvalue()
        logger.info(
            f"图片最终压缩: {file_size / 1024 / 1024:.2f}MB -> {len(final_data) / 1024 / 1024:.2f}MB"
        )
        return final_data

    except Exception as e:
        logger.error(f"压缩图片时出错: {e}")
        # 出错时降级策略：返回原图
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
        max_token_count: int = 700,
        max_messages: int = 50,
) -> list[list[ChatHistory]]:
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
    # ✅ 转成 Pydantic 模型（一次性完全脱离数据库）
    all_messages = [ChatHistorySchema.model_validate(m) for m in all_messages]


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

# 缓存 encoder，避免重复加载
@lru_cache
def get_encoder():
    return tiktoken.get_encoding("cl100k_base")



def estimate_token_count(text: str) -> int:
    try:
        encoder = get_encoder()
        return len(encoder.encode(text))
    except Exception as e:
        logger.warning(f"Tiktoken 计算失败，回退到字符估算: {e}")
        return len(text)  # 最后的保底


async def process_and_vectorize_session_chats(
        db_session: AsyncSession,
        session_id: str,
        max_time_gap: timedelta = timedelta(hours=1),
        max_token_count: int = 1000,
        chunk_size: int = 100,
        commit_interval: int = 500,
) -> dict | None:
    """
    处理并向量化一个会话内的聊天记录，按上下文智能切分

    参数:
        db_session: 数据库会话
        session_id: 会话ID
        max_time_gap: 时间间隔切分标准
        max_token_count: token数量切分标准
        chunk_size: 每批次向量化的数量
        commit_interval: 数据库提交间隔（消息数）
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

    processed_groups = 0
    failed_groups = 0
    total_groups = len(context_groups)

    # 2. 分块处理，避免一次性处理过多数据
    for i in tqdm(range(0, total_groups, chunk_size), desc="向量化处理"):
        chunk = context_groups[i:i + chunk_size]

        # 准备当前批次的数据
        batch_contexts = []
        batch_msg_ids = []

        for group in chunk:
            context_text, msg_ids = combine_messages_into_context(group)
            batch_contexts.append(context_text)
            batch_msg_ids.extend(msg_ids)

        if not batch_contexts:
            continue

        # 3. 批量插入向量到 Milvus（带重试）
        try:
            await insert_vectors_with_retry(batch_contexts, session_id)
        except Exception as e:
            logger.error(
                f"批量向量化失败 (chunk {i}-{i + chunk_size}): {str(e)}\n{traceback.format_exc()}"
            )
            failed_groups += len(chunk)
            continue

        # 4. 分批更新数据库状态，避免单次更新过多
        try:
            await update_messages_in_batches(
                db_session,
                batch_msg_ids,
                commit_interval
            )
            processed_groups += len(chunk)
        except Exception as e:
            logger.error(
                f"批量更新数据库失败 (chunk {i}-{i + chunk_size}): {str(e)}\n{traceback.format_exc()}"
            )
            failed_groups += len(chunk)
            # 发生错误时回滚当前事务
            await db_session.rollback()

    # 5. 最终提交
    try:
        await db_session.commit()
    except Exception as e:
        logger.error(f"最终提交失败: {str(e)}")
        await db_session.rollback()

    return {
        "session_id": session_id,
        "processed_groups": processed_groups,
        "failed_groups": failed_groups,
        "total_groups": total_groups,
        "success_rate": f"{processed_groups / total_groups * 100:.2f}%" if total_groups > 0 else "0%"
    }


def combine_messages_into_context(
        messages: list[ChatHistory]
) -> tuple[str, list[int]]:
    context_parts = []
    msg_ids = []

    for msg in messages:
        # 确保所有关系和延迟加载的字段都已加载
        msg_id = msg.msg_id
        sender_name = msg.user_name
        content = msg.content
        created_at = msg.created_at

        # 格式化时间（现在所有数据都已加载到内存）
        time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")

        # 组合消息
        context_parts.append(f"[{time_str}] {sender_name}: {content}")
        msg_ids.append(msg_id)

    return "\n".join(context_parts), msg_ids


async def insert_vectors_with_retry(
        contexts: list[str],
        session_id: str,
        max_retries: int = 3
) -> None:
    """
    带重试机制的向量插入
    """
    for attempt in range(max_retries):
        try:
            await DB.batch_insert(contexts, session_id)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                f"向量插入失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            )
            await asyncio.sleep(2 ** attempt)


async def update_messages_in_batches(
        db_session: AsyncSession,
        msg_ids: list[int],
        batch_size: int = 500
) -> None:
    """
    分批更新消息状态
    """
    total_updated = 0

    for i in range(0, len(msg_ids), batch_size):
        batch_ids = msg_ids[i:i + batch_size]

        try:
            await mark_messages_as_vectorized_batch(db_session, batch_ids)
            await db_session.commit()
            total_updated += len(batch_ids)
            logger.debug(f"已更新 {total_updated}/{len(msg_ids)} 条消息")

        except Exception as e:
            logger.error(f"更新批次失败: {str(e)}")
            await db_session.rollback()
            await update_messages_one_by_one(db_session, batch_ids)


async def update_messages_one_by_one(
        db_session: AsyncSession,
        msg_ids: list[int]
) -> None:
    """
    逐条更新（降级方案）
    """
    for msg_id in msg_ids:
        try:
            await mark_messages_as_vectorized_batch(db_session, [msg_id])
            await db_session.commit()
        except Exception as e:
            logger.error(f"更新单条消息失败 {msg_id}: {str(e)}")
            await db_session.rollback()


async def mark_message_as_vectorized(db_session: AsyncSession, msg_id: int):
    """
    将消息标记为已向量化
    """
    await db_session.execute(
        Update(ChatHistory).where(ChatHistory.msg_id == msg_id).values(vectorized=True)
    )


async def mark_messages_as_vectorized_batch(db_session: AsyncSession, msg_ids: list[int]):
    """批量更新消息向量化状态"""
    if not msg_ids:
        return

    # 使用SQLAlchemy的批量更新
    await db_session.execute(
        Update(ChatHistory)
        .where(ChatHistory.msg_id.in_(msg_ids))
        .values(vectorized=True)
    )
    await db_session.commit()
