import base64
import mimetypes
from typing import Any
from pathlib import Path

from sqlalchemy import Select
from nonebot.log import logger
from nonebot_plugin_uninfo import SceneType, QryItrface
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..model import ChatHistory, ChatHistorySchema


def get_image_data_uri(file_name: str, *, pic_dir: Path) -> str | None:
    file_path = pic_dir / file_name
    if not file_path.exists():
        return None

    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"读取图片失败 {file_name}: {e}")
        return None


def fallback_qq_avatar_url(user_id: str | None) -> str | None:
    if not user_id:
        return None
    uid = str(user_id).strip()
    if not uid.isdigit():
        return None
    return f"https://q1.qlogo.cn/g?b=qq&nk={uid}&s=100"


def user_display_name_from_history(
    history: list[ChatHistorySchema], user_id: str
) -> str:
    for msg in reversed(history):
        if msg.user_id == user_id and msg.user_name:
            return msg.user_name
    return user_id


async def build_avatar_context_messages(
    history: list[ChatHistorySchema],
    *,
    interface: QryItrface | None,
    session_id: str,
    current_user_id: str | None,
    current_user_name: str | None,
    max_users: int = 4,
    group_members: list[Any] | None = None,
) -> list[BaseMessage]:
    user_ids: list[str] = []
    if current_user_id:
        user_ids.append(str(current_user_id))
    for msg in reversed(history):
        if msg.content_type == "bot":
            continue
        uid = str(msg.user_id)
        if uid and uid not in user_ids:
            user_ids.append(uid)
        if len(user_ids) >= max_users:
            break

    if not user_ids:
        return []

    avatar_by_id: dict[str, str] = {}
    name_by_id: dict[str, str] = {}

    members = group_members
    if members is None and interface is not None:
        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
        except Exception as e:
            logger.warning(f"获取群友头像信息失败，降级使用可推导头像: {e}")
            members = None
    if members is not None:
        wanted = set(user_ids)
        for member in members:
            uid = str(member.id)
            if uid not in wanted:
                continue
            user = getattr(member, "user", None)
            avatar = getattr(user, "avatar", None)
            if avatar:
                avatar_by_id[uid] = str(avatar)
            member_name = (
                getattr(member, "nick", None)
                or getattr(user, "nick", None)
                or getattr(user, "name", None)
            )
            if member_name:
                name_by_id[uid] = str(member_name)

    content_parts: list[Any] = [
        {
            "type": "text",
            "text": (
                "【群友头像上下文】下面是最近发言者的头像，仅用于识别群友形象、头像梗和轻度玩笑。"
                "不要根据头像推断敏感身份、年龄、性别、种族、健康等属性。"
            ),
        }
    ]
    added = 0
    for uid in user_ids:
        avatar_url = avatar_by_id.get(uid) or fallback_qq_avatar_url(uid)
        if not avatar_url:
            continue
        if uid == str(current_user_id) and current_user_name:
            display_name = current_user_name
        else:
            display_name = name_by_id.get(uid) or user_display_name_from_history(
                history, uid
            )
        content_parts.append(
            {"type": "text", "text": f"\n{display_name}({uid}) 的头像："}
        )
        content_parts.append({"type": "image_url", "image_url": {"url": avatar_url}})
        added += 1

    if added == 0:
        return []
    return [HumanMessage(content_parts)]  # type: ignore[arg-type]


def should_include_avatar_context(
    history: list[ChatHistorySchema], limit: int = 4
) -> bool:
    avatar_keywords = (
        "头像",
        "头图",
        "头像框",
        "看我头",
        "看下我头",
        "看看我头",
        "你看我头",
        "avatar",
        "pfp",
    )
    checked = 0
    for msg in reversed(history):
        if msg.content_type == "bot":
            continue
        _, _, body = parse_msg_meta(msg.content)
        text = body.lower()
        if any(keyword in text for keyword in avatar_keywords):
            return True
        checked += 1
        if checked >= limit:
            break
    return False


def parse_msg_meta(content: str) -> tuple[str | None, str | None, str]:
    lines = content.splitlines()
    if not lines:
        return None, None, ""

    own_id: str | None = None
    reply_to_id: str | None = None
    body_start = 0

    if lines[0].startswith("id:"):
        own_id = lines[0].split(":", 1)[1].strip()
        body_start = 1

        if len(lines) > 1 and lines[1].startswith("回复id:"):
            reply_to_id = lines[1].split(":", 1)[1].strip()
            body_start = 2

    body = "\n".join(lines[body_start:]).strip()
    return own_id, reply_to_id, body


def image_file_name_from_history(msg: ChatHistorySchema) -> str:
    for line in reversed(msg.content.strip().splitlines()):
        line = line.strip()
        if line.startswith("图片文件:"):
            return line.split(":", 1)[1].strip()
    return msg.content.strip().split("\n")[-1].strip()


def is_image_history(msg: ChatHistorySchema) -> bool:
    return msg.content_type == "image" or (
        msg.content_type == "bot" and msg.media_id is not None
    )


async def load_replied_message_histories(
    db_session: AsyncSession,
    session_id: str,
    reply_to_id: str | None,
) -> list[ChatHistorySchema]:
    if not reply_to_id:
        return []

    normalized_reply_id = str(reply_to_id).strip()
    if not normalized_reply_id:
        return []

    for marker in (f"id: {normalized_reply_id}\n", f"id:{normalized_reply_id}\n"):
        rows = (
            (
                await db_session.execute(
                    Select(ChatHistory)
                    .where(
                        ChatHistory.session_id == session_id,
                        ChatHistory.content.contains(marker),
                    )
                    .order_by(ChatHistory.msg_id.asc())
                )
            )
            .scalars()
            .all()
        )
        if rows:
            history_msg_ids = ", ".join(str(msg.msg_id) for msg in rows)
            logger.info(
                f"命中被回复消息记录 reply_id={normalized_reply_id} history_msg_ids={history_msg_ids}"
            )
            return [ChatHistorySchema.model_validate(msg) for msg in rows]

    return []


def format_chat_history(
    history: list[ChatHistorySchema],
    *,
    pic_dir: Path,
    bot_name: str,
    max_inline_images: int = 3,
    user_roles: dict[str, str] | None = None,
    extra_inline_images: list[ChatHistorySchema] | None = None,
) -> list[BaseMessage]:
    messages = []
    user_roles = user_roles or {}
    extra_inline_images = extra_inline_images or []

    def role_prefix(uid: str) -> str:
        role = user_roles.get(uid)
        if role == "owner":
            return "[群主] "
        if role == "admin":
            return "[管理员] "
        return ""

    id_to_summary: dict[str, str] = {}
    for msg in [*history, *extra_inline_images]:
        own_id, _, body = parse_msg_meta(msg.content)
        if own_id:
            if is_image_history(msg):
                snippet = "[图片]"
            else:
                snippet = body[:30] + ("…" if len(body) > 30 else "")
            id_to_summary[own_id] = f'{msg.user_name} "{snippet}"'

    has_extra_inline_images = any(
        is_image_history(msg) for msg in extra_inline_images
    )

    image_indices = [
        i
        for i, msg in enumerate(history)
        if is_image_history(msg) and msg.content_type != "bot"
    ]
    if has_extra_inline_images or max_inline_images <= 0:
        inline_image_set = set()
    else:
        inline_image_set = set(image_indices[-max_inline_images:])

    text_to_images: dict[str, tuple[int, list[int]]] = {}
    for i in range(len(history)):
        own_id, _, _ = parse_msg_meta(history[i].content)
        if not own_id:
            continue
        if history[i].content_type == "text":
            img_idxs: list[int] = []
            for j in range(i + 1, len(history)):
                next_own_id, _, _ = parse_msg_meta(history[j].content)
                if (
                    next_own_id == own_id
                    and is_image_history(history[j])
                    and history[j].user_id == history[i].user_id
                ):
                    img_idxs.append(j)
                else:
                    break
            if img_idxs:
                text_to_images[own_id] = (i, img_idxs)

    merged_image_indices: set[int] = set()
    for _tid, (_, img_idxs) in text_to_images.items():
        merged_image_indices.update(img_idxs)

    for idx, msg in enumerate(history):
        if idx in merged_image_indices:
            continue

        time_str = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        own_id, reply_to_id, body = parse_msg_meta(msg.content)

        if reply_to_id and reply_to_id in id_to_summary:
            reply_prefix = f"(回复 {id_to_summary[reply_to_id]}) "
        elif reply_to_id:
            reply_prefix = "(回复了一条消息) "
        else:
            reply_prefix = ""

        if msg.content_type == "text" and own_id and own_id in text_to_images:
            prefix = role_prefix(msg.user_id)
            text_line = f"[{time_str}] {prefix}{msg.user_name}: {reply_prefix}{body}"

            merged_inline_images: list[int] = [
                i for i in text_to_images[own_id][1] if i in inline_image_set
            ]
            merged_fallback_images: list[int] = [
                i for i in text_to_images[own_id][1] if i not in inline_image_set
            ]

            if not merged_inline_images:
                for _ in merged_fallback_images:
                    text_line += " [图片]"
                messages.append(HumanMessage(content=text_line))
                continue

            content_parts: list[Any] = [{"type": "text", "text": text_line}]
            for img_idx in merged_inline_images:
                img_msg = history[img_idx]
                file_name = image_file_name_from_history(img_msg)
                image_data = get_image_data_uri(file_name, pic_dir=pic_dir)
                if image_data:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": image_data}}
                    )
                else:
                    text_line += " [图片已过期]"
                    content_parts[0]["text"] = text_line

            for _ in merged_fallback_images:
                content_parts[0]["text"] += " [图片]"

            messages.append(HumanMessage(content_parts))  # type: ignore[arg-type]
            continue

        if msg.content_type == "text":
            prefix = role_prefix(msg.user_id)
            content = f"[{time_str}] {prefix}{msg.user_name}: {reply_prefix}{body}"
            messages.append(HumanMessage(content=content))
            continue

        if msg.content_type == "bot" and not is_image_history(msg):
            messages.append(AIMessage(content=body or msg.content))
            continue

        if is_image_history(msg):
            file_name = image_file_name_from_history(msg)
            is_bot_image = msg.content_type == "bot"

            if idx in inline_image_set:
                image_data = get_image_data_uri(file_name, pic_dir=pic_dir)
                if image_data:
                    prefix = role_prefix(msg.user_id)
                    text = (
                        f"[{time_str}] {bot_name} {reply_prefix}发送了一张图片："
                        if is_bot_image
                        else f"[{time_str}] {prefix}{msg.user_name} {reply_prefix}发送了一张图片："
                    )
                    content_parts = [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]
                    message_cls = AIMessage if is_bot_image else HumanMessage
                    messages.append(message_cls(content_parts))  # type: ignore[arg-type]
                else:
                    prefix = role_prefix(msg.user_id)
                    content = (
                        f"[{time_str}] {bot_name} {reply_prefix}[图片已过期/无法加载]"
                        if is_bot_image
                        else f"[{time_str}] {prefix}{msg.user_name} {reply_prefix}[图片已过期/无法加载]"
                    )
                    message_cls = AIMessage if is_bot_image else HumanMessage
                    messages.append(message_cls(content=content))
            else:
                prefix = role_prefix(msg.user_id)
                content = (
                    f"[{time_str}] {bot_name} {reply_prefix}[图片]"
                    if is_bot_image
                    else f"[{time_str}] {prefix}{msg.user_name} {reply_prefix}[图片]"
                )
                message_cls = AIMessage if is_bot_image else HumanMessage
                messages.append(message_cls(content=content))
            continue

    return messages
