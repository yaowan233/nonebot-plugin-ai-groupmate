import io
import base64
import json
from typing import Any
from pathlib import Path

from PIL import Image, ImageOps
from nonebot.log import logger
from nonebot_plugin_uninfo import SceneType, QryItrface
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..model import ChatHistorySchema
from .file_workspace import FileWorkspace
from .image_resources import ImageResources, ImageResourceError, is_image_history as is_image_history, published_image_file_id as published_image_file_id, image_file_name_from_history as image_file_name_from_history
from ..message_mentions import format_member_mention
from ..message_metadata import platform_message_ids, deduplicate_bot_echoes, parse_message_metadata, is_reliable_platform_id
from .structured_context import StructuredSnapshot, visible_quote_text
from .turn_context import TurnHistoryResolver
from ..message_repository import find_by_platform_id
from ..media_message import format_media_markers
from ..forward_message import format_forward_marker

BOT_IMAGE_DESCRIPTION_MAX_CHARS = 500


def _image_bytes_to_data_uri(image_bytes: bytes, *, source: str) -> str | None:
    """Decode and normalize an image before sending it to an OpenAI-compatible API."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.seek(0)
            image.load()
            has_transparency = image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            )
            normalized = ImageOps.exif_transpose(image)
            output = io.BytesIO()
            if has_transparency:
                normalized.convert("RGBA").save(output, format="PNG", optimize=True)
                mime_type = "image/png"
            else:
                normalized.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=90,
                    optimize=True,
                )
                mime_type = "image/jpeg"
    except Exception as e:
        logger.warning(f"图片无法解码，跳过模型图片输入 source={source}: {e}")
        return None

    encoded_string = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_string}"


def get_image_data_uri(file_name: str, *, pic_dir: Path) -> str | None:
    try:
        data = ImageResources(pic_dir).read_media(file_name)
        return _image_bytes_to_data_uri(data, source=file_name)
    except ImageResourceError:
        return None


def get_history_image_data_uri(msg: ChatHistorySchema, *, pic_dir: Path, workspace: FileWorkspace | None = None) -> str | None:
    """Read either an ordinary media image or a confirmed, scoped bot publication."""
    try:
        image_bytes = ImageResources(pic_dir, workspace).read_history(msg)
    except ImageResourceError:
        return None
    return _image_bytes_to_data_uri(image_bytes, source="history_image")


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
    history = deduplicate_bot_echoes(history)
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
    for msg in reversed(deduplicate_bot_echoes(history)):
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
    return parse_message_metadata(content)


def _bot_image_description(msg: ChatHistorySchema) -> str:
    """Recover the stored send-time caption without exposing transport metadata."""
    if msg.content_type != "bot":
        return ""
    _, _, body = parse_msg_meta(msg.content)
    if msg.media_id is None and published_image_file_id(msg) is not None:
        for line in body.splitlines():
            if line.startswith("搜索描述:") and (description := line.partition(":")[2].strip()):
                return "搜索描述（未必准确）：" + description[:BOT_IMAGE_DESCRIPTION_MAX_CHARS]
        return ""
    if msg.media_id is None:
        return ""
    for prefix in ("发送了图片，图片描述是:", "发送了图片，图片描述是："):
        if body.startswith(prefix):
            description = body[len(prefix):].rsplit("\n图片文件:", 1)[0].strip()
            if not description or description == "[图片]":
                return ""
            if len(description) > BOT_IMAGE_DESCRIPTION_MAX_CHARS:
                return description[:BOT_IMAGE_DESCRIPTION_MAX_CHARS] + "…"
            return description
    return ""


def current_message_images(
    history: list[ChatHistorySchema],
) -> list[ChatHistorySchema]:
    """返回历史中最后一条（当前）消息携带的图片记录。"""
    history = deduplicate_bot_echoes(history)
    if not history:
        return []
    last_id, _, _ = parse_msg_meta(history[-1].content)
    if not last_id:
        return []
    images: list[ChatHistorySchema] = []
    for msg in reversed(history):
        own_id, _, _ = parse_msg_meta(msg.content)
        if own_id != last_id:
            break
        if msg.content_type == "image":
            images.append(msg)
    images.reverse()
    return images


async def load_replied_message_histories(
    db_session: AsyncSession,
    session_id: str,
    reply_to_id: str | None,
) -> list[ChatHistorySchema]:
    rows = await find_by_platform_id(db_session, session_id, reply_to_id)
    return [ChatHistorySchema.model_validate(msg) for msg in rows]


def _match_bot_image_rows(snapshot: StructuredSnapshot, indices: list[int], history: list[ChatHistorySchema]) -> list[int] | None:
    """Match each structured image to its exact legacy row before reading it."""
    parts = [part for part in snapshot.parts if part.get("type") == "image"]
    available = [index for index in indices if is_image_history(history[index])]
    if len(parts) != len(available):
        return None
    matched = []
    for part in parts:
        file_id = part.get("file_id", "")
        media_id = part.get("media_id", "")
        if not file_id and not media_id:
            return None
        if media_id and not media_id.isdigit():
            return None
        index = next((i for i in available if
                      (not file_id or published_image_file_id(history[i]) == file_id) and
                      (not media_id or history[i].media_id == int(media_id))), None)
        if index is None:
            return None
        available.remove(index)
        matched.append(index)
    return matched


def format_recent_message_references(
    history: list[ChatHistorySchema],
    *,
    session_id: str,
    bot_name: str,
    structured_messages: dict[tuple[str, str], StructuredSnapshot] | None = None,
    message_limit: int = 12,
    image_limit: int = 5,
) -> str:
    """Expose bounded, scoped transport IDs for the messages visible in this turn."""
    if message_limit <= 0 and image_limit <= 0:
        return ""
    resolver = TurnHistoryResolver(structured_messages)
    records: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    ambiguous_ids: set[str] = set()
    kinds_by_row: dict[tuple[int, str], set[str]] = {}
    image_counts_by_row: dict[tuple[int, str], int] = {}
    structured_image_counts: dict[str, int] = {}
    for (_, sender_id), snapshot in (structured_messages or {}).items():
        kinds = {"audio" if part.get("type") == "record" else part.get("type", "") for part in snapshot.parts}
        kinds.intersection_update({"text", "image", "audio", "video", "file", "forward"})
        image_count = sum(part.get("type") == "image" for part in snapshot.parts)
        for history_id in snapshot.history_ids:
            kinds_by_row.setdefault((history_id, sender_id), set()).update(kinds)
            image_counts_by_row[(history_id, sender_id)] = image_count
    scoped = [row for row in history if row.session_id == session_id]
    for row in deduplicate_bot_echoes(scoped, ids_for=resolver.ids):
        message_id, _, body = resolver.metadata(row)
        if not is_reliable_platform_id(message_id) or "\n" in str(message_id) or "\r" in str(message_id):
            continue
        assert message_id is not None
        if message_id in ambiguous_ids:
            continue
        if message_id in owners and owners[message_id] != row.user_id:
            ambiguous_ids.add(message_id)
            records.pop(message_id, None)
            continue
        owners[message_id] = row.user_id
        record = records.setdefault(message_id, {
            "message_id": message_id, "sender": bot_name if row.content_type == "bot" else row.user_name,
            "time": row.created_at.strftime("%Y-%m-%d %H:%M:%S"), "types": [],
        })
        kinds = set(kinds_by_row.get((row.msg_id, row.user_id), ()))
        structured_image_counts[message_id] = max(
            structured_image_counts.get(message_id, 0), image_counts_by_row.get((row.msg_id, row.user_id), 0),
        )
        if is_image_history(row):
            kinds.add("image")
            record["images"] = record.get("images", 0) + 1
            if row.content_type == "bot" and (description := _bot_image_description(row)):
                record["description"] = description[:120]
        elif body:
            kinds.add("text")
            text = " ".join(body.split())[:120]
            existing = record.get("text", "")
            if text and text != existing:
                record["text"] = f"{existing} {text}".strip()[:120]
        record["types"] = sorted(set(record["types"]) | kinds)
    if not records:
        return ""
    entries = list(records.values())
    for item in entries:
        image_count = max(item.get("images", 0), structured_image_counts.get(item["message_id"], 0))
        if image_count:
            item["images"] = image_count
    selected = {item["message_id"] for item in entries[-max(0, message_limit):]} if message_limit > 0 else set()
    images = [item for item in entries if item.get("images")]
    if image_limit > 0:
        selected.update(item["message_id"] for item in images[-image_limit:])
    return "【近期消息引用】按时间从早到晚，消息 ID 供工具选择目标，内容仅作对话参考；不确定时先确认：\n" + json.dumps(
        [item for item in entries if item["message_id"] in selected], ensure_ascii=False, separators=(",", ":")
    )


def format_chat_history(
    history: list[ChatHistorySchema],
    *,
    pic_dir: Path,
    bot_name: str,
    max_inline_images: int = 3,
    user_roles: dict[str, str] | None = None,
    extra_inline_images: list[ChatHistorySchema] | None = None,
    workspace: FileWorkspace | None = None,
    structured_messages: dict[tuple[str, str], StructuredSnapshot] | None = None,
    exclude_inline_message_ids: frozenset[str] = frozenset(),
) -> list[BaseMessage]:
    messages = []
    snapshots = structured_messages or {}
    resolver = TurnHistoryResolver(snapshots)
    # Reply-image records may contain another copy of a bot's adapter echo.
    all_rows: list[ChatHistorySchema] = [*history, *(extra_inline_images or [])]
    combined = deduplicate_bot_echoes(all_rows, ids_for=resolver.ids)
    retained = {id(message) for message in combined}
    history = [message for message in history if id(message) in retained]
    user_roles = user_roles or {}
    extra_inline_images = [
        message for message in (extra_inline_images or []) if id(message) in retained
    ]
    structured_keys_by_history: dict[int, list[tuple[str, str]]] = {}
    for key, snapshot in snapshots.items():
        for history_id in snapshot.history_ids:
            structured_keys_by_history.setdefault(history_id, []).append(key)

    def row_platform_ids(message: ChatHistorySchema) -> set[str]:
        indexed_keys = structured_keys_by_history.get(message.msg_id)
        if indexed_keys is not None:
            ids: set[str] = set()
            for key in indexed_keys:
                if key[1] == message.user_id:
                    ids.update(snapshots[key].history_platform_ids.get(message.msg_id, (key[0],)))
            return ids
        return set(platform_message_ids(message.content))

    def row_own_id(message: ChatHistorySchema) -> str | None:
        indexed_keys = structured_keys_by_history.get(message.msg_id)
        if indexed_keys is not None:
            return next((key[0] for key in sorted(indexed_keys) if key[1] == message.user_id), None)
        return parse_msg_meta(message.content)[0]

    def role_prefix(uid: str) -> str:
        role = user_roles.get(uid)
        if role == "owner":
            return "[群主] "
        if role == "admin":
            return "[管理员] "
        return ""

    id_to_summary: dict[str, str] = {}
    for msg in [*history, *extra_inline_images]:
        own_id, _, body = resolver.metadata(msg)
        summary_ids = resolver.ids(msg) or ({own_id} if own_id else set())
        if summary_ids:
            if is_image_history(msg):
                snippet = "[图片]"
            else:
                snippet = body[:30] + ("…" if len(body) > 30 else "")
            for summary_id in summary_ids:
                id_to_summary[summary_id] = f'{msg.user_name} "{snippet}"'

    has_extra_inline_images = any(
        is_image_history(msg) for msg in extra_inline_images
    )

    image_indices = [
        i
        for i, msg in enumerate(history)
        if is_image_history(msg) and not (row_platform_ids(msg) & exclude_inline_message_ids)
    ]
    if has_extra_inline_images or max_inline_images <= 0:
        inline_image_set = set()
    else:
        inline_image_set = set(image_indices[-max_inline_images:])

    text_to_images: dict[str, tuple[int, list[int]]] = {}
    for i in range(len(history)):
        own_id = row_own_id(history[i])
        if not own_id:
            continue
        if history[i].content_type == "text":
            img_idxs: list[int] = []
            for j in range(i + 1, len(history)):
                next_own_id = row_own_id(history[j])
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

    # Image downloads can finish after later messages. Anchor a structured
    # message at its first compatibility row, even when its rows are separated.
    structured_anchors: dict[int, tuple[StructuredSnapshot, list[int]]] = {}
    structured_bot_images: dict[int, list[int]] = {}
    structured_skip: set[int] = set()
    grouped: dict[tuple[str, str], list[int]] = {}
    if structured_messages:
        for i, msg in enumerate(history):
            indexed_keys = structured_keys_by_history.get(msg.msg_id)
            keys = (sorted(indexed_keys) if indexed_keys is not None else
                    [(message_id, msg.user_id) for message_id in sorted(platform_message_ids(msg.content))])
            for key in keys:
                if key[1] != msg.user_id:
                    continue
                if key not in structured_messages:
                    continue
                if msg.content_type == "bot":
                    if structured_messages[key].parts and all(part.get("type") in {"text", "mention", "image"}
                                                              for part in structured_messages[key].parts):
                        grouped.setdefault(key, []).append(i)
                        break
                    continue
                if (structured_messages[key].parts and visible_quote_text(structured_messages[key]) is not None and all(
                    part.get("type") in {"text", "mention", "image", "face", "mface", "card", "file", "audio", "record", "video", "forward"} and
                    (part.get("type") != "file" or (part.get("file_id") and part.get("name"))) and
                    (part.get("type") != "image" or (part.get("media_id", "").isdigit() and
                     int(part["media_id"]) in structured_messages[key].media_paths))
                    for part in structured_messages[key].parts
                )):
                    grouped.setdefault(key, []).append(i)
                    break
        for key, indices in grouped.items():
            snapshot = structured_messages[key]
            contains_text = any((part.get("type") in {"text", "mention", "card", "file"} and
                                 (part.get("text") or part.get("target") or part.get("name"))) or
                                part.get("type") in {"audio", "record", "video", "forward"} for part in snapshot.parts)
            is_bot = history[indices[0]].content_type == "bot"
            if is_bot:
                bot_images = _match_bot_image_rows(snapshot, indices, history)
                has_text_row = any(not is_image_history(history[i]) for i in indices)
                if bot_images is None or contains_text != has_text_row:
                    continue
                structured_bot_images[indices[0]] = bot_images
            elif contains_text and not any(history[i].content_type == "text" for i in indices):
                # An image can arrive after this message was already cached in
                # an append-only thread. Do not replay its old text as new.
                continue
            structured_anchors[indices[0]] = (snapshot, indices)
            structured_skip.update(indices[1:])

    for idx, msg in enumerate(history):
        if idx in structured_skip or (idx in merged_image_indices and idx not in structured_anchors):
            continue

        time_str = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if idx in structured_anchors:
            snapshot = structured_anchors[idx][0]
            own_id, reply_to_id, body = snapshot.platform_id, snapshot.reply_to_id, ""
        else:
            own_id, reply_to_id, body = parse_msg_meta(msg.content)

        if reply_to_id and reply_to_id in id_to_summary:
            reply_prefix = f"(回复 {id_to_summary[reply_to_id]}) "
        elif reply_to_id:
            reply_prefix = "(回复了一条消息) "
        else:
            reply_prefix = ""

        if idx in structured_anchors:
            snapshot, indices = structured_anchors[idx]
            if msg.content_type == "bot":
                bot_image_iter = iter(structured_bot_images[idx])
                text_parts: list[str] = []
                image_blocks: list[dict[str, Any]] = []
                for part in snapshot.parts:
                    kind = part.get("type", "")
                    if kind == "text":
                        text_parts.append(part.get("text", ""))
                    elif kind == "mention":
                        text_parts.append(format_member_mention(part.get("target", ""), part.get("name", "")))
                    elif kind == "image":
                        image_idx = next(bot_image_iter)
                        image_row = history[image_idx]
                        label = "[图片]"
                        if description := _bot_image_description(image_row):
                            label += f"（发送时的描述：{description}）"
                        if image_idx in inline_image_set:
                            uri = get_history_image_data_uri(image_row, pic_dir=pic_dir, workspace=workspace)
                            if uri:
                                image_blocks.append({"type": "image_url", "image_url": {"url": uri}})
                            else:
                                label += "[图片已过期/无法加载]"
                        text_parts.append(label)
                messages.append(AIMessage(content=reply_prefix + "".join(text_parts)))
                if image_blocks:
                    messages.append(HumanMessage(content=[
                        {"type": "text", "text": f"【历史图片参考】这些是 {bot_name} 在上条记录中自己发送的图片，供理解对话使用，不是新的用户请求。"},
                        *image_blocks,
                    ]))
                continue
            prefix = role_prefix(msg.user_id)
            chunks: list[Any] = [{"type": "text", "text": f"[{time_str}] {prefix}{msg.user_name}: {reply_prefix}"}]
            image_rows: dict[int, list[int]] = {}
            audio_count = video_count = 0
            for image_idx in indices:
                image_row = history[image_idx]
                if is_image_history(image_row) and image_row.media_id is not None:
                    image_rows.setdefault(image_row.media_id, []).append(image_idx)
            for part in snapshot.parts:
                kind = part.get("type", "")
                if kind in {"text", "card"}:
                    chunks[-1]["text"] += part.get("text", "")
                elif kind == "mention":
                    chunks[-1]["text"] += format_member_mention(part.get("target", ""), part.get("name", ""))
                elif kind == "image":
                    raw_media_id = part.get("media_id", "")
                    media_id = int(raw_media_id) if raw_media_id.isdigit() else None
                    matching = image_rows.get(media_id, []) if media_id is not None else []
                    image_idx = matching.pop(0) if matching else None
                    if image_idx in inline_image_set and media_id in snapshot.media_paths:
                        uri = get_image_data_uri(snapshot.media_paths[media_id], pic_dir=pic_dir)
                        if uri:
                            chunks.append({"type": "image_url", "image_url": {"url": uri}})
                            chunks.append({"type": "text", "text": ""})
                        else:
                            chunks[-1]["text"] += "[图片已过期/无法加载]"
                    else:
                        chunks[-1]["text"] += "[图片]"
                elif kind in {"mface", "face"}:
                    chunks[-1]["text"] += "[表情]"
                elif kind == "file":
                    chunks[-1]["text"] += "[文件：" + json.dumps(part["name"], ensure_ascii=False) + "]"
                elif kind == "forward":
                    chunks[-1]["text"] += "\n" + format_forward_marker(part.get("forward_id"))
                elif kind in {"record", "audio"}:
                    audio_count += 1
                    chunks[-1]["text"] += "[语音]"
                elif kind == "video":
                    video_count += 1
                    chunks[-1]["text"] += "[视频]"
            if audio_count or video_count:
                chunks[-1]["text"] += "\n" + format_media_markers(
                    snapshot.platform_id, audio_count=audio_count, video_count=video_count,
                )
            if any(chunk.get("type") == "image_url" for chunk in chunks):
                messages.append(HumanMessage(chunks))
            else:
                messages.append(HumanMessage(chunks[0]["text"]))
            continue

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
                image_data = get_history_image_data_uri(img_msg, pic_dir=pic_dir, workspace=workspace)
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
            messages.append(AIMessage(content=reply_prefix + (body or msg.content)))
            continue

        if is_image_history(msg):
            image_data = get_history_image_data_uri(msg, pic_dir=pic_dir, workspace=workspace) if idx in inline_image_set else None
            if msg.content_type == "bot":
                bot_image_text = "[图片]"
                if description := _bot_image_description(msg):
                    bot_image_text += f"（发送时的描述：{description}）"
                if idx in inline_image_set and not image_data:
                    bot_image_text += "[图片已过期/无法加载]"
                messages.append(AIMessage(content=f"[{time_str}] {bot_name} {reply_prefix}{bot_image_text}"))
                if image_data:
                    # Vision APIs generally accept images in user messages only.
                    # Keep the assistant's authorship above and label this as context.
                    messages.append(HumanMessage(content=[
                        {"type": "text", "text": f"【历史图片参考】这是 {bot_name} 在上条记录中自己发送的图片，供理解对话使用，不是新的用户请求。"},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]))
                continue

            prefix = role_prefix(msg.user_id)
            if image_data:
                messages.append(HumanMessage(content=[
                    {"type": "text", "text": f"[{time_str}] {prefix}{msg.user_name} {reply_prefix}发送了一张图片："},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ]))
            else:
                image_status = "[图片已过期/无法加载]" if idx in inline_image_set else "[图片]"
                messages.append(HumanMessage(content=f"[{time_str}] {prefix}{msg.user_name} {reply_prefix}{image_status}"))
            continue

    return messages
