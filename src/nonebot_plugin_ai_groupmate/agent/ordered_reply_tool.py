"""Prepare and deliver one ordered text/image/mention message."""

import json
import uuid
import asyncio
import hashlib
from typing import Any, Literal
from pathlib import Path

from sqlalchemy import Select, update
from nonebot.log import logger
from langchain_core.tools import tool
from nonebot_plugin_uninfo import SceneType

from ..model import DrawingJob, ChatHistory, MediaStorage, OrderedDeliveryJob
from .delivery import PublishedImage, send_and_finalize
from .reply_tools import resolve_reply_target
from ..reply_guard import is_request_active
from .file_content import image_preview
from .tool_results import tool_failure, tool_skipped, tool_success
from ..drawing_jobs import utcnow
from .outbound_plan import ReplyPlan, compose_reply
from .file_workspace import FileTaskError


def create_ordered_reply_tool(db_session, session_id, request_id, *, workspace, pic_dir: Path,
                              approved_meme_ids: set[int], send_target, bot_name: str,
                              members: list[Any] | None, current_message_id: str | None,
                              replied_message_id: str | None, outbound_identity, recovery_enabled: bool, interface=None):
    attempted: set[str] = set()
    prepared_memes: set[str] = set()

    @tool("prepare_meme_image")
    async def prepare_meme_image(pic_id: str) -> str:
        """将本轮搜索审核通过的表情包准备为图片资源，供 send_ordered_message 使用；不会发送。"""
        if request_id and not await is_request_active(session_id, request_id):
            return tool_skipped("request_expired", "请求已过期。")
        try:
            if not pic_id.isdigit() or int(pic_id) not in approved_meme_ids:
                return tool_failure("candidate_not_approved", "请先搜索并选择本轮通过审核的表情包。")
            pic = await db_session.scalar(Select(MediaStorage).where(MediaStorage.media_id == int(pic_id)))
            if pic is None:
                return tool_failure("image_not_available", "表情包不可用。")
            path = (pic_dir / pic.file_path).resolve()
            await db_session.commit()
            if not path.is_relative_to(pic_dir.resolve()) or path.stat().st_size > 10 * 1024 * 1024:
                return tool_failure("image_not_available", "表情包不可用或过大。")
            data = await asyncio.to_thread(path.read_bytes)
            await asyncio.to_thread(image_preview, data, 0)
            artifact = await asyncio.to_thread(workspace.save, data, path.name)
            prepared_memes.add(artifact.file_id)
            return tool_success("meme_prepared", "表情包已准备好，尚未发送。",
                                data={"resource_id": "file:" + artifact.file_id, "pic_id": pic_id})
        except Exception as error:
            logger.warning(f"准备表情包失败 error_type={type(error).__name__}")
            return tool_failure("meme_prepare_failed", "表情包无法读取，未发送。")

    @tool("send_ordered_message")
    async def send_ordered_message(parts: list[dict[str, str]], reply_to: str | None = None,
                                   next_step: Literal["end", "continue"] = "end") -> str:
        """发送一条按顺序排列的图文消息。parts 每项为 text/text、image/resource_id 或 mention/user_id。
        图片用 file:<file_id>、meme:<pic_id> 或 drawing:<job_id>；表情包必须先搜索审核，绘图须先设置 defer_delivery=true。
        包含 drawing 的消息提交后台等待生成完成后一起发送。reply_to=current 可引用当前消息。
        所有图片必须准备好才发送；没有实际图片时不要用文字占位。"""
        started = False
        delivery_id = None
        try:
            plan = ReplyPlan.model_validate({"parts": parts, "reply_to": reply_to})
            meme_count = sum(part.type == "image" and (part.resource_id.startswith("meme:") or
                             part.resource_id.removeprefix("file:") in prepared_memes) for part in plan.parts)
            if meme_count > 5:
                raise ValueError("每条消息最多发送 5 张表情包")
            if request_id and not await is_request_active(session_id, request_id):
                return tool_skipped("request_expired", "请求已过期，未发送。", delivery_state="not_attempted")
            quoted = await resolve_reply_target(db_session, session_id, reply_to,
                    current_message_id=current_message_id, replied_message_id=replied_message_id,
                    outbound_identity=outbound_identity)
            resolved_members = members
            if resolved_members is None and interface is not None and any(part.type == "mention" for part in plan.parts):
                resolved_members = await interface.get_members(SceneType.GROUP, session_id)
            allowed_members = {str(member.id) for member in resolved_members or []}
            images: dict[str, bytes] = {}
            file_ids: list[str | None] = []
            media_ids: list[int | None] = []
            publications: list[PublishedImage] = []
            descriptions: dict[str, tuple[str, int | None]] = {}
            drawing_ids: set[str] = set()
            persisted_parts: list[dict[str, str]] = []
            total_bytes = 0
            has_drawing = any(part.type == "image" and part.resource_id.startswith("drawing:") for part in plan.parts)
            for part in plan.parts:
                if part.type == "mention":
                    if part.user_id not in allowed_members:
                        raise ValueError("无法确认 @ 的群成员 ID")
                if part.type != "image":
                    persisted_parts.append(part.model_dump(exclude_none=True))
                    continue
                resource_id = part.resource_id
                kind, sep, identifier = resource_id.partition(":")
                if not sep:
                    raise ValueError("图片资源标识格式无效")
                if kind == "drawing":
                    if len(identifier) != 32 or any(char not in "0123456789abcdef" for char in identifier):
                        raise ValueError("绘图任务 ID 无效")
                    drawing_ids.add(identifier)
                    persisted_parts.append(part.model_dump(exclude_none=True))
                    continue
                if kind == "file":
                    artifact = await asyncio.to_thread(workspace.get, identifier)
                    data = await asyncio.to_thread(workspace.read, identifier)
                    file_ids.append(artifact.file_id)
                    media_ids.append(None)
                    publications.append(PublishedImage(workspace, artifact.file_id))
                    descriptions[resource_id] = (f"[图片文件] {artifact.name} file_id={artifact.file_id}", None)
                elif kind == "meme" and identifier.isdigit():
                    media_id = int(identifier)
                    if media_id not in approved_meme_ids:
                        raise ValueError("表情包未通过本轮搜索审核")
                    pic = await db_session.scalar(Select(MediaStorage).where(MediaStorage.media_id == media_id))
                    if pic is None:
                        raise ValueError("表情包不存在")
                    path = (pic_dir / pic.file_path).resolve()
                    if not path.is_relative_to(pic_dir.resolve()) or path.stat().st_size > 10 * 1024 * 1024:
                        raise ValueError("表情包路径无效或图片过大")
                    data = await asyncio.to_thread(path.read_bytes)
                    file_ids.append(None)
                    media_ids.append(media_id)
                    descriptions[resource_id] = (f"发送了图片，图片描述是: {pic.description}\n图片文件: {pic.file_path}", media_id)
                else:
                    raise ValueError("图片资源类型无效")
                await asyncio.to_thread(image_preview, data, 0)
                total_bytes += len(data)
                if total_bytes > 30 * 1024 * 1024:
                    raise ValueError("整条消息图片超过 30 MiB")
                images[resource_id] = data
                if kind == "meme" and has_drawing:
                    artifact = await asyncio.to_thread(workspace.save, data, "表情包.png")
                    persisted_parts.append({"type": "image", "resource_id": "file:" + artifact.file_id})
                elif kind == "file" and has_drawing:
                    # Freeze peer-visible files in this job's own scope so the
                    # background worker can resolve them after reconnect/restart.
                    artifact = await asyncio.to_thread(workspace.save, data, artifact.name, content_type=artifact.content_type)
                    persisted_parts.append({"type": "image", "resource_id": "file:" + artifact.file_id})
                else:
                    persisted_parts.append(part.model_dump(exclude_none=True))
            fingerprint = hashlib.sha256(json.dumps({"parts": plan.model_dump()["parts"], "reply_to": quoted}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if fingerprint in attempted:
                return tool_skipped("duplicate_reply", "本轮已尝试发送此消息。", delivery_state="not_attempted")
            if outbound_identity is not None and request_id:
                existing = await db_session.scalar(Select(OrderedDeliveryJob).where(
                    OrderedDeliveryJob.platform == outbound_identity.platform, OrderedDeliveryJob.session_id == session_id,
                    OrderedDeliveryJob.is_private == workspace.is_private, OrderedDeliveryJob.user_id == workspace.user_id,
                    OrderedDeliveryJob.request_id == request_id, OrderedDeliveryJob.fingerprint == fingerprint))
                if existing is not None:
                    if existing.status == "waiting":
                        return tool_success("ordered_delivery_queued", "组合回复已提交。", data={"job_id": existing.job_id, "job_status": existing.status})
                    if existing.status == "completed":
                        return tool_success("message_sent", "这条消息已经发送。",
                                            data={"message_id": existing.result_message_id, "meme_count": meme_count}, delivery_state="completed")
                    if existing.status in {"delivering", "delivery_unknown"}:
                        return tool_failure("delivery_unknown", "这条消息可能已发送，请勿重复发送。", delivery_state="unknown")
                    return tool_failure("delivery_unavailable", "此回复已取消或未能完成，请重新提出请求。", delivery_state="not_attempted")
            if drawing_ids:
                if outbound_identity is None or not request_id:
                    raise ValueError("组合绘图缺少会话信息")
                batch_id = uuid.uuid4().hex
                now = utcnow()
                for drawing_id in sorted(drawing_ids):
                    result = await db_session.execute(update(DrawingJob).where(
                        DrawingJob.job_id == drawing_id, DrawingJob.status == "held", DrawingJob.batch_id.is_(None),
                        DrawingJob.platform == outbound_identity.platform, DrawingJob.session_id == session_id,
                        DrawingJob.is_private == workspace.is_private, DrawingJob.user_id == workspace.user_id,
                        DrawingJob.request_id == request_id, DrawingJob.bot_id == workspace.bot_id,
                    ).values(batch_id=batch_id, status="queued", updated_at=now))
                    if result.rowcount != 1:
                        await db_session.rollback()
                        raise ValueError("绘图任务不可用或已经被其他组合使用")
                if not await is_request_active(session_id, request_id):
                    await db_session.rollback()
                    return tool_skipped("request_expired", "请求已过期，未提交组合回复。", delivery_state="not_attempted")
                db_session.add(OrderedDeliveryJob(job_id=batch_id, platform=outbound_identity.platform,
                    session_id=session_id, is_private=workspace.is_private, bot_id=workspace.bot_id,
                    bot_name=bot_name, user_id=workspace.user_id, request_id=request_id, reply_to_id=quoted,
                    fingerprint=fingerprint, parts=persisted_parts, status="waiting", created_at=now, updated_at=now))
                await db_session.commit()
                attempted.add(fingerprint)
                return tool_success("ordered_delivery_queued", "组合回复已提交，图片准备齐后会按顺序一起发送。",
                                    data={"job_id": batch_id, "job_status": "waiting"})
            message = compose_reply(plan, images, quoted)
            if outbound_identity is not None and outbound_identity.platform == "OneBot V11":
                await message.export(adapter="OneBot V11", fallback=False)
            def record(message_id):
                header = f"id: {message_id}\n" + (f"回复id: {quoted}\n" if quoted else "")
                rows = []
                for part in plan.parts:
                    content, media_id = (descriptions[part.resource_id] if part.type == "image" else
                                         (part.text if part.type == "text" else "@" + part.user_id, None))
                    rows.append(ChatHistory(session_id=session_id, user_id=bot_name, user_name=bot_name,
                                            content_type="bot", content=header + content, media_id=media_id))
                return rows
            if request_id and not await is_request_active(session_id, request_id):
                return tool_skipped("request_expired", "请求已过期，未发送。", delivery_state="not_attempted")
            if outbound_identity is not None and request_id:
                delivery_id = uuid.uuid4().hex
                now = utcnow()
                db_session.add(OrderedDeliveryJob(job_id=delivery_id, platform=outbound_identity.platform,
                    session_id=session_id, is_private=workspace.is_private, bot_id=workspace.bot_id,
                    bot_name=bot_name, user_id=workspace.user_id, request_id=request_id, reply_to_id=quoted,
                    fingerprint=fingerprint, parts=persisted_parts, status="delivering", created_at=now, updated_at=now))
            await db_session.commit()
            if request_id and not await is_request_active(session_id, request_id):
                if delivery_id:
                    now = utcnow()
                    await db_session.execute(update(OrderedDeliveryJob).where(
                        OrderedDeliveryJob.job_id == delivery_id, OrderedDeliveryJob.status == "delivering",
                    ).values(status="cancelled", updated_at=now, finished_at=now))
                    await db_session.commit()
                return tool_skipped("request_expired", "请求已过期，未发送。", delivery_state="not_attempted")
            attempted.add(fingerprint)
            started = True
            receipt = await send_and_finalize(db_session, session_id, message, send_options={"target": send_target, "fallback": False},
                    record=record, publications=publications, image_file_ids=file_ids, image_media_ids=media_ids,
                    recovery_enabled=recovery_enabled, outbound_identity=outbound_identity, reply_to_id=quoted)
            if delivery_id:
                try:
                    now = utcnow()
                    await db_session.execute(update(OrderedDeliveryJob).where(
                        OrderedDeliveryJob.job_id == delivery_id, OrderedDeliveryJob.status == "delivering",
                    ).values(status="completed", result_message_id=receipt.message_id, updated_at=now, finished_at=now))
                    await db_session.commit()
                except Exception as error:
                    await db_session.rollback()
                    logger.warning(f"保存有序消息状态失败 error_type={type(error).__name__}")
            return tool_success("message_sent", receipt.note or "有序消息已发送。",
                                data={"message_id": receipt.message_id,
                                      "meme_count": meme_count}, delivery_state="completed")
        except (ValueError, FileTaskError, OSError) as error:
            if not started:
                await db_session.rollback()
                return tool_failure("invalid_message", str(error), delivery_state="not_attempted")
            logger.warning(f"有序消息投递结果未知 error_type={type(error).__name__}")
        except Exception as error:
            logger.warning(f"有序消息发送异常 error_type={type(error).__name__}")
            if not started:
                await db_session.rollback()
                return tool_failure("prepare_failed", "消息准备失败，未发送。", delivery_state="not_attempted")
        if delivery_id:
            try:
                await db_session.rollback()
                now = utcnow()
                await db_session.execute(update(OrderedDeliveryJob).where(
                    OrderedDeliveryJob.job_id == delivery_id, OrderedDeliveryJob.status == "delivering",
                ).values(status="delivery_unknown", error_code="delivery_outcome_unknown", updated_at=now, finished_at=now))
                await db_session.commit()
            except Exception as error:
                logger.warning(f"保存未知投递状态失败 error_type={type(error).__name__}")
        return tool_failure("send_unknown", "投递结果未知，请勿立即重发。", delivery_state="unknown")

    return send_ordered_message, prepare_meme_image
