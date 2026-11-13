"""Generate artifacts in the same user-scoped workspace as read_file/send_file."""

import json
import asyncio
import hashlib
from typing import Any
from pathlib import Path

from nonebot.log import logger
from langchain_core.tools import tool
from nonebot_plugin_uninfo import SceneType, QryItrface

from ..model import MediaStorage
from ..config import ScopedConfig
from .web_fetch import download_public_url
from ..concurrency import suspend_reply_for_image
from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from ..message_trace import trace_id
from .file_workspace import FileTaskError, FileWorkspace
from .history_format import is_image_history, fallback_qq_avatar_url, published_image_file_id, load_replied_message_histories
from .image_resources import ImageResources, ImageResourceError
from ..message_rollout import message_pilot_enabled
from .image_generation import MAX_IMAGE_BYTES, ImageGenerationError, request_image, prepare_markup, prepare_reference, resolve_image_credentials
from .structured_image_resources import read_scoped_message_images

IMAGE_GENERATION_PROMPT = (
    "\n【GPT 绘图】\n创作或改图用 generate_image，参考图传真实消息 ID（或 current/replied）、file_id；可从近期消息引用中选择，目标不清楚时先确认。"
    "成员头像用 reference_user_ids，未知成员 ID 先查询。生成后 send_file(as_image=true) 交付，配文放 text；连续改图复用 file_id，失败不自动重复生成。\n"
)
BACKGROUND_IMAGE_PROMPT = (
    "\n【GPT 绘图】\n创作或改图用 generate_image，参考图传真实消息 ID（或 current/replied）、file_id；可从近期消息引用中选择，目标不清楚时先确认。"
    "成员头像用 reference_user_ids。普通任务完成后自动发图；图文穿插任务先 defer_delivery=true，再把 drawing:<job_id> 放入 send_ordered_message。失败不自动重复生成。\n"
)
_generation_gate = asyncio.Semaphore(1)


def _read_reference_path(root: Path, file_path: str) -> bytes:
    try:
        return ImageResources(root).read_media(file_path, max_bytes=MAX_IMAGE_BYTES)
    except ImageResourceError as error:
        if error.code == "too_large":
            raise ImageGenerationError("invalid_reference", "参考图超过 10 MiB。") from None
        raise ImageGenerationError("reference_not_found", "参考图原文件已清理或不可访问，请重新发送。") from None


def create_image_generation_tool(db_session, session_id: str, request_id: str | None, *, config: ScopedConfig, workspace: FileWorkspace,
                                 pic_dir: Path, current_message_id: str | None, replied_message_id: str | None,
                                 current_user_id: str | None = None, group_members: list[Any] | None = None, interface: QryItrface | None = None,
                                 background: bool = False, bot_id: str = "", bot_name: str = "bot", platform: str = "", is_private: bool = False):
    config = config.model_copy(deep=True)
    results: dict[str, str] = {}
    attempted: set[str] = set()

    async def active() -> bool:
        return request_id is None or await is_request_active(session_id, request_id)

    @tool("generate_image")
    async def generate_image(prompt: str, reference_message_ids: list[str] | None = None, reference_file_ids: list[str] | None = None,
                             size: str = "1024x1024", transparent_background: bool = False, reference_user_ids: list[str] | None = None,
                             defer_delivery: bool = False, reference_image_index: int | None = None,
                             complexity: str | None = None, output_format: str = "png",
                             frame_count: int | None = None, frame_duration_ms: int | None = None,
                             markup_file_id: str | None = None, size_explicit: bool = False) -> str:
        """用 GPT 创作或修改一张图片。普通后台任务完成会自动发图，defer_delivery=true 时等待组合；返回 file_id 时用 send_file(as_image=true) 交付。

        prompt 保留用户的内容和风格要求；改图须传参考图。reference_message_ids 是本会话图片的真实消息ID，
        current=当前消息，replied=当前用户引用的消息；reference_file_ids 可传已生成或下载的图片 file_id。
        reference_user_ids 取当前群已知成员头像作为真实参考图，current=当前用户；不确定 ID 先用 query_platform 的 list_members 查询，不猜昵称或ID。
        合计最多4张参考图。size 支持 1024x1024、1536x1024、1024x1536、auto。可用透明背景。
        图文穿插需要等多张绘图一起发送时设置 defer_delivery=true，返回待组合的 job_id，不会单独发送。
        需要引用消息中的某一张图时，传单个 reference_message_ids 和从 1 开始的 reference_image_index。
        GPT Image 2.6 可选 complexity=low/medium/high；动态 GIF 使用 output_format=gif，默认自动尺寸，仅用户明确要求具体像素尺寸时传 size_explicit=true；可选 2-16 帧和每帧 20-2000 毫秒；GIF 不能用参考图修改。
        标注引导修改需原图及单独的透明 PNG 标注层，传 markup_file_id，标注层须与原图尺寸一致。
        """
        if not await active():
            return tool_skipped("request_expired", "这轮请求已结束或取消，没有继续绘图。")
        if not config.image_generation_enabled:
            return tool_failure("image_generation_disabled", "GPT 绘图尚未启用。")
        if defer_delivery and not background:
            return tool_failure("deferred_unavailable", "组合绘图需要启用后台绘图。")
        prompt = prompt.strip()
        # Resolve aliases before fingerprinting so the same reference cannot
        # be generated twice merely by switching between "current" and its ID.
        message_aliases = {"current": current_message_id, "replied": replied_message_id}
        message_ids = list(dict.fromkeys(message_aliases.get(value) or value for value in (reference_message_ids or [])))
        file_ids = list(dict.fromkeys(reference_file_ids or []))
        avatar_ids = list(dict.fromkeys(str(current_user_id or "") if uid == "current" else uid.strip() for uid in (reference_user_ids or [])))
        if output_format == "gif" and not size_explicit:
            size = "auto"
        options = {key: value for key, value in {"complexity": complexity, "output_format": output_format if output_format != "png" else None,
                                                  "frame_count": frame_count, "frame_duration_ms": frame_duration_ms,
                                                  "markup_file_id": markup_file_id}.items() if value is not None}
        if (complexity not in {None, "low", "medium", "high"} or output_format not in {"png", "gif"}
                or (output_format == "gif" and (message_ids or file_ids or avatar_ids or transparent_background or markup_file_id))
                or (markup_file_id is not None and (not markup_file_id or not (message_ids or file_ids or avatar_ids)))
                or (output_format != "gif" and (frame_count is not None or frame_duration_ms is not None))
                or (frame_count is not None and (type(frame_count) is not int or not 2 <= frame_count <= 16))
                or (frame_duration_ms is not None and (type(frame_duration_ms) is not int or not 20 <= frame_duration_ms <= 2000))
                or (options and not config.image_generation_model.startswith("gpt-image-2.6-"))):
            return tool_failure("invalid_image_arguments", "当前模型或参数不支持此绘图方式；动态 GIF 暂不支持参考图修改。")
        if not prompt or len(prompt) > 32000 or size not in {"1024x1024", "1536x1024", "1024x1536", "auto"}:
            return tool_failure("invalid_image_arguments", "请提供 1 到 32000 字描述以及支持的图片尺寸。")
        if len(message_ids) + len(file_ids) + len(avatar_ids) > 4:
            return tool_failure("too_many_references", "每次最多使用 4 张参考图。")
        if reference_image_index is not None and (reference_image_index < 1 or len(message_ids) != 1):
            return tool_failure("invalid_reference_index", "选择第几张图片时，需要指定一条消息和从 1 开始的序号。")
        fingerprint_parts = [prompt, message_ids, file_ids, avatar_ids, size, transparent_background]
        if options:
            fingerprint_parts.append(options)
        if defer_delivery:
            fingerprint_parts.append("deferred")
        if reference_image_index is not None:
            fingerprint_parts.extend(["image_index", reference_image_index])
        fingerprint = hashlib.sha256(json.dumps(fingerprint_parts, ensure_ascii=False).encode()).hexdigest()
        if fingerprint in results:
            return results[fingerprint]
        if fingerprint in attempted or len(attempted) >= 3:
            return tool_failure("image_generation_limit", "本轮已尝试该绘图请求或已达到 3 次生成上限；不要自动重复生成。")
        try:
            _, provider_base = resolve_image_credentials(config)
            references = []
            resources = ImageResources(pic_dir, workspace)
            for file_id in file_ids:
                try:
                    data = await asyncio.to_thread(resources.read_file_reference, file_id, max_bytes=MAX_IMAGE_BYTES)
                except ImageResourceError as error:
                    if error.code == "not_found":
                        raise FileTaskError("file_not_found", "文件不存在、已过期或不属于当前会话。") from None
                    raise
                references.append(await asyncio.to_thread(prepare_reference, data))
            for supplied_id in message_ids:
                message_id = message_aliases.get(supplied_id, supplied_id)
                if message_pilot_enabled(config, session_id, is_private=is_private):
                    structured = await read_scoped_message_images(
                        db_session, resources, session_id=session_id, platform=platform, is_private=is_private,
                        message_id=message_id, max_bytes=MAX_IMAGE_BYTES, max_images=4 - len(references),
                        image_index=reference_image_index,
                    )
                    if structured is not None:
                        if not structured:
                            raise ImageGenerationError("reference_not_found", "指定消息没有可用图片；请发送或引用图片，不会替换成其他人的图。")
                        for data in structured:
                            references.append(await asyncio.to_thread(prepare_reference, data))
                        continue
                rows = await load_replied_message_histories(db_session, session_id, message_id)
                images = [row for row in rows if is_image_history(row)]
                if reference_image_index is not None:
                    if reference_image_index > len(images):
                        raise ImageGenerationError("reference_not_found", "指定序号的图片不存在。")
                    images = [images[reference_image_index - 1]]
                if len(references) + len(images) > 4:
                    raise ImageGenerationError("too_many_references", "指定消息包含过多图片，每次最多使用 4 张参考图。")
                if not images:
                    raise ImageGenerationError("reference_not_found", "指定消息没有可用图片；请发送或引用图片，不会替换成其他人的图。")
                for row in images:
                    media = await db_session.get(MediaStorage, row.media_id) if row.media_id is not None else None
                    if published_image_file_id(row) is None and (media is None or not media.file_path):
                        raise ImageGenerationError("reference_not_found", "指定图片的原文件不可用，请重新发送。")
                    data = await asyncio.to_thread(resources.read_history, row, max_bytes=MAX_IMAGE_BYTES,
                                                   allow_owned_legacy=True, media_path=media.file_path if media else None)
                    references.append(await asyncio.to_thread(prepare_reference, data))
                if len(references) > 4:
                    raise ImageGenerationError("too_many_references", "指定消息包含过多图片，每次最多使用 4 张参考图。")
            # No transaction/pooled connection is held while waiting for a GPU.
            await db_session.commit()
            # Waiting for the image queue/provider must not monopolize a chat
            # worker. Resume the same request before returning to the Agent.
            async with suspend_reply_for_image():
                if avatar_ids:
                    if len(references) + len(avatar_ids) > 4:
                        raise ImageGenerationError("too_many_references", "每次最多使用 4 张参考图。")
                    members = group_members
                    if members is None and interface is not None:
                        try:
                            members = list(await interface.get_members(SceneType.GROUP, session_id))
                        except Exception:
                            raise ImageGenerationError("avatar_members_unavailable", "无法获取当前群成员，未开始绘图。") from None
                    avatar_urls = {str(member.id): getattr(getattr(member, "user", None), "avatar", None) or fallback_qq_avatar_url(str(member.id)) for member in (members or [])}
                    if current_user_id:
                        avatar_urls.setdefault(str(current_user_id), fallback_qq_avatar_url(str(current_user_id)))
                    if any(not uid or uid not in avatar_urls for uid in avatar_ids):
                        raise ImageGenerationError("unknown_avatar_member", "参考头像的成员不在当前会话中；请先确认成员 ID，未开始绘图。")
                    if any(not avatar_urls[uid] for uid in avatar_ids):
                        raise ImageGenerationError("avatar_unavailable", "无法获取该成员的头像，未开始绘图。")

                    async def fetch_avatar(uid: str) -> tuple[bytes, str, str]:
                        downloaded = await download_public_url(str(avatar_urls[uid]))
                        return await asyncio.to_thread(prepare_reference, downloaded.data)

                    loaded = await asyncio.gather(*(fetch_avatar(uid) for uid in avatar_ids), return_exceptions=True)
                    for reference in loaded:
                        if isinstance(reference, BaseException):
                            raise reference
                        references.append(reference)
                markup = None
                if markup_file_id:
                    markup_data = await asyncio.to_thread(workspace.read, markup_file_id)
                    markup = await asyncio.to_thread(prepare_markup, markup_data, references[0][0])
                if background:
                    if not request_id or not current_user_id or not bot_id or not platform:
                        raise ImageGenerationError("drawing_context_missing", "无法确认绘图任务的会话，未开始绘图。")
                    from ..drawing_jobs import DrawingRequest, enqueue_drawing
                    from ..drawing_service import endpoint_hash

                    reference_ids = []
                    for index, (data, suffix, mime) in enumerate(references):
                        source_key = "drawing-reference:" + hashlib.sha256(
                            f"{request_id}:{fingerprint}:{index}".encode()).hexdigest()
                        artifact = await asyncio.to_thread(workspace.save, data, f"reference-{index}.{suffix}",
                                                           content_type=mime, source_key=source_key)
                        reference_ids.append(artifact.file_id)
                    if not await active():
                        return tool_skipped("request_expired", "这轮请求已结束或取消，没有继续绘图。")
                    attempted.add(fingerprint)
                    queued = await enqueue_drawing(db_session, DrawingRequest(
                        platform=platform, session_id=session_id, is_private=is_private, bot_id=bot_id,
                        bot_name=bot_name, user_id=current_user_id, request_id=request_id,
                        reply_to_id=current_message_id, fingerprint=fingerprint, prompt=prompt,
                        reference_file_ids=tuple(reference_ids), size=size,
                        transparent_background=transparent_background, model=config.image_generation_model,
                        quality=config.image_generation_quality, endpoint_hash=endpoint_hash(provider_base), deferred=defer_delivery,
                        image_options=options or None,
                    ))
                    await db_session.commit()
                    if queued.status not in {"held", "queued", "running", "generated", "delivering", "completed"}:
                        return tool_failure("drawing_job_unavailable", "该绘图任务已结束或结果未知，请勿自动重复生成。",
                                            data={"job_id": queued.job_id, "job_status": queued.status})
                    logger.info(f"[MessageTrace] trace={trace_id(session_id, request_id)} stage=drawing_queued job={queued.job_id}")
                    result = tool_success("drawing_queued", "绘图待组合，请用 send_ordered_message 提交整条排版。" if defer_delivery else "绘图任务已提交，完成后会自动发送图片。",
                                          data={"job_id": queued.job_id, "job_status": queued.status})
                    results[fingerprint] = result
                    return result
                async with _generation_gate:
                    if not await active():
                        return tool_skipped("request_expired", "这轮请求已结束或取消，没有继续绘图。")
                    attempted.add(fingerprint)
                    request_options = {k: v for k, v in options.items() if k != "markup_file_id"}
                    if markup is not None:
                        request_options["markup"] = markup
                    generated = await asyncio.wait_for(
                        request_image(config, prompt, references, size=size, transparent_background=transparent_background, **request_options),
                        timeout=config.image_generation_timeout_seconds,
                    )
            artifact = await asyncio.to_thread(workspace.save, generated.data, f"gpt-image.{generated.output_format}", content_type=f"image/{generated.output_format}")
            result = tool_success("image_generated", "图已经生成并保存，还没发出去；用 send_file(as_image=true) 发给对方，不用重新生成。", data={**artifact.public(), "model": config.image_generation_model, "usage": generated.usage})
            results[fingerprint] = result
            return result
        except ImageResourceError as error:
            if error.code == "unavailable":
                return tool_failure("reference_unavailable", "暂时无法确认参考图，请稍后再试。")
            if error.code == "expired":
                return tool_failure("reference_expired", "参考图已过期，请重新发送。")
            if error.code == "too_many":
                return tool_failure("too_many_references", "指定消息包含过多图片，每次最多使用 4 张参考图。")
            return tool_failure("invalid_reference" if error.code == "too_large" else "reference_not_found",
                                "参考图过大。" if error.code == "too_large" else "参考图不存在、已过期或不可访问，请重新发送。")
        except (ImageGenerationError, FileTaskError) as error:
            return tool_failure(error.code, str(error))
        except TimeoutError:
            return tool_failure("image_timeout", "绘图等待超时，服务端可能已计费；请勿自动重复生成。")

    generate_image.metadata = {"image_generation_timeout_seconds": config.image_generation_timeout_seconds + 35}
    return generate_image
