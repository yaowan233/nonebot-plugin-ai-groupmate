import base64
import mimetypes

from ollama import Image, Message, AsyncClient as OllamaClient
from openai import AsyncOpenAI
from nonebot import logger, get_plugin_config

from .config import Config

plugin_config = get_plugin_config(Config).ai_groupmate

ollama_client = None
openai_client = None

if plugin_config.vlm_provider == "ollama":
    ollama_client = OllamaClient(
        host=plugin_config.vlm_ollama_base_url, timeout=15
    )
elif plugin_config.vlm_provider == "openai":
    openai_client = AsyncOpenAI(
        api_key=plugin_config.vlm_openai_api_key,
        base_url=plugin_config.vlm_openai_base_url,
        timeout=15.0
    )


def encode_image_to_base64(file_path: str) -> str:
    """将本地图片转换为 Base64 字符串"""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(file_path: str) -> str:
    """获取文件的 mime type，默认为 image/jpeg"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type if mime_type else "image/jpeg"


async def image_vl(file_path, prompt: str = "请描述一下这个图片") -> str | None:
    try:
        if plugin_config.vlm_provider == "ollama":
            if not ollama_client:
                logger.error("Ollama client not initialized")
                return None

            response = await ollama_client.chat(
                model=plugin_config.vlm_model,
                messages=[
                    Message(role="user", content=prompt, images=[Image(value=file_path)])
                ],
                options={"repeat_penalty": 1.5, "num_ctx": 1024}
            )
            content = response.message.content

        elif plugin_config.vlm_provider == "openai":
            if not openai_client:
                logger.error("OpenAI client not initialized")
                return None

            # 1. 获取 Base64 和 MimeType
            base64_image = encode_image_to_base64(file_path)
            mime_type = get_mime_type(file_path)

            # 2. 构造 OpenAI 多模态消息格式
            response = await openai_client.chat.completions.create(
                model=plugin_config.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024
            )
            content = response.choices[0].message.content

        else:
            logger.error(f"Unknown provider: {plugin_config.vlm_provider}")
            return None

        # 统一的后处理逻辑
        if not content:
            return None

        # 防止输出重复的内容或过长
        if len(content) > 2000:
            return None

        return content

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None
