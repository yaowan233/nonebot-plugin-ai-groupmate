import httpx
from .config import Config
from nonebot import logger
from typing import Optional
from nonebot import get_plugin_config

plugin_config = get_plugin_config(Config)


async def image_vl(base64_image, prompt: str = "请描述一下这个图片") -> Optional[str]:
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 1024,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
    }
    headers = {
        "Authorization": f"Bearer {plugin_config.siliconflow_bearer_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=300)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.info(e)
        logger.info(response.text)
