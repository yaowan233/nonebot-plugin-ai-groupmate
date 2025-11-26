from .config import Config
from nonebot import get_plugin_config
from ollama import ChatResponse, Message, Image, AsyncClient


plugin_config = get_plugin_config(Config)
client = AsyncClient(
  host=plugin_config.vlm_ollama_base_url, timeout=15
)

async def image_vl(file_path, prompt: str = "请描述一下这个图片") -> str | None:
    global client
    try:
        response: ChatResponse = await client.chat(model=plugin_config.vlm_model, messages=[
            Message(role="user", content=prompt, images=[Image(value=file_path)])],
                                                   options={"repeat_penalty": 1.5, "num_ctx": 1024})
    except Exception as e:
        print(file_path)
        print(e)
        return
    # 防止输出重复的内容
    if len(response.message.content) > 2000:
        return
    return response.message.content
