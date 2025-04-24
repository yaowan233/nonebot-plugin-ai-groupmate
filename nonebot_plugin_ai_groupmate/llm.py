import httpx
import datetime
from .config import Config
from .model import ChatHistory
from typing import Optional, List
from nonebot import get_plugin_config

plugin_config = get_plugin_config(Config)

async def choice_response_strategy(
    contexts: list[str], history: List[ChatHistory], setting: Optional[str] = None
) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    choice_response_prompt = f"""\
你现在是QQ群里的一位普通群友，名叫"{plugin_config.bot_name}"。你的目标是基于提供的历史消息和最近聊天记录，以贴近群友表达形式来参与群聊，避免任何暴露AI身份的表达。
{setting}
规则
1. 冒号前面的是发言人名称，后面的是发言内容，回复的时候不要保留发言人名称
2. 以 ! / ！ # 作为开头的发言内容是调用其他机器人的指令，请勿回复
3. 请不要发送以 ! / ！ # 作为开头的语句来调用其他指令
【角色设定】
- 你是群里的活跃成员
- 你对群内成员的互动风格和关系有基本了解
- 你会先通过学习知识库内其他群友的发言，学习群友的发言习惯
- 请根据历史记录理解群内的主要话题和互动模式

【交流风格】
- 模仿群友的普遍的语言风格和表达习惯
- 使用群友的常见的表情和语气词
- 如果群友有特定的称呼习惯，请采用

【注意事项】
- 不要自称AI、机器人或助手，你是{plugin_config.bot_name}
- 发言风格请学习他人的发言习惯以及风格，尽可能融入他们
- 回答尽可能简短，一般回复1-20字，视情况回复1-5字
- 请注意自己发言的逻辑连贯性但不要重复发句子内容相似的内容
- 尽量不要一直发相似的表情包
- 当其他人都在复读时，请不要复读
- 请基于已知知识对来进行对话，不要对不理解的内容随意回复
- 回复中不要包含“笑死”、“确实”等词语
- 你要注意到其他机器人并且无视他们的发言

你将通过上下文来判断回复文字、回复表情包或不进行回复
- 请注意分析上下文自己回复的时间与内容来进行判断，请不要回复的过于频繁
- 别人叫你发图时，可以选择发送表情包
- 当回复类型为image/both时，必须生成image_desc以及image_emotion字段
- 图片描述需具体（如描述画面元素、文字内容、表情特征）
- 带表情包的回复优先考虑群内常见梗图类型
- 图片情感类型有，请务必从以下几个情感中选择一个，请归类为以下的情感类型，不要输出别的情感：[搞笑,讽刺,愤怒,无奈,喜爱,惊讶,中立]
请用以下JSON格式回应：
{{
  "need_reply": boolean,
  "reply_type": "none/text/image/both",
  "image_desc": string(optional),
  "image_emotion": string(optional),
  "text": string(optional)
}}
"""
    history_contexts = "\n".join(contexts)
    history_messages = combine_messages(history)
    prompt = f"""以下是相关历史对话，你可以通过这些作为知识库来进行判断，现在是{datetime.datetime.now()}：\n{history_contexts}"""
    messages = [
        {"role": "system", "content": choice_response_prompt},
        {"role": "user", "content": prompt},
    ]
    messages += history_messages
    messages.append(
        {"role": "user", "content": "请接着最下面的主题进行选择回复json数据:"}
    )
    # print(prompt)
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
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
        "Authorization": f"Bearer {plugin_config.deepseek_bearer_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=300)
    # print(response.json())
    return response.json()["choices"][0]["message"]["content"]


def combine_messages(messages: List[ChatHistory]) -> List[dict]:
    """
    将消息组合成一个上下文文本，并返回消息ID列表
    """
    res = []
    for msg in messages:
        combined_text = ""
        time = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        combined_text += f"[{time}] "
        if msg.content_type == "bot":
            res.append({"role": "assistant", "content": msg.content, "name": plugin_config.bot_name})
        elif msg.content_type == "text":
            res.append(
                {
                    "role": "user",
                    "content": f"{msg.user_name}: {msg.content}\n",
                    "name": msg.user_name,
                }
            )
        elif msg.content_type == "image":
            res.append(
                {
                    "role": "user",
                    "content": f"{msg.user_name} 发送了一张图片\n该图片的描述为: {msg.content}\n",
                    "name": msg.user_name,
                }
            )
    return res
