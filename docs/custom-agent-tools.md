# 自定义 Agent Tools

`ai-groupmate` 支持从外部插件注册 LangChain tool。工具会在每次创建 agent 时自动加入工具列表，适合给自己的机器人扩展查询、控制、业务系统调用等能力。

## 最小示例

```python
from langchain.tools import tool
from nonebot_plugin_ai_groupmate.agent import (
    AgentToolBundle,
    AgentToolContext,
    register_agent_tool,
)


@register_agent_tool
def build_my_tools(ctx: AgentToolContext) -> AgentToolBundle:
    @tool("get_current_session_id")
    async def get_current_session_id() -> str:
        """获取当前会话 ID。"""
        return ctx.session_id

    return AgentToolBundle(
        tools=[get_current_session_id],
        instructions=[
            "- get_current_session_id：当需要知道当前群/私聊会话 ID 时使用。"
        ],
    )
```

只要这段代码所在模块被 NoneBot 加载，工具就会注册到 agent。

## 按需 Skill Prompt

如果外部插件很多，或者每个插件都有较长的使用说明，不建议把长文本都放进 `instructions`。可以注册 `AgentSkill`，agent 初始 prompt 里只会看到技能名称和简短描述；只有当前任务需要时，模型才会调用内置 `load_agent_skill` 读取完整 prompt。

```python
from nonebot_plugin_ai_groupmate.agent import (
    AgentSkill,
    AgentToolBundle,
    AgentToolContext,
    register_agent_tool,
)


@register_agent_tool
def build_my_skills(ctx: AgentToolContext) -> AgentToolBundle:
    return AgentToolBundle(
        skills=[
            AgentSkill(
                name="score_report",
                description="查询、解读和发送成绩报告相关能力。",
                prompt="""
当用户要查询成绩、生成成绩图、分析成绩走势时使用这个技能。
- 如果只是查询文字结果，直接说明需要查询的字段。
- 如果需要发图，先调用成绩插件提供的发送能力。
- 如果用户要求评价发挥，结合返回的数据给出简短分析。
""",
            )
        ]
    )
```

`description` 应该短，只用于路由；`prompt` 可以写完整规则。`prompt` 也可以是同步或异步函数，用于按当前群、用户或权限动态生成说明：

```python
async def build_score_prompt(ctx: AgentToolContext) -> str:
    return f"当前会话 ID 是 {ctx.session_id}，只允许查询本群成绩。"
```

如果插件确实需要让模型调用函数，仍然返回 `tools=[...]`；如果只是让模型掌握一段专用流程、格式或业务规则，优先用 `skills=[...]`。

## AgentToolContext

自定义工具 factory 会收到 `AgentToolContext`：

```python
ctx.db_session   # 当前数据库会话
ctx.session_id   # 当前群/私聊会话 ID
ctx.request_id   # 当前回复请求 ID，可用于过期判断
ctx.user_id      # 触发用户 ID
ctx.user_name    # 触发用户昵称
ctx.interface    # nonebot-plugin-uninfo 查询接口
ctx.send_target  # UniMessage 发送目标
ctx.is_private   # 是否私聊
ctx.bot_id       # bot 自身 ID
ctx.bot          # NoneBot Bot，上下文不存在时为 None
ctx.event        # NoneBot Event，上下文不存在时为 None
ctx.model        # 当前聊天模型
```

## 返回值

factory 可以返回：

```python
tool_object
[tool_object_1, tool_object_2]
AgentToolBundle(tools=[...], instructions=[...])
AgentToolBundle(skills=[...])
None
```

推荐返回 `AgentToolBundle`。短规则放在 `instructions`，长规则放在 `AgentSkill.prompt`，避免每次对话都消耗大量 token。

## Tool 返回多模态内容

普通 tool 返回字符串即可：

```python
return "查询完成"
```

如果 tool 生成了图片，并希望主聊天模型继续基于图片分析，可以返回 OpenAI 兼容的 content block 列表：

```python
import base64


def image_result(text: str, raw: bytes):
    image_data = base64.b64encode(raw).decode("utf-8")
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
    ]
```

`ai-groupmate` 会把其中的文字作为 tool 执行结果，同时额外追加一条带图片的多模态消息给主模型。这样可以满足“先调用工具生成图片，再让模型评价图片内容”的场景。

建议只在用户明确要求分析、评价、解释图片时返回图片内容；如果用户只是要求查询或发图，返回普通字符串即可，避免每次 tool 调用都消耗大量多模态上下文。

示例：

```python
@tool("send_score_image")
async def send_score_image(include_image_for_analysis: bool = False):
    raw = await draw_score_image()
    await UniMessage.image(raw=raw).send(target=ctx.send_target)

    if include_image_for_analysis:
        return image_result("已发送成绩图，请根据图片简短评价发挥。", raw)
    return "已发送成绩图"
```

在 `instructions` 中应明确告诉模型什么时候打开这类参数：

```python
instructions=[
    "- 用户只要求查询/发图时，不要传 include_image_for_analysis。",
    "- 用户问“怎么样/分析/评价/发挥如何”时，传 include_image_for_analysis=true，并基于返回图片回复。",
]
```
