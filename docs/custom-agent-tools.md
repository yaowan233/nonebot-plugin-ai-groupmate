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
None
```

推荐返回 `AgentToolBundle`，并提供简短 `instructions`，这样模型会知道什么时候调用你的工具。
