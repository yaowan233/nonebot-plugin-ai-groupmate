<div align="center">
    <a href="https://v2.nonebot.dev/store">
    <img src="https://raw.githubusercontent.com/fllesser/nonebot-plugin-template/refs/heads/resource/.docs/NoneBotPlugin.svg" width="310" alt="logo"></a>

## ✨ nonebot-plugin-ai-groupmate ✨
[![LICENSE](https://img.shields.io/github/license/yaowan233/nonebot-plugin-ai-groupmate.svg)](./LICENSE)
[![pypi](https://img.shields.io/pypi/v/nonebot-plugin-ai-groupmate.svg)](https://pypi.python.org/pypi/nonebot-plugin-ai-groupmate)
[![python](https://img.shields.io/badge/python-3.10|3.11|3.12|3.13-blue.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/package%20manager-uv-black?style=flat-square&logo=uv)](https://github.com/astral-sh/uv)
<br/>
[![ruff](https://img.shields.io/badge/code%20style-ruff-black?style=flat-square&logo=ruff)](https://github.com/astral-sh/ruff)
[![pre-commit](https://results.pre-commit.ci/badge/github/yaowan233/nonebot-plugin-ai-groupmate/master.svg)](https://results.pre-commit.ci/latest/github/yaowan233/nonebot-plugin-ai-groupmate/master)
[![codecov](https://codecov.io/gh/yaowan233/nonebot-plugin-ai-groupmate/graph/badge.svg?token=TMR6QZ6C6I)](https://codecov.io/gh/yaowan233/nonebot-plugin-ai-groupmate)

</div>

## 📖 介绍
2.0 版本更新，轻量化依赖，全部使用 API 进行调用，基本上任何设备都能运行。

本插件主体使用 langchain 的 agent 进行决策，由 langchain 调用 tools 进行一系列任务。

- **群体认知档案**：每 6 小时自动归纳群内话题、活跃成员特征、内部梗等，让 bot 对群文化有持续感知。
- **长记忆**（需配置 Qdrant）：RAG 自动存储聊天历史，学习群友发言习惯，使 bot 更像真人。
- **表情包学习**（需配置 Qdrant）：使用 `qwen-vl-max` 理解图片内容，自动从群内偷学表情包并存入向量库，回复时按语义匹配发出。
- **自定义 Agent Tools**：可以注册自己的 LangChain tools 扩展 agent 能力，详见 [自定义 Agent Tools](./docs/custom-agent-tools.md)。

对于主模型选择：推荐使用支持 Function Calling 的通义千问系列模型（如 `qwen-plus`、`qwen-max`）。图片理解固定使用 `qwen-vl-max`，群档案摘要固定使用 `qwen-flash`。

## 💿 安装

<details open>
<summary>使用 nb-cli 安装</summary>
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install nonebot-plugin-ai-groupmate --upgrade
使用 **pypi** 源安装

    nb plugin install nonebot-plugin-ai-groupmate --upgrade -i "https://pypi.org/simple"
使用**清华源**安装

    nb plugin install nonebot-plugin-ai-groupmate --upgrade -i "https://pypi.tuna.tsinghua.edu.cn/simple"


</details>

<details>
<summary>使用包管理器安装</summary>
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details open>
<summary>uv</summary>

    uv add nonebot-plugin-ai-groupmate
安装仓库 master 分支

    uv add git+https://github.com/yaowan233/nonebot-plugin-ai-groupmate@master
</details>

<details>
<summary>pdm</summary>

    pdm add nonebot-plugin-ai-groupmate
安装仓库 master 分支

    pdm add git+https://github.com/yaowan233/nonebot-plugin-ai-groupmate@master
</details>
<details>
<summary>poetry</summary>

    poetry add nonebot-plugin-ai-groupmate
安装仓库 master 分支

    poetry add git+https://github.com/yaowan233/nonebot-plugin-ai-groupmate@master
</details>

打开 nonebot2 项目根目录下的 `pyproject.toml` 文件, 在 `[tool.nonebot]` 部分追加写入

    plugins = ["nonebot-plugin-ai-groupmate"]

</details>

<details>
<summary>使用 nbr 安装(使用 uv 管理依赖可用)</summary>

[nbr](https://github.com/fllesser/nbr) 是一个基于 uv 的 nb-cli，可以方便地管理 nonebot2

    nbr plugin install nonebot-plugin-ai-groupmate
使用 **pypi** 源安装

    nbr plugin install nonebot-plugin-ai-groupmate -i "https://pypi.org/simple"
使用**清华源**安装

    nbr plugin install nonebot-plugin-ai-groupmate -i "https://pypi.tuna.tsinghua.edu.cn/simple"

</details>


## ⚙️ 配置

配置说明
| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| ai_groupmate__bot_name | 否 | `"bot"` | bot 名 |
| ai_groupmate__reply_probability | 否 | `0.01` | 群内主动发言概率 |
| ai_groupmate__personality_setting | 否 | 无 | 自定义人设 prompt |
| ai_groupmate__tavily_api_key | 否 | 无 | Tavily 搜索 API 密钥（联网搜索功能） |
| ai_groupmate__llm_api_key | 推荐 | 无 | 通用 LLM API Key，未单独配置各角色 key 时使用 |
| ai_groupmate__llm_base_url | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 通用 OpenAI 兼容接口地址 |
| ai_groupmate__chat_model | 否 | `qwen3.5-plus` | 主聊天/工具调用模型，推荐 `qwen3.5-plus` 或 `qwen3.7-plus` |
| ai_groupmate__chat_api_key | 否 | 无 | 主聊天模型专用 API Key，留空则使用 `llm_api_key` / `qwen_token` |
| ai_groupmate__chat_base_url | 否 | 无 | 主聊天模型专用 Base URL，留空则使用 `llm_base_url` |
| ai_groupmate__chat_temperature | 否 | `0.7` | 主聊天模型温度 |
| ai_groupmate__chat_api_format | 否 | `openai` | 主聊天接口格式，可选 `openai` / `anthropic` |
| ai_groupmate__agent_timeout_seconds | 否 | `180` | 单次 agent 总运行超时（秒） |
| ai_groupmate__agent_llm_timeout_seconds | 否 | `60` | 每次主模型调用超时（秒） |
| ai_groupmate__agent_tool_timeout_seconds | 否 | `30` | 每次工具调用超时（秒） |
| ai_groupmate__agent_max_llm_calls | 否 | `8` | 单次 agent 最多调用主模型次数 |
| ai_groupmate__agent_max_total_tokens | 否 | `64000` | 单次 agent 最多累计模型 token 数 |
| ai_groupmate__agent_tool_result_max_chars | 否 | `6000` | 写回后续上下文的单次工具结果最大字符数 |
| ai_groupmate__flash_model | 否 | `qwen-flash` | 快速判断是否需要回复的模型 |
| ai_groupmate__flash_api_key | 否 | 无 | 快速判断模型专用 API Key |
| ai_groupmate__flash_base_url | 否 | 无 | 快速判断模型专用 Base URL |
| ai_groupmate__flash_temperature | 否 | `0.0` | 快速判断模型温度 |
| ai_groupmate__flash_max_tokens | 否 | `10` | 快速判断模型最大输出 token |
| ai_groupmate__summary_model | 否 | `qwen-flash` | 群体记忆档案更新模型 |
| ai_groupmate__summary_api_key | 否 | 无 | 群体记忆模型专用 API Key |
| ai_groupmate__summary_base_url | 否 | 无 | 群体记忆模型专用 Base URL |
| ai_groupmate__summary_temperature | 否 | `0.3` | 群体记忆模型温度 |
| ai_groupmate__summary_max_tokens | 否 | `800` | 群体记忆模型最大输出 token |
| ai_groupmate__tagging_model | 否 | `qwen-vl-max` | 图片/表情包标注模型 |
| ai_groupmate__tagging_api_key | 否 | 无 | 图片标注模型专用 API Key |
| ai_groupmate__tagging_base_url | 否 | 无 | 图片标注模型专用 Base URL |
| ai_groupmate__tagging_temperature | 否 | `0.01` | 图片标注模型温度 |
| ai_groupmate__tagging_api_format | 否 | `openai` | 图片标注接口格式，可选 `openai` / `anthropic` |
| ai_groupmate__qwen_token | 否 | 无 | 兼容旧配置的 DashScope API Key；新配置推荐使用 `llm_api_key` |
| ai_groupmate__base_model | 否 | 无 | 兼容旧配置的默认模型名；新配置推荐使用 `chat_model` |
| ai_groupmate__qdrant_uri | 否 | 无 | Qdrant 地址，不填则禁用表情包、RAG 等向量功能 |
| ai_groupmate__qdrant_api_key | 否 | 无 | Qdrant API Key（使用 Qdrant Cloud 时需要） |
| ai_groupmate__embedding_api_key | 否 | 无 | Embedding API Key，启用 Qdrant 时必填（推荐硅基流动，免费） |
| ai_groupmate__embedding_base_url | 否 | 无 | Embedding Base URL，启用 Qdrant 时必填（推荐硅基流动，免费） |
| ai_groupmate__rerank_api_url | 否 | 无 | Rerank API URL，启用 Qdrant 时使用（推荐硅基流动，免费） |
| ai_groupmate__rerank_api_key | 否 | 无 | Rerank API Key，启用 Qdrant 时使用（推荐硅基流动，免费） |

用量 WebUI 默认地址为 `/ai-groupmate/usage`。升级数据库后，页面会额外展示每轮 agent 的 LLM/工具调用次数、平均耗时、工具超时、结果截断与副作用去重情况；旧记录会以 0 显示这些新增指标。

最小配置示例：

```dotenv
AI_GROUPMATE__BOT_NAME=bot
AI_GROUPMATE__LLM_API_KEY=sk-xxxx
AI_GROUPMATE__CHAT_MODEL=qwen3.5-plus
```

如果想使用更强的主聊天模型：

```dotenv
AI_GROUPMATE__CHAT_MODEL=qwen3.7-plus
```

插件会尽量复用稳定 system prompt、固定工具 schema，并在连续对话中复用 append-only history，以提高输入缓存命中率。日志中可通过 `[LLM缓存]` 查看缓存命中 token；如果服务商未返回缓存字段，会显示 `缓存命中=未返回`。

## 🎉 使用

@bot 即可触发回复，也会以 `reply_probability` 的概率主动发言。

### 自定义 Agent Tools

如果你想给 agent 增加自己的工具（例如查询业务系统、控制设备、调用自定义 API），可以参考 [自定义 Agent Tools](./docs/custom-agent-tools.md)。

内置了好感度系统，增加了趣味性。
![Screenshot_20251201_134157](https://github.com/user-attachments/assets/68b8d563-7ad5-4d83-be4b-0a05f16df09a)

> 以下功能需要配置 Qdrant

配置 Qdrant 后，ai 会自动偷群内使用的表情包并存入向量库，回复时通过 VLM 语义匹配发出，准确率非常高。
![Screenshot_20251201_134203](https://github.com/user-attachments/assets/cbf95194-ac33-45e0-a83d-cb6639c204fb)
发送群内偷学到的表情包
![Screenshot_20251201_132723](https://github.com/user-attachments/assets/6fbd036f-e7ec-4ced-9cd7-557976306553)
利用 RAG 对聊天历史进行语义检索，可进行总结、查询等功能。
![Screenshot_20251201_133320](https://github.com/user-attachments/assets/b7e96bd0-8245-4da5-b28b-33e8aad5fc63)

### 指令表
由于 AI 功能需要记录聊天记录，基于已记录的聊天记录，可以很轻松的做到词频统计，所以顺带加上了。

|     指令      |    说明    |
|:-----------:|:--------:|
|  /词频 <统计天数> | 生成个人词频词云 |
| /群词频 <统计天数> | 生成群词频词云  |
