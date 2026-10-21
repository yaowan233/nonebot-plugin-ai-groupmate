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

- **群体认知档案**：由 bot 在发现值得长期记住的新话题、成员特征、内部梗或氛围变化时自主更新，让 bot 对群文化有持续感知。
- **长记忆**（需配置 Qdrant）：RAG 自动存储聊天历史，学习群友发言习惯，使 bot 更像真人。
- **表情包学习**（需配置 Qdrant）：默认使用 `qwen3-vl-plus` 理解图片内容，分别索引描述语义与视觉特征，自动从群内偷学表情包；回复时通过 RRF 多路召回，结合当前群的真实使用频率重排，并在发送前按当前对话审核相关性。
- **自定义 Agent Tools**：可以注册自己的 LangChain tools 扩展 agent 能力，详见 [自定义 Agent Tools](./docs/custom-agent-tools.md)。
- **定时任务管理**：支持通过聊天创建、查看、修改和取消定时消息及 Agent 任务，详见 [定时任务](#定时任务)。
- **网页与文件任务**：打开公开网页、提取正文与下载链接，下载并读取 PDF、Word、Excel、CSV，按列汇总表格并发送结果文件，详见 [网页与文件](./docs/agent-web-files.md)。

对于主模型选择：推荐使用支持 Function Calling 的通义千问系列模型（如 `qwen-plus`、`qwen-max`）。图片标注默认使用 `qwen3-vl-plus`，群档案摘要默认使用 `qwen-flash`，均可通过对应角色配置修改。若图片标注报 `404 model_not_found`，请核对 `tagging_model` 是否仍在服务商的可用模型列表中；已有配置可在 WebUI 中切换模型并测试连接，无需重启。

### 明确记忆

用户明确提出的事实和聊天要求会通过 Agent 工具保存到 ORM 数据库，按会话、用户、Bot 和群聊/私聊隔离，无需 Qdrant。保存用户原话和来源消息，只有数据库提交成功才确认记住。

- “接下来模仿雪人的文笔聊天”：临时要求，30 分钟无该用户与 Bot 的互动后失效，最长保留 24 小时；“今天”的要求最晚当天结束失效。
- “以后叫我小陈”：明确的长期要求，持续保留，直到用户修改或取消。未明确长期范围的要求默认临时。
- “恢复正常聊天”“忘掉这个称呼”：Agent 删除对应记忆；更正同一事项覆盖原值。明确要求忘掉全部时才批量删除。
- 有效聊天要求在后续回复读取，相关事实通过工具查询正文。当前请求优先，不会把用户记忆当成系统规则。普通闲聊和第三方引用不自动保存。

每个作用范围最多 12 项，每项原话最多 400 字；满额时提示清理，不自动淘汰长期记忆。到期记忆在读取前清理，后台每 5 分钟也会清理一次；忘记操作删除明确记忆条目，原始聊天日志仍按现有历史记录策略保存。

升级后先在 Bot 项目目录执行 `nb orm upgrade` 创建记忆表，再启动 Bot。

### 定时任务

在群聊中 @ Bot，或在私聊中直接用自然语言管理定时任务：

| 操作 | 示例 |
|:--|:--|
| 创建提醒 | “10 分钟后提醒我喝水” |
| 创建 Agent 任务 | “1 小时后查一下上海天气并告诉我” |
| 查看任务 | “看看当前有哪些定时任务” |
| 查看历史 | “看看已经执行、失败或取消的定时任务” |
| 修改时间 | “把喝水提醒改成从现在起 30 分钟后执行” |
| 修改内容 | “把喝水提醒的内容改成起来走走” |
| 取消任务 | “取消刚才那个喝水提醒” |

Bot 会先查询任务列表，再定位需要修改或取消的任务；存在多个候选时会询问具体目标。固定消息和 Agent 任务均支持管理，修改内容不会改变任务类型，修改时间从当前时刻重新计算；只改内容时保留原执行时间。

任务以当前会话和 Bot 为管理范围，同一群聊内共享任务列表，不跨群、跨私聊或跨 Bot 管理。支持 10 秒至 7 天内的一次性任务；默认列表只显示待执行任务，也可查询执行中、已完成、失败、已取消、过期和中断记录。已经开始执行的任务无法撤回。

任务内容、执行时间和状态均存入插件 ORM 数据库，复用现有数据库配置，无需额外数据库。升级后请在 Bot 项目目录执行 `nb orm upgrade` 创建任务表，再启动 Bot。APScheduler 只负责每秒扫描数据库，不保存单个待执行任务；重启后会自动继续处理待执行任务，目标 Bot 未连接时会等待连接。超过原定时间 5 分钟的任务标记为过期，不补发。

执行前通过数据库原子领取，防止多个执行器重复领取同一任务；失败不自动重试。若执行进程中断，记录会在执行租约到期后（最长约 3 分钟）标记为中断；由于外部消息可能已部分发送，中断任务不会自动重跑。旧版仅存在内存中的任务无法在重启后迁移，需要重新创建。

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

### 从旧版本升级

表情包多向量集合 `media_collection_v3` 从 `v2.1.1` 开始引入。升级前请按当前版本和向量化模式确认迁移路径：

| 当前版本/模式 | 升级方式 |
|:--|:--|
| `v2.1.1` 至 `v2.1.10`，且使用 `multimodal` 模式 | 等待日志中的“待重建旧表情包”为 0、确认 `media_collection_v3` 已完成重建后，可直接升级。 |
| `v2.1.0` 及更早版本，且使用 `multimodal` 模式 | 先升级到 `v2.1.10`，保留旧 `media_collection` 并等待 v3 重建完成，再升级到最新版。不要跨过该迁移步骤。 |
| `v2.1.7` 及更高版本，且始终使用 `text` 模式 | 保持 `meme_embedding_mode=text` 时可直接升级，继续使用 `media_collection_text`。切换到 `multimodal` 仍需重新向量化。 |

新版聊天 RAG 会创建 `chat_collection_v2`，并在启动后于后台从 SQL 历史记录重建干净的 Dense + 稀疏混合索引。升级前需要先执行 `nb orm upgrade`；旧 `chat_collection` 会保留且不会阻塞 Bot 启动。待日志显示历史会话向量化完成、RAG 查询正常后，可按需人工删除旧集合。

历史检索在重排后按索引中的来源消息 ID 回查当前会话的 SQL 原文，返回每条消息的稳定 `user_id`、数据库 `msg_id`、发言时昵称和 `role`。同一用户改名后仍按 ID 识别，同名但不同 ID 不合并；`msg_id` 可用于展开前后文。现有 v2 索引无需重新生成向量。缺少来源 ID 或回查失败的片段会标明来源未核实；已删除或属于其他会话的来源不从向量文本恢复。

可用下面的命令确认 v3 集合存在并查看向量数量（请按实际地址修改 Qdrant URL）：

```bash
curl http://127.0.0.1:6333/collections/media_collection_v3
```

迁移依赖原始表情包文件和可用的 Qwen 向量接口。如果旧文件已经丢失，对应数据无法重建到 v3；此时不要删除旧集合，也不要升级到不再读取 `media_collection` 的版本。完成升级并确认搜索正常后，旧 `media_collection` 才可以删除。


## ⚙️ 配置

配置说明
| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| ai_groupmate__bot_name | 否 | `"bot"` | bot 名 |
| ai_groupmate__reply_probability | 否 | `0.01` | 普通群消息的独立主动接话抽样概率，默认 1%；`0` 关闭，实际发言还受语境、冷却与频率限制 |
| ai_groupmate__proactive_observation_enabled | 否 | `True` | 允许 Bot 偶尔自主查看近期群聊；配置中心可即时关闭，本群主动回复概率为 `0` 时也不查看 |
| ai_groupmate__global_model_daily_group_limit_enabled | 否 | `true` | 是否启用每群每日公共主模型额度；关闭后不查询、不记录额度 |
| ai_groupmate__global_model_daily_group_limit | 否 | `50` | 每个群每天可使用公共主模型的回复次数，按 Bot 本地时间零点重置；`0` 表示不限额，配置了群聊独立 API 的群不计入 |
| ai_groupmate__global_model_daily_private_user_limit_enabled | 否 | `true` | 是否启用每位用户每日私聊公共主模型额度；关闭后不查询、不记录额度 |
| ai_groupmate__global_model_daily_private_user_limit | 否 | `10` | 每位用户每天私聊可使用公共主模型的回复次数，按 Bot 本地时间零点重置；`0` 表示不限额 |
| ai_groupmate__repeat_probability | 否 | `0.15` | Bot 已在当前连续对话窗口参与，且至少两名不同群友连续发送同一句短文本后，Bot 对每条新跟读加入队形的概率 |
| ai_groupmate__proactive_reaction_probability | 否 | `0.05` | 兼容旧配置；非定向群消息的主动 reaction 采样已停用，避免额外模型调用 |
| ai_groupmate__proactive_meme_probability | 否 | `0.02` | 兼容旧配置；非定向群消息的主动表情包采样已停用，避免额外模型调用 |
| ai_groupmate__personality_setting | 否 | 无 | 自定义人设和固定业务知识 prompt |
| ai_groupmate__tavily_api_key | 否 | 无 | Tavily 搜索 API 密钥（联网搜索及关键词搜图发送功能，支持群聊和私聊） |
| ai_groupmate__web_search_provider | 否 | `tavily` | 网页搜索服务，可选 `tavily` / `deepseek`；保存后立即生效，关键词搜图仍使用 Tavily |
| ai_groupmate__deepseek_search_api_key | DeepSeek 搜索时必填 | 无 | DeepSeek 官方搜索 API 密钥，独立于主聊天模型配置；使用 `https://api.deepseek.com/v1/tools/search` |
| ai_groupmate__llm_api_key | 推荐 | 无 | 通用 LLM API Key，未单独配置各角色 key 时使用 |
| ai_groupmate__llm_base_url | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 通用 OpenAI 兼容接口地址 |
| ai_groupmate__chat_model | 否 | `qwen3.7-plus` | 主聊天/工具调用模型；使用百炼官方工具时推荐 `qwen3.8-max` |
| ai_groupmate__chat_api_key | 否 | 无 | 主聊天模型专用 API Key，留空则使用 `llm_api_key` / `qwen_token` |
| ai_groupmate__chat_base_url | 否 | 无 | 主聊天模型专用 Base URL，留空则使用 `llm_base_url` |
| ai_groupmate__chat_temperature | 否 | `0.7` | 主聊天模型温度 |
| ai_groupmate__chat_reasoning_effort | 否 | 留空 | OpenAI 兼容主模型思考强度：`none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`；须选择模型支持的值，DeepSeek 留空默认 `low` |
| ai_groupmate__chat_max_tokens | 否 | `32768` | 主模型单次输出 Token 上限；推理是否计入由供应商决定，显式留空使用供应商默认 |
| ai_groupmate__chat_api_format | 否 | `openai` | 主聊天接口格式，可选 `openai` / `anthropic` / `vertex` |
| ai_groupmate__chat_multimodal | 否 | `true` | 主聊天模型是否支持图片输入；若使用纯文本模型请设为 `false`，将跳过图片上传只发文本 |
| ai_groupmate__chat_responses_builtin_tools | 否 | `[]` | 千问 Responses API 官方工具白名单，可选 `code_interpreter`、`web_search`、`web_extractor`、`web_search_image`、`image_search`；仅官方百炼 OpenAI 兼容地址生效，留空完全关闭 |
| ai_groupmate__chat_responses_enable_thinking | 否 | `false` | 官方工具调用启用 medium 思考强度；启用 `code_interpreter` 时会按官方要求自动开启 |
| ai_groupmate__chat_explicit_prompt_cache | 否 | `true` | 为支持的接口添加显式 Prompt 缓存断点；支持 DashScope、Anthropic，以及 OpenRouter 上的 Gemini/Claude，并自动使用匿名群级粘性路由 |
| ai_groupmate__group_api_relay_url | 否 | `https://mayumi.xyz` | 群 API 公网中转服务根地址；默认使用 Mayumi，也可覆盖为自建中转服务 |
| ai_groupmate__group_api_relay_registration_token | 私有注册模式首次必填 | 无 | 中转服务签发的部署者注册码；服务器开放注册时留空，实例注册成功后也可移除 |
| ai_groupmate__group_api_local_encryption_key | 否 | 自动生成 | 可选的 32 字节 Base64URL 本地主密钥；留空时在插件数据目录自动生成并持久化 |
| ai_groupmate__group_api_relay_timeout_seconds | 否 | `15` | 调用中转服务的 HTTP 超时（秒） |
| ai_groupmate__group_api_ticket_ttl_seconds | 否 | `900` | 配置链接和一次性配置码的有效期（秒，范围 60～3600） |
| ai_groupmate__group_api_allowed_provider_hosts | 否 | `[]` | 群 API 模型服务主机白名单，支持精确主机及 `*.example.com`；留空时仅允许 DNS 全部解析到公网 IP 的 HTTPS 地址，配置私有模型端点时必须显式填写 |
| ai_groupmate__vertex_project | 标准 Vertex 时推荐 | 无 | Google Cloud 项目 ID；ADC / 服务账号模式使用，Express Mode API Key 不需要 |
| ai_groupmate__vertex_location | 否 | `global` | ADC / 服务账号模式的 Vertex AI 区域；Express Mode API Key 下忽略 |
| ai_groupmate__vertex_api_key | Express Mode 必填 | 无 | Vertex AI Express Mode API Key，可直接写入 NoneBot 的 `.env`；服务账号路径已配置时忽略 |
| ai_groupmate__vertex_credentials_path | 否 | 无 | 容器内服务账号 JSON 路径；留空时使用 Application Default Credentials (ADC) |
| ai_groupmate__google_web_detection_enabled | 否 | `false` | 启用 Google Cloud Vision Web Detection 反向搜图；仅在用户明确要求以图搜图、找原图、图源或出处时向 Agent 暴露 |
| ai_groupmate__google_web_detection_credentials_path | 否 | 无 | 反向搜图服务账号 JSON 路径；留空时复用 `vertex_credentials_path`，再回退 ADC |
| ai_groupmate__google_web_detection_project_id | 否 | 无 | 反向搜图计费项目 ID；留空时复用 `vertex_project`，再使用凭据自带项目 |
| ai_groupmate__google_web_detection_monthly_limit | 否 | `900` | 整个 Bot 每个自然月的本地硬限额，范围 1～1000；缓存命中不计数，失败请求保守计数 |
| ai_groupmate__google_web_detection_cache_days | 否 | `30` | 相同图片 SHA-256 的反向搜图结果缓存天数 |
| ai_groupmate__google_web_detection_timeout_seconds | 否 | `20` | Google Web Detection 请求超时（秒） |
| ai_groupmate__google_web_detection_max_results | 否 | `5` | 每类匹配网页、完整匹配图、局部匹配图和相似图的最大返回数，范围 1～10 |
| ai_groupmate__vision_model | 否 | 无 | 图片回读辅助模型（如 `qwen3-vl-plus`）；主模型不支持图片时用它总结工具返回的图片内容，留空则跳过图片回读 |
| ai_groupmate__vision_api_key | 否 | 无 | 图片回读辅助模型专用 API Key，留空则使用 `llm_api_key` / `qwen_token` |
| ai_groupmate__vision_base_url | 否 | 无 | 图片回读辅助模型专用 Base URL，留空则使用 `llm_base_url` |
| ai_groupmate__vision_temperature | 否 | `0.01` | 图片回读辅助模型温度 |
| ai_groupmate__vision_api_format | 否 | `openai` | 图片回读辅助接口格式，可选 `openai` / `anthropic` / `vertex` |
| ai_groupmate__vision_input_cost_per_million | 否 | `0` | 图片回读辅助模型每百万输入 Token 费用，用于 WebUI 成本统计 |
| ai_groupmate__vision_output_cost_per_million | 否 | `0` | 图片回读辅助模型每百万输出 Token 费用，用于 WebUI 成本统计 |
| ai_groupmate__agent_timeout_seconds | 否 | `180` | 单次 agent 总运行超时（秒） |
| ai_groupmate__agent_llm_timeout_seconds | 否 | `60` | 主模型首个流式片段及后续片段的等待超时（秒）；收到片段后重新计时，整轮仍受 agent_timeout_seconds 限制 |
| ai_groupmate__agent_tool_timeout_seconds | 否 | `30` | 每次工具调用超时（秒） |
| ai_groupmate__agent_max_concurrency | 否 | `4` | 全局同时运行的 Agent 上限，超出的请求在不占用数据库连接的状态下等待 |
| ai_groupmate__agent_max_concurrency_per_group | 否 | `2` | 同一个群可同时处理的 @Bot 或连续追问数；后台插话仍保持单任务 |
| ai_groupmate__agent_max_parallel_tools | 否 | `4` | 单轮可同时执行的安全只读工具上限；未知或有副作用的工具仍串行执行 |
| ai_groupmate__background_image_max_concurrency | 否 | `2` | 后台图片下载、压缩和入库的并发上限 |
| ai_groupmate__background_image_max_pending | 否 | `100` | 后台图片任务的最大待处理数，防止高峰期无界堆积 |
| ai_groupmate__maintenance_max_concurrency | 否 | `1` | 向量化、媒体清理和群档案维护的共享并发上限 |
| ai_groupmate__media_vectorize_min_references | 否 | `3` | 图片进入表情包识别与向量化队列所需的最低引用次数 |
| ai_groupmate__media_vectorize_batch_size | 否 | `1000` | 每轮最多处理的新图片数及旧向量重建数 |
| ai_groupmate__media_vectorize_concurrency | 否 | `8` | 表情包标注与向量化的并发数（过高可能触发接口限流） |
| ai_groupmate__group_memory_update_timeout_seconds | 否 | `120` | 群档案后台更新超时（秒） |
| ai_groupmate__agent_max_llm_calls | 否 | `8` | 单次 agent 常规处理阶段最多调用主模型次数；任务尚未完成时额外允许一次无工具收尾 |
| ai_groupmate__agent_max_total_tokens | 否 | `2000000` | 单次 agent 常规工作阶段最多累计模型 token 数；达到上限且任务尚未完成时会额外执行一次受控收尾 |
| ai_groupmate__agent_tool_result_max_chars | 否 | `6000` | 写回后续上下文的单次工具结果最大字符数 |
| ai_groupmate__agent_file_tools_enabled | 否 | `true` | 启用公开网页抓取、文件下载、文档读取、表格汇总与文件发送；文件保留 24 小时，详见 [使用说明](./docs/agent-web-files.md) |
| ai_groupmate__agent_execution_backend | 否 | `disabled` | 通用代码执行：`local` 本机执行（无隔离）、`sandbox` 按平台自动选择写入限制、`docker` 临时容器或 `remote` 独立服务；容器模式需构建执行镜像，详见 [部署说明](./docs/agent-web-files.md#可选的通用执行环境) |
| ai_groupmate__agent_execution_sandbox_windows_runner | 否 | 空 | Windows 沙箱 runner.js 绝对路径；留空查找管理员安装的 DSH ACL 运行器 |
| ai_groupmate__agent_execution_image | 否 | `ai-groupmate-executor:local` | 直接 Docker 模式使用的执行镜像 |
| ai_groupmate__agent_execution_url | 否 | 空 | 远程执行服务地址，仅管理员配置 |
| ai_groupmate__agent_execution_token | 否 | 空 | 远程执行服务访问令牌 |
| ai_groupmate__agent_execution_timeout_seconds | 否 | `20` | 单次执行时限，1–120 秒，同时受工具时限和 Agent 总预算约束 |
| ai_groupmate__flash_model | 否 | `qwen-flash` | 快速判断是否需要回复的模型 |
| ai_groupmate__flash_api_key | 否 | 无 | 快速判断模型专用 API Key |
| ai_groupmate__flash_base_url | 否 | 无 | 快速判断模型专用 Base URL |
| ai_groupmate__flash_temperature | 否 | `0.0` | 快速判断模型温度 |
| ai_groupmate__flash_max_tokens | 否 | `10` | 快速判断模型最大输出 token；主动接话的结构化判断至少使用 `64` |
| ai_groupmate__summary_model | 否 | `qwen-flash` | 群体记忆档案更新模型 |
| ai_groupmate__summary_api_key | 否 | 无 | 群体记忆模型专用 API Key |
| ai_groupmate__summary_base_url | 否 | 无 | 群体记忆模型专用 Base URL |
| ai_groupmate__summary_temperature | 否 | `0.3` | 群体记忆模型温度 |
| ai_groupmate__summary_max_tokens | 否 | `800` | 群体记忆模型最大输出 token |
| ai_groupmate__tagging_model | 否 | `qwen3-vl-plus` | 图片/表情包标注模型；须使用服务商当前可用的视觉模型 |
| ai_groupmate__tagging_api_key | 否 | 无 | 图片标注模型专用 API Key |
| ai_groupmate__tagging_base_url | 否 | 无 | 图片标注模型专用 Base URL |
| ai_groupmate__tagging_temperature | 否 | `0.01` | 图片标注模型温度 |
| ai_groupmate__tagging_api_format | 否 | `openai` | 图片标注接口格式，可选 `openai` / `anthropic` / `vertex` |
| ai_groupmate__qwen_token | 否 | 无 | 兼容旧配置的 DashScope API Key；新配置推荐使用 `llm_api_key` |
| ai_groupmate__base_model | 否 | 无 | 兼容旧配置的默认模型名；新配置推荐使用 `chat_model` |
| ai_groupmate__qdrant_uri | 否 | 无 | Qdrant 地址，不填则禁用表情包、RAG 等向量功能；要求 Qdrant 服务端版本不低于 `1.16` |
| ai_groupmate__qdrant_api_key | 否 | 无 | Qdrant API Key（使用 Qdrant Cloud 时需要） |
| ai_groupmate__embedding_api_key | 否 | 无 | Embedding API Key，启用 Qdrant 时必填（推荐硅基流动，免费） |
| ai_groupmate__embedding_base_url | 否 | 无 | Embedding Base URL，启用 Qdrant 时必填（推荐硅基流动，免费）。填根地址如 `https://api.siliconflow.cn/v1`，也可填完整路径 `…/v1/embeddings`（插件会自动去重） |
| ai_groupmate__embedding_model | 否 | `BAAI/bge-m3` | 文本 Embedding 模型名称，用于聊天 RAG 和 `text` 表情包模式。 |
| ai_groupmate__embedding_dimension | 否 | 无 | 文本 Embedding 向量维度。模型不支持维度参数或使用模型默认输出维度时无需填写。 |
| ai_groupmate__meme_embedding_mode | 否 | `multimodal` | 表情包向量化模式。`multimodal`：qwen3-vl-embedding 生成描述+原图双向量（需配 `qwen_token`，支持图找图）；`text`：仅用描述文本向量化（无需 `qwen_token`，图找图不可用） |
| ai_groupmate__rerank_api_url | 否 | 无 | Rerank API URL，启用 Qdrant 时使用（推荐硅基流动，免费） |
| ai_groupmate__rerank_api_key | 否 | 无 | Rerank API Key，启用 Qdrant 时使用（推荐硅基流动，免费） |

向量功能要求 Qdrant 服务端版本不低于 `1.16`，因为插件使用 collection metadata 记录并校验 Embedding 模型与维度。

修改 `embedding_model` 或 `embedding_dimension` 前，请自行重建聊天与纯文本表情包的 Qdrant 向量。插件不会迁移既有向量；旧文本集合缺少 metadata 且当前模型不是历史默认的 `BAAI/bge-m3` 时，插件会拒绝使用该集合，避免混用不同模型生成的向量。

如果多个插件共用 `nonebot-plugin-orm`，建议同时将 SQLAlchemy 连接池设为快速失败，避免外部插件耗尽连接时每条消息卡住 30 秒：

```dotenv
SQLALCHEMY_ENGINE_OPTIONS={"pool_size":5,"max_overflow":10,"pool_timeout":5,"pool_pre_ping":true}
```

并发限制用于避免本插件耗尽连接池；`pool_timeout=5` 是其他插件或数据库异常时的快速降级保护，不建议只靠扩大连接池解决泄漏。

用量 WebUI 默认地址为 `/ai-groupmate/usage`。升级数据库后，页面会额外展示每轮 agent 的 LLM/工具调用次数、平均耗时、工具超时、结果截断与副作用去重情况；旧记录会以 0 显示这些新增指标。

用量统计页及其 JSON 接口可直接访问，不需要 Token。页面右上角的“配置中心”可维护插件运行配置；配置中心使用前必须在环境变量中设置非空的 `AI_GROUPMATE__USAGE_WEBUI_TOKEN`，并通过独立的 HttpOnly Cookie 登录，不会在页面中回显 API Key。网页保存的值存入插件数据库，加载顺序为“代码默认值 → 环境变量 → 网页覆盖值”；可随时一键恢复环境变量配置。

配置中心顶部的“群聊 API”页面可由 Bot 管理员按群 ID 新增、修改和删除独立主聊天模型，并可随独立 API 配置设置该群的主动参与倾向。群级倾向值范围为 `0`～`0.1`，留空则跟随全局值；没有独立大模型 API 的群不能单独调整。新增或修改时会先测试模型连接，成功后立即生效；编辑已有记录时 API Key 留空会保留原 Key。列表与接口只返回脱敏信息，删除后该群立即恢复使用全局主聊天 API 和主动参与倾向。启用配置中心后，插件也会自动生成本地群 API 加密密钥，因此即使不启用公网中转也可以使用这个管理页。

针对内网部署 Bot 的会话级自带模型 API（BYOK）能力已经完成端到端实现：群管理员可为群聊配置，私聊用户也可为自己配置；公网配置页生成一次性配置码，插件解密并测试后按群或按用户保存主模型配置。协议与部署说明见 [分群模型 API 中转配置设计](./docs/group-model-api-relay.md)。配套 FastAPI 中转服务位于 `../backend`，Mayumi 配置页位于 `../mayumi-front`；生产启用前仍需配置域名、HTTPS 与 Secret。

插件默认使用 `https://mayumi.xyz` 作为群 API 中转地址，不需要额外配置 URL。只有使用自建中转服务时才需要覆盖：

```dotenv
AI_GROUPMATE__GROUP_API_RELAY_URL=https://relay.example.com
# 仅中转服务器未开放公开注册时需要
AI_GROUPMATE__GROUP_API_RELAY_REGISTRATION_TOKEN=replace-with-issued-token
# 可选；建议生产环境显式限制群管理员可选的模型服务域名
AI_GROUPMATE__GROUP_API_ALLOWED_PROVIDER_HOSTS=["api.openai.com","openrouter.ai","*.example-provider.com"]
```

使用默认 Mayumi 公开注册服务时，不需要配置 `AI_GROUPMATE__GROUP_API_RELAY_REGISTRATION_TOKEN`。

插件默认在数据目录生成 `group_api_local_encryption.key`。请把该文件与数据库一起持久化并备份；丢失后已经保存的群 API Key、实例令牌和私钥均无法恢复。使用 Secret 管理平台时，也可以继续显式设置本地主密钥：

```bash
python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))"
```

`GROUP_API_RELAY_REGISTRATION_TOKEN` 只在 Bot 第一次向中转服务注册时使用，注册成功并确认数据库已持久化后可以从环境中移除。上述启动配置不会进入插件配置中心，也不会在 WebUI 回显。

回复概率、Agent 限制、模型与费用配置会在保存后热更新；Qdrant、Embedding、表情包向量化模式、兼容 Qwen Token 和 Rerank 连接配置会标记为“等待重启”。WebUI 开关、访问路径和管理密码属于启动配置，仍需通过环境变量修改。升级后请先执行：

```bash
nb orm upgrade
```

最小配置示例：

```dotenv
AI_GROUPMATE__BOT_NAME=bot
AI_GROUPMATE__LLM_API_KEY=sk-xxxx
AI_GROUPMATE__CHAT_MODEL=qwen3.5-plus
```

DeepSeek 官方主模型示例：

```dotenv
AI_GROUPMATE__CHAT_API_FORMAT=openai
AI_GROUPMATE__CHAT_BASE_URL=https://api.deepseek.com/v1
AI_GROUPMATE__CHAT_API_KEY=你的DeepSeek密钥
AI_GROUPMATE__CHAT_MODEL=deepseek-flash
AI_GROUPMATE__CHAT_REASONING_EFFORT=low
AI_GROUPMATE__CHAT_MAX_TOKENS=32768
```

官方地址会自动使用保留推理字段的 Chat Completions 适配器，支持连续工具调用。
`chat_reasoning_effort` 和 `chat_max_tokens` 是通用配置，不限定 DeepSeek。OpenAI 兼容接口会传递显式配置的思考强度；留空不发送该参数，DeepSeek 内部默认 `low`。开启思考时不传温度。不同模型支持的档位不同，请按模型能力配置。
快速判断、摘要等辅助角色使用官方 DeepSeek 地址时关闭思考，继续遵守各角色的输出上限。
这些设置不会切换其他角色的模型或密钥，也不会向百炼、OpenRouter 或其他中转地址添加 DeepSeek 专用参数；中转服务需单独核对协议。
分群/私聊配置为官方地址时同样自动适配，思考强度和输出上限继承全局配置。

Vertex AI 主模型示例（推荐使用服务账号或其他 ADC，不要把 JSON 内容写进 `.env`）：

```dotenv
AI_GROUPMATE__CHAT_API_FORMAT=vertex
AI_GROUPMATE__CHAT_MODEL=gemini-3.7-flash
AI_GROUPMATE__VERTEX_PROJECT=your-google-cloud-project
AI_GROUPMATE__VERTEX_LOCATION=global
AI_GROUPMATE__VERTEX_CREDENTIALS_PATH=/app/secrets/vertex-service-account.json
```

把服务账号 JSON 只读挂载到上述容器路径，并授予服务账号 `Vertex AI User`
（`roles/aiplatform.user`）角色。也可以不设置 `VERTEX_CREDENTIALS_PATH`，改用
`GOOGLE_APPLICATION_CREDENTIALS`、Workload Identity 或运行环境已有的 ADC。插件会自动
刷新 OAuth 凭据。若从 OpenRouter 迁移，`google/gemini-...` 模型名也会自动移除
`google/` 前缀；`CHAT_API_KEY` 和 `CHAT_BASE_URL` 在 Vertex 模式下不会使用。

Vertex AI Express Mode API Key 不需要项目 ID 和区域：

```dotenv
AI_GROUPMATE__CHAT_API_FORMAT=vertex
AI_GROUPMATE__CHAT_MODEL=gemini-3.7-flash
# 将 Google 官方示例中 GOOGLE_CLOUD_API_KEY 对应的 Key 值填在这里
AI_GROUPMATE__VERTEX_API_KEY=AIza...
```

如果 Google 控制台或官方示例将 Express Mode Key 称为 `GOOGLE_CLOUD_API_KEY`，
直接把它的**值**填给 `AI_GROUPMATE__VERTEX_API_KEY` 即可；不需要再配置一个名为
`GOOGLE_CLOUD_API_KEY` 的环境变量。把上述插件配置直接写入 NoneBot 项目使用的
`.env` 即可，也不需要在 Docker Compose 中额外声明 `environment`，或配置 Project
ID、Location、服务账号。插件检测到该 Key 后会使用 Express Mode，并忽略
`VERTEX_PROJECT` 和 `VERTEX_LOCATION`；认证优先级为服务账号 JSON → Vertex API
Key → ADC。`AI_GROUPMATE__CHAT_API_KEY` 是其他接口使用的聊天模型 Key，在 Vertex
模式下不会读取。

网页搜索可在配置中心选择 `web_search_provider=deepseek` 并填写 `deepseek_search_api_key`，保存后立即生效。该密钥调用 [DeepSeek 官方 Search API](https://api-docs.deepseek.com/zh-cn/guides/search/)，与聊天模型选择独立，其他模型也可使用。域名及日期条件会转为 `site:` / `after:` / `before:`，组合后的查询不能超过 128 个字符，超长会提示缩短而不会静默截断。工具返回最多 3 条带来源链接的摘要；需要正文时继续调用 `fetch_url`。切换回 `tavily` 可恢复原搜索服务，关键词搜图始终使用 Tavily。

联网搜图发送：配置 `ai_groupmate__tavily_api_key` 后，可在群聊或私聊中说“网上搜一张猫咪图片发给我”或“搜索风景图片，发两张”。Agent 用 `search_web_images` 获取最多 5 个候选，按描述筛选 1～2 张后调用 `preview_web_images` 看图，再用 `send_web_image` 发送；默认发送 1 张，每轮最多 3 张。支持视觉的主模型直接查看图片，否则使用已配置的 `vision_model` 逐张总结。未配置视觉模型、下载失败或识别失败时会标记为仅按描述选择；只有显式允许降级发送并向用户说明未经视觉确认后才发送。预览成功的图片会缓存到本轮内存并用于发送，避免重复下载。预览下载限制为每张 10 MB。此功能复用 Tavily 搜索额度，无需配置 Google 反向搜图。

Google Cloud Vision 反向搜图示例：

```dotenv
AI_GROUPMATE__GOOGLE_WEB_DETECTION_ENABLED=true
AI_GROUPMATE__GOOGLE_WEB_DETECTION_PROJECT_ID=your-google-cloud-project
AI_GROUPMATE__GOOGLE_WEB_DETECTION_CREDENTIALS_PATH=/app/secrets/google-vision-service-account.json
# 默认 900，给 Google 每月前 1000 个免费单位留出余量
AI_GROUPMATE__GOOGLE_WEB_DETECTION_MONTHLY_LIMIT=900
```

先在对应 Google Cloud 项目中启用 Cloud Vision API 和结算，再把服务账号 JSON
只读挂载到容器。若已经配置 Vertex 服务账号和项目，可省略反向搜图专用的路径与
项目 ID；也可全部留空使用 ADC。升级后必须执行 `nb orm upgrade` 创建缓存、月用量和最近搜图记录表。
启用后，正常对话可使用反向搜图工具；仅在用户要求新搜索时调用，不再依赖关键词决定工具是否可见。
有引用时只搜索被回复的图片，引用找不到会报错；没有引用时才搜索当前用户最近发送的图片。相同图片在默认 30 天缓存期内
不会再次请求 Google。这里的 900 次限制只统计本 Bot 发出的请求；若同一 Google 项目
还被其他程序使用，应把限制设得更低，并同时在 Google Cloud 设置预算告警或配额。

最近一次搜图结果按会话和用户持久化保存，包含目标图片、时间、匹配链接和失败状态。
追问“搜到了什么”时使用 `get_last_image_search_result` 读取证据，不重新联网；新搜图之后最多补查两次网页，再根据证据回答。
明确叫 Bot 处理任务时保留文字回复工具；只发表情或 reaction 的模式仅用于 Bot 自主插话。

固定知识示例（将群号和入群方式替换为自己的信息）：

```dotenv
AI_GROUPMATE__PERSONALITY_SETTING="【固定知识】当用户询问加群、群号、入群方式或请求拉群时，明确告诉对方：请搜索 QQ 群 123456789 申请加入，验证信息填写‘来自 Bot’。不要编造其他群号或链接。"
```

修改该配置后需要重启 Bot。固定知识会同时用于群聊和私聊；涉及口令、密钥等敏感内容时不要放在这里。

如果想使用更强的主聊天模型：

```dotenv
AI_GROUPMATE__CHAT_MODEL=qwen3.7-plus
```

如果要让 `qwen3.8-max` 使用百炼服务器提供的代码解释器、联网搜索和网页抓取：

```dotenv
AI_GROUPMATE__CHAT_MODEL=qwen3.8-max
AI_GROUPMATE__CHAT_RESPONSES_BUILTIN_TOOLS='["code_interpreter","web_search","web_extractor"]'
AI_GROUPMATE__CHAT_RESPONSES_ENABLE_THINKING=true
```

这些工具只在官方百炼 OpenAI 兼容地址上生效，配置后会自动使用 Responses API。所有工具默认关闭；工具列表只是授权模型按需调用，并不代表每轮都会执行。`web_extractor` 依赖 `web_search`，只填写前者时插件也会自动启用后者。代码解释器因为不能与 Bot 的 Function Calling 同请求，会由插件隔离调用，再把结果交回主 Agent。

华北 2（北京）当前参考工具价为：`code_interpreter`、`web_extractor` 限时免工具费，`web_search` 4 元/千次，`web_search_image`（文搜图）24 元/千次，`image_search`（图搜图）48 元/千次；所有工具产生的额外模型 Token 仍收费，实际价格以百炼控制台为准。Token Plan Harness 的 `t2i_search` / `i2i_search` 个人版 Key 不可用于机器人后端，因此这里使用普通按量 API 的正式工具名。接口、地域和计费依据见 [千问 Responses API 官方工具调研](./docs/qwen-responses-built-in-tools.md)。

插件会尽量复用稳定 system prompt、固定工具 schema，并在连续对话中复用 append-only history，以提高输入缓存命中率。日志中可通过 `[LLM缓存]` 查看缓存命中 token；如果服务商未返回缓存字段，会显示 `缓存命中=未返回`。

## 🎉 使用

@bot 即可触发回复；普通群消息按 `reply_probability` 独立抽样决定是否进入主动接话判断（默认 1%，`0` 关闭）；额度开关默认开启，未配置群聊独立 API 时，每群每天默认可使用公共主模型回复 50 次，私聊则为每位用户每天 10 次，额度按 Bot 本地时间零点重置。额度耗尽后会提示 `/配置群API` 或 `/配置个人API`；已配置相应独立大模型 API 的群或私聊用户不受公共额度限制。可在管理界面关闭对应额度开关，关闭后不查询或记录额度。群管理员还可覆盖本群主动参与倾向，但用户配置上限为 `0.1`。Bot 已在当前连续对话窗口参与、随后至少两名不同群友连续发送同一句短文本时，会按 `repeat_probability` 独立决定是否原样加入队形。reaction 和图片表情不再对未提及 Bot 的普通群消息主动采样；只有当前消息明确呼叫 Bot、回复 Bot、处于同一用户的连续对话窗口，或用户明确要求时，才会交给 Agent 选择或执行。

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

### 群聊互动与历史查询

群聊历史工具可用于话题回顾、梗或说法的来历、成员过往发言及观点变化、约定和事件核对、共同经历与多人互动调查。例如“回顾一下上周大家聊了什么”“这个梗是怎么来的”“对比小王前后对这个游戏的看法”。成员评价或排名只是其中一种用法。

Agent 按问题选择工具：具体话题用现有 `search_history_context` RAG；已知成员可用 `read_member_history` 按成员、正文关键词和日期读取原话，或按来源消息 ID 展开前后文；需要发现多人跟读和代表互动时才用 `search_group_interactions`。后者只统计短时间内多人发送相同文本的事件，不能代表全部讨论、完整话题热度或全群共识。

每轮 Agent 的常规工具调用上限为 30 次。达到工具次数、模型次数或累计 token 的处理预算时，若尚未交付完整回复，会停止搜索并额外执行至多一次无工具的模型调用，整理已取得的证据；这次收尾的 token 仍计入实际用量。资料不足时应说明限制，不为补足排名或结论继续检索。

两个 SQL 工具不依赖 Qdrant，也不调用后台提炼模型；读取的片段会占用回复模型输入。互动扫描最多读取日期范围内最近 12 万条文本/BOT记录，结果可翻页；消息包含说话者、时间和来源消息 ID，需结合上下文核对。

群聊中 `search_history_context`、`read_member_history`、`search_group_interactions` 从第一轮就直接可用，无需匹配关键词或先加载技能，由 Agent 按当前问题选择是否查询。查询仅限当前群，保留只读、请求有效期和结果预算；私聊不开放两个群 SQL 工具，主动表情、reaction 和复读专用轮次仍使用各自的工具范围。

### 指令表

由于 AI 功能需要记录聊天记录，基于已记录的聊天记录，可以很轻松的做到词频统计，所以顺带加上了。

|     指令      |    说明    |
|:-----------:|:--------:|
|  /词频 <统计天数> | 生成个人词频词云 |
| /群词频 <统计天数> | 生成群词频词云  |
| /重置负面关系 | 仅超级用户；预览需要重置的历史负面关系数量 |
| /重置负面关系 确认 | 仅超级用户；备份后将负好感度归零并清空这些用户的旧标签 |
| /配置群API | 群主、群管理员或超级用户创建配置链接；链接通过私聊发送 |
| /提交群API <配置码> | 发起人私聊提交一次性配置码，测试并应用群主模型配置 |
| /查看群API | 群内查看当前群级主模型配置的脱敏摘要 |
| /删除群API 确认 | 群主、群管理员或超级用户删除群配置并恢复全局主模型 |
| /配置个人API | 私聊创建只属于自己的模型配置链接 |
| /提交个人API <配置码> | 私聊提交一次性配置码，测试并应用个人主模型配置 |
| /查看个人API | 私聊查看个人主模型配置的脱敏摘要 |
| /删除个人API 确认 | 删除个人配置，恢复公共主模型和每日额度 |

关系备份保存在 NoneBot 插件数据目录下的 `relation_backups` 文件夹中。重复执行是安全的；没有负好感度记录时不会创建备份或修改数据库。

### 动态参与群聊

Bot 会按群形成长期的相处感受：结合人设兴趣、最近的话题、接梗和互动方式，聊得来就更愿意冒泡，持续相处不舒服就逐渐少插话。公共 API 的群同样适用，无需配置本群独立 API。群级手动覆盖仍是基础倾向，自动感受在此基础上调整；`0` 始终关闭主动接话。

每个 Bot、每个群分别保存印象、依据与进度，重启后恢复。相处感受在主 Agent 的正常群聊循环中顺带更新，结合完整人设、群档案、明确记忆、连续对话和本轮实际交流；使用主循环已经加载的最多 20 条近期消息（通常最近 1 小时，不足时扩展至 6 小时）以及已有线程上下文，不为评估另查聊天记录。私聊、复读、主动表情包和 reaction 专用轮次不评估。积累至少 12 条新群友文本后，初始最短间隔为 15 分钟；连续没有新变化或依据无效时，间隔逐步扩大到 30、60、120 分钟。有新变化时恢复 15 分钟；群友重新明确呼叫或叫停 Bot 也可提前到最短间隔重新评估，但不会直接加减好感。没有正常主循环运行时不会为了更新感受而单独调用模型，也不设每天两次的硬限制。

只有轮到更新时才在主循环增加简短评估说明与消息编号对照，主模型可在正常 `reply_user(next_step=end)` 或 `finish` 参数中附带私有的 `group_feeling` 字段，随这次正常结束一起保存；它不发送到群里、不占用额外工具轮次、不增加模型调用或重复输入聊天正文。模型可省略此字段；字段无效、缺失或保存失败都不会要求模型补写，也不影响正常回复。Gatekeeper 继续只做轻量接话判断，预算保持原值。顺带评估仍会增加少量输入和输出 Token。正面或负面变化须引用两条不同的新群友原话（每条最多 60 字符），负面还需来自不同成员；只保存最多 100 字符的感受原因，不重述聊天记录。结合语境判断是否聊得投机，不按回复数直接加减好感；没人接话、群里安静、正常纠错、礼貌结束和个别人的一次抱怨都不能直接变成对全群的负面印象。

Bot 可在主循环的可选 `group_feeling.desired_reply_probability` 字段中直接选择本群想使用的主动接话抽样概率，无固定加减步长；例如默认基准 1% 时，可自主选择 0.1%、0.8%、1.3%、3% 或 5%。代码只校验范围和依据：允许值为当前手动基础概率的 `0.1`～`5` 倍，默认配置下为 0.1%～5%，有效概率最高 `0.1`；改变概率须引用两条不同的新真人原话，降低概率还须来自两名不同真人。字段省略或为 null 时保留原意愿；非法数值、越界、依据不足或评估期间手动基准已改变时不调整，也不额外调用模型修正。手动基础值为 `0` 时保持关闭。选择结果按 Bot/群保存为相对偏好，重启后恢复；旧偏好按 30 天半衰期逐渐回归手动基准，不改变明确提问的回应质量，也不绕过冷却、叫停、公共额度或发言上限。更新后需先执行 `nb orm upgrade` 创建群相处感受表；模型失败或依据不可靠时保留原印象。

普通随机接话只在收到新消息时考虑；群里沉默不会触发开场白。`reply_probability` 保留原配置名和群级覆盖规则：相处感受为中性、使用默认 `0.01` 时，每条符合条件的文本消息独立以 1% 的概率进入模型判断；近期参与过也不会提高抽样概率，未抽中不会积累机会，群聊沉默也不会增加下次概率。是否发言仍由 Gatekeeper 判断语境，实际发言概率更低；不存在聊够固定条数就触发的门槛。命令、无文本消息及明确指向其他人的消息不抽样。

- 同一 Bot、同一群的主动判断至少间隔 20 秒。实际发言后，近期参与的对话至少冷却 45 秒，其他情况至少 120 秒。明确呼叫与现有的同用户连续对话不受这些主动接话冷却限制。
- 主动消息最多 3 条 / 10 分钟；最近 10 分钟内最多 40 条活动中，Bot 至少发过 3 条且占比达到 25% 时，暂停主动接话。限制仅作用于主动接话，不妨碍正常响应明确请求。
- 一次主动发言后，至少经过 120 秒且出现 5 条后续群友文本、仍未发现明确承接，才提高退让程度。冷却和判断间隔最多扩大 8 倍，每 10 分钟逐级恢复；明确回应或模型识别到承接则恢复。未抽中判断的语义回应可能漏判，因此这只是短期频率调节，不保存为成员好感或长期评价。
- Gatekeeper 看最近 12 条消息（包含 Bot 发言，每条最多 1200 字符），判断是否值得接话及是否承接 Bot；不同群友可以接续同一话题。无效输出或调用失败时不发言。
- 明确叫停后，普通主动接话暂停 5 分钟；再次明确呼叫 Bot 可恢复。已在思考中的主动轮次在发送前也会检查叫停状态。
- 短期参与状态（冷却、最近判断、未承接计数）只驻留内存，30 分钟无活动后重置，最多保存 2048 个 Bot/群组合；进程重启会重置，多进程部署不共享这些计数。长期相处感受和回顾间隔保存在数据库，重启后恢复。

Bot 还可以偶尔自己醒来看看群聊，由 `proactive_observation_enabled` 控制（默认开启）。每个 Bot、每个群随机间隔 45～90 分钟，至少积累 12 条新的普通群友文本，而且最近 15 分钟仍有人聊天时，才产生一次查看机会。刚参与过对话的 45 分钟内、最近 5 分钟已经做过接话判断、本群主动回复概率为 `0`、Bot 未连接或已有回复任务时均跳过；叫停后暂停此类查看，再次明确呼叫可恢复。Bot 实际发言后重新等待随机间隔和新消息。跳过或看完沉默也会消耗本次机会，不会反复重试；同一进程最多每 5 分钟发起一次自主查看。

这项功能平时只用消息事件在内存中计数；每分钟的定时器只检查本地时间和状态，不扫描群记录、不调用模型。真正醒来时只读本群最近 1 小时内最多 20 条文本（包含 Bot 发言，每条最多 600 字符），直接由主模型结合人设、群档案、明确记忆和相处感受决定沉默或自然接一句，无需额外调用 Gatekeeper。查看不会复用或延长旧对话线程，不分析图片或自动检索更多历史；历史中的命令不视为当前请求。查看轮次只开放回复和结束工具，最多发送一条，模型调用最多两轮，出错或达到预算时安静结束；不会因预算耗尽而发送完成提示，也不能自行安排持续查看。符合相处感受回顾条件时，可随本轮结束动作顺带更新感受，不增加评估轮次。实际模型调用照常计入用量和公共群额度，即使最终选择沉默也会消耗 Token。

查看进度仅驻留内存，最多保存 256 个 Bot/群组合，6 小时无新活动后清理；重启会重新等待随机间隔，不会立刻补看所有群。此限制按进程计，多进程部署不共享查看计划。
