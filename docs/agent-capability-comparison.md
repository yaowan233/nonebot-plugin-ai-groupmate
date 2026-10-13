# Bot 的复杂任务执行能力对比

调研日期：2026-10-13。本文比较的是复杂目标能否持续推进、保留执行证据、从失败中恢复并确认完成，而不是接入平台或插件数量。证据来自官方文档和公开源码；没有对四个项目运行同一套任务，因此下述判断是机制比较，不是成功率排行榜。文档、开发分支和已发布版本可能不同；“未确认”不代表项目一定没有。

实现进展：下文的本地核查与差距描述是本次修改前的基线。本次已新增公开网页抓取、文件下载和解析、用户文件区、确定性表格汇总、文件创建与发送接口；用预设模型和模拟 HTTP/聊天接口验证了工具链。浏览器、OCR、通用沙箱代码执行仍未实现；真实公网和 QQ 投递尚未验收。当前功能与限制见 [网页与文件任务](./agent-web-files.md)。

核对过的上游项目：[AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)、[langbot-app/LangBot](https://github.com/langbot-app/LangBot)、[Mai-with-u/MaiBot](https://github.com/Mai-with-u/MaiBot)、[openclaw/openclaw](https://github.com/openclaw/openclaw)。MaiBot 的旧组织地址 `MaiM-with-u` 会重定向到 `Mai-with-u`。

## 优先级修正：先补网页和文件的实际操作能力

用户进一步明确：主要痛点是 Bot 无法自主下载文件、抓取网页信息。这是资料获取和处理链路的缺口，应优先于复杂规划、持久任务和多 Agent 编排解决。

本地核查结果：

- 通用 `search_web` 使用 Tavily basic 搜索，最多返回 3 条结果，每条 `content` 最多保留 1200 字符；当前工具接口没有打开指定 URL、跟进正文链接、分页读取全文的操作。[搜索实现](../src/nonebot_plugin_ai_groupmate/agent/common_tools.py)
- 已有图片下载预览及音视频读取工具，但它们面向专门的消息/媒体流程，不是通用 PDF、CSV、Excel、Word 文件的下载与解析工具。[图片工具](../src/nonebot_plugin_ai_groupmate/agent/image_search_tools.py)、[媒体工具](../src/nonebot_plugin_ai_groupmate/agent/media_tools.py)
- 代码支持可选的千问官方 `web_extractor`，默认工具白名单为空，是否启用取决于部署配置和服务商。这条可选路径不能替代跨模型通用的文件工作区。[配置](../src/nonebot_plugin_ai_groupmate/config.py)
- 当前千问代码解释器包装接收 `task` 文本并返回文本；本地未见把下载文件、平台附件与解释器产物统一接通的通用文件接口。[解释器包装](../src/nonebot_plugin_ai_groupmate/agent/qwen_responses_tools.py)

OpenClaw 官方将搜索、指定 URL 抓取和浏览器操作明确分开：`web_search` 查来源，`web_fetch` 读取指定网页，浏览器处理需要 JavaScript 或交互的页面。这种分层直接对应本项目当前的缺口。[官方 Web 工具说明](https://docs.openclaw.ai/tools/web)

建议先建设以下连通的能力，名称为建议接口而非已有实现：

| 能力 | 作用 |
| --- | --- |
| `fetch_url` | 读取指定网页正文、表格和链接；长内容保存后可分段读取，返回来源 |
| `download_file` | 从 URL 下载文件，保存到按任务隔离的工作区，返回文件 ID、类型和大小 |
| `read_file` | 按格式读取 PDF、CSV、Excel、Word 等，支持页码/工作表/范围定位；扫描件 OCR 单独处理 |
| 沙箱内 Python | 对已保存的数据计算、清洗和制图，并保存结果文件 |
| `send_file` | 将生成文件交付到当前会话，按适配器实际能力实现 |
| 可选浏览器 | 处理动态页面、点击下载、翻页等普通 HTTP 抓取无法完成的步骤 |

工具应共享同一任务的文件引用，形成“搜索 → 打开原始网页 → 找到下载链接 → 下载 → 读取/计算 → 生成并发送结果”的完整流程。新增工具本身不意味着模型会可靠调用，仍需用这条完整链路做端到端验证。

建议开发顺序：先打通公开网页抓取、文件下载、文件读取和统一工作区；随后接通沙箱计算与产物发送；遇到具体动态网页需求时补浏览器。下载实施公共网络地址校验和体积/超时限制，代码执行放在隔离环境内，不直接获得 Bot 主机的完整权限。

下文的目标状态、完成检查和续做建议保留，作为上述基础操作能力到位后的改进。

## 本项目：复杂任务能力判断

核查日期：2026-10-13；本地提交：`fb13802`。用户最终确认的范围是“多步骤办事：查资料、比较、计算、整合结果”。以下是代码审查与已有测试的结论，不是不同 bot 在同一任务集上的能力排名。

**结论：本项目具备多步执行的基础，但缺少保证复杂目标完整交付的任务管理机制。** 模型可以在现有循环内自行规划并完成复杂任务；没有独立 Planner 不能直接等同于不能处理复杂任务。当前的短轮次预算、证据保留和结束判断，使任务越长、约束越多，稳定完成越难保证。

### 已有的有效基础

- LangGraph 实现模型与工具的循环，工具结果进入下一次模型调用；支持技能按需加载、部分只读工具并发、工具失败反馈和副作用去重。不是单次 Function Calling。[执行图](../src/nonebot_plugin_ai_groupmate/agent/graph.py)
- 有网页搜索、历史检索和计算工具；可配置独立的千问代码解释器工具，主 Agent 能得到其文本结果。代码解释器是否可用取决于配置与服务商。[通用工具](../src/nonebot_plugin_ai_groupmate/agent/common_tools.py)、[代码解释器包装](../src/nonebot_plugin_ai_groupmate/agent/qwen_responses_tools.py)
- 已限制“正在查，稍后回复”等空承诺，拦截同一批工具工作尚未完成时提前确认；部分发图任务要求实际执行成功才能结束。这些保护应保留。[完成规则与工具节点](../src/nonebot_plugin_ai_groupmate/agent/graph.py)

### 六个直接影响多步骤办事的限制

| 限制 | 已核实的实现 | 对复杂任务的影响 |
|---|---|---|
| 没有通用的目标与子任务状态 | `AgentState` 有消息、调用次数、已加载技能等，没有要求清单、步骤依赖、证据对应关系、未完成项 | 用户同时要求价格、性能、总成本和推荐理由时，主要依赖模型从长对话中记住所有要求 |
| 完成判断仍主要依赖模型 | 普通任务成功发送 `reply_user(next_step="end")` 后结束；直接文本也可兜底发送并结束。发图等有专门保护，但没有通用任务验收节点 | 回答了一部分问题，也可能被当作整项任务完成；“工具成功”不等于“用户要求全部满足” |
| 执行预算适合较短对话 | 默认 8 次 LLM 调用、180 秒；工具总数硬上限 20；累计 Token 默认 128000。配置可覆盖部分默认值 | 技能加载、查找、补查、计算、验证和写结果共同消耗轮次。普通路径达到 LLM/工具次数上限会结束；Token 上限等特定路径有收尾逻辑，并非所有预算出口都统一收尾 |
| 工具资料缺少可重读的通用保存机制 | 单工具结果默认超过 6000 字符时保留头尾、中间截断；未看到通用结果文件/证据存储与按需读取机制 | 长网页、多方案材料中的条件可能丢失；累积全部原始内容也会占用后续上下文和预算 |
| 聊天延续不等于任务延续 | 跨轮活跃上下文保存聊天和视觉摘要，没有把 `graph_result["messages"]` 的工具轨迹接续；缓存为内存字典，TTL 10 分钟、最多 24 条消息；图 `compile()` 未配置 checkpointer | 用户说“继续，把第三个方案也加进去”时，之前搜索的完整证据、未完成步骤和阶段结果没有通用续接机制。原始聊天、明确记忆、RAG、最近搜图结果等独立保存仍然存在 |
| 聊天提示与任务交付混用 | 主提示强调像群友、短句、复杂问题拆成 2–3 条短消息；同时把 `end` 描述为“本条已经说完”。执行图的完成规则则要求它表示整项任务结束 | 这是提示层的语义张力，可能促使模型优先完成一条聊天回复；需要实测其影响，不能仅凭提示推断必然失败 |

证据位置：[状态与结束判断](../src/nonebot_plugin_ai_groupmate/agent/graph.py)、[默认预算](../src/nonebot_plugin_ai_groupmate/config.py)、[聊天提示](../src/nonebot_plugin_ai_groupmate/agent/prompts.py)、[活跃会话缓存](../src/nonebot_plugin_ai_groupmate/agent/conversation.py)、[图调用及跨轮保存](../src/nonebot_plugin_ai_groupmate/agent/__init__.py)。

已有定时任务保存的是“在某时执行某段任务描述”，并通过租约维护执行状态；执行中断后标记 interrupted、不盲目重跑。这是有价值的调度能力，但不等于从复杂任务完成到一半的步骤恢复。[调度实现](../src/nonebot_plugin_ai_groupmate/scheduled_tasks.py)

### 基础操作能力到位后的执行机制升级

优先采用“单 Agent + 可更新任务计划 + 证据记录 + 完成检查”。是否另加独立规划模型或多个子 Agent，交给对照评测决定。

1. **为复杂请求建立任务状态。** 保存用户目标、硬约束、需交付的内容、步骤状态和阻塞原因。由同一个模型生成并更新简短步骤；不记录内部思维过程。简单闲聊保持轻量。复杂任务允许完整答案，群友语气用于表达，不限制信息完整性。
2. **保存并按需读取证据。** 搜索结果、关键数值、来源 URL、适用日期、计算输入与输出关联到对应步骤。长结果保存到任务附件，仅把摘要和引用放入上下文。来源冲突、缺少维度要显式保留，不能被摘要掩盖。
3. **把“回复用户”与“任务完成”分开。** 最终交付前检查原始要求是否覆盖、推荐是否符合约束、数字能否追溯、计算是否验证。缺项则继续补查；无法获得的信息明确标记未知。达到预算时交付部分结果和未完成项，统一处理各预算出口。
4. **根据进展调整执行。** 保留时间/Token/费用硬边界，为复杂任务设置独立预算和预留收尾空间。重复查同一内容而没有新增证据时换查询或来源；关键输入不足时阻止下游计算/结论继续。循环检测不依赖简单提高 max_llm_calls。
5. **再增加续做和可选委派。** 任务状态按 Bot、会话、发起用户和任务 ID 隔离，允许用户追加条件、继续或取消。需要重启恢复时持久化步骤和操作记录；具有外部副作用的步骤不得因恢复而重复执行。只有多个独立对象的资料查询值得并行时，再引入带独立上下文的子 Agent，并由父任务检查合并结果。

示例任务：“查询 A/B/C 三个方案的官方资料，在预算和功能条件下比较，计算一年总成本，推荐一个并给出依据。”

- 拆解：确认比较维度和约束 → 获取各方案来源 → 对齐价格周期与口径 → 补齐缺项/处理冲突 → 计算 → 检查约束与证据 → 交付。
- 有价格但缺附加费用：继续补查或把总成本标成不完整，不直接宣布最便宜。
- 某来源不可用：尝试合适替代来源、标注证据等级；无法确认则说明缺项。
- 任务完成：每个要求都有结果或明确的未解决说明，推荐可追溯到证据，不能只因为发过一条回复就结束。

### 如何验证是否真的变强

沿用现有 fixture runner，增加长链任务，不以“调用了多少工具”作为成功标准。建议覆盖：

| 场景 | 验收重点 |
|---|---|
| 三方案、多约束、总成本比较 | 所有维度覆盖、口径一致、预算与功能约束同时满足 |
| 多来源价格冲突 | 识别日期、版本和套餐差异，不任取一个数字 |
| 搜索成功但缺关键字段 | 补查或明确缺失，不能进入无依据计算 |
| 某工具超时或连续空结果 | 有限重试、切换策略、明确剩余障碍 |
| 很长的工具结果，关键信息在中段 | 证据保存后可重读，不被头尾截断永久丢失 |
| 用户中途追加条件或要求继续 | 保留已验证资料，更新受影响步骤 |
| 临近预算上限 | 保留交付空间，说明完成部分与未完成项 |
| 首稿只覆盖一半要求 | 验收发现遗漏并继续处理 |

对同一模型、同一任务、同一预算运行现状和升级版，分别记录要求覆盖率、事实/计算正确率、最终完成率、延迟和成本；对子 Agent 再做单独消融比较。先控制模型因素，才能判断架构变化是否有效。

### 验证范围与已有评测的限制

本次执行：
```powershell
.venv\Scripts\python.exe -m pytest tests/test_agent_graph.py -q -o addopts= --disable-warnings
```
结果：**66 passed，5 warnings**。这些是执行机制测试，不是用真实模型做的新一轮复杂任务评测；本次没有付费调用模型，也没有实际运行其他 bot 的同题评测。

现有数据集共有 40 题，其中 8 题归类 multi_tool，多数是查询后计算、搜索后发图、查询后设置提醒等短链任务，尚不足以证明长链复杂任务能力。[评测说明](../evals/README.md)、[用例](../evals/agent_cases.json)

本地现存最新文件名的历史报告 `evals/results/qwen3.7-plus-20260729-114042.json`，记录总体 113/120（94.17%）、multi_tool 20/24（83.33%）。报告日期为 2026-07-29，并非当前代码重跑结果，且工具暴露方式已经与当前实现有差别，不能用来给当前版本定分。复核发现部分 Judge 判断与原始工具证据冲突：multi_tool_008 的工具明确返回 Nova 5.2，裁判仍称其捏造；multi_tool_005 的关键统计同样存在于工具结果，虽然该条另有回复次数超限问题。因此这些百分比只作为历史记录，不作为本次能力结论或已确认幻觉率。

## 判断复杂任务能力的标准

一个具体例子：“收集三个来源的数据，处理缺失与口径冲突，生成表格和结论；其中一个接口失败时换路；用户补充约束后接着做；结束前核对文件和所有要求。”

这要求运行时支持一条连续链路：目标和约束 → 可调整的步骤 → 工具反馈 → 证据与中间产物 → 失败后的改路 → 完成检查。多轮工具调用只是其中一环。下面把有证据的运行机制与尚未证明的能力分开。

| 项目 | 对复杂任务已经确认的支撑 | 本次不能据此认定的能力 |
| --- | --- | --- |
| OpenClaw | 持久目标、子任务委派与回传、上下文压缩、循环防护、重启恢复。[Goal](https://docs.openclaw.ai/tools/goal)、[子 Agent](https://docs.openclaw.ai/tools/subagents/operations)、[恢复](https://docs.openclaw.ai/gateway/restart-recovery) | 自动拆解一定正确、每个完成状态都经过独立验收、任意外部副作用能够恰好执行一次 |
| AstrBot | 工具循环、实验性子 Agent 与后台唤醒、任务导向的上下文摘要、重复调用提示、超长结果落文件。[子 Agent](https://docs.astrbot.app/use/subagent.html)、[上下文](https://docs.astrbot.app/use/context-compress.html)、[执行器源码](https://github.com/AstrBotDevs/AstrBot/blob/48ba71ad7477fc81c8aa99ad241f41ef5b1c0bfe/astrbot/core/agent/runners/tool_loop_agent_runner.py) | 通用持久任务图、子任务任意断点恢复、独立的通用验收器 |
| LangBot | 可替换 Runner、执行日志和状态事件、取消与期限管理、外部工作流接入、运行环境与产物管理。[架构](https://github.com/langbot-app/LangBot/blob/master/ARCHITECTURE.md)、[编排器](https://github.com/langbot-app/LangBot/blob/master/src/langbot/pkg/agent/runner/orchestrator.py) | 接入 n8n/Dify 就代表本体具备自主规划；一次运行完成日志就代表用户目标全部达成 |
| MaiBot | 面向聊天场景的循环决策、信息搜集、工具发现与调用、错误后尝试替代方案。[当前决策提示词](https://github.com/Mai-with-u/MaiBot/blob/0f94cfa83fee87a93275951c9ba2207a90df9070/prompts/zh-CN/maisaka_chat.prompt) | 聊天 Planner 等同于通用长任务规划器；长期人物记忆等同于任务检查点 |

## OpenClaw：最值得借鉴的是目标、执行、恢复分别建模

**目标不会随着某一次回复结束而消失。** 官方 Goal 文档定义会话级持久目标，并区分 active、paused、blocked、budget_limited、complete 等状态。活动目标会在后续轮次重新注入；目标和状态可跨进程重启保留。其完成规则要求核对完整目标，不能因为预算将尽就标为完成。但这些仍包含模型遵循的行为约束，不能当成独立验收器；Goal 本身也不是任务队列或定时调度器。[Goal](https://docs.openclaw.ai/tools/goal)

**委派具备执行生命周期。** 子 Agent 有独立会话、运行标识和结果回传；父任务可以让出执行，再由子任务完成事件触发续做。官方同时限定了持久性的含义：已接纳的排队回传和记录能跨重启保留，重用子会话恢复的是上下文，不会重放被中断的命令。它比“把子 Agent 当成普通同步工具”更适合分支较多、等待较长的任务。[子 Agent](https://docs.openclaw.ai/subagents)、[回传与恢复边界](https://docs.openclaw.ai/tools/subagents/operations)、[yield 交接](https://docs.openclaw.ai/concepts/subagent-yield-handoff)

**压缩试图保住继续工作的依据。** 内置上下文机制支持旧历史摘要与近期上下文保留、标识符保留策略，以及压缩前保存长期信息；工具调用和返回值按完整配对处理。文档明确承认降级摘要可能丢信息，某些失败则保留原历史。值得借鉴的是压缩失败和证据丢失都有显式处理，不是简单把全部返回文本截短。[Compaction](https://docs.openclaw.ai/concepts/compaction)

**运行结束、目标完成、消息送达是不同事实。** Agent loop 有独立运行 ID、工具事件、终态和投递回执；重启恢复文档分别描述会话历史、中断主任务、子任务记录、待投递队列和定时计划。不能由“后台执行完了”直接推断“用户已收到结果”。[Agent loop](https://docs.openclaw.ai/concepts/agent-loop)、[Restart recovery](https://docs.openclaw.ai/gateway/restart-recovery)

**防止无效坚持有运行时手段。** 文档记载连续三次相同工具、参数和错误会终止该轮；另有滚动历史循环检测和压缩后重复检测，且各自启用条件不同。这样能给模型改路机会，又避免“继续努力”变成无限重复。不能把所有检测概括为默认开启。[Tool-loop detection](https://docs.openclaw.ai/tools/loop-detection)

本次判断：四个项目中，OpenClaw 的公开材料对长任务生命周期覆盖最完整，最适合作为运行时设计参照。该判断不表示同一模型下它必然取得最高任务成功率，也不表示所有接入的外部执行器都具备相同能力。

## AstrBot：在现有工具循环上补复杂任务韧性的直接参照

当前 `ToolLoopAgentRunner` 源码中，模型调用、工具执行、结果回填形成循环；同一工具与参数连续出现 3、4、5 次时，会逐级提示检查进展、换参数或换工具。步数接近上限时提示收束，耗尽后要求总结已完成与未完成项。这是策略调整提示，仍需模型实际做出有效决策。[执行器源码](https://github.com/AstrBotDevs/AstrBot/blob/48ba71ad7477fc81c8aa99ad241f41ef5b1c0bfe/astrbot/core/agent/runners/tool_loop_agent_runner.py)

同一源码还提供超长工具结果落文件的路径：配置了溢出目录和读取工具时，把全文保存，只将预览与读取路径放入上下文。这样后续步骤能针对原始证据继续查询。不能忽略它的配置前提，或把预览误认为全文。[大结果处理实现](https://github.com/AstrBotDevs/AstrBot/blob/48ba71ad7477fc81c8aa99ad241f41ef5b1c0bfe/astrbot/core/agent/runners/tool_loop_agent_runner.py)

官方上下文文档规定，摘要关注目标、进度、结论、工具结果和读过的材料，每次工具后再次调用模型前也检查上下文。摘要失败仍可能退回截断，因此关键产物应另存文件。这个机制直接服务于“任务还没做完，但上下文已经很长”的情况。[上下文压缩](https://docs.astrbot.app/use/context-compress.html)

子 Agent 通过 `transfer_to_*` 接收任务描述并返回结果，也支持后台完成后唤醒主 Agent；不会自动获得主会话完整历史，所以委派必须带上路径、约束与背景。官方仍将其标为实验性功能。本次没有验证通用任务图或独立验收器；执行器中的普通无工具文本回复会进入 DONE，因此不能把运行结束当成目标已验证完成。[子 Agent 编排](https://docs.astrbot.app/use/subagent.html)、[结束逻辑](https://github.com/AstrBotDevs/AstrBot/blob/48ba71ad7477fc81c8aa99ad241f41ef5b1c0bfe/astrbot/core/agent/runners/tool_loop_agent_runner.py)

定时任务可持久保存、重启后重新加载计划，但官方明确不要依赖补发全部停机提醒；一次性任务即使执行失败也会在定时触发后删除。因此“有 Cron”不是“复杂任务中断后可靠续做”的充分证据。[主动型能力](https://docs.astrbot.app/use/proactive-agent.html)

## LangBot：宿主运行管理和具体 Agent 策略要分开评估

官方架构将 IM 输入、Pipeline、Runner、工具与持久化分层。当前主分支的 Host 编排器逐条记录 Runner 事件，处理 `state.updated`，保存运行终态、错误与用量，并支持取消、期限以及向活动运行送入后续指令。这说明它有运行状态管理，不能简单归为“只有聊天转发”。但这些日志和状态接口本身，没有证明所选 Runner 会制定好计划或验证交付物。[架构](https://github.com/langbot-app/LangBot/blob/master/ARCHITECTURE.md)、[AgentRunOrchestrator](https://github.com/langbot-app/LangBot/blob/master/src/langbot/pkg/agent/runner/orchestrator.py)

LangBot 可把 IM 接到 Dify、n8n、Langflow 等外部平台。此时复杂流程的依赖、重试与验收要看外部流程怎么配置；不能将外部平台的全部能力算成本体的自主规划能力。内置 Skills 的价值是按需加载某类任务的方法，而“能激活操作手册”也不等于已经有动态计划或任务完成证明。[Pipelines](https://docs.langbot.app/en/usage/pipelines/readme)、[Skills](https://docs.langbot.app/en/usage/skills/readme)

本次判断：适合借鉴 Host 对执行事实的记录和 Runner 边界。没有进一步固定并实测各 Runner 的配置，因此不判断它的自主拆解成功率，也不认定所有 Runner 有跨崩溃检查点恢复。

## MaiBot：聊天决策器能做多步行动，但规划对象不同

当前上游提示词要求分析互动、搜集信息并选择下一步行为；支持发现延迟工具、独立工具并行、依赖工具顺序执行、遇错换方案。可见它并非只能进行一次模型问答。代码中也有 Action Loop 的工具筛选、历史上下文及可被新消息打断的 Planner 请求。[决策提示词](https://github.com/Mai-with-u/MaiBot/blob/0f94cfa83fee87a93275951c9ba2207a90df9070/prompts/zh-CN/maisaka_chat.prompt)、[推理引擎](https://github.com/Mai-with-u/MaiBot/blob/0f94cfa83fee87a93275951c9ba2207a90df9070/src/maisaka/reasoning_engine.py)

不过，这里的中心对象是“机器人在当前聊天中下一步做什么”。本次未确认它存在通用的持久目标、步骤依赖、交付物验收与断点恢复组合。把名字里的 Planner 直接等同于能完成任意复杂任务，会过度推断。对本项目若要增强复杂任务执行，MaiBot 的优先借鉴价值低于前两者；若要优化持续参与聊天，则是另一项比较。

## 对本项目设计的启发

以下是由外部机制推导的设计建议，不是四个项目都已经证明有效的统一方案：

1. 让未完成目标、关键约束、当前步骤、已确认事实与产物引用成为显式任务状态。不要只依靠聊天上下文记住“还要做什么”。
2. 在工具循环中加入进展判断：这一步解决了什么，失败属于参数问题、资源问题还是路线问题；连续无进展时换路线或明确报告阻塞。
3. 长结果保存全文与来源，模型接收摘要和可重新读取的位置。压缩后仍能核对先前数字、文件和失败原因。
4. 将“回复用户”与“任务完成”分离。完成前逐项检查请求约束和实际产物；部分完成、等待输入、预算耗尽必须有不同状态。
5. 先把单 Agent 的任务状态、恢复和验收做稳，再按独立子问题加入委派。多 Agent 会扩大协调成本，不会自动补好目标遗漏与错误验收。

验证这些设计需要同模型、同工具、同预算的任务集，至少覆盖分支依赖、接口失败改路、超长结果、中途补充约束、跨轮续做和验收失败后的返工。单纯看工具调用次数或几步调用成功率，无法回答复杂目标是否真正完成。
