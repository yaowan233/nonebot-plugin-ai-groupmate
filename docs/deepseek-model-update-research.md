# DeepSeek 模型/API 更新核查

查询日期：2026-10-19。以下来自 DeepSeek 官方文档当日页面；除更新日志外，指南页未标注最近修订日期，不能据此断言某项协议变化发生于 10 月 17 日。未调用付费模型做线上验证。

## 最新模型

- 2026-10-17 发布 DeepSeek-V4.1-Flash-1017，调用名仍为 `deepseek-flash`，调用方式与价格不变。官方特别建议 DeepSeek Harness 升级到 `0.2.1-rc.1` 或更高版本。[更新日志](https://api-docs.deepseek.com/updates/)
- `deepseek-v4-flash` / `deepseek-v4-flash-vision-exp` 自 9 月 10 日暂时转向 V4.1 Flash；旧 `deepseek-chat` / `deepseek-reasoner` 的公告停用日期是 2026-07-24，不应继续作为新配置默认值。[更新日志](https://api-docs.deepseek.com/updates/)
- 当前列出的两个模型为 `deepseek-flash`（V4.1-Flash-1017）与 `deepseek-v4-pro`（V4-Pro-0813）；上下文 1M，最大输出 384K，同时支持思考/非思考、工具调用、JSON 和 Responses API。[模型详情](https://api-docs.deepseek.com/quick_start/pricing/)

## 接入契约

| 项目 | 当前官方行为 |
| --- | --- |
| 思考开关 | 默认开启，默认强度 `high`；Chat Completions 使用 `thinking: {type: enabled/disabled}` |
| 强度 | `reasoning_effort` 支持 `none/low/high/max`；`none` 关闭思考；`minimal→low`，`medium/xhigh→high` |
| SDK | OpenAI SDK 的 `thinking` 放进 `extra_body`，`reasoning_effort` 为直接参数 |
| 温度 | 思考模式忽略 `temperature`；不能靠降低温度来减少推理 |
| 强制工具 | 思考模式不支持 `tool_choice=required` 或指定工具名，否则 400 |
| 默认输出额度 | 非思考 8K，思考 64K，`max` 强度 128K；可显式设置上限 |

来源：[Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)、[思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)。不要发送互相冲突的开关和 effort；文档没有在这些页面明确冲突优先级。

**多轮工具调用必须保留推理字段。** 请求携带 `tools` 时，所有历史轮次的 `reasoning_content` 都要完整回传，即使某一轮没有实际调用工具；缺失可能报 400。没有 `tools` 时，回传推理会被忽略。官方示例完整保存 assistant 的 `content`、`reasoning_content`、`tool_calls`，下一用户轮仍保留前轮推理。[思考模式工具调用说明](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/#工具调用)

`top_p` 有模型差异：Flash 仅思考模式有效，范围 0.95–1.0；Pro 在思考模式忽略传入值（带 tools 固定 0.95，否则 1.0），非思考模式有效。以思考模式指南的分模型说明为准，避免把 API 通用字段说明外推到 Pro。[思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)

## 对群聊 Agent 的建议（工程判断，非官方强制要求）

1. 先确认模型名、思考开关、消息序列化及工具回传正确，再调整人设提示词；提示词不能修复协议字段丢失。
2. 群聊可使用 `low` 或显式关闭思考控制时延，复杂工具任务再选 `high`；不要直接把编码 Harness 的 `max` 套用到聊天。官方一般建议简单任务 low、日常 Agent high、复杂任务 max。[2026-08-13 更新说明](https://api-docs.deepseek.com/updates/)
3. 提示词应明确输出渠道、允许的工具、先看工具结果再宣称成功、失败时停止重复相同调用；不需要把内部推理过程写进群消息。保留群聊人设和业务边界，避免整套复制编码助手职责。
4. 如使用 JSON Output，系统或用户提示必须明确要求 JSON；仅设置 `response_format` 不足以防止持续空白输出。[API JSON Output 说明](https://api-docs.deepseek.com/api/create-chat-completion/)
5. 本次阅读的模型更新公告没有要求统一更换某套自然语言提示词；是否精简或重排提示词，应根据当前代码及 Harness 实际差异决定，不能只从“发了新模型”推导出全量重写。

检验重点：有/无工具的思考请求、工具后继续推理、没有实际工具调用的最终 assistant 消息在下一轮的保留、显式关闭思考、`low/high/max` 参数、缺少历史推理时的处理。

## Harness 当前源码与发布差异

已读取官方仓库当前 HEAD `687f282e8625c6b8d11ff58211a90524be2e4317`，对应最新预发布 `dsh-v0.2.1-rc.2`（10 月 17 日）。比较基点为 rc.1 `1600d8a1e140dcde43a1a0941940a0480712c011`。[发布记录](https://github.com/deepseek-ai/deepseek-harness/releases/tag/dsh-v0.2.1-rc.2)、[完整差异](https://github.com/deepseek-ai/deepseek-harness/compare/1600d8a1e140dcde43a1a0941940a0480712c011...687f282e8625c6b8d11ff58211a90524be2e4317)

- rc.1 → rc.2 中，`core/system-prompt`、`core/agent-loop`、`llm/llm-deepseek` 仅升级 package.json 版本，没有这些包的核心源码或提示词改动。这个结论仅适用于这两个版本之间，不表示 rc.1 之前没有适配工作。
- 1017 专门新增的 UI 插件给官方路由模型添加显示标识，明确不改变请求、Session 事件或缓存前缀。[显示插件说明](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/client/ui-deepseek-model-label-1017/README.zh.md)
- 新增 DeepSeek Search/Fetch 后端；旧辅助搜索 Messages 请求关闭 thinking，原因是只取搜索结果、丢弃生成答案。这不是关闭主 Agent 思考。[搜索辅助实现](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/web/web-search-deepseek/src/provider.ts)
- 当前 Harness 将身份、人设、工具规则与动态上下文分开组装；本项目已有稳定提示词/动态上下文、技能按需加载，不必重建为 Cordis 插件体系。[提示词组装说明](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/system-prompt/README.zh.md)
- Harness 使用自己的 DeepSeek Messages 适配器，保留 thinking 内容及可用的 replay signature，并检查工具调用/结果配对。不能直接把其 Messages 字段照抄到本项目 Chat Completions 请求。[序列化源码](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/serialize.ts)
- 重复工具提醒在相同工具、相同参数重复 3/5/8 次时建议检查进展或换方法，不是强制中止。[重复提醒](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/guard/repeat-tool-reminder/README.zh.md)

## 本项目需要改什么

以下是首次代码审查时的结论；首次仅做评估和本地复现，后续实施与真实验证见文末。默认模型仍是 Qwen，不能断言此前运行中的机器人已受到 DeepSeek 问题影响。

### P0：接 DeepSeek 思考模式前修复协议适配

`src/nonebot_plugin_ai_groupmate/config.py:create_chat_openai` 对所有 OpenAI 格式供应商统一构造 `ChatOpenAI`。本地安装版本为 `langchain-openai 1.2.2`，其源码明确不保留第三方 `reasoning_content`。用合成 assistant 响应测试 `_convert_dict_to_message`、`_convert_message_to_dict`、`_convert_delta_to_message_chunk`，三处得到的 reasoning_content 均为 None，没有发出网络请求。

因此，在当前路径直接切到 DeepSeek 的默认思考模式，连续工具调用存在确定的推理字段丢失，按官方契约可能触发 400；这不是提示词能修复的问题。应选择并验证供应商专用适配器，覆盖接收、流式拼接、请求回传、跨用户轮保留，以及各分群模型入口。适配器版本是否满足新契约仍需测试，不应仅凭类名认定兼容。中转商如改用其他推理字段，也需要单独验证。

### P1：明确控制思考强度和输出预算

当前配置没有 DeepSeek 的 thinking/effort 开关，仅有千问 Responses 专用开关；普通聊天默认 temperature=0.7，不能控制 DeepSeek 思考模式。建议新增供应商范围内的思考配置，日常群聊从 low 或关闭思考开始比较，复杂任务再使用 high。设置合理的输出上限，避免盲目套用官方最大上下文/最大输出或编码评测的 max。现有按流事件空闲时间计时的超时逻辑可以保留。

### P1：修正已有提示词矛盾，保留群聊定位

`agent/prompts.py` 私聊和群聊两处写“本条已经说完填 end”，但 `agent/graph.py:COMPLETION_RULE` 写“整个任务本轮已结束，不是这一句话写完”。应统一成：

> reply_user 的 next_step=end 表示当前请求已经处理完毕，并已交付结果或明确说明阻塞；仍有必要操作时先继续处理。只有还需要发送有新增信息的消息时才使用 continue。不得用“正在查、稍后告诉你”等进度承诺结束本轮。

同时明确“短句/2–3 条消息”是表达偏好，不能导致遗漏用户要求；有依赖的工具先取得上一步结果再调用下一步。现有工具成功确认、投递未知不得重试、拒绝插话、人物关系只影响语气等约束应保留，不需要用编码助手的身份或长任务默认行为替换。

### P2：补执行保护和评测

可以借鉴重复调用提醒，针对无新证据的重复检索提示换查询或报告限制；保留本项目更严格的发送/禁言等副作用去重。提示词调整是否改善新模型，应在现有 40 题 eval 上同模型、同工具、同预算比较任务完成率、提前结束率、重复发送、延迟及 token。本次没有真实模型 A/B 数据，不能声称修改后效果已提高。

建议实施顺序：DeepSeek 协议适配及离线测试 → 明确思考配置 → 统一完成语义 → 真实模型评测。没有证据要求因 1017 发布而整体重写 Agent 或整套替换提示词。

## 后续实施与真实验证（2026-10-19）

用户授权使用本机 `.env` 中的 `ds_api_key` 做验证。仅向 DeepSeek 官方 API 提交合成测试对话及公开仓库的 fixture 用例；没有发送真实群消息、执行真实群管理动作或使用生产聊天记录。生产模型配置未切换。

已实施：

- 新增 `deepseek.py` 的专用 Chat Completions 适配器，保留普通响应与流式响应的 `reasoning_content`，在工具轮与后续用户轮请求中完整回传所持有的字段，并记录 DeepSeek 缓存命中 Token。历史消息本来没有推理时传空字段，不生成或恢复不存在的推理。原有群聊历史管理未改造为持久化完整推理轨迹。
- `config.py` 对官方主机 `api.deepseek.com` 自动选用适配器；新增 `chat_deepseek_reasoning_effort`（默认 low）与 `chat_deepseek_max_tokens`（默认 8192，包含推理）。思考开启时不传 temperature，辅助角色默认关闭思考。分群/私聊工厂同样适用，其他主机不自动推断协议。
- WebUI 和 README 提供新配置；评测 runner 复用生产模型工厂，支持相同的 DeepSeek 参数。
- 统一 `next_step=end` 的任务完成语义；任务型回答默认整合成一条，资料足够时不再无关检索，不用短句风格省略必要内容，明确禁止将证据补写成未支持的事实。

本地测试先观察到普通/流式两条回归用例在旧适配器上失败，再修复验证。相关测试共 120 个不同用例通过（首批 118 个，新增入口测试后 DeepSeek 文件 12 个通过，其中 10 个已在首批；最终提示词修改后另重跑 6 个提示词测试）。Ruff 和 `git diff --check` 通过。测试覆盖实际 SDK 请求/响应转换、流式片段拼接、工具结果后回传、下一用户轮保留、思考配置、供应商隔离、分群工厂及评测工厂。

### 真实 API 冒烟

`/models` 认证通过，返回 `deepseek-flash` 与 `deepseek-v4-pro`。用 `deepseek-flash` / low / 输出上限 2048 验证：调用只读 fixture 工具 → 使用其数值回答 → 下一用户轮继续计算，三轮均完成，无协议错误。另用 none 验证非思考请求成功。第二/三轮模型可以不输出推理，这是正常结果。四次响应总 Token 1111。

记录：[冒烟结果](../evals/results/deepseek-flash-1017-smoke.json)。报告不包含密钥或推理原文。

### 同一组 16 题的提示词比较

模型为 `deepseek-flash`，思考 low，单次输出上限 4096，并发 2，每题一次；工具为模拟实现，Judge 使用同模型。覆盖 8 个 multi_tool、4 个 failure_recovery，以及停止回复、旁观沉默、单工具计算、单次回复各一题。两轮都已使用修复后的协议适配与完成语义；第二轮仅增加任务回答整合、证据足够即停止和事实约束。没有更改数据集或评分标准。

| 指标 | 第一轮 | 提示词细化后 |
| --- | ---: | ---: |
| 现有评分规则通过 | 12/16 | 16/16 |
| 平均分 | 86.28 | 97.08 |
| 平均模型调用 | 3.12 | 2.56 |
| P95 Agent 耗时（不含 Judge） | 6188 ms | 3732 ms |
| 被测 Agent 累计 Token（含重复输入） | 178767 | 148220 |
| 协议/运行错误 | 0 | 0 |

第一轮三个用例因拆成两条消息超过用例发送上限而失败，另一个因报告素材充分后仍额外检索而失败；第二轮这些硬失败消失。第二轮一次 Judge JSON 解析失败，已只对原轨迹重新判分，没有重跑 Agent 或丢弃失败轨迹；补判成功，分数不变。

记录：[第一轮](../evals/results/deepseek-flash-1017-fixtures.json)、[第二轮原始结果](../evals/results/deepseek-flash-1017-fixtures-refined.json)、[仅补判裁判解析失败的结果](../evals/results/deepseek-flash-1017-fixtures-reviewed.json)。`evals/results` 按仓库现有规则忽略提交，这些是本地验证产物。

**限制与仍存在的行为问题：** 16/16 是现有阈值的通过数，不代表所有检查满分。第二轮金额差计算正确但跳过了要求的计算工具；空结果搜图额外查了一次；年度报告把“最活跃月份”补成“基本天天冒泡”，违反新增事实约束，Judge 未识别这项补写；另有 `10 月` / `10月` 字面匹配差异造成扣分。没有为提高分数修改评测。这只是单次小样本比较，不能证明生产通过率或全面优于旧模型；目前证据支持协议修复有效，以及合并任务型回复能减少本组不必要拆条，不能声称已经消除幻觉或工具遗漏。
