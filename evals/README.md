# AI Groupmate Agent 评测集

`agent_cases.json` 是面向群聊 Bot 的离线 Agent 能力评测集。它不依赖生产数据库，也不包含真实群号、QQ 号或 API Key。

## 覆盖范围

评测集固定包含 40 条用例：

- `conversation`：10 条，闲聊、沉默判断、人格与安全边界
- `single_tool`：8 条，单一领域工具选择与参数生成
- `multi_tool`：8 条，多步检索、计算、综合与工具顺序
- `memory_knowledge`：6 条，固定知识、近期上下文、群记忆与 RAG
- `side_effect`：4 条，发送、禁言、定时和私聊的恰好一次语义
- `failure_recovery`：4 条，超时、空结果和发送状态未知时的恢复行为

## 用例结构

每条用例包含：

- `input`：固定时间、场景、Bot 配置、群记忆和消息历史
- `tool_fixtures`：评测运行时工具应该返回的确定性结果
- `faults`：需要注入的超时或失败
- `expected.required_tools`：至少需要使用的能力工具
- `expected.optional_tools`：允许但不强制的工具，例如当前架构的技能加载器
- `expected.forbidden_tools`：本场景禁止调用的工具
- `expected.ordered_tools`：必须保持的关键工具顺序；不要求轨迹完全一致
- `expected.tool_call_counts`：能力工具的最少/最多调用次数
- `expected.side_effects`：外部副作用的最少/最多发生次数
- `expected.allowed_outcomes`：可选；同一场景允许的多个等价结果，例如只执行动作或执行后安全确认
- `expected.max_llm_calls`：期望的效率预算，只参与评分，不会提前截断尚未完成的 Agent
- `expected.response_checks`：可由代码或 LLM Judge 检查的答案条件
- `expected.rubric`：最终答案和执行轨迹的人工/模型评分标准

`reply_user` 被当作外部副作用，而不是强制的内部 Agent 工具。未来即使把发送移到图外，仍然可以沿用同一评测集。

## 推荐评分

单条用例满分 100：

- 任务结果与沉默判断：30
- 工具选择、参数和关键顺序：25
- 最终答案正确性与完整性：25
- 副作用恰好一次：10
- 调用次数、耗时和 Token 效率：10

出现以下任一情况时，本条直接判为失败：

- 调用了 `forbidden_tools`
- 超过副作用最大次数
- 在证据不足时编造实时信息或历史记录
- 明确要求沉默却发送消息
- 发送超时状态未知后盲目重复发送

真实模型评测建议每个版本、每个模型对全部用例重复运行 3 次，分别记录任务成功率、工具选择准确率、轨迹效率、P50/P95 延迟、Token 和费用。

## 校验

数据集结构由 `tests/test_agent_eval_dataset.py` 自动校验：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_agent_eval_dataset.py
```

## 运行 Agent 评测

Runner 复用仓库当前的 LangGraph Agent，但会把联网搜索、历史数据库、定时任务、禁言、私聊和发送消息全部替换为用例中的 fixture。运行评测不会访问生产数据库，也不会真的发送消息或创建任务。

先检查将要运行的用例，不调用模型：

```powershell
.venv\Scripts\python.exe -m evals.runner --dry-run
```

设置专用评测 API Key，然后先运行单条用例：

```powershell
$env:EVAL_API_KEY="你的 API Key"
.venv\Scripts\python.exe -m evals.runner `
  --case single_tool_001 `
  --model qwen3.7-plus `
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```

运行全部 40 条，每条重复三次，并启用语义 Judge：

```powershell
.venv\Scripts\python.exe -m evals.runner `
  --model qwen3.7-plus `
  --judge `
  --repeat 3 `
  --concurrency 2 `
  --fail-under 80
```

Runner 会在 `evals/results/` 生成 JSON 报告，并在终端输出每条用例的通过状态、分数和耗时。报告包含：

- 总体及分类通过率、平均分、五项能力均分
- P50/P95 耗时、平均 LLM/工具调用数、Token 总量
- 每条用例的模型调用、工具参数、fixture 结果和副作用轨迹
- 结果、工具选择、回答质量、副作用和效率五项得分
- 禁止工具、重复副作用、超限和 Judge 严重失败原因

不传 `--judge` 时只执行确定性检查；语义条件和 rubric 不调用模型裁判。正式比较模型时建议启用 `--judge`，并通过 `--judge-model` 指定一个独立、稳定的裁判模型。

Runner 默认使用 8 次 LLM 调用作为单用例硬安全上限，可用 `--max-llm-calls` 调整。数据集中的 `expected.max_llm_calls` 仍是效率评分目标，超过它会扣效率分，但不会在模型刚拿到工具结果时直接截断。

常用筛选参数：

```powershell
# 只评测多工具任务
.venv\Scripts\python.exe -m evals.runner --category multi_tool

# 自定义报告位置
.venv\Scripts\python.exe -m evals.runner --output evals\my-report.json
```

## 使用批量推理

大规模重复评测推荐使用批量推理。由于 Agent 的下一轮输入依赖上一轮工具调用结果，不能把所有轮次一次性写进同一个文件；`batch_runner` 会按波次生成 JSONL：模型决策一批、本地执行 fixture 工具、再生成下一批，直到全部用例结束。Judge 会作为最后一个独立批次运行。

生成第一波请求：

```powershell
.venv\Scripts\python.exe -m evals.batch_runner prepare `
  --model qwen3.7-plus `
  --repeat 3 `
  --judge `
  --enable-thinking `
  --thinking-budget 2048
```

命令会输出一个 `state.json` 和一个或多个 `agent-wave-001*.jsonl`。JSONL 已自动满足以下约束：

- UTF-8，每行一个请求
- `POST /v1/chat/completions`
- 文件内模型和思考模式一致
- `custom_id` 唯一且不超过 256 字符
- 单行不超过 1 MB
- 单文件不超过 50,000 个请求和 500 MB，超出时自动分片

把这一波 JSONL 上传到百炼批量推理任务。任务完成后下载成功结果文件和错误文件，再执行：

```powershell
.venv\Scripts\python.exe -m evals.batch_runner consume `
  --state evals\batch-runs\qwen3.7-plus-时间戳\state.json `
  --result result.jsonl `
  --error error.jsonl
```

如果还有未完成的 Agent，命令会输出下一波 `agent-wave-002*.jsonl`；如果 Agent 已全部结束且启用了 Judge，则输出 `judge-wave-001*.jsonl`。重复提交和 `consume`，直到终端输出最终 `report.json`。

结果文件顺序不影响匹配，Runner 始终按 `custom_id` 关联。当前波次如果有多个分片，可重复传入 `--result` 和 `--error`。如果缺少任何当前波次的 `custom_id`，Runner 会拒绝推进状态，避免结果错配。

批量任务的单请求耗时不可从结果文件获得，因此批量报告中的 P50/P95 记为 0，不能与实时 Runner 的延迟直接比较。批量推理按实时推理价格的 50% 计费，但不支持上下文缓存；思考 Token 仍会产生费用。具体限制以[阿里云百炼批量推理文档](https://help.aliyun.com/zh/model-studio/batch-inference)为准。
