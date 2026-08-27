# Qwen3.8-Max Responses API 官方内置工具调研

> 调研日期：2026-08-27  
> 资料范围：仅使用阿里云百炼/通义千问第一方官方文档。价格和“限时免费”状态可能变化，生产环境应以调用地域的百炼控制台与官方价格页为准。

## 结论

可以接入。`qwen3.8-max` 是阿里云官方文档为代码解释器、联网搜索、网页抓取、文搜图和图搜图列出的推荐模型，用户给出的 `client.responses.create(...)` 写法也是官方 Python 示例采用的请求形式。

但不能只把这几个 `tools` 塞进当前的 Chat Completions 请求：这是 **Responses API 的服务端内置工具**。接入方需要增加 Responses 协议调用路径、解析 Responses 的 `output`/流式事件，并通过 `usage.x_tools` 统计实际工具调用次数。文搜图和图搜图在普通按量 Responses API 中的正式类型名也不是 `t2i_search`、`i2i_search`。

`t2i_search`、`i2i_search` 是 **Token Plan 个人版 Harness** 页面使用的工具名。该产品的[订阅前须知](https://help.aliyun.com/zh/model-studio/token-plan-personal-overview)明确禁止把套餐 API Key 用于自动化脚本、自定义应用后端或非交互批量调用。因此本机器人不能按 Token Plan 的工具名和套餐 Key 实现，应接普通百炼按量 Responses API，使用 `web_search_image`、`image_search`。

官方名称与请求值映射如下：

| 用户/计费侧叫法 | Responses API 的 `tools[].type` | 输出项 `type` | 北京工具价 |
| --- | --- | --- | --- |
| 代码解释器 | `code_interpreter` | `code_interpreter_call` | 限时免费 |
| 联网搜索 | `web_search` | `web_search_call` | 4 元/千次 |
| 网页抓取 | `web_extractor` | `web_extractor_call` | 限时免费 |
| 文搜图（Token Plan Harness 名为 `t2i_search`） | `web_search_image` | `web_search_image_call` | 24 元/千次 |
| 图搜图（Token Plan Harness 名为 `i2i_search`） | `image_search` | `image_search_call` | 48 元/千次 |

上述请求值、输出类型来自[创建响应](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses)、[文搜图](https://help.aliyun.com/zh/model-studio/web-search-image)和[图搜图](https://help.aliyun.com/zh/model-studio/image-search)；价格来自各工具官方页面，详见下文。

## Python SDK 请求格式

北京地域的官方写法如下。`{WorkspaceId}` 必须替换为实际业务空间 ID：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=(
        "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1"
    ),
)

response = client.responses.create(
    model="qwen3.8-max",
    input="12的3次方",
    tools=[
        {"type": "code_interpreter"},
        {"type": "web_search"},
        {"type": "web_extractor"},
    ],
    extra_body={"enable_thinking": True},
)

print(response.output_text)
print(response.usage.x_tools)
```

这与[代码解释器官方示例](https://help.aliyun.com/zh/model-studio/qwen-code-interpreter)一致。`qwen3.8-max` 使用代码解释器时需要开启思考模式。`enable_thinking` 是百炼扩展参数，Python SDK 需要放在 `extra_body`；不过[创建响应参数文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses)已经建议新代码优先使用 `reasoning.effort`，并说明 `enable_thinking` 后续将不再支持。两者同时存在时，`reasoning.effort` 优先。

推荐的新式写法可表达为：

```python
response = client.responses.create(
    model="qwen3.8-max",
    input="12的3次方",
    tools=[
        {"type": "code_interpreter"},
        {"type": "web_search"},
        {"type": "web_extractor"},
    ],
    reasoning={"effort": "high"},
)
```

工具列表表示“允许模型使用”，并不等于每次都会调用全部工具。模型会根据任务自主决定是否调用以及调用几次；代码解释器的执行与结果整合阶段也可能循环多次。[代码解释器文档](https://help.aliyun.com/zh/model-studio/qwen-code-interpreter)给出的例子中，同一个请求运行了两次代码解释器，`usage.x_tools.code_interpreter.count` 因而为 2。

## 五种工具的行为与限制

### `code_interpreter`

- 由模型在百炼沙箱内生成并运行 Python，适合精确计算和数据分析。
- Responses 输出依次可能包含 `reasoning`、一个或多个 `code_interpreter_call`、最终 `message`；最终文本仍可直接通过 `response.output_text` 读取。
- 单个用户请求可能触发多轮模型推理和多次代码执行，`usage` 汇总所有 Token，`usage.x_tools` 记录工具次数。
- 官方特别注明代码解释器与 Function Calling 互斥，同时启用会报错。因此不能假设它能和现有自定义 function 工具无条件混用。
- 工具调用限时免费，但工具产生的额外推理和上下文 Token 仍按模型价格计费。

来源：[代码解释器](https://help.aliyun.com/zh/model-studio/qwen-code-interpreter)。

### `web_search`

- 模型自主判断是否需要联网；仅提供工具并不保证触发。需要强制搜索时，官方文档要求在 `search_options` 中设置 `forced_search: true`。
- 搜索来源位于 `response.output` 中 `type == "web_search_call"` 项的 `action.sources`。
- Responses API 暂不支持 `enable_source`、`enable_citation`、`citation_format`，不会自动在回答正文插入 `[1]` 角标；应用若要展示来源，需要自行读取 `action.sources` 并渲染。
- 联网搜索限流为账号级 15 RPS。超限时不会报错，而是跳过搜索链路，所以不能只靠异常判断工具是否执行。
- 北京按 Responses API 对应的 agent 策略为 4 元/千次，新加坡为 73.392381 元/千次；检索内容还会增加输入 Token 费用。该内置搜索与“联网搜索 MCP”是两个独立产品，不能混用其免费额度或价格。

来源：[联网搜索](https://help.aliyun.com/zh/model-studio/web-search/)。

### `web_extractor`

- 必须和 `web_search` 一起放进 `tools`，不能单独启用。
- 用来访问并抽取网页内容；Responses 输出以 `web_extractor_call` 标识。
- 官方推荐涉及网页与计算的复杂任务同时开启 `web_search`、`web_extractor`、`code_interpreter`。
- 网页抓取当前限时免费；联网搜索仍按北京 4 元/千次或新加坡 73.392381 元/千次收费，抓取回来的网页内容会增加模型输入 Token。

来源：[网页抓取](https://help.aliyun.com/zh/model-studio/web-extractor)。

### `web_search_image`（文搜图，不是 `t2i_search`）

- 按文本描述搜索互联网上已有图片，不是图片生成。
- 请求为 `tools=[{"type": "web_search_image"}]`；输出项为 `web_search_image_call`。
- 官方将 Qwen3.8-Max 系列列为推荐模型，并说明该工具仅支持 Responses API。
- 北京 24 元/千次，新加坡 58.713905 元/千次；搜索结果拼接到上下文后还会产生模型 Token 费用。
- 工具较慢，官方建议使用流式输出；完成后可从 `usage.x_tools.web_search_image.count` 获取调用次数。

来源：[文搜图](https://help.aliyun.com/zh/model-studio/web-search-image)。

### `image_search`（图搜图，不是 `i2i_search`）

- 根据输入图片搜索互联网中的相似或相关图片，不是图生图。
- `input` 必须含 `input_image`，官方 OpenAI SDK 示例使用公网图片 URL；可同时提供 `input_text` 补充搜索意图。
- 请求为 `tools=[{"type": "image_search"}]`；输出项为 `image_search_call`。工具调用的 `arguments` 可包含图片索引 `img_idx` 和搜索区域 `bbox`。
- 每次工具执行只搜索一张图片，但模型可在一次请求内多次调用；返回图片数由模型决定，最多 100 张。OpenAI SDK 不支持用本地文件路径直接传入。
- 北京 48 元/千次，新加坡 58.713905 元/千次；搜索结果同样增加模型输入 Token 费用。

示例：

```python
response = client.responses.create(
    model="qwen3.8-max",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "找相似风格的图片"},
                {"type": "input_image", "image_url": "https://example.com/a.png"},
            ],
        }
    ],
    tools=[{"type": "image_search"}],
)
```

来源：[图搜图](https://help.aliyun.com/zh/model-studio/image-search)。

## 支持模型、地域与 Endpoint

`qwen3.8-max` 的正式模型 ID 就是 `qwen3.8-max`。Responses 兼容文档在华北 2（北京）、新加坡、美国（弗吉尼亚）、德国（法兰克福）、日本（东京）的支持模型列表中都列出了它；各地域 Endpoint 如下：

| 地域 | OpenAI SDK `base_url` |
| --- | --- |
| 华北 2（北京） | `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| 新加坡 | `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| 美国（弗吉尼亚） | `https://{WorkspaceId}.us-east-1.maas.aliyuncs.com/compatible-mode/v1` |
| 德国（法兰克福） | `https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1` |
| 日本（东京） | `https://{WorkspaceId}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1` |

HTTP 创建响应是在相应地址后追加 `/responses`。官方还说明旧路径 `/api/v2/apps/protocols/compatible-mode/v1/responses` 即将停止维护，应使用 `/compatible-mode/v1/responses`。北京和新加坡旧的通用域名目前仍可使用，但官方建议迁移到业务空间专属域名。

来源：[OpenAI Responses 接口兼容与迁移指南](https://help.aliyun.com/zh/model-studio/compatibility-with-openai-responses-api)、[qwen3.8-max 模型信息](https://help.aliyun.com/zh/model-studio/qwen3-8-max)。

需要区分“该地域存在 qwen3.8-max Responses Endpoint”和“该地域明确公布了某项工具价格”。五个工具的专项文档目前明确给出的是北京、新加坡价格；未给出美国、德国、日本的工具价格。因此首版若要给用户展示确定的费用说明，宜只展示当前地域的官方价格，不应把北京价格外推到其他地域；其他地域能力与价格应以当地百炼控制台为准。

## 响应结构和流式事件

非流式响应的常用读取方式：

```python
print(response.output_text)

for item in response.output:
    print(item.type, item.status)

tool_counts = response.usage.x_tools or {}
```

`output` 是有顺序的输出项数组，可能包含：

- `reasoning`：思考输出；
- `code_interpreter_call`、`web_search_call`、`web_extractor_call`、`web_search_image_call`、`image_search_call`：服务端工具执行记录；
- `message`：最终助手消息，正文内容项类型为 `output_text`。

`usage` 除输入、输出、总 Token 外，还会以 `x_tools` 给出实际调用次数，例如：

```json
{
  "input_tokens": 8371,
  "output_tokens": 417,
  "total_tokens": 8788,
  "x_tools": {"image_search": {"count": 1}}
}
```

流式模式使用 `stream=True`，返回的是 Responses 事件而不是 Chat Completions 的 `choices[].delta`。常见事件包括 `response.output_item.added`、各工具的进行中/完成事件、`response.output_text.delta`、`response.output_item.done`，最后以 `response.completed` 返回完整 Response 和 `usage`。因此现有只解析 `choices[0].delta.content` 的代码不能直接复用。

来源：[创建响应](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses)、[获取响应](https://help.aliyun.com/zh/model-studio/retrieve-a-response)。

## 与 Chat Completions 的关键差异

| 项目 | Responses API | Chat Completions |
| --- | --- | --- |
| 五种工具统一调用 | 通过 `tools[].type` 使用服务端内置工具 | 不具备这套统一的内置工具协议；个别能力有各自扩展参数 |
| 输入 | 可直接传字符串，也可传消息/多模态内容数组 | 主要使用 `messages` |
| 输出 | `output` 多类型数组，最终文本有 `output_text` 便捷属性 | `choices[].message.content` |
| 多轮上下文 | 可用 `previous_response_id` 关联上一轮（官方说明响应 ID 有效期 7 天） | 应用通常自行重传完整消息历史 |
| 流式解析 | 事件类型驱动，如 `response.output_text.delta` | `choices[].delta` |
| 工具执行 | 百炼服务端执行内置工具，应用读取执行记录与最终结果 | 自定义 function 通常由应用执行并回传结果 |

代码解释器虽然也有 Chat Completions 扩展方式，但官方要求 `enable_code_interpreter=true`、思考模式和流式调用，而且 Chat Completions 无法获得代码解释器运行代码；Responses API 才会返回 `code_interpreter_call`。文搜图和图搜图官方明确为 Responses-only。迁移收益与限制见[Responses 迁移指南](https://help.aliyun.com/zh/model-studio/compatibility-with-openai-responses-api)和[代码解释器](https://help.aliyun.com/zh/model-studio/qwen-code-interpreter)。

## 对本项目接入的注意事项

1. 增加显式协议选项，例如 `chat_completions` / `responses`。仅从模型名猜协议会让同一模型在不同兼容服务上的行为不可控。
2. 内置工具应做成用户可选择的白名单，默认不开启收费工具，并在界面直接显示“按实际工具调用次数 + 额外 Token”计费。
3. 配置层使用官方 API 名 `web_search_image`、`image_search`；界面可显示“文搜图（t2i search）”“图搜图（i2i search）”，但不可把显示别名直接发给 API。
4. Qwen3.8-Max + `code_interpreter` 需要思考模式。新实现优先使用 `reasoning.effort`，兼容旧配置时再回退到 `extra_body.enable_thinking`。
5. `web_extractor` 勾选时必须自动同时加入 `web_search`，并提示会产生搜索费；不能把“网页抓取限时免费”显示成整个请求免费。
6. 不要让 `code_interpreter` 与现有 Function Calling 同时启用，官方明确指出两者互斥。
7. 以 `usage.x_tools` 作为工具次数审计与费用提示依据，不能以请求中的工具列表计数，因为模型可能不调用、调用一次或调用多次。
8. 若要在机器人回复中展示搜索引用，解析 `web_search_call.action.sources`；`response.output_text` 本身不会自动带官方角标。
9. 流式和非流式都要解析工具输出项，同时为未知 `output.type` 保留向前兼容处理。百炼只处理其文档明确列出的参数，其他 OpenAI Responses 参数可能被忽略；例如当前不支持 `background` 异步执行。
10. 图片搜索会把外部 URL 和搜索结果带入第三方服务流程，应在私聊/群聊配置说明中提示隐私、版权和额外费用；图搜图不能接受 SDK 本地路径，需先得到可访问的 URL 或使用官方支持的文件传入方式。
11. 不支持把 Token Plan 个人版 Key 接入机器人后端；该套餐官方只允许在指定类型的交互式编程/智能体工具中使用。机器人应使用普通按量 API Key 与对应地域的业务空间 Endpoint。

## 费用核对（2026-08-27）

| 工具 | 华北 2（北京） | 新加坡 | 额外模型 Token |
| --- | ---: | ---: | --- |
| `code_interpreter` | 限时免费 | 官方专项页仅写“限时免费”，未分地域价格 | 会增加 |
| `web_search` | 4 元/千次 | 73.392381 元/千次 | 搜索内容计入输入 |
| `web_extractor` | 限时免费 | 限时免费 | 抓取内容计入输入；同时启用的搜索另收费 |
| `web_search_image` | 24 元/千次 | 58.713905 元/千次 | 搜索结果计入输入 |
| `image_search` | 48 元/千次 | 58.713905 元/千次 | 搜索结果计入输入 |

所以用户列出的“代码解释器限时免费、网页抓取限时免费、联网搜索 4 元/千次、文搜图 24 元/千次、图搜图 48 元/千次”与 **北京地域当前官方文档**一致。这里的免费只指工具调用费，不包含模型 Token；“限时”页面没有给出固定结束日期，不能在代码或文案中承诺长期免费。

价格来源：[代码解释器](https://help.aliyun.com/zh/model-studio/qwen-code-interpreter)、[联网搜索](https://help.aliyun.com/zh/model-studio/web-search/)、[网页抓取](https://help.aliyun.com/zh/model-studio/web-extractor)、[文搜图](https://help.aliyun.com/zh/model-studio/web-search-image)、[图搜图](https://help.aliyun.com/zh/model-studio/image-search)。
