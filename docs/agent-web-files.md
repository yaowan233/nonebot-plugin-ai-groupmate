# 网页与文件任务

Bot 可以读取 QQ 上传的文件、打开公开链接、从网页找到附件、汇总表格，再把结果作为文件发送到当前会话。接收和发送使用 Alconna 的统一 `File` 消息段，工具直接加入正常对话的 Agent；模型需要支持 Function Calling。

## 使用

在群聊中 @ Bot，或在私聊中直接说：

- “打开这个网页，整理价格和对应来源：<公开网页链接>。”
- “下载这个页面里的 Excel，按‘部门’汇总‘金额’，把汇总表发给我：<网页链接>。”
- “阅读这个 PDF，整理主要结论并保存成 Markdown 发给我：<PDF 直链>。”
- 后续追问：“把刚才的表按地区统计数量，再发一份 Excel。”
- 在 QQ 群上传文件，再 @ Bot：“按部门汇总刚才上传的表格，把结果 Excel 发给我。”
- 在私聊上传文件，再说：“读取刚才的文档，整理主要结论。”

普通文件消息通过 Alconna `File` 登记文件名、文件 ID、下载链接等元信息，不下载或保存原件。OneBot V11 的 QQ 群文件上传通知（`group_upload`）以及协议端提供的私聊文件通知（`offline_file` / `friend_upload`）也会转成统一文件段登记。单纯上传文件不会触发模型处理，即使正在连续对话或文件消息 @ 了 Bot；上传后请再 @ Bot 或在私聊中说明处理要求，也可以把文件与处理说明一起发送。

`list_files` 只列出元信息。用户要求读取、汇总或发送某个附件时，工具才下载选中的附件并保存原件；不会批量下载同一用户收到的其他文件。每条消息最多登记 3 个文件，重复收到同一个平台文件 ID 不重复下载或保存。文件只提供适配器内存数据、没有平台 ID 或可下载链接时，无法登记为可稍后读取的附件。

文件下载优先使用消息段的 URL；只有文件 ID 时，调用协议端的群文件/私聊文件 URL 接口，兼容 `get_file` 返回的 URL。协议端需要提供 Bot 可访问的公开 HTTP/HTTPS 下载地址；不能读取协议端返回的本地路径或内网下载地址。已知超过 10 MiB 的文件在登记时拒绝，大小未知的下载到上限后中止；登记或下载失败时模型应说明原因。上传文件按发送者隔离，群友不能让 Bot 读取其他人的上传文件。

OneBot 文件段的 `url` 为空、缺失或为 `null` 时，`file` 可能只是文件名。即使 Alconna 将其补成 `https://文件名`，也不把这个值当作下载地址，而是使用平台文件 ID 查询真实 URL。`v2.5.12` 同时兼容 `v2.5.11` 已登记的这类伪 URL 记录，可继续处理原来的附件。

网页需要登录、JavaScript 渲染或点击按钮才出现的内容不在支持范围内。抓取到的 HTML 正文与表格可以用于整理比较，表格统计工具的输入则是 CSV、TSV 或 XLSX 文件。

默认开启，无需新增 API Key。更新本仓库后执行 `uv sync` 安装新增解析依赖，再重启 Bot；本功能不新增数据库表。其他安装方式需要更新插件及其依赖。可通过配置中心“网页抓取与文件处理工具”开关，或环境变量关闭：

```dotenv
AI_GROUPMATE__AGENT_FILE_TOOLS_ENABLED=false
```

Bot 主机需要能够解析目标域名并直接访问公网 HTTP/HTTPS。下载不使用环境变量代理、登录 Cookie 或模型服务的代理配置。HTTP 下载最长 25 秒，仍受现有 Agent 总时限和工具调用时限约束；降低工具时限会提前中止下载。现有默认 8 次模型调用、20 次工具调用的预算继续生效，来源很多或需要反复翻页的任务可能需要调高预算或拆分执行。

## 工具如何衔接

| 工具 | 作用 |
| --- | --- |
| `fetch_url` | 保存网页或文档原件，返回正文预览、来源、文件 ID；HTML 同时返回下载链接等页面链接 |
| `download_file` | 下载原件并返回文件 ID；不会自动发送 |
| `read_file` | 提取正文并分页；`mode=source` 读取源码；`mode=image` 查看图片或 GIF 的指定帧（需要模型支持图片输入） |
| `list_files` | 查询收到/已保存的文件元信息，不下载附件，支持分页 |
| `summarize_table` | 对完整表格按列分组求和、计数、均值、最小值或最大值，生成 XLSX |
| `create_file` | 保存文本、表格以及 HTML、SVG、CSS、JavaScript、Python、YAML、TOML 源码 |
| `run_code` | 管理员配置执行后端后，可运行 Python、JavaScript、Shell，输入和产物均通过文件 ID 传递 |
| `send_file` | 调用当前聊天平台的文件发送接口，成功后才报告已发送 |

例如 Excel 任务会依次读取网页、取得附件链接、下载、确认表头、统计、检查汇总结果、发送文件。统计遍历完整工作表，不用正文预览代替全表计算。模型仍负责选择链接和列名；工具接通不等于所有模型都会可靠完成每种任务。

读取长正文时，`next_offset` 非空表示还有未读内容。来源、文件名、文档内容和链接均作为资料处理，不作为可以执行的指令。

## 文档支持与限制

| 类型 | 当前支持 |
| --- | --- |
| HTML | 提取静态正文、表格行和最多 500 个链接；不运行脚本，也不递归爬取整站 |
| PDF | 读取文本层，最多 200 页；不支持扫描件 OCR 或加密 PDF |
| DOCX | 读取正文段落和表格；不支持旧版 DOC、密码文档和完整排版还原 |
| XLSX | 选择工作表，首行为列名；保留公式文本，不计算公式，不执行宏 |
| CSV / TSV | 首行为列名，支持常见 UTF-8、GB18030 文本；列名需要唯一 |
| TXT / Markdown / JSON | 读取文字内容 |

单个表格最多 10,000 行数据和 100 列，提取文字最多 100 万字符。统计数值列遇到空值、公式、带单位的文字或非有限数字时会报错，不静默跳过。均值按组内行数计算；计数统计行数。结果写入 Excel 数值单元格，受 Excel 数值精度限制。生成表格时，疑似公式的文本单元格转为普通文本。

默认没有启用代码执行；配置下述执行环境后，模型可自行编写处理、转换和验证代码。对未实际解析的文件，不能声称已经读取。

## 可选的通用执行环境

`run_code` 默认关闭，需要保持 `agent_file_tools_enabled=true`。它按需加载，不限定绘图流程或修改轮数，不改变模型的思考强度。每次执行使用新工作目录，输入在 `inputs/`，产物写入 `out/`；后续执行通过返回的文件 ID 接续。容器模式的工作目录为 `/workspace`，标准镜像预装 Python、Node、Pillow、NumPy、pandas、matplotlib、openpyxl 和 Python Playwright/Chromium。本机模式使用 Bot 主机现有环境。图像模型可以通过 `read_file(mode="image", frame=0)` 查看渲染图片或动画帧；纯文本模型无法获得视觉反馈。

### 按平台选择的写入沙箱（sandbox）

统一配置，不需要子容器或 Docker-in-Docker：

```dotenv
AI_GROUPMATE__AGENT_EXECUTION_BACKEND=sandbox
```

| Bot 实际运行平台 | 自动选择 | 依赖 |
| --- | --- | --- |
| Linux x86_64/aarch64，64 位 Python | Landlock | 内核启用 ABI 3 或以上，允许相关系统调用 |
| macOS | Seatbelt | 系统 `/usr/bin/sandbox-exec` 可用 |
| Windows | DSH ACL 受限令牌 | Node.js 与 `@deepseek-ai/dsh-sandbox-windows-acl` |

Docker Desktop 的 Linux 容器和 WSL 按 Linux 处理，不按宿主桌面系统处理。缺少依赖、不支持的平台或限制初始化失败，均拒绝执行，**不自动降级为 local**。不要求特权容器，也不会为了兼容自动放宽权限。

Windows 可复用已有 DSH 安装，或者由管理员安装独立运行器（本项目验证版本为 `0.2.1-rc.2`）：

```sh
npm install -g @deepseek-ai/dsh-sandbox-windows-acl@0.2.1-rc.2
```

默认从 Node 安装目录、用户 npm 全局目录及其中 DSH 的依赖目录查找；不从模型任务目录加载运行器。自定义 npm 安装目录可显式指定运行器绝对路径：

```dotenv
AI_GROUPMATE__AGENT_EXECUTION_SANDBOX_WINDOWS_RUNNER=C:/path/to/node_modules/@deepseek-ai/dsh-sandbox-windows-acl/lib/runner.js
```

任务运行时不自动安装 npm 包。Node 和运行器必须对 Bot 服务账号可用；Windows ACL 模式会修改本次临时工作目录的安全描述符，按 DSH 设计保留工作目录授权到目录删除，并在退出时尝试撤销私有临时区授权。不会对仓库或用户指定目录添加授权。为兼容 Python 3.13 的私有临时目录 ACL，启动前只在本次新建目录上为当前用户添加显式 Modify 权限；降权后的限制仍由 DSH 的能力 SID 与完整性级别共同执行，不给 Everyone 增加允许权限。

三个平台都在执行模型代码前施加文件写入限制，后代进程继承限制。允许写入本次工作目录及私有临时区，Unix 另允许 `/dev/null`。文件内容写入、创建、删除等操作受到平台策略限制；日志、输出文件大小和执行时限沿用现有接口。Windows Shell 为 PowerShell，Linux/macOS 为 `/bin/sh`。

这是 DSH 式的**文件写入限制**，不是完整容器隔离：读取和网络仍可用，不提供统一的 IPC、文件元数据、CPU、内存或磁盘隔离。Windows 后端还有上游记录的硬链接和 ACL 边界。凭据读取风险仍存在；宿主机与资源隔离由外层运行环境负责。Linux 使用独立实现的 Landlock UAPI 启动器，macOS 的策略参考 DSH，Windows 调用上游运行器。许可声明见 [DSH MIT notice](../LICENSES/DSH-MIT.txt)。参考 [Linux 官方说明](https://docs.kernel.org/userspace-api/landlock.html)、[DSH 文件策略](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/sandbox/sandbox-local/src/profiles.ts) 和 [Windows 后端说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/sandbox/sandbox-windows-acl/README.md)。

验证：`tests/test_execution_sandbox.py` 包含不可用时拒绝执行，以及各平台的真实限制测试。运行在其他系统上的内核测试会明确跳过；Windows 缺少运行器、Linux 内核未开放 Landlock 时也会跳过对应实测，不能据此声称该部署环境已经验收通过。

### 本机执行（local）

无需 Docker，显式配置后重启 Bot：

```dotenv
AI_GROUPMATE__AGENT_EXECUTION_BACKEND=local
AI_GROUPMATE__AGENT_EXECUTION_TIMEOUT_SECONDS=20
```

Python 使用运行 Bot 的同一解释器及依赖；JavaScript 使用 PATH 中的 Node.js；`shell` 在 Windows 使用 PowerShell（优先 pwsh），在 Linux/macOS 使用 `/bin/sh`。每次运行在新的临时目录中，使用相对路径 `inputs/` 和 `out/`。需要跨次保留的文件必须放到 `out/`，再通过文件 ID 传入下一次运行。缺失 Node、Playwright 或浏览器时，需管理员在本机准备对应依赖；后端不自动安装。

**本机后端不是沙盒**：代码拥有 Bot 运行用户的文件和网络权限，没有容器的 CPU、内存、磁盘限额。它仅适用于可信使用者和允许本机代码执行的部署。子进程不会直接继承 API key 等任意 Bot 环境变量，但这不阻止代码通过主机文件访问凭据。临时目录也不构成文件访问隔离。

超时或取消会尝试结束进程树并清理临时目录（Windows 使用 taskkill，POSIX 使用进程组）；主动脱离进程树的程序不能保证被回收。返回日志和文件仍有大小限制。此配置不改变默认模型或思考强度。

### Bot 能访问 Docker

在 Docker 主机上从仓库根目录构建镜像：

```sh
docker build -t ai-groupmate-executor:local -f deploy/executor/Dockerfile .
```

配置 Bot 并重启：

```dotenv
AI_GROUPMATE__AGENT_EXECUTION_BACKEND=docker
AI_GROUPMATE__AGENT_EXECUTION_IMAGE=ai-groupmate-executor:local
AI_GROUPMATE__AGENT_EXECUTION_TIMEOUT_SECONDS=20
```

Bot 进程需要 Docker CLI 和 daemon 访问权限。文件通过 `docker exec` 中的 tar 流写入和读取容器的 tmpfs 工作目录，不依赖 Bot 与 Docker 主机共享目录；执行镜像需要提供 `tar`。Bot 本身在容器中时，也可使用下面的独立服务方式。

构建镜像后，可运行 `uv run pytest tests/test_execution_docker.py` 验证真实容器的内联代码、输入/输出文件传输和输出目录删除错误。默认使用 `ai-groupmate-executor:local`，可通过 `AI_GROUPMATE_TEST_EXECUTION_IMAGE` 指定已存在的兼容镜像。Docker 或镜像不可用时明确跳过，测试不会拉取镜像。

### Bot 通过独立执行服务访问

在有 Docker 的机器构建上述镜像，并准备 Python 和 `httpx`。将 `agent/execution_service.py` 与 `agent/execution_backend.py` 放在同一目录，设置以下服务环境变量后运行 `python execution_service.py`：

```dotenv
EXECUTION_SERVICE_TOKEN=<至少16字符的随机令牌>
EXECUTION_SERVICE_HOST=127.0.0.1
EXECUTION_SERVICE_PORT=8091
EXECUTION_SERVICE_IMAGE=ai-groupmate-executor:local
EXECUTION_SERVICE_TIMEOUT=20
```

服务默认仅监听本机。跨机器或跨容器部署时，改为实际监听地址，并通过受限的内网或 HTTPS 反向代理提供给 Bot；不要直接向公网开放开发用 HTTP 服务。Bot 中配置：

```dotenv
AI_GROUPMATE__AGENT_EXECUTION_BACKEND=remote
AI_GROUPMATE__AGENT_EXECUTION_URL=http://executor:8091
AI_GROUPMATE__AGENT_EXECUTION_TOKEN=<与服务相同的令牌>
AI_GROUPMATE__AGENT_EXECUTION_TIMEOUT_SECONDS=20
```

`executor` 是示例服务名，须替换为 Bot 实际可达的地址；Bot 容器中的 `localhost` 指向 Bot 容器自身。此方式不需要给 Bot 安装 Docker、挂载 Docker socket 或共享文件目录。服务端仍需 Docker CLI 和 daemon 权限。通过容器运行服务时应自行配置这些管理权限；执行任务的临时容器不会获得它们。

### 限制与验证

临时容器无网络、无宿主机目录挂载，根文件系统只读，限制为 1 CPU、512 MiB 内存、128 个进程；工作目录和临时目录为限额内存文件系统。不传入 Bot 的环境变量或密钥。依赖需要在管理员构建镜像时安装，任务中不能联网安装。Docker 隔离依赖宿主机和 daemon 的安全配置。

单次最多传入 12 个文件、总计 10 MiB；输出最多 12 个文件，单个 10 MiB、总计 20 MiB，同时计入既有文件工作区额度。输出子目录会展开为文件名，不允许同名冲突或符号链接。超时和取消会请求删除容器；容器自身也有到时退出机制。服务进程异常退出后可能留下已停止的容器，可由管理员清理 `groupmate-job-` 前缀的容器。

执行时限同时受 Agent 工具时限与总预算约束；本机和直接 Docker 模式会预留清理时间。远程服务使用服务端时限，建议与 Bot 配置一致，调高执行时限时也应相应调高 `agent_tool_timeout_seconds`。自动化测试覆盖协议、鉴权、资源限制参数、超时清理和模拟模型的生成→看图→修改→交付流程；真实 Docker/Chromium 和聊天平台投递仍需在部署环境验收。

## 文件保存与交付

- 文件存放在插件 localstore 数据目录下的 `agent_files`，按 Bot、群聊/私聊、会话、发起用户隔离；其他群友不能直接读取该用户的文件工作区。
- 工具只接受不透明的文件 ID，不接受本机路径。文件保留 24 小时；到期后立即不可读，写入时和每 30 分钟的后台清理会删除到期原件。
- 未下载附件的元信息保留 24 小时，每个用户在一个会话中最多保留 30 条，全插件最多 1000 条；超过数量上限时移除最早的元信息。这些记录只包含下载所需信息，不包含文件正文。
- 单个文件最多 10 MiB；每个用户在一个会话中最多 30 个文件、50 MiB；整个插件文件区最多 512 MiB。下载原件和生成文件都计入额度。
- 仅访问公开 HTTP/HTTPS 的 80/443 端口；解析域名并校验每次重定向，拒绝本机和内网地址。下载有体积限制，Office 解压内容也有限制。
- 每轮最多尝试发送 3 个文件。同一文件本轮不重复发送；接口异常时保留“未确认送达”状态，避免重复投递。
- 文件投递结果未知时，停止文件操作，只尝试发送一次“文件未确认送达”的文字提示；提示失败也不重试。请求已经过期时不再发送提示。

文件投递依赖平台适配器。OneBot V11 使用 Alconna 自定义 `File` 导出，将已验证工作区文件转成 `base64://` 数据交给群文件/私聊文件上传接口，协议端无需访问 NoneBot 的本地路径，也无需共享文件目录。单文件仍限制为 10 MiB，不使用文字链接代替文件；协议端需支持 OneBot 的 Base64 文件输入。其他适配器沿用普通 Alconna `File` 发送方式，具体支持和路径要求由适配器决定。

## 验证范围

自动化测试覆盖 Alconna 文件消息、QQ 群/私聊文件通知、文件 URL 接口和兼容回退、重复上传、下载期间追问、关闭接收开关，以及实际 PDF、DOCX、XLSX、CSV 解析与生成、完整表格计算、分页、文件隔离和过期、下载地址与大小限制、发送失败状态，以及真实 Agent 图中的“网页 → 附件下载 → 表格统计 → 文件发送”流程。此流程使用预设模型响应、HTTP 模拟服务和模拟聊天接口，不能据此推断真实模型的任务成功率或 QQ 投递成功。

2026-10-13 使用真实 `deepseek-flash` 和公网 HTTP 完成 CSV、XLSX、网页找附件、后续追问和失败处理验收：244 行 CSV 和 700 行 Excel 的汇总数值与独立计算一致。文件投递未知路径重复测试，均只尝试发送文件一次并给出状态提示。聊天发送接口在验收中被模拟，QQ 实际收件及 Bot/协议端共享路径仍需部署后验证。

同日补充上传文件验收，4/4 通过：普通 OneBot 文件消息经 Alconna 转换，以及 QQ 群文件通知形状，分别处理 244 行 CSV 和 700 行 XLSX。模型先 `list_files` 找回附件元信息，再按需下载、读取、完整汇总、检查并发送，生成数值与独立计算一致，同一工作区的无关附件保持未下载。模型、公开样本下载和文件处理真实执行；QQ 事件由脚本构造，协议端 URL 查询和文件发送采用模拟，尚未验证真实 QQ 上传与收件。验收机器使用代理的 fake-IP DNS，脚本通过公开 DNS 查询取得真实公网地址；生产代码的公网地址校验保持开启。

随后用用户环境捕获到的真实私聊文件段补充回归，覆盖空/缺失/null URL、群聊与私聊、旧附件记录，以及真实 URL 与文件名相同的情况。通过真实 LLBot 协议端 `get_private_file_url` 查询接口取得 QQ 下载地址，使用 Bot 容器中的公网下载器完整下载 29,253 字节，与上传记录的文件大小一致；临时本机查询接口在验证后恢复。此项验证覆盖协议端查询和实际下载，不代表 QQ 文件发送已经验收。

补充发送回归：复现用户环境中“Excel 已生成，但协议端容器找不到 NoneBot 本地路径”的错误。使用真实 Alconna OneBot V11 导出器，模拟没有共享目录的协议端，验证群聊/私聊均收到可解码并用 openpyxl 打开的完整 XLSX，且保持每轮只发送一次；另验证非 OneBot V11 沿用原导出器，以及内联数据的大小上限。此回归的 QQ 上传接口为模拟，真实 QQ 收件需单独验收。
