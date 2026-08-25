# 分群模型 API 中转配置设计

> 状态：插件端、中转服务和浏览器配置页均已实现
> 范围：定义并实现插件与中转服务之间的注册、配置提交和兑换协议
> 代码位置：插件位于当前仓库；中转服务位于 `../backend`；配置页位于 `../mayumi-front`

## 当前实现状态

插件端已经包含：

- 实例首次注册、身份复用和本地加密存储；
- 创建配置单、一次性配置码兑换、浏览器密文解密和处理结果确认；
- 群主、群管理员和超级用户权限校验；
- 群模型配置连接测试、加密持久化、查看和删除命令；
- 主聊天模型按群解析和缓存失效，未配置群继续使用全局配置；
- 可替换的中转 transport 与不访问公网的协议测试。

配套实现已经包含：

- FastAPI `/v1` 注册、配置单、密文提交、兑换与确认接口；
- SQLite 密文信箱、HMAC 凭据摘要、TTL 清理、应用层与 Nginx 双层限流；
- 与现有 Mayumi UI 一致的 HeroUI 配置页；
- 浏览器端 RSA-OAEP-256 + AES-256-GCM 加密和一次性配置码展示；
- Docker、Nginx、环境变量与上线检查文档。

## 背景

`nonebot-plugin-ai-groupmate` 通常运行在家庭网络、容器网络或其他没有公网入站地址的环境中。现有配置中心只能由能够访问 Bot WebUI 的部署者使用，群管理员无法安全地为自己的群填写模型 API Key。

本设计引入一个公网中转服务。内网 Bot 只主动发起 HTTPS 请求，不接收公网回调。群管理员在网页填写模型配置后取得一次性配置码，再将配置码私聊发送给 Bot。中转服务只转交浏览器加密后的密文，不应获得明文 API Key。

## 目标

- 不要求 Bot 拥有公网域名、开放端口或建立反向隧道。
- 群主、群管理员可以为自己的群配置主聊天模型 API。
- API Key 不通过群消息、QQ 私聊或中转数据库以明文传输和保存。
- 一个 Bot 实例只注册一次，之后可以为多个群创建配置单。
- 未配置群继续使用现有全局配置，保持向后兼容。
- 中转配置失败时保留该群原有配置，不影响其他群。
- 中转协议与具体网页框架、数据库和部署平台解耦。

## 非目标

第一期不处理以下内容：

- 为中转服务提供用户账号、自助签发注册码或运营后台。
- 由仓库自动执行公网域名、证书和生产 Secret 的实际开通。
- 将 Bot 的现有用量 WebUI 暴露到公网。
- 允许普通群成员修改配置。
- 通过聊天消息直接提交明文 API Key。
- 分群配置 Qdrant、Embedding、Rerank、人格或并发限制。
- 将图片标注等共享后台任务的费用归属到某个群。
- 在模型接口失败时默认回退到 Bot 所有者的全局 API。

## 核心决策

### 中转服务是短期密文信箱

中转服务负责：

- 注册 Bot 实例的公钥；
- 创建短期配置单；
- 向浏览器提供对应 Bot 的公钥；
- 暂存浏览器上传的加密信封；
- 生成一次性配置码；
- 只向目标 Bot 实例交付密文；
- 在确认处理或超时后删除密文。

中转服务不负责：

- 解密或验证模型 API Key；
- 调用模型供应商；
- 保存 Bot 私钥；
- 决定 QQ 用户是否拥有群管理权限；
- 保存群号、QQ 号等业务身份信息。

### 权限判断留在 Bot 内部

Bot 能通过适配器取得当前群成员角色，因此创建配置单前由 Bot 验证操作者是否为群主、群管理员或 NoneBot 超级用户。中转服务只处理不透明的配置单，不需要知道群号或 QQ 号。

Bot 本地保存配置单与业务身份的对应关系：

```text
ticket_id -> group_id、operator_id、expires_at
```

兑换时必须再次核对发送配置码的用户，不能只依赖配置码本身。

### 浏览器端加密

网页使用 Web Crypto 在浏览器本地加密配置：

1. 随机生成 256 位 AES 密钥和 96 位 nonce；
2. 使用 AES-256-GCM 加密 UTF-8 JSON 配置；
3. 使用目标 Bot 的 RSA-OAEP 公钥加密 AES 密钥；
4. 只把加密信封上传到中转服务。

推荐算法组合：

```text
RSA-OAEP-3072 + SHA-256
AES-256-GCM
```

RSA 只加密随机 AES 密钥，模型配置由 AES-GCM 加密，避免 RSA 明文长度限制。

### 群配置不静默消耗全局主模型额度

一旦某群启用自有主聊天 API，以下情况默认停止该群本次主模型调用，而不是自动回退到全局主模型：

- API Key 无效或欠费；
- 模型不存在；
- 供应商限流；
- 接口持续超时；
- 群级额度耗尽。

是否允许回退必须是显式群配置，页面应醒目标注“回退将消耗 Bot 所有者的额度”。

## 总体结构

```text
┌──────────────┐       出站 HTTPS       ┌──────────────────┐
│ 内网 NoneBot │ ─────────────────────> │ 公网中转服务     │
│              │ <───────────────────── │ 短期保存加密信封 │
└──────┬───────┘                         └────────┬─────────┘
       │                                          │ HTTPS
       │ QQ 私聊                                  │
       │ 配置链接 / 配置码                        │
       ▼                                          ▼
┌──────────────────────────────────────────────────────────┐
│ 群管理员浏览器                                            │
│ 获取 Bot 公钥 -> 本地加密配置 -> 上传密文 -> 显示配置码   │
└──────────────────────────────────────────────────────────┘
```

插件侧应把复杂性收进一个较深的中转模块。其他调用者只需要了解以下接口：

```python
class GroupModelRelay:
    async def ensure_registered(self) -> InstanceIdentity: ...
    async def create_ticket(self, context: GroupConfigContext) -> ConfigTicket: ...
    async def redeem(self, code: str, context: GroupConfigContext) -> RedeemResult: ...
```

注册凭据保存、请求认证、重试、配置码标准化、解密、确认删除和错误归类都属于该模块的实现，不应散落在命令处理器中。命令处理器只负责权限、交互和结果提示。

## 完整流程

### 1. Bot 实例注册

首次启用中转功能时：

1. Bot 在本地生成 RSA-3072 密钥对；
2. Bot 使用部署者注册码调用注册接口；
3. 中转服务保存公钥并返回实例 ID 和实例令牌；
4. Bot 加密保存私钥、实例令牌和实例 ID；
5. 后续启动读取本地身份，不重复注册。

```http
POST /v1/instances/register
Authorization: Bearer <registration_token>
Content-Type: application/json

{
  "protocol_version": 1,
  "public_key_jwk": {},
  "plugin_version": "2.3.5"
}
```

服务器使用私有注册模式时必须携带上述 Authorization。服务器运营者也可以显式开放公开注册；此时插件在未配置注册码时省略 Authorization，由服务端的每 IP 限流和全局注册额度控制滥用。不得把通用注册码硬编码到插件中，因为客户端内置值不具备保密性。

成功响应：

```json
{
  "instance_id": "ins_01J6EXAMPLE",
  "instance_token": "agt_EXAMPLE",
  "key_id": "key_01J6EXAMPLE",
  "created_at": "2026-08-24T12:00:00Z"
}
```

`instance_token` 只返回一次。中转数据库只保存它的不可逆摘要。

部署者注册码用于限制垃圾注册。第一版可以由中转服务运营者人工签发；后续再考虑账户、自助申请或开放注册加限流。

### 2. 创建配置单

群管理员在群内执行：

```text
/配置群API
```

Bot 完成以下操作：

1. 确认事件来自群聊；
2. 确认操作者是群主、群管理员或超级用户；
3. 创建本地 `GroupConfigContext`；
4. 使用实例令牌请求中转服务创建配置单；
5. 将配置链接私聊发送给操作者。

```http
POST /v1/config-tickets
Authorization: Bearer <instance_token>
Idempotency-Key: <uuid>
Content-Type: application/json

{
  "protocol_version": 1,
  "expires_in": 900
}
```

成功响应：

```json
{
  "ticket_id": "tkt_01J6EXAMPLE",
  "config_url": "https://relay.example.com/config/tkt_01J6EXAMPLE#token=submit_EXAMPLE",
  "expires_at": "2026-08-24T12:15:00Z"
}
```

原始群号和 QQ 号不上传；它们只与 `ticket_id` 一起保存在 Bot 本地。`submit_token` 只放在 URL fragment 中，首次请求网页时不会进入服务器 URL 或访问日志；页面脚本在上传密文时再通过请求头提交。中转数据库只保存其摘要。

### 3. 用户填写并加密配置

配置页面展示第一期允许填写的字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `api_format` | 是 | 插件端支持 `openai` / `anthropic` / `vertex`；Vertex 使用 Express Mode API Key |
| `base_url` | 是 | OpenAI 兼容接口根地址 |
| `api_key` | 是 | 只存在于浏览器内存和加密后载荷中 |
| `chat_model` | 是 | 当前群主聊天模型名称 |
| `chat_multimodal` | 否 | 主模型是否接收图片，默认开启 |
| `reply_probability` | 否 | 本群主动发言概率，范围 `0`～`0.1`；`null` 或省略时跟随 Bot 全局配置 |
| `allow_global_fallback` | 否 | 当前版本必须为 `false`；群 API 失败时不消耗 Bot 所有者的全局额度 |

页面先读取目标实例的公开加密信息：

```http
GET /v1/config-tickets/tkt_01J6EXAMPLE/public
```

```json
{
  "protocol_version": 1,
  "ticket_id": "tkt_01J6EXAMPLE",
  "instance_id": "ins_01J6EXAMPLE",
  "key_id": "key_01J6EXAMPLE",
  "public_key_jwk": {},
  "expires_at": "2026-08-24T12:15:00Z"
}
```

此接口只返回公钥和非敏感元数据，不返回实例令牌或业务身份。

加密前载荷：

```json
{
  "schema_version": 1,
  "ticket_id": "tkt_01J6EXAMPLE",
  "api_format": "openai",
  "base_url": "https://api.example.com/v1",
  "api_key": "sk-example",
  "chat_model": "example-model",
  "chat_multimodal": true,
  "reply_probability": 0.01,
  "allow_global_fallback": false,
  "created_at": "2026-08-24T12:05:00Z"
}
```

AES-GCM 的附加认证数据必须固定为：

```text
ai-groupmate-config:v1:<ticket_id>:<instance_id>:<key_id>
```

上传的是加密信封：

```json
{
  "protocol_version": 1,
  "ticket_id": "tkt_01J6EXAMPLE",
  "key_id": "key_01J6EXAMPLE",
  "wrapped_key": "<base64url RSA-OAEP ciphertext>",
  "nonce": "<base64url 12 bytes>",
  "ciphertext": "<base64url AES-GCM ciphertext>"
}
```

上传接口：

```http
POST /v1/config-tickets/tkt_01J6EXAMPLE/payload
Authorization: Ticket submit_EXAMPLE
Content-Type: application/json

{
  "protocol_version": 1,
  "ticket_id": "tkt_01J6EXAMPLE",
  "key_id": "key_01J6EXAMPLE",
  "wrapped_key": "...",
  "nonce": "...",
  "ciphertext": "..."
}
```

每张配置单只接受一次有效上传。重复上传返回冲突错误，避免后来提交的内容覆盖用户已经取得配置码的内容。

上传成功后服务器生成至少 80 位随机性的 Base32 配置码：

```text
AGC-7K3M-P9DX-2FWA-8QRT
```

配置码是查找凭据，不是加密密钥。截获配置码的人没有目标 Bot 的实例令牌和私钥，不能兑换或解密载荷。

### 4. 用户提交配置码

用户应在 Bot 私聊中发送：

```text
/提交群API AGC-7K3M-P9DX-2FWA-8QRT
```

Bot 必须：

1. 标准化配置码的大小写、空格和连字符；
2. 根据本地待处理记录确认操作者身份；
3. 使用实例令牌兑换加密信封；
4. 使用本地 RSA 私钥解开 AES 密钥；
5. 验证 AES-GCM 和附加认证数据；
6. 验证载荷中的 `ticket_id`、版本和时间；
7. 使用候选配置测试模型连接；
8. 测试成功后加密保存群配置并切换运行配置；
9. 无论测试成功与否，处理完成后通知中转服务删除密文；
10. 向用户返回不包含任何密钥内容的结果。

### 5. 兑换与确认

兑换接口：

```http
POST /v1/config-payloads/redeem
Authorization: Bearer <instance_token>
Idempotency-Key: <uuid>
Content-Type: application/json

{
  "code": "AGC-7K3M-P9DX-2FWA-8QRT"
}
```

响应返回加密信封及交付凭据：

```json
{
  "delivery_id": "del_01J6EXAMPLE",
  "ticket_id": "tkt_01J6EXAMPLE",
  "envelope": {
    "protocol_version": 1,
    "key_id": "key_01J6EXAMPLE",
    "wrapped_key": "...",
    "nonce": "...",
    "ciphertext": "..."
  }
}
```

兑换应当幂等。网络中断后，同一目标 Bot 在配置单过期前可以重新取得相同密文，避免响应丢失导致配置永久丢失。

处理结束后确认：

```http
POST /v1/config-payloads/{delivery_id}/ack
Authorization: Bearer <instance_token>
Content-Type: application/json

{
  "outcome": "applied"
}
```

`outcome` 可为 `applied`、`rejected` 或 `invalid`，但不得包含供应商错误原文。中转服务收到确认后删除密文；未确认的密文在配置单过期后自动删除。

## 配置继承与第一期范围

配置解析顺序：

```text
群级主聊天配置
        ↓ 未启用
全局主聊天专用配置
        ↓ 未填写
全局通用 LLM 配置
        ↓
兼容旧配置
```

第一期群级配置只接管主聊天模型。下列调用仍使用全局配置，并应在网页中明确提示：

- Flash/Gatekeeper；
- 群摘要；
- 图片标注；
- 图片回读；
- Embedding 和 Rerank。

这样可以先转移占比最大的主模型费用，同时避免图片和向量等共享后台任务产生模糊的群归属。后续若扩展到辅助模型，应继续使用“群角色配置 → 群通用配置 → 全局角色配置 → 全局通用配置”的解析顺序，并在用量统计中标记实际凭据来源。

## 本地数据模型

建议新增三类本地记录。

### RelayInstanceIdentity

每个插件数据库最多一条：

```text
instance_id
instance_token_ciphertext
public_key_jwk
private_key_ciphertext
key_id
relay_url
registered_at
```

### PendingGroupConfig

短期记录，不含 API Key：

```text
ticket_id
group_id
operator_id
expires_at
created_at
```

### GroupModelConfig

每群最多一条：

```text
group_id
enabled
api_format
base_url
api_key_ciphertext
chat_model
chat_multimodal
reply_probability nullable
allow_global_fallback
updated_by
updated_at
last_tested_at
last_test_status
version
```

群 API Key、实例令牌和 RSA 私钥必须使用本地长期主密钥加密。建议新增环境变量：

```dotenv
AI_GROUPMATE__GROUP_API_RELAY_URL=https://relay.example.com
# 私有注册模式才需要
AI_GROUPMATE__GROUP_API_RELAY_REGISTRATION_TOKEN=replace-me
# 可选；留空时在插件数据目录自动生成
AI_GROUPMATE__GROUP_API_LOCAL_ENCRYPTION_KEY=<32-byte-base64url-key>
# 可选；生产环境建议显式配置
AI_GROUPMATE__GROUP_API_ALLOWED_PROVIDER_HOSTS=["api.openai.com","openrouter.ai"]
```

可用下面的命令生成本地主密钥：

```bash
python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))"
```

`GROUP_API_LOCAL_ENCRYPTION_KEY` 留空时，插件会原子创建数据目录下的 `group_api_local_encryption.key`，并在支持 POSIX 权限的平台设置为 `0600`。插件数据目录必须使用持久卷并与数据库一起备份；不得把自动生成文件留在容器临时层。也可以通过 Secret、只读文件或受保护的环境变量显式提供该值。密钥不可直接轮换；需要轮换时必须先重新配置所有群并重新注册实例。

`GROUP_API_RELAY_REGISTRATION_TOKEN` 只在中转服务使用私有注册模式且本实例首次注册时需要。中转服务显式开放公开注册时可以留空；这与把通用注册码写死在插件中不同，服务运营者可以独立关闭公开注册并设置全局注册额度。私有模式注册成功且本地数据库已经持久化身份后，可以从环境中移除。

群管理员可控制模型 `base_url`，因此插件会拒绝 HTTP 地址。`GROUP_API_ALLOWED_PROVIDER_HOSTS` 支持精确主机和 `*.example.com` 通配形式：配置白名单后只允许其中的主机，且白名单也可用于由部署者明确授权私有模型端点；未配置白名单时，插件会在连接测试前解析域名，并拒绝任何非公网 IP、localhost、`.local`、`.internal` 和内网短主机名，以降低内网 Bot 的 SSRF 风险。对于安全要求较高的部署，建议始终配置精确白名单，并在网络出口层继续限制 Bot 可访问的地址。

## 中转服务数据模型

中转端只需保存：

### Instance

```text
instance_id
instance_token_hash
public_key_jwk
key_id
status
created_at
last_seen_at
```

### ConfigTicket

```text
ticket_id
instance_id
key_id
code_hash
encrypted_envelope
status
expires_at
created_at
submitted_at
acknowledged_at
```

状态转换：

```text
CREATED -> SUBMITTED -> DELIVERED -> ACKNOWLEDGED
    └─────────────── 任意未完成状态 ───────────────> EXPIRED
```

配置码使用高熵随机值，中转数据库只保存带服务器 pepper 的摘要。`encrypted_envelope` 在确认或过期后立即物理删除。

## 插件侧模块划分

插件实现把 seam 放在以下位置：

### relay 模块

负责注册、创建配置单、兑换、确认、认证、重试、协议版本和加解密。命令层不直接调用 HTTP 客户端。

实现位置：

```text
src/nonebot_plugin_ai_groupmate/group_api_relay.py
```

### group_config 模块

负责群配置校验、加密持久化、继承解析和模型客户端缓存失效。

实现位置：

```text
src/nonebot_plugin_ai_groupmate/group_model_config.py
```

它向模型调用方提供尽量小的接口：

```python
def resolve_chat_config(group_id: str | None) -> ScopedConfig: ...
```

调用方不应自行查询数据库或拼接配置优先级。

### command adapter

命令 adapter 只负责：

- 取得群和用户身份；
- 验证群管理权限；
- 调用中转模块；
- 私聊发送链接；
- 接收配置码并展示安全错误。

已实现命令：

```text
/配置群API
/提交群API <配置码>
/查看群API
/删除群API 确认
```

### Bot 管理员配置中心

配置中心的“群聊 API”页面提供同一份群配置的管理员入口：

- 按群 ID 新增或编辑主聊天模型配置；
- 随独立模型配置设置 `0`～`0.1` 的主动发言概率，留空跟随全局值；
- API Key 不回显，编辑时留空保留原密钥；
- 保存前解析目标地址并实际测试一次模型连接；
- 列表只展示接口格式、服务主机、模型、图片能力和版本；
- 删除配置后立即清除模型缓存并恢复全局主模型。

页面及其 JSON 接口复用配置中心的 HttpOnly Cookie 登录。配置中心启用时，插件会在数据目录自动生成本地加密密钥，所以管理员入口不依赖公网中转服务。

## 错误处理

插件向用户展示稳定的中文错误，不透传内部异常：

| 场景 | 用户提示 | 是否保留旧配置 |
| --- | --- | --- |
| 无群管理权限 | 只有群主或管理员可以配置 | 是 |
| 配置单过期 | 配置码已过期，请重新生成 | 是 |
| 配置码不属于当前 Bot | 配置码无效或不属于当前 Bot | 是 |
| 操作者不匹配 | 请由发起配置的管理员本人提交 | 是 |
| 密文校验失败 | 配置内容无法验证，请重新配置 | 是 |
| Base URL 指向内网或不在白名单 | 模型连接测试失败，请检查地址、Key 和模型名 | 是 |
| 模型连接失败 | 模型连接测试失败，请检查地址、Key 和模型名 | 是 |
| 本地保存失败 | 配置未生效，请稍后重试 | 是 |
| 中转暂不可用 | 配置服务暂不可用，请稍后重试 | 是 |

日志可以记录错误类型、HTTP 状态、实例 ID 后六位和 ticket ID 后六位，但不能记录：

- API Key；
- Bearer Token；
- 完整配置码；
- 私钥；
- 解密后的配置载荷；
- 可能包含上述内容的供应商原始响应。

## 安全模型

### 能够防御

- 中转数据库只读泄露：攻击者只能取得短期密文和公钥；
- 配置码被其他 Bot 获取：配置单绑定目标实例，其他实例无法兑换；
- 网络窃听：所有连接使用 HTTPS，载荷本身仍为端到端密文；
- 密文篡改：AES-GCM 和附加认证数据验证失败；
- 重放提交：配置单、操作者和一次性确认共同限制；
- Bot 进程重启：本地待处理记录和幂等兑换允许恢复。

### 不能完全防御

- Bot 主机已经被攻陷；
- 用户浏览器或扩展已经被攻陷；
- 本地主密钥和数据库同时泄露；
- 中转服务恶意替换配置页面 JavaScript 或目标公钥。

因此配置页面应使用固定版本的静态资源、严格 CSP、禁止第三方脚本和统计代码，并公开构建产物供审计。端到端加密主要降低数据库泄露和被动运维访问的风险，不能把恶意网页运营方排除在信任模型之外。

## 接口通用约束

- 所有接口必须使用 HTTPS。
- JSON 请求体限制在合理大小，例如 32 KiB。
- 时间使用 UTC RFC 3339。
- ID 使用不可预测的随机值，不使用自增整数。
- Base64 使用无填充 Base64URL。
- 实例认证第一期使用 Bearer Token；后续可以增加 HMAC 请求签名。
- 写接口支持 `Idempotency-Key`。
- 服务端为注册、创建配置单、上传和兑换分别限流。
- 错误响应使用稳定错误码，不返回内部堆栈。
- 协议版本不兼容时拒绝处理，不进行猜测性降级。

统一错误响应：

```json
{
  "error": {
    "code": "ticket_expired",
    "message": "configuration ticket expired",
    "request_id": "req_01J6EXAMPLE"
  }
}
```

## 可观测性

后续用量记录建议增加不含密钥的信息：

```text
config_scope = global | group
credential_id = 群配置记录的随机 ID
config_version
provider_host
```

WebUI 可以显示某群当前使用全局还是群级主模型、模型名称、最近测试状态和更新时间。绝不显示完整 Base URL 查询参数、API Key 或实例令牌。

## 测试要求

插件实现至少覆盖：

- 首次注册与重复启动不重复注册；
- 注册令牌和实例令牌不进入日志；
- 仅群主、群管理员、超级用户能创建配置单；
- 普通成员不能借用其他人的 ticket 或配置码；
- 过期、错误实例和重复兑换得到稳定错误；
- RSA-OAEP/AES-GCM 正常解密及任意字段篡改失败；
- AAD 中 ticket、instance 或 key ID 不匹配时失败；
- 连接测试失败时旧群配置保持不变；
- 保存成功后只清理对应群的模型缓存；
- 群级配置缺失时继续使用全局配置；
- 群 API 失败时默认不回退全局主模型；
- 中转响应丢失后幂等兑换能够恢复；
- 确认或过期后中转端密文被删除；
- 所有用户可见错误和日志均不包含密钥。

## 分阶段实施

### 阶段一：插件基础模块（已完成）

- 本地身份、待处理配置单和群模型配置数据模型；
- 本地加密存储；
- 中转客户端接口及可替换的内存 adapter；
- 群级主聊天配置解析；
- 单元测试，不连接真实公网服务。

### 阶段二：中转服务（已完成）

- 实例注册；
- 配置单和兑换接口；
- 浏览器端加密页面；
- TTL 清理、限流、审计和部署文档。

### 阶段三：Bot 命令接入（主体已完成）

- 群权限校验；
- 私聊发送配置链接；
- 配置码兑换、连接测试、应用和删除；
- Bot 配置中心按群新增、修改和删除独立主模型；
- 用量 WebUI 展示配置来源（待实现）。

### 阶段四：扩展模型角色

- Flash、摘要和视觉模型的可选群级配置；
- 分角色费用归属和配额；
- 凭据轮换、实例注销和公钥轮换。

## 验收标准

整个功能完成时应满足：

1. 内网 Bot 无公网入口也能完成群级 API 配置；
2. 中转服务的请求、日志和数据库中均不出现明文 API Key；
3. 非群管理人员无法创建或应用群配置；
4. 配置失败不会破坏旧配置，也不会影响其他群；
5. 群级 API 生效后，该群主聊天请求使用群级凭据；
6. 群级 API 故障默认不会消耗全局主模型额度；
7. 配置密文确认处理或过期后从中转端删除；
8. 插件可以通过内存 adapter 在无公网中转服务时完成自动化测试。
