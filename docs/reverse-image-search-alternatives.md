# 不使用 Google Cloud 凭据的以图搜图方案

核实日期：2026-09-09。这里区分“不申请 Google Key”与“完全不需要任何 Key”；未实测命中率，不比较价格。

| 方案 | 凭据 | 适用场景与限制 |
| --- | --- | --- |
| trace.moe | 可以不带 Key；按 IP 计算额度和并发限制 | 动画截图定位番名、集数、时间点；支持图片 URL 或直接上传。它是动画场景搜索，不是通用全网图片搜索。额度以 `/me` 返回值为准。见[官方介绍](https://raw.githubusercontent.com/soruly/trace.moe-api/master/docs/README.md)、[API 文档](https://raw.githubusercontent.com/soruly/trace.moe-api/master/docs/docs.md)。 |
| SauceNAO | 网页可上传；API 账户与 Key、额度需登录官方账户页进一步确认，本次该页返回 403 | 索引包含 Pixiv、Danbooru、漫画等，因而适合插画来源查询；覆盖取决于其数据库。重度裁剪、拼图会影响效果。见[官方索引列表](https://saucenao.com/)、[使用说明](https://saucenao.com/options.php)、[API 账户页](https://saucenao.com/user.php?page=search-api)。 |
| SerpApi Google Lens API | 需要 SerpApi Key，不需要 Google Cloud Key | 提供相似图片、完全匹配、商品等搜索类型，可传 URL 或先上传图片。仍然使用 Google Lens 结果，只是改用第三方接口；不能视为不依赖 Google 的搜索引擎。见[官方文档](https://serpapi.com/google-lens-api)。 |
| 阿里云百炼 `image_search` | 需要百炼 API Key | Responses API 提供图搜图工具，以 `input_image` 传公网可访问图片 URL；受支持模型与接口限制。见[官方图搜图文档](https://help.aliyun.com/zh/model-studio/image-search)。 |

建议：若主要处理群聊中的动画截图，可先接 trace.moe；插画来源可评估 SauceNAO；通用图片检索可评估百炼或第三方 Lens。以上是接入方向，不能仅通过替换 Google API 地址完成，需要适配各自请求和结果格式。
