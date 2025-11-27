<div align="center">
  <a href="https://v2.nonebot.dev/store"><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/nbp_logo.png" width="180" height="180" alt="NoneBotPluginLogo"></a>
  <br>
  <p><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/NoneBotPlugin.svg" width="240" alt="NoneBotPluginText"></p>
</div>

<div align="center">

# nonebot-plugin-ai-groupmate

_✨ ai groupmate ✨_


<a href="./LICENSE">
    <img src="https://img.shields.io/github/license/yaowan233/nonebot-plugin-ai-groupmate.svg" alt="license">
</a>
<a href="https://pypi.python.org/pypi/nonebot-plugin-ai-groupmate">
    <img src="https://img.shields.io/pypi/v/nonebot-plugin-ai-groupmate.svg" alt="pypi">
</a>
<img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="python">

</div>


## 📖 介绍
使用 RAG 技术，自动对聊天历史储存，储存长记忆。学习群内群友发言习惯，使得 bot 更像真人。

接入vlm，并且自动学习表情包，自动在群内学习并偷取表情包。


## 💿 安装

<details>
<summary>使用 nb-cli 安装</summary>
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install nonebot-plugin-ai-groupmate

</details>

<details>
<summary>使用包管理器安装</summary>
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details>
<summary>pip</summary>

    pip install nonebot-plugin-ai-groupmate
</details>
<details>
<summary>pdm</summary>

    pdm add nonebot-plugin-ai-groupmate
</details>
<details>
<summary>poetry</summary>

    poetry add nonebot-plugin-ai-groupmate
</details>


打开 nonebot2 项目的 `bot.py` 文件, 在其中写入

    nonebot.load_plugin('nonebot_plugin_ai_groupmate')

</details>


## ⚙️ 配置

配置说明
| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| bot_name | 是 | 无 | bot 名 |
| reply_probability | 否 | 0.01 | 群内发言概率 |
| personality_setting | 否 | 无 | 自定义人设 |
| milvus_uri | 否 | 无 | milvus 地址 |
| milvus_user | 否 | 无| milvus 用户名 |
| milvus_password | 否 | 无 | milvus 密码 |
| tavily_api_key | 否 | 无 | tavily api 密钥 |
| openai_base_url | 否 | 无| openai 请求地址 |
| openai_token | 否 | 无 | openai token |
| openai_model | 否 | 无 | openai 模型名 |
| vlm_ollama_base_url | 否 | 无| vlm 地址 |
| vlm_model | 否 | 无 | vlm 模型名 |
| vlm_provider | 否 | ollama| ollama 或 openai |
| vlm_openai_base_url | 否 | 无 | vlm openai 请求地址 |
| vlm_openai_api_key | 否 | 无 | vlm openai api key |

## 🎉 使用

待补充
### 指令