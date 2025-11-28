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
使用 RAG 技术，自动对聊天历史储存，储存长记忆。学习群内群友发言习惯，使得 bot 更像真人。

接入vlm，并且自动学习表情包，自动在群内学习并偷取表情包。

使用 langchain，利用 agent 进行决策
## 💿 安装

<details open>
<summary>使用 nb-cli 安装</summary>
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install {plugin-name} --upgrade
使用 **pypi** 源安装

    nb plugin install {plugin-name} --upgrade -i "https://pypi.org/simple"
使用**清华源**安装

    nb plugin install {plugin-name} --upgrade -i "https://pypi.tuna.tsinghua.edu.cn/simple"


</details>

<details>
<summary>使用包管理器安装</summary>
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details open>
<summary>uv</summary>

    uv add {plugin-name}
安装仓库 master 分支

    uv add git+https://github.com/{owner}/{plugin-name}@master
</details>

<details>
<summary>pdm</summary>

    pdm add {plugin-name}
安装仓库 master 分支

    pdm add git+https://github.com/{owner}/{plugin-name}@master
</details>
<details>
<summary>poetry</summary>

    poetry add {plugin-name}
安装仓库 master 分支

    poetry add git+https://github.com/{owner}/{plugin-name}@master
</details>

打开 nonebot2 项目根目录下的 `pyproject.toml` 文件, 在 `[tool.nonebot]` 部分追加写入

    plugins = ["nonebot_plugin_template"]

</details>

<details>
<summary>使用 nbr 安装(使用 uv 管理依赖可用)</summary>

[nbr](https://github.com/fllesser/nbr) 是一个基于 uv 的 nb-cli，可以方便地管理 nonebot2

    nbr plugin install {plugin-name}
使用 **pypi** 源安装

    nbr plugin install {plugin-name} -i "https://pypi.org/simple"
使用**清华源**安装

    nbr plugin install {plugin-name} -i "https://pypi.tuna.tsinghua.edu.cn/simple"

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