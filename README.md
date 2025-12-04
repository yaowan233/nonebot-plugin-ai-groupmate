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
本插件主体使用使用 langchain 的 agent 进行决策，由 langchain 调用 tools 进行一系列任务。

tools 中包含 RAG ，可以自动对聊天历史储存，储存长记忆。学习群内群友发言习惯，使得 bot 更像真人。

对于群内的表情包，使用了 vlm + clip 模型，自动从群内学习并偷取表情包，然后从向量库内选取合适表情包进行回答。

对于模型选择方面：推荐使用 glm-4.6 模型，以及本地部署 qwen3-vl 作为 vlm 模型，如果 embedding、 rerank、vlm 和 clip 模型都使用了显卡加速的话，大约需要 16G 显存的显卡。若只有 8G 显存，将 vlm 模型调整为 api 调用的话，就可以流畅使用了。

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

    plugins = [nonebot-plugin-ai-groupmate"]

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

ai会自动偷群内使用的表情包，增加至向量库当中，在回答时通过向量库内容搜索表情包，由于使用了vlm模型，搜索的准确率十分高。
![Screenshot_20251201_134203](https://github.com/user-attachments/assets/cbf95194-ac33-45e0-a83d-cb6639c204fb)
内置了好感度系统，增加了趣味性。
![Screenshot_20251201_134157](https://github.com/user-attachments/assets/68b8d563-7ad5-4d83-be4b-0a05f16df09a)
利用强大的 RAG，进行总结或进行任何检索聊天相关功能。
![Screenshot_20251201_133320](https://github.com/user-attachments/assets/b7e96bd0-8245-4da5-b28b-33e8aad5fc63)
发送群内偷学到的表情包
![Screenshot_20251201_132723](https://github.com/user-attachments/assets/6fbd036f-e7ec-4ced-9cd7-557976306553)

### 指令表
由于 AI 功能需要记录聊天记录，基于已记录的聊天记录，可以很轻松的做到词频统计，所以顺带加上了。

|     指令      |    说明    |
|:-----------:|:--------:|
|  /词频 <统计天数> | 生成个人词频词云 |
| /群词频 <统计天数> | 生成群词频词云  |