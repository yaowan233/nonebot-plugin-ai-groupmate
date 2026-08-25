import json
from html import escape
from typing import Any

from .config import ScopedConfig
from .runtime_config import (
    SECRET_FIELDS,
    CONFIGURABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
)

SETTING_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "基础行为",
        "Bot 名称、回复策略和人格设定，保存后立即生效。",
        (
            "bot_name",
            "reply_probability",
            "repeat_probability",
            "proactive_reaction_probability",
            "proactive_meme_probability",
            "proactive_private_message",
            "continuous_conversation_minutes",
            "personality_setting",
        ),
    ),
    (
        "Agent 运行",
        "控制单轮运行时间、调用次数和工具结果大小，保存后立即生效。",
        (
            "agent_timeout_seconds",
            "agent_llm_timeout_seconds",
            "agent_tool_timeout_seconds",
            "agent_max_concurrency",
            "background_image_max_concurrency",
            "background_image_max_pending",
            "maintenance_max_concurrency",
            "group_memory_update_timeout_seconds",
            "agent_max_llm_calls",
            "agent_max_total_tokens",
            "agent_tool_result_max_chars",
            "chat_explicit_prompt_cache",
        ),
    ),
    (
        "通用模型",
        "各模型未填写专用地址或密钥时，会回退使用这里的配置。",
        ("llm_api_key", "llm_base_url"),
    ),
    (
        "主聊天模型",
        "负责 Agent 推理、工具调用和最终回复。",
        (
            "chat_model",
            "chat_api_key",
            "chat_base_url",
            "chat_temperature",
            "chat_api_format",
            "chat_multimodal",
        ),
    ),
    (
        "Google Vertex AI",
        "chat/tagging/vision 选择 Vertex 时共用；API Key 使用无需项目 ID 的 Express Mode，服务账号或 ADC 使用标准模式。",
        (
            "vertex_project",
            "vertex_location",
            "vertex_api_key",
            "vertex_credentials_path",
        ),
    ),
    (
        "快速决策模型",
        "用于 Gatekeeper 判断消息是否值得回复。",
        (
            "flash_model",
            "flash_api_key",
            "flash_base_url",
            "flash_temperature",
            "flash_max_tokens",
        ),
    ),
    (
        "群摘要模型",
        "用于更新群体认知档案。",
        (
            "summary_model",
            "summary_api_key",
            "summary_base_url",
            "summary_temperature",
            "summary_max_tokens",
        ),
    ),
    (
        "图片标注模型",
        "负责识别和描述收到的图片、表情包。",
        (
            "tagging_model",
            "tagging_api_key",
            "tagging_base_url",
            "tagging_temperature",
            "tagging_api_format",
        ),
    ),
    (
        "图片回读模型",
        "主聊天模型不支持图片时，用于把图片转换为文本摘要。",
        (
            "vision_model",
            "vision_api_key",
            "vision_base_url",
            "vision_temperature",
            "vision_api_format",
            "vision_input_cost_per_million",
            "vision_output_cost_per_million",
        ),
    ),
    (
        "搜索与向量库",
        "联网搜索立即生效；Qdrant 服务端须为 1.16 或更高版本；Qdrant、Embedding 与 Rerank 连接配置重启后生效；注意：修改 Embedding 模型或者维度后需要人工重建向量。",
        (
            "tavily_api_key",
            "qdrant_uri",
            "qdrant_api_key",
            "embedding_api_key",
            "embedding_base_url",
            "embedding_model",
            "embedding_dimension",
            "meme_embedding_mode",
            "rerank_api_url",
            "rerank_api_key",
            "media_vectorize_min_references",
            "media_vectorize_batch_size",
            "media_vectorize_concurrency",
        ),
    ),
    (
        "费用统计",
        "每百万 Token 的价格，仅影响 WebUI 费用估算。",
        (
            "chat_input_cost_per_million",
            "chat_output_cost_per_million",
            "chat_cached_input_cost_per_million",
            "chat_explicit_cached_input_cost_per_million",
            "chat_cache_creation_input_cost_per_million",
            "chat_long_context_threshold_tokens",
            "chat_long_input_cost_per_million",
            "chat_long_output_cost_per_million",
            "chat_long_cached_input_cost_per_million",
            "chat_long_explicit_cached_input_cost_per_million",
            "chat_long_cache_creation_input_cost_per_million",
        ),
    ),
    (
        "兼容旧配置",
        "仅为旧版配置保留；新部署优先使用通用模型配置。",
        ("base_model", "qwen_token"),
    ),
)

_DIRECT_LABELS = {
    "bot_name": "Bot 名称",
    "reply_probability": "随机回复概率",
    "repeat_probability": "加入复读概率",
    "proactive_reaction_probability": "主动消息表情回应概率",
    "proactive_meme_probability": "主动表情包采样概率",
    "proactive_private_message": "允许主动私聊",
    "continuous_conversation_minutes": "连续对话窗口（分钟）",
    "personality_setting": "人格设定",
    "agent_timeout_seconds": "Agent 总超时（秒）",
    "agent_llm_timeout_seconds": "单次模型超时（秒）",
    "agent_tool_timeout_seconds": "单次工具超时（秒）",
    "agent_max_concurrency": "Agent 全局并发上限",
    "background_image_max_concurrency": "后台图片并发上限",
    "background_image_max_pending": "后台图片最大积压数",
    "maintenance_max_concurrency": "维护任务并发上限",
    "media_vectorize_min_references": "表情包入库最低引用次数",
    "media_vectorize_batch_size": "表情包向量化每轮数量",
    "media_vectorize_concurrency": "表情包向量化并发数",
    "group_memory_update_timeout_seconds": "群档案更新超时（秒）",
    "agent_max_llm_calls": "最多模型调用次数",
    "agent_max_total_tokens": "最大 Token 预算",
    "agent_tool_result_max_chars": "工具结果最大字符数",
    "chat_explicit_prompt_cache": "启用显式 Prompt 缓存",
    "chat_multimodal": "主模型支持图片",
    "vertex_project": "Google Cloud 项目 ID",
    "vertex_location": "Vertex 区域",
    "vertex_api_key": "Vertex API Key",
    "vertex_credentials_path": "服务账号 JSON 路径",
    "tavily_api_key": "Tavily API Key",
    "qdrant_uri": "Qdrant 地址",
    "qdrant_api_key": "Qdrant API Key",
    "embedding_api_key": "Embedding API Key",
    "embedding_base_url": "Embedding Base URL",
    "embedding_model": "Embedding 模型名称",
    "embedding_dimension": "Embedding 向量维度",
    "meme_embedding_mode": "表情包向量化模式 (multimodal/text)",
    "rerank_api_url": "Rerank API URL",
    "rerank_api_key": "Rerank API Key",
    "base_model": "旧版默认模型",
    "qwen_token": "旧版 Qwen Token",
}
_ROLE_LABELS = {
    "llm": "通用模型",
    "chat": "主聊天模型",
    "flash": "快速模型",
    "summary": "群摘要模型",
    "tagging": "图片标注模型",
    "vision": "图片回读模型",
}
_SUFFIX_LABELS = {
    "model": "模型名称",
    "api_key": "API Key",
    "base_url": "Base URL",
    "temperature": "温度",
    "api_format": "接口格式",
    "max_tokens": "最大输出 Tokens",
    "input_cost_per_million": "输入价格 / 百万 Tokens",
    "output_cost_per_million": "输出价格 / 百万 Tokens",
}
_TEST_ROLES = {
    "主聊天模型": "chat",
    "快速决策模型": "flash",
    "群摘要模型": "summary",
    "图片标注模型": "tagging",
    "图片回读模型": "vision",
}


def _field_label(field_name: str) -> str:
    if field_name in _DIRECT_LABELS:
        return _DIRECT_LABELS[field_name]
    for role, role_label in _ROLE_LABELS.items():
        prefix = f"{role}_"
        if not field_name.startswith(prefix):
            continue
        suffix = field_name[len(prefix):]
        if suffix in _SUFFIX_LABELS:
            return f"{role_label} · {_SUFFIX_LABELS[suffix]}"
    return field_name


def _safe_script_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def _display_environment_value(field_name: str, value: Any) -> str:
    if field_name in SECRET_FIELDS:
        return "环境变量中已配置" if value else "环境变量中未配置"
    if isinstance(value, bool):
        return "开启" if value else "关闭"
    if value is None:
        return "未配置"
    if value == "":
        return "空"
    return str(value)


def _render_field(
    field_name: str,
    config: ScopedConfig,
    environment_config: ScopedConfig,
    overridden_fields: set[str],
    pending_restart_fields: set[str],
) -> str:
    value = getattr(config, field_name)
    environment_value = getattr(environment_config, field_name)
    label = escape(_field_label(field_name))
    key = escape(field_name)
    badges: list[str] = []
    if field_name in overridden_fields:
        badges.append('<span class="badge override">网页覆盖</span>')
    if field_name in RESTART_REQUIRED_FIELDS:
        badges.append('<span class="badge restart">重启生效</span>')
    if field_name in pending_restart_fields:
        badges.append('<span class="badge pending">等待重启</span>')
    badge_html = "".join(badges)
    environment_hint = escape(
        _display_environment_value(field_name, environment_value)
    )

    if field_name in SECRET_FIELDS:
        configured = bool(value)
        control = (
            f'<input type="password" data-setting="{key}" data-secret="true" '
            f'autocomplete="new-password" placeholder="'
            f'{"已配置，留空保持不变" if configured else "尚未配置"}" />'
            f'<label class="clear-secret"><input type="checkbox" '
            f'data-clear-secret="{key}" /> 清除此密钥</label>'
        )
    elif isinstance(value, bool):
        control = (
            f'<label class="switch"><input type="checkbox" data-setting="{key}" '
            f'{"checked" if value else ""} /><span>开启</span></label>'
        )
    elif field_name.endswith("_api_format"):
        control = (
            f'<select data-setting="{key}">'
            f'<option value="openai" {"selected" if value == "openai" else ""}>OpenAI 兼容</option>'
            f'<option value="anthropic" {"selected" if value == "anthropic" else ""}>Anthropic</option>'
            f'<option value="vertex" {"selected" if value == "vertex" else ""}>Google Vertex AI</option>'
            "</select>"
        )
    elif field_name == "personality_setting":
        control = (
            f'<textarea data-setting="{key}" rows="5">'
            f'{escape(str(value))}</textarea>'
        )
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        step = "1" if isinstance(value, int) else "any"
        control = (
            f'<input type="number" step="{step}" data-setting="{key}" '
            f'value="{escape(str(value))}" />'
        )
    elif value is None and field_name.endswith("_dimension"):
        # 可空数值字段（如 embedding_dimension）值为 None 时渲染为空白，
        # 避免显示字面量 "None" 导致保存时解析失败。
        control = (
            f'<input type="number" step="1" data-setting="{key}" />'
        )
    else:
        control = (
            f'<input type="text" data-setting="{key}" '
            f'value="{escape(str(value))}" />'
        )

    return f"""
      <div class="setting-field">
        <div class="setting-title"><span>{label}</span>{badge_html}</div>
        <code>{key}</code>
        {control}
        <small>环境变量：{environment_hint}</small>
      </div>"""


def render_settings_page(
    config: ScopedConfig,
    environment_config: ScopedConfig,
    *,
    overridden_fields: set[str],
    pending_restart_fields: set[str],
    dashboard_path: str,
    settings_path: str,
    group_models_path: str | None = None,
) -> str:
    group_models_path = group_models_path or f"{settings_path}/groups"
    configured_fields = {
        field_name
        for _, _, fields in SETTING_GROUPS
        for field_name in fields
    }
    missing_fields = sorted(CONFIGURABLE_FIELDS - configured_fields)
    groups = list(SETTING_GROUPS)
    if missing_fields:
        groups.append(("其他", "尚未分类的配置项。", tuple(missing_fields)))

    group_html = "".join(
        f"""
        <section class="settings-section">
          <div class="section-head">
            <div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>
            <div class="section-actions">
              <span>{len(fields)} 项</span>
              {f'<button class="test-connection" type="button" data-test-role="{_TEST_ROLES[title]}" title="使用已保存配置测试">测试连接</button>' if title in _TEST_ROLES else ''}
            </div>
          </div>
          <div class="settings-grid">
            {''.join(_render_field(field_name, config, environment_config, overridden_fields, pending_restart_fields) for field_name in fields)}
          </div>
        </section>"""
        for title, description, fields in groups
    )
    restart_banner = (
        '<div class="notice warning">部分连接配置已保存，重启 Bot 后生效：'
        + escape("、".join(sorted(pending_restart_fields)))
        + "</div>"
        if pending_restart_fields
        else ""
    )
    api_path = f"{settings_path}/api"
    reset_path = f"{settings_path}/reset"
    logout_path = f"{settings_path}/logout"
    test_path = f"{settings_path}/test"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Groupmate · 配置中心</title>
  <style>
    :root {{ --bg:#f5f7fb; --panel:#fff; --text:#152238; --muted:#64748b; --line:#dce3ee; --accent:#0f766e; --warn:#b45309; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,"Segoe UI",Arial,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ position:sticky; top:0; z-index:5; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); backdrop-filter:blur(8px); }}
    .header-inner, main {{ max-width:1380px; margin:auto; padding-left:24px; padding-right:24px; }}
    .header-inner {{ min-height:72px; display:flex; align-items:center; gap:18px; }}
    h1 {{ margin:0; font-size:23px; }}
    .subtitle {{ color:var(--muted); font-size:13px; }}
    nav {{ margin-left:auto; display:flex; gap:10px; }}
    a, button {{ border-radius:8px; font-weight:700; text-decoration:none; cursor:pointer; }}
    nav a, nav button {{ padding:9px 13px; border:1px solid var(--line); background:#fff; color:var(--text); }}
    main {{ padding-top:22px; padding-bottom:110px; }}
    .notice {{ margin-bottom:16px; padding:13px 15px; border:1px solid #bae6fd; border-radius:10px; background:#f0f9ff; color:#075985; font-size:13px; }}
    .notice.warning {{ border-color:#fed7aa; background:#fff7ed; color:#9a3412; }}
    .settings-section {{ margin-bottom:16px; border:1px solid var(--line); border-radius:13px; background:var(--panel); overflow:hidden; }}
    .section-head {{ display:flex; justify-content:space-between; gap:16px; padding:18px 19px; border-bottom:1px solid var(--line); }}
    .section-head h2 {{ margin:0; font-size:17px; }}
    .section-head p {{ margin:6px 0 0; color:var(--muted); font-size:13px; }}
    .section-actions {{ display:flex; align-items:center; gap:9px; color:var(--muted); font-size:12px; white-space:nowrap; }}
    .test-connection {{ min-height:31px; padding:0 10px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--accent); font-weight:700; cursor:pointer; }}
    .settings-grid {{ display:grid; grid-template-columns:repeat(3,minmax(240px,1fr)); }}
    .setting-field {{ min-width:0; padding:16px 18px; border-right:1px solid #edf1f6; border-bottom:1px solid #edf1f6; }}
    .setting-title {{ min-height:24px; display:flex; flex-wrap:wrap; align-items:center; gap:6px; font-size:14px; font-weight:750; }}
    code {{ display:block; margin:3px 0 10px; color:#0f766e; font-size:11px; overflow-wrap:anywhere; }}
    input[type=text],input[type=password],input[type=number],select,textarea {{ width:100%; border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:#fff; color:var(--text); font:inherit; }}
    textarea {{ resize:vertical; }}
    small {{ display:block; margin-top:7px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }}
    .switch,.clear-secret {{ display:flex; align-items:center; gap:7px; color:var(--muted); font-size:12px; }}
    .clear-secret {{ margin-top:8px; }}
    .badge {{ padding:3px 6px; border-radius:999px; font-size:10px; font-weight:750; }}
    .badge.override {{ background:#ecfdf5; color:#047857; }} .badge.restart {{ background:#f1f5f9; color:#475569; }} .badge.pending {{ background:#fff7ed; color:#b45309; }}
    .actions {{ position:fixed; left:0; right:0; bottom:0; z-index:6; border-top:1px solid var(--line); background:rgba(255,255,255,.96); box-shadow:0 -8px 30px rgba(15,23,42,.07); }}
    .actions-inner {{ max-width:1380px; min-height:76px; margin:auto; padding:12px 24px; display:flex; align-items:center; gap:10px; }}
    .actions button {{ min-height:40px; padding:0 16px; border:1px solid var(--accent); background:var(--accent); color:#fff; }}
    .actions .secondary {{ border-color:var(--line); background:#fff; color:var(--text); }}
    #save-status {{ margin-left:auto; color:var(--muted); font-size:13px; }}
    @media(max-width:960px) {{ .settings-grid {{ grid-template-columns:repeat(2,minmax(220px,1fr)); }} }}
    @media(max-width:640px) {{
      .header-inner,main {{ padding-left:13px; padding-right:13px; }}
      .header-inner {{ align-items:flex-start; padding-top:14px; padding-bottom:14px; }}
      .subtitle {{ display:none; }}
      nav {{ display:grid; margin-left:auto; }}
      .settings-grid {{ grid-template-columns:1fr; }}
      .actions-inner {{ padding-left:13px; padding-right:13px; }}
      #save-status {{ display:none; }}
    }}
  </style>
</head>
<body>
  <header><div class="header-inner">
    <div><h1>配置中心</h1><div class="subtitle">环境变量作为默认值，网页保存值作为覆盖项。</div></div>
    <nav><a href="{escape(group_models_path)}">群聊 API</a><a href="{escape(dashboard_path)}">运行概览</a><button id="logout" type="button">退出</button></nav>
  </div></header>
  <main>
    <div class="notice">密钥字段不会回显；留空表示保持原值。WebUI 地址、开关和管理密码仍由环境变量控制。</div>
    {restart_banner}
    <form id="settings-form">{group_html}</form>
  </main>
  <div class="actions"><div class="actions-inner"><button id="save" type="button">保存配置</button><button id="reset" class="secondary" type="button">恢复环境变量</button><span id="save-status">{len(overridden_fields)} 项网页覆盖</span></div></div>
  <script>
    (() => {{
      const apiPath = {_safe_script_value(api_path)};
      const resetPath = {_safe_script_value(reset_path)};
      const logoutPath = {_safe_script_value(logout_path)};
      const testPath = {_safe_script_value(test_path)};
      const status = document.getElementById("save-status");
      const setStatus = (text, error=false) => {{ status.textContent=text; status.style.color=error?"#b91c1c":"#64748b"; }};
      document.getElementById("save").addEventListener("click", async () => {{
        const updates = {{}};
        document.querySelectorAll("[data-setting]").forEach((input) => {{
          if (input.dataset.secret === "true" && !input.value) return;
          updates[input.dataset.setting] = input.type === "checkbox" ? input.checked : input.value;
        }});
        const clearSecrets = [...document.querySelectorAll("[data-clear-secret]:checked")].map((input) => input.dataset.clearSecret);
        setStatus("正在保存…");
        try {{
          const response = await fetch(apiPath, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{updates, clear_secrets:clearSecrets}})}});
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || "保存失败");
          setStatus(data.restart_required ? "已保存，部分配置等待重启" : "已保存并生效");
          setTimeout(() => location.reload(), 700);
        }} catch (error) {{ setStatus(error.message || "保存失败", true); }}
      }});
      document.getElementById("reset").addEventListener("click", async () => {{
        if (!confirm("确定删除全部网页覆盖项并恢复环境变量配置吗？")) return;
        setStatus("正在恢复…");
        try {{
          const response = await fetch(resetPath, {{method:"POST"}});
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || "恢复失败");
          location.reload();
        }} catch (error) {{ setStatus(error.message || "恢复失败", true); }}
      }});
      document.querySelectorAll("[data-test-role]").forEach((button) => {{
        button.addEventListener("click", async () => {{
          const original = button.textContent;
          button.disabled = true;
          button.textContent = "测试中…";
          try {{
            const response = await fetch(testPath, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{role:button.dataset.testRole}})}});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "连接失败");
            button.textContent = "连接正常";
            setTimeout(() => {{ button.textContent = original; button.disabled = false; }}, 1600);
          }} catch (error) {{
            button.textContent = "连接失败";
            setStatus(error.message || "连接失败", true);
            setTimeout(() => {{ button.textContent = original; button.disabled = false; }}, 2200);
          }}
        }});
      }});
      document.getElementById("logout").addEventListener("click", async () => {{ await fetch(logoutPath, {{method:"POST"}}); location.reload(); }});
    }})();
  </script>
</body>
</html>"""


def render_settings_login(settings_path: str, *, auth_configured: bool) -> str:
    login_path = f"{settings_path}/login"
    if not auth_configured:
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>配置中心未启用</title>
  <style>
    body { font-family:Segoe UI,Arial; background:#f5f7fb; color:#152238; display:grid; place-items:center; min-height:100vh; margin:0; }
    .card { max-width:540px; padding:28px; border:1px solid #dce3ee; border-radius:14px; background:#fff; }
    code { color:#0f766e; }
  </style>
</head>
<body><div class="card"><h1>配置中心未启用</h1><p>配置中心会管理 API Key，必须先在环境变量中设置非空的 <code>ai_groupmate__usage_webui_token</code>，重启后再访问。</p></div></body>
</html>"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>登录配置中心</title>
  <style>
    body {{ font-family:Segoe UI,Arial; background:#f5f7fb; color:#152238; display:grid; place-items:center; min-height:100vh; margin:0; }}
    .card {{ width:min(92vw,420px); padding:28px; border:1px solid #dce3ee; border-radius:14px; background:#fff; }}
    input,button {{ width:100%; height:42px; margin-top:10px; border-radius:8px; border:1px solid #dce3ee; padding:0 11px; }}
    button {{ border-color:#0f766e; background:#0f766e; color:#fff; font-weight:700; cursor:pointer; }}
    #error {{ min-height:20px; color:#b91c1c; font-size:13px; }}
  </style>
</head>
<body>
  <div class="card"><h1>配置中心</h1><p>请输入 WebUI 管理密码。</p><input id="token" type="password" autocomplete="current-password" autofocus><button id="login">登录</button><p id="error"></p></div>
  <script>
    (() => {{
      const path = {_safe_script_value(login_path)};
      const input = document.getElementById("token");
      const error = document.getElementById("error");
      async function login() {{
        error.textContent = "";
        const response = await fetch(path, {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{token: input.value}}),
        }});
        if (response.ok) location.reload();
        else error.textContent = "管理密码错误";
      }}
      document.getElementById("login").onclick = login;
      input.addEventListener("keydown", event => {{ if (event.key === "Enter") login(); }});
    }})();
  </script>
</body>
</html>"""
