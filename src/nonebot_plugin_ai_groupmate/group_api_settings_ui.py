import json
from html import escape


def _safe_script_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def render_group_api_settings_page(
    *,
    dashboard_path: str,
    settings_path: str,
    group_models_path: str,
) -> str:
    api_path = f"{group_models_path}/api"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="referrer" content="no-referrer" />
  <title>AI Groupmate · 群聊 API</title>
  <style>
    :root {{ --bg:#f5f7fb; --panel:#fff; --text:#152238; --muted:#64748b; --line:#dce3ee; --accent:#0f766e; --accent-soft:#ecfdf5; --danger:#b91c1c; --danger-soft:#fef2f2; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,"Segoe UI",Arial,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ position:sticky; top:0; z-index:5; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); backdrop-filter:blur(8px); }}
    .header-inner,main {{ max-width:1380px; margin:auto; padding-left:24px; padding-right:24px; }}
    .header-inner {{ min-height:72px; display:flex; align-items:center; gap:18px; }}
    h1 {{ margin:0; font-size:23px; }}
    .subtitle {{ color:var(--muted); font-size:13px; }}
    nav {{ margin-left:auto; display:flex; gap:10px; }}
    a,button {{ border-radius:8px; font-weight:700; text-decoration:none; cursor:pointer; }}
    nav a {{ padding:9px 13px; border:1px solid var(--line); background:#fff; color:var(--text); }}
    main {{ padding-top:22px; padding-bottom:48px; }}
    .notice {{ margin-bottom:16px; padding:13px 15px; border:1px solid #bae6fd; border-radius:10px; background:#f0f9ff; color:#075985; font-size:13px; line-height:1.6; }}
    .layout {{ display:grid; grid-template-columns:minmax(320px,430px) minmax(0,1fr); gap:16px; align-items:start; }}
    section {{ border:1px solid var(--line); border-radius:13px; background:var(--panel); overflow:hidden; }}
    .section-head {{ display:flex; justify-content:space-between; gap:16px; padding:18px 19px; border-bottom:1px solid var(--line); }}
    .section-head h2 {{ margin:0; font-size:17px; }}
    .section-head p {{ margin:6px 0 0; color:var(--muted); font-size:13px; line-height:1.5; }}
    .count {{ align-self:center; padding:4px 8px; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-size:12px; font-weight:750; white-space:nowrap; }}
    form {{ display:grid; gap:14px; padding:18px 19px 20px; }}
    label {{ display:grid; gap:6px; color:var(--text); font-size:13px; font-weight:700; }}
    label small {{ color:var(--muted); font-size:11px; font-weight:400; line-height:1.5; }}
    input,select {{ width:100%; min-height:40px; border:1px solid var(--line); border-radius:8px; padding:8px 10px; background:#fff; color:var(--text); font:inherit; }}
    input:focus,select:focus {{ outline:2px solid rgba(15,118,110,.16); border-color:var(--accent); }}
    .switch {{ display:flex; grid-template-columns:none; align-items:center; gap:8px; color:var(--muted); font-weight:500; }}
    .switch input {{ width:auto; min-height:auto; }}
    .form-actions {{ display:flex; gap:9px; padding-top:3px; }}
    button {{ min-height:40px; padding:0 15px; border:1px solid var(--accent); background:var(--accent); color:#fff; }}
    button.secondary {{ border-color:var(--line); background:#fff; color:var(--text); }}
    button.danger {{ border-color:#fecaca; background:var(--danger-soft); color:var(--danger); }}
    button:disabled {{ cursor:wait; opacity:.65; }}
    #form-status {{ min-height:19px; margin:0; color:var(--muted); font-size:12px; line-height:1.5; }}
    #form-status.error {{ color:var(--danger); }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:12px 13px; border-top:1px solid #edf1f6; text-align:left; white-space:nowrap; }}
    th {{ border-top:0; color:var(--muted); background:#fbfcfe; font-size:12px; }}
    td code {{ padding:2px 5px; border-radius:4px; background:#edf6f5; color:#0f5f58; }}
    .provider {{ max-width:210px; overflow:hidden; text-overflow:ellipsis; }}
    .row-actions {{ display:flex; gap:7px; }}
    .row-actions button {{ min-height:32px; padding:0 10px; font-size:12px; }}
    .empty {{ padding:44px 20px; color:var(--muted); text-align:center; }}
    .status {{ display:inline-flex; padding:3px 7px; border-radius:999px; background:var(--accent-soft); color:#047857; font-size:11px; font-weight:750; }}
    @media(max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} }}
    @media(max-width:640px) {{
      .header-inner,main {{ padding-left:13px; padding-right:13px; }}
      .header-inner {{ align-items:flex-start; padding-top:14px; padding-bottom:14px; }}
      .subtitle {{ display:none; }}
      nav {{ display:grid; margin-left:auto; }}
      .form-actions {{ display:grid; }}
    }}
  </style>
</head>
<body>
  <header><div class="header-inner"><div><h1>群聊 API</h1><div class="subtitle">为指定群配置独立的主聊天模型。</div></div><nav><a href="{escape(settings_path)}">全局配置</a><a href="{escape(dashboard_path)}">运行概览</a></nav></div></header>
  <main>
    <div class="notice">未配置的群继续使用全局主聊天 API。独立配置只影响指定群；API Key 加密保存且不会回显，编辑时留空即可保留原 Key。</div>
    <div class="layout">
      <section>
        <div class="section-head"><div><h2 id="form-title">新增群配置</h2><p>保存前会实际调用一次模型验证连接。</p></div></div>
        <form id="group-form" autocomplete="off">
          <label>群 ID
            <input id="group-id" name="group_id" type="text" maxlength="160" required placeholder="例如：123456789" />
            <small>填写适配器上报的群号或会话 ID；编辑已有配置时不可修改。</small>
          </label>
          <label>接口格式
            <select id="api-format" name="api_format">
              <option value="openai">OpenAI 兼容</option>
              <option value="anthropic">Anthropic Messages</option>
              <option value="vertex">Google Vertex AI</option>
            </select>
          </label>
          <label>API Base URL
            <input id="base-url" name="base_url" type="url" maxlength="2048" required value="https://api.openai.com/v1" />
            <small>必须使用 HTTPS；内网地址需要在 Bot 配置中显式加入白名单。</small>
          </label>
          <label>API Key
            <input id="api-key" name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="新建时必填" />
            <small id="api-key-hint">密钥不会回显。</small>
          </label>
          <label>模型名称
            <input id="chat-model" name="chat_model" type="text" maxlength="256" required placeholder="例如：gpt-4.1-mini" />
          </label>
          <label class="switch"><input id="chat-multimodal" name="chat_multimodal" type="checkbox" checked /> 模型支持图片输入</label>
          <div class="form-actions"><button id="save" type="submit">测试并保存</button><button id="cancel" class="secondary" type="button" hidden>取消编辑</button></div>
          <p id="form-status" aria-live="polite"></p>
        </form>
      </section>
      <section>
        <div class="section-head"><div><h2>已配置群聊</h2><p>删除独立配置后，该群立即恢复使用全局 API。</p></div><span id="group-count" class="count">0 个群</span></div>
        <div id="group-list" class="table-wrap"><div class="empty">正在加载…</div></div>
      </section>
    </div>
  </main>
  <script>
    (() => {{
      const apiPath = {_safe_script_value(api_path)};
      const defaults = {{openai:"https://api.openai.com/v1",anthropic:"https://api.anthropic.com",vertex:"https://aiplatform.googleapis.com"}};
      const form = document.getElementById("group-form");
      const groupId = document.getElementById("group-id");
      const apiFormat = document.getElementById("api-format");
      const baseUrl = document.getElementById("base-url");
      const apiKey = document.getElementById("api-key");
      const chatModel = document.getElementById("chat-model");
      const multimodal = document.getElementById("chat-multimodal");
      const save = document.getElementById("save");
      const cancel = document.getElementById("cancel");
      const status = document.getElementById("form-status");
      const list = document.getElementById("group-list");
      let groups = [];
      let editing = null;

      const setStatus = (message, error=false) => {{ status.textContent=message; status.classList.toggle("error", error); }};
      const resetForm = () => {{
        editing = null;
        form.reset();
        groupId.disabled = false;
        baseUrl.value = defaults.openai;
        document.getElementById("form-title").textContent = "新增群配置";
        document.getElementById("api-key-hint").textContent = "新建配置时必须填写；密钥不会回显。";
        apiKey.placeholder = "新建时必填";
        cancel.hidden = true;
        setStatus("");
      }};
      const startEdit = (id) => {{
        const item = groups.find(group => group.group_id === id);
        if (!item) return;
        editing = id;
        groupId.value = item.group_id;
        groupId.disabled = true;
        apiFormat.value = item.api_format;
        baseUrl.value = item.base_url;
        apiKey.value = "";
        apiKey.placeholder = "留空保持原 Key";
        chatModel.value = item.chat_model;
        multimodal.checked = item.chat_multimodal;
        document.getElementById("form-title").textContent = `编辑群 ${{item.group_id}}`;
        document.getElementById("api-key-hint").textContent = "已配置 API Key；留空将保留原 Key。";
        cancel.hidden = false;
        setStatus("");
        window.scrollTo({{top:0,behavior:"smooth"}});
      }};
      const render = () => {{
        document.getElementById("group-count").textContent = `${{groups.length}} 个群`;
        if (!groups.length) {{ list.innerHTML='<div class="empty">暂无独立群 API 配置</div>'; return; }}
        const table = document.createElement("table");
        table.innerHTML = "<thead><tr><th>群 ID</th><th>接口</th><th>服务地址</th><th>模型</th><th>图片</th><th>版本</th><th>操作</th></tr></thead>";
        const body = document.createElement("tbody");
        groups.forEach(item => {{
          const row = document.createElement("tr");
          const values = [item.group_id,item.api_format,item.provider_host,item.chat_model,item.chat_multimodal?"开启":"关闭",`v${{item.version}}`];
          values.forEach((value,index) => {{ const cell=document.createElement("td"); if(index===0){{const code=document.createElement("code");code.textContent=value;cell.append(code);}}else{{cell.textContent=value;}} if(index===2)cell.className="provider"; row.append(cell); }});
          const actions = document.createElement("td");
          actions.className = "row-actions";
          const edit = document.createElement("button"); edit.type="button"; edit.className="secondary"; edit.textContent="编辑"; edit.onclick=()=>startEdit(item.group_id);
          const remove = document.createElement("button"); remove.type="button"; remove.className="danger"; remove.textContent="删除"; remove.onclick=()=>deleteGroup(item.group_id);
          actions.append(edit,remove); row.append(actions); body.append(row);
        }});
        table.append(body); list.replaceChildren(table);
      }};
      const load = async () => {{
        const response = await fetch(apiPath, {{headers:{{"Accept":"application/json"}},cache:"no-store"}});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "加载失败");
        groups = Array.isArray(data.groups) ? data.groups : [];
        render();
      }};
      const deleteGroup = async (id) => {{
        if (!confirm(`确定删除群 ${{id}} 的独立 API 配置吗？该群会恢复使用全局 API。`)) return;
        const response = await fetch(`${{apiPath}}/${{encodeURIComponent(id)}}`, {{method:"DELETE"}});
        const data = await response.json();
        if (!response.ok) {{ alert(data.detail || "删除失败"); return; }}
        if (editing === id) resetForm();
        await load();
      }};
      apiFormat.addEventListener("change", () => {{
        if (!baseUrl.value || Object.values(defaults).includes(baseUrl.value)) baseUrl.value = defaults[apiFormat.value];
      }});
      cancel.addEventListener("click", resetForm);
      form.addEventListener("submit", async event => {{
        event.preventDefault();
        save.disabled = true;
        save.textContent = "正在测试…";
        setStatus("正在连接模型服务，成功后会自动保存…");
        try {{
          const payload = {{group_id:editing || groupId.value.trim(),api_format:apiFormat.value,base_url:baseUrl.value.trim(),api_key:apiKey.value,chat_model:chatModel.value.trim(),chat_multimodal:multimodal.checked}};
          const response = await fetch(apiPath, {{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(payload)}});
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || "保存失败");
          resetForm();
          setStatus(`群 ${{data.group.group_id}} 已保存并生效`);
          await load();
        }} catch (error) {{ setStatus(error.message || "保存失败", true); }}
        finally {{ save.disabled=false; save.textContent="测试并保存"; }}
      }});
      resetForm();
      load().catch(error => {{ list.innerHTML='<div class="empty">加载失败</div>'; setStatus(error.message || "加载失败", true); }});
    }})();
  </script>
</body>
</html>"""
