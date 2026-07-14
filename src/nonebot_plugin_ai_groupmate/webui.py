from __future__ import annotations

from html import escape
from urllib.parse import urlencode

from nonebot import logger, get_driver
from nonebot_plugin_orm import get_session

from .usage import get_usage_dashboard_data
from .config import ScopedConfig


def _money(value: float) -> str:
    return f"¥{value:.6f}"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_ms(value: int) -> str:
    if value < 1_000:
        return f"{value} ms"
    return f"{value / 1_000:.2f} s"


def _fmt_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "—"
    return f"{numerator / denominator:.1%}"


def _fmt_average(total: int, count: int) -> str:
    if count <= 0:
        return "—"
    return f"{total / count:.2f}"


def _agent_issue_summary(row: dict) -> str:
    issues: list[str] = []
    if row["agent_tool_timeouts"]:
        issues.append(f"工具超时 {row['agent_tool_timeouts']}")
    if row["agent_result_truncations"]:
        issues.append(f"截断 {row['agent_result_truncations']}")
    if row["agent_side_effect_deduplications"]:
        issues.append(f"去重 {row['agent_side_effect_deduplications']}")
    if not issues:
        return '<span class="status ok">无工具异常</span>'
    return f'<span class="status warn">{" · ".join(issues)}</span>'


def _token_ok(config: ScopedConfig, token: str | None) -> bool:
    return not config.usage_webui_token or token == config.usage_webui_token


def _auth_query(config: ScopedConfig, token: str | None) -> str:
    return urlencode({"token": token}) if config.usage_webui_token and token else ""


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    if not rows:
        body = f"<tr><td colspan='{len(headers)}' class='empty'>暂无数据</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_dashboard(data: dict, *, path: str, token: str | None, config: ScopedConfig) -> str:
    total = data["total"]
    agent = data["agent"]
    auth_query = _auth_query(config, token)
    auth_suffix = f"&{auth_query}" if auth_query else ""
    cache_rate = _fmt_ratio(total["cached_tokens"], total["prompt_tokens"])
    llm_per_run = _fmt_average(agent["llm_calls"], agent["runs"])
    tools_per_run = _fmt_average(agent["tool_calls"], agent["runs"])
    agent_avg_duration = _fmt_ms(agent["avg_duration_ms"]) if agent["runs"] else "—"
    agent_health = (
        '<span class="status ok">工具无超时</span>'
        if not agent["tool_timeouts"]
        else '<span class="status warn">存在工具超时</span>'
    )

    session_rows = [
        [
            f"<code>{escape(row['session_id'])}</code>",
            escape(row["session_type"]),
            _fmt_int(row["requests"]),
            _fmt_int(row["total_tokens"]),
            _fmt_int(row["cached_tokens"]),
            _fmt_int(row["cache_creation_tokens"]),
            _money(row["estimated_cost"]),
        ]
        for row in data["by_session"]
    ]
    user_rows = [
        [
            f"<code>{escape(row['user_id'])}</code>",
            escape(row["user_name"]),
            _fmt_int(row["requests"]),
            _fmt_int(row["total_tokens"]),
            _fmt_int(row["cached_tokens"]),
            _fmt_int(row["cache_creation_tokens"]),
            _money(row["estimated_cost"]),
        ]
        for row in data["by_user"]
    ]
    model_rows = [
        [
            escape(row["model"]),
            _fmt_int(row["requests"]),
            _fmt_int(row["prompt_tokens"]),
            _fmt_int(row["completion_tokens"]),
            _fmt_int(row["cached_tokens"]),
            _fmt_int(row["cache_creation_tokens"]),
            _fmt_int(row["total_tokens"]),
            _money(row["estimated_cost"]),
        ]
        for row in data["by_model"]
    ]
    recent_rows = [
        [
            escape(row["created_at"][:19].replace("T", " ")),
            f"<code>{escape(row['session_id'])}</code>",
            f"<code>{escape(row['user_id'])}</code>",
            escape(row["user_name"]),
            escape(row["model"]),
            _fmt_int(row["total_tokens"]),
            _fmt_int(row["cached_tokens"]),
            _fmt_int(row["cache_creation_tokens"]),
            _money(row["estimated_cost"]),
        ]
        for row in data["recent"]
    ]
    agent_session_rows = [
        [
            f"<code>{escape(row['session_id'])}</code>",
            _fmt_int(row["requests"]),
            _fmt_ms(row["agent_avg_duration_ms"]),
            f"{_fmt_int(row['agent_llm_calls'])} / {_fmt_int(row['agent_tool_calls'])}",
            _agent_issue_summary(row),
        ]
        for row in data["agent_by_session"]
    ]
    agent_recent_rows = [
        [
            escape(row["created_at"][:19].replace("T", " ")),
            f"<code>{escape(row['session_id'])}</code>",
            _fmt_ms(row["agent_duration_ms"]),
            f"{_fmt_int(row['agent_llm_calls'])} / {_fmt_int(row['agent_tool_calls'])}",
            _agent_issue_summary(row),
        ]
        for row in data["agent_recent"]
    ]
    agent_sessions = {
        row["session_id"]: row for row in data["agent_by_session"]
    }
    group_rows = []
    for row in data["by_session"]:
        agent_row = agent_sessions.get(row["session_id"])
        agent_runs = agent_row["requests"] if agent_row else 0
        group_rows.append(
            [
                f"<code>{escape(row['session_id'])}</code>",
                escape(row["session_type"]),
                _fmt_int(row["requests"]),
                _fmt_int(row["total_tokens"]),
                _money(row["estimated_cost"]),
                _fmt_int(agent_runs),
                _fmt_ms(agent_row["agent_avg_duration_ms"]) if agent_row else "—",
                (
                    f"{_fmt_average(agent_row['agent_llm_calls'], agent_runs)} / "
                    f"{_fmt_average(agent_row['agent_tool_calls'], agent_runs)}"
                    if agent_row
                    else "—"
                ),
                _agent_issue_summary(agent_row)
                if agent_row
                else '<span class="status neutral">暂无指标</span>',
            ]
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Groupmate · 运行概览</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #152238;
      --muted: #64748b;
      --line: #dce3ee;
      --accent: #0f766e;
      --accent-soft: #ecfdf5;
      --blue: #2563eb;
      --blue-soft: #eff6ff;
      --warn: #b45309;
      --warn-soft: #fff7ed;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .page-header {{ background: var(--panel); border-bottom: 1px solid var(--line); }}
    .header-inner, main {{ max-width: 1440px; margin: 0 auto; padding-left: 28px; padding-right: 28px; }}
    .header-top {{ display: flex; justify-content: space-between; gap: 20px; padding-top: 26px; align-items: start; }}
    .eyebrow {{ margin: 0 0 6px; color: var(--accent); font-size: 12px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }}
    h1 {{ margin: 0; font-size: 27px; letter-spacing: -.03em; }}
    .subtitle {{ margin: 8px 0 0; color: var(--muted); font-size: 14px; }}
    .period {{ padding: 7px 10px; border-radius: 999px; background: var(--blue-soft); color: var(--blue); font-size: 13px; font-weight: 700; white-space: nowrap; }}
    .filter-panel {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin-top: 22px; padding: 14px 0 18px; }}
    label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; }}
    input, select, button {{ height: 36px; border: 1px solid var(--line); border-radius: 7px; padding: 0 10px; background: var(--panel); color: var(--text); }}
    input {{ min-width: 160px; }}
    button {{ padding: 0 16px; border-color: var(--accent); background: var(--accent); color: white; cursor: pointer; font-weight: 700; }}
    .links {{ margin-left: auto; color: var(--muted); font-size: 12px; }}
    .links a {{ color: var(--accent); text-decoration: none; }}
    main {{ padding-top: 22px; padding-bottom: 42px; }}
    .tabs {{ display: flex; gap: 6px; margin-bottom: 16px; padding: 5px; border: 1px solid var(--line); border-radius: 11px; background: var(--panel); overflow-x: auto; }}
    .tab-button {{ flex: 0 0 auto; height: 38px; padding: 0 17px; border: 0; border-radius: 7px; background: transparent; color: var(--muted); font-weight: 750; }}
    .tab-button:hover {{ background: #f1f5f9; color: var(--text); }}
    .tab-button.active {{ background: var(--accent); color: white; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    section {{ margin: 16px 0; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); overflow: hidden; }}
    .tab-panel > section:first-child {{ margin-top: 0; }}
    .section-head {{ display: flex; justify-content: space-between; align-items: start; gap: 16px; padding: 17px 18px 0; }}
    .section-head h2 {{ margin: 0; font-size: 17px; letter-spacing: -.01em; }}
    .section-copy {{ margin: 6px 18px 16px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; }}
    .metric-grid.agent {{ grid-template-columns: repeat(6, minmax(130px, 1fr)); padding: 0 18px 18px; }}
    .metric {{ min-height: 114px; padding: 15px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }}
    .metric.primary {{ border-color: #bfdbfe; background: var(--blue-soft); }}
    .metric.accent {{ border-color: #a7f3d0; background: var(--accent-soft); }}
    .metric-label {{ display: block; color: var(--muted); font-size: 12px; font-weight: 650; }}
    .metric strong {{ display: block; margin-top: 9px; font-size: 24px; letter-spacing: -.03em; }}
    .metric-hint {{ display: block; margin-top: 6px; color: var(--muted); font-size: 12px; }}
    .overview {{ border: 0; background: transparent; overflow: visible; }}
    .status {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .status.ok {{ color: #047857; background: var(--accent-soft); }}
    .status.warn {{ color: var(--warn); background: var(--warn-soft); }}
    .status.neutral {{ color: var(--muted); background: #f1f5f9; }}
    .agent-panel {{ border-color: #b7ead5; }}
    .table-wrap {{ overflow-x: auto; }}
    .table-scroll {{ max-height: calc(100vh - 300px); overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 11px 14px; border-top: 1px solid #edf1f6; text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; background: #fbfcfe; }}
    .table-scroll th {{ position: sticky; top: 0; z-index: 1; box-shadow: 0 1px 0 #edf1f6; }}
    code {{ padding: 2px 5px; border-radius: 4px; background: #edf6f5; color: #0f5f58; }}
    .empty {{ color: var(--muted); text-align: center; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .table-card {{ min-width: 0; }}
    .table-card h3 {{ margin: 0; padding: 15px 16px 9px; font-size: 14px; }}
    details {{ margin-top: 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }}
    summary {{ display: flex; justify-content: space-between; gap: 16px; padding: 16px 18px; cursor: pointer; font-size: 15px; font-weight: 750; }}
    summary span {{ color: var(--muted); font-size: 12px; font-weight: 400; }}
    .details-content {{ padding: 0 16px 16px; }}
    @media (max-width: 1080px) {{
      .metric-grid.agent {{ grid-template-columns: repeat(3, minmax(150px, 1fr)); }}
    }}
    @media (max-width: 760px) {{
      .header-inner, main {{ padding-left: 14px; padding-right: 14px; }}
      .header-top, .section-head {{ display: block; }}
      .period {{ display: inline-flex; margin-top: 12px; }}
      .filter-panel {{ align-items: stretch; }}
      .filter-panel label, input, select, button {{ width: 100%; }}
      .links {{ margin-left: 0; }}
      .metric-grid, .metric-grid.agent, .grid {{ grid-template-columns: 1fr; }}
      .tabs {{ margin-left: -14px; margin-right: -14px; border-left: 0; border-right: 0; border-radius: 0; }}
      .table-scroll {{ max-height: calc(100vh - 260px); }}
      summary {{ display: block; }}
      summary span {{ display: block; margin-top: 5px; }}
    }}
  </style>
</head>
<body>
  <header class="page-header">
    <div class="header-inner">
      <div class="header-top">
        <div>
          <p class="eyebrow">AI Groupmate</p>
          <h1>运行与用量概览</h1>
          <p class="subtitle">先看 Agent 是否健康，再按会话追踪耗时与调用情况。</p>
        </div>
        <span class="period">近 {int(data["days"])} 天</span>
      </div>
      <form class="filter-panel" method="get" action="{escape(path)}">
        <label>时间范围
          <select name="days">
            {"".join(f'<option value="{d}" {"selected" if int(data["days"]) == d else ""}>近 {d} 天</option>' for d in (1, 7, 30, 90))}
          </select>
        </label>
        <label>群/会话 ID <input name="session_id" value="{escape(data["filters"]["session_id"])}" placeholder="可选" /></label>
        <label>用户 ID <input name="user_id" value="{escape(data["filters"]["user_id"])}" placeholder="可选" /></label>
        {f'<input type="hidden" name="token" value="{escape(token or "")}" />' if config.usage_webui_token else ""}
        <button type="submit">更新数据</button>
        <div class="links">JSON：<a href="{escape(path)}/api?days={int(data["days"])}{auth_suffix}">{escape(path)}/api</a></div>
      </form>
    </div>
  </header>
  <main>
    <nav class="tabs" role="tablist" aria-label="统计视图">
      <button class="tab-button active" type="button" role="tab" aria-selected="true" aria-controls="panel-groups" data-tab="groups">分群对比</button>
      <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="panel-overview" data-tab="overview">整体概览</button>
      <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="panel-recent" data-tab="recent">最近运行</button>
      <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="panel-usage" data-tab="usage">用量明细</button>
    </nav>

    <div class="tab-panel active" id="panel-groups" role="tabpanel" data-panel="groups">
      <section>
        <div class="section-head"><h2>分群效果对比</h2><span class="status ok">{_fmt_int(len(group_rows))} 个会话</span></div>
        <p class="section-copy">在同一行比较各群的消耗与 Agent 表现；LLM / 工具显示每次运行的平均调用数。</p>
        <div class="table-wrap table-scroll">{_render_table(["群/会话 ID", "类型", "请求", "Tokens", "费用", "Agent 运行", "平均耗时", "LLM / 工具", "状态"], group_rows)}</div>
      </section>
    </div>

    <div class="tab-panel" id="panel-overview" role="tabpanel" data-panel="overview" hidden>
      <section class="overview">
        <div class="section-head"><h2>用量概览</h2></div>
        <p class="section-copy">所有已落库的模型请求；缓存占比按输入 Tokens 计算。</p>
        <div class="metric-grid">
          <div class="metric primary"><span class="metric-label">请求记录</span><strong>{_fmt_int(total["requests"])}</strong><span class="metric-hint">当前筛选范围内</span></div>
          <div class="metric"><span class="metric-label">总 Tokens</span><strong>{_fmt_int(total["total_tokens"])}</strong><span class="metric-hint">输入 {_fmt_int(total["prompt_tokens"])} · 输出 {_fmt_int(total["completion_tokens"])} </span></div>
          <div class="metric"><span class="metric-label">缓存占比</span><strong>{cache_rate}</strong><span class="metric-hint">缓存 {_fmt_int(total["cached_tokens"])} Tokens</span></div>
          <div class="metric accent"><span class="metric-label">估算费用</span><strong>{_money(total["estimated_cost"])}</strong><span class="metric-hint">按当前价格配置估算</span></div>
        </div>
      </section>
      <section class="agent-panel">
        <div class="section-head"><h2>Agent 整体运行</h2>{agent_health}</div>
        <p class="section-copy">仅统计启用 Agent 指标后成功完成的运行。</p>
        <div class="metric-grid agent">
          <div class="metric primary"><span class="metric-label">已观测运行</span><strong>{_fmt_int(agent["runs"])}</strong><span class="metric-hint">可用于性能分析</span></div>
          <div class="metric"><span class="metric-label">平均耗时</span><strong>{agent_avg_duration}</strong><span class="metric-hint">每次完整 Agent 运行</span></div>
          <div class="metric"><span class="metric-label">LLM / 运行</span><strong>{llm_per_run}</strong><span class="metric-hint">共 {_fmt_int(agent["llm_calls"])} 次调用</span></div>
          <div class="metric"><span class="metric-label">工具 / 运行</span><strong>{tools_per_run}</strong><span class="metric-hint">共 {_fmt_int(agent["tool_calls"])} 次调用</span></div>
          <div class="metric"><span class="metric-label">工具超时</span><strong>{_fmt_int(agent["tool_timeouts"])}</strong><span class="metric-hint">超时会标记为需关注</span></div>
          <div class="metric"><span class="metric-label">结果控制</span><strong>{_fmt_int(agent["result_truncations"])} / {_fmt_int(agent["side_effect_deduplications"])}</strong><span class="metric-hint">截断 / 去重</span></div>
        </div>
      </section>
    </div>

    <div class="tab-panel" id="panel-recent" role="tabpanel" data-panel="recent" hidden>
      <div class="grid">
        <section class="table-card"><h3>最近 Agent 运行</h3><div class="table-wrap table-scroll">{_render_table(["时间", "会话", "耗时", "LLM / 工具", "状态"], agent_recent_rows)}</div></section>
        <section class="table-card"><h3>最近模型请求</h3><div class="table-wrap table-scroll">{_render_table(["时间", "会话", "用户", "名称", "模型", "Tokens", "缓存", "创建", "费用"], recent_rows)}</div></section>
      </div>
    </div>

    <div class="tab-panel" id="panel-usage" role="tabpanel" data-panel="usage" hidden>
      <div class="grid">
        <section class="table-card"><h3>按模型</h3><div class="table-wrap table-scroll">{_render_table(["模型", "请求", "输入", "输出", "缓存", "创建", "总计", "费用"], model_rows)}</div></section>
        <section class="table-card"><h3>按会话</h3><div class="table-wrap table-scroll">{_render_table(["会话 ID", "类型", "请求", "Tokens", "缓存", "创建", "费用"], session_rows)}</div></section>
        <section class="table-card"><h3>按用户</h3><div class="table-wrap table-scroll">{_render_table(["用户 ID", "名称", "请求", "Tokens", "缓存", "创建", "费用"], user_rows)}</div></section>
        <section class="table-card"><h3>Agent 分群指标</h3><div class="table-wrap table-scroll">{_render_table(["会话 ID", "运行", "平均耗时", "LLM / 工具", "状态"], agent_session_rows)}</div></section>
      </div>
    </div>
  </main>
  <script>
    (() => {{
      const buttons = [...document.querySelectorAll("[data-tab]")];
      const panels = [...document.querySelectorAll("[data-panel]")];
      const validTabs = new Set(buttons.map((button) => button.dataset.tab));
      const activate = (name, updateHash = true) => {{
        if (!validTabs.has(name)) name = "groups";
        buttons.forEach((button) => {{
          const active = button.dataset.tab === name;
          button.classList.toggle("active", active);
          button.setAttribute("aria-selected", String(active));
        }});
        panels.forEach((panel) => {{
          const active = panel.dataset.panel === name;
          panel.classList.toggle("active", active);
          panel.hidden = !active;
        }});
        if (updateHash) history.replaceState(null, "", `#${{name}}`);
      }};
      buttons.forEach((button) => button.addEventListener("click", () => activate(button.dataset.tab)));
      activate(location.hash.slice(1) || "groups", false);
    }})();
  </script>
</body>
</html>"""


def register_usage_webui(config: ScopedConfig) -> None:
    if not config.usage_webui_enabled:
        return

    driver = get_driver()
    app = getattr(driver, "server_app", None)
    if app is None:
        logger.warning("Token 用量 WebUI 需要 FastAPI driver，当前 driver 不支持 server_app，已跳过注册")
        return

    try:
        from fastapi import Query, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
    except Exception as e:
        logger.warning(f"Token 用量 WebUI 依赖 FastAPI，导入失败，已跳过注册: {e}")
        return

    path = "/" + config.usage_webui_path.strip("/")
    api_path = f"{path}/api"

    async def _load_data(days: int, session_id: str, user_id: str) -> dict:
        async with get_session() as db_session:
            return await get_usage_dashboard_data(
                db_session,
                config=config,
                days=days,
                session_id=session_id.strip() or None,
                user_id=user_id.strip() or None,
            )

    @app.get(path, response_class=HTMLResponse, include_in_schema=False)
    async def usage_page(
        days: int = Query(7, ge=1, le=3650),
        session_id: str = "",
        user_id: str = "",
        token: str | None = None,
    ):
        if not _token_ok(config, token):
            raise HTTPException(status_code=401, detail="invalid token")
        data = await _load_data(days, session_id, user_id)
        return HTMLResponse(_render_dashboard(data, path=path, token=token, config=config))

    @app.get(api_path, response_class=JSONResponse, include_in_schema=False)
    async def usage_api(
        days: int = Query(7, ge=1, le=3650),
        session_id: str = "",
        user_id: str = "",
        token: str | None = None,
    ):
        if not _token_ok(config, token):
            raise HTTPException(status_code=401, detail="invalid token")
        return JSONResponse(await _load_data(days, session_id, user_id))

    logger.info(f"Token 用量 WebUI 已注册: {path}")
