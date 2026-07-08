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
    auth_query = _auth_query(config, token)
    auth_suffix = f"&{auth_query}" if auth_query else ""

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

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Groupmate Usage</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent2: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 22px 28px 14px; border-bottom: 1px solid var(--line); background: var(--panel); }}
    h1 {{ margin: 0 0 12px; font-size: 22px; font-weight: 650; letter-spacing: 0; }}
    main {{ padding: 22px 28px 36px; max-width: 1440px; margin: 0 auto; }}
    form {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }}
    label {{ display: grid; gap: 5px; font-size: 12px; color: var(--muted); }}
    input, select, button {{ height: 34px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; background: white; color: var(--text); }}
    button {{ background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; font-weight: 600; }}
    .metrics {{ display: grid; grid-template-columns: repeat(7, minmax(130px, 1fr)); gap: 12px; margin: 18px 0 22px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .metric strong {{ font-size: 22px; font-weight: 680; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin: 16px 0; overflow: hidden; }}
    h2 {{ margin: 0; padding: 14px 16px; font-size: 15px; border-bottom: 1px solid var(--line); background: #fbfcfd; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #edf0f5; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 650; background: #fbfcfd; }}
    code {{ background: #eef6f4; color: #115e59; padding: 2px 5px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .empty {{ color: var(--muted); text-align: center; }}
    .links {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .links a {{ color: var(--accent2); text-decoration: none; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .metrics, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>AI Groupmate Token Usage</h1>
    <form method="get" action="{escape(path)}">
      <label>时间范围
        <select name="days">
          {"".join(f'<option value="{d}" {"selected" if int(data["days"]) == d else ""}>近 {d} 天</option>' for d in (1, 7, 30, 90))}
        </select>
      </label>
      <label>群/会话 ID <input name="session_id" value="{escape(data["filters"]["session_id"])}" placeholder="可选" /></label>
      <label>用户 ID <input name="user_id" value="{escape(data["filters"]["user_id"])}" placeholder="可选" /></label>
      {f'<input type="hidden" name="token" value="{escape(token or "")}" />' if config.usage_webui_token else ""}
      <button type="submit">筛选</button>
    </form>
    <div class="links">JSON API: <a href="{escape(path)}/api?days={int(data["days"])}{auth_suffix}">{escape(path)}/api</a></div>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><span>请求数</span><strong>{_fmt_int(total["requests"])}</strong></div>
      <div class="metric"><span>总 Tokens</span><strong>{_fmt_int(total["total_tokens"])}</strong></div>
      <div class="metric"><span>输入 Tokens</span><strong>{_fmt_int(total["prompt_tokens"])}</strong></div>
      <div class="metric"><span>输出 Tokens</span><strong>{_fmt_int(total["completion_tokens"])}</strong></div>
      <div class="metric"><span>缓存命中</span><strong>{_fmt_int(total["cached_tokens"])}</strong></div>
      <div class="metric"><span>缓存创建</span><strong>{_fmt_int(total["cache_creation_tokens"])}</strong></div>
      <div class="metric"><span>估算费用</span><strong>{_money(total["estimated_cost"])}</strong></div>
    </div>
    <div class="grid">
      <section><h2>按群/会话</h2><div class="table-wrap">{_render_table(["会话 ID", "类型", "请求", "Tokens", "缓存", "创建", "费用"], session_rows)}</div></section>
      <section><h2>按用户</h2><div class="table-wrap">{_render_table(["用户 ID", "名称", "请求", "Tokens", "缓存", "创建", "费用"], user_rows)}</div></section>
    </div>
    <section><h2>按模型</h2><div class="table-wrap">{_render_table(["模型", "请求", "输入", "输出", "缓存", "创建", "总计", "费用"], model_rows)}</div></section>
    <section><h2>最近请求</h2><div class="table-wrap">{_render_table(["时间", "会话", "用户", "名称", "模型", "Tokens", "缓存", "创建", "费用"], recent_rows)}</div></section>
  </main>
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
