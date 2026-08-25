import asyncio
import secrets
import datetime
from html import escape
from typing import Any
from collections.abc import Callable, Iterable

from nonebot import logger, get_driver
from pydantic import ValidationError
from nonebot_plugin_orm import get_session
from langchain_core.messages import HumanMessage

from .usage import get_usage_dashboard_data
from .config import (
    ScopedConfig,
    create_chat_llm,
    create_vision_llm,
    create_chat_openai,
    create_tagging_llm,
)
from .settings_ui import render_settings_page, render_settings_login
from .runtime_config import (
    SECRET_FIELDS,
    get_runtime_config,
    get_config_overrides,
    get_environment_config,
    get_pending_restart_fields,
    save_runtime_config_updates,
    reset_runtime_config_overrides,
)
from .group_model_config import (
    GroupModelPayload,
    LocalSecretCipher,
    GroupModelConfigError,
    LocalEncryptionKeyError,
    save_group_model_config,
    list_group_model_configs,
    delete_group_model_config,
    build_candidate_chat_config,
    get_group_model_config_detail,
    get_decrypted_group_model_config,
    validate_group_model_test_response,
    validate_group_provider_resolution,
)
from .group_api_settings_ui import render_group_api_settings_page

SETTINGS_COOKIE_NAME = "ai_groupmate_settings"
SETTINGS_COOKIE_MAX_AGE = 7 * 24 * 60 * 60


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


def _tool_timeout_summary(counts: dict) -> str:
    if not isinstance(counts, dict):
        return ""
    items = sorted(
        (
            (str(name), int(count))
            for name, count in counts.items()
            if isinstance(count, int) and count > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return "、".join(
        f"{escape(name)}×{_fmt_int(count)}"
        for name, count in items
    )


def _agent_issue_summary(row: dict) -> str:
    issues: list[str] = []
    if row["agent_tool_timeouts"]:
        timeout_summary = _tool_timeout_summary(
            row.get("agent_tool_timeout_tools", {})
        )
        timeout_detail = f"（{timeout_summary}）" if timeout_summary else ""
        issues.append(f"工具超时 {row['agent_tool_timeouts']}{timeout_detail}")
    if row["agent_result_truncations"]:
        issues.append(f"截断 {row['agent_result_truncations']}")
    if row["agent_side_effect_deduplications"]:
        issues.append(f"去重 {row['agent_side_effect_deduplications']}")
    if not issues:
        return '<span class="status ok">无工具异常</span>'
    return f'<span class="status warn">{" · ".join(issues)}</span>'


def _settings_token_ok(config: ScopedConfig, request: Any) -> bool:
    expected = config.usage_webui_token
    supplied = request.cookies.get(SETTINGS_COOKIE_NAME, "")
    return bool(expected and supplied) and secrets.compare_digest(supplied, expected)


def _validation_detail(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_input=False):
        location = ".".join(str(part) for part in item.get("loc", ()))
        message = str(item.get("msg", "配置值无效"))
        details.append(f"{location}: {message}" if location else message)
    return "；".join(details) or "配置值无效"


async def _test_model_connection(role: str, config: ScopedConfig) -> None:
    if role == "chat":
        model = create_chat_llm(config)
    elif role == "tagging":
        model = create_tagging_llm(config)
    elif role == "vision":
        if not config.vision_model:
            raise ValueError("尚未配置图片回读模型")
        model = create_vision_llm(config)
    elif role in {"flash", "summary"}:
        model = create_chat_openai(config, role, max_tokens=8)
    else:
        raise ValueError("不支持测试这个模型角色")
    response = await asyncio.wait_for(
        model.ainvoke([HumanMessage(content="请只回复 OK")]),
        timeout=min(config.agent_llm_timeout_seconds, 20.0),
    )
    validate_group_model_test_response(response)


def _safe_connection_error(
    error: Exception,
    config: ScopedConfig,
    *,
    extra_secrets: Iterable[str] = (),
) -> str:
    message = str(error)
    configured_secrets = (
        str(getattr(config, field_name, "") or "")
        for field_name in SECRET_FIELDS
    )
    for secret_value in (*configured_secrets, *extra_secrets):
        if secret_value:
            message = message.replace(secret_value, "***")
    message = " ".join(message.split())
    return message[:240] or type(error).__name__


async def _test_group_model_connection(
    payload: GroupModelPayload,
    config: ScopedConfig,
) -> None:
    candidate = build_candidate_chat_config(payload, config)
    await validate_group_provider_resolution(
        payload.base_url,
        config.group_api_allowed_provider_hosts,
    )
    model = create_chat_llm(candidate)
    await asyncio.wait_for(
        model.ainvoke([HumanMessage(content="请只回复 OK")]),
        timeout=min(config.agent_llm_timeout_seconds, 20.0),
    )


def _clear_group_chat_model_cache(group_id: str) -> None:
    from .agent import clear_group_chat_model_cache

    clear_group_chat_model_cache(group_id)


def _validate_group_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("群 ID 必须是字符串")
    group_id = value.strip()
    if not group_id:
        raise ValueError("请填写群 ID")
    if len(group_id) > 160:
        raise ValueError("群 ID 不能超过 160 个字符")
    if any(ord(character) < 32 for character in group_id) or any(
        character in "/\\" for character in group_id
    ):
        raise ValueError("群 ID 包含不支持的字符")
    return group_id


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    if not rows:
        body = f"<tr><td colspan='{len(headers)}' class='empty'>暂无数据</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_dashboard(data: dict, *, path: str) -> str:
    total = data["total"]
    agent = data["agent"]
    cache_rate = _fmt_ratio(total["cached_tokens"], total["prompt_tokens"])
    llm_per_run = _fmt_average(agent["llm_calls"], agent["runs"])
    tools_per_run = _fmt_average(agent["tool_calls"], agent["runs"])
    agent_avg_duration = _fmt_ms(agent["avg_duration_ms"]) if agent["runs"] else "—"
    agent_health = (
        '<span class="status ok">工具无超时</span>'
        if not agent["tool_timeouts"]
        else '<span class="status warn">存在工具超时</span>'
    )
    agent_timeout_summary = _tool_timeout_summary(
        agent.get("tool_timeout_tools", {})
    )
    agent_timeout_hint = agent_timeout_summary or "超时会标记为需关注"

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
        <button type="submit">更新数据</button>
        <div class="links">JSON：<a href="{escape(path)}/api?days={int(data["days"])}">{escape(path)}/api</a> · <a href="{escape(path)}/settings">配置中心</a></div>
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
          <div class="metric"><span class="metric-label">工具超时</span><strong>{_fmt_int(agent["tool_timeouts"])}</strong><span class="metric-hint">{agent_timeout_hint}</span></div>
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
      const storageKey = `ai-groupmate:usage-tab:${{location.pathname}}`;
      const readStoredTab = () => {{
        try {{ return localStorage.getItem(storageKey) || ""; }} catch {{ return ""; }}
      }};
      const storeTab = (name) => {{
        try {{ localStorage.setItem(storageKey, name); }} catch {{}}
      }};
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
        storeTab(name);
        if (updateHash) history.replaceState(null, "", `#${{name}}`);
      }};
      buttons.forEach((button) => button.addEventListener("click", () => activate(button.dataset.tab)));
      const filterForm = document.querySelector(".filter-panel");
      filterForm?.addEventListener("submit", () => {{
        const activeButton = document.querySelector("[data-tab].active");
        if (activeButton?.dataset.tab) {{
          storeTab(activeButton.dataset.tab);
          filterForm.action = `${{location.pathname}}#${{activeButton.dataset.tab}}`;
        }}
      }});
      activate(location.hash.slice(1) || readStoredTab() || "groups", false);
    }})();
  </script>
</body>
</html>"""


def register_usage_webui(
    config: ScopedConfig,
    *,
    on_config_change: Callable[[set[str]], None] | None = None,
) -> None:
    if not config.usage_webui_enabled:
        return

    driver = get_driver()
    app = getattr(driver, "server_app", None)
    if app is None:
        logger.warning("Token 用量 WebUI 需要 FastAPI driver，当前 driver 不支持 server_app，已跳过注册")
        return

    try:
        from fastapi import Query, Request, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
    except Exception as e:
        logger.warning(f"Token 用量 WebUI 依赖 FastAPI，导入失败，已跳过注册: {e}")
        return

    path = "/" + config.usage_webui_path.strip("/")
    api_path = f"{path}/api"
    settings_path = f"{path}/settings"
    settings_api_path = f"{settings_path}/api"
    group_models_path = f"{settings_path}/groups"
    group_models_api_path = f"{group_models_path}/api"

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
    ):
        data = await _load_data(days, session_id, user_id)
        return HTMLResponse(_render_dashboard(data, path=path))

    @app.get(api_path, response_class=JSONResponse, include_in_schema=False)
    async def usage_api(
        days: int = Query(7, ge=1, le=3650),
        session_id: str = "",
        user_id: str = "",
    ):
        return JSONResponse(await _load_data(days, session_id, user_id))

    @app.get(settings_path, response_class=HTMLResponse, include_in_schema=False)
    async def settings_page(request: Request):
        if not _settings_token_ok(config, request):
            return HTMLResponse(
                render_settings_login(
                    settings_path,
                    auth_configured=bool(config.usage_webui_token),
                )
            )
        runtime_config = get_runtime_config()
        return HTMLResponse(
            render_settings_page(
                runtime_config,
                get_environment_config(),
                overridden_fields=set(get_config_overrides()),
                pending_restart_fields=get_pending_restart_fields(),
                dashboard_path=path,
                settings_path=settings_path,
                group_models_path=group_models_path,
            )
        )

    @app.get(
        group_models_path,
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def group_models_page(request: Request):
        if not _settings_token_ok(config, request):
            return HTMLResponse(
                render_settings_login(
                    settings_path,
                    auth_configured=bool(config.usage_webui_token),
                ),
                headers={"Cache-Control": "no-store"},
            )
        return HTMLResponse(
            render_group_api_settings_page(
                dashboard_path=path,
                settings_path=settings_path,
                group_models_path=group_models_path,
            ),
            headers={"Cache-Control": "no-store"},
        )

    @app.get(
        group_models_api_path,
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def list_group_models(request: Request):
        if not _settings_token_ok(config, request):
            raise HTTPException(status_code=401, detail="invalid token")
        async with get_session() as db_session:
            groups = await list_group_model_configs(db_session)
        return JSONResponse(
            {"groups": [group.model_dump(mode="json") for group in groups]},
            headers={"Cache-Control": "no-store"},
        )

    @app.post(
        group_models_api_path,
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def save_group_model(request: Request):
        if not _settings_token_ok(config, request):
            raise HTTPException(status_code=401, detail="invalid token")
        raw_payload = await request.json()
        if not isinstance(raw_payload, dict):
            raise HTTPException(status_code=400, detail="invalid group model payload")
        try:
            group_id = _validate_group_id(raw_payload.get("group_id"))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        runtime_config = get_runtime_config()
        if not runtime_config.group_api_local_encryption_key:
            raise HTTPException(
                status_code=503,
                detail="群 API 本地加密密钥尚未初始化，请检查插件数据目录后重启 Bot",
            )
        try:
            cipher = LocalSecretCipher(
                runtime_config.group_api_local_encryption_key
            )
        except LocalEncryptionKeyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        submitted_api_key = str(raw_payload.get("api_key", "")).strip()
        try:
            async with get_session() as db_session:
                if not submitted_api_key:
                    existing = await get_decrypted_group_model_config(
                        db_session,
                        group_id,
                        cipher,
                    )
                    if existing is None:
                        raise ValueError("新建群配置时必须填写 API Key")
                    submitted_api_key = existing.api_key

                candidate = GroupModelPayload(
                    ticket_id="webui-admin",
                    api_format=raw_payload.get("api_format", "openai"),
                    base_url=raw_payload.get("base_url", ""),
                    api_key=submitted_api_key,
                    chat_model=raw_payload.get("chat_model", ""),
                    chat_multimodal=raw_payload.get("chat_multimodal", True),
                    reply_probability=raw_payload.get("reply_probability"),
                    allow_global_fallback=False,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                )
                await _test_group_model_connection(candidate, runtime_config)
                active = await save_group_model_config(
                    db_session,
                    group_id=group_id,
                    operator_id="webui-admin",
                    payload=candidate,
                    cipher=cipher,
                )
                detail = await get_group_model_config_detail(
                    db_session,
                    active.group_id,
                )
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=_validation_detail(error),
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except LocalEncryptionKeyError as error:
            raise HTTPException(
                status_code=503,
                detail="无法解密已有群配置，请检查本地加密密钥",
            ) from error
        except GroupModelConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except asyncio.TimeoutError as error:
            raise HTTPException(status_code=504, detail="模型连接测试超时") from error
        except Exception as error:
            safe_error = _safe_connection_error(
                error,
                runtime_config,
                extra_secrets=(submitted_api_key,),
            )
            logger.warning(
                f"群模型连接测试失败 group={group_id}: {safe_error}"
            )
            raise HTTPException(
                status_code=502,
                detail=f"模型连接失败：{safe_error}",
            ) from error
        _clear_group_chat_model_cache(group_id)
        if detail is None:
            raise HTTPException(status_code=500, detail="群配置保存后无法读取")
        return JSONResponse(
            {"ok": True, "group": detail.model_dump(mode="json")},
            headers={"Cache-Control": "no-store"},
        )

    @app.delete(
        f"{group_models_api_path}/{{group_id}}",
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def remove_group_model(group_id: str, request: Request):
        if not _settings_token_ok(config, request):
            raise HTTPException(status_code=401, detail="invalid token")
        try:
            normalized_group_id = _validate_group_id(group_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        async with get_session() as db_session:
            deleted = await delete_group_model_config(
                db_session,
                normalized_group_id,
            )
        _clear_group_chat_model_cache(normalized_group_id)
        return JSONResponse(
            {"ok": True, "deleted": deleted},
            headers={"Cache-Control": "no-store"},
        )

    @app.post(
        f"{settings_path}/login",
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def settings_login(request: Request):
        expected = config.usage_webui_token
        if not expected:
            raise HTTPException(status_code=503, detail="settings auth is not configured")
        payload = await request.json()
        supplied = str(payload.get("token", "")) if isinstance(payload, dict) else ""
        if not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid token")
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SETTINGS_COOKIE_NAME,
            expected,
            max_age=SETTINGS_COOKIE_MAX_AGE,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path=path,
        )
        return response

    @app.post(
        f"{settings_path}/logout",
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def settings_logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie(SETTINGS_COOKIE_NAME, path=path)
        return response

    @app.post(
        settings_api_path,
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def save_settings(request: Request):
        if not _settings_token_ok(config, request):
            raise HTTPException(status_code=401, detail="invalid token")
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("updates"), dict):
            raise HTTPException(status_code=400, detail="invalid settings payload")
        clear_secrets_value = payload.get("clear_secrets", [])
        if not isinstance(clear_secrets_value, list) or not all(
            isinstance(item, str) for item in clear_secrets_value
        ):
            raise HTTPException(status_code=400, detail="invalid clear_secrets")
        try:
            async with get_session() as db_session:
                changed_fields, restart_fields = await save_runtime_config_updates(
                    db_session,
                    payload["updates"],
                    clear_secrets=set(clear_secrets_value),
                )
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=_validation_detail(error),
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if on_config_change is not None:
            on_config_change(changed_fields)
        return JSONResponse({
            "ok": True,
            "changed_fields": sorted(changed_fields),
            "restart_fields": sorted(restart_fields),
            "restart_required": bool(restart_fields),
        })

    @app.post(
        f"{settings_path}/test",
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def test_settings_connection(request: Request):
        if not _settings_token_ok(config, request):
            raise HTTPException(status_code=401, detail="invalid token")
        payload = await request.json()
        role = str(payload.get("role", "")) if isinstance(payload, dict) else ""
        try:
            await _test_model_connection(role, get_runtime_config())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except asyncio.TimeoutError as error:
            raise HTTPException(status_code=504, detail="模型连接测试超时") from error
        except Exception as error:
            safe_error = _safe_connection_error(error, get_runtime_config())
            logger.warning(f"模型连接测试失败 role={role}: {safe_error}")
            raise HTTPException(
                status_code=502,
                detail=f"模型连接失败：{safe_error}",
            ) from error
        return JSONResponse({"ok": True, "role": role})

    @app.post(
        f"{settings_path}/reset",
        response_class=JSONResponse,
        include_in_schema=False,
    )
    async def reset_settings(request: Request):
        if not _settings_token_ok(config, request):
            raise HTTPException(status_code=401, detail="invalid token")
        async with get_session() as db_session:
            changed_fields, restart_fields = await reset_runtime_config_overrides(
                db_session
            )
        if on_config_change is not None:
            on_config_change(changed_fields)
        return JSONResponse({
            "ok": True,
            "changed_fields": sorted(changed_fields),
            "restart_fields": sorted(restart_fields),
            "restart_required": bool(restart_fields),
        })

    logger.info(f"Token 用量与配置 WebUI 已注册: {path}")
