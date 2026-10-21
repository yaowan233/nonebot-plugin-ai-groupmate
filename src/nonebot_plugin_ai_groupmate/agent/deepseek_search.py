"""DeepSeek's standalone Search API, adapted to the existing search tool."""

import re
import datetime
from typing import Any

import httpx

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
MAX_QUERY_CHARS = 128


class SearchFilterError(ValueError):
    pass


def build_search_query(search_input: dict[str, Any]) -> str:
    query = search_input["query"]
    domains = search_input.get("include_domains", [])
    for domain in domains:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", domain):
            raise SearchFilterError("来源限制只接受域名，例如 example.com；不要填写链接或搜索运算符。")
    if domains:
        query += " " + " OR ".join(f"site:{domain}" for domain in domains)
    start_date = search_input.get("start_date")
    if time_range := search_input.get("time_range"):
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[time_range]
        relative_start = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        start_date = max(start_date, relative_start) if start_date else relative_start
    end_date = search_input.get("end_date")
    if start_date and end_date and start_date > end_date:
        raise SearchFilterError("起始日期不能晚于结束日期，请调整时间范围。")
    if start_date:
        query += f" after:{start_date}"
    if end_date:
        query += f" before:{end_date}"
    if len(query) > MAX_QUERY_CHARS:
        raise SearchFilterError("DeepSeek 搜索关键词连同域名、日期条件最多 128 个字符，请缩短关键词或减少筛选条件。")
    return query


class DeepSeekSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def ainvoke(self, search_input: dict[str, Any]) -> dict[str, Any]:
        query = build_search_query(search_input)
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.post(
                SEARCH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "count": 3, "need_content": False},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError("Invalid DeepSeek search response")
        results = []
        for row in payload["results"][:3]:
            if not isinstance(row, dict):
                continue
            results.append({
                "title": row.get("title"),
                "url": row.get("url"),
                "content": row.get("summary") or row.get("snippet") or row.get("content"),
                "published_date": row.get("date"),
                "score": row.get("rerank_score"),
            })
        return {"results": results}
