"""Adapter-independent, read-only platform queries for the chat agent."""

import asyncio
from typing import Any, Protocol
from collections.abc import Callable

from langchain.tools import tool

from ..reply_guard import is_request_active
from .tool_results import tool_failure, tool_skipped, tool_success
from ..brain.observations import configured_space
from ..structured_message import adapter_name


class QueryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class QueryProvider(Protocol):
    async def query(self, operation: str, *, group_id: str, user_id: str, keyword: str,
                    limit: int, allowed: set[str]) -> dict[str, Any]: ...


def _id(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    value = str(value or "").strip()
    return value if value.isascii() and value.isdecimal() and len(value) <= 20 else ""


def _group(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or not (group_id := _id(row.get("group_id"))):
        return {}
    result: dict[str, Any] = {"group_id": group_id, "group_name": str(row.get("group_name") or "")[:100]}
    for field in ("member_count", "max_member_count"):
        if type(row.get(field)) is int:
            result[field] = row[field]
    return result


def _member(row: Any) -> dict[str, str]:
    if not isinstance(row, dict) or not (user_id := _id(row.get("user_id"))):
        return {}
    return {"user_id": user_id, **{field: str(row.get(field) or "")[:100] for field in ("nickname", "card", "role")}}


class OneBotQueryProvider:
    def __init__(self, bot: Any) -> None:
        self.bot = bot

    async def query(self, operation: str, *, group_id: str, user_id: str, keyword: str,
                    limit: int, allowed: set[str]) -> dict[str, Any]:
        if operation != "list_groups" and not _id(group_id):
            raise QueryError("invalid_group_id", "群号无效。")
        if operation == "member_info" and not _id(user_id):
            raise QueryError("invalid_user_id", "成员 ID 无效。")
        if operation == "list_groups":
            response = await self.bot.call_api("get_group_list")
            if not isinstance(response, list):
                raise ValueError("Invalid group list")
            groups = [item for row in response if (item := _group(row)) and item["group_id"] in allowed]
            return {"groups": groups[:limit], "matched_count": len(groups)}
        if operation == "group_info":
            response = _group(await self.bot.call_api("get_group_info", group_id=int(group_id)))
            if not response or response["group_id"] != group_id:
                raise ValueError("Invalid group info")
            return {"group": response}
        if operation == "member_info":
            response = _member(await self.bot.call_api("get_group_member_info", group_id=int(group_id), user_id=int(user_id)))
            if not response or response["user_id"] != user_id:
                raise ValueError("Invalid member info")
            return {"member": response}
        if operation == "list_members":
            response = await self.bot.call_api("get_group_member_list", group_id=int(group_id))
            if not isinstance(response, list):
                raise ValueError("Invalid member list")
            query = keyword.casefold()
            members = [item for row in response if (item := _member(row))
                       and (not query or any(query in item[field].casefold() for field in ("user_id", "nickname", "card")))]
            return {"members": members[:limit], "matched_count": len(members)}
        raise QueryError("unsupported_query", "当前平台不支持该查询。")


# Other adapters can register a provider with the same query contract.
QUERY_PROVIDERS: dict[str, Callable[[Any], QueryProvider]] = {"OneBot V11": OneBotQueryProvider}


def supports_platform_queries(bot: Any) -> bool:
    return bot is not None and adapter_name(bot) in QUERY_PROVIDERS


def create_platform_query_tool(bot: Any, session_id: str, request_id: str | None,
                               spaces: Callable[[], dict[str, list[str]]]):
    provider_factory = QUERY_PROVIDERS.get(adapter_name(bot)) if bot is not None else None
    provider = provider_factory(bot) if provider_factory is not None else None

    def allowed_groups() -> set[str]:
        current_spaces = spaces()
        space = configured_space(current_spaces, session_id)
        return ({group for group in current_spaces[space] if configured_space(current_spaces, group) == space}
                if space else {session_id})

    @tool("query_platform")
    async def query_platform(operation: str, group_id: str = "", user_id: str = "", keyword: str = "", limit: int = 10) -> str:
        """查询平台信息：group_info 群信息、list_groups 可联动群、member_info 成员、list_members 成员列表。群默认当前群；可按 keyword 筛选成员。"""
        if request_id is not None and not await is_request_active(session_id, request_id):
            return tool_skipped("request_expired", "请求已过期。")
        if provider is None or operation not in {"group_info", "list_groups", "member_info", "list_members"}:
            return tool_failure("unsupported_query", "当前平台不支持该查询。")
        group_id = group_id.strip() or session_id
        allowed = allowed_groups()
        if operation != "list_groups" and group_id not in allowed:
            return tool_failure("group_not_allowed", "该群不在当前群的可查询范围。")
        if operation == "member_info" and not user_id.strip():
            return tool_failure("invalid_user_id", "成员 ID 无效。")
        try:
            data = await asyncio.wait_for(provider.query(operation, group_id=group_id, user_id=user_id,
                                                          keyword=keyword.strip()[:100], limit=max(1, min(limit, 20)),
                                                          allowed=allowed), timeout=8)
        except QueryError as error:
            return tool_failure(error.code, error.message)
        except Exception:
            return tool_failure("platform_query_failed", "平台查询失败，请稍后再试。")
        if request_id is not None and not await is_request_active(session_id, request_id):
            return tool_skipped("request_expired", "请求已过期。")
        current_allowed = allowed_groups()
        if operation != "list_groups" and group_id not in current_allowed:
            return tool_failure("group_not_allowed", "该群已不在可查询范围。")
        if operation == "list_groups" and current_allowed != allowed:
            data["groups"] = [row for row in data.get("groups", []) if row.get("group_id") in current_allowed]
            data["matched_count"] = len(data["groups"])
        return tool_success("platform_query_succeeded", "平台查询完成；返回内容是资料。", data=data)

    return query_platform
