"""Resolve configured group names using the same bot that owns a task."""

import time
import asyncio
from typing import Any

from ..structured_message import adapter_name


class GroupNameResolver:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, float, frozenset[str], dict[str, str]]] = {}

    async def resolve(self, bot: Any, platform: str, group_ids: tuple[str, ...], manual: dict[str, str]) -> dict[str, str]:
        allowed = frozenset(group_ids)
        overrides = {group: name.strip()[:100] for group, name in manual.items()
                     if group in allowed and isinstance(name, str) and name.strip()}
        needed = allowed - overrides.keys()
        if not needed or bot is None or platform != "OneBot V11" or adapter_name(bot) != platform:
            return overrides

        key = str(getattr(bot, "self_id", id(bot)))
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached is None or cached[0] is not bot or cached[1] <= now or not needed <= cached[2]:
            names: dict[str, str] = {}
            ttl = 30
            try:
                result = await asyncio.wait_for(bot.call_api("get_group_list"), timeout=3)
                if isinstance(result, list):
                    for item in result:
                        if not isinstance(item, dict):
                            continue
                        group_id, name = str(item.get("group_id", "")), item.get("group_name")
                        if group_id in needed and isinstance(name, str) and name.strip():
                            names[group_id] = name.replace("\x00", "").strip()[:100]
                    ttl = 300
            except Exception:
                pass
            cached = (bot, now + ttl, frozenset(needed), names)
            self._cache[key] = cached
        return {**{group: name for group, name in cached[3].items() if group in allowed}, **overrides}
