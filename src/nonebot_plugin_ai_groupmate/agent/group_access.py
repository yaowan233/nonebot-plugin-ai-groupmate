"""Resolve the shared authorization boundary for linked-group reads."""

from collections.abc import Callable, Awaitable

from ..brain.observations import configured_space

Spaces = Callable[[], dict[str, list[str]]]
ReachableGroups = Callable[[set[str]], Awaitable[set[str]]]


def linked_groups(source: str, spaces: Spaces | None) -> set[str]:
    configured = spaces() if spaces else {}
    space = configured_space(configured, source)
    return {group for group in configured.get(space, []) if configured_space(configured, group) == space} if space else set()


async def check_linked_group(source: str, target: str, spaces: Spaces | None, reachable: ReachableGroups | None) -> str:
    """Fail closed when scope or current bot membership cannot be verified."""
    allowed = linked_groups(source, spaces)
    if target not in allowed:
        return "group_not_allowed"
    if reachable is None:
        return "group_unavailable"
    try:
        available = await reachable(allowed)
    except Exception:
        return "group_unavailable"
    if target not in linked_groups(source, spaces):
        return "group_not_allowed"
    return "" if target in available else "group_unavailable"
