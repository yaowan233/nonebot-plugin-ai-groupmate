"""Attach private group reflection to normal Agent completion, without a new turn."""

from .graph import TerminalMetadata
from ..model import ChatHistorySchema
from ..reply_guard import is_request_active
from ..group_affinity import prepare_affinity_review, complete_affinity_review


async def prepare_main_group_review(
    bot_id: str, session_id: str, history: list[ChatHistorySchema], request_id: str | None,
) -> tuple[TerminalMetadata | None, str]:
    review = await prepare_affinity_review((bot_id, session_id), history)
    if review is None:
        return None, ""
    completed = False

    async def on_complete(value: object) -> None:
        nonlocal completed
        if completed or (request_id is not None and not await is_request_active(session_id, request_id)):
            return
        completed = True
        await complete_affinity_review(review, value)

    return TerminalMetadata("group_feeling", review.schema(), on_complete), review.prompt()
