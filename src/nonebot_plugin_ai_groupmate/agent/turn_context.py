"""Keep a queued turn attached to its actual speaker and transport message."""

import json
from collections.abc import Sequence

from ..model import ChatHistorySchema
from ..message_metadata import platform_message_ids, parse_message_metadata, is_reliable_platform_id


def bind_turn_history(
    history: Sequence[ChatHistorySchema], user_id: str,
    message_id: str | None = None,
) -> tuple[list[ChatHistorySchema], ChatHistorySchema | None]:
    """Later traffic is context for its own turn, not a replacement request."""
    index = next((i for i in range(len(history) - 1, -1, -1)
                  if history[i].content_type != "bot" and history[i].user_id == user_id
                  and (not is_reliable_platform_id(message_id) or str(message_id) in platform_message_ids(history[i].content))), None)
    if index is None:
        return list(history), None
    end = index + 1
    own_ids = platform_message_ids(history[index].content)
    while end < len(history) and history[end].user_id == user_id and own_ids and own_ids & platform_message_ids(history[end].content):
        end += 1
    return list(history[:end]), history[index]


def build_turn_context(
    history: Sequence[ChatHistorySchema], *, user_id: str, user_name: str | None,
    current_text: str, message_id: str | None = None,
    continuous: bool = False, observation: bool = False,
) -> str:
    """Explicit roles distinguish our Bot from every other conversation party."""
    bound, current = bind_turn_history(history, user_id, message_id)
    if current is not None and not current_text:
        current_text = parse_message_metadata(current.content)[2] if current.content_type == "text" else "[图片或其他媒体]"
    participants = {row.user_id: row.user_name for row in bound[-12:] if row.content_type != "bot"}
    if continuous and current is not None:
        current_index = bound.index(current)
        previous = next((i for i in range(current_index - 1, -1, -1) if bound[i].content_type != "bot"), None)
        if previous is not None and previous < current_index - 1 and bound[previous].user_id == user_id:
            # Focus an uninterrupted exchange, without assuming the new turn is
            # addressed to us. Keep all parts of the preceding transport message.
            own_ids = platform_message_ids(bound[previous].content)
            while previous > 0 and bound[previous - 1].user_id == user_id and own_ids and own_ids & platform_message_ids(bound[previous - 1].content):
                previous -= 1
            bound = bound[previous:]
    turns = []
    for row in bound[-12:]:
        own_id, reply_id, body = parse_message_metadata(row.content)
        turns.append({"message_id": own_id, "speaker_id": row.user_id, "speaker_name": row.user_name,
                      "role": "self_bot" if row.content_type == "bot" else "other_participant",
                      "reply_to": reply_id, "text": body[:600] if row.content_type != "image" else "[图片]"})
    return json.dumps({
        "candidate_reason": "observation" if observation else "recent_conversation" if continuous else "ambient",
        "participants": participants,
        "recent_messages": turns,
        "latest_message": {"message_id": message_id, "speaker_id": user_id,
                           "speaker_name": user_name or (current.user_name if current else user_id), "text": current_text[:1600]},
    }, ensure_ascii=False)
