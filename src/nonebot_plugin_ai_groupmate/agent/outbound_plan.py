"""Ordered message plans and their native transport representation.

Plans carry opaque resource IDs, never paths or URLs. Resolution and permission
checks must finish before constructing or sending a native message.
"""

from typing import Literal

from pydantic import Field, BaseModel, ConfigDict, model_validator
from nonebot_plugin_alconna import UniMessage


class ReplyPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image", "mention"]
    text: str | None = None
    resource_id: str | None = None
    user_id: str | None = None

    @model_validator(mode="after")
    def valid_payload(self):
        expected = {"text": "text", "image": "resource_id", "mention": "user_id"}[self.type]
        for field in ("text", "resource_id", "user_id"):
            value = getattr(self, field)
            if field == expected:
                if not isinstance(value, str) or not value or (field != "text" and not value.strip()):
                    raise ValueError(f"{field} is required")
                if "\x00" in value:
                    raise ValueError(f"{field} contains an invalid character")
            elif value is not None:
                raise ValueError(f"{field} is not valid for {self.type}")
        return self


class ReplyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[ReplyPart] = Field(min_length=1, max_length=20)
    reply_to: str | None = None

    @model_validator(mode="after")
    def bounded(self):
        if not any(part.type != "text" or (part.text or "").strip() for part in self.parts):
            raise ValueError("message is empty")
        if sum(part.type == "image" for part in self.parts) > 10:
            raise ValueError("at most 10 images per message")
        if sum(len(part.text or "") for part in self.parts) > 12000:
            raise ValueError("message text too long")
        return self


def compose_segments(segments: list[UniMessage], quoted_id: str | None = None) -> UniMessage:
    """Combine already-resolved native segments in their original order."""
    result = UniMessage()
    if quoted_id is not None:
        result += UniMessage.reply(quoted_id)
    for segment in segments:
        result += segment
    return result


def compose_reply(plan: ReplyPlan, images: dict[str, bytes], quoted_id: str | None = None) -> UniMessage:
    """Build a single ordered message after the caller resolves every resource."""
    segments = []
    for part in plan.parts:
        if part.type == "text":
            assert part.text is not None
            segments.append(UniMessage.text(part.text))
        elif part.type == "mention":
            assert part.user_id is not None
            segments.append(UniMessage.at(part.user_id))
        else:
            assert part.resource_id is not None
            data = images.get(part.resource_id)
            if not data:
                raise ValueError("image resource unavailable")
            segments.append(UniMessage.image(raw=data))
    return compose_segments(segments, quoted_id)
