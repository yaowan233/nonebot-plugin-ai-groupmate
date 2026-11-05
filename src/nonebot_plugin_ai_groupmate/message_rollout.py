"""Decide whether one conversation is in the message-architecture rollout."""

from .config import ScopedConfig


def message_pilot_enabled(config: ScopedConfig, session_id: str, *, is_private: bool) -> bool:
    if is_private:
        return config.message_refactor_pilot_private
    groups = config.message_refactor_pilot_groups
    return "*" in groups or str(session_id) in groups
