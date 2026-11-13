from datetime import datetime, timedelta

import pytest


def test_active_group_needs_real_distinct_interactions_and_new_messages():
    from nonebot_plugin_ai_groupmate.brain.attention import Interaction, ObservationState, evaluate

    now = datetime(2026, 11, 10, 12)
    a = Interaction("turn-a", now - timedelta(minutes=5), True)
    b = Interaction("turn-b", now - timedelta(minutes=2), False)
    state = ObservationState((a, a), now, False, 100, now)
    assert evaluate(state, now=now, budget_available=True) == (False, False)
    state = ObservationState((a, b), b.at, False, 10, now)
    assert evaluate(state, now=now, budget_available=True) == (True, True)
    assert evaluate(state, now=now, budget_available=False) == (True, False)


@pytest.mark.parametrize(("paused", "enabled", "new_messages", "expected"), [
    (True, True, 10, (True, False)),
    (False, False, 10, (False, False)),
    (False, True, 9, (True, False)),
])
def test_observation_respects_pause_opt_out_and_material(paused, enabled, new_messages, expected):
    from nonebot_plugin_ai_groupmate.brain.attention import ObservationState, evaluate

    now = datetime(2026, 11, 10, 12)
    state = ObservationState((), now - timedelta(hours=1), True, new_messages, now, enabled=enabled, paused=paused)
    assert evaluate(state, now=now, budget_available=True) == expected


def test_inactive_group_eventually_leaves_observation():
    from nonebot_plugin_ai_groupmate.brain.attention import ObservationState, evaluate

    now = datetime(2026, 11, 10, 12)
    state = ObservationState((), now - timedelta(hours=2), True, 100, now)
    assert evaluate(state, now=now, budget_available=True) == (False, False)
