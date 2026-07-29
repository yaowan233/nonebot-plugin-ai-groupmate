import json
import datetime
from types import SimpleNamespace

import pytest


class _Result:
    def __init__(self, relations):
        self.relations = relations

    def scalars(self):
        return self

    def all(self):
        return self.relations


class _Session:
    def __init__(self, relations):
        self.relations = relations
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, _statement):
        return _Result(self.relations)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


@pytest.mark.asyncio
async def test_reset_negative_relations_backs_up_before_soft_reset(tmp_path):
    from nonebot_plugin_ai_groupmate.relation_maintenance import (
        reset_negative_relations,
    )

    relation = SimpleNamespace(
        id=7,
        user_id="10001",
        user_name="tester",
        favorability=-80,
        tags=["讨厌", "曾经争吵"],
        updated_at=datetime.datetime(2026, 7, 28, 12, 30),
    )
    session = _Session([relation])
    reset_time = datetime.datetime(
        2026, 7, 29, 12, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8))
    )

    result = await reset_negative_relations(
        session, tmp_path / "backups", now=reset_time
    )

    assert result.affected_count == 1
    assert result.backup_path is not None
    backup = json.loads(result.backup_path.read_text(encoding="utf-8"))
    assert backup["schema_version"] == 1
    assert backup["affected_count"] == 1
    assert backup["relations"] == [
        {
            "id": 7,
            "user_id": "10001",
            "user_name": "tester",
            "favorability": -80,
            "tags": ["讨厌", "曾经争吵"],
            "updated_at": "2026-07-28T12:30:00",
        }
    ]
    assert relation.favorability == 0
    assert relation.tags == []
    assert session.commits == 1
    assert session.rollbacks == 0


@pytest.mark.asyncio
async def test_reset_negative_relations_is_noop_when_nothing_matches(tmp_path):
    from nonebot_plugin_ai_groupmate.relation_maintenance import (
        reset_negative_relations,
    )

    session = _Session([])

    result = await reset_negative_relations(session, tmp_path / "backups")

    assert result.affected_count == 0
    assert result.backup_path is None
    assert not (tmp_path / "backups").exists()
    assert session.commits == 0
    assert session.rollbacks == 1


@pytest.mark.asyncio
async def test_backup_failure_never_changes_relation_data(monkeypatch, tmp_path):
    from nonebot_plugin_ai_groupmate import relation_maintenance

    relation = SimpleNamespace(
        id=8,
        user_id="10002",
        user_name="tester2",
        favorability=-30,
        tags=["旧标签"],
        updated_at=None,
    )
    session = _Session([relation])

    def fail_to_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        relation_maintenance, "_write_json_atomically", fail_to_write
    )

    with pytest.raises(OSError, match="disk full"):
        await relation_maintenance.reset_negative_relations(
            session, tmp_path / "backups"
        )

    assert relation.favorability == -30
    assert relation.tags == ["旧标签"]
    assert session.commits == 0
    assert session.rollbacks == 1
