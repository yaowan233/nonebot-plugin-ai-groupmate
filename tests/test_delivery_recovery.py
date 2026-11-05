from uuid import uuid4
from datetime import datetime, timedelta
from importlib import import_module

from sqlalchemy import text, inspect, create_engine
from alembic.migration import MigrationContext
from alembic.operations import Operations


async def test_pruning_keeps_receipts_with_pending_journal(tmp_path):
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import DeliveryRecoveryReceipt
    from nonebot_plugin_ai_groupmate.delivery_recovery import HistoryEntry, RecoveryRecord, prune_receipts, write_recovery

    now = datetime.now()
    retained, removable = uuid4().hex, uuid4().hex
    record = RecoveryRecord(receipt_id=retained, entries=[HistoryEntry(session_id="g", user_id="b", user_name="b",
                                                                      content="id: 1\nhello", created_at=now)])
    write_recovery(tmp_path, record)
    async with get_session() as db:
        db.add_all([DeliveryRecoveryReceipt(receipt_id=key, created_at=now - timedelta(days=31)) for key in (retained, removable)])
        await db.commit()
    assert await prune_receipts(tmp_path, now=now) == 1
    async with get_session() as db:
        assert await db.get(DeliveryRecoveryReceipt, retained) is not None
        assert await db.get(DeliveryRecoveryReceipt, removable) is None


def test_recovery_migration_keeps_existing_tables():
    migration = import_module("nonebot_plugin_ai_groupmate.migrations.b5e9c2a6d741_add_delivery_recovery")
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE old_messages (id INTEGER PRIMARY KEY)"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert "nonebot_plugin_ai_groupmate_deliveryrecoveryreceipt" in inspect(connection).get_table_names()
            migration.downgrade()
        assert "old_messages" in inspect(connection).get_table_names()
