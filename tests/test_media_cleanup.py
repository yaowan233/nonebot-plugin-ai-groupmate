import os
import time
import datetime


def test_orphan_cleanup_runs_full_scan_with_grace_period(tmp_path):
    from nonebot_plugin_ai_groupmate import _delete_orphaned_files

    known = tmp_path / "known.jpg"
    old_orphan = tmp_path / "old-orphan.jpg"
    recent_orphan = tmp_path / "recent-orphan.jpg"
    child_directory = tmp_path / "nested"

    known.write_bytes(b"known")
    old_orphan.write_bytes(b"old")
    recent_orphan.write_bytes(b"recent")
    child_directory.mkdir()

    old_timestamp = time.time() - 3600
    os.utime(known, (old_timestamp, old_timestamp))
    os.utime(old_orphan, (old_timestamp, old_timestamp))

    orphaned, deleted = _delete_orphaned_files(
        tmp_path,
        {known.name},
        datetime.timedelta(minutes=10),
    )

    assert (orphaned, deleted) == (1, 1)
    assert known.exists()
    assert not old_orphan.exists()
    assert recent_orphan.exists()
    assert child_directory.exists()
