import os
from pathlib import Path


def test_orm_data_directory_is_isolated_per_test_process():
    import nonebot_plugin_orm

    configured_data_dir = os.environ.get("LOCALSTORE_DATA_DIR")

    assert configured_data_dir is not None
    assert nonebot_plugin_orm._data_dir.parent == Path(configured_data_dir)
    assert str(os.getpid()) in Path(configured_data_dir).parts
