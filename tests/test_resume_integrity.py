"""Exact recovery regressions for missing checkpoints and Windows map handles."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sst_preprocessing.constants import FEATURE_GROUPS
from sst_preprocessing.errors import ResumeMismatchError
from sst_preprocessing.normalization import _open_maps


def _map_context(output_root):
    config = SimpleNamespace(start_date="2011-01-01", end_date="2011-02-28")
    return SimpleNamespace(root=output_root, config=config, months=["2011-01", "2011-02"])


def _write_group_map(context, group, *, has_correct_shape=True):
    folder = context.root / "normalized"
    folder.mkdir(exist_ok=True)
    days = len(pd.date_range(context.config.start_date, context.config.end_date))
    shape = (days, len(FEATURE_GROUPS[group]), 2, 2) if has_correct_shape else (1,)
    path = folder / (group + ".npy.part")
    with path.open("wb") as handle:
        np.save(handle, np.full(shape, 7.0, dtype=np.float32), allow_pickle=False)
    return path


def test_checkpointed_missing_group_is_rejected_without_recreating_zeros(tmp_path):
    context = _map_context(tmp_path)
    _write_group_map(context, "front")
    _write_group_map(context, "thermal")
    missing_path = tmp_path / "normalized" / "base.npy.part"
    progress = {"months_done": ["2011-01"]}
    grid = {"ocean_mask": np.ones((2, 2), dtype=bool)}

    with pytest.raises(ResumeMismatchError, match="Checkpointed normalized map is missing"):
        _open_maps(context, grid, progress)

    assert not missing_path.exists()
    assert not (tmp_path / "normalized" / "base.npy").exists()


def test_second_group_failure_closes_all_open_maps_before_windows_rename(tmp_path, monkeypatch):
    context = _map_context(tmp_path)
    first_path = _write_group_map(context, "base")
    failing_path = _write_group_map(context, "front", has_correct_shape=False)
    original_load, opened_maps = np.load, []

    def track_load(path, **options):
        mapped = original_load(path, **options)
        opened_maps.append(mapped)
        return mapped

    monkeypatch.setattr(np, "load", track_load)
    with pytest.raises(ResumeMismatchError, match="unexpected shape/dtype") as failure:
        _open_maps(context, {"ocean_mask": np.ones((2, 2), dtype=bool)},
                   {"months_done": ["2011-01"]})

    assert "front.npy.part" in str(failure.value)
    assert len(opened_maps) == 2
    assert all(mapped._mmap.closed for mapped in opened_maps)
    # Keep the exception traceback alive: garbage collection must not hide leaked handles.
    for path in (first_path, failing_path):
        released_path = path.with_name(path.name + ".released")
        path.rename(released_path)
        assert released_path.is_file()
