"""Small offline contracts for train-only statistics and historical windows."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.constants import FEATURE_GROUPS
from sst_preprocessing.dataset import PreparedSSTDataset
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.indexing import build_sample_index
from sst_preprocessing.statistics import (
    StreamingMoments, compute_percentile_inplace, denormalize_values, normalize_values,
)


def test_streaming_moments_equal_full_calculation():
    values = np.arange(84, dtype=np.float64).reshape(7, 3, 4) + 1e6
    values[1, 1, 1] = np.nan
    mask = np.ones((3, 4), dtype=bool)
    mask[0, 0] = False
    moments = StreamingMoments()
    for chunk in np.array_split(values, 3):
        moments.update(chunk, mask)
    selected = values[np.isfinite(values) & mask]
    result = moments.result()
    assert result["count"] == selected.size
    assert result["mean"] == pytest.approx(selected.mean())
    assert result["std"] == pytest.approx(selected.std())
    assert result["min"] == selected.min()
    assert result["max"] == selected.max()


def test_empty_statistics_are_an_explicit_failure():
    moments = StreamingMoments()
    moments.update(np.array([np.nan, np.inf]))
    with pytest.raises(DataValidationError, match="No finite training"):
        moments.result()


def test_normalization_roundtrip_and_invalid_cells():
    values = np.array([[4.0, 10.0], [np.nan, 6.0]])
    mask = np.array([[True, False], [True, True]])
    normalized = normalize_values(values, {"mean": 5.0, "std": 2.0}, mask)
    restored = denormalize_values(normalized, 5.0, 2.0)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[-0.5, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(restored[0, 0], values[0, 0])
    constant = normalize_values(np.array([5.0]), {"mean": 5.0, "std": 0.0}, np.array([True]))
    np.testing.assert_array_equal(denormalize_values(constant, 5.0, 0.0), [5.0])


def test_nearly_constant_float32_gradients_remain_centered():
    lower = np.float32(0.01)
    upper = np.nextafter(lower, np.float32(np.inf))
    values = np.array([lower] * 7 + [upper] * 3, dtype=np.float32)
    moments = StreamingMoments()
    moments.update(values)
    statistics = moments.result()
    normalized = normalize_values(values, statistics, np.ones(values.shape, dtype=bool))
    assert abs(normalized.astype(np.float64).mean()) < 1e-6
    assert normalized.astype(np.float64).std() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_array_equal(
        denormalize_values(normalized, statistics["mean"], statistics["std"]), values)


def test_statistics_ignore_later_splits():
    dates = pd.date_range("2019-01-01", "2021-12-31")
    values = np.where(dates.year <= 2020, 2.0, 1e10)
    moments = StreamingMoments()
    moments.update(values[dates.year <= 2020])
    assert moments.result()["mean"] == 2.0


def test_percentile_memmap_matches_numpy(tmp_path):
    values = np.random.RandomState(8).normal(size=103).astype(np.float32)
    path = tmp_path / "front.dat"
    values.tofile(str(path))
    assert compute_percentile_inplace(path, len(values)) == pytest.approx(np.percentile(values, 90))
    path.rename(path.with_name("partition-released.dat"))


def test_default_counts_and_no_split_leakage(tmp_path):
    config = PreprocessConfig(data_root=tmp_path / "raw", output_root=tmp_path / "processed")
    dates = pd.date_range(config.start_date, config.end_date)
    index = build_sample_index(dates, config)
    assert index.groupby("split").size().to_dict() == {"train": 3636, "validation": 713, "test": 714}
    assert "2012-02-29" in set(index.origin_date)
    assert "2011-12-31" in set(index.origin_date)
    for split, (start, end) in config.splits.items():
        rows = index.loc[index.split == split]
        assert dates[rows.environment_start_index.min()] >= pd.Timestamp(start)
        assert dates[rows.target_3_index.max()] <= pd.Timestamp(end)


def test_missing_day_excludes_every_window_that_spans_it(tmp_path):
    config = PreprocessConfig(data_root=tmp_path / "raw", output_root=tmp_path / "processed")
    dates = pd.date_range("2011-01-01", "2011-02-28")
    missing_date = pd.Timestamp("2011-01-28")
    index = build_sample_index(dates[dates != missing_date], config)
    for row in index.itertuples():
        support_start = pd.Timestamp(row.origin_date) - pd.Timedelta(days=14)
        support_end = pd.Timestamp(row.target_3_date)
        assert not support_start <= missing_date <= support_end


def test_bad_dates_fail_instead_of_reordering(tmp_path):
    config = PreprocessConfig(data_root=tmp_path / "raw", output_root=tmp_path / "processed")
    with pytest.raises(DataValidationError, match="unique"):
        build_sample_index(["2011-01-01", "2011-01-01"], config)


def _write_prepared_fixture(output_root: Path):
    normalized_root = output_root / "normalized"
    normalized_root.mkdir()
    dates = pd.date_range("2011-01-01", periods=22).to_numpy()
    np.save(output_root / "dates.npy", dates)
    mask = np.array([[True, False], [True, True]])
    np.savez(output_root / "grid_masks.npz", ocean_mask=mask, gradient_mask=mask)
    statistics = {"run_identity": "offline-fixture", "variables": {name: {"mean": 5.0, "std": 2.0}
                                for group in FEATURE_GROUPS.values() for name in group}}
    for group, variables in FEATURE_GROUPS.items():
        values = np.empty((len(dates), len(variables), 2, 2), dtype=np.float32)
        for day in range(len(dates)):
            values[day] = day
        np.save(normalized_root / (group + ".npy"), values)
    (output_root / "stats.json").write_text(json.dumps(statistics), encoding="utf-8")
    ready = {"status": "complete", "run_identity": "offline-fixture"}
    (output_root / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    config = PreprocessConfig(data_root=output_root / "raw", output_root=output_root)
    build_sample_index(dates, config).to_csv(output_root / "sample_index.csv", index=False)


def test_dataset_clock_alignment_targets_and_original_heat_prior(tmp_path):
    _write_prepared_fixture(tmp_path)
    dataset = PreparedSSTDataset(tmp_path)
    sample = dataset[0]
    assert len(dataset) == 5
    np.testing.assert_array_equal(sample["base"][:, 0, 0, 0], np.arange(1, 15))
    np.testing.assert_array_equal(sample["base"][:, 1, 0, 0], np.arange(14))
    np.testing.assert_array_equal(sample["front"][:, 0, 0, 0], np.arange(1, 15))
    np.testing.assert_array_equal(sample["thermal"][:, 0, 0, 0], np.arange(14))
    np.testing.assert_array_equal(sample["target"][:, 0, 0, 0], [15, 16, 17])
    np.testing.assert_array_equal(sample["heat_increment"][:, 0, 0, 0], [31, 62, 93])
    assert sample["last_sst"].shape == (1, 2, 2)
    assert sample["last_sst"][0, 0, 0] == 33
    for group in ("base", "front", "thermal", "target", "last_sst", "heat_increment"):
        assert np.all(sample[group][..., 0, 1] == 0)
    sample["base"][:] = -999
    assert dataset[0]["base"][0, 0, 0, 0] == 1


def test_dataset_requires_completion_marker(tmp_path):
    with pytest.raises(DataValidationError, match="READY.json"):
        PreparedSSTDataset(tmp_path)


@pytest.mark.parametrize("status", [None, "partial", "verified", "failed"])
def test_dataset_rejects_noncomplete_ready_status(tmp_path, status):
    _write_prepared_fixture(tmp_path)
    ready = {"run_identity": "offline-fixture", "status": status}
    (tmp_path / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(DataValidationError, match="status must be complete"):
        PreparedSSTDataset(tmp_path)


@pytest.mark.parametrize("identity", [None, "", "  ", 7])
def test_dataset_rejects_empty_ready_identity(tmp_path, identity):
    _write_prepared_fixture(tmp_path)
    ready = {"run_identity": identity, "status": "complete"}
    (tmp_path / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(DataValidationError, match="nonempty run_identity"):
        PreparedSSTDataset(tmp_path)


@pytest.mark.parametrize("filename", ["stats.json", "run.json"])
def test_dataset_rejects_inconsistent_artifact_identity(tmp_path, filename):
    _write_prepared_fixture(tmp_path)
    path = tmp_path / filename
    contents = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    contents["run_identity"] = "different-run"
    path.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(DataValidationError, match="inconsistent run_identity"):
        PreparedSSTDataset(tmp_path)


def test_dataset_accepts_consistent_optional_run_identity(tmp_path):
    _write_prepared_fixture(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"run_identity": "offline-fixture"}), encoding="utf-8")
    with PreparedSSTDataset(tmp_path) as dataset:
        assert len(dataset) == 5


def test_dataset_reopens_memmaps_when_sent_to_spawned_workers(tmp_path):
    _write_prepared_fixture(tmp_path)
    dataset = PreparedSSTDataset(tmp_path)
    serialized = pickle.dumps(dataset)
    restored = pickle.loads(serialized)
    assert isinstance(restored.arrays["base"], np.memmap)
    assert restored.arrays["base"].mode == "r"
    np.testing.assert_array_equal(restored[0]["base"], dataset[0]["base"])
    restored.close()
    dataset.close()


def test_dataset_context_manager_releases_windows_file_handles(tmp_path):
    _write_prepared_fixture(tmp_path)
    with PreparedSSTDataset(tmp_path) as dataset:
        mapping = dataset.arrays["base"]._mmap
        assert not mapping.closed
    assert mapping.closed
    with pytest.raises(DataValidationError, match="closed"):
        dataset[0]
    source = tmp_path / "normalized" / "base.npy"
    source.rename(source.with_name("base-released.npy"))


def test_dataset_releases_partial_maps_after_validation_error(tmp_path, monkeypatch):
    _write_prepared_fixture(tmp_path)
    path = tmp_path / "normalized" / "front.npy"
    np.save(path, np.zeros((1,), dtype=np.float32))
    original_load = np.load
    opened_maps = []

    def track_load(filename, **options):
        loaded = original_load(filename, **options)
        if isinstance(loaded, np.memmap):
            opened_maps.append(loaded)
        return loaded

    monkeypatch.setattr(np, "load", track_load)
    with pytest.raises(DataValidationError, match="front array shape"):
        PreparedSSTDataset(tmp_path)
    assert len(opened_maps) == 3
    assert all(array._mmap.closed for array in opened_maps)
