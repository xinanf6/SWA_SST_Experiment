"""Small physical-field fixtures for exact, training-only MLD stratification."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst_analysis.context import AnalysisContext
import sst_analysis.stratification as stratification
from sst_analysis.stratification import fit_training_mld_threshold
from sst_preprocessing.errors import ConfigurationError, DataValidationError


RUN_IDENTITY = "synthetic-training-mld"


def _source_fields(dates: pd.DatetimeIndex) -> np.ndarray:
    day = np.asarray((dates - pd.Timestamp("2020-01-01")).days, dtype=np.float32)[:, None, None]
    north, east = np.arange(5)[None, :, None], np.arange(5)[None, None, :]
    fields = (1.0 + 2.0 * day + 0.25 * north + 0.01 * east).astype(np.float32)
    fields[:, 2, 2] = np.nan
    return fields


def _write_month(context: AnalysisContext, month: str) -> None:
    period = pd.Period(month, freq="M")
    dates = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    dataset = xr.Dataset(
        {"mixed_layer_depth": (("time", "latitude", "longitude"), _source_fields(dates), {"units": "m"})},
        coords={"time": dates, "latitude": context.grid["latitude"], "longitude": context.grid["longitude"]},
        attrs={"run_identity": context.run_identity},
    )
    dataset.to_netcdf(context.root / "physical" / (month + ".nc"))
    dataset.close()


@pytest.fixture
def mld_fixture(tmp_path):
    root = tmp_path / "prepared"
    (root / "physical").mkdir(parents=True)
    core_mask, ocean_mask = np.zeros((5, 5), bool), np.ones((5, 5), bool)
    core_mask[1:4, 1:4] = True
    ocean_mask[2, 2] = False
    grid = {"latitude": np.linspace(-48.0, -44.0, 5), "longitude": np.linspace(-60.0, -56.0, 5),
            "core_mask": core_mask, "ocean_mask": ocean_mask, "area_weights": np.ones((5, 5))}
    run = {"run_identity": RUN_IDENTITY, "config": {"splits": {"train": ["2020-01-27", "2020-02-12"]}}}
    context = AnalysisContext(root, run, {}, grid)
    for month in ("2020-01", "2020-02"):
        _write_month(context, month)
    return {"context": context, "output": tmp_path / "stratification"}


def _expected_values(context: AnalysisContext) -> np.ndarray:
    dates = pd.date_range(*context.train_period, freq="D")
    values = _source_fields(dates)[:, context.grid["core_mask"] & context.grid["ocean_mask"]]
    return values[np.isfinite(values) & (values > 0)]


def _rewrite_month(context: AnalysisContext, month: str, transform) -> None:
    path = context.root / "physical" / (month + ".nc")
    with xr.open_dataset(path) as source:
        dataset = source.load()
    changed = transform(dataset)
    changed.to_netcdf(path)
    changed.close()


def _assert_no_scratch(output: Path) -> None:
    assert list(output.glob(".training_mld_values_*.float32.tmp")) == []


def _source_digests(context: AnalysisContext) -> dict:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (context.root / "physical").glob("*.nc")}


def test_training_median_is_exact_unweighted_and_ignores_other_dates(mld_fixture):
    context, output = mld_fixture["context"], mld_fixture["output"]
    context.grid["area_weights"] = np.arange(25).reshape(5, 5) ** 8
    (context.root / "physical" / "2020-03.nc").write_text("Do not open validation data", encoding="utf-8")
    original_sources = _source_digests(context)
    expected = _expected_values(context)
    report = fit_training_mld_threshold(context, output)
    assert report["threshold_m"] == float(np.median(expected.astype(np.float64)))
    assert report["count"] == len(expected) == 136
    assert report["status"] == "complete" and report["run_identity"] == RUN_IDENTITY
    assert report["train_period"] == list(context.train_period)
    assert report["training_day_count"] == 17 and report["chunk_days"] == 7
    assert len(report["source_files"]) == 2
    assert all(row["size_bytes"] > 0 and row["mtime_ns"] > 0 for row in report["source_files"])
    assert json.loads((output / "training_mld_median.json").read_text(encoding="utf-8")) == report
    assert _source_digests(context) == original_sources
    _assert_no_scratch(output)


@pytest.mark.parametrize("day_count", [2, 3])
def test_odd_and_even_counts_use_the_correct_middle_values(mld_fixture, day_count):
    context = mld_fixture["context"]
    context.run["config"]["splits"]["train"] = ["2020-02-01", "2020-02-%02d" % day_count]
    context.grid["core_mask"][:] = False
    context.grid["core_mask"][1, 1] = True
    expected = _expected_values(context)
    report = fit_training_mld_threshold(context, mld_fixture["output"])
    assert report["count"] == day_count
    assert report["threshold_m"] == float(np.median(expected.astype(np.float64)))


def test_matching_complete_cache_avoids_rereading_mld(mld_fixture, monkeypatch):
    context, output = mld_fixture["context"], mld_fixture["output"]
    expected = fit_training_mld_threshold(context, output)

    def unexpected_fit(*arguments):
        raise AssertionError("Matching complete cache should avoid data rereads")

    monkeypatch.setattr(stratification, "_fit_exact", unexpected_fit)
    assert fit_training_mld_threshold(context, output) == expected


@pytest.mark.parametrize("key", ["status", "run_identity", "train_period", "source_files", "method", "grid_signature", "units"])
def test_cache_is_not_reused_when_identity_or_completeness_differs(mld_fixture, key):
    context, output = mld_fixture["context"], mld_fixture["output"]
    expected = fit_training_mld_threshold(context, output)
    changed = dict(expected, threshold_m=999.0)
    changed[key] = "stale"
    (output / "training_mld_median.json").write_text(json.dumps(changed), encoding="utf-8")
    actual = fit_training_mld_threshold(context, output)
    assert actual == expected
    _assert_no_scratch(output)


def test_source_size_or_modification_time_invalidates_cache(mld_fixture):
    context, output = mld_fixture["context"], mld_fixture["output"]
    original = fit_training_mld_threshold(context, output)

    def add_depth(dataset):
        dataset.mixed_layer_depth.values[:] += 100.0
        return dataset

    for month in ("2020-01", "2020-02"):
        _rewrite_month(context, month, add_depth)
        path = context.root / "physical" / (month + ".nc")
        metadata = path.stat()
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1000000))
    current = fit_training_mld_threshold(context, output)
    assert current["source_files"] != original["source_files"]
    assert current["threshold_m"] == pytest.approx(original["threshold_m"] + 100.0, abs=1e-5)


def _corrupt_source(dataset: xr.Dataset, case: str) -> xr.Dataset:
    if case == "units":
        dataset.mixed_layer_depth.attrs["units"] = "cm"
    elif case == "identity":
        dataset.attrs["run_identity"] = "different-run"
    elif case == "grid":
        dataset = dataset.assign_coords(latitude=dataset.latitude.values + 0.01)
    elif case == "dimensions":
        dataset["mixed_layer_depth"] = dataset.mixed_layer_depth.transpose("time", "longitude", "latitude")
    elif case == "missing_day":
        dataset = dataset.isel(time=np.arange(1, len(dataset.time)))
    elif case == "subdaily":
        dataset = dataset.assign_coords(time=dataset.time.values + np.timedelta64(1, "h"))
    elif case == "duplicate_day":
        dates = dataset.time.values.copy()
        dates[1] = dates[0]
        dataset = dataset.assign_coords(time=dates)
    return dataset


@pytest.mark.parametrize("case", ["units", "identity", "grid", "dimensions", "missing_day", "subdaily", "duplicate_day"])
def test_strict_physical_contract_rejects_invalid_sources_and_cleans_scratch(mld_fixture, case):
    context, output = mld_fixture["context"], mld_fixture["output"]
    _rewrite_month(context, "2020-02", lambda dataset: _corrupt_source(dataset, case))
    with pytest.raises(DataValidationError):
        fit_training_mld_threshold(context, output)
    assert not (output / "training_mld_median.json").exists()
    _assert_no_scratch(output)


def test_nonpositive_and_missing_depths_are_excluded_from_grid_day_count(mld_fixture):
    context = mld_fixture["context"]
    context.run["config"]["splits"]["train"] = ["2020-02-01", "2020-02-03"]
    context.grid["core_mask"][:] = False
    context.grid["core_mask"][1, 1:4] = True

    def set_values(dataset):
        dataset.mixed_layer_depth.values[:3, 1, 1:4] = [[0, -5, np.nan], [2, 4, 6], [8, 10, 12]]
        return dataset

    _rewrite_month(context, "2020-02", set_values)
    report = fit_training_mld_threshold(context, mld_fixture["output"])
    assert report["count"] == 6 and report["threshold_m"] == 7.0


def test_no_valid_depths_fail_without_publishing_or_leaving_scratch(mld_fixture):
    context, output = mld_fixture["context"], mld_fixture["output"]

    def set_missing(dataset):
        dataset.mixed_layer_depth.values[:] = np.nan
        return dataset

    for month in ("2020-01", "2020-02"):
        _rewrite_month(context, month, set_missing)
    with pytest.raises(DataValidationError, match="no finite positive"):
        fit_training_mld_threshold(context, output)
    assert not (output / "training_mld_median.json").exists()
    _assert_no_scratch(output)


def test_reads_at_most_seven_days_per_chunk(mld_fixture, monkeypatch):
    original, lengths = xr.DataArray.isel, []

    def record_selection(self, *arguments, **options):
        if "time" in options:
            lengths.append(len(options["time"]))
        return original(self, *arguments, **options)

    monkeypatch.setattr(xr.DataArray, "isel", record_selection)
    fit_training_mld_threshold(mld_fixture["context"], mld_fixture["output"])
    assert lengths and max(lengths) <= 7 and sum(lengths) == 17


def test_output_cannot_overlap_preprocessing_artifacts(mld_fixture):
    context = mld_fixture["context"]
    with pytest.raises(ConfigurationError):
        fit_training_mld_threshold(context, context.root / "stratification")


def test_partition_operates_in_place_on_a_closed_then_removed_file_map(mld_fixture, monkeypatch):
    original_map, calls = np.memmap, []

    class TrackingMap(original_map):
        def partition(self, *arguments, **options):
            calls.append({"size": self.size, "mapping": self._mmap, "path": Path(self.filename)})
            return super().partition(*arguments, **options)

    monkeypatch.setattr(stratification.np, "memmap", TrackingMap)
    fit_training_mld_threshold(mld_fixture["context"], mld_fixture["output"])
    assert len(calls) == 1 and calls[0]["size"] == 136
    assert calls[0]["mapping"].closed and not calls[0]["path"].exists()


def test_source_change_during_fitting_prevents_cache_publication(mld_fixture, monkeypatch):
    original_fit = stratification._fit_exact
    context, output = mld_fixture["context"], mld_fixture["output"]

    def change_source_after_fit(*arguments):
        result = original_fit(*arguments)
        path = context.root / "physical" / "2020-01.nc"
        metadata = path.stat()
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1000000))
        return result

    monkeypatch.setattr(stratification, "_fit_exact", change_source_after_fit)
    with pytest.raises(DataValidationError, match="changed during"):
        fit_training_mld_threshold(context, output)
    assert not (output / "training_mld_median.json").exists()
    _assert_no_scratch(output)
