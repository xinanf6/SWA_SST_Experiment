"""Offline file-contract tests using synthetic supplied forecasts, not baselines."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst_analysis.config import load_analysis_config, validate_output_location
from sst_analysis.context import load_context
import sst_analysis.__main__ as analysis_cli
import sst_analysis.evaluation as evaluation_module
from sst_analysis.evaluation import evaluate_forecast_file
from sst_analysis.forecasts import ForecastFile
from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.errors import ConfigurationError, DataValidationError
from sst_preprocessing.grid import build_grid_metadata, gradient_valid_mask
from sst_preprocessing.indexing import build_sample_index


RUN_IDENTITY = "synthetic-analysis-io-fixture"
SPLIT_NAMES = ("train", "validation", "test")
TEMPERATURE_MEAN = 10.0
TEMPERATURE_STANDARD_DEVIATION = 2.0
TOLERANCE = 1e-5


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_config(root: Path, dates: pd.DatetimeIndex) -> PreprocessConfig:
    split_bounds = [(dates[index], dates[index + 17]) for index in (0, 18, 36)]
    splits = {name: (str(start.date()), str(end.date()))
              for name, (start, end) in zip(SPLIT_NAMES, split_bounds)}
    return PreprocessConfig(
        data_root=root / "raw", output_root=root / "prepared",
        start_date=str(dates[0].date()), end_date=str(dates[-1].date()), splits=splits,
        core_south=-40.075, core_north=-39.825, core_west=-55.075, core_east=-54.825,
        expected_grid_shape=(7, 7), expected_core_shape=(5, 5),
    )


def _fixture_grid(config: PreprocessConfig) -> dict:
    latitude = np.linspace(-40.1, -39.8, 7)
    longitude = np.linspace(-55.1, -54.8, 7)
    grid = build_grid_metadata(latitude, longitude, config)
    ocean_mask = np.ones((7, 7), dtype=bool)
    ocean_mask[3, 3] = False
    grid.update(ocean_mask=ocean_mask, gradient_mask=gradient_valid_mask(ocean_mask))
    return grid


def _fixture_base() -> np.ndarray:
    base = np.zeros((54, 7, 7, 7), dtype=np.float32)
    day = np.arange(54, dtype=np.float32)[:, None, None]
    north = np.arange(7, dtype=np.float32)[None, :, None]
    east = np.arange(7, dtype=np.float32)[None, None, :]
    base[:, 0] = 0.01 * day + 0.02 * north + 0.03 * east
    base[:, :, 3, 3] = 0.0
    return base


def _fixture_thermal(day_count: int) -> np.ndarray:
    depth = np.full((7, 7), 15.0, np.float32)
    depth[:, :3] = 5.0
    depth[:, 3] = 10.0
    thermal = np.zeros((day_count, 3, 7, 7), np.float32)
    thermal[:, 1] = (depth - 10.0) / 5.0
    return thermal


@pytest.fixture
def prepared(tmp_path):
    dates = pd.date_range("2020-01-01", periods=54, freq="D")
    config = _fixture_config(tmp_path, dates)
    root = config.output_root
    (root / "normalized").mkdir(parents=True)
    statistics = {"run_identity": RUN_IDENTITY, "front_threshold_p90": 0.01,
                  "variables": {"sst": {"mean": TEMPERATURE_MEAN,
                                         "std": TEMPERATURE_STANDARD_DEVIATION},
                                "mixed_layer_depth": {"mean": 10.0, "std": 5.0}}}
    _write_json(root / "READY.json", {"status": "complete", "run_identity": RUN_IDENTITY})
    _write_json(root / "run.json", {"run_identity": RUN_IDENTITY, "config": config.to_dict()})
    _write_json(root / "stats.json", statistics)
    np.savez(root / "grid_masks.npz", **_fixture_grid(config))
    np.save(root / "dates.npy", dates.to_numpy())
    index = build_sample_index(dates, config)
    index.to_csv(root / "sample_index.csv", index=False)
    base = _fixture_base()
    np.save(root / "normalized" / "base.npy", base)
    np.save(root / "normalized" / "thermal.npy", _fixture_thermal(len(dates)))
    return {"root": root, "context": load_context(root), "index": index, "base": base}


def _forecast_payload(prepared: dict, split: str = "test", units: str = "degree_Celsius") -> dict:
    index = prepared["index"].loc[prepared["index"].split == split]
    indices = index[["target_%d_index" % lead for lead in (1, 2, 3)]].to_numpy()
    physical_targets = prepared["base"][indices, 0].astype(np.float64) * 2.0 + 10.0
    prediction = physical_targets + 1.0
    if units == "K":
        prediction = prediction + 273.15
    elif units == "sst_zscore":
        prediction = (prediction - TEMPERATURE_MEAN) / TEMPERATURE_STANDARD_DEVIATION
    return {
        "predictions": prediction,
        "origin_dates": index.origin_date.to_numpy(dtype="datetime64[D]"),
        "target_dates": np.stack([index["target_%d_date" % lead].to_numpy(dtype="datetime64[D]")
                                  for lead in (1, 2, 3)], axis=1),
        "latitude": prepared["context"].grid["latitude"].copy(),
        "longitude": prepared["context"].grid["longitude"].copy(),
        "units": np.asarray(units), "run_identity": np.asarray(RUN_IDENTITY),
    }


def _forecast_dataset(payload: dict) -> xr.Dataset:
    dataset = xr.Dataset(
        {"predictions": (("origin", "lead", "latitude", "longitude"), payload["predictions"]),
         "target_dates": (("origin", "lead"), payload["target_dates"])},
        coords={"origin": payload["origin_dates"], "lead": [1, 2, 3],
                "latitude": payload["latitude"], "longitude": payload["longitude"]},
        attrs={"run_identity": str(payload["run_identity"])},
    )
    if "units" in payload:
        dataset.predictions.attrs["units"] = str(payload["units"])
    return dataset


def _write_forecast(path: Path, payload: dict) -> None:
    if path.suffix == ".npz":
        np.savez(path, **payload)
        return
    with _forecast_dataset(payload) as dataset:
        dataset.to_netcdf(path)


def _tree_digest(root: Path) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_context_accepts_complete_fixed_grid_and_one_origin_per_split(prepared):
    context = prepared["context"]
    assert context.run_identity == RUN_IDENTITY
    assert context.train_period == ("2020-01-01", "2020-01-18")
    assert context.front_threshold == 0.01
    assert prepared["index"].groupby("split").size().to_dict() == dict.fromkeys(SPLIT_NAMES, 1)
    assert context.grid["core_mask"].sum() == 25
    assert context.grid["ocean_mask"].sum() == 48
    assert context.evaluation_grid()["front_tolerance_km"] == 10.0


@pytest.mark.parametrize("extension", [".npz", ".nc"])
@pytest.mark.parametrize("units", ["degree_Celsius", "K", "sst_zscore"])
def test_forecast_formats_and_explicit_units_are_equivalent(prepared, extension, units):
    payload = _forecast_payload(prepared, units=units)
    path = prepared["root"].parent / ("supplied" + extension)
    _write_forecast(path, payload)
    before = _tree_digest(prepared["root"])
    expected = _forecast_payload(prepared)["predictions"]
    with ForecastFile(path, prepared["context"], "test") as forecasts:
        np.testing.assert_allclose(forecasts.read(slice(0, 1)), expected, atol=TOLERANCE)
        assert forecasts.units == units
        np.testing.assert_array_equal(forecasts.target_dates,
                                      payload["target_dates"].astype("datetime64[ns]"))
    assert forecasts.handle is None
    assert _tree_digest(prepared["root"]) == before


def test_npz_accepts_explicit_singleton_temperature_channel(prepared, tmp_path):
    payload = _forecast_payload(prepared)
    payload["predictions"] = payload["predictions"][:, :, None]
    path = tmp_path / "supplied.npz"
    _write_forecast(path, payload)
    with ForecastFile(path, prepared["context"], "test") as forecasts:
        assert forecasts.read(slice(0, 1)).shape == (1, 3, 1, 7, 7)


@pytest.mark.parametrize("split", SPLIT_NAMES)
@pytest.mark.parametrize("extension", [".npz", ".nc"])
def test_evaluation_reports_three_leads_from_canonical_truth(prepared, split, extension):
    tmp_path = prepared["root"].parent
    path = tmp_path / ("supplied" + extension)
    _write_forecast(path, _forecast_payload(prepared, split=split))
    before = _tree_digest(prepared["root"])
    output_root = tmp_path / "evaluation"
    report = evaluate_forecast_file(prepared["context"], {
        "predictions": path, "split": split, "output_root": output_root,
        "model_name": "synthetic_supplied_forecast", "front_lines": True,
    })
    assert report["status"] == "complete"
    assert report["run_identity"] == RUN_IDENTITY
    assert report["sample_count"] == 1
    assert report["daily_metric_records"] == 3
    assert [row["lead_day"] for row in report["metrics"]] == [1, 2, 3]
    for row in report["metrics"]:
        for metric in ("rmse", "mae", "bias"):
            assert row[metric] == pytest.approx(1.0, abs=TOLERANCE)
    _assert_evaluation_tables(output_root, report)
    assert _tree_digest(prepared["root"]) == before


def _assert_evaluation_tables(output_root: Path, report: dict) -> None:
    daily = pd.read_csv(output_root / "daily_errors.csv")
    summary = pd.read_csv(output_root / "metrics_summary.csv")
    assert daily.lead_day.tolist() == summary.lead_day.tolist() == [1, 2, 3]
    assert daily.origin_date.nunique() == 1
    origin = pd.Timestamp(daily.origin_date.iloc[0])
    assert pd.to_datetime(daily.target_date).tolist() == [origin + pd.Timedelta(days=day)
                                                       for day in (1, 2, 3)]
    for metric in ("rmse", "mae", "bias"):
        np.testing.assert_allclose(daily[metric], 1.0, atol=TOLERANCE)
        np.testing.assert_allclose(summary[metric], 1.0, atol=TOLERANCE)
    saved = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    assert saved["prediction_sha256"] == report["prediction_sha256"]
    assert (output_root / "evaluation_report.md").is_file()


def _corrupt_payload(payload: dict, case: str) -> dict:
    if case in {"origin_dates", "target_dates"}:
        payload[case] = payload[case] + np.timedelta64(1, "D")
    elif case in {"latitude", "longitude"}:
        payload[case] = payload[case] + 0.01
    elif case == "run_identity":
        payload[case] = np.asarray("different-run")
    elif case == "units":
        payload[case] = np.asarray("degree_C")
    elif case == "missing_units":
        del payload["units"]
    elif case == "empty_split":
        for key in ("origin_dates", "target_dates", "predictions"):
            payload[key] = payload[key][:0]
    elif case == "duplicate_origin":
        for key in ("origin_dates", "target_dates", "predictions"):
            payload[key] = np.concatenate([payload[key], payload[key]], axis=0)
    return payload


@pytest.mark.parametrize("extension", [".npz", ".nc"])
@pytest.mark.parametrize("case", ["origin_dates", "target_dates", "latitude", "longitude",
                                  "run_identity", "units", "missing_units", "empty_split",
                                  "duplicate_origin"])
def test_forecast_contract_rejects_mismatched_metadata(prepared, extension, case):
    path = prepared["root"].parent / ("invalid" + extension)
    _write_forecast(path, _corrupt_payload(_forecast_payload(prepared), case))
    with pytest.raises(DataValidationError):
        with ForecastFile(path, prepared["context"], "test"):
            pass


@pytest.mark.parametrize("case", ["lead_count", "spatial_shape", "channel_count", "nonnumeric"])
def test_npz_rejects_incompatible_prediction_shapes_and_types(prepared, tmp_path, case):
    payload = _forecast_payload(prepared)
    if case == "lead_count":
        payload["predictions"] = payload["predictions"][:, :2]
    elif case == "spatial_shape":
        payload["predictions"] = payload["predictions"][..., :-1]
    elif case == "channel_count":
        payload["predictions"] = np.repeat(payload["predictions"][:, :, None], 2, axis=2)
    else:
        payload["predictions"] = payload["predictions"].astype("U20")
    path = tmp_path / "invalid.npz"
    _write_forecast(path, payload)
    with pytest.raises(DataValidationError):
        with ForecastFile(path, prepared["context"], "test"):
            pass


def test_netcdf_rejects_wrong_dimension_order(prepared, tmp_path):
    path = tmp_path / "invalid.nc"
    with _forecast_dataset(_forecast_payload(prepared)) as dataset:
        dataset["predictions"] = dataset.predictions.transpose("origin", "latitude", "lead", "longitude")
        dataset.to_netcdf(path)
    with pytest.raises(DataValidationError, match="dimensions"):
        with ForecastFile(path, prepared["context"], "test"):
            pass


@pytest.mark.parametrize("is_ocean_nan", [False, True])
@pytest.mark.parametrize("extension", [".npz", ".nc"])
def test_evaluation_rejects_ocean_nans_and_accepts_land_nans(prepared, is_ocean_nan, extension):
    tmp_path = prepared["root"].parent
    payload = _forecast_payload(prepared)
    position = (1, 1) if is_ocean_nan else (3, 3)
    payload["predictions"][..., position[0], position[1]] = np.nan
    path = tmp_path / ("supplied" + extension)
    _write_forecast(path, payload)
    options = {"predictions": path, "split": "test", "output_root": tmp_path / "evaluation"}
    if is_ocean_nan:
        with pytest.raises(DataValidationError, match="ocean"):
            evaluate_forecast_file(prepared["context"], options)
        return
    report = evaluate_forecast_file(prepared["context"], options)
    assert all(row["rmse"] == pytest.approx(1.0, abs=TOLERANCE) for row in report["metrics"])


@pytest.mark.parametrize("case", ["ready_status", "ready_id", "run_id", "stats_id", "threshold"])
def test_context_rejects_incomplete_or_mismatched_runs(prepared, case):
    root = prepared["root"]
    name = "stats.json" if case in {"stats_id", "threshold"} else "run.json" if case == "run_id" else "READY.json"
    value = json.loads((root / name).read_text(encoding="utf-8"))
    key = "status" if case == "ready_status" else "front_threshold_p90" if case == "threshold" else "run_identity"
    value[key] = "incomplete" if case == "ready_status" else 0.0 if case == "threshold" else "different-run"
    _write_json(root / name, value)
    with pytest.raises(DataValidationError):
        load_context(root)


@pytest.mark.parametrize("case", ["core_mask", "area_weights", "gradient_mask"])
def test_context_rejects_inconsistent_fixed_grid_masks(prepared, case):
    grid = dict(prepared["context"].grid)
    grid[case] = grid[case].copy()
    grid[case][1, 1] = 0 if case != "area_weights" else 2.0
    np.savez(prepared["root"] / "grid_masks.npz", **grid)
    with pytest.raises(DataValidationError):
        load_context(prepared["root"])


def test_analysis_config_resolves_paths_relative_to_its_file(tmp_path):
    folder = tmp_path / "config"
    folder.mkdir()
    path = folder / "analysis.json"
    _write_json(path, {"prepared_root": "../prepared", "diagnostics_output": "../diagnostics",
                       "figure_dpi": 150, "batch_size": 2})
    config = load_analysis_config(path)
    assert config.prepared_root == (tmp_path / "prepared").resolve()
    assert config.diagnostics_output == (tmp_path / "diagnostics").resolve()
    assert config.figure_dpi == 150 and config.batch_size == 2


@pytest.mark.parametrize("options", [{"figure_dpi": 71}, {"batch_size": 0}])
def test_analysis_config_rejects_invalid_resource_settings(tmp_path, options):
    path = tmp_path / "analysis.json"
    _write_json(path, dict({"prepared_root": "prepared", "diagnostics_output": "diagnostics"}, **options))
    with pytest.raises(ConfigurationError):
        load_analysis_config(path)


@pytest.mark.parametrize("relationship", ["same", "descendant", "ancestor"])
def test_analysis_output_cannot_overwrite_preprocessed_tree(prepared, relationship):
    root = prepared["root"]
    output = root if relationship == "same" else root / "metrics" if relationship == "descendant" else root.parent
    with pytest.raises(ConfigurationError):
        validate_output_location(output, root)


@pytest.mark.parametrize("dtype", [np.complex64, np.bool_])
def test_npz_rejects_complex_and_boolean_temperatures(prepared, tmp_path, dtype):
    payload = _forecast_payload(prepared)
    payload["predictions"] = payload["predictions"].astype(dtype)
    path = tmp_path / "non_real_temperature.npz"
    _write_forecast(path, payload)
    with pytest.raises(DataValidationError, match="real numeric"):
        with ForecastFile(path, prepared["context"], "test"):
            pass


@pytest.mark.parametrize("name", ["units", "run_identity", "predictions"])
def test_npz_rejects_object_arrays_with_a_contextual_error(prepared, monkeypatch, name):
    payload = _forecast_payload(prepared)
    payload[name] = payload[name].astype(object)
    path = prepared["root"].parent / "unsafe_object_array.npz"
    _write_forecast(path, payload)
    tracked = _track_evaluation_handles(monkeypatch)
    with pytest.raises(DataValidationError, match="NPZ"):
        with ForecastFile(path, prepared["context"], "test"):
            pass
    assert len(tracked["archives"]) == 1 and tracked["archives"][0].zip is None


def _with_two_training_origins(prepared: dict) -> dict:
    root = prepared["root"]
    dates = pd.date_range("2020-01-01", periods=55, freq="D")
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    options = run["config"]
    options["end_date"] = str(dates[-1].date())
    bounds = [(0, 18), (19, 36), (37, 54)]
    options["splits"] = {name: [str(dates[start].date()), str(dates[end].date())]
                         for name, (start, end) in zip(SPLIT_NAMES, bounds)}
    config = PreprocessConfig(**options)
    run["config"] = config.to_dict()
    _write_json(root / "run.json", run)
    np.save(root / "dates.npy", dates.to_numpy())
    base = np.concatenate([prepared["base"], prepared["base"][-1:]], axis=0)
    base[-1, 0] += 0.01
    base[:, :, 3, 3] = 0.0
    np.save(root / "normalized" / "base.npy", base)
    np.save(root / "normalized" / "thermal.npy", _fixture_thermal(len(dates)))
    index = build_sample_index(dates, config)
    index.to_csv(root / "sample_index.csv", index=False)
    return dict(prepared, context=load_context(root), index=index, base=base)


@pytest.mark.parametrize("extension", [".npz", ".nc"])
def test_file_evaluation_is_independent_of_batch_partition(prepared, tmp_path, extension):
    prepared = _with_two_training_origins(prepared)
    payload = _forecast_payload(prepared, split="train")
    payload["predictions"][1] += 2.0
    path = tmp_path / ("supplied" + extension)
    _write_forecast(path, payload)
    reports, daily_tables = [], []
    for batch_size in (1, 2):
        output = tmp_path / ("batch_%d" % batch_size)
        reports.append(evaluate_forecast_file(prepared["context"], {
            "predictions": path, "split": "train", "output_root": output,
            "batch_size": batch_size, "front_lines": True,
        }))
        daily_tables.append(pd.read_csv(output / "daily_errors.csv"))
    assert reports[0]["metrics"] == reports[1]["metrics"]
    pd.testing.assert_frame_equal(daily_tables[0], daily_tables[1])
    assert len(daily_tables[0]) == 6 and daily_tables[0].origin_date.nunique() == 2
    for row in reports[0]["metrics"]:
        assert row["sample_count"] == 2
        assert row["rmse"] == pytest.approx(np.sqrt(5.0), abs=TOLERANCE)
        assert row["mae"] == row["bias"] == pytest.approx(2.0, abs=TOLERANCE)


def _track_evaluation_handles(monkeypatch) -> dict:
    tracked = {"forecasts": [], "memmaps": [], "archives": []}
    original_forecast, original_load = ForecastFile, np.load

    def open_forecast(path, context, split):
        forecasts = original_forecast(path, context, split)
        tracked["forecasts"].append(forecasts)
        return forecasts

    def load_array(*arguments, **options):
        values = original_load(*arguments, **options)
        if isinstance(values, np.memmap):
            tracked["memmaps"].append(values)
        elif isinstance(values, np.lib.npyio.NpzFile):
            tracked["archives"].append(values)
        return values

    monkeypatch.setattr(evaluation_module, "ForecastFile", open_forecast)
    monkeypatch.setattr(evaluation_module.np, "load", load_array)
    return tracked


@pytest.mark.parametrize("case", [(".npz", False), (".npz", True), (".nc", False), (".nc", True)])
def test_evaluation_releases_sources_after_success_and_failure(prepared, monkeypatch, case):
    extension, should_fail = case
    payload = _forecast_payload(prepared)
    if should_fail:
        payload["predictions"][0, 0, 1, 1] = np.nan
    path = prepared["root"].parent / ("supplied" + extension)
    _write_forecast(path, payload)
    tracked = _track_evaluation_handles(monkeypatch)
    options = {"predictions": path, "split": "test",
               "output_root": prepared["root"].parent / "evaluation"}
    if should_fail:
        with pytest.raises(DataValidationError, match="ocean"):
            evaluate_forecast_file(prepared["context"], options)
    else:
        evaluate_forecast_file(prepared["context"], options)
    assert len(tracked["forecasts"]) == len(tracked["memmaps"]) == 1
    assert tracked["forecasts"][0].handle is None
    assert tracked["memmaps"][0]._mmap.closed
    assert all(archive.zip is None for archive in tracked["archives"])


@pytest.mark.parametrize("extension", [".npz", ".nc"])
@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_grouped_evaluation_keeps_all_daily_rows_and_emits_all_groups(prepared, extension, split):
    root = prepared["root"].parent
    path = root / ("grouped_supplied" + extension)
    _write_forecast(path, _forecast_payload(prepared, split=split))
    before = _tree_digest(prepared["root"])
    output = root / "grouped_evaluation"
    report = evaluate_forecast_file(prepared["context"], {
        "predictions": path, "split": split, "output_root": output,
        "mld_threshold_m": 10.0, "batch_size": 1})
    assert len(report["group_metrics"]) == 24
    assert report["mld_stratification"]["threshold_m"] == 10.0
    _assert_grouped_tables(output, report)
    _assert_group_conservation(report)
    assert _tree_digest(prepared["root"]) == before
    _read_strict_json(output / "metrics.json")
    assert not list(output.glob("*.tmp"))


def _read_strict_json(path: Path) -> dict:
    def reject_nonfinite(value):
        raise AssertionError("JSON contains nonstandard nonfinite value: %s" % value)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def _assert_grouped_tables(output: Path, report: dict) -> None:
    daily = pd.read_csv(output / "daily_errors.csv")
    grouped_daily = pd.read_csv(output / "group_errors.csv")
    summary = pd.read_csv(output / "group_metrics.csv")
    groups = {"DJF", "MAM", "JJA", "SON", "strong_front", "weak_front", "shallow_mld", "deep_mld"}
    assert len(daily) == report["daily_metric_records"] == 3 * report["sample_count"]
    assert len(grouped_daily) == 8 * len(daily)
    assert len(summary) == 24 and set(summary.group) == groups
    assert grouped_daily.groupby("group").size().to_dict() == dict.fromkeys(groups, len(daily))
    expected_dates = daily[["origin_date", "target_date", "lead_day"]].sort_values(["origin_date", "lead_day"]).reset_index(drop=True)
    for _, rows in grouped_daily.groupby("group"):
        actual = rows[["origin_date", "target_date", "lead_day"]].sort_values(["origin_date", "lead_day"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_dates, actual)
    assert grouped_daily.loc[grouped_daily.group.isin(["MAM", "JJA", "SON"]), "valid_count"].eq(0).all()
    assert grouped_daily.loc[grouped_daily.group.isin(["MAM", "JJA", "SON"]), "rmse"].isna().all()


def _assert_group_conservation(report: dict) -> None:
    groups = {(row["group"], row["lead_day"]): row for row in report["group_metrics"]}
    for all_row in report["metrics"]:
        lead = all_row["lead_day"]
        seasons = [groups[(name, lead)] for name in ("DJF", "MAM", "JJA", "SON")]
        depth = [groups[(name, lead)] for name in ("shallow_mld", "deep_mld")]
        fronts = [groups[(name, lead)] for name in ("strong_front", "weak_front")]
        for partition in (seasons, depth):
            for metric in ("valid_count", "weight_sum", "weighted_sse", "gradient_count", "gradient_weight_sum"):
                assert sum(row[metric] for row in partition) == pytest.approx(all_row[metric])
        assert sum(row["valid_count"] for row in fronts) == all_row["gradient_count"]
        assert sum(row["weight_sum"] for row in fronts) == pytest.approx(all_row["gradient_weight_sum"])
        assert sum(row["gradient_vector_error_sum"] for row in fronts) == pytest.approx(all_row["gradient_vector_error_sum"])


@pytest.mark.parametrize("bad_index", ("negative", "beyond_end", "fractional", "wrong_date"))
def test_target_indices_are_validated_before_truth_lookup(prepared, monkeypatch, bad_index):
    root = prepared["root"].parent
    path = root / "supplied.npz"
    _write_forecast(path, _forecast_payload(prepared))
    index = prepared["index"].copy()
    row = index.index[index.split == "test"][0]
    replacement = {"negative": -1, "beyond_end": len(prepared["base"]),
                   "fractional": 1.5, "wrong_date": index.loc[row, "target_1_index"] + 1}[bad_index]
    index.loc[row, "target_1_index"] = replacement
    index.to_csv(prepared["root"] / "sample_index.csv", index=False)
    tracked = _track_evaluation_handles(monkeypatch)
    output = root / "invalid_truth"
    with pytest.raises(DataValidationError, match="Target indices"):
        evaluate_forecast_file(prepared["context"], {"predictions": path, "split": "test",
                               "output_root": output, "mld_threshold_m": 10.0})
    assert not (output / "metrics.json").exists()
    assert len(tracked["memmaps"]) == 2
    assert all(values._mmap.closed for values in tracked["memmaps"])
    assert all(forecasts.handle is None for forecasts in tracked["forecasts"])


def test_grouped_evaluation_releases_thermal_base_and_forecasts(prepared, monkeypatch):
    root = prepared["root"].parent
    path = root / "supplied.npz"
    _write_forecast(path, _forecast_payload(prepared))
    tracked = _track_evaluation_handles(monkeypatch)
    evaluate_forecast_file(prepared["context"], {"predictions": path, "split": "test",
                           "output_root": root / "grouped", "mld_threshold_m": 10.0})
    assert len(tracked["memmaps"]) == 2
    assert all(values._mmap.closed for values in tracked["memmaps"])
    assert all(archive.zip is None for archive in tracked["archives"])
    assert all(forecasts.handle is None for forecasts in tracked["forecasts"])


def test_cli_evaluate_runs_real_grouped_pipeline_and_writes_complete_manifest(prepared, monkeypatch, capsys):
    import sst_analysis.stratification as stratification_module
    root = prepared["root"].parent
    path, output = root / "supplied.npz", root / "cli_evaluation"
    _write_forecast(path, _forecast_payload(prepared))
    config = root / "analysis.json"
    _write_json(config, {"prepared_root": str(prepared["root"]), "diagnostics_output": str(root / "diagnostics")})
    fitted = []

    def synthetic_training_threshold(context, destination):
        fitted.append((context.train_period, destination))
        return {"status": "complete", "threshold_m": 10.0, "run_identity": context.run_identity}

    monkeypatch.setattr(stratification_module, "fit_training_mld_threshold", synthetic_training_threshold)
    monkeypatch.setattr(sys, "argv", ["sst_analysis", "evaluate", "--config", str(config),
                                     "--predictions", str(path), "--split", "test", "--output-root", str(output),
                                     "--model-name", "supplied_cli_model"])
    assert analysis_cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_name"] == "supplied_cli_model" and len(report["group_metrics"]) == 24
    assert fitted == [(prepared["context"].train_period, root / "diagnostics")]
    _assert_cli_evaluation_manifest(output, report)


def _assert_cli_evaluation_manifest(output: Path, report: dict) -> None:
    manifest = _read_strict_json(output / "analysis_manifest.json")
    assert manifest["run_identity"] == report["run_identity"] == RUN_IDENTITY
    assert manifest["kind"] == "supplied_model_forecast_evaluation"
    expected = {"metrics.json", "metrics_summary.csv", "daily_errors.csv", "evaluation_report.md",
                "group_metrics.csv", "group_errors.csv"}
    assert expected <= set(manifest["artifacts"])
    assert all((output / name).is_file() for name in manifest["artifacts"])
    saved = _read_strict_json(output / "metrics.json")
    assert saved["mld_stratification"] == report["mld_stratification"]
    assert len(saved["group_metrics"]) == 24


def test_cli_compare_parser_accepts_paired_evaluation_paths_and_seed():
    arguments = analysis_cli.build_parser().parse_args([
        "compare", "--evaluation-a", "model_a", "--evaluation-b", "model_b",
        "--output-root", "comparison", "--seed", "27"])
    assert arguments.command == "compare"
    assert arguments.evaluation_a == Path("model_a") and arguments.evaluation_b == Path("model_b")
    assert arguments.output_root == Path("comparison") and arguments.seed == 27
