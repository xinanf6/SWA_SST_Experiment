"""Small offline diagnostics fixtures verify split isolation and aggregation rules."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst_analysis.context import AnalysisContext
from sst_analysis.diagnostics import run_diagnostics
from sst_preprocessing.constants import VARIABLE_UNITS
from sst_preprocessing.errors import DataValidationError


PHYSICAL_VARIABLES = {"sst": 10.0, "uo": 3.0, "vo": 4.0,
                      "gradient_magnitude": 0.2, "net_heat_flux": 100.0,
                      "mixed_layer_depth": 20.0, "heat_increment": 0.1}
TRAIN_VALUES = {"2011-01-30": 10.0, "2011-01-31": 12.0,
                "2011-02-01": 15.0, "2011-02-02": 19.0}


def _context(root: Path) -> AnalysisContext:
    latitude, longitude = np.arange(3.0), np.arange(4.0)
    core = np.zeros((3, 4), bool)
    core[1:, 1:3] = True
    ocean = np.ones_like(core)
    ocean[1, 1] = False
    gradient = ocean.copy()
    gradient[2, 2] = False
    weights = np.broadcast_to(np.array([1., 2., 4.])[:, None], core.shape).copy()
    grid = {"latitude": latitude, "longitude": longitude, "core_mask": core,
            "ocean_mask": ocean, "gradient_mask": gradient, "area_weights": weights}
    run = {"run_identity": "offline-diagnostics", "config": {
        "splits": {"train": ["2011-01-30", "2011-02-02"]}}}
    return AnalysisContext(root, run, {"front_threshold_p90": 0.2}, grid)


def _write_month(context: AnalysisContext, month: str) -> None:
    dates = pd.date_range(month + "-01", periods=pd.Period(month).days_in_month)
    shape = (len(dates), 3, 4)
    variables = {name: np.full(shape, value, np.float32) for name, value in PHYSICAL_VARIABLES.items()}
    for offset, date in enumerate(dates):
        variables["sst"][offset] = TRAIN_VALUES.get(str(date.date()), 1.e6)
    for values in variables.values():
        values[:, ~context.grid["ocean_mask"]] = 9.e9
        values[:, ~context.grid["core_mask"]] = 8.e8
    dataset = xr.Dataset({name: (("time", "latitude", "longitude"), values)
                          for name, values in variables.items()},
                         coords={"time": dates, "latitude": context.grid["latitude"],
                                 "longitude": context.grid["longitude"]},
                         attrs={"run_identity": context.run_identity})
    for name in PHYSICAL_VARIABLES:
        dataset[name].attrs["units"] = VARIABLE_UNITS[name]
    (context.root / "physical").mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(context.root / "physical" / (month + ".nc"), engine="netcdf4")


@pytest.fixture
def diagnostic_case(tmp_path):
    context = _context(tmp_path / "processed")
    for month in ("2011-01", "2011-02", "2011-03"):
        _write_month(context, month)
    return context, tmp_path / "diagnostics"


def test_training_only_climatology_masks_and_counts(diagnostic_case, monkeypatch):
    context, destination = diagnostic_case
    opened = []
    original_open = xr.open_dataset

    def tracked_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("sst_analysis.diagnostics.xr.open_dataset", tracked_open)
    summary = run_diagnostics(context, destination)
    assert opened == ["2011-01.nc", "2011-02.nc"]
    assert summary["training_day_count"] == 4
    assert summary["source_files"] == ["physical/2011-01.nc", "physical/2011-02.nc"]
    with original_open(destination / "seasonal_maps.nc") as maps:
        annual = maps.sel(season="annual")
        valid = context.grid["core_mask"] & context.grid["ocean_mask"]
        np.testing.assert_allclose(annual.mean_sst.values[valid], 14.)
        np.testing.assert_allclose(annual.std_sst.values[valid], np.sqrt(11.5), rtol=1.e-6)
        np.testing.assert_allclose(annual.mean_current_speed.values[valid], 5.)
        np.testing.assert_allclose(annual.mean_net_heat_flux.values[valid], 100.)
        assert np.isnan(annual.mean_sst.values[~valid]).all()
        assert annual.mean_sst_count.dtype == np.dtype("int32")
        assert (annual.mean_sst_count.values[valid] == 4).all()
        assert (maps.sel(season="MAM").mean_sst_count.values == 0).all()
        assert np.isnan(maps.sel(season="MAM").mean_sst.values).all()
        assert maps.season.values.tolist() == ["annual", "DJF", "MAM", "JJA", "SON"]


def test_monthly_cycle_and_cross_month_change(diagnostic_case):
    context, destination = diagnostic_case
    run_diagnostics(context, destination)
    daily = pd.read_csv(destination / "daily_regional.csv")
    np.testing.assert_allclose(daily.mean_sst, [10., 12., 15., 19.])
    np.testing.assert_allclose(daily.mean_abs_sst_change.iloc[1:], [2., 3., 4.])
    assert pd.isna(daily.mean_abs_sst_change.iloc[0])
    assert daily.mean_abs_sst_change_valid_count.tolist() == [0, 3, 3, 3]
    assert daily.strong_front_area_fraction.tolist() == [1.] * 4
    assert daily.strong_front_area_fraction_valid_count.tolist() == [2] * 4
    cycle = pd.read_csv(destination / "monthly_cycle.csv").set_index("month")
    assert len(cycle) == 12
    assert cycle.loc[1, "mean_sst"] == 11.
    assert cycle.loc[2, "mean_sst"] == 17.
    assert cycle.loc[1, "mean_abs_sst_change_day_count"] == 1
    assert cycle.loc[2, "mean_abs_sst_change"] == 3.5
    assert cycle.loc[3, "day_count"] == 0
    assert pd.isna(cycle.loc[3, "mean_sst"])
    with xr.open_dataset(destination / "seasonal_maps.nc") as maps:
        valid = context.grid["core_mask"] & context.grid["ocean_mask"]
        np.testing.assert_allclose(maps.mean_abs_sst_change.sel(season="annual").values[valid], 3.)
        assert (maps.mean_abs_sst_change_count.sel(season="annual").values[valid] == 3).all()
        assert np.isnan(maps.mean_gradient.sel(season="annual").values[2, 2])


def test_field_denominators_use_area_weights(diagnostic_case):
    context, destination = diagnostic_case
    path = context.root / "physical" / "2011-01.nc"
    with xr.open_dataset(path) as source:
        changed = source.load()
    changed["uo"].values[-2:, 1, 2] = np.nan
    changed["vo"].values[-2:, 2, 1] = 10.
    changed["gradient_magnitude"].values[-2:, 1, 2] = 0.
    changed.to_netcdf(path, engine="netcdf4")
    run_diagnostics(context, destination)
    first = pd.read_csv(destination / "daily_regional.csv").iloc[0]
    assert first.mean_uo == 3.
    assert first.mean_uo_valid_count == 2
    assert first.mean_vo == pytest.approx((4. * 2. + 10. * 4. + 4. * 4.) / 10.)
    assert first.strong_front_area_fraction == pytest.approx(4. / 6.)


def test_inputs_unchanged_atomic_outputs_and_json(diagnostic_case):
    context, destination = diagnostic_case
    paths = list((context.root / "physical").glob("*.nc"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    run_diagnostics(context, destination)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after
    assert not list(destination.glob("*.tmp"))
    assert set(path.name for path in destination.iterdir()) == {
        "seasonal_maps.nc", "daily_regional.csv", "monthly_cycle.csv", "diagnostics_summary.json"}
    summary = json.loads((destination / "diagnostics_summary.json").read_text("utf-8"))
    assert summary["coverage"]["mean_abs_sst_change"]["valid_grid_observations"] == 9
    assert summary["coverage"]["mean_sst"]["valid_grid_observations"] == 12


def test_missing_training_dates_raise(diagnostic_case):
    context, destination = diagnostic_case
    path = context.root / "physical" / "2011-02.nc"
    with xr.open_dataset(path) as source:
        changed = source.isel(time=slice(1, None)).load()
    changed.to_netcdf(path, engine="netcdf4")
    with pytest.raises(DataValidationError, match="incomplete"):
        run_diagnostics(context, destination)


def test_chunking_preserves_results(diagnostic_case, monkeypatch):
    context, destination = diagnostic_case
    run_diagnostics(context, destination)
    expected = pd.read_csv(destination / "daily_regional.csv")
    monkeypatch.setattr("sst_analysis.diagnostics.CHUNK_DAYS", 1)
    second = destination.parent / "one_day_chunks"
    run_diagnostics(context, second)
    pd.testing.assert_frame_equal(expected, pd.read_csv(second / "daily_regional.csv"))


def test_leap_year_counts_do_not_overflow_uint8(tmp_path):
    context = _context(tmp_path / "processed")
    context.run["config"]["splits"]["train"] = ["2012-01-01", "2012-12-31"]
    for month in range(1, 13):
        _write_month(context, "2012-%02d" % month)
    destination = tmp_path / "diagnostics"
    summary = run_diagnostics(context, destination)
    assert summary["training_day_count"] == 366
    with xr.open_dataset(destination / "seasonal_maps.nc") as maps:
        assert maps.mean_sst_count.sel(season="annual").values[2, 1] == 366
        assert maps.mean_abs_sst_change_count.sel(season="annual").values[2, 1] == 365
    assert "2012-02-29" in pd.read_csv(destination / "daily_regional.csv").date.tolist()


@pytest.mark.parametrize("relative_output", (".", "derived", "physical/derived", ".."))
def test_overlapping_output_directories_are_rejected(diagnostic_case, relative_output):
    context, _ = diagnostic_case
    forbidden = context.root / relative_output
    with pytest.raises(DataValidationError, match="separate"):
        run_diagnostics(context, forbidden)
    assert not (forbidden / "diagnostics_summary.json").exists()


@pytest.mark.parametrize("identity", (None, "another-preprocessing-run"))
def test_absent_or_wrong_source_identity_is_rejected(diagnostic_case, identity):
    context, destination = diagnostic_case
    path = context.root / "physical" / "2011-01.nc"
    with xr.open_dataset(path) as source:
        changed = source.load()
    changed.attrs.pop("run_identity")
    if identity is not None:
        changed.attrs["run_identity"] = identity
    changed.to_netcdf(path, engine="netcdf4")
    with pytest.raises(DataValidationError, match="another preprocessing run"):
        run_diagnostics(context, destination)


@pytest.mark.parametrize("variable", tuple(PHYSICAL_VARIABLES))
def test_incorrect_physical_units_are_rejected(diagnostic_case, variable):
    context, destination = diagnostic_case
    path = context.root / "physical" / "2011-01.nc"
    with xr.open_dataset(path) as source:
        changed = source.load()
    changed[variable].attrs["units"] = "incorrect-units"
    changed.to_netcdf(path, engine="netcdf4")
    with pytest.raises(DataValidationError, match="has units"):
        run_diagnostics(context, destination)
