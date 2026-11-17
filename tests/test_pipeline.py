"""Offline end-to-end NetCDF, memmap, split and resume regression tests."""

import contextlib
import dataclasses
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.constants import ENERGY_VARIABLES, FEATURE_GROUPS
from sst_preprocessing.dataset import PreparedSSTDataset
from sst_preprocessing.errors import DataValidationError, ResumeMismatchError
from sst_preprocessing.pipeline import run_preprocessing
from sst_preprocessing.verification import verify_output

FIXTURE_DATES = pd.date_range("2020-01-15", periods=54, freq="D")
FIXTURE_LATITUDE = np.linspace(-48.15, -47.85, 7)
FIXTURE_LONGITUDE = np.linspace(-60.15, -59.85, 7)
EXPECTED_COUNTS = {"train": 1, "validation": 1, "test": 1}
ERA5_VARIABLES = ("u10", "v10", "t2m", "msl") + ENERGY_VARIABLES


def create_pipeline_fixture(root: Path) -> PreprocessConfig:
    """Create 54 original-date daily fields with nonconstant features and known land."""
    for variable in ("sst", "uo", "vo", "mlotst"):
        path = _source_path(root, variable, FIXTURE_DATES)
        path.parent.mkdir(parents=True, exist_ok=True)
        _source_dataset(variable, FIXTURE_DATES).to_netcdf(path)
    for month in FIXTURE_DATES.to_period("M").unique():
        dates = FIXTURE_DATES[FIXTURE_DATES.to_period("M") == month]
        path = root / "ERA5" / "2020" / ("ERA5_2020_%02d.nc" % month.month)
        path.parent.mkdir(parents=True, exist_ok=True)
        _source_dataset("era5", dates).to_netcdf(path)
    split_dates = [(FIXTURE_DATES[index], FIXTURE_DATES[index + 17]) for index in (0, 18, 36)]
    splits = {name: (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
              for name, (start, end) in zip(EXPECTED_COUNTS, split_dates)}
    return PreprocessConfig(data_root=root, output_root=root.parent / "processed",
        start_date="2020-01-15", end_date="2020-03-08", splits=splits,
        core_south=float(FIXTURE_LATITUDE[1]), core_north=float(FIXTURE_LATITUDE[-2]),
        core_west=float(FIXTURE_LONGITUDE[1]), core_east=float(FIXTURE_LONGITUDE[-2]),
        expected_grid_shape=(7, 7), expected_core_shape=(5, 5), chunk_days=4)


def _source_path(root: Path, variable: str, dates: pd.DatetimeIndex) -> Path:
    time_range = "%sT00-00-00-%sT00-00-00.nc" % (dates[0].strftime("%Y-%m-%d"),
                                                       dates[-1].strftime("%Y-%m-%d"))
    region = "60.15W-59.85W_48.15S-47.85S_"
    if variable == "sst":
        name = "METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_" + region + time_range
        return root / "OSTIA" / name
    depth = "0.49m_" if variable in ("uo", "vo") else ""
    name = "cmems_mod_glo_phy_my_0.083deg_P1D-m_%s_%s%s%s" % (variable, region, depth, time_range)
    return root / "CMEMS" / ("MLD" if variable == "mlotst" else "current") / name


def _source_dataset(variable: str, dates: pd.DatetimeIndex) -> xr.Dataset:
    names = ERA5_VARIABLES if variable == "era5" else ("analysed_sst" if variable == "sst" else variable,)
    fields = {name: (("time", "latitude", "longitude"), _source_values(name, dates)) for name in names}
    dataset = xr.Dataset(fields, coords={"time": dates, "latitude": FIXTURE_LATITUDE,
                                        "longitude": FIXTURE_LONGITUDE})
    dataset.latitude.attrs["units"] = "degrees_north"
    dataset.longitude.attrs["units"] = "degrees_east"
    for name in names:
        dataset[name].attrs["units"] = _source_units(name)
    if variable == "sst":
        dataset.analysed_sst.attrs["standard_name"] = "sea_surface_foundation_temperature"
    if variable in ("uo", "vo"):
        dataset = dataset.expand_dims(depth=[0.494])
        dataset.depth.attrs["units"] = "m"
    if variable == "era5":
        dataset.attrs.update(time_zone="UTC", processing="Daily UTC mean for air; daily UTC sum for energy.")
    return dataset


def _source_values(name: str, dates: pd.DatetimeIndex) -> np.ndarray:
    day = np.asarray((dates - FIXTURE_DATES[0]).days, dtype=np.float64)[:, None, None]
    north = np.arange(7, dtype=np.float64)[None, :, None]
    east = np.arange(7, dtype=np.float64)[None, None, :]
    fields = {"analysed_sst": 280 + .07 * day + .2 * north + .3 * east + .01 * day * east,
              "uo": .01 * day + .1 * east - .08 * north,
              "vo": -.05 * day + .1 * north + .02 * east,
              "u10": 2 + .1 * day + .01 * east + .02 * north,
              "v10": -3 + .03 * day + .2 * north + .01 * east,
              "t2m": 285 + .09 * day + .2 * north + .1 * east,
              "msl": 100000 + 3 * day + 5 * east + 2 * north,
              "mlotst": 15 + .2 * day + east + north / 2,
              "ssr": 12e6 + 15000 * day + 2000 * east + 1000 * north,
              "str": -3e6 - 1000 * day - 100 * north - 200 * east,
              "slhf": -4e6 - 3000 * day - 400 * east - 200 * north,
              "sshf": -1e6 - 2000 * day - 100 * east - 300 * north}
    values = np.asarray(fields[name], dtype=np.float32)
    if name == "analysed_sst":
        values += np.where(day >= 18, 50.0, 0.0).astype(np.float32)
    if name in ("analysed_sst", "uo", "vo", "mlotst"):
        values[:, 0, 0] = np.nan
    if name == "mlotst":
        values[:, 2, 2] = 0.0
    return values


def _source_units(name: str) -> str:
    if name in ENERGY_VARIABLES:
        return "J m-2"
    if name in ("analysed_sst", "t2m"):
        return "K"
    if name == "mlotst":
        return "m"
    return "Pa" if name == "msl" else "m s-1"


def _quiet_run(config: PreprocessConfig, **options) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return run_preprocessing(config, **options)


def _load_physical(root: Path) -> xr.Dataset:
    pieces = []
    for path in sorted((root / "physical").glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            pieces.append(dataset.load())
    return xr.concat(pieces, dim="time")


def _close_dataset(dataset: PreparedSSTDataset) -> None:
    for array in dataset.arrays.values():
        array._mmap.close()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "output" / "offline_pipeline_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = temporary_root / ("swa_pipeline_" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.config = create_pipeline_fixture(self.directory / "data")

    def tearDown(self):
        if self.directory.parent.name != "offline_pipeline_tests":
            raise RuntimeError("Refusing to remove fixtures outside their dedicated test root")
        shutil.rmtree(self.directory)

    def test_complete_pipeline_verification_and_dataset_windows(self):
        report = _quiet_run(self.config)
        self.assertEqual(report["sample_counts"], EXPECTED_COUNTS)
        self.assertEqual(report["days"], 54)
        self.assertEqual(report["input_grid_shape"], [7, 7])
        self.assertEqual(report["core_grid_shape"], [5, 5])
        self.assertEqual(verify_output(self.config.output_root)["status"], "verified")
        physical = _load_physical(self.config.output_root)
        self.assertEqual(pd.DatetimeIndex(physical.time.values).tolist(), FIXTURE_DATES.tolist())
        self.assertGreater(report["mld_filled_target_grid_days"], 0)
        self._assert_training_statistics(physical)
        for split, origin in (("train", 14), ("validation", 32), ("test", 50)):
            dataset = PreparedSSTDataset(self.config.output_root, split=split)
            try:
                self._assert_sample(dataset, physical, origin)
            finally:
                _close_dataset(dataset)

    def _assert_training_statistics(self, physical: xr.Dataset) -> None:
        with (self.config.output_root / "stats.json").open(encoding="utf-8") as handle:
            statistics = json.load(handle)
        with np.load(self.config.output_root / "grid_masks.npz") as masks:
            mask = masks["ocean_mask"]
            self.assertEqual(int(masks["core_mask"].sum()), 25)
        training = physical.sst.values[:18, mask]
        self.assertAlmostEqual(statistics["variables"]["sst"]["mean"], float(np.mean(training, dtype=np.float64)), places=6)
        self.assertLess(statistics["variables"]["sst"]["mean"], 20.0)

    def _assert_sample(self, dataset: PreparedSSTDataset, physical: xr.Dataset, origin: int) -> None:
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["base"].shape, (14, 7, 7, 7))
        self.assertEqual(sample["front"].shape, (14, 3, 7, 7))
        self.assertEqual(sample["thermal"].shape, (14, 3, 7, 7))
        self.assertEqual(sample["target"].shape, (3, 1, 7, 7))
        self.assertEqual(sample["origin_date"], FIXTURE_DATES[origin].strftime("%Y-%m-%d"))
        np.testing.assert_array_equal(sample["target_dates"], FIXTURE_DATES[origin + 1:origin + 4].values)
        np.testing.assert_array_equal(sample["base"][:, 0], dataset.arrays["base"][origin - 13:origin + 1, 0])
        np.testing.assert_array_equal(sample["base"][:, 1:], dataset.arrays["base"][origin - 14:origin, 1:])
        expected_sst = np.where(dataset.ocean_mask, physical.sst.values[origin], 0.0)
        np.testing.assert_allclose(sample["last_sst"][0], expected_sst, atol=2e-5)
        last_heat = np.where(dataset.ocean_mask, physical.heat_increment.values[origin - 1], 0.0)
        for lead in (1, 2, 3):
            np.testing.assert_allclose(sample["heat_increment"][lead - 1, 0], lead * last_heat, atol=1e-6)

    def test_month_resume_matches_uninterrupted_arrays_and_statistics(self):
        _quiet_run(self.config)
        resumed = dataclasses.replace(self.config, output_root=self.directory / "resumed")
        partial = _quiet_run(resumed, stop_after_months=1)
        self.assertEqual(partial["status"], "partial")
        self.assertFalse((resumed.output_root / "READY.json").exists())
        with self.assertRaisesRegex(DataValidationError, "incomplete"):
            PreparedSSTDataset(resumed.output_root)
        report = _quiet_run(resumed, resume=True)
        self.assertEqual(report["sample_counts"], EXPECTED_COUNTS)
        for group in FEATURE_GROUPS:
            np.testing.assert_array_equal(np.load(self.config.output_root / "normalized" / (group + ".npy")),
                                          np.load(resumed.output_root / "normalized" / (group + ".npy")))
        for name in ("stats.json", "sample_index.csv", "dates.npy"):
            self.assertEqual((self.config.output_root / name).read_bytes(), (resumed.output_root / name).read_bytes())
        xr.testing.assert_equal(_load_physical(self.config.output_root), _load_physical(resumed.output_root))

    def test_changed_configuration_rejects_resume(self):
        _quiet_run(self.config, stop_after_months=1)
        changed = dataclasses.replace(self.config, seawater_density=1026.0)
        with self.assertRaisesRegex(ResumeMismatchError, "different inputs/settings"):
            _quiet_run(changed, resume=True)

    def test_daily_and_four_day_chunks_produce_identical_features(self):
        _quiet_run(self.config)
        daily = dataclasses.replace(self.config, output_root=self.directory / "daily_chunks", chunk_days=1)
        _quiet_run(daily)
        for group in FEATURE_GROUPS:
            np.testing.assert_array_equal(np.load(self.config.output_root / "normalized" / (group + ".npy")),
                                          np.load(daily.output_root / "normalized" / (group + ".npy")))
        baseline = _load_physical(self.config.output_root)
        chunked = _load_physical(daily.output_root)
        for name in baseline.data_vars:
            xr.testing.assert_equal(baseline[name], chunked[name])

    def test_changed_source_rejects_resume(self):
        _quiet_run(self.config, stop_after_months=1)
        path = next(self.config.data_root.glob("ERA5/2020/ERA5_2020_02.nc"))
        with xr.open_dataset(path) as dataset:
            changed = dataset.load()
        changed["ssr"].values[0, 1, 1] += np.float32(12345.0)
        changed.to_netcdf(path, mode="w")
        with self.assertRaisesRegex(ResumeMismatchError, "different inputs/settings"):
            _quiet_run(self.config, resume=True)


if __name__ == "__main__":
    unittest.main()
