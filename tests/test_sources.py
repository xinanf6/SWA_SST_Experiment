"""Offline source contracts, using only synthetic NetCDF fixtures in a temp directory."""

import dataclasses
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.constants import ENERGY_VARIABLES
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.sources import build_inventory, fit_mld_climatology, load_source_month

LATITUDE = np.array([-35.0, -48.0], dtype=np.float32)
LONGITUDE = np.array([300.0, 301.0, 302.0], dtype=np.float32)
START_DATE = "2019-12-25"
END_DATE = "2020-03-05"
ERA5_NAMES = ("u10", "v10", "t2m", "msl") + ENERGY_VARIABLES


def create_source_fixture(root: Path) -> PreprocessConfig:
    """Write a small cross-year fixture with a leap day, marine gaps and source land."""
    dates = pd.date_range(START_DATE, END_DATE)
    for year in sorted(set(dates.year)):
        yearly = dates[dates.year == year]
        for variable in ("sst", "uo", "vo", "mlotst"):
            path = _fixture_path(root, variable, yearly)
            path.parent.mkdir(parents=True, exist_ok=True)
            _fixture_dataset(variable, yearly).to_netcdf(path)
    for month in dates.to_period("M").unique():
        selected = dates[dates.to_period("M") == month]
        path = root / "ERA5" / str(month.year) / ("ERA5_%04d_%02d.nc" % (month.year, month.month))
        path.parent.mkdir(parents=True, exist_ok=True)
        _fixture_dataset("era5", selected).to_netcdf(path)
    (root / "CMEMS" / "current" / "current_202301.nc").write_text("not a NetCDF; must never open")
    return PreprocessConfig(data_root=root, output_root=root.parent / "processed",
                            start_date=START_DATE, end_date=END_DATE,
                            splits={"train": (START_DATE, "2020-02-29"),
                                    "validation": ("2020-03-01", "2020-03-02"),
                                    "test": ("2020-03-03", END_DATE)},
                            expected_grid_shape=None, expected_core_shape=None)


def _fixture_path(root: Path, variable: str, dates: pd.DatetimeIndex) -> Path:
    time_range = "%sT00-00-00-%sT00-00-00.nc" % (dates[0].strftime("%Y-%m-%d"),
                                                       dates[-1].strftime("%Y-%m-%d"))
    if variable == "sst":
        return root / "OSTIA" / ("METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_fixture_" + time_range)
    folder = "MLD" if variable == "mlotst" else "current"
    filename = "cmems_mod_glo_phy_my_0.083deg_P1D-m_%s_fixture_%s" % (variable, time_range)
    return root / "CMEMS" / folder / filename


def _fixture_dataset(variable: str, dates: pd.DatetimeIndex) -> xr.Dataset:
    coordinates = {"time": dates, "latitude": LATITUDE, "longitude": LONGITUDE}
    names = ERA5_NAMES if variable == "era5" else ("analysed_sst" if variable == "sst" else variable,)
    fields = {name: (("time", "latitude", "longitude"), _fixture_values(name, dates)) for name in names}
    dataset = xr.Dataset(fields, coords=coordinates)
    dataset.latitude.attrs["units"] = "degrees_north"
    dataset.longitude.attrs["units"] = "degrees_east"
    for name in names:
        dataset[name].attrs["units"] = _fixture_units(name)
    if variable == "sst":
        dataset.analysed_sst.attrs["standard_name"] = "sea_surface_foundation_temperature"
    if variable in ("uo", "vo"):
        dataset = dataset.expand_dims(depth=[0.494])
        dataset.depth.attrs["units"] = "m"
    if variable == "era5":
        dataset.attrs.update(time_zone="UTC", processing="Daily UTC mean; daily UTC sum of energy.")
    return dataset


def _fixture_values(name: str, dates: pd.DatetimeIndex) -> np.ndarray:
    values = np.ones((len(dates), 2, 3), dtype=np.float32)
    if name in ("analysed_sst", "t2m"):
        values *= 290.0
    if name in ENERGY_VARIABLES:
        values *= -86400.0 if name != "ssr" else 8640000.0
    if name in ("uo", "vo", "analysed_sst", "mlotst"):
        values[:, 0, 0] = np.nan
    if name == "mlotst":
        values *= 20.0
        values[:, 1, 0] = 1000.0
        values[dates.month == 3] *= 100.0
        values[:, 1, 1] = 0.0
    return values


def _fixture_units(name: str) -> str:
    if name in ENERGY_VARIABLES:
        return "J m-2"
    if name in ("analysed_sst", "t2m"):
        return "K"
    if name == "mlotst":
        return "m"
    return "Pa" if name == "msl" else "m s**-1"


class SourceTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "output" / "offline_source_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = temporary_root / ("swa_sources_" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.config = create_source_fixture(self.directory / "data")

    def tearDown(self):
        if self.directory.parent.name != "offline_source_tests":
            raise RuntimeError("Refusing to remove a test directory outside its dedicated root")
        shutil.rmtree(self.directory)

    def test_inventory_hashes_only_canonical_inputs(self):
        inventory = build_inventory(self.config)
        self.assertEqual(len(inventory["files"]), 12)
        self.assertEqual(len(inventory["excluded_files"]), 1)
        self.assertIn("Duplicate", inventory["excluded_files"][0]["reason"])
        self.assertTrue(all(len(record["sha256"]) == 64 for record in inventory["files"]))
        self.assertEqual(inventory["fingerprint"], build_inventory(self.config)["fingerprint"])
        json.dumps(inventory)

    def test_load_month_preserves_dates_units_signed_energy_and_leap_day(self):
        inventory = build_inventory(self.config)
        month = load_source_month(inventory, "2020-02-01")
        self.assertEqual(month["sst"].dims, ("time", "latitude", "longitude"))
        self.assertEqual(len(month["sst"].time), 29)
        self.assertEqual(str(month["sst"].time.values[-1])[:10], "2020-02-29")
        np.testing.assert_array_equal(month["sst"].latitude, [-48.0, -35.0])
        np.testing.assert_array_equal(month["sst"].longitude, [-60.0, -59.0, -58.0])
        self.assertEqual(month["sst"].attrs["units"], "K")
        self.assertEqual(float(month["era5"].slhf.values[0, 0, 0]), -86400.0)
        self.assertEqual(len(load_source_month(inventory, "2019-12-01")["sst"].time), 7)

    def test_climatology_is_training_only_marine_and_does_not_cap_mld(self):
        inventory = build_inventory(self.config)
        climatology = fit_mld_climatology(self.config, inventory)
        self.assertEqual(climatology["monthly_mean"].dtype, np.float32)
        self.assertFalse(climatology["ocean_mask"][1, 0])
        self.assertTrue(np.isnan(climatology["monthly_mean"][:, 1, 0]).all())
        self.assertEqual(float(climatology["monthly_mean"][0, 0, 0]), 1000.0)
        self.assertTrue(np.isnan(climatology["monthly_mean"][2]).all())
        self.assertEqual(float(climatology["monthly_median"][0]), 20.0)
        self.assertEqual(float(climatology["annual_mean"][0, 2]), 20.0)

    def test_missing_and_duplicate_source_dates_are_rejected(self):
        missing = sorted(self.config.data_root.glob("ERA5/2020/ERA5_2020_02.nc"))[0]
        missing.unlink()
        with self.assertRaisesRegex(DataValidationError, "Missing era5 source coverage"):
            build_inventory(self.config)

    def test_duplicate_canonical_files_are_rejected(self):
        original = sorted(self.config.data_root.glob("OSTIA/*.nc"))[0]
        duplicate = original.with_name(original.name.replace("fixture", "copy"))
        duplicate.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(DataValidationError, "duplicate sst source dates"):
            build_inventory(self.config)

    def test_incorrect_units_daily_provenance_and_changed_input_fail(self):
        inventory = build_inventory(self.config)
        path = Path(next(record["path"] for record in inventory["files"] if record["variable"] == "era5"))
        with xr.open_dataset(path) as source:
            modified = source.load()
        modified.ssr.attrs["units"] = "W m-2"
        modified.to_netcdf(path, mode="w")
        with self.assertRaisesRegex(DataValidationError, "differs from hashed input"):
            load_source_month(inventory, "2019-12-01")
        with self.assertRaisesRegex(DataValidationError, "Unsupported units"):
            build_inventory(self.config)

    def test_missing_day_is_rejected_before_processing(self):
        path = next(self.config.data_root.glob("ERA5/2020/ERA5_2020_02.nc"))
        with xr.open_dataset(path) as source:
            modified = source.isel(time=[0, 2]).load()
        modified.to_netcdf(path, mode="w")
        with self.assertRaisesRegex(DataValidationError, "Missing day"):
            build_inventory(self.config)

    def test_expected_grid_shape_and_invalid_month_are_rejected(self):
        invalid = dataclasses.replace(self.config, expected_grid_shape=(270, 250))
        with self.assertRaisesRegex(DataValidationError, "OSTIA grid shape"):
            build_inventory(invalid)
        inventory = build_inventory(self.config)
        with self.assertRaisesRegex(DataValidationError, "first day"):
            load_source_month(inventory, "2020-02-02")
        with self.assertRaisesRegex(DataValidationError, "outside input coverage"):
            load_source_month(inventory, "2018-01-01")


if __name__ == "__main__":
    unittest.main()
