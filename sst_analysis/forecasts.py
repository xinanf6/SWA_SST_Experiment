"""Explicit forecast-file contract; observations are always read from prepared truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.constants import KELVIN_OFFSET
from sst_preprocessing.statistics import denormalize_values

from .context import AnalysisContext


ALLOWED_UNITS = {"degree_Celsius", "K", "sst_zscore"}
FORECAST_KEYS = {"predictions", "origin_dates", "target_dates", "latitude", "longitude",
                 "units", "run_identity"}


def _string_scalar(value: np.ndarray, name: str) -> str:
    values = np.asarray(value)
    if values.size != 1 or values.dtype.kind not in "US":
        raise DataValidationError("Forecast %s must be a scalar Unicode/ASCII string" % name)
    scalar = values.reshape(-1)[0]
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def _date_array(values: np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(values).astype("datetime64[ns]")
    except (ValueError, TypeError) as error:
        raise DataValidationError("Cannot decode forecast %s: %s" % (name, error))
    if np.isnat(result).any():
        raise DataValidationError("Forecast %s contains NaT" % name)
    return result


def _expected_index(context: AnalysisContext, split: str) -> pd.DataFrame:
    if split not in {"train", "validation", "test"}:
        raise DataValidationError("Unknown evaluation split: %s" % split)
    frame = pd.read_csv(context.root / "sample_index.csv")
    selected = frame.loc[frame["split"] == split].reset_index(drop=True)
    if selected.empty:
        raise DataValidationError("There are no usable forecast origins in split: %s" % split)
    return selected


class ForecastFile:
    """Read full-split forecasts with explicit dates, grid, identity and temperature units."""

    def __init__(self, path: Path, context: AnalysisContext, split: str) -> None:
        self.path, self.context = Path(path), context
        self.index = _expected_index(context, split)
        self.handle = None
        try:
            self.metadata, self.predictions = self._open()
            self._validate()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.close()
            raise DataValidationError("Cannot read/validate forecast %s %s: %s" %
                                      (self.path.suffix.upper(), self.path, error)) from error
        except Exception:
            self.close()
            raise

    def _open(self) -> tuple:
        if self.path.suffix.lower() == ".npz":
            self.handle = np.load(self.path, allow_pickle=False)
            if not FORECAST_KEYS.issubset(self.handle.files):
                raise DataValidationError("Forecast NPZ is missing: %s" %
                                          sorted(FORECAST_KEYS - set(self.handle.files)))
            metadata = {name: self.handle[name] for name in FORECAST_KEYS - {"predictions"}}
            return metadata, self.handle["predictions"]
        if self.path.suffix.lower() == ".nc":
            return self._open_netcdf()
        raise DataValidationError("Forecast files must use .npz or .nc")

    def _open_netcdf(self) -> tuple:
        try:
            self.handle = xr.open_dataset(self.path)
        except (ValueError, OSError) as error:
            raise DataValidationError("Cannot open/decode forecast NetCDF %s: %s" % (self.path, error))
        expected_dims = ("origin", "lead", "latitude", "longitude")
        if "predictions" not in self.handle or self.handle.predictions.dims != expected_dims:
            raise DataValidationError("NetCDF predictions dimensions must be %s" % (expected_dims,))
        if "target_dates" not in self.handle or self.handle.target_dates.dims != ("origin", "lead"):
            raise DataValidationError("NetCDF target_dates must have dimensions (origin, lead)")
        if not np.array_equal(self.handle.lead.values, np.arange(1, 4)):
            raise DataValidationError("NetCDF lead coordinate must be [1, 2, 3] days")
        metadata = {name: self.handle[name].values for name in ("latitude", "longitude", "target_dates")}
        metadata["origin_dates"] = self.handle.origin.values
        metadata["units"] = np.asarray(self.handle.predictions.attrs.get("units", ""))
        metadata["run_identity"] = np.asarray(self.handle.attrs.get("run_identity", ""))
        return metadata, self.handle.predictions

    def _validate(self) -> None:
        self.units = _string_scalar(self.metadata["units"], "units")
        if self.units not in ALLOWED_UNITS:
            raise DataValidationError("Forecast units must be one of %s" % sorted(ALLOWED_UNITS))
        identity = _string_scalar(self.metadata["run_identity"], "run_identity")
        if identity != self.context.run_identity:
            raise DataValidationError("Forecasts belong to a different preprocessing run")
        _validate_grid(self.metadata, self.context.grid)
        self.origin_dates = _date_array(self.metadata["origin_dates"], "origin_dates")
        self.target_dates = _date_array(self.metadata["target_dates"], "target_dates")
        self._validate_dates_and_shape()

    def _validate_dates_and_shape(self) -> None:
        expected_origins = _date_array(self.index.origin_date.to_numpy(), "origin_dates")
        expected_targets = np.stack([_date_array(self.index["target_%d_date" % lead].to_numpy(),
                                                "target_dates") for lead in range(1, 4)], axis=1)
        if not np.array_equal(self.origin_dates, expected_origins):
            raise DataValidationError("Forecast origins must exactly match the ordered, quality-filtered split")
        if not np.array_equal(self.target_dates, expected_targets):
            raise DataValidationError("Forecast target dates are inconsistent with lead days")
        shape = tuple(self.predictions.shape)
        spatial = tuple(self.context.grid["ocean_mask"].shape)
        allowed = ((len(self.index), 3) + spatial, (len(self.index), 3, 1) + spatial)
        if shape not in allowed:
            raise DataValidationError("Forecast array shape %s does not match %s" % (shape, allowed))
        if not (np.issubdtype(self.predictions.dtype, np.floating) or
                np.issubdtype(self.predictions.dtype, np.integer)):
            raise DataValidationError("Forecast predictions must be real numeric temperatures")

    def read(self, selection: slice) -> np.ndarray:
        """Return only the requested batch, converted to physical degrees Celsius."""
        values = self.predictions.isel(origin=selection).values if isinstance(
            self.predictions, xr.DataArray) else self.predictions[selection]
        values = np.asarray(values, dtype=np.float64)
        if self.units == "K":
            return values - KELVIN_OFFSET
        if self.units == "sst_zscore":
            info = self.context.statistics["variables"]["sst"]
            return denormalize_values(values, info["mean"], info["std"])
        return values

    def close(self) -> None:
        """Side effect: release file handles, including validation-failure paths."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "ForecastFile":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def _validate_grid(metadata: dict, grid: dict) -> None:
    for name in ("latitude", "longitude"):
        actual = np.asarray(metadata[name])
        expected = np.asarray(grid[name])
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise DataValidationError("Forecast %s differs from the prepared grid" % name)
