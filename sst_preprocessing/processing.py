"""Build physical monthly fields while retaining original UTC date meanings."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

from .config import PreprocessConfig
from .constants import ENERGY_VARIABLES, KELVIN_OFFSET, VARIABLE_UNITS
from .errors import DataValidationError
from .grid import GridInterpolator, compute_temperature_gradients
from .physics import compute_heat_features, fill_mixed_layer_depth


def _interpolator(source: xr.DataArray, target: xr.DataArray,
                  config: PreprocessConfig) -> GridInterpolator:
    source_grid = {"latitude": source.latitude.values, "longitude": source.longitude.values}
    target_grid = {"latitude": target.latitude.values, "longitude": target.longitude.values}
    return GridInterpolator(source_grid, target_grid,
                            minimum_weight=config.minimum_interpolation_weight)


def _check_source_dates(sources: dict) -> None:
    expected = pd.DatetimeIndex(sources["sst"].time.values)
    for name, source in sources.items():
        actual = pd.DatetimeIndex(source.time.values)
        if not actual.equals(expected):
            raise DataValidationError("Source timestamps differ within physical month: %s" % name)


def _regrid_ocean(sources: dict, config: PreprocessConfig, climatology: dict) -> tuple:
    target = sources["sst"]
    ocean_interpolator = _interpolator(sources["uo"], target, config)
    month = int(pd.Timestamp(target.time.values[0]).month)
    depth, was_filled = fill_mixed_layer_depth(sources["mlotst"].values, month, climatology)
    fields = {}
    minimum_weight = np.ones_like(target.values, dtype=np.float32)
    for name in ("uo", "vo"):
        fields[name], weight = ocean_interpolator.interpolate(sources[name].values)
        minimum_weight = np.minimum(minimum_weight, weight)
    fields["mixed_layer_depth"], weight = ocean_interpolator.interpolate(depth)
    minimum_weight = np.minimum(minimum_weight, weight)
    fill_fraction, _ = ocean_interpolator.interpolate(
        np.where(climatology["ocean_mask"], was_filled.astype(np.float32), np.nan))
    fields["minimum_ocean_interpolation_weight"] = minimum_weight
    return fields, np.isfinite(fill_fraction) & (fill_fraction > 0)


def _regrid_air(sources: dict, config: PreprocessConfig) -> dict:
    target = sources["sst"]
    interpolator = _interpolator(sources["era5"]["u10"], target, config)
    fields = {}
    minimum_weight = np.ones_like(target.values, dtype=np.float32)
    for name in ("u10", "v10", "t2m", "msl") + ENERGY_VARIABLES:
        fields[name], weight = interpolator.interpolate(sources["era5"][name].values)
        minimum_weight = np.minimum(minimum_weight, weight)
    fields["minimum_air_interpolation_weight"] = minimum_weight
    return fields


def build_physical_chunk(sources: dict, config: PreprocessConfig,
                         climatology: dict) -> xr.Dataset:
    """Return physical-unit arrays; do not shift daily labels or normalize here."""
    _check_source_dates(sources)
    target = sources["sst"]
    fields, was_filled = _regrid_ocean(sources, config, climatology)
    fields.update(_regrid_air(sources, config))
    fields["sst"] = target.values.astype(np.float32) - np.float32(KELVIN_OFFSET)
    fields["t2m"] = fields["t2m"] - np.float32(KELVIN_OFFSET)
    gradients = compute_temperature_gradients(
        fields["sst"], target.latitude.values, target.longitude.values)
    for name, values in zip(("gradient_x", "gradient_y", "gradient_magnitude"), gradients[:3]):
        fields[name] = values
    fields.update(compute_heat_features(
        {name: fields[name] for name in ENERGY_VARIABLES}, fields["mixed_layer_depth"], config))
    dimensions = ("time", "latitude", "longitude")
    variables = {name: (dimensions, values.astype(np.float32)) for name, values in fields.items()}
    variables["mld_was_filled"] = (dimensions, was_filled.astype(np.uint8))
    dataset = xr.Dataset(variables, coords={name: target[name] for name in dimensions})
    for name in fields:
        dataset[name].attrs["units"] = VARIABLE_UNITS[name]
    dataset.attrs.update(time_zone="UTC", environment_lag_applied="false; applied by sample index",
                         heat_convention="positive into the ocean",
                         heat_model="net surface energy; no shortwave penetration; 3 m default floor")
    return dataset


def build_physical_month(sources: dict, config: PreprocessConfig,
                         climatology: dict) -> xr.Dataset:
    """Limit interpolation temporaries to configured daily chunks within one month."""
    chunks = []
    for start in range(0, sources["sst"].sizes["time"], config.chunk_days):
        selection = slice(start, start + config.chunk_days)
        chunk_sources = {name: source.isel(time=selection) for name, source in sources.items()}
        chunks.append(build_physical_chunk(chunk_sources, config, climatology))
    return xr.concat(chunks, dim="time", combine_attrs="override")
