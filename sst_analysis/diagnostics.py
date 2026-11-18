"""Stream observed training fields into fixed-domain regional climatologies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from sst_preprocessing.artifacts import write_json_atomic
from sst_preprocessing.constants import VARIABLE_UNITS
from sst_preprocessing.errors import DataValidationError

from .context import AnalysisContext


SEASONS = ("annual", "DJF", "MAM", "JJA", "SON")
CHUNK_DAYS = 7
SOURCE_VARIABLES = ("sst", "uo", "vo", "gradient_magnitude", "net_heat_flux",
                    "mixed_layer_depth", "heat_increment")
SOURCE_METRICS = {"mean_sst": "sst", "mean_uo": "uo", "mean_vo": "vo",
                  "mean_gradient": "gradient_magnitude", "mean_net_heat_flux": "net_heat_flux",
                  "mean_mixed_layer_depth": "mixed_layer_depth", "mean_heat_increment": "heat_increment"}
METRIC_UNITS = {"mean_sst": "degree_Celsius", "std_sst": "degree_Celsius",
                "mean_uo": "m s-1", "mean_vo": "m s-1", "mean_current_speed": "m s-1",
                "mean_gradient": "degree_Celsius km-1", "front_frequency": "1",
                "mean_net_heat_flux": "W m-2", "mean_mixed_layer_depth": "m",
                "mean_heat_increment": "degree_Celsius day-1",
                "mean_abs_heat_increment": "degree_Celsius day-1",
                "mean_abs_sst_change": "degree_Celsius day-1"}
ACCUMULATED_METRICS = tuple(name for name in METRIC_UNITS if name != "std_sst")
GRADIENT_METRICS = ("mean_gradient", "front_frequency")


@dataclass
class _Accumulator:
    sums: Dict[str, np.ndarray]
    counts: Dict[str, np.ndarray]
    sst_square_sum: np.ndarray


def run_diagnostics(context: AnalysisContext, output_root: Path) -> dict:
    """Write training-only diagnostic artifacts; never form or evaluate forecasts.

    Side effect: create/atomically replace four artifacts inside ``output_root``.
    Only training-month files are opened, and physical variables load in <=7-day chunks.
    """
    output_root = Path(output_root).resolve()
    prepared_root = context.root.resolve()
    is_overlapping_root = (output_root == prepared_root or prepared_root in output_root.parents
                           or output_root in prepared_root.parents)
    if is_overlapping_root:
        raise DataValidationError("Diagnostics output must be separate from preprocessing artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    source_files = _training_files(context)
    accumulator = _new_accumulator(context)
    rows = _process_training(context, source_files, accumulator)
    daily = pd.DataFrame(rows)
    maps = _build_maps(context, accumulator)
    summary = _build_summary(context, {"files": source_files, "daily": daily, "maps": maps})
    _write_netcdf_atomic(output_root / "seasonal_maps.nc", maps)
    _write_csv_atomic(output_root / "daily_regional.csv", daily)
    _write_csv_atomic(output_root / "monthly_cycle.csv", _monthly_cycle(daily))
    write_json_atomic(output_root / "diagnostics_summary.json", summary)
    return summary


def _training_files(context: AnalysisContext) -> list:
    start, end = pd.to_datetime(context.train_period)
    if start > end or start != start.normalize() or end != end.normalize():
        raise DataValidationError("Training period must contain ordered whole UTC days")
    paths = [context.root / "physical" / (str(month) + ".nc")
             for month in pd.period_range(start, end, freq="M")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DataValidationError("Missing training physical files: %s" % missing)
    return paths


def _new_accumulator(context: AnalysisContext) -> _Accumulator:
    shape = (len(SEASONS),) + np.asarray(context.grid["core_mask"]).shape
    return _Accumulator(
        sums={name: np.zeros(shape, np.float64) for name in ACCUMULATED_METRICS},
        counts={name: np.zeros(shape, np.int32) for name in ACCUMULATED_METRICS},
        sst_square_sum=np.zeros(shape, np.float64))


def _process_training(context: AnalysisContext, paths: list, accumulator: _Accumulator) -> list:
    rows, previous_sst = [], None
    for path in paths:
        for dates, source in _monthly_chunks(context, path):
            _check_next_dates(context, rows, dates)
            fields, previous_sst = _derive_fields(source, previous_sst, context.front_threshold)
            for offset, date in enumerate(dates):
                daily_fields = {name: values[offset] for name, values in fields.items()}
                _accumulate_day(accumulator, daily_fields, (date, context))
                rows.append(_regional_row(context, date, daily_fields))
    expected_count = len(pd.date_range(*context.train_period, freq="D"))
    if len(rows) != expected_count:
        raise DataValidationError("Training diagnostics read %d days; expected %d" % (len(rows), expected_count))
    return rows


def _monthly_chunks(context: AnalysisContext, path: Path) -> Iterator[Tuple[pd.DatetimeIndex, dict]]:
    with xr.open_dataset(path) as dataset:
        _validate_source(context, path, dataset)
        dates = pd.DatetimeIndex(dataset.time.values)
        start, end = pd.to_datetime(context.train_period)
        selected = np.flatnonzero((dates >= start) & (dates <= end))
        for offset in range(0, len(selected), CHUNK_DAYS):
            indices = selected[offset:offset + CHUNK_DAYS]
            source = {name: dataset[name].isel(time=indices).values.astype(np.float64)
                      for name in SOURCE_VARIABLES}
            yield dates[indices], source


def _validate_source(context: AnalysisContext, path: Path, dataset: xr.Dataset) -> None:
    missing = set(SOURCE_VARIABLES) - set(dataset.data_vars)
    if missing:
        raise DataValidationError("%s lacks physical fields %s" % (path, sorted(missing)))
    if "time" not in dataset.coords:
        raise DataValidationError("%s has no time coordinate" % path)
    dates = pd.DatetimeIndex(dataset.time.values)
    is_valid_dates = (not dates.hasnans and not dates.has_duplicates and dates.is_monotonic_increasing
                      and np.array_equal(dates.values, dates.normalize().values))
    if not is_valid_dates or any(str(date.to_period("M")) != path.stem for date in dates):
        raise DataValidationError("%s has duplicate, unordered or unexpected daily dates" % path)
    for coordinate in ("latitude", "longitude"):
        if coordinate not in dataset.coords or not np.array_equal(dataset[coordinate], context.grid[coordinate]):
            raise DataValidationError("%s %s differs from the fixed preprocessing grid" % (path, coordinate))
    _validate_source_variables(path, dataset)
    if dataset.attrs.get("run_identity") != context.run_identity:
        raise DataValidationError("%s belongs to another preprocessing run" % path)


def _validate_source_variables(path: Path, dataset: xr.Dataset) -> None:
    for name in SOURCE_VARIABLES:
        if dataset[name].dims != ("time", "latitude", "longitude"):
            raise DataValidationError("%s variable %s must have time/latitude/longitude dimensions" % (path, name))
        actual_units = dataset[name].attrs.get("units")
        if actual_units != VARIABLE_UNITS[name]:
            raise DataValidationError("%s variable %s has units %r; expected %r" %
                                      (path, name, actual_units, VARIABLE_UNITS[name]))


def _check_next_dates(context: AnalysisContext, rows: list, dates: pd.DatetimeIndex) -> None:
    expected_start = pd.Timestamp(rows[-1]["date"]) + pd.Timedelta(days=1) if rows else pd.Timestamp(context.train_period[0])
    expected = pd.date_range(expected_start, periods=len(dates), freq="D")
    if not dates.equals(expected):
        raise DataValidationError("Training daily sequence is incomplete near %s" % expected_start.date())


def _derive_fields(source: dict, previous_sst: np.ndarray, threshold: float) -> Tuple[dict, np.ndarray]:
    fields = {name: source[variable] for name, variable in SOURCE_METRICS.items()}
    fields["mean_current_speed"] = np.hypot(source["uo"], source["vo"])
    fields["mean_abs_heat_increment"] = np.abs(source["heat_increment"])
    gradient = source["gradient_magnitude"]
    fields["front_frequency"] = np.where(np.isfinite(gradient), gradient >= threshold, np.nan)
    change = np.full_like(source["sst"], np.nan)
    change[1:] = np.abs(np.diff(source["sst"], axis=0))
    if previous_sst is not None:
        change[0] = np.abs(source["sst"][0] - previous_sst)
    fields["mean_abs_sst_change"] = change
    return fields, source["sst"][-1].copy()


def _field_mask(context: AnalysisContext, name: str) -> np.ndarray:
    domain = "gradient_mask" if name in GRADIENT_METRICS else "ocean_mask"
    return np.asarray(context.grid["core_mask"], bool) & np.asarray(context.grid[domain], bool)


def _season(date: pd.Timestamp) -> str:
    if date.month in (12, 1, 2):
        return "DJF"
    if date.month in (3, 4, 5):
        return "MAM"
    if date.month in (6, 7, 8):
        return "JJA"
    return "SON"


def _accumulate_day(accumulator: _Accumulator, fields: dict, details: tuple) -> None:
    date, context = details
    season_indices = (0, SEASONS.index(_season(date)))
    for name, values in fields.items():
        is_valid = _field_mask(context, name) & np.isfinite(values)
        valid_values = np.where(is_valid, values, 0.0)
        for season_index in season_indices:
            accumulator.sums[name][season_index] += valid_values
            accumulator.counts[name][season_index] += is_valid
            if name == "mean_sst":
                accumulator.sst_square_sum[season_index] += valid_values * valid_values


def _regional_row(context: AnalysisContext, date: pd.Timestamp, fields: dict) -> dict:
    row = {"date": date.strftime("%Y-%m-%d"), "season": _season(date)}
    weights = np.asarray(context.grid["area_weights"], np.float64)
    for name in ACCUMULATED_METRICS:
        output_name = "strong_front_area_fraction" if name == "front_frequency" else name
        values = fields[name]
        is_valid = _field_mask(context, name) & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        weight_sum = weights[is_valid].sum()
        row[output_name] = float(np.sum(values[is_valid] * weights[is_valid]) / weight_sum) if weight_sum > 0 else np.nan
        row[output_name + "_valid_count"] = int(np.count_nonzero(is_valid))
    return row


def _safe_mean(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def _build_maps(context: AnalysisContext, accumulator: _Accumulator) -> xr.Dataset:
    dimensions = ("season", "latitude", "longitude")
    variables = {}
    for name in ACCUMULATED_METRICS:
        values = _safe_mean(accumulator.sums[name], accumulator.counts[name])
        variables[name] = (dimensions, values.astype(np.float32), _map_attributes(name))
        variables[name + "_count"] = (dimensions, accumulator.counts[name], {"units": "1"})
    square_mean = _safe_mean(accumulator.sst_square_sum, accumulator.counts["mean_sst"])
    variance = np.maximum(square_mean - _safe_mean(accumulator.sums["mean_sst"], accumulator.counts["mean_sst"]) ** 2, 0)
    variables["std_sst"] = (dimensions, np.sqrt(variance).astype(np.float32), _map_attributes("std_sst"))
    variables["std_sst_count"] = (dimensions, accumulator.counts["mean_sst"], {"units": "1"})
    return xr.Dataset(variables, coords={"season": list(SEASONS), "latitude": context.grid["latitude"],
                                       "longitude": context.grid["longitude"]},
                      attrs={"run_identity": context.run_identity, "training_start": context.train_period[0],
                             "training_end": context.train_period[1], "std_sst_ddof": 0,
                             "season_definition": "Calendar-month climatology: DJF/MAM/JJA/SON; no complete-season filtering",
                             "spatial_domain": "Core and fixed ocean mask; gradient fields use the gradient template"})


def _map_attributes(name: str) -> dict:
    mask = "core_mask AND gradient_mask" if name in GRADIENT_METRICS else "core_mask AND ocean_mask"
    return {"units": METRIC_UNITS[name], "valid_domain": mask,
            "temporal_aggregation": "Valid daily observations per cell; population standard deviation for std_sst"}


def _monthly_cycle(daily: pd.DataFrame) -> pd.DataFrame:
    metrics = [name for name in daily.columns if name not in ("date", "season") and not name.endswith("_valid_count")]
    months = pd.to_datetime(daily["date"]).dt.month
    rows = []
    for month in range(1, 13):
        selected = daily.loc[months == month]
        row = {"month": month, "day_count": len(selected)}
        for name in metrics:
            row[name] = selected[name].mean()
            row[name + "_day_count"] = int(selected[name].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary(context: AnalysisContext, results: dict) -> dict:
    files, daily, maps = results["files"], results["daily"], results["maps"]
    return {"status": "complete", "run_identity": context.run_identity,
            "train_period": list(context.train_period), "training_day_count": len(daily),
            "source_files": [str(path.relative_to(context.root)).replace("\\", "/") for path in files],
            "front_threshold": context.front_threshold, "front_threshold_units": "degree_Celsius km-1",
            "units": dict(METRIC_UNITS, strong_front_area_fraction="1"),
            "coverage": _coverage(maps), "methods": _methods(),
            "artifacts": {"seasonal_maps": "seasonal_maps.nc", "daily_regional": "daily_regional.csv",
                          "monthly_cycle": "monthly_cycle.csv", "summary": "diagnostics_summary.json"}}


def _coverage(maps: xr.Dataset) -> dict:
    coverage = {}
    for name in METRIC_UNITS:
        counts = maps[name + "_count"].sel(season="annual").values
        supported = counts[counts > 0]
        coverage[name] = {"valid_grid_observations": int(counts.sum()),
                          "cells_with_observations": int(supported.size),
                          "minimum_days_per_observed_cell": int(supported.min()) if supported.size else None,
                          "maximum_days_per_observed_cell": int(supported.max()) if supported.size else None}
    return coverage


def _methods() -> dict:
    return {"data_scope": "Observed concurrent physical fields; training dates only; no forecasts or baseline models",
            "chunk_days": CHUNK_DAYS, "season_order": list(SEASONS),
            "season_definition": "Calendar-month climatology; DJF is austral summer. All training months are retained, including boundary Dec/Jan/Feb; no complete-season filtering.",
            "map_aggregation": "Equal valid daily observations per cell; SST population standard deviation (ddof=0).",
            "regional_aggregation": "Area-weighted valid core-ocean cells per day, with a separate valid-weight denominator for each field. Gradient fields use core AND gradient_mask.",
            "monthly_cycle": "Equal-day mean of valid daily regional statistics; each metric carries its own valid day count. Empty months are undefined.",
            "sst_change": "Absolute difference of consecutive chronological training days, attributed to the later date. First training day excluded; month/chunk boundaries retained.",
            "front_frequency": "Fixed training P90 threshold with >= comparison; denominator is valid gradient observations. No spatial smoothing or component filtering.",
            "heat_convention": "Positive into ocean; original net surface heat and 3 m effective MLD floor; no penetration correction or damping.",
            "missing_values": "NetCDF/CSV undefined values are NaN/empty; JSON undefined statistics are null."}


def _write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(str(temporary), str(path))


def _write_netcdf_atomic(path: Path, dataset: xr.Dataset) -> None:
    temporary = path.with_name(path.name + ".tmp")
    encoding = {}
    for name, variable in dataset.data_vars.items():
        settings = {"zlib": True, "complevel": 2}
        settings.update({"dtype": "int32"} if np.issubdtype(variable.dtype, np.integer)
                        else {"dtype": "float32", "_FillValue": np.float32(np.nan)})
        encoding[name] = settings
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    os.replace(str(temporary), str(path))
