"""Inspect immutable NetCDF inputs and read UTC months without downloader imports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import PreprocessConfig
from .constants import ENERGY_VARIABLES, SCHEMA_VERSION
from .errors import DataValidationError

SOURCE_VARIABLES = ("sst", "uo", "vo", "mlotst", "era5")
ERA5_VARIABLES = ("u10", "v10", "t2m", "msl") + ENERGY_VARIABLES
SOURCE_NAMES = {"sst": ("analysed_sst",), "uo": ("uo",), "vo": ("vo",),
                "mlotst": ("mlotst",), "era5": ERA5_VARIABLES}
SUPPORTED_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
HASH_CHUNK_BYTES = 8 * 1024 * 1024
SURFACE_DEPTH_LIMIT_METERS = 1.0


@dataclass(frozen=True)
class _SourceSelection:
    variable: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class _TrainingPeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    months: pd.DatetimeIndex


@dataclass(frozen=True)
class _CalendarMonthSelection:
    period: _TrainingPeriod
    calendar_month: int


@dataclass
class _ClimatologyAccumulator:
    ocean_mask: np.ndarray
    total: np.ndarray
    count: np.ndarray


def build_inventory(config: PreprocessConfig) -> dict:
    """Hash and validate selected source files; never import or execute downloaders."""
    data_root = Path(config.data_root).resolve()
    if not data_root.is_dir():
        raise DataValidationError("Raw data directory does not exist: %s" % data_root)
    records, excluded = _inspect_candidates(config, data_root)
    _validate_inventory_coverage(config, records)
    _validate_shared_grids(records)
    identity = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": SCHEMA_VERSION, "data_root": str(data_root),
        "start_date": config.start_date, "end_date": config.end_date,
        "files": records, "excluded_files": excluded,
        "fingerprint": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "date_basis": "Decoded daily UTC time coordinate; provider coverage attributes may be stale.",
        "energy_basis": "ERA5 inputs are daily accumulated J m-2; no further time aggregation.",
    }


def _inspect_candidates(config: PreprocessConfig, data_root: Path) -> Tuple[list, list]:
    records, excluded = [], []
    for path in sorted(data_root.rglob("*.nc")):
        variable, reason = _classify_source(path)
        if variable is None:
            excluded.append({"relative_path": path.relative_to(data_root).as_posix(), "reason": reason})
            continue
        if not _filename_overlaps(path, variable, config):
            excluded.append({"relative_path": path.relative_to(data_root).as_posix(),
                             "reason": "Outside configured date coverage"})
            continue
        records.append(_inspect_file(path=path, variable=variable, config=config))
    return records, excluded


def _classify_source(path: Path) -> Tuple[Optional[str], str]:
    filename = path.name
    if filename == "current_202301.nc":
        return None, "Duplicate/download-test current file; canonical yearly uo/vo files are selected"
    if filename.startswith("METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_"):
        return "sst", ""
    for variable in ("uo", "vo", "mlotst"):
        prefix = "cmems_mod_glo_phy_my_0.083deg_P1D-m_%s_" % variable
        if filename.startswith(prefix):
            return variable, ""
    if re.fullmatch(r"ERA5_\d{4}_\d{2}\.nc", filename):
        return "era5", ""
    return None, "Unrecognized NetCDF filename; not part of the selected input contract"


def _filename_overlaps(path: Path, variable: str, config: PreprocessConfig) -> bool:
    if variable == "era5":
        match = re.fullmatch(r"ERA5_(\d{4})_(\d{2})\.nc", path.name)
        start = pd.Timestamp("%s-%s-01" % match.groups())
        end = start + pd.offsets.MonthEnd(0)
    else:
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})T\d{2}-\d{2}-\d{2}", path.name)
        if len(dates) != 2:
            raise DataValidationError("Canonical source filename needs start/end dates: %s" % path)
        start, end = map(pd.Timestamp, dates)
        if start.year != end.year:
            raise DataValidationError("Expected a single yearly source file: %s" % path)
    return start <= pd.Timestamp(config.end_date) and end >= pd.Timestamp(config.start_date)


def _open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, decode_times=True, mask_and_scale=True, cache=False)
    except (OSError, ValueError, TypeError) as error:
        raise DataValidationError("Cannot decode source NetCDF %s: %s" % (path, error)) from error


def _inspect_file(*, path: Path, variable: str, config: PreprocessConfig) -> dict:
    with _open_dataset(path) as raw:
        dataset = _normalize_dataset(raw, path)
        dates = _validate_dates(dataset, path)
        _validate_source_fields(dataset, variable, path)
        _validate_filename_dates(path, variable, dates)
        grid = _grid_description(dataset)
        if variable == "sst" and config.expected_grid_shape is not None:
            if tuple(grid["shape"]) != tuple(config.expected_grid_shape):
                raise DataValidationError("OSTIA grid shape %s differs from expected %s: %s" %
                                          (grid["shape"], config.expected_grid_shape, path))
        metadata = _source_metadata(dataset, variable)
    before = path.stat()
    checksum = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise DataValidationError("Source changed while computing its identity: %s" % path)
    return {"path": str(path.resolve()), "relative_path": path.relative_to(Path(config.data_root).resolve()).as_posix(),
            "variable": variable, "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date": dates[-1].strftime("%Y-%m-%d"), "days": len(dates),
            "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": checksum, "grid": grid, **metadata}


def _normalize_dataset(dataset: xr.Dataset, path: Path) -> xr.Dataset:
    aliases = {name: canonical for name, canonical in
               (("lat", "latitude"), ("lon", "longitude"), ("valid_time", "time"))
               if name in dataset.dims and canonical not in dataset.dims}
    if aliases:
        dataset = dataset.rename(aliases)
    for coordinate in ("latitude", "longitude", "time"):
        if coordinate not in dataset.coords or dataset[coordinate].dims != (coordinate,):
            raise DataValidationError("Missing one-dimensional %s coordinate: %s" % (coordinate, path))
    longitude_attrs = dict(dataset.longitude.attrs)
    longitude = (dataset.longitude.values.astype(np.float64) + 180.0) % 360.0 - 180.0
    dataset = dataset.assign_coords(longitude=("longitude", longitude, longitude_attrs))
    dataset = dataset.sortby("latitude").sortby("longitude")
    _validate_horizontal_coordinates(dataset, path)
    if "depth" in dataset.dims:
        _validate_surface_depth(dataset, path)
        dataset = dataset.isel(depth=0, drop=True)
    return dataset


def _validate_horizontal_coordinates(dataset: xr.Dataset, path: Path) -> None:
    for name, limit, units in (("latitude", 90.0, "degrees_north"),
                               ("longitude", 180.0, "degrees_east")):
        values = dataset[name].values
        if len(values) < 2 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
            raise DataValidationError("%s must contain distinct finite grid coordinates: %s" % (name, path))
        if np.any(np.abs(values) > limit):
            raise DataValidationError("%s outside its geographic range: %s" % (name, path))
        actual_units = dataset[name].attrs.get("units", "").lower()
        if actual_units != units:
            raise DataValidationError("Unexpected %s units %r in %s" % (name, actual_units, path))


def _validate_surface_depth(dataset: xr.Dataset, path: Path) -> None:
    if dataset.sizes["depth"] != 1:
        raise DataValidationError("Only a single surface-current depth is supported: %s" % path)
    depth = float(dataset.depth.values[0])
    if not np.isfinite(depth) or not 0 <= depth <= SURFACE_DEPTH_LIMIT_METERS:
        raise DataValidationError("Expected surface-current depth within 0..1 m; got %s: %s" % (depth, path))
    if dataset.depth.attrs.get("units", "").lower() not in ("m", "meter", "metre"):
        raise DataValidationError("Expected current depth in meters: %s" % path)


def _validate_dates(dataset: xr.Dataset, path: Path) -> pd.DatetimeIndex:
    calendar = dataset.time.encoding.get("calendar", dataset.time.attrs.get("calendar", "standard"))
    if calendar not in SUPPORTED_CALENDARS:
        raise DataValidationError("Unsupported calendar %r in %s" % (calendar, path))
    values = dataset.time.values
    if not np.issubdtype(values.dtype, np.datetime64):
        raise DataValidationError("Expected Gregorian UTC datetime64 dates: %s" % path)
    dates = pd.DatetimeIndex(values)
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise DataValidationError("Time axis is empty, duplicated, missing, or unordered: %s" % path)
    if not dates.equals(dates.normalize()):
        raise DataValidationError("Daily inputs must use midnight UTC date labels: %s" % path)
    expected = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected):
        raise DataValidationError("Missing day in source time axis: %s" % path)
    timezone = str(dataset.attrs.get("time_zone", "UTC")).upper()
    if timezone != "UTC":
        raise DataValidationError("Only UTC daily source dates are supported: %s" % path)
    return dates


def _normalized_unit(value: str) -> str:
    return str(value).lower().replace("**", "").replace("^", "").replace(" ", "")


def _validate_source_fields(dataset: xr.Dataset, variable: str, path: Path) -> None:
    for name in SOURCE_NAMES[variable]:
        if name not in dataset.data_vars:
            raise DataValidationError("Required variable %s missing: %s" % (name, path))
        if set(dataset[name].dims) != {"time", "latitude", "longitude"}:
            raise DataValidationError("Unexpected dimensions for %s: %s" % (name, path))
        units = _normalized_unit(dataset[name].attrs.get("units", ""))
        if units not in _expected_units(name):
            raise DataValidationError("Unsupported units %r for %s in %s" % (units, name, path))
    if variable == "era5":
        processing = str(dataset.attrs.get("processing", "")).lower()
        if "daily" not in processing or "sum" not in processing:
            raise DataValidationError("ERA5 requires explicit daily accumulated-energy provenance: %s" % path)
    if variable == "sst":
        description = " ".join(str(dataset.analysed_sst.attrs.get(key, ""))
                               for key in ("standard_name", "comment", "long_name")).lower()
        if "foundation" not in description:
            raise DataValidationError("OSTIA input must be identified as foundation SST: %s" % path)


def _expected_units(name: str) -> set:
    if name in ("analysed_sst", "t2m"):
        return {"k", "kelvin"}
    if name in ("u10", "v10", "uo", "vo"):
        return {"ms-1", "m/s"}
    if name in ENERGY_VARIABLES:
        return {"jm-2", "j/m2"}
    if name == "mlotst":
        return {"m", "meter", "metre"}
    return {"pa"}


def _validate_filename_dates(path: Path, variable: str, dates: pd.DatetimeIndex) -> None:
    if variable == "era5":
        expected = pd.Timestamp("%s-%s-01" % tuple(path.stem.split("_")[1:]))
        if np.any(dates.to_period("M") != expected.to_period("M")):
            raise DataValidationError("ERA5 filename month disagrees with actual UTC dates: %s" % path)
        return
    labels = re.findall(r"(\d{4}-\d{2}-\d{2})T\d{2}-\d{2}-\d{2}", path.name)
    if pd.Timestamp(labels[0]) != dates[0] or pd.Timestamp(labels[-1]) != dates[-1]:
        raise DataValidationError("Source filename date range disagrees with actual UTC dates: %s" % path)


def _grid_description(dataset: xr.Dataset) -> dict:
    latitude = np.asarray(dataset.latitude.values, dtype="<f8")
    longitude = np.asarray(dataset.longitude.values, dtype="<f8")
    checksum = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
    return {"shape": [len(latitude), len(longitude)], "fingerprint": checksum,
            "latitude_bounds": [float(latitude[0]), float(latitude[-1])],
            "longitude_bounds": [float(longitude[0]), float(longitude[-1])]}


def _source_metadata(dataset: xr.Dataset, variable: str) -> dict:
    return {"units": {name: str(dataset[name].attrs["units"]) for name in SOURCE_NAMES[variable]},
            "standard_names": {name: str(dataset[name].attrs.get("standard_name", ""))
                               for name in SOURCE_NAMES[variable]},
            "calendar": dataset.time.encoding.get("calendar", "standard"),
            "processing": str(dataset.attrs.get("processing", "")),
            "time_zone": str(dataset.attrs.get("time_zone", "UTC"))}


def _sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _validate_inventory_coverage(config: PreprocessConfig, records: List[dict]) -> None:
    expected = pd.date_range(config.start_date, config.end_date, freq="D")
    if expected.empty:
        raise DataValidationError("Configured date range is empty")
    for variable in SOURCE_VARIABLES:
        selected = [record for record in records if record["variable"] == variable]
        dates = []
        for record in selected:
            interval = pd.date_range(max(record["start_date"], config.start_date),
                                     min(record["end_date"], config.end_date), freq="D")
            dates.extend(interval)
        actual = pd.DatetimeIndex(dates)
        if actual.has_duplicates:
            duplicates = actual[actual.duplicated()].strftime("%Y-%m-%d").tolist()[:3]
            raise DataValidationError("Overlapping/duplicate %s source dates: %s" % (variable, duplicates))
        missing = expected.difference(actual)
        if len(missing):
            raise DataValidationError("Missing %s source coverage: %s (%d days)" %
                                      (variable, missing[0].strftime("%Y-%m-%d"), len(missing)))


def _validate_shared_grids(records: List[dict]) -> None:
    fingerprints = {}
    for record in records:
        variable = record["variable"]
        identity = record["grid"]["fingerprint"]
        if variable in fingerprints and fingerprints[variable] != identity:
            raise DataValidationError("Source grid changes between %s files: %s" % (variable, record["path"]))
        fingerprints[variable] = identity
    if len({fingerprints[name] for name in ("uo", "vo", "mlotst")}) != 1:
        raise DataValidationError("uo, vo and MLD must share the same native source grid")


def load_source_month(inventory: dict, month_start: str) -> dict:
    """Load one original-date UTC month, clipping only to inventory coverage."""
    requested = pd.Timestamp(month_start)
    if requested != requested.normalize().replace(day=1):
        raise DataValidationError("month_start must name the first day of a UTC month: %s" % month_start)
    start = max(requested, pd.Timestamp(inventory["start_date"]))
    end = min(requested + pd.offsets.MonthEnd(0), pd.Timestamp(inventory["end_date"]))
    if start > end:
        raise DataValidationError("Requested month is outside input coverage: %s" % month_start)
    return {variable: _load_variable_range(inventory, _SourceSelection(variable, start, end))
            for variable in SOURCE_VARIABLES}


def _load_variable_range(inventory: dict, selection: _SourceSelection):
    variable, start, end = selection.variable, selection.start, selection.end
    parts = []
    for record in inventory["files"]:
        if record["variable"] != variable:
            continue
        if pd.Timestamp(record["start_date"]) > end or pd.Timestamp(record["end_date"]) < start:
            continue
        path = Path(record["path"])
        _verify_source_identity(record, path)
        with _open_dataset(path) as raw:
            dataset = _normalize_dataset(raw, path)
            selected = dataset[list(SOURCE_NAMES[variable])].sel(time=slice(start, end))
            parts.append(selected.transpose("time", "latitude", "longitude").load())
    if not parts:
        raise DataValidationError("No %s input for %s..%s" % (variable, start.date(), end.date()))
    combined = xr.concat(parts, dim="time").sortby("time") if len(parts) > 1 else parts[0]
    _validate_loaded_range(combined, selection)
    if variable == "era5":
        return combined
    result = combined[SOURCE_NAMES[variable][0]]
    return result.rename("sst") if variable == "sst" else result


def _verify_source_identity(record: dict, path: Path) -> None:
    try:
        status = path.stat()
    except OSError as error:
        raise DataValidationError("Source no longer available: %s" % path) from error
    if (status.st_size, status.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
        raise DataValidationError("Source differs from hashed input inventory; rebuild inventory: %s" % path)


def _validate_loaded_range(dataset: xr.Dataset, selection: _SourceSelection) -> None:
    expected = pd.date_range(selection.start, selection.end, freq="D")
    actual = pd.DatetimeIndex(dataset.time.values)
    if not actual.equals(expected):
        raise DataValidationError("Loaded %s dates do not exactly cover %s..%s" %
                                  (selection.variable, selection.start, selection.end))


def fit_mld_climatology(config: PreprocessConfig, inventory: dict) -> dict:
    """Fit native-ocean MLD statistics from positive training samples only."""
    start, end = map(pd.Timestamp, config.splits["train"])
    months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    period = _TrainingPeriod(start, end, months)
    ocean_mask, latitude, longitude = _training_ocean_grid(inventory, period)
    monthly_sum = np.zeros((12,) + ocean_mask.shape, dtype=np.float64)
    monthly_count = np.zeros(monthly_sum.shape, dtype=np.int64)
    monthly_median = np.full(12, np.nan, dtype=np.float32)
    accumulator = _ClimatologyAccumulator(ocean_mask, monthly_sum, monthly_count)
    for calendar_month in range(1, 13):
        selection = _CalendarMonthSelection(period, calendar_month)
        samples = _accumulate_calendar_month(inventory, selection, accumulator)
        if samples:
            monthly_median[calendar_month - 1] = np.median(np.concatenate(samples))
    annual_mean = _positive_mean(monthly_sum.sum(axis=0), monthly_count.sum(axis=0))
    marine_values = annual_mean[ocean_mask & np.isfinite(annual_mean)]
    if not marine_values.size:
        raise DataValidationError("Training ocean grid has no positive finite MLD samples")
    return {"monthly_mean": _positive_mean(monthly_sum, monthly_count),
            "monthly_median": monthly_median, "ocean_mask": ocean_mask,
            "annual_mean": annual_mean, "annual_median": float(np.median(marine_values)),
            "latitude": latitude, "longitude": longitude,
            "train_period": np.asarray(config.splits["train"]),
            "median_definition": "monthly: all positive marine training samples; annual: annual-mean marine grid"}


def _training_ocean_grid(inventory: dict, period: _TrainingPeriod) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ocean_mask, latitude, longitude = None, None, None
    for month in period.months:
        start = max(month, period.start)
        end = min(month + pd.offsets.MonthEnd(0), period.end)
        eastward = _load_variable_range(inventory, _SourceSelection("uo", start, end))
        northward = _load_variable_range(inventory, _SourceSelection("vo", start, end))
        is_ocean = np.any(np.isfinite(eastward.values) & np.isfinite(northward.values), axis=0)
        ocean_mask = is_ocean if ocean_mask is None else ocean_mask | is_ocean
        latitude, longitude = eastward.latitude.values.copy(), eastward.longitude.values.copy()
    if ocean_mask is None or not ocean_mask.any():
        raise DataValidationError("Training surface currents contain no jointly finite source ocean cells")
    return ocean_mask, latitude, longitude


def _accumulate_calendar_month(inventory: dict, selection: _CalendarMonthSelection,
                               accumulator: _ClimatologyAccumulator) -> list:
    period, calendar_month = selection.period, selection.calendar_month
    samples = []
    for month in period.months[period.months.month == calendar_month]:
        start = max(month, period.start)
        end = min(month + pd.offsets.MonthEnd(0), period.end)
        field = _load_variable_range(inventory, _SourceSelection("mlotst", start, end))
        values = np.asarray(field.values, dtype=np.float32)
        is_valid = np.isfinite(values) & (values > 0) & accumulator.ocean_mask[None, :, :]
        accumulator.total[calendar_month - 1] += np.where(is_valid, values, 0).sum(axis=0, dtype=np.float64)
        accumulator.count[calendar_month - 1] += is_valid.sum(axis=0)
        if is_valid.any():
            samples.append(values[is_valid])
    return samples


def _positive_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean.astype(np.float32)
