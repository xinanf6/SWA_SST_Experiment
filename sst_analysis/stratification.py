"""Fit a frozen MLD split from training grid-days without loading them into RAM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional
import uuid

import numpy as np
import pandas as pd
import xarray as xr

from sst_preprocessing.artifacts import write_json_atomic
from sst_preprocessing.errors import DataValidationError

from .config import validate_output_location
from .context import AnalysisContext
from .diagnostics import CHUNK_DAYS, _training_files


CACHE_FILENAME = "training_mld_median.json"
SCHEMA_VERSION = 1
METHOD = ("Exact median of finite positive training core-ocean grid-day MLD; equal grid-day "
          "weights; float32 in-place partition; even counts average both middle values")
TEMPORARY_PREFIX = ".training_mld_values_"
TEMPORARY_SUFFIX = ".float32.tmp"


def fit_training_mld_threshold(context: AnalysisContext, output_root: Path) -> dict:
    """Side effect: cache an exact training-only MLD median and delete its scratch map.

    Only ``mixed_layer_depth`` values are read, in at most seven-day chunks.
    Each finite positive core-ocean grid-day has one vote; area weights are not
    used. Cache reuse requires complete status and matching source size/mtime.
    """
    output_root = Path(output_root).resolve()
    validate_output_location(output_root, context.root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _training_files(context)
    mask = _core_ocean_mask(context)
    capacity = len(pd.date_range(*context.train_period)) * int(mask.sum())
    identity = _cache_identity(context, paths, mask)
    cache_path = output_root / CACHE_FILENAME
    cached = _read_cache(cache_path, identity, capacity)
    if cached is not None:
        return cached
    threshold, count = _fit_exact(context, paths, {
        "output_root": output_root, "mask": mask, "capacity": capacity,
    })
    if _source_inventory(context, paths) != identity["source_files"]:
        raise DataValidationError("Training MLD source files changed during median fitting")
    report = dict(identity, status="complete", count=count, threshold_m=threshold,
                  training_day_count=capacity // int(mask.sum()),
                  core_ocean_point_count=int(mask.sum()), chunk_days=CHUNK_DAYS, units="m")
    write_json_atomic(cache_path, report)
    return report


def _core_ocean_mask(context: AnalysisContext) -> np.ndarray:
    mask = np.asarray(context.grid["core_mask"], bool) & np.asarray(context.grid["ocean_mask"], bool)
    if not mask.any():
        raise DataValidationError("Training MLD median requires at least one core-ocean cell")
    return mask


def _cache_identity(context: AnalysisContext, paths: list, mask: np.ndarray) -> dict:
    digest = hashlib.sha256()
    for name in ("latitude", "longitude"):
        digest.update(np.asarray(context.grid[name], dtype=np.float64).tobytes())
    digest.update(mask.tobytes())
    return {"schema_version": SCHEMA_VERSION, "run_identity": context.run_identity,
            "train_period": list(context.train_period), "method": METHOD,
            "source_files": _source_inventory(context, paths), "grid_signature": digest.hexdigest()}


def _source_inventory(context: AnalysisContext, paths: list) -> list:
    inventory = []
    for path in paths:
        metadata = path.stat()
        inventory.append({"path": path.relative_to(context.root).as_posix(),
                          "size_bytes": metadata.st_size, "mtime_ns": metadata.st_mtime_ns})
    return inventory


def _read_cache(path: Path, identity: dict, capacity: int) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise DataValidationError("Cannot decode training MLD median cache %s: %s" % (path, error)) from error
    if not isinstance(cached, dict) or cached.get("status") != "complete" or cached.get("units") != "m":
        return None
    if any(cached.get(name) != value for name, value in identity.items()):
        return None
    count, threshold = cached.get("count"), cached.get("threshold_m")
    has_valid_count = isinstance(count, int) and not isinstance(count, bool) and 0 < count <= capacity
    has_valid_threshold = (isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                           and np.isfinite(threshold) and threshold > 0)
    return cached if has_valid_count and has_valid_threshold else None


def _fit_exact(context: AnalysisContext, paths: list, state: dict) -> tuple:
    output_root = state["output_root"]
    path = output_root / (TEMPORARY_PREFIX + uuid.uuid4().hex + TEMPORARY_SUFFIX)
    values = None
    try:
        values = np.memmap(path, mode="w+", dtype=np.float32, shape=(state["capacity"],))
        count = _fill_values(context, paths, {"values": values, "mask": state["mask"]})
        if not count:
            raise DataValidationError("Training MLD has no finite positive core-ocean observations")
        return _exact_median(values, count), count
    finally:
        _release_temporary(values, path, output_root)


def _fill_values(context: AnalysisContext, paths: list, state: dict) -> int:
    expected_dates = pd.date_range(*context.train_period, freq="D")
    days_read, count = 0, 0
    for path in paths:
        for dates, fields in _monthly_mld_chunks(context, path):
            expected = expected_dates[days_read:days_read + len(dates)]
            if not dates.equals(expected):
                raise DataValidationError("Training MLD daily sequence is incomplete near %s" % dates[0])
            core_values = fields[:, state["mask"]]
            valid_values = core_values[np.isfinite(core_values) & (core_values > 0)]
            state["values"][count:count + valid_values.size] = valid_values
            count += valid_values.size
            days_read += len(dates)
    if days_read != len(expected_dates):
        raise DataValidationError("Training MLD read %d daily fields; expected %d" %
                                  (days_read, len(expected_dates)))
    return count


def _monthly_mld_chunks(context: AnalysisContext, path: Path):
    try:
        dataset = xr.open_dataset(path)
    except (OSError, ValueError) as error:
        raise DataValidationError("Cannot open training MLD source %s: %s" % (path, error)) from error
    with dataset:
        dates = _validate_source(context, path, dataset)
        start, end = pd.to_datetime(context.train_period)
        selected = np.flatnonzero((dates >= start) & (dates <= end))
        for offset in range(0, len(selected), CHUNK_DAYS):
            indices = selected[offset:offset + CHUNK_DAYS]
            yield dates[indices], dataset.mixed_layer_depth.isel(time=indices).values.astype(np.float32)


def _validate_source(context: AnalysisContext, path: Path, dataset: xr.Dataset) -> pd.DatetimeIndex:
    if "mixed_layer_depth" not in dataset or "time" not in dataset.coords:
        raise DataValidationError("%s lacks mixed_layer_depth or its daily time coordinate" % path)
    field = dataset.mixed_layer_depth
    if field.dims != ("time", "latitude", "longitude"):
        raise DataValidationError("%s MLD must have time/latitude/longitude dimensions" % path)
    if field.attrs.get("units") != "m":
        raise DataValidationError("%s MLD units must be m" % path)
    if not (np.issubdtype(field.dtype, np.floating) or np.issubdtype(field.dtype, np.integer)):
        raise DataValidationError("%s MLD must contain real numeric depths" % path)
    for name in ("latitude", "longitude"):
        if name not in dataset.coords or not np.array_equal(dataset[name], context.grid[name]):
            raise DataValidationError("%s %s differs from the fixed preprocessing grid" % (path, name))
    if dataset.attrs.get("run_identity") != context.run_identity:
        raise DataValidationError("%s MLD belongs to another preprocessing run" % path)
    dates = pd.DatetimeIndex(dataset.time.values)
    has_valid_dates = (not dates.hasnans and not dates.has_duplicates and dates.is_monotonic_increasing
                       and np.array_equal(dates.values, dates.normalize().values))
    if not has_valid_dates or any(str(date.to_period("M")) != path.stem for date in dates):
        raise DataValidationError("%s has duplicate, unordered, subdaily or unexpected MLD dates" % path)
    return dates


def _exact_median(values: np.memmap, count: int) -> float:
    used_values = values[:count]
    middle = count // 2
    if count % 2:
        used_values.partition(middle)
        return float(used_values[middle])
    used_values.partition((middle - 1, middle))
    return (float(used_values[middle - 1]) + float(used_values[middle])) / 2.0


def _release_temporary(values: np.memmap, path: Path, output_root: Path) -> None:
    try:
        if values is not None:
            try:
                values.flush()
            finally:
                values._mmap.close()
    finally:
        is_safe_path = (path.resolve().parent == output_root and path.name.startswith(TEMPORARY_PREFIX)
                         and path.name.endswith(TEMPORARY_SUFFIX))
        if not is_safe_path:
            raise DataValidationError("Refusing removal of unexpected MLD scratch file: %s" % path)
        if path.exists():
            path.unlink()
