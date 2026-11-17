"""Resumable monthly preprocessing; raw NetCDF files are opened read-only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .artifacts import (build_run_identity, check_identity, read_json, write_json_atomic,
                        write_netcdf_atomic, write_npz_atomic, write_npy_atomic)
from .config import PreprocessConfig, validate_config
from .constants import BASE_VARIABLES, FEATURE_GROUPS, FRONT_VARIABLES, SCHEMA_VERSION
from .errors import DataValidationError, ResumeMismatchError
from .grid import build_grid_metadata, gradient_valid_mask
from .indexing import build_sample_index
from .processing import build_physical_month
from .sources import build_inventory, fit_mld_climatology, load_source_month
from .statistics import StreamingMoments, compute_percentile_inplace


@dataclass
class PipelineContext:
    config: PreprocessConfig
    inventory: dict
    identity: str
    months: list

    @property
    def root(self) -> Path:
        return self.config.output_root

    def physical_path(self, month: str) -> Path:
        return self.root / "physical" / (month + ".nc")


def month_starts(config: PreprocessConfig) -> list:
    first = pd.Timestamp(config.start_date).to_period("M").to_timestamp()
    last = pd.Timestamp(config.end_date).to_period("M").to_timestamp()
    return [date.strftime("%Y-%m") for date in pd.date_range(first, last, freq="MS")]


def _prepare_context(config: PreprocessConfig, inventory: dict, resume: bool) -> PipelineContext:
    validate_config(config)
    identity = build_run_identity(config.to_dict(), inventory)
    context = PipelineContext(config, inventory, identity, month_starts(config))
    state_path = context.root / "run.json"
    if state_path.is_file():
        if not resume:
            raise ResumeMismatchError("Output already exists; use --resume or a new output directory")
        check_identity(read_json(state_path), identity, "run.json")
        return context
    if context.root.exists() and any(context.root.iterdir()):
        raise ResumeMismatchError("Nonempty output directory has no run.json: %s" % context.root)
    context.root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(state_path, {"run_identity": identity, "schema_version": SCHEMA_VERSION,
                                   "config": config.to_dict()})
    write_json_atomic(context.root / "manifest.json", inventory)
    return context


def _get_climatology(context: PipelineContext) -> dict:
    path = context.root / "mld_climatology.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    print("Fitting training-only MLD monthly climatology...", flush=True)
    climatology = fit_mld_climatology(context.config, context.inventory)
    arrays = {key: value for key, value in climatology.items()
              if isinstance(value, (np.ndarray, float, int, np.number))}
    write_npz_atomic(path, arrays)
    notes = {key: value for key, value in climatology.items() if key not in arrays}
    write_json_atomic(context.root / "mld_climatology.json", notes)
    return arrays


def _validate_cached_month(context: PipelineContext, month: str) -> None:
    with xr.open_dataset(context.physical_path(month)) as dataset:
        check_identity(dataset.attrs, context.identity, "physical/" + month)
        expected = _month_dates(context.config, month)
        if not pd.DatetimeIndex(dataset.time.values).equals(expected):
            raise DataValidationError("Cached month has missing/duplicate dates: %s" % month)
        for variables in FEATURE_GROUPS.values():
            if not set(variables).issubset(dataset.data_vars):
                raise DataValidationError("Cached physical variables are incomplete: %s" % month)


def _month_dates(config: PreprocessConfig, month: str) -> pd.DatetimeIndex:
    start = max(pd.Timestamp(month + "-01"), pd.Timestamp(config.start_date))
    last = pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(1)
    end = min(last, pd.Timestamp(config.end_date))
    return pd.date_range(start, end, freq="D")


def _prepare_month(context: PipelineContext, month: str, climatology: dict) -> None:
    path = context.physical_path(month)
    if path.is_file():
        _validate_cached_month(context, month)
        print("Reuse physical %s" % month, flush=True)
        return
    sources = load_source_month(context.inventory, month + "-01")
    dataset = build_physical_month(sources, context.config, climatology)
    dataset.attrs["run_identity"] = context.identity
    write_netcdf_atomic(path, dataset, context.config.compression_level)
    _validate_cached_month(context, month)
    print("Prepared physical %s (%d days)" % (month, dataset.sizes["time"]), flush=True)


def _training_dates(dataset: xr.Dataset, config: PreprocessConfig) -> xr.Dataset:
    start, end = config.splits["train"]
    return dataset.sel(time=slice(start, end))


def _fit_grid_masks(context: PipelineContext) -> dict:
    ocean_mask = None
    metadata = None
    for month in context.months:
        with xr.open_dataset(context.physical_path(month)) as dataset:
            if metadata is None:
                metadata = build_grid_metadata(dataset.latitude.values,
                                               dataset.longitude.values, context.config)
            training = _training_dates(dataset, context.config)
            if not training.sizes["time"]:
                continue
            valid = np.ones(metadata["core_mask"].shape, dtype=bool)
            for name in BASE_VARIABLES + ("mixed_layer_depth", "net_heat_flux", "heat_increment"):
                valid &= np.isfinite(training[name].values).all(axis=0)
            ocean_mask = valid if ocean_mask is None else ocean_mask & valid
    if ocean_mask is None or not (ocean_mask & metadata["core_mask"]).any():
        raise DataValidationError("Training sources have no common valid ocean in the core")
    metadata["ocean_mask"] = ocean_mask
    metadata["gradient_mask"] = gradient_valid_mask(ocean_mask)
    return metadata


def _variable_mask(name: str, grid: dict) -> np.ndarray:
    return grid["gradient_mask"] if name in FRONT_VARIABLES else grid["ocean_mask"]


def _fit_statistics(context: PipelineContext, grid: dict) -> dict:
    variables = [name for group in FEATURE_GROUPS.values() for name in group]
    moments = {name: StreamingMoments() for name in variables}
    scratch = context.root / "gradient_percentile.dat"
    count = 0
    with scratch.open("wb") as handle:
        state = {"grid": grid, "moments": moments, "quantile_file": handle}
        for month in context.months:
            with xr.open_dataset(context.physical_path(month)) as dataset:
                training = _training_dates(dataset, context.config)
                if not training.sizes["time"]:
                    continue
                count += _update_training_statistics(training, state)
    threshold = compute_percentile_inplace(scratch, count, percentile=90.0)
    from .constants import VARIABLE_UNITS
    stats = {"run_identity": context.identity, "schema_version": SCHEMA_VERSION,
             "train_period": list(context.config.splits["train"]),
             "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
             "variables": {name: dict(value.result(), units=VARIABLE_UNITS[name])
                           for name, value in moments.items()},
             "front_threshold_p90": threshold, "front_percentile_count": count,
             "normalization": "Unweighted population moments over training ocean; constant std uses scale=1"}
    write_json_atomic(context.root / "stats.json", stats)
    return stats


def _update_training_statistics(training: xr.Dataset, state: dict) -> int:
    grid = state["grid"]
    for name, accumulator in state["moments"].items():
        accumulator.update(training[name].values, mask=_variable_mask(name, grid))
    values = training["gradient_magnitude"].values
    mask = grid["gradient_mask"] & grid["core_mask"]
    selected = values[:, mask]
    selected = selected[np.isfinite(selected)].astype(np.float32)
    selected.tofile(state["quantile_file"])
    return selected.size


def _get_statistics(context: PipelineContext) -> tuple:
    statistics_path = context.root / "stats.json"
    grid_path = context.root / "grid_masks.npz"
    if statistics_path.is_file() and grid_path.is_file():
        stats = read_json(statistics_path)
        check_identity(stats, context.identity, "stats.json")
        with np.load(grid_path, allow_pickle=False) as archive:
            return stats, {name: archive[name] for name in archive.files}
    print("Fitting fixed masks and training-only statistics...", flush=True)
    grid = _fit_grid_masks(context)
    write_npz_atomic(grid_path, grid)
    return _fit_statistics(context, grid), grid


def _write_index(context: PipelineContext) -> dict:
    dates = pd.date_range(context.config.start_date, context.config.end_date, freq="D")
    write_npy_atomic(context.root / "dates.npy", dates.values.astype("datetime64[D]"))
    index = build_sample_index(dates, context.config)
    from .quality import filter_sample_index
    quality = pd.read_csv(context.root / "daily_quality.csv")
    index, excluded = filter_sample_index(index, quality)
    excluded_temporary = context.root / "excluded_samples.csv.tmp"
    excluded.to_csv(excluded_temporary, index=False, encoding="utf-8")
    excluded_temporary.replace(context.root / "excluded_samples.csv")
    temporary = context.root / "sample_index.csv.tmp"
    index.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(context.root / "sample_index.csv")
    return {name: int((index["split"] == name).sum()) for name in context.config.splits}


def _prepare_daily_quality(context: PipelineContext, grid: dict) -> None:
    from .quality import assess_daily_quality
    path = context.root / "daily_quality.csv"
    metadata_path = context.root / "daily_quality.json"
    if path.is_file() and metadata_path.is_file():
        check_identity(read_json(metadata_path), context.identity, "daily_quality.json")
        return
    frames = []
    for month in context.months:
        with xr.open_dataset(context.physical_path(month)) as dataset:
            frames.append(assess_daily_quality(dataset, grid))
    quality = pd.concat(frames, ignore_index=True)
    temporary = path.with_name(path.name + ".tmp")
    quality.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)
    columns = ["sst_valid", "front_valid", "environment_valid"]
    invalid = quality.loc[~quality[columns].all(axis=1)]
    write_json_atomic(metadata_path, {"run_identity": context.identity,
                      "invalid_days": invalid.to_dict(orient="records"),
                      "policy": "No imputation; exclude windows with invalid inputs/targets"})


def run_preprocessing(config: PreprocessConfig, *, resume: bool = False,
                      stop_after_months: Optional[int] = None) -> dict:
    """Side effect: run the selected pipeline or publish a clearly incomplete smoke run."""
    print("Inspecting and hashing source files...", flush=True)
    inventory = build_inventory(config)
    context = _prepare_context(config, inventory, resume)
    climatology = _get_climatology(context)
    selected_months = context.months[:stop_after_months] if stop_after_months else context.months
    for month in selected_months:
        _prepare_month(context, month, climatology)
    if len(selected_months) != len(context.months):
        report = {"status": "partial", "run_identity": context.identity,
                  "prepared_months": selected_months, "total_months": len(context.months)}
        write_json_atomic(context.root / "smoke_report.json", report)
        return report
    statistics, grid = _get_statistics(context)
    _prepare_daily_quality(context, grid)
    from .normalization import prepare_normalized_arrays
    prepare_normalized_arrays(context, statistics, grid)
    counts = _write_index(context)
    from .verification import verify_output
    report = verify_output(config.output_root, deep=True, expected_identity=context.identity)
    report.update(sample_counts=counts, source_file_count=len(inventory["files"]))
    write_json_atomic(context.root / "quality_report.json", report)
    write_json_atomic(context.root / "READY.json", {"run_identity": context.identity,
                       "schema_version": SCHEMA_VERSION, "sample_counts": counts,
                       "status": "complete"})
    from .verification import write_quality_markdown
    write_quality_markdown(context.root, report)
    return report
