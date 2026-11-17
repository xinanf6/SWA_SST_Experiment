"""Independent artifact, clock, heat-unit and normalization verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .artifacts import build_run_identity, check_identity, close_memmap, read_json
from .config import PreprocessConfig
from .constants import ENERGY_VARIABLES, FEATURE_GROUPS, FRONT_VARIABLES, SECONDS_PER_DAY
from .errors import DataValidationError
from .grid import build_grid_metadata, gradient_valid_mask
from .indexing import build_sample_index
from .statistics import StreamingMoments, denormalize_values


def _verify_index(root: Path, config: PreprocessConfig, dates: np.ndarray) -> dict:
    actual = pd.read_csv(root / "sample_index.csv")
    expected = build_sample_index(dates, config)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except AssertionError as error:
        raise DataValidationError("Sample index violates date/lag/split contract: %s" % error)
    return {name: int((actual["split"] == name).sum()) for name in config.splits}


def _load_arrays(root: Path, dates: np.ndarray, grid: dict) -> dict:
    arrays = {}
    try:
        for group, variables in FEATURE_GROUPS.items():
            path = root / "normalized" / (group + ".npy")
            arrays[group] = np.load(path, mmap_mode="r", allow_pickle=False)
            expected = (len(dates), len(variables)) + grid["ocean_mask"].shape
            if arrays[group].shape != expected or arrays[group].dtype != np.float32:
                raise DataValidationError("Normalized array shape/dtype mismatch: %s" % path)
    except Exception:
        _close_arrays(arrays)
        raise
    return arrays


def _close_arrays(arrays: dict) -> None:
    for array in arrays.values():
        close_memmap(array)


def _verify_heat(dataset: xr.Dataset, config: PreprocessConfig) -> None:
    energy = sum(dataset[name].values.astype(np.float64) for name in ENERGY_VARIABLES)
    depth = dataset["mixed_layer_depth"].values
    effective = np.maximum(depth, config.mixed_layer_minimum_meters)
    increment = energy / (config.seawater_density * config.seawater_heat_capacity * effective)
    for name, expected in (("net_heat_flux", energy / SECONDS_PER_DAY), ("heat_increment", increment)):
        if not np.allclose(dataset[name].values, expected, rtol=2e-5, atol=1e-6, equal_nan=True):
            raise DataValidationError("Daily energy/flux/increment inconsistency: %s" % name)


def _verify_normalized_month(dataset: xr.Dataset, state: dict) -> None:
    dates = pd.DatetimeIndex(dataset.time.values)
    start = (dates[0] - pd.Timestamp(state["config"].start_date)).days
    stop = start + len(dates)
    train_start, train_end = map(pd.Timestamp, state["config"].splits["train"])
    is_train = (dates >= train_start) & (dates <= train_end)
    for group, variables in FEATURE_GROUPS.items():
        for channel, name in enumerate(variables):
            values = state["arrays"][group][start:stop, channel]
            mask = state["grid"]["gradient_mask"] if name in FRONT_VARIABLES else state["grid"]["ocean_mask"]
            _check_variable_roundtrip(dataset[name].values, values,
                                      {"name": name, "mask": mask, "statistics": state["statistics"]})
            if is_train.any():
                state["moments"][name].update(values[is_train], mask=mask)


def _check_variable_roundtrip(physical: np.ndarray, normalized: np.ndarray,
                              details: dict) -> None:
    name, mask = details["name"], details["mask"]
    if not np.isfinite(normalized).all() or not np.isfinite(physical[:, mask]).all():
        raise DataValidationError("Nonfinite values in a completed feature: %s" % name)
    if np.any(normalized[:, ~mask] != 0):
        raise DataValidationError("Unmasked fill values in normalized feature: %s" % name)
    info = details["statistics"]["variables"][name]
    restored = denormalize_values(normalized[:, mask], info["mean"], info["std"])
    if not np.allclose(restored, physical[:, mask], rtol=2e-5, atol=2e-5):
        raise DataValidationError("Normalization round-trip failed: %s" % name)


def _deep_verify(root: Path, state: dict) -> dict:
    config, grid, statistics = state["config"], state["grid"], state["statistics"]
    variables = [name for group in FEATURE_GROUPS.values() for name in group]
    state["moments"] = {name: StreamingMoments() for name in variables}
    all_dates, filled_points = [], 0
    for path in sorted((root / "physical").glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            check_identity(dataset.attrs, state["identity"], str(path))
            all_dates.extend(dataset.time.values)
            _verify_heat(dataset, config)
            _verify_normalized_month(dataset, state)
            filled_points += int(dataset["mld_was_filled"].values[:, grid["ocean_mask"]].sum())
    expected = pd.date_range(config.start_date, config.end_date)
    if not pd.DatetimeIndex(all_dates).equals(expected):
        raise DataValidationError("Physical monthly files do not cover the exact daily timeline")
    normalized_stats = {name: moment.result() for name, moment in state["moments"].items()}
    for name, info in normalized_stats.items():
        expected_std = 1.0 if statistics["variables"][name]["std"] > 1e-12 else 0.0
        if abs(info["mean"]) > 1e-3 or abs(info["std"] - expected_std) > 1e-3:
            raise DataValidationError("Training normalization is not centered/scaled: %s" % name)
    return {"normalized_training_statistics": normalized_stats, "mld_filled_target_grid_days": filled_points}


def _verification_state(root: Path, expected_identity: str) -> dict:
    run, inventory = read_json(root / "run.json"), read_json(root / "manifest.json")
    config = PreprocessConfig(**run["config"])
    identity = build_run_identity(run["config"], inventory)
    check_identity(run, expected_identity or identity, "run.json")
    statistics = read_json(root / "stats.json")
    check_identity(statistics, identity, "stats.json")
    with np.load(root / "grid_masks.npz", allow_pickle=False) as archive:
        grid = {name: archive[name] for name in archive.files}
    _verify_grid(grid, config)
    dates = np.load(root / "dates.npy", allow_pickle=False)
    if not pd.DatetimeIndex(dates).equals(pd.date_range(config.start_date, config.end_date)):
        raise DataValidationError("dates.npy does not cover the configured period")
    return {"config": config, "inventory": inventory, "identity": identity,
            "statistics": statistics, "grid": grid, "dates": dates}


def _verify_grid(grid: dict, config: PreprocessConfig) -> None:
    expected = build_grid_metadata(grid["latitude"], grid["longitude"], config)
    for name in ("core_mask", "area_weights"):
        if not np.allclose(grid[name], expected[name], rtol=1e-6, atol=1e-7):
            raise DataValidationError("Stored grid metadata is inconsistent: %s" % name)
    if not np.array_equal(grid["gradient_mask"], gradient_valid_mask(grid["ocean_mask"])):
        raise DataValidationError("Gradient mask is inconsistent with fixed ocean neighbors")
    if not (grid["ocean_mask"] & grid["core_mask"]).any():
        raise DataValidationError("No fixed ocean points remain inside the core")


def _verification_report(root: Path, state: dict, deep: bool) -> dict:
    grid, statistics = state["grid"], state["statistics"]
    core_shape = [int(grid["core_mask"].any(axis=1).sum()),
                  int(grid["core_mask"].any(axis=0).sum())]
    return {"status": "verified", "run_identity": state["identity"], "days": len(state["dates"]),
              "input_grid_shape": list(grid["ocean_mask"].shape),
              "core_grid_shape": core_shape,
              "ocean_points": int(grid["ocean_mask"].sum()),
              "core_ocean_points": int((grid["ocean_mask"] & grid["core_mask"]).sum()),
              "gradient_points": int(grid["gradient_mask"].sum()),
              "sample_counts": _verify_index(root, state["config"], state["dates"]),
              "front_threshold_p90": statistics["front_threshold_p90"],
              "source_file_count": len(state["inventory"]["files"]), "deep_verification": deep}


def verify_output(output_root: Path, *, deep: bool = True,
                   expected_identity: str = None) -> dict:
    """Validate artifacts without READY; always release Windows mapped-file handles."""
    root = Path(output_root)
    state = _verification_state(root, expected_identity)
    arrays = _load_arrays(root, state["dates"], state["grid"])
    state["arrays"] = arrays
    try:
        report = _verification_report(root, state, deep)
        if deep:
            report.update(_deep_verify(root, state))
        return report
    finally:
        _close_arrays(arrays)


def write_quality_markdown(root: Path, report: dict) -> None:
    """Side effect: write a compact, reproducible quality summary."""
    lines = ["# SWA SST preprocessing quality report", "",
             "- Status: %s" % report["status"],
             "- Source files: %d" % report["source_file_count"],
             "- Daily dates: %d" % report["days"],
             "- Input grid: %s" % report["input_grid_shape"],
             "- Core ocean points: %d" % report["core_ocean_points"],
             "- Samples: %s" % report["sample_counts"],
             "- Front P90: %.8f degree_Celsius/km" % report["front_threshold_p90"],
             "- Deep verification: %s" % report["deep_verification"], "",
             "Original protocol: 3 m floor, net surface energy, no shortwave penetration.",
             "Statistics and static masks use training dates only; environment lag is applied by index.",
             "Arrays are daily memmaps, not expanded windows. READY.json marks completed output.", ""]
    temporary = root / "quality_report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(root / "quality_report.md")
