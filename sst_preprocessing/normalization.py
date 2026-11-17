"""Bounded-memory normalization into restartable, non-expanded daily arrays."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .artifacts import check_identity, close_memmap, read_json, write_json_atomic
from .constants import FEATURE_GROUPS, FRONT_VARIABLES
from .errors import DataValidationError, ResumeMismatchError
from .statistics import normalize_values


def _normalization_identity(context, statistics: dict, grid: dict) -> str:
    digest = hashlib.sha256(json.dumps(statistics, sort_keys=True).encode("utf-8"))
    digest.update(context.identity.encode("ascii"))
    for name in ("ocean_mask", "gradient_mask"):
        digest.update(grid[name].tobytes())
    return digest.hexdigest()


def _open_maps(context, grid: dict, progress: dict) -> dict:
    folder = context.root / "normalized"
    folder.mkdir(exist_ok=True)
    days = len(pd.date_range(context.config.start_date, context.config.end_date))
    arrays = {}
    try:
        for group, variables in FEATURE_GROUPS.items():
            _open_group_map({"context": context, "grid": grid, "progress": progress,
                             "folder": folder, "days": days, "arrays": arrays}, group)
    except Exception:
        for array in arrays.values():
            close_memmap(array)
        raise
    return arrays


def _open_group_map(state: dict, group: str) -> None:
    context, progress, arrays = state["context"], state["progress"], state["arrays"]
    folder, grid, days = state["folder"], state["grid"], state["days"]
    variables = FEATURE_GROUPS[group]
    shape = (days, len(variables)) + grid["ocean_mask"].shape
    temporary, final = folder / (group + ".npy.part"), folder / (group + ".npy")
    if final.exists() and not temporary.exists():
        if len(progress["months_done"]) != len(context.months):
            raise ResumeMismatchError("Final normalized file exists before all months completed: %s" % final)
        return
    if temporary.exists():
        arrays[group] = np.load(temporary, mmap_mode="r+", allow_pickle=False)
        if arrays[group].shape != shape or arrays[group].dtype != np.float32:
            raise ResumeMismatchError("Partial normalized file has unexpected shape/dtype: %s" % temporary)
    else:
        if progress["months_done"]:
            raise ResumeMismatchError("Checkpointed normalized map is missing: %s" % temporary)
        arrays[group] = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=shape)


def _write_normalized_month(state: dict, month: str) -> None:
    context, grid = state["context"], state["grid"]
    with xr.open_dataset(context.physical_path(month)) as dataset:
        start = (pd.Timestamp(dataset.time.values[0]) - pd.Timestamp(context.config.start_date)).days
        stop = start + dataset.sizes["time"]
        for group, variables in FEATURE_GROUPS.items():
            for channel, name in enumerate(variables):
                values = dataset[name].values
                mask = grid["gradient_mask"] if name in FRONT_VARIABLES else grid["ocean_mask"]
                if not np.isfinite(values[:, mask]).all():
                    raise DataValidationError("Unexpected missing %s at fixed ocean cells in %s; no silent imputation" %
                                              (name, month))
                info = state["statistics"]["variables"][name]
                state["arrays"][group][start:stop, channel] = normalize_values(values, info, mask)
    for array in state["arrays"].values():
        array.flush()


def prepare_normalized_arrays(context, statistics: dict, grid: dict) -> None:
    """Side effect: checkpoint each flushed month and atomically publish final arrays."""
    state_path = context.root / "normalization.json"
    identity = _normalization_identity(context, statistics, grid)
    progress = {"run_identity": identity, "months_done": []}
    if state_path.is_file():
        progress = read_json(state_path)
        check_identity(progress, identity, "normalization.json")
    if progress["months_done"] != context.months[:len(progress["months_done"])]:
        raise ResumeMismatchError("Normalization checkpoint months are not a valid prefix")
    arrays = _open_maps(context, grid, progress)
    state = {"context": context, "grid": grid, "arrays": arrays, "statistics": statistics}
    try:
        for month in context.months:
            if month in progress["months_done"]:
                continue
            _write_normalized_month(state, month)
            progress["months_done"].append(month)
            write_json_atomic(state_path, progress)
            print("Normalized %s" % month, flush=True)
    finally:
        for array in arrays.values():
            close_memmap(array)
    for group in FEATURE_GROUPS:
        temporary = context.root / "normalized" / (group + ".npy.part")
        if temporary.exists():
            os.replace(str(temporary), str(temporary.with_name(group + ".npy")))
