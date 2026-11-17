"""Small atomic artifact operations and explicit resume identities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from .constants import SCHEMA_VERSION
from .errors import ResumeMismatchError


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: dict) -> None:
    """Side effect: replace one complete JSON file, never a partially written one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def write_npz_atomic(path: Path, arrays: dict) -> None:
    """Side effect: atomically save non-object arrays, without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def write_npy_atomic(path: Path, values: np.ndarray) -> None:
    """Side effect: atomically save a NumPy array."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(str(temporary), str(path))


def write_netcdf_atomic(path: Path, dataset: xr.Dataset, compression: int) -> None:
    """Side effect: save a complete monthly physical dataset before publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoding = {}
    for name, variable in dataset.data_vars.items():
        settings = {"zlib": compression > 0, "complevel": compression}
        if np.issubdtype(variable.dtype, np.floating):
            settings.update(dtype="float32", _FillValue=np.float32(np.nan))
        else:
            settings.update(dtype="uint8")
        encoding[name] = settings
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    os.replace(str(temporary), str(path))


def build_run_identity(config: dict, inventory: dict) -> str:
    """Hash numerical settings and the source manifest, independent of output location."""
    settings = dict(config)
    settings.pop("output_root", None)
    payload = {"schema_version": SCHEMA_VERSION, "config": settings,
               "source_fingerprint": inventory["fingerprint"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def check_identity(metadata: dict, identity: str, context: str) -> None:
    """Reject resume against different source files or configuration."""
    if metadata.get("run_identity") != identity:
        raise ResumeMismatchError("%s belongs to different inputs/settings; use a new output directory" % context)


def close_memmap(array: np.memmap) -> None:
    """Side effect: flush and close a map so Windows can atomically rename it."""
    if array.mode != "r":
        array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()
