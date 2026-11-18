"""Reuse unchanged physical fields while rebuilding masks, statistics and windows."""

from __future__ import annotations

import shutil
import json
from pathlib import Path

import netCDF4

from .artifacts import check_identity, read_json, write_json_atomic
from .errors import DataValidationError
from .pipeline import _prepare_context, _validate_cached_month, run_preprocessing
from .sources import build_inventory


def _validate_source(config, source_root: Path, inventory: dict) -> dict:
    source_root = source_root.resolve()
    target = config.output_root.resolve()
    if target == source_root or target in source_root.parents or source_root in target.parents:
        raise DataValidationError("Rebuild requires separate source and destination directories")
    ready = read_json(source_root / "READY.json")
    run = read_json(source_root / "run.json")
    if ready.get("status") != "complete" or not ready.get("run_identity"):
        raise DataValidationError("Rebuild source must have a completed READY identity")
    check_identity(run, ready["run_identity"], "rebuild source run")
    old_settings, new_settings = dict(run["config"]), config.to_dict()
    old_settings.pop("output_root", None)
    new_settings.pop("output_root", None)
    if json.dumps(old_settings, sort_keys=True) != json.dumps(new_settings, sort_keys=True):
        raise DataValidationError("Physical reuse requires identical numerical/source settings")
    old_manifest = read_json(source_root / "manifest.json")
    if old_manifest["fingerprint"] != inventory["fingerprint"]:
        raise DataValidationError("Physical reuse source manifest differs from current raw inputs")
    return run


def _copy_month(context, source: Path, source_identity: str) -> None:
    destination = context.physical_path(source.stem)
    if destination.exists():
        _validate_cached_month(context, source.stem)
        return
    with netCDF4.Dataset(str(source), "r") as dataset:
        check_identity({"run_identity": dataset.getncattr("run_identity")}, source_identity, str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".nc.tmp")
    shutil.copyfile(str(source), str(temporary))
    # Always copy before updating provenance; the old physical archive stays immutable.
    with netCDF4.Dataset(str(temporary), "r+") as dataset:
        dataset.setncattr("run_identity", context.identity)
        dataset.setncattr("physical_source_run_identity", source_identity)
    temporary.replace(destination)
    _validate_cached_month(context, source.stem)


def rebuild_from_physical(config, source_root: Path, resume: bool = False) -> dict:
    """Side effect: copy validated physical fields and rebuild all dependent artifacts."""
    source_root = Path(source_root).resolve()
    inventory = build_inventory(config)
    source_run = _validate_source(config, source_root, inventory)
    context = _prepare_context(config, inventory, resume)
    for name in ("mld_climatology.npz", "mld_climatology.json"):
        destination = context.root / name
        if not destination.exists():
            shutil.copyfile(str(source_root / name), str(destination))
    for month in context.months:
        _copy_month(context, source_root / "physical" / (month + ".nc"), source_run["run_identity"])
        print("Validated physical reuse %s" % month, flush=True)
    write_json_atomic(context.root / "rebuild_provenance.json", {
        "source_root": str(source_root), "source_run_identity": source_run["run_identity"],
        "run_identity": context.identity, "reason": "Separate static ocean geography from daily missingness",
        "physical_values": "Unchanged; copied files with updated run provenance"})
    return run_preprocessing(config, resume=True)
