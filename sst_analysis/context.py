"""Validated access to completed preprocessing metadata, without opening all data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sst_preprocessing.artifacts import read_json
from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.grid import build_grid_metadata, gradient_valid_mask


@dataclass
class AnalysisContext:
    """Identity, training bounds and fixed spatial rules shared by every analysis."""

    root: Path
    run: dict
    statistics: dict
    grid: dict

    @property
    def run_identity(self) -> str:
        return self.run["run_identity"]

    @property
    def train_period(self) -> tuple:
        return tuple(self.run["config"]["splits"]["train"])

    @property
    def front_threshold(self) -> float:
        return float(self.statistics["front_threshold_p90"])

    def evaluation_grid(self) -> dict:
        """Return a fresh dictionary; arrays remain immutable by convention."""
        return dict(self.grid, front_threshold=self.front_threshold, front_tolerance_km=10.0)


def load_context(root: Path) -> AnalysisContext:
    """Require a completed matching run and independently validate fixed grid metadata."""
    root = Path(root).resolve()
    required = ("READY.json", "stats.json", "run.json", "grid_masks.npz")
    for name in required:
        if not (root / name).is_file():
            raise DataValidationError("Completed preprocessing artifact is missing: %s" % (root / name))
    ready, run = read_json(root / "READY.json"), read_json(root / "run.json")
    statistics = read_json(root / "stats.json")
    identity = ready.get("run_identity")
    if ready.get("status") != "complete" or not isinstance(identity, str) or not identity:
        raise DataValidationError("Analysis requires READY.json status=complete and a run identity")
    if run.get("run_identity") != identity or statistics.get("run_identity") != identity:
        raise DataValidationError("READY, run and statistics identities do not match")
    with np.load(root / "grid_masks.npz", allow_pickle=False) as archive:
        grid = {name: archive[name] for name in archive.files}
    _validate_grid(grid, PreprocessConfig(**run["config"]))
    threshold = statistics.get("front_threshold_p90")
    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        raise DataValidationError("A finite positive training-only front P90 is required")
    return AnalysisContext(root, run, statistics, grid)


def _validate_grid(grid: dict, config: PreprocessConfig) -> None:
    names = {"latitude", "longitude", "core_mask", "ocean_mask", "gradient_mask", "area_weights"}
    if not names.issubset(grid):
        raise DataValidationError("Grid metadata is incomplete: %s" % sorted(names - set(grid)))
    expected = build_grid_metadata(grid["latitude"], grid["longitude"], config)
    shape = expected["core_mask"].shape
    for name in ("core_mask", "ocean_mask", "gradient_mask", "area_weights"):
        if grid[name].shape != shape:
            raise DataValidationError("Grid shape mismatch for %s" % name)
        if name != "area_weights" and grid[name].dtype != np.bool_:
            raise DataValidationError("Grid %s must have boolean dtype" % name)
    if not np.array_equal(grid["core_mask"], expected["core_mask"]):
        raise DataValidationError("Core mask does not match configured coordinates")
    if not np.allclose(grid["area_weights"], expected["area_weights"], rtol=1e-6):
        raise DataValidationError("Area weights do not match the observed latitude grid")
    if not np.array_equal(grid["gradient_mask"], gradient_valid_mask(grid["ocean_mask"])):
        raise DataValidationError("Gradient template mask does not match the ocean mask")
    if not (grid["core_mask"] & grid["ocean_mask"]).any():
        raise DataValidationError("There are no valid ocean cells in the configured core")
