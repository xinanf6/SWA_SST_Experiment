"""Frozen seasonal/front/lagged-MLD strata for supplied forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.grid import temperature_gradients
from sst_preprocessing.statistics import denormalize_values

from .context import AnalysisContext
from .metrics import ForecastEvaluator


GROUP_NAMES = ("DJF", "MAM", "JJA", "SON", "strong_front", "weak_front", "shallow_mld", "deep_mld")
SEASON_BY_MONTH = np.array(["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"])


class GroupedForecastEvaluator:
    """Accumulate group scores; only center masks change, never derivative stencils."""

    def __init__(self, context: AnalysisContext, options: dict) -> None:
        self.context = context
        self.threshold = float(options["mld_threshold_m"])
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise DataValidationError("Group MLD threshold must be finite and positive")
        self.evaluators = {name: ForecastEvaluator(context.evaluation_grid()) for name in GROUP_NAMES}
        self.thermal = np.load(context.root / "normalized" / "thermal.npy", mmap_mode="r", allow_pickle=False)
        try:
            self.dates = np.load(context.root / "dates.npy", allow_pickle=False).astype("datetime64[D]")
        except Exception:
            self.close()
            raise
        shape = (len(self.dates), 3) + context.grid["ocean_mask"].shape
        if self.thermal.shape != shape or self.thermal.dtype != np.float32:
            self.close()
            raise DataValidationError("Prepared thermal array has an unexpected shape or dtype")

    def update(self, predictions: np.ndarray, targets: np.ndarray, index: pd.DataFrame) -> list:
        """Return all origin/lead/group rows, retaining zero-weight seasonal rows."""
        masks = self._masks(targets, index)
        records = []
        for name, evaluator in self.evaluators.items():
            rows = evaluator.update(predictions, targets, masks[name])
            records.extend(dict(row, group=name) for row in rows)
        return records

    def _masks(self, targets: np.ndarray, index: pd.DataFrame) -> dict:
        values = np.asarray(targets)
        if values.ndim == 5 and values.shape[2] == 1:
            values = values[:, :, 0]
        spatial = self.context.grid["ocean_mask"].shape
        if values.shape != (len(index), 3) + spatial:
            raise DataValidationError("Grouped targets do not match the sample index and fixed grid")
        target_months = np.stack([pd.to_datetime(index["target_%d_date" % lead]).dt.month.to_numpy()
                                  for lead in (1, 2, 3)], axis=1)
        seasons = SEASON_BY_MONTH[target_months - 1]
        masks = {name: np.broadcast_to((seasons == name)[..., None, None], values.shape)
                 for name in GROUP_NAMES[:4]}
        full_field = np.where(self.context.grid["ocean_mask"], values, np.nan).reshape((-1,) + spatial)
        gradient = temperature_gradients(full_field, self.context.grid["latitude"],
                                         self.context.grid["longitude"])[2].reshape(values.shape)
        valid = self.context.grid["gradient_mask"]
        masks["strong_front"] = valid & (gradient >= self.context.front_threshold)
        masks["weak_front"] = valid & (gradient < self.context.front_threshold)
        depth = self._lagged_mld(index)
        masks["shallow_mld"] = np.broadcast_to(depth[:, None] <= self.threshold, values.shape)
        masks["deep_mld"] = np.broadcast_to(depth[:, None] > self.threshold, values.shape)
        return masks

    def _lagged_mld(self, index: pd.DataFrame) -> np.ndarray:
        raw_indices = index.environment_end_index.to_numpy()
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise DataValidationError("Environment end indices must be integers")
        if ((raw_indices < 0) | (raw_indices >= len(self.dates))).any():
            raise DataValidationError("Environment end index is outside prepared dates")
        expected = pd.to_datetime(index.origin_date).to_numpy(dtype="datetime64[D]") - np.timedelta64(1, "D")
        if not np.array_equal(self.dates[raw_indices], expected):
            raise DataValidationError("MLD stratification must use the latest lagged environment day t-1")
        info = self.context.statistics["variables"]["mixed_layer_depth"]
        values = denormalize_values(self.thermal[raw_indices, 1], info["mean"], info["std"])
        core = self.context.grid["core_mask"] & self.context.grid["ocean_mask"]
        if not np.isfinite(values[:, core]).all() or (values[:, core] <= 0).any():
            raise DataValidationError("Lagged MLD must be finite and positive on the core ocean domain")
        return values

    def result(self) -> list:
        """Return one sufficient-statistics record per group and lead."""
        return [dict(row, group=name) for name, evaluator in self.evaluators.items()
                for row in evaluator.result()]

    def close(self) -> None:
        """Side effect: release the read-only thermal memory mapping."""
        if getattr(self, "thermal", None) is not None:
            self.thermal._mmap.close()
            self.thermal = None
