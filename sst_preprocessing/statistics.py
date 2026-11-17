"""Training-only streaming statistics and reversible feature normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .constants import STANDARD_DEVIATION_EPSILON
from .errors import DataValidationError


class StreamingMoments:
    """Accumulate finite, masked values using stable population moments."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.squared_deviation = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Merge one chunk; a spatial mask broadcasts over leading dimensions."""
        values = np.asarray(values, dtype=np.float64)
        is_valid = np.isfinite(values)
        if mask is not None:
            is_valid &= np.broadcast_to(np.asarray(mask, dtype=bool), values.shape)
        selected = values[is_valid]
        if selected.size == 0:
            return
        chunk_mean = float(selected.mean(dtype=np.float64))
        chunk_deviation = float(np.square(selected - chunk_mean).sum(dtype=np.float64))
        total_count = self.count + selected.size
        mean_difference = chunk_mean - self.mean
        self.squared_deviation += chunk_deviation + (
            mean_difference ** 2 * self.count * selected.size / total_count
        )
        self.mean += mean_difference * selected.size / total_count
        self.count = int(total_count)
        self.minimum = min(self.minimum, float(selected.min()))
        self.maximum = max(self.maximum, float(selected.max()))

    def result(self) -> dict:
        """Return JSON-safe population statistics; reject empty selections."""
        if self.count == 0:
            raise DataValidationError("No finite training values available for statistics")
        return {
            "count": self.count,
            "mean": float(self.mean),
            "std": float(np.sqrt(max(self.squared_deviation / self.count, 0.0))),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


def _normalization_scale(standard_deviation: float) -> float:
    if not np.isfinite(standard_deviation) or standard_deviation < 0:
        raise DataValidationError("Normalization standard deviation must be finite and nonnegative")
    # Constant training channels carry no variation; unit scaling keeps their inverse exact.
    return float(standard_deviation) if standard_deviation > STANDARD_DEVIATION_EPSILON else 1.0


def normalize_values(values: np.ndarray, statistics: dict, valid_mask: np.ndarray) -> np.ndarray:
    """Return float32 z-scores, setting masked/nonfinite cells to zero."""
    mean, std = statistics["mean"], statistics["std"]
    if not np.isfinite(mean):
        raise DataValidationError("Normalization mean must be finite")
    values = np.asarray(values)
    is_valid = np.broadcast_to(np.asarray(valid_mask, dtype=bool), values.shape)
    is_valid = is_valid & np.isfinite(values)
    result = np.zeros(values.shape, dtype=np.float32)
    # Nearly constant float32 gradients need subtraction before rounding the training mean.
    selected = values[is_valid].astype(np.float64)
    result[is_valid] = (selected - mean) / _normalization_scale(std)
    return result


def denormalize_values(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Recover physical values with the same constant-channel policy as normalization."""
    if not np.isfinite(mean):
        raise DataValidationError("Normalization mean must be finite")
    physical = np.asarray(values, dtype=np.float64) * _normalization_scale(std) + mean
    return physical.astype(np.float32)


def compute_percentile_inplace(path: Path, count: int, percentile: float = 90.0) -> float:
    """Partition a disposable raw float32 memmap without a full-array copy."""
    if count < 1 or not 0.0 <= percentile <= 100.0:
        raise DataValidationError("Percentile requires nonempty values and a value in [0, 100]")
    values = np.memmap(Path(path), dtype=np.float32, mode="r+", shape=(count,))
    try:
        position = (count - 1) * percentile / 100.0
        lower_index, upper_index = int(np.floor(position)), int(np.ceil(position))
        values.partition((lower_index, upper_index))
        lower_value, upper_value = float(values[lower_index]), float(values[upper_index])
        result = lower_value + (upper_value - lower_value) * (position - lower_index)
    finally:
        values.flush()
        values._mmap.close()
    if not np.isfinite(result):
        raise DataValidationError("Percentile scratch file contains invalid selected values")
    return result
