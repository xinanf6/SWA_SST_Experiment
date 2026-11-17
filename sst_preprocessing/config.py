"""Load an explicit, portable JSON configuration without importing download code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .errors import ConfigurationError


@dataclass(frozen=True)
class PreprocessConfig:
    """Public settings for the original 3 m, net-surface-heat experiment."""

    data_root: Path
    output_root: Path
    start_date: str = "2011-01-01"
    end_date: str = "2024-12-31"
    splits: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "train": ("2011-01-01", "2020-12-31"),
        "validation": ("2021-01-01", "2022-12-31"),
        "test": ("2023-01-01", "2024-12-31"),
    })
    history_days: int = 14
    prediction_days: int = 3
    environment_lag_days: int = 1
    mixed_layer_minimum_meters: float = 3.0
    seawater_density: float = 1025.0
    seawater_heat_capacity: float = 3990.0
    minimum_interpolation_weight: float = 0.999999
    core_south: float = -48.0
    core_north: float = -35.0
    core_west: float = -60.0
    core_east: float = -48.0
    expected_grid_shape: Optional[Tuple[int, int]] = (270, 250)
    expected_core_shape: Optional[Tuple[int, int]] = (260, 240)
    chunk_days: int = 7
    compression_level: int = 1

    def to_dict(self) -> dict:
        """Return a JSON-safe configuration, including resolved local paths."""
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["output_root"] = str(self.output_root)
        return result


def validate_config(config: PreprocessConfig) -> None:
    """Reject unsupported dates and settings before creating any output."""
    raw_root, output_root = config.data_root.resolve(), config.output_root.resolve()
    if raw_root == output_root or raw_root in output_root.parents:
        raise ConfigurationError("output_root must be outside the raw data_root")
    if config.history_days < 1 or config.prediction_days < 1:
        raise ConfigurationError("history_days and prediction_days must be positive")
    if config.environment_lag_days != 1:
        raise ConfigurationError("The selected protocol requires environment_lag_days=1")
    if min(config.mixed_layer_minimum_meters, config.seawater_density,
           config.seawater_heat_capacity) <= 0:
        raise ConfigurationError("Physical constants and the MLD lower bound must be positive")
    if not 0 < config.minimum_interpolation_weight <= 1:
        raise ConfigurationError("minimum_interpolation_weight must be in (0, 1]")
    if config.chunk_days < 1 or not 0 <= config.compression_level <= 9:
        raise ConfigurationError("chunk_days must be positive; compression_level must be 0..9")
    _validate_splits(config)


def _validate_splits(config: PreprocessConfig) -> None:
    if set(config.splits) != {"train", "validation", "test"}:
        raise ConfigurationError("splits must contain train, validation, and test")
    previous_end = None
    for name in ("train", "validation", "test"):
        start, end = map(pd.Timestamp, config.splits[name])
        if start > end or start < pd.Timestamp(config.start_date) or end > pd.Timestamp(config.end_date):
            raise ConfigurationError("Invalid or out-of-coverage split: %s" % name)
        if previous_end is not None and start <= previous_end:
            raise ConfigurationError("Overlapping or unordered split: %s" % name)
        previous_end = end


def load_config(path: Path, *, output_root: Optional[Path] = None) -> PreprocessConfig:
    """Resolve paths relative to the config file, with an optional output override."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        options = json.load(handle)
    for name in ("data_root", "output_root"):
        options[name] = (path.parent / options[name]).resolve()
    if output_root is not None:
        options["output_root"] = Path(output_root).resolve()
    for name in ("expected_grid_shape", "expected_core_shape"):
        if options.get(name) is not None:
            options[name] = tuple(options[name])
    if "splits" in options:
        options["splits"] = {key: tuple(value) for key, value in options["splits"].items()}
    config = PreprocessConfig(**options)
    validate_config(config)
    return config
