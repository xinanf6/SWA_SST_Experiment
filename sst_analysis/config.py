"""Portable paths and output safety rules for the analysis stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sst_preprocessing.errors import ConfigurationError


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis settings; numerical masks/thresholds always come from preprocessing."""

    prepared_root: Path
    diagnostics_output: Path
    figure_dpi: int = 300
    batch_size: int = 1


def load_analysis_config(path: Path, prepared_root: Optional[Path] = None) -> AnalysisConfig:
    """Resolve configuration paths relative to the JSON file, not the terminal cwd."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        options = json.load(handle)
    for name in ("prepared_root", "diagnostics_output"):
        options[name] = (path.parent / options[name]).resolve()
    if prepared_root is not None:
        options["prepared_root"] = Path(prepared_root).resolve()
    settings = AnalysisConfig(**options)
    if settings.figure_dpi < 72 or settings.batch_size < 1:
        raise ConfigurationError("figure_dpi must be at least 72 and batch_size must be positive")
    validate_output_location(settings.diagnostics_output, settings.prepared_root)
    return settings


def validate_output_location(output_root: Path, prepared_root: Path) -> None:
    """Keep diagnostic/evaluation writes outside the immutable preprocessing output."""
    output_root, prepared_root = Path(output_root).resolve(), Path(prepared_root).resolve()
    if output_root == prepared_root or prepared_root in output_root.parents:
        raise ConfigurationError("Analysis output must be outside the preprocessed data directory")
    if output_root in prepared_root.parents:
        raise ConfigurationError("Analysis output must not be an ancestor of the preprocessed directory")
