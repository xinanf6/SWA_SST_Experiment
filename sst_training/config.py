"""Explicit C0 training settings, independent of the original DSPG globals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

from sst_analysis.config import validate_output_location
from sst_preprocessing.errors import ConfigurationError


ORIGINAL_MODEL_VARIANT = "dspg"
COMPACT_MODEL_VARIANT = "parallel8"


@dataclass(frozen=True)
class TrainingConfig:
    """Defaults follow the frozen experiment; architecture changes create a new run."""

    prepared_root: Path
    output_root: Path
    seed: int = 2026
    batch_size: int = 2
    accumulation_steps: int = 2
    workers: int = 0
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    max_epochs: int = 50
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    gradient_clip: float = 1.0
    auxiliary_weight: float = 0.1
    amp: bool = True
    device: str = "auto"
    architecture: dict = field(default_factory=dict)
    model_variant: str = ORIGINAL_MODEL_VARIANT

    @property
    def model_name(self) -> str:
        return "C0_parallel8" if self.model_variant == COMPACT_MODEL_VARIANT else "C0"

    def to_dict(self) -> dict:
        result = asdict(self)
        for name in ("prepared_root", "output_root"):
            result[name] = str(result[name])
        return result

    def fingerprint(self) -> str:
        values = self.to_dict()
        values.pop("output_root")
        if self.model_variant == ORIGINAL_MODEL_VARIANT:
            # The original DSPG checkpoints predate explicit variant selection.
            values.pop("model_variant")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def load_training_config(path: Path, output_root: Path = None) -> TrainingConfig:
    """Resolve paths relative to the config file and reject ambiguous settings."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    for name in ("prepared_root", "output_root"):
        values[name] = (path.parent / values[name]).resolve()
    if output_root is not None:
        values["output_root"] = Path(output_root).resolve()
    try:
        config = TrainingConfig(**values)
    except TypeError as error:
        raise ConfigurationError("Invalid C0 training configuration: %s" % error) from error
    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    validate_output_location(config.output_root, config.prepared_root)
    _validate_types(config)
    positive = (config.batch_size, config.accumulation_steps, config.max_epochs,
                config.early_stopping_patience, config.gradient_clip, config.learning_rate)
    if any(value <= 0 for value in positive) or config.workers < 0 or config.scheduler_patience < 0:
        raise ConfigurationError("Training batch/epoch/rate settings must be positive; workers/patience nonnegative")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("Training device must be auto, cpu or cuda")
    if config.weight_decay < 0 or config.auxiliary_weight < 0:
        raise ConfigurationError("Training loss/decay weights must be nonnegative")
    if config.model_variant not in (ORIGINAL_MODEL_VARIANT, COMPACT_MODEL_VARIANT):
        raise ConfigurationError("Training model_variant must be dspg or parallel8")
    if config.model_variant == COMPACT_MODEL_VARIANT and config.architecture:
        raise ConfigurationError("parallel8 uses fixed widths; leave architecture empty")


def _validate_types(config: TrainingConfig) -> None:
    integer_names = ("seed", "batch_size", "accumulation_steps", "workers", "max_epochs",
                     "early_stopping_patience", "scheduler_patience")
    for name in integer_names:
        if type(getattr(config, name)) is not int:
            raise ConfigurationError("Training %s must be an integer" % name)
    if not 0 <= config.seed < 2 ** 32:
        raise ConfigurationError("Training seed must be in [0, 2**32)")
    for name in ("learning_rate", "weight_decay", "gradient_clip", "auxiliary_weight"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigurationError("Training %s must be a finite number" % name)
    if type(config.amp) is not bool or not isinstance(config.architecture, dict):
        raise ConfigurationError("Training amp must be boolean; architecture must be an object")
