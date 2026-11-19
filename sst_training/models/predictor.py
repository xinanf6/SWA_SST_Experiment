"""Select the C0 residual head or the seven-variable direct SST reference."""

import torch
from torch import nn
from dataclasses import asdict

from sst_preprocessing.statistics import _normalization_scale

from ..config import BASELINE_MODEL_VARIANT, COMPACT_MODEL_VARIANT, ORIGINAL_MODEL_VARIANT, TrainingConfig
from .dspg import DSPGBackbone, ModelSettings
from .parallel import CompactParallelBackbone, FEATURE_CHANNELS, compact_architecture


def build_predictor(config: TrainingConfig, statistics: dict) -> nn.Module:
    """Select the same model for training, restart and forecast export."""
    settings = ModelSettings(**config.architecture)
    if config.model_variant == BASELINE_MODEL_VARIANT:
        return DSPG7Predictor(settings, statistics)
    return C0Predictor(settings, statistics, config.model_variant)


def describe_architecture(config: TrainingConfig) -> dict:
    """Record resolved model settings even when configuration uses defaults."""
    if config.model_variant == COMPACT_MODEL_VARIANT:
        return compact_architecture()
    return asdict(ModelSettings(**config.architecture))


class C0Predictor(nn.Module):
    """All returned temperatures and residuals are Celsius; PGFE stays standardized."""

    def __init__(self, settings: ModelSettings, statistics: dict,
                 model_variant: str = ORIGINAL_MODEL_VARIANT) -> None:
        super().__init__()
        is_compact = model_variant == COMPACT_MODEL_VARIANT
        self.backbone = CompactParallelBackbone() if is_compact else DSPGBackbone(settings)
        feature_channels = FEATURE_CHANNELS if is_compact else settings.spatial_channels
        thermal_hidden = FEATURE_CHANNELS if is_compact else 16
        # Isolate extra-module RNG so its initialization does not consume backbone seed state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + 17031) % (2 ** 63))
            self.thermal_encoder = nn.Sequential(nn.Conv2d(3, thermal_hidden, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(thermal_hidden, feature_channels, 3, padding=1), nn.ReLU())
            self.thermal_projection = nn.Conv2d(feature_channels, feature_channels, 1)
        self.thermal_gain = nn.Parameter(torch.tensor(0.1))
        scale = _normalization_scale(statistics["variables"]["sst"]["std"])
        self.register_buffer("sst_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, batch: dict) -> dict:
        base, thermal = batch["base"], batch["thermal"]
        _validate_inputs(base, thermal, batch["last_sst"])
        features = self.backbone.spatial_encoder(base)
        count, steps, _, height, width = thermal.shape
        encoded = self.thermal_projection(self.thermal_encoder(thermal.reshape(count * steps, 3, height, width)))
        features = features + self.thermal_gain * encoded.reshape_as(features)
        residual, displacement = self.backbone(base, features)
        residual = residual.float() * self.sst_scale
        prediction = batch["last_sst"].float()[:, None] + residual
        return {"predictions": prediction, "residual": residual, "displacement_pixels": displacement}


class DSPG7Predictor(nn.Module):
    """Preserve the seven-channel backbone and decode absolute SST in Celsius."""

    def __init__(self, settings: ModelSettings, statistics: dict) -> None:
        super().__init__()
        self.backbone = DSPGBackbone(settings)
        sst = statistics["variables"]["sst"]
        self.register_buffer("sst_scale", torch.tensor(_normalization_scale(sst["std"]), dtype=torch.float32))
        self.register_buffer("sst_mean", torch.tensor(sst["mean"], dtype=torch.float32))

    def forward(self, batch: dict) -> dict:
        base = batch["base"]
        _validate_base(base, batch["last_sst"])
        normalized_sst, displacement = self.backbone(base, self.backbone.spatial_encoder(base))
        prediction = normalized_sst.float() * self.sst_scale + self.sst_mean
        # Export the change from origin SST under the shared forecast schema.
        change = prediction - batch["last_sst"].float()[:, None]
        return {"predictions": prediction, "residual": change, "displacement_pixels": displacement}


def _validate_inputs(base: torch.Tensor, thermal: torch.Tensor, last_sst: torch.Tensor) -> None:
    _validate_base(base, last_sst)
    if thermal.shape != (base.shape[0], 14, 3) + tuple(base.shape[-2:]):
        raise ValueError("C0 thermal must have shape [B,14,3,H,W] on the same grid")


def _validate_base(base: torch.Tensor, last_sst: torch.Tensor) -> None:
    if base.ndim != 5 or base.shape[1:3] != (14, 7):
        raise ValueError("base must have shape [B,14,7,H,W]")
    if last_sst.shape != (base.shape[0], 1) + tuple(base.shape[-2:]):
        raise ValueError("last_sst must have shape [B,1,H,W] in Celsius")
