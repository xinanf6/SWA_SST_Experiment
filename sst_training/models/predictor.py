"""C0: identical thermal information, residual output, no explicit heat addition."""

import torch
from torch import nn

from sst_preprocessing.statistics import _normalization_scale

from .dspg import DSPGBackbone, ModelSettings


class C0Predictor(nn.Module):
    """All returned temperatures and residuals are Celsius; PGFE stays standardized."""

    def __init__(self, settings: ModelSettings, statistics: dict) -> None:
        super().__init__()
        self.backbone = DSPGBackbone(settings)
        # Isolate extra-module RNG so its initialization does not consume backbone seed state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + 17031) % (2 ** 63))
            self.thermal_encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(16, settings.spatial_channels, 3, padding=1), nn.ReLU())
            self.thermal_projection = nn.Conv2d(settings.spatial_channels, settings.spatial_channels, 1)
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


def _validate_inputs(base: torch.Tensor, thermal: torch.Tensor, last_sst: torch.Tensor) -> None:
    if base.ndim != 5 or base.shape[1:3] != (14, 7):
        raise ValueError("C0 base must have shape [B,14,7,H,W]")
    if thermal.shape != (base.shape[0], 14, 3) + tuple(base.shape[-2:]):
        raise ValueError("C0 thermal must have shape [B,14,3,H,W] on the same grid")
    if last_sst.shape != (base.shape[0], 1) + tuple(base.shape[-2:]):
        raise ValueError("C0 last_sst must have shape [B,1,H,W] in Celsius")
