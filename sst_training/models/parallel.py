"""Full-grid, fixed-width parallel temporal backbone for the C0 experiment."""

import torch
from torch import nn

from .physics import AdvectionDiffusionBlock
from .spatial import AttentionFusion


HISTORY_STEPS = 14
FEATURE_CHANNELS = 8
SPATIAL_BRANCH_CHANNELS = 4
SPATIAL_DILATIONS = (1, 2, 3)
DISPLACEMENT_CHANNELS = 4
GLOBAL_HIDDEN_CHANNELS = 16
LOCAL_CHANNELS = 16
LOCAL_CONVOLUTION_GROUPS = 4
FUSION_CHANNELS = 24
DECODER_HIDDEN_CHANNELS = 12
FORECAST_LEADS = 3


def compact_architecture() -> dict:
    """Describe the fixed widths and spatial operators in checkpoint metadata."""
    return {"spatial_channels": FEATURE_CHANNELS, "spatial_branch_channels": SPATIAL_BRANCH_CHANNELS,
            "spatial_dilations": list(SPATIAL_DILATIONS), "thermal_hidden": FEATURE_CHANNELS,
            "displacement_channels": DISPLACEMENT_CHANNELS, "global_hidden": GLOBAL_HIDDEN_CHANNELS,
            "local_channels": LOCAL_CHANNELS, "local_groups": LOCAL_CONVOLUTION_GROUPS,
            "fusion_channels": FUSION_CHANNELS, "decoder_hidden": DECODER_HIDDEN_CHANNELS}


class DilatedSpatialEncoder(nn.Module):
    """Encode each day at native resolution with three receptive field spans."""

    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(7, SPATIAL_BRANCH_CHANNELS, 3, dilation=rate, padding=rate)
            for rate in SPATIAL_DILATIONS)
        self.project = nn.Conv2d(len(SPATIAL_DILATIONS) * SPATIAL_BRANCH_CHANNELS, FEATURE_CHANNELS, 1)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        daily = base.flatten(0, 1)
        features = torch.cat([branch(daily).relu() for branch in self.branches], dim=1)
        return self.project(features).relu().unflatten(0, base.shape[:2])


class ParallelTemporalFusion(nn.Module):
    """Mix all ordered lags at every grid cell, then propagate at full resolution."""

    def __init__(self) -> None:
        super().__init__()
        channels_per_day = FEATURE_CHANNELS + 1
        self.mix = nn.Conv2d(HISTORY_STEPS * channels_per_day, LOCAL_CHANNELS, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(LOCAL_CHANNELS, LOCAL_CHANNELS, 3, padding=1, groups=LOCAL_CONVOLUTION_GROUPS),
            nn.ReLU(inplace=True),
            nn.Conv2d(LOCAL_CHANNELS, LOCAL_CHANNELS, 3, padding=2, dilation=2, groups=LOCAL_CONVOLUTION_GROUPS))

    def forward(self, daily_features: torch.Tensor) -> torch.Tensor:
        # Lag order is encoded in the input channels, not averaged across time.
        local = self.mix(daily_features.flatten(1, 2)).relu()
        return local + self.spatial(local)


class CompactParallelBackbone(nn.Module):
    """Preserve the original displacement and global context with a parallel local branch."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_encoder = DilatedSpatialEncoder()
        self.physics_block = AdvectionDiffusionBlock(FEATURE_CHANNELS, DISPLACEMENT_CHANNELS)
        channels_per_day = FEATURE_CHANNELS + 1
        self.global_bilstm = nn.LSTM(channels_per_day, GLOBAL_HIDDEN_CHANNELS, batch_first=True,
                                    bidirectional=True)
        self.temporal_fusion = ParallelTemporalFusion()
        self.attention_fusion = AttentionFusion({
            "global_channels": 2 * GLOBAL_HIDDEN_CHANNELS, "pixel_channels": LOCAL_CHANNELS,
            "fusion_channels": FUSION_CHANNELS})
        self.decoder = nn.Sequential(
            nn.Conv2d(FUSION_CHANNELS, FUSION_CHANNELS, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(FUSION_CHANNELS, DECODER_HIDDEN_CHANNELS, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(DECODER_HIDDEN_CHANNELS, FORECAST_LEADS, 1))

    def forward(self, base: torch.Tensor, features: torch.Tensor) -> tuple:
        # Each daily physics calculation is independent, so use a single batched warp.
        temperatures, displacement = self.physics_block(features.flatten(0, 1), base[:, :, :5].flatten(0, 1))
        daily_features = torch.cat((features, temperatures.unflatten(0, base.shape[:2])), dim=2)
        return self.decode(daily_features), displacement.unflatten(0, base.shape[:2])

    def decode(self, daily_features: torch.Tensor) -> torch.Tensor:
        pooled = daily_features.mean(dim=(-1, -2))
        global_history, _ = self.global_bilstm(pooled)
        height, width = daily_features.shape[-2:]
        context = global_history[:, -1, :, None, None].expand(-1, -1, height, width)
        local = self.temporal_fusion(daily_features)
        return self.decoder(self.attention_fusion(context, local)).unsqueeze(2)
