"""DSPG backbone port; no imports from the original project or global config."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .physics import AdvectionDiffusionBlock
from .recurrent import ConvBiLSTM
from .spatial import AttentionFusion, SpatialEncoder


@dataclass(frozen=True)
class ModelSettings:
    """One recurrent layer, three horizons and original default architecture widths."""

    spatial_channels: int = 32
    global_hidden: int = 128
    pixel_hidden: int = 32
    fusion_channels: int = 256
    decoder_channels: int = 128
    decoder_hidden: int = 64

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()) or self.spatial_channels < 8:
            raise ValueError("DSPG widths must be positive; spatial_channels must be at least 8")


class DSPGBackbone(nn.Module):
    """Separated encode/decode allows thermal fusion before original PGFE evolution."""

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.spatial_encoder = SpatialEncoder(settings.spatial_channels)
        self.physics_block = AdvectionDiffusionBlock(settings.spatial_channels)
        combined = settings.spatial_channels + 1
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bilstm = nn.LSTM(combined, settings.global_hidden, batch_first=True, bidirectional=True)
        self.pixel_conv_bilstm = ConvBiLSTM({"input_dim": combined, "hidden_dim": settings.pixel_hidden})
        self.attention_fusion = AttentionFusion({"global_channels": 2 * settings.global_hidden,
                    "pixel_channels": 2 * settings.pixel_hidden, "fusion_channels": settings.fusion_channels})
        self.decoder = nn.Sequential(nn.Conv2d(settings.fusion_channels, settings.decoder_channels, 3, padding=1),
                    nn.ReLU(inplace=True), nn.Conv2d(settings.decoder_channels, settings.decoder_hidden, 3, padding=1),
                    nn.ReLU(inplace=True), nn.Conv2d(settings.decoder_hidden, 3, 1))

    def forward(self, base: torch.Tensor, features: torch.Tensor) -> tuple:
        evolved, displacements = [], []
        for step in range(base.shape[1]):
            temperature, displacement = self.physics_block(features[:, step], base[:, step, :5])
            evolved.append(torch.cat((features[:, step], temperature), dim=1))
            displacements.append(displacement)
        return self.decode(torch.stack(evolved, dim=1)), torch.stack(displacements, dim=1)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = features.shape
        pooled = self.global_pool(features.reshape(batch * steps, -1, height, width)).reshape(batch, steps, -1)
        global_sequence, _ = self.global_bilstm(pooled)
        global_context = global_sequence[:, -1, :, None, None].expand(-1, -1, height, width)
        pixel_context = self.pixel_conv_bilstm(features)[:, -1]
        fused = self.attention_fusion(global_context, pixel_context)
        return self.decoder(fused).unsqueeze(2)
