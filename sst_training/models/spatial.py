"""DSPG spatial and attention fusion blocks, preserving original parameter keys."""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // 16, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(channels // 16, channels, 1, bias=False))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (self.fc(self.avg_pool(values)) + self.fc(self.max_pool(values))).sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        summaries = torch.cat((values.mean(1, keepdim=True), values.max(1, keepdim=True)[0]), dim=1)
        return self.conv1(summaries).sigmoid()


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ca, self.sa = ChannelAttention(channels), SpatialAttention()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.ca(values) * values
        return self.sa(values) * values


class SpatialEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3x3 = nn.Conv2d(7, channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(7, channels, 5, padding=2)
        self.conv7x7 = nn.Conv2d(7, channels, 7, padding=3)
        self.cbam, self.final_conv = CBAM(3 * channels), nn.Conv2d(3 * channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = values.shape
        flat = values.reshape(batch * steps, 7, height, width)
        features = [layer(flat).relu() for layer in (self.conv3x3, self.conv5x5, self.conv7x7)]
        features = self.final_conv(self.cbam(torch.cat(features, dim=1))).relu()
        return features.reshape(batch, steps, -1, height, width)


class AttentionFusion(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        global_channels, pixel_channels = options["global_channels"], options["pixel_channels"]
        self.map_global_to_pixel = nn.Conv2d(global_channels, pixel_channels, 1)
        fused = global_channels + pixel_channels
        self.bottleneck = nn.Sequential(nn.Conv2d(fused, fused // 2, 3, padding=1), nn.ReLU(inplace=True),
                                        nn.Conv2d(fused // 2, options["fusion_channels"], 1))

    def forward(self, global_context: torch.Tensor, pixel_context: torch.Tensor) -> torch.Tensor:
        attention = (self.map_global_to_pixel(global_context) + pixel_context).sigmoid()
        return self.bottleneck(torch.cat((global_context, pixel_context * attention), dim=1))
