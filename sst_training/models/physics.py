"""Inherited learned pixel displacements; deliberately not SI ocean velocities."""

import torch
from torch import nn
import torch.nn.functional as functional


def displacement_network(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(32, 2, 3, padding=1), nn.Tanh())


class AdvectionDiffusionBlock(nn.Module):
    """Pin original trainable diffusion, learned correction and 5/2 pixel scales."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.velocity_net, self.residual_net = displacement_network(5), displacement_network(channels)
        self.diffusion_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            for network in (self.velocity_net, self.residual_net):
                network[-2].weight.normal_(0, 0.001)
                network[-2].bias.zero_()
            kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
            self.diffusion_conv.weight.copy_(kernel.reshape(1, 1, 3, 3) * 0.1)

    def flow_warp(self, values: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        height, width = values.shape[-2:]
        north, east = torch.meshgrid(torch.arange(height, device=values.device),
                                     torch.arange(width, device=values.device), indexing="ij")
        grid = torch.stack((east, north)).float()[None] + displacement.float()
        longitude = 2.0 * grid[:, 0] / max(width - 1, 1) - 1.0
        latitude = 2.0 * grid[:, 1] / max(height - 1, 1) - 1.0
        normalized = torch.stack((longitude, latitude), dim=-1)
        return functional.grid_sample(values, normalized, align_corners=True,
                                      mode="bilinear", padding_mode="reflection")

    def forward(self, features: torch.Tensor, inputs: torch.Tensor) -> tuple:
        displacement = self.velocity_net(inputs) * 5.0
        correction = self.residual_net(features) * 2.0
        advected = self.flow_warp(inputs[:, :1], displacement + correction)
        return advected + self.diffusion_conv(advected), displacement
