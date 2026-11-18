"""C0 core-ocean Celsius MSE plus the frozen inherited auxiliary objective."""

import torch
from torch import nn

from sst_preprocessing.statistics import _normalization_scale


class C0Loss(nn.Module):
    """Do not interpret the standardized-current auxiliary loss as an SI PDE."""

    def __init__(self, context, auxiliary_weight: float = 0.1) -> None:
        super().__init__()
        mask = context.grid["core_mask"] & context.grid["ocean_mask"]
        weights = context.grid["area_weights"] * mask
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        info = context.statistics["variables"]["sst"]
        self.register_buffer("sst_mean", torch.tensor(info["mean"], dtype=torch.float32))
        self.register_buffer("sst_scale", torch.tensor(_normalization_scale(info["std"]), dtype=torch.float32))
        self.auxiliary_weight = auxiliary_weight

    def forward(self, output: dict, batch: dict) -> dict:
        target = batch["target"].float() * self.sst_scale + self.sst_mean
        errors = output["predictions"].float() - target
        errors = torch.where(self.weights > 0, errors, torch.zeros_like(errors))
        sst_numerator = (errors.square() * self.weights).sum()
        sst_denominator = errors.shape[0] * errors.shape[1] * self.weights.sum()
        displacement_error = output["displacement_pixels"].float() - batch["base"][:, :, 3:5].float()
        displacement_error = torch.where(self.weights > 0, displacement_error, torch.zeros_like(displacement_error))
        auxiliary_numerator = (displacement_error.square() * self.weights).sum()
        auxiliary_denominator = displacement_error.shape[0] * 14 * 2 * self.weights.sum()
        sst, auxiliary = sst_numerator / sst_denominator, auxiliary_numerator / auxiliary_denominator
        total = sst + self.auxiliary_weight * auxiliary
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite C0 loss on the fixed core ocean domain")
        return {"total": total, "sst": sst, "auxiliary": auxiliary,
                "sst_numerator": sst_numerator, "sst_denominator": sst_denominator,
                "auxiliary_numerator": auxiliary_numerator, "auxiliary_denominator": auxiliary_denominator}
