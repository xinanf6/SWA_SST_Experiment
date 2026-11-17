"""Daily net surface heating and training-climatology MLD gap filling."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import PreprocessConfig
from .constants import ENERGY_VARIABLES, SECONDS_PER_DAY
from .errors import DataValidationError


MONTHS_PER_YEAR = 12


def compute_heat_features(energy_components: Dict[str, np.ndarray],
                          mixed_layer_depth: np.ndarray,
                          config: PreprocessConfig) -> Dict[str, np.ndarray]:
    """Apply the original 3 m floor to already integrated daily energy in J/m²."""
    missing = sorted(set(ENERGY_VARIABLES) - set(energy_components))
    if missing:
        raise DataValidationError("Missing daily heat-energy components: %s" % missing)
    mixed_layer_depth = np.asarray(mixed_layer_depth, dtype=np.float64)
    net_energy = np.zeros_like(mixed_layer_depth)
    for name in ENERGY_VARIABLES:
        energy = np.asarray(energy_components[name], dtype=np.float64)
        if energy.shape != mixed_layer_depth.shape:
            raise DataValidationError("%s energy shape %s differs from MLD shape %s" %
                                      (name, energy.shape, mixed_layer_depth.shape))
        net_energy += energy
    effective_depth = np.maximum(mixed_layer_depth, config.mixed_layer_minimum_meters)
    is_valid_depth = np.isfinite(mixed_layer_depth) & (mixed_layer_depth > 0)
    effective_depth = np.where(is_valid_depth, effective_depth, np.nan)
    # Energy is already integrated over one day; a second 86400 factor changes its meaning.
    heat_increment = net_energy / (config.seawater_density * config.seawater_heat_capacity * effective_depth)
    return {"net_heat_flux": (net_energy / SECONDS_PER_DAY).astype(np.float32),
            "heat_increment": heat_increment.astype(np.float32)}


def fill_mixed_layer_depth(values: np.ndarray, month: int,
                           climatology: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Fill invalid ocean MLD from training months; preserve every positive depth."""
    if not isinstance(month, (int, np.integer)) or not 1 <= month <= MONTHS_PER_YEAR:
        raise DataValidationError("MLD climatology month must be an integer in 1..12")
    values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
    ocean_mask = np.asarray(climatology["ocean_mask"], dtype=bool)
    if values.ndim not in (2, 3) or values.shape[-2:] != ocean_mask.shape:
        raise DataValidationError("MLD shape %s differs from source ocean mask %s" %
                                  (values.shape, ocean_mask.shape))
    replacement = _monthly_replacement(month, climatology, ocean_mask)
    is_ocean = np.broadcast_to(ocean_mask, values.shape)
    should_fill = is_ocean & (~np.isfinite(values) | (values <= 0))
    filled = np.where(should_fill, replacement, values)
    if not np.isfinite(filled[is_ocean]).all() or not (filled[is_ocean] > 0).all():
        raise DataValidationError("Training MLD climatology cannot fill ocean gaps for month %02d" % month)
    filled = np.where(is_ocean, filled, np.nan)
    return filled.astype(np.float32), should_fill


def _monthly_replacement(month: int, climatology: dict, ocean_mask: np.ndarray) -> np.ndarray:
    monthly_mean = np.asarray(climatology["monthly_mean"], dtype=np.float64)
    if monthly_mean.shape != (MONTHS_PER_YEAR,) + ocean_mask.shape:
        raise DataValidationError("MLD monthly_mean must have shape (12, latitude, longitude)")
    monthly_field = monthly_mean[month - 1]
    monthly_median = np.asarray(climatology["monthly_median"])[month - 1]
    is_valid_local = np.isfinite(monthly_field) & (monthly_field > 0) & ocean_mask
    is_valid_median = bool(np.isfinite(monthly_median) and monthly_median > 0)
    if is_valid_local.any() or is_valid_median:
        fallback = monthly_median if is_valid_median else np.nan
        return np.where(is_valid_local, monthly_field, fallback)
    # The annual path permits intentionally short offline fixtures, never partial-month gaps.
    annual_field = np.asarray(climatology["annual_mean"], dtype=np.float64)
    annual_median = climatology["annual_median"]
    if annual_field.shape != ocean_mask.shape:
        raise DataValidationError("MLD annual_mean shape differs from the source ocean mask")
    is_valid_annual = np.isfinite(annual_field) & (annual_field > 0) & ocean_mask
    fallback = annual_median if np.isfinite(annual_median) and annual_median > 0 else np.nan
    return np.where(is_valid_annual, annual_field, fallback)
