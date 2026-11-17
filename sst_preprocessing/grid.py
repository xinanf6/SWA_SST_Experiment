"""Explicit coastal interpolation and spherical, centered SST derivatives."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .config import PreprocessConfig
from .constants import EARTH_RADIUS_METERS
from .errors import DataValidationError


METERS_PER_KILOMETER = 1000.0
MINIMUM_COORDINATE_COUNT = 2


def _coordinate_vector(values: np.ndarray, name: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < MINIMUM_COORDINATE_COUNT:
        raise DataValidationError("%s must be a 1D coordinate with at least two points" % name)
    if not np.isfinite(coordinates).all() or not (np.diff(coordinates) > 0).all():
        raise DataValidationError("%s must be finite and strictly increasing" % name)
    return coordinates


def _axis_interpolation(source: np.ndarray, target: np.ndarray) -> tuple:
    lower = np.searchsorted(source, target, side="right") - 1
    lower = np.clip(lower, 0, source.size - 2)
    fraction = (target - source[lower]) / (source[lower + 1] - source[lower])
    is_inside = (target >= source[0]) & (target <= source[-1])
    return lower, fraction, is_inside


def _validated_grid(grid: Mapping[str, np.ndarray], name: str) -> Dict[str, np.ndarray]:
    missing = {"latitude", "longitude"} - set(grid)
    if missing:
        raise DataValidationError("%s grid is missing coordinates: %s" % (name, sorted(missing)))
    return {coordinate: _coordinate_vector(grid[coordinate], "%s %s" % (name, coordinate))
            for coordinate in ("latitude", "longitude")}


class GridInterpolator:
    """Reuse bilinear weights while retaining the valid coastal template weight."""

    def __init__(self, source_grid: Mapping[str, np.ndarray],
                 target_grid: Mapping[str, np.ndarray], minimum_weight: float = 0.999999) -> None:
        """Prepare source and target coordinate objects and a coastal acceptance threshold."""
        if not 0 < minimum_weight <= 1:
            raise DataValidationError("minimum_weight must be in (0, 1]")
        source_grid = _validated_grid(source_grid, "source")
        target_grid = _validated_grid(target_grid, "target")
        self.source_shape = (source_grid["latitude"].size, source_grid["longitude"].size)
        self.target_shape = (target_grid["latitude"].size, target_grid["longitude"].size)
        self.minimum_weight = minimum_weight
        self._prepare_template(source_grid, target_grid)

    def _prepare_template(self, source_grid: Dict[str, np.ndarray],
                          target_grid: Dict[str, np.ndarray]) -> None:
        south, north_fraction, is_inside_y = _axis_interpolation(source_grid["latitude"], target_grid["latitude"])
        west, east_fraction, is_inside_x = _axis_interpolation(source_grid["longitude"], target_grid["longitude"])
        north_fraction = north_fraction[:, None]
        east_fraction = east_fraction[None, :]
        self._is_inside = is_inside_y[:, None] & is_inside_x[None, :]
        self._corners = (
            (south, west, (1 - north_fraction) * (1 - east_fraction)),
            (south, west + 1, (1 - north_fraction) * east_fraction),
            (south + 1, west, north_fraction * (1 - east_fraction)),
            (south + 1, west + 1, north_fraction * east_fraction),
        )

    def interpolate(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate 2D or daily 3D arrays; return values and finite-source weights."""
        values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
        if values.ndim not in (2, 3) or values.shape[-2:] != self.source_shape:
            raise DataValidationError("Interpolation input shape %s does not match %s" %
                                      (values.shape, self.source_shape))
        result_shape = values.shape[:-2] + self.target_shape
        numerator = np.zeros(result_shape, dtype=np.float64)
        valid_weight = np.zeros(result_shape, dtype=np.float64)
        for latitude_index, longitude_index, corner_weight in self._corners:
            corner = values[..., latitude_index[:, None], longitude_index[None, :]]
            is_finite = np.isfinite(corner)
            numerator += np.where(is_finite, corner, 0.0) * corner_weight
            valid_weight += is_finite * corner_weight
        valid_weight = np.where(self._is_inside, valid_weight, 0.0)
        is_accepted = self._is_inside & (valid_weight >= self.minimum_weight)
        result = np.full(result_shape, np.nan, dtype=np.float64)
        np.divide(numerator, valid_weight, out=result, where=is_accepted)
        return result.astype(np.float32), valid_weight.astype(np.float32)


def build_grid_metadata(latitude: np.ndarray, longitude: np.ndarray,
                        config: PreprocessConfig) -> Dict[str, np.ndarray]:
    """Preserve the observed grid and validate the requested core, without resampling."""
    latitude = _coordinate_vector(latitude, "OSTIA latitude")
    longitude = _coordinate_vector(longitude, "OSTIA longitude")
    grid_shape = (latitude.size, longitude.size)
    is_core_latitude = (latitude >= config.core_south) & (latitude <= config.core_north)
    is_core_longitude = (longitude >= config.core_west) & (longitude <= config.core_east)
    core_shape = (int(is_core_latitude.sum()), int(is_core_longitude.sum()))
    _validate_grid_shape(grid_shape, config.expected_grid_shape, "input")
    _validate_grid_shape(core_shape, config.expected_core_shape, "core")
    if not all(core_shape):
        raise DataValidationError("The configured core does not intersect the OSTIA grid")
    latitude_weights = np.cos(np.deg2rad(latitude))[:, None]
    return {
        "latitude": latitude.copy(), "longitude": longitude.copy(),
        "core_mask": is_core_latitude[:, None] & is_core_longitude[None, :],
        "area_weights": np.broadcast_to(latitude_weights, grid_shape).astype(np.float32).copy(),
    }


def _validate_grid_shape(actual: tuple, expected: Optional[Tuple[int, int]], name: str) -> None:
    if expected is not None and actual != tuple(expected):
        raise DataValidationError("OSTIA %s grid shape %s does not match expected %s" %
                                  (name, actual, tuple(expected)))


def gradient_valid_mask(ocean_mask: np.ndarray) -> np.ndarray:
    """Require the center and all four ocean neighbors for a centered derivative."""
    ocean_mask = np.asarray(ocean_mask, dtype=bool)
    if ocean_mask.ndim != 2:
        raise DataValidationError("The gradient ocean mask must be two-dimensional")
    result = np.zeros_like(ocean_mask)
    result[1:-1, 1:-1] = (ocean_mask[1:-1, 1:-1] & ocean_mask[:-2, 1:-1] &
                          ocean_mask[2:, 1:-1] & ocean_mask[1:-1, :-2] &
                          ocean_mask[1:-1, 2:])
    return result


def temperature_gradients(temperature: np.ndarray, latitude: np.ndarray,
                          longitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eastward, northward and magnitude gradients in degrees Celsius per km."""
    latitude = _coordinate_vector(latitude, "gradient latitude")
    longitude = _coordinate_vector(longitude, "gradient longitude")
    temperature = np.asarray(temperature, dtype=np.float64)
    if temperature.ndim not in (2, 3) or temperature.shape[-2:] != (latitude.size, longitude.size):
        raise DataValidationError("Temperature dimensions do not match gradient coordinates")
    gradient_x = np.full(temperature.shape, np.nan, dtype=np.float64)
    gradient_y = np.full(temperature.shape, np.nan, dtype=np.float64)
    longitude_span = np.deg2rad(longitude[2:] - longitude[:-2])[None, :]
    latitude_span = np.deg2rad(latitude[2:] - latitude[:-2])[:, None]
    eastward_distance = EARTH_RADIUS_METERS * np.cos(np.deg2rad(latitude[1:-1]))[:, None] * longitude_span
    northward_distance = EARTH_RADIUS_METERS * latitude_span
    gradient_x[..., 1:-1, 1:-1] = (temperature[..., 1:-1, 2:] - temperature[..., 1:-1, :-2]) / eastward_distance * METERS_PER_KILOMETER
    gradient_y[..., 1:-1, 1:-1] = (temperature[..., 2:, 1:-1] - temperature[..., :-2, 1:-1]) / northward_distance * METERS_PER_KILOMETER
    is_valid = _finite_gradient_template(temperature)
    gradient_x[~is_valid] = np.nan
    gradient_y[~is_valid] = np.nan
    magnitude = np.hypot(gradient_x, gradient_y)
    return tuple(values.astype(np.float32) for values in (gradient_x, gradient_y, magnitude))


def _finite_gradient_template(values: np.ndarray) -> np.ndarray:
    is_finite = np.isfinite(values)
    is_valid = np.zeros(values.shape, dtype=bool)
    is_valid[..., 1:-1, 1:-1] = (is_finite[..., 1:-1, 1:-1] & is_finite[..., :-2, 1:-1] &
                                 is_finite[..., 2:, 1:-1] & is_finite[..., 1:-1, :-2] &
                                 is_finite[..., 1:-1, 2:])
    return is_valid


def compute_temperature_gradients(temperature: np.ndarray, latitude: np.ndarray,
                                  longitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the three gradients and their per-day centered finite-source mask."""
    gradient_x, gradient_y, magnitude = temperature_gradients(temperature, latitude, longitude)
    return gradient_x, gradient_y, magnitude, np.isfinite(magnitude)
