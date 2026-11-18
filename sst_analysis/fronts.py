"""Raw skeleton front locations and spherical, bidirectional matching metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from sst_preprocessing.constants import EARTH_RADIUS_METERS
from sst_preprocessing.errors import DataValidationError


EARTH_RADIUS_KM = EARTH_RADIUS_METERS / 1000.0
DEFAULT_TOLERANCE_KM = 10.0
MINIMUM_THINNING_NEIGHBORS = 2
MAXIMUM_THINNING_NEIGHBORS = 6


@dataclass(frozen=True)
class _FrontGrid:
    latitude: np.ndarray
    longitude: np.ndarray
    valid_mask: np.ndarray
    core_mask: np.ndarray
    threshold: float
    tolerance_km: float


def front_line_scores(prediction_gradient: np.ndarray, target_gradient: np.ndarray,
                      grid: dict) -> dict:
    """Score raw core front skeletons in km from 2D gradient magnitudes in C/km.

    Skeletonization uses the entire valid input domain before the core crop.
    Precision and recall count skeleton points, without component filtering or
    area weighting. Undefined empty-line scores and distances are ``None``.
    """
    front_grid = _prepare_grid(grid)
    predicted_points = _front_points(prediction_gradient, front_grid, "prediction")
    target_points = _front_points(target_gradient, front_grid, "target")
    result = _empty_scores(len(predicted_points), len(target_points))
    if not len(predicted_points) or not len(target_points):
        return result
    predicted_distances = _nearest_distances(predicted_points, target_points)
    target_distances = _nearest_distances(target_points, predicted_points)
    return _matched_scores(result, (predicted_distances, target_distances), front_grid)


def _prepare_grid(grid: dict) -> _FrontGrid:
    required = {"latitude", "longitude", "gradient_mask", "core_mask", "front_threshold"}
    missing = required - set(grid)
    if missing:
        raise DataValidationError("Front evaluation grid is missing: %s" % sorted(missing))
    latitude = _coordinate(grid["latitude"], "latitude")
    longitude = _coordinate(grid["longitude"], "longitude")
    shape = (latitude.size, longitude.size)
    gradient_mask = _mask(grid["gradient_mask"], shape, "gradient_mask")
    core_mask = _mask(grid["core_mask"], shape, "core_mask")
    ocean_mask = _mask(grid.get("ocean_mask", np.ones(shape, dtype=bool)), shape, "ocean_mask")
    threshold = _positive_scalar(grid["front_threshold"], "front_threshold")
    tolerance = _positive_scalar(grid.get("front_tolerance_km", DEFAULT_TOLERANCE_KM),
                                 "front_tolerance_km", allow_zero=True)
    valid_mask = gradient_mask & ocean_mask
    if not np.any(valid_mask & core_mask):
        raise DataValidationError("Front evaluation has no valid core gradient points")
    return _FrontGrid(latitude, longitude, valid_mask, core_mask, threshold, tolerance)


def _coordinate(values: np.ndarray, name: str) -> np.ndarray:
    coordinate = np.asarray(values, dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size == 0 or not np.isfinite(coordinate).all():
        raise DataValidationError("Front %s must be a nonempty finite 1D coordinate" % name)
    differences = np.diff(coordinate)
    if differences.size and not (np.all(differences > 0) or np.all(differences < 0)):
        raise DataValidationError("Front %s must be strictly monotonic" % name)
    lower, upper = (-90.0, 90.0) if name == "latitude" else (-180.0, 360.0)
    if np.any(coordinate < lower) or np.any(coordinate > upper):
        raise DataValidationError("Front %s is outside [%s, %s]" % (name, lower, upper))
    if name == "longitude" and np.ptp(coordinate) >= 360.0:
        raise DataValidationError("Front longitude must not duplicate the 360-degree meridian")
    return coordinate


def _mask(values: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.shape != shape:
        raise DataValidationError("Front %s shape %s does not match %s" % (name, mask.shape, shape))
    if not np.isin(mask, [False, True]).all():
        raise DataValidationError("Front %s must contain only boolean mask values" % name)
    return mask.astype(bool, copy=False)


def _positive_scalar(value: float, name: str, allow_zero: bool = False) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Front %s must be a finite scalar" % name) from error
    is_valid = np.isfinite(scalar) and (scalar >= 0 if allow_zero else scalar > 0)
    if not is_valid:
        raise DataValidationError("Front %s must be finite and %s" %
                                  (name, "nonnegative" if allow_zero else "positive"))
    return scalar


def _front_points(values: np.ndarray, grid: _FrontGrid, name: str) -> np.ndarray:
    gradient = np.asarray(values, dtype=np.float64)
    if gradient.shape != grid.valid_mask.shape:
        raise DataValidationError("Front %s gradient shape %s does not match %s" %
                                  (name, gradient.shape, grid.valid_mask.shape))
    if not np.isfinite(gradient[grid.valid_mask]).all():
        raise DataValidationError("Front %s gradient has nonfinite valid core/buffer values" % name)
    if np.any(gradient[grid.valid_mask] < 0):
        raise DataValidationError("Front %s gradient magnitude is negative on the valid domain" % name)
    binary_front = grid.valid_mask & np.isfinite(gradient) & (gradient >= grid.threshold)
    # Cropping before skeletonization changes endpoints and topology at the core boundary.
    skeleton = _skeletonize(binary_front) & grid.core_mask
    row_indices, column_indices = np.nonzero(skeleton)
    return _earth_coordinates(grid.latitude[row_indices], grid.longitude[column_indices])


def _skeletonize(binary_front: np.ndarray) -> np.ndarray:
    """Apply the original Zhang-Suen two-subiteration rules until convergence.

    Reference: Zhang & Suen (1984), doi:10.1145/357994.358023, section 3.
    Paper: https://teaching.vda.univie.ac.at/ipa/18w/Lab3/Zhan-Suen_algorithm.pdf
    The implementation uses NumPy, not scikit-image's lookup-table variant.
    Each subiteration deletes simultaneously from its own starting state.
    """
    skeleton = np.pad(np.asarray(binary_front, dtype=bool), 1)
    maximum_iterations = int(skeleton.sum()) + 1
    for _ in range(maximum_iterations):
        has_changed = _thin_pass(skeleton, is_second_pass=False)
        has_changed = _thin_pass(skeleton, is_second_pass=True) or has_changed
        if not has_changed:
            return skeleton[1:-1, 1:-1].copy()
    raise DataValidationError("Front thinning failed to converge within its pixel deletion bound")


def _thin_pass(skeleton: np.ndarray, is_second_pass: bool) -> bool:
    neighbors = _clockwise_neighbors(skeleton)
    neighbor_count = np.add.reduce(neighbors, dtype=np.uint8)
    transitions = np.add.reduce([~neighbors[index] & neighbors[(index + 1) % 8]
                                 for index in range(8)], dtype=np.uint8)
    has_valid_neighbors = ((neighbor_count >= MINIMUM_THINNING_NEIGHBORS) &
                           (neighbor_count <= MAXIMUM_THINNING_NEIGHBORS))
    north, east, south, west = (neighbors[index] for index in (0, 2, 4, 6))
    if is_second_pass:
        can_delete_direction = ~(north & east & west) & ~(north & south & west)
    else:
        can_delete_direction = ~(north & east & south) & ~(east & south & west)
    center = skeleton[1:-1, 1:-1]
    can_delete = center & has_valid_neighbors & (transitions == 1) & can_delete_direction
    has_changed = bool(np.any(can_delete))
    center[can_delete] = False
    return has_changed


def _clockwise_neighbors(skeleton: np.ndarray) -> tuple:
    return (skeleton[:-2, 1:-1], skeleton[:-2, 2:], skeleton[1:-1, 2:],
            skeleton[2:, 2:], skeleton[2:, 1:-1], skeleton[2:, :-2],
            skeleton[1:-1, :-2], skeleton[:-2, :-2])


def _earth_coordinates(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude_radians, longitude_radians = np.deg2rad(latitude), np.deg2rad(longitude)
    horizontal_radius = EARTH_RADIUS_KM * np.cos(latitude_radians)
    return np.column_stack((horizontal_radius * np.cos(longitude_radians),
                            horizontal_radius * np.sin(longitude_radians),
                            EARTH_RADIUS_KM * np.sin(latitude_radians)))


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    chord_distances, _ = cKDTree(target).query(source, k=1)
    # Chord distance is monotonic with great-circle distance, so the nearest node is exact.
    sine_half_angle = np.clip(chord_distances / (2.0 * EARTH_RADIUS_KM), 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(sine_half_angle)


def _empty_scores(predicted_count: int, target_count: int) -> dict:
    has_prediction, has_target = predicted_count > 0, target_count > 0
    return {
        "front_line_precision": 0.0 if has_prediction else None,
        "front_line_recall": 0.0 if has_target else None,
        "front_line_f1": 0.0 if has_prediction or has_target else None,
        "front_distance_km": None,
        "predicted_line_points": predicted_count,
        "target_line_points": target_count,
        "predicted_matched_points": 0,
        "target_matched_points": 0,
        "empty_predicted_line": not has_prediction,
        "empty_target_line": not has_target,
    }


def _matched_scores(result: dict, distances: tuple, grid: _FrontGrid) -> dict:
    predicted_distances, target_distances = distances
    predicted_matches = int(np.count_nonzero(predicted_distances <= grid.tolerance_km))
    target_matches = int(np.count_nonzero(target_distances <= grid.tolerance_km))
    precision = predicted_matches / result["predicted_line_points"]
    recall = target_matches / result["target_line_points"]
    result.update({
        "front_line_precision": precision,
        "front_line_recall": recall,
        "front_line_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "front_distance_km": float((predicted_distances.mean() + target_distances.mean()) / 2.0),
        "predicted_matched_points": predicted_matches,
        "target_matched_points": target_matches,
    })
    return result
