"""Streaming SST and front metrics in physical degrees Celsius on fixed masks."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.grid import gradient_valid_mask, temperature_gradients


FORECAST_LEAD_COUNT = 3
SUM_FIELDS = (
    "sample_count", "valid_count", "weight_sum", "weighted_sse",
    "weighted_absolute_error", "weighted_error_sum", "weighted_target_sum",
    "weighted_target_square_sum", "strong_front_count", "strong_front_weight_sum",
    "strong_front_weighted_sse", "strong_front_weighted_absolute_error",
    "gradient_count", "gradient_weight_sum", "gradient_vector_error_sum",
    "gradient_magnitude_error_sum", "front_region_predicted_weight",
    "front_region_target_weight", "front_region_intersection_weight",
)
LINE_SUM_FIELDS = (
    "predicted_line_points", "target_line_points", "predicted_matched_points",
    "target_matched_points", "front_distance_sum_km", "front_distance_frame_count",
    "front_line_empty_prediction_frames", "front_line_empty_target_frames",
    "front_line_both_empty_frames",
)


class ForecastEvaluator:
    """Accumulate three-lead forecast errors without averaging batch RMSE values.

    Predictions and targets must already be physical Celsius arrays with shape
    [batch, 3, height, width] or [batch, 3, 1, height, width]. Reductions use the
    core ocean mask; centered gradients use the full, land-masked input grid.
    ``gradient_vector_mae`` averages the absolute eastward and northward component
    errors, then applies area weights. Gradient units are Celsius per kilometer.
    """

    def __init__(self, grid: dict, include_front_lines: bool = False) -> None:
        self.grid = _validated_grid(grid)
        self.include_front_lines = bool(include_front_lines)
        self._shape = self.grid["ocean_mask"].shape
        self._temperature_mask = self.grid["core_mask"] & self.grid["ocean_mask"]
        self._gradient_mask = self.grid["core_mask"] & self.grid["gradient_mask"]
        self._sums = [_empty_statistics(self.include_front_lines)
                      for _ in range(FORECAST_LEAD_COUNT)]
        self._front_line_scores = None
        if self.include_front_lines:
            from .fronts import front_line_scores
            self._front_line_scores = front_line_scores

    def update(self, predictions: np.ndarray, targets: np.ndarray,
               region_masks: Optional[np.ndarray] = None) -> List[dict]:
        """Return sample/lead metrics and add their sufficient statistics atomically."""
        predictions = self._validated_fields(predictions, "predictions")
        targets = self._validated_fields(targets, "targets")
        if predictions.shape != targets.shape:
            raise DataValidationError("Prediction and target shapes differ: %s versus %s" %
                                      (predictions.shape, targets.shape))
        masks = _validated_region_masks(region_masks, predictions.shape)
        records = []
        for sample_index in range(predictions.shape[0]):
            for lead_index in range(FORECAST_LEAD_COUNT):
                statistics = self._frame_statistics(predictions[sample_index, lead_index],
                                                    targets[sample_index, lead_index],
                                                    masks[sample_index, lead_index])
                records.append(dict(_scores(statistics), sample_index=sample_index,
                                    lead_day=lead_index + 1))
        for record in records:
            destination = self._sums[record["lead_day"] - 1]
            for name in destination:
                destination[name] += record[name]
        return records

    def result(self) -> List[dict]:
        """Return one globally aggregated metric record for each of the three leads."""
        return [dict(_scores(statistics), lead_day=index + 1)
                for index, statistics in enumerate(self._sums)]

    def _validated_fields(self, values: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 5 and values.shape[2] == 1:
            values = values[:, :, 0]
        expected = "[batch, 3, height, width] or [batch, 3, 1, height, width]"
        if (values.ndim != 4 or values.shape[0] < 1 or
                values.shape[1] != FORECAST_LEAD_COUNT or values.shape[-2:] != self._shape):
            raise DataValidationError("%s shape %s must be %s on grid %s" %
                                      (name, values.shape, expected, self._shape))
        if not np.isfinite(values[..., self.grid["ocean_mask"]]).all():
            raise DataValidationError("%s contain nonfinite values on the fixed ocean domain" % name)
        return np.where(self.grid["ocean_mask"], values, np.nan)

    def _frame_statistics(self, predictions: np.ndarray, targets: np.ndarray,
                          region_mask: np.ndarray) -> dict:
        statistics = _empty_statistics(self.include_front_lines)
        temperature_mask = self._temperature_mask & region_mask
        statistics["sample_count"] = int(temperature_mask.any())
        _temperature_statistics(statistics, {
            "prediction": predictions, "target": targets,
            "mask": temperature_mask, "weights": self.grid["area_weights"],
        })
        predicted_gradient = temperature_gradients(predictions, self.grid["latitude"],
                                                   self.grid["longitude"])
        target_gradient = temperature_gradients(targets, self.grid["latitude"],
                                                self.grid["longitude"])
        fields = {"prediction": predictions, "target": targets,
                  "predicted_gradient": predicted_gradient, "target_gradient": target_gradient,
                  "mask": self._gradient_mask & region_mask, "weights": self.grid["area_weights"],
                  "threshold": self.grid["front_threshold"]}
        _gradient_statistics(statistics, fields)
        if self.include_front_lines:
            scores = self._front_line_scores(predicted_gradient[2], target_gradient[2], self.grid)
            _line_statistics(statistics, scores)
        return statistics


def _validated_region_masks(values: Optional[np.ndarray], shape: tuple) -> np.ndarray:
    if values is None:
        return np.broadcast_to(np.array(True), shape)
    masks = np.asarray(values)
    if masks.shape != shape or masks.dtype != np.bool_:
        raise DataValidationError("Evaluation region masks must be boolean and match forecast shape")
    return masks


def _validated_grid(grid: dict) -> dict:
    required = {"latitude", "longitude", "core_mask", "ocean_mask",
                "gradient_mask", "area_weights", "front_threshold"}
    if not required.issubset(grid):
        raise DataValidationError("Evaluation grid is missing: %s" % sorted(required - set(grid)))
    result = dict(grid)
    for name in ("latitude", "longitude", "area_weights"):
        result[name] = np.asarray(grid[name], dtype=np.float64)
    shape = (result["latitude"].size, result["longitude"].size)
    _validate_coordinates(result)
    for name in ("core_mask", "ocean_mask", "gradient_mask"):
        if not np.isin(grid[name], [False, True]).all():
            raise DataValidationError("Evaluation %s must contain only boolean mask values" % name)
        result[name] = np.asarray(grid[name], dtype=bool)
    for name in ("core_mask", "ocean_mask", "gradient_mask", "area_weights"):
        if result[name].shape != shape:
            raise DataValidationError("Evaluation grid %s has shape %s, expected %s" %
                                      (name, result[name].shape, shape))
    _validate_spatial_rules(result)
    return result


def _validate_coordinates(grid: dict) -> None:
    for name in ("latitude", "longitude"):
        values = grid[name]
        if (values.ndim != 1 or values.size < 3 or not np.isfinite(values).all() or
                not (np.diff(values) > 0).all()):
            raise DataValidationError("Evaluation %s must be finite, increasing, and contain >=3 cells" % name)
    if (np.abs(grid["latitude"]) >= 90).any():
        raise DataValidationError("Evaluation latitude must lie strictly between the poles")


def _validate_spatial_rules(grid: dict) -> None:
    weights = grid["area_weights"]
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise DataValidationError("Evaluation area weights must all be finite and positive")
    if not (grid["core_mask"] & grid["ocean_mask"]).any():
        raise DataValidationError("Evaluation core ocean mask must contain at least one cell")
    if (grid["gradient_mask"] & ~gradient_valid_mask(grid["ocean_mask"])).any():
        raise DataValidationError("Evaluation gradient mask includes an invalid ocean stencil")
    threshold = grid["front_threshold"]
    if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold <= 0:
        raise DataValidationError("Evaluation requires a positive, finite training front threshold")
    grid["front_threshold"] = float(threshold)


def _empty_statistics(include_front_lines: bool) -> dict:
    names = SUM_FIELDS + LINE_SUM_FIELDS if include_front_lines else SUM_FIELDS
    return dict.fromkeys(names, 0)


def _temperature_statistics(statistics: dict, fields: dict) -> None:
    mask = fields["mask"]
    weights = fields["weights"][mask]
    targets = fields["target"][mask]
    errors = fields["prediction"][mask] - targets
    statistics.update({
        "valid_count": int(mask.sum()), "weight_sum": float(weights.sum()),
        "weighted_sse": float(np.sum(weights * errors ** 2)),
        "weighted_absolute_error": float(np.sum(weights * np.abs(errors))),
        "weighted_error_sum": float(np.sum(weights * errors)),
        "weighted_target_sum": float(np.sum(weights * targets)),
        "weighted_target_square_sum": float(np.sum(weights * targets ** 2)),
    })


def _gradient_statistics(statistics: dict, fields: dict) -> None:
    mask, weights = fields["mask"], fields["weights"]
    prediction, target = fields["predicted_gradient"], fields["target_gradient"]
    if not all(np.isfinite(values[mask]).all() for values in prediction + target):
        raise DataValidationError("Nonfinite centered derivative inside the fixed gradient mask")
    vector_error = (np.abs(prediction[0] - target[0]) + np.abs(prediction[1] - target[1])) / 2
    magnitude_error = np.abs(prediction[2] - target[2])
    predicted_front = mask & (prediction[2] >= fields["threshold"])
    target_front = mask & (target[2] >= fields["threshold"])
    statistics.update({
        "gradient_count": int(mask.sum()), "gradient_weight_sum": float(weights[mask].sum()),
        "gradient_vector_error_sum": float(np.sum(weights[mask] * vector_error[mask])),
        "gradient_magnitude_error_sum": float(np.sum(weights[mask] * magnitude_error[mask])),
        "front_region_predicted_weight": float(weights[predicted_front].sum()),
        "front_region_target_weight": float(weights[target_front].sum()),
        "front_region_intersection_weight": float(weights[predicted_front & target_front].sum()),
    })
    _strong_front_statistics(statistics, dict(fields, mask=target_front))


def _strong_front_statistics(statistics: dict, fields: dict) -> None:
    mask = fields["mask"]
    weights = fields["weights"][mask]
    errors = fields["prediction"][mask] - fields["target"][mask]
    statistics.update({
        "strong_front_count": int(mask.sum()), "strong_front_weight_sum": float(weights.sum()),
        "strong_front_weighted_sse": float(np.sum(weights * errors ** 2)),
        "strong_front_weighted_absolute_error": float(np.sum(weights * np.abs(errors))),
    })


def _line_statistics(statistics: dict, scores: dict) -> None:
    for name in LINE_SUM_FIELDS[:4]:
        statistics[name] = int(scores[name])
    distance = scores["front_distance_km"]
    if distance is not None:
        if not np.isfinite(distance):
            raise DataValidationError("Front line distance must be finite or None")
        statistics["front_distance_sum_km"] = float(distance)
        statistics["front_distance_frame_count"] = 1
    is_prediction_empty = statistics["predicted_line_points"] == 0
    is_target_empty = statistics["target_line_points"] == 0
    statistics["front_line_empty_prediction_frames"] = int(is_prediction_empty)
    statistics["front_line_empty_target_frames"] = int(is_target_empty)
    statistics["front_line_both_empty_frames"] = int(is_prediction_empty and is_target_empty)


def _scores(statistics: dict) -> dict:
    result = dict(statistics)
    result.update({
        "rmse": _root_mean(statistics["weighted_sse"], statistics["weight_sum"]),
        "mae": _ratio(statistics["weighted_absolute_error"], statistics["weight_sum"]),
        "bias": _ratio(statistics["weighted_error_sum"], statistics["weight_sum"]),
        "r2": _r_squared(statistics),
        "strong_front_rmse": _root_mean(statistics["strong_front_weighted_sse"],
                                         statistics["strong_front_weight_sum"]),
        "strong_front_mae": _ratio(statistics["strong_front_weighted_absolute_error"],
                                   statistics["strong_front_weight_sum"]),
        "gradient_vector_mae": _ratio(statistics["gradient_vector_error_sum"],
                                      statistics["gradient_weight_sum"]),
        "gradient_magnitude_mae": _ratio(statistics["gradient_magnitude_error_sum"],
                                         statistics["gradient_weight_sum"]),
    })
    _region_scores(result)
    if "predicted_line_points" in statistics:
        _line_scores(result)
    return result


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator > 0 else None


def _root_mean(numerator: float, denominator: float) -> Optional[float]:
    ratio = _ratio(numerator, denominator)
    return float(np.sqrt(ratio)) if ratio is not None else None


def _r_squared(statistics: dict) -> Optional[float]:
    weight = statistics["weight_sum"]
    if weight <= 0:
        return None
    target_squares = statistics["weighted_target_square_sum"]
    denominator = target_squares - statistics["weighted_target_sum"] ** 2 / weight
    tolerance = max(1e-14, np.finfo(np.float64).eps * abs(target_squares) * 16)
    return float(1 - statistics["weighted_sse"] / denominator) if denominator > tolerance else None


def _region_scores(result: dict) -> None:
    predicted = result["front_region_predicted_weight"]
    target = result["front_region_target_weight"]
    intersection = result["front_region_intersection_weight"]
    result.update({
        "front_region_precision": _ratio(intersection, predicted),
        "front_region_recall": _ratio(intersection, target),
        "front_region_f1": _ratio(2 * intersection, predicted + target),
    })


def _line_scores(result: dict) -> None:
    predicted = result["predicted_line_points"]
    target = result["target_line_points"]
    predicted_matched = result["predicted_matched_points"]
    target_matched = result["target_matched_points"]
    precision, recall = _ratio(predicted_matched, predicted), _ratio(target_matched, target)
    has_any_points = predicted + target > 0
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if has_any_points else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    result.update({"front_line_precision": precision, "front_line_recall": recall,
                   "front_line_f1": f1,
                   "front_distance_km": _ratio(result["front_distance_sum_km"],
                                                result["front_distance_frame_count"])})
