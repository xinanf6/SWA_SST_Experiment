"""Numerically known physical-unit, fixed-mask and streaming metric contracts."""

import sys
import types

import numpy as np
import pytest

from sst_analysis.metrics import ForecastEvaluator
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.grid import gradient_valid_mask, temperature_gradients


def evaluation_grid():
    latitude = np.linspace(-45.0, -35.0, 5)
    longitude = np.linspace(-60.0, -50.0, 5)
    core = np.zeros((5, 5), dtype=bool)
    core[1:-1, 1:-1] = True
    ocean = np.ones((5, 5), dtype=bool)
    return {
        "latitude": latitude, "longitude": longitude, "core_mask": core,
        "ocean_mask": ocean, "gradient_mask": gradient_valid_mask(ocean),
        "area_weights": np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], (5, 5)),
        "front_threshold": 0.01, "front_tolerance_km": 10.0,
    }


def batch(values):
    return np.broadcast_to(np.asarray(values, dtype=float), (1, 3, 5, 5)).copy()


def test_physical_celsius_known_constant_error_and_empty_fronts():
    evaluator = ForecastEvaluator(evaluation_grid())
    records = evaluator.update(batch(12.0), batch(10.0))
    assert len(records) == 3
    for lead_day, row in enumerate(evaluator.result(), start=1):
        assert row["lead_day"] == lead_day
        assert row["sample_count"] == 1
        assert row["valid_count"] == 9
        assert row["rmse"] == pytest.approx(2.0)
        assert row["mae"] == pytest.approx(2.0)
        assert row["bias"] == pytest.approx(2.0)
        assert row["r2"] is None
        assert row["strong_front_count"] == 0
        assert row["strong_front_rmse"] is None
        assert row["front_region_f1"] is None
        assert row["gradient_vector_mae"] == 0.0


def test_unequal_area_weights_and_buffer_exclusion():
    grid = evaluation_grid()
    target = batch(10.0)
    prediction = target.copy()
    prediction[:, :, 1, 1:-1] += 2.0
    prediction[:, :, 0, :] = 1000.0
    row = ForecastEvaluator(grid).update(prediction, target)[0]
    weights = grid["area_weights"][grid["core_mask"]]
    error = (prediction - target)[0, 0][grid["core_mask"]]
    assert row["weighted_sse"] == pytest.approx(np.sum(weights * error ** 2))
    assert row["rmse"] == pytest.approx(np.sqrt(np.average(error ** 2, weights=weights)))
    assert row["mae"] == pytest.approx(np.average(np.abs(error), weights=weights))
    assert row["rmse"] != pytest.approx(np.sqrt(np.mean(error ** 2)), rel=1e-3)
    assert row["gradient_magnitude_mae"] > 0


def test_streaming_reconstructs_global_rmse_r2_and_is_batch_independent():
    predictions = np.concatenate([batch(1.0), batch(5.0)])
    targets = np.concatenate([batch(0.0), batch(2.0)])
    together = ForecastEvaluator(evaluation_grid())
    together.update(predictions, targets)
    separate = ForecastEvaluator(evaluation_grid())
    separate.update(predictions[:1], targets[:1])
    separate.update(predictions[1:], targets[1:])
    assert separate.result() == together.result()
    for row in separate.result():
        assert row["sample_count"] == 2
        assert row["rmse"] == pytest.approx(np.sqrt(5.0))
        assert row["rmse"] != pytest.approx(2.0)
        assert row["bias"] == pytest.approx(2.0)
        assert row["r2"] == pytest.approx(-4.0)


def test_gradients_use_components_not_only_magnitude_and_strong_truth_mask():
    grid = evaluation_grid()
    field = np.broadcast_to(np.arange(5)[None, :] * 10.0, (5, 5))
    targets, predictions = batch(field), batch(-field)
    row = ForecastEvaluator(grid).update(predictions, targets)[0]
    gx, gy, magnitude = temperature_gradients(field, grid["latitude"], grid["longitude"])
    mask = grid["core_mask"] & grid["gradient_mask"]
    expected = np.average((np.abs(2 * gx[mask]) + np.abs(2 * gy[mask])) / 2,
                          weights=grid["area_weights"][mask])
    assert row["gradient_vector_mae"] == pytest.approx(expected)
    assert row["gradient_magnitude_mae"] == 0.0
    assert row["strong_front_count"] == 9
    assert row["strong_front_rmse"] == pytest.approx(row["rmse"])
    assert row["front_region_f1"] == 1.0
    assert (magnitude[mask] >= grid["front_threshold"]).all()


def test_front_region_one_sided_empty_and_micro_aggregation():
    grid = evaluation_grid()
    front = batch(np.broadcast_to(np.arange(5)[None, :] * 10.0, (5, 5)))
    evaluator = ForecastEvaluator(grid)
    empty_prediction = evaluator.update(batch(0), front)[0]
    assert empty_prediction["front_region_precision"] is None
    assert empty_prediction["front_region_recall"] == 0.0
    assert empty_prediction["front_region_f1"] == 0.0
    evaluator.update(front, front)
    row = evaluator.result()[0]
    assert row["front_region_precision"] == 1.0
    assert row["front_region_recall"] == 0.5
    assert row["front_region_f1"] == pytest.approx(2.0 / 3)


def test_front_region_recall_uses_area_weights_instead_of_pixel_counts():
    grid = evaluation_grid()
    grid["front_threshold"] = 0.1
    field = np.broadcast_to((np.arange(5) ** 2 * 10.0)[:, None], (5, 5))
    row = ForecastEvaluator(grid).update(batch(field / 2), batch(field))[0]
    target_magnitude = temperature_gradients(field, grid["latitude"], grid["longitude"])[2]
    prediction_magnitude = temperature_gradients(field / 2, grid["latitude"], grid["longitude"])[2]
    target_front = grid["core_mask"] & (target_magnitude >= grid["front_threshold"])
    prediction_front = grid["core_mask"] & (prediction_magnitude >= grid["front_threshold"])
    expected = grid["area_weights"][prediction_front].sum() / grid["area_weights"][target_front].sum()
    assert target_front.sum() == 6
    assert prediction_front.sum() == 3
    assert row["front_region_precision"] == 1.0
    assert row["front_region_recall"] == pytest.approx(expected)
    assert row["front_region_recall"] != pytest.approx(0.5, abs=1e-3)


def test_land_nan_is_masked_but_nonfinite_ocean_buffer_is_rejected():
    grid = evaluation_grid()
    grid["ocean_mask"][0, 0] = False
    grid["gradient_mask"] = gradient_valid_mask(grid["ocean_mask"])
    prediction, target = batch(11.0), batch(10.0)
    prediction[:, :, 0, 0], target[:, :, 0, 0] = np.nan, np.inf
    evaluator = ForecastEvaluator(grid)
    assert evaluator.update(prediction, target)[0]["rmse"] == 1.0
    prediction[:, :, 0, 1] = np.nan
    with pytest.raises(DataValidationError, match="fixed ocean domain"):
        evaluator.update(prediction, target)
    assert evaluator.result()[0]["sample_count"] == 1


def test_single_channel_and_channel_free_shapes_are_equivalent():
    evaluator = ForecastEvaluator(evaluation_grid())
    records = evaluator.update(batch(11.0)[:, :, None], batch(10.0))
    assert [row["sample_index"] for row in records] == [0, 0, 0]
    assert [row["lead_day"] for row in records] == [1, 2, 3]


@pytest.mark.parametrize("shape", [(1, 2, 5, 5), (1, 3, 2, 5, 5), (3, 5, 5),
                                  (1, 3, 4, 5), (0, 3, 5, 5)])
def test_invalid_input_shapes_are_rejected(shape):
    evaluator = ForecastEvaluator(evaluation_grid())
    with pytest.raises(DataValidationError, match="shape"):
        evaluator.update(np.zeros(shape), batch(0))
    assert evaluator.result()[0]["sample_count"] == 0


def test_mismatched_batch_sizes_are_rejected_without_partial_accumulation():
    evaluator = ForecastEvaluator(evaluation_grid())
    with pytest.raises(DataValidationError, match="shapes differ"):
        evaluator.update(np.concatenate([batch(0), batch(1)]), batch(0))
    assert evaluator.result()[0]["rmse"] is None


def test_optional_front_lines_micro_aggregate_counts_and_defined_distances(monkeypatch):
    calls = []

    def synthetic_scores(prediction, target, grid):
        calls.append((prediction.shape, target.shape))
        return {"predicted_line_points": 2, "target_line_points": 4,
                "predicted_matched_points": 1, "target_matched_points": 4,
                "front_distance_km": 3.0}

    module = types.ModuleType("sst_analysis.fronts")
    module.front_line_scores = synthetic_scores
    monkeypatch.setitem(sys.modules, "sst_analysis.fronts", module)
    without_lines = ForecastEvaluator(evaluation_grid())
    assert "front_line_f1" not in without_lines.update(batch(0), batch(0))[0]
    assert not calls
    with_lines = ForecastEvaluator(evaluation_grid(), include_front_lines=True)
    with_lines.update(np.concatenate([batch(0), batch(1)]),
                      np.concatenate([batch(0), batch(1)]))
    row = with_lines.result()[0]
    assert len(calls) == 6
    assert row["predicted_line_points"] == 4
    assert row["target_line_points"] == 8
    assert row["front_line_precision"] == 0.5
    assert row["front_line_recall"] == 1.0
    assert row["front_line_f1"] == pytest.approx(2 / 3)
    assert row["front_distance_km"] == 3.0
    assert row["front_distance_frame_count"] == 2


def test_empty_line_distances_are_excluded_and_empty_frames_are_counted(monkeypatch):
    def empty_scores(prediction, target, grid):
        return {"predicted_line_points": 0, "target_line_points": 4,
                "predicted_matched_points": 0, "target_matched_points": 0,
                "front_distance_km": None}

    module = types.ModuleType("sst_analysis.fronts")
    module.front_line_scores = empty_scores
    monkeypatch.setitem(sys.modules, "sst_analysis.fronts", module)
    evaluator = ForecastEvaluator(evaluation_grid(), include_front_lines=True)
    evaluator.update(batch(0), batch(0))
    row = evaluator.result()[0]
    assert row["front_line_f1"] == 0.0
    assert row["front_line_precision"] is None
    assert row["front_line_recall"] == 0.0
    assert row["front_distance_km"] is None
    assert row["front_distance_frame_count"] == 0
    assert row["front_line_empty_prediction_frames"] == 1
    assert row["front_line_empty_target_frames"] == 0


def test_actual_front_geometry_integrates_with_streaming_evaluator():
    field = np.broadcast_to(np.arange(5)[None, :] * 10.0, (5, 5))
    evaluator = ForecastEvaluator(evaluation_grid(), include_front_lines=True)
    records = evaluator.update(batch(field), batch(field))
    assert len(records) == 3
    for row in evaluator.result():
        assert row["predicted_line_points"] > 0
        assert row["front_line_precision"] == 1.0
        assert row["front_line_recall"] == 1.0
        assert row["front_line_f1"] == 1.0
        assert row["front_distance_km"] == 0.0
        assert row["front_distance_frame_count"] == 1


@pytest.mark.parametrize("invalid", ["area_weights", "front_threshold", "gradient_mask", "ocean_mask"])
def test_invalid_fixed_grid_rules_are_rejected(invalid):
    grid = evaluation_grid()
    if invalid == "area_weights":
        grid[invalid] = np.zeros((5, 5))
    elif invalid == "front_threshold":
        grid[invalid] = np.nan
    elif invalid == "ocean_mask":
        grid[invalid] = np.full((5, 5), 0.5)
    else:
        grid[invalid] = np.ones((5, 5), dtype=bool)
    with pytest.raises(DataValidationError):
        ForecastEvaluator(grid)
