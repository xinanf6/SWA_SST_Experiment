"""Offline geometry and empty-set tests for optional raw skeleton diagnostics."""

import numpy as np
import pytest
from scipy.ndimage import label

from sst_analysis.fronts import _skeletonize, front_line_scores
from sst_preprocessing.errors import DataValidationError


EARTH_RADIUS_KM = 6371.0


def _grid(shape=(11, 13)):
    return {
        "latitude": -40.0 + 0.05 * np.arange(shape[0]),
        "longitude": -55.0 + 0.05 * np.arange(shape[1]),
        "gradient_mask": np.ones(shape, dtype=bool),
        "ocean_mask": np.ones(shape, dtype=bool),
        "core_mask": np.ones(shape, dtype=bool),
        "front_threshold": 1.0,
        "front_tolerance_km": 10.0,
    }


def _zonal_distance(latitude, longitude_difference):
    half_angle = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude_difference) / 2.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.abs(half_angle))


def test_identical_fronts_have_perfect_scores():
    gradient = np.zeros((11, 13))
    gradient[2:9, 4] = 1.0
    result = front_line_scores(gradient, gradient, _grid())
    assert result["front_line_precision"] == 1.0
    assert result["front_line_recall"] == 1.0
    assert result["front_line_f1"] == 1.0
    assert result["front_distance_km"] == pytest.approx(0.0, abs=1e-12)
    assert result["predicted_line_points"] == result["predicted_matched_points"] == 7
    assert result["target_line_points"] == result["target_matched_points"] == 7
    assert not result["empty_predicted_line"]


def test_shifted_straight_lines_use_great_circle_distances():
    target, prediction = np.zeros((11, 13)), np.zeros((11, 13))
    target[2:9, 4], prediction[2:9, 5] = 1.0, 1.0
    grid = _grid()
    result = front_line_scores(prediction, target, grid)
    expected = _zonal_distance(grid["latitude"][2:9], 0.05).mean()
    assert result["front_distance_km"] == pytest.approx(expected, rel=1e-10)
    assert result["front_line_f1"] == 1.0
    grid["front_tolerance_km"] = 1.0
    result = front_line_scores(prediction, target, grid)
    assert result["front_line_f1"] == 0.0
    assert result["predicted_matched_points"] == result["target_matched_points"] == 0


def test_isolated_points_are_not_removed_by_component_filtering():
    gradient = np.zeros((11, 13))
    gradient[2, 2] = gradient[8, 10] = 1.0
    result = front_line_scores(gradient, gradient, _grid())
    assert result["predicted_line_points"] == 2
    assert result["front_line_f1"] == 1.0


def test_skeletonization_precedes_core_crop():
    gradient = np.zeros((11, 13))
    gradient[3:8, 1:12] = 1.0
    grid = _grid()
    grid["core_mask"][:] = False
    grid["core_mask"][:, 5:8] = True
    binary = gradient >= 1.0
    expected = _skeletonize(binary) & grid["core_mask"]
    incorrectly_cropped = _skeletonize(binary & grid["core_mask"])
    assert not np.array_equal(expected, incorrectly_cropped)
    result = front_line_scores(gradient, gradient, grid)
    assert result["predicted_line_points"] == int(expected.sum())


def test_thinning_preserves_single_pixels_and_one_pixel_lines():
    binary = np.zeros((11, 13), dtype=bool)
    binary[1, 1] = True
    binary[4:9, 5] = True
    binary[4, 9:12] = True
    np.testing.assert_array_equal(_skeletonize(binary), binary)


def test_rectangle_strip_converges_to_connected_thin_line():
    binary = np.zeros((15, 21), dtype=bool)
    binary[4:11, 2:19] = True
    skeleton = _skeletonize(binary)
    assert skeleton.sum() > 1
    assert np.flatnonzero(skeleton.any(axis=1)).tolist() == [7]
    assert label(skeleton, structure=np.ones((3, 3)))[1] == 1
    np.testing.assert_array_equal(_skeletonize(skeleton), skeleton)
    assert np.all(~skeleton | binary)


def test_thinning_retains_ring_hole_and_connected_outline():
    binary = np.zeros((21, 21), dtype=bool)
    binary[3:18, 3:18] = True
    binary[7:14, 7:14] = False
    skeleton = _skeletonize(binary)
    assert 0 < skeleton.sum() < binary.sum()
    assert label(skeleton, structure=np.ones((3, 3)))[1] == 1
    # Eight-connected foreground uses four-connected background for its hole count.
    background_labels, _ = label(~skeleton)
    assert background_labels[10, 10] != background_labels[0, 0]
    np.testing.assert_array_equal(_skeletonize(skeleton), skeleton)


def test_thinning_is_stable_for_empty_and_border_touching_fields():
    empty = np.zeros((3, 4), dtype=bool)
    np.testing.assert_array_equal(_skeletonize(empty), empty)
    solid = np.ones((15, 17), dtype=bool)
    skeleton = _skeletonize(solid)
    assert 0 < skeleton.sum() < solid.sum()
    np.testing.assert_array_equal(_skeletonize(skeleton), skeleton)


def test_bidirectional_distance_gives_each_direction_equal_weight():
    prediction, target = np.zeros((11, 13)), np.zeros((11, 13))
    prediction[2, 2] = prediction[8, 10] = target[2, 2] = 1.0
    grid = _grid()
    result = front_line_scores(prediction, target, grid)
    latitude = np.deg2rad(grid["latitude"][[2, 8]])
    delta_longitude = np.deg2rad(grid["longitude"][10] - grid["longitude"][2])
    haversine = np.sin(np.diff(latitude)[0] / 2.0) ** 2
    haversine += np.cos(latitude).prod() * np.sin(delta_longitude / 2.0) ** 2
    point_distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))
    assert result["front_line_precision"] == 0.5
    assert result["front_line_recall"] == 1.0
    assert result["front_line_f1"] == pytest.approx(2.0 / 3.0)
    assert result["front_distance_km"] == pytest.approx(point_distance / 4.0)


@pytest.mark.parametrize("has_prediction,has_target", [(False, False), (False, True), (True, False)])
def test_empty_fronts_remain_explicit_without_distance_penalties(has_prediction, has_target):
    prediction, target = np.zeros((11, 13)), np.zeros((11, 13))
    prediction[5, 5], target[5, 5] = float(has_prediction), float(has_target)
    result = front_line_scores(prediction, target, _grid())
    assert result["front_line_precision"] == (0.0 if has_prediction else None)
    assert result["front_line_recall"] == (0.0 if has_target else None)
    assert result["front_line_f1"] == (0.0 if has_prediction or has_target else None)
    assert result["front_distance_km"] is None
    assert result["empty_predicted_line"] == (not has_prediction)
    assert result["empty_target_line"] == (not has_target)


def test_threshold_includes_equal_values_only_on_valid_domain():
    gradient = np.zeros((11, 13))
    gradient[2, 2], gradient[5, 5], gradient[8, 8] = 1.0, 0.999, 10.0
    grid = _grid()
    grid["gradient_mask"][8, 8] = False
    result = front_line_scores(gradient, gradient, grid)
    assert result["predicted_line_points"] == 1


def test_longitude_distance_changes_with_latitude():
    distances = []
    for latitude in [0.0, 60.0]:
        grid = _grid((3, 3))
        grid["latitude"] = latitude + np.arange(3) * 0.05
        prediction, target = np.zeros((3, 3)), np.zeros((3, 3))
        prediction[0, 1], target[0, 0] = 1.0, 1.0
        distances.append(front_line_scores(prediction, target, grid)["front_distance_km"])
    assert distances[0] == pytest.approx(_zonal_distance(0.0, 0.05), rel=1e-10)
    assert distances[1] == pytest.approx(_zonal_distance(60.0, 0.05), rel=1e-10)
    assert distances[1] / distances[0] == pytest.approx(0.5, rel=1e-6)


def test_land_and_outer_stencil_nans_are_ignored():
    grid = _grid()
    grid["gradient_mask"][0, :] = False
    grid["ocean_mask"][5, 5] = False
    gradient = np.zeros((11, 13))
    gradient[0, :] = np.nan
    gradient[5, 5] = np.nan
    assert front_line_scores(gradient, gradient, grid)["predicted_line_points"] == 0


@pytest.mark.parametrize("case", ["shape", "nan", "negative", "latitude", "longitude",
                                  "threshold", "tolerance", "mask"])
def test_invalid_evaluation_data_raise_contextual_errors(case):
    gradient, grid = np.zeros((11, 13)), _grid()
    if case == "shape":
        gradient = gradient[:, :-1]
    elif case == "nan":
        gradient[5, 5] = np.nan
    elif case == "negative":
        gradient[5, 5] = -1.0
    elif case == "latitude":
        grid["latitude"][0] = np.nan
    elif case == "longitude":
        grid["longitude"][0] = grid["longitude"][1]
    elif case == "threshold":
        grid["front_threshold"] = 0.0
    elif case == "tolerance":
        grid["front_tolerance_km"] = -1.0
    else:
        grid["gradient_mask"] = np.zeros((1, 1), dtype=bool)
    with pytest.raises(DataValidationError, match="Front"):
        front_line_scores(gradient, gradient, grid)
