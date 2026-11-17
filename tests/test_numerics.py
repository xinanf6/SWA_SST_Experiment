"""Offline physical and spatial contracts; no download scripts are imported."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.constants import EARTH_RADIUS_METERS, ENERGY_VARIABLES, SECONDS_PER_DAY
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.grid import (GridInterpolator, build_grid_metadata,
                                    gradient_valid_mask, temperature_gradients)
from sst_preprocessing.physics import compute_heat_features, fill_mixed_layer_depth


@pytest.fixture
def config():
    return PreprocessConfig(data_root=Path("unused-data"), output_root=Path("unused-output"))


def test_linear_bilinear_interpolation_and_time_axis():
    coordinate = np.array([0.0, 1.0, 2.0])
    target = np.array([0.25, 1.25])
    field = 2 * coordinate[:, None] + 3 * coordinate[None, :]
    interpolator = GridInterpolator(
        source_grid={"latitude": coordinate, "longitude": coordinate},
        target_grid={"latitude": target, "longitude": target})
    result, weights = interpolator.interpolate(np.stack((field, field + 7)))
    expected = 2 * target[:, None] + 3 * target[None, :]
    np.testing.assert_allclose(result[0], expected)
    np.testing.assert_allclose(result[1], expected + 7)
    np.testing.assert_allclose(weights, 1)


def test_coast_requires_complete_valid_weight_and_reports_missing_fraction():
    coordinate = np.array([0.0, 1.0, 2.0])
    field = np.ones((3, 3), dtype=np.float32)
    field[0, 0] = np.nan
    target = np.array([0.5, 1.5])
    interpolator = GridInterpolator(
        source_grid={"latitude": coordinate, "longitude": coordinate},
        target_grid={"latitude": target, "longitude": target})
    result, weights = interpolator.interpolate(field)
    assert np.isnan(result[0, 0])
    assert weights[0, 0] == pytest.approx(0.75)
    np.testing.assert_allclose(result[1], 1)
    permissive = GridInterpolator(
        source_grid={"latitude": coordinate, "longitude": coordinate},
        target_grid={"latitude": target, "longitude": target}, minimum_weight=0.7)
    np.testing.assert_allclose(permissive.interpolate(field)[0], 1)


def test_outside_interpolation_never_extrapolates():
    source = np.array([0.0, 1.0])
    target = np.array([-0.1, 1.0, 1.1])
    interpolator = GridInterpolator(
        source_grid={"latitude": source, "longitude": source},
        target_grid={"latitude": target, "longitude": target})
    result, weights = interpolator.interpolate(np.ones((2, 2)))
    assert result[1, 1] == pytest.approx(1)
    assert weights[1, 1] == pytest.approx(1)
    assert np.isnan(result[0]).all() and np.isnan(result[2]).all()
    assert not weights[0].any() and not weights[2].any()


@pytest.mark.parametrize("coordinate", [[1, 0], [0, 0], [0, np.nan]])
def test_interpolation_rejects_unsorted_or_invalid_coordinates(coordinate):
    with pytest.raises(DataValidationError, match="strictly increasing"):
        GridInterpolator(source_grid={"latitude": coordinate, "longitude": [0, 1]},
                         target_grid={"latitude": [0, 1], "longitude": [0, 1]})


def test_original_ostia_grid_and_core_shape(config):
    latitude = -48.225 + np.arange(270) * 0.05
    longitude = -60.225 + np.arange(250) * 0.05
    metadata = build_grid_metadata(latitude, longitude, config)
    assert metadata["core_mask"].shape == (270, 250)
    assert metadata["core_mask"].sum() == 260 * 240
    assert metadata["core_mask"][5:-5, 5:-5].all()
    assert metadata["area_weights"].dtype == np.float32
    np.testing.assert_allclose(metadata["area_weights"][:, 0], np.cos(np.deg2rad(latitude)))
    with pytest.raises(DataValidationError, match="input grid shape"):
        build_grid_metadata(latitude[:-1], longitude, config)


def test_core_shape_is_checked_independently(config):
    latitude = -48.225 + np.arange(270) * 0.05
    longitude = -60.225 + np.arange(250) * 0.05
    with pytest.raises(DataValidationError, match="core grid shape"):
        build_grid_metadata(latitude, longitude, replace(config, core_north=-36))


def test_constant_gradient_and_land_stencil():
    coordinate = np.arange(5, dtype=float)
    field = np.full((2, 5, 5), 14.0)
    field[1, 2, 2] = np.nan
    gradient_x, gradient_y, magnitude = temperature_gradients(field, coordinate, coordinate)
    np.testing.assert_allclose(magnitude[0, 1:-1, 1:-1], 0)
    assert np.isnan(gradient_x[:, 0]).all()
    expected_mask = gradient_valid_mask(np.isfinite(field[1]))
    np.testing.assert_array_equal(np.isfinite(magnitude[1]), expected_mask)
    np.testing.assert_array_equal(np.isfinite(gradient_y[1]), expected_mask)


def test_linear_spherical_gradient_has_east_and_north_signs():
    latitude = np.array([-42, -41, -40, -39, -38], dtype=float)
    longitude = np.array([-55, -54, -53, -52, -51], dtype=float)
    field = 2 * latitude[:, None] + 3 * longitude[None, :]
    gradient_x, gradient_y, magnitude = temperature_gradients(field, latitude, longitude)
    north = 2 * 1000 / (EARTH_RADIUS_METERS * np.deg2rad(1))
    east = 3 * 1000 / (EARTH_RADIUS_METERS * np.cos(np.deg2rad(latitude[1:-1])) * np.deg2rad(1))
    np.testing.assert_allclose(gradient_x[1:-1, 1:-1], np.broadcast_to(east[:, None], (3, 3)), rtol=1e-6)
    np.testing.assert_allclose(gradient_y[1:-1, 1:-1], north, rtol=1e-6)
    np.testing.assert_allclose(magnitude[1:-1, 1:-1], np.hypot(east[:, None], north) * np.ones((3, 3)), rtol=1e-6)


def _climatology():
    monthly_mean = np.full((12, 2, 3), 20.0)
    monthly_mean[0, 0, 1] = np.nan
    return {"monthly_mean": monthly_mean, "monthly_median": np.full(12, 30.0),
            "ocean_mask": np.array([[True, True, False], [True, True, True]]),
            "annual_mean": np.full((2, 3), 40.0), "annual_median": 50.0}


def test_mld_uses_local_then_monthly_ocean_fallback_and_preserves_deep_water():
    values = np.array([[[0, np.nan, np.nan], [600, -1, 2]]], dtype=float)
    filled, flags = fill_mixed_layer_depth(values, 1, _climatology())
    np.testing.assert_allclose(filled[0, 0, :2], [20, 30])
    np.testing.assert_allclose(filled[0, 1], [600, 20, 2])
    assert np.isnan(filled[0, 0, 2]) and not flags[0, 0, 2]
    np.testing.assert_array_equal(flags[0, 1], [False, True, False])


def test_mld_annual_fallback_only_for_an_entirely_missing_month():
    climatology = _climatology()
    climatology["monthly_mean"][1] = np.nan
    climatology["monthly_median"][1] = np.nan
    climatology["annual_mean"][0, 1] = np.nan
    filled, _ = fill_mixed_layer_depth(np.zeros((2, 3)), 2, climatology)
    np.testing.assert_allclose(filled[0, :2], [40, 50])
    climatology["monthly_median"][0] = np.nan
    with pytest.raises(DataValidationError, match="cannot fill ocean gaps"):
        fill_mixed_layer_depth(np.zeros((2, 3)), 1, climatology)


def test_daily_energy_heat_signs_3_meter_floor_and_no_extra_day_factor(config):
    depth = np.array([[20, 1, 3, 600]], dtype=float)
    components = {name: np.zeros_like(depth) for name in ENERGY_VARIABLES}
    components["ssr"][:] = 150 * SECONDS_PER_DAY
    components["slhf"][:] = -50 * SECONDS_PER_DAY
    result = compute_heat_features(components, depth, config)
    np.testing.assert_allclose(result["net_heat_flux"], 100)
    expected = 100 * SECONDS_PER_DAY / (1025 * 3990 * np.maximum(depth, 3))
    np.testing.assert_allclose(result["heat_increment"], expected, rtol=1e-6)
    assert result["heat_increment"][0, 0] == pytest.approx(0.10563, abs=1e-5)
    np.testing.assert_equal(result["heat_increment"][0, 1], result["heat_increment"][0, 2])
    components["slhf"][:] = -250 * SECONDS_PER_DAY
    assert (compute_heat_features(components, depth, config)["heat_increment"] < 0).all()


def test_interpolation_and_gradients_are_chunk_invariant():
    coordinate = np.arange(6, dtype=float)
    target = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    field = np.arange(4 * 6 * 6, dtype=float).reshape(4, 6, 6)
    interpolator = GridInterpolator(
        source_grid={"latitude": coordinate, "longitude": coordinate},
        target_grid={"latitude": target, "longitude": target})
    whole = interpolator.interpolate(field)[0]
    chunked = np.concatenate([interpolator.interpolate(field[:1])[0],
                              interpolator.interpolate(field[1:])[0]])
    np.testing.assert_array_equal(whole, chunked)
    whole_gradient = temperature_gradients(whole, target, target)
    for index in range(3):
        chunked_gradient = np.concatenate([temperature_gradients(whole[:1], target, target)[index],
                                           temperature_gradients(whole[1:], target, target)[index]])
        np.testing.assert_array_equal(whole_gradient[index], chunked_gradient)
