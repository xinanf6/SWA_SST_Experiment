"""Offline seasonal, reference-front and lagged-MLD forecast group contracts."""

import numpy as np
import pandas as pd
import pytest

from sst_analysis.context import AnalysisContext
from sst_analysis.groups import GROUP_NAMES, GroupedForecastEvaluator
from sst_preprocessing.grid import gradient_valid_mask, temperature_gradients
from sst_preprocessing.errors import DataValidationError


MLD_MEAN_METERS = 50.0
MLD_STANDARD_DEVIATION_METERS = 5.0
MLD_THRESHOLD_METERS = 20.0


def _grid() -> dict:
    latitude, longitude = -40.2 + np.arange(7) * 0.05, -55.2 + np.arange(7) * 0.05
    core = np.zeros((7, 7), bool)
    core[1:-1, 1:-1] = True
    ocean = np.ones_like(core)
    ocean[3, 3] = False
    return {"latitude": latitude, "longitude": longitude, "core_mask": core,
            "ocean_mask": ocean, "gradient_mask": gradient_valid_mask(ocean),
            "area_weights": np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], core.shape).copy()}


def _sample_index(dates: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    for origin_date in ("2020-02-28", "2020-02-29"):
        origin = dates.get_loc(origin_date)
        row = {"origin_date": origin_date, "origin_index": origin,
               "environment_end_index": origin - 1}
        for lead in (1, 2, 3):
            row["target_%d_index" % lead] = origin + lead
            row["target_%d_date" % lead] = str(dates[origin + lead].date())
        records.append(row)
    return pd.DataFrame(records)


def _thermal_fields(index: pd.DataFrame, day_count: int) -> np.ndarray:
    depth = np.ones((day_count, 7, 7), np.float32)
    depth[index.environment_end_index.iloc[0]] = 30.0
    depth[index.environment_end_index.iloc[0], :, :3] = 10.0
    depth[index.environment_end_index.iloc[0], :, 3] = MLD_THRESHOLD_METERS
    depth[index.environment_end_index.iloc[1]] = 30.0
    thermal = np.zeros((day_count, 3, 7, 7), np.float32)
    thermal[:, 1] = (depth - MLD_MEAN_METERS) / MLD_STANDARD_DEVIATION_METERS
    return thermal


def _forecast_fields() -> tuple:
    north, east = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    spatial_target = 10. + 0.2 * east ** 2 + 0.1 * north
    targets = np.broadcast_to(spatial_target, (2, 3, 7, 7)).copy()
    errors = 1. + 0.1 * north + 0.2 * east
    predictions = targets + errors
    return predictions, targets


@pytest.fixture
def grouped_case(tmp_path):
    root = tmp_path / "prepared"
    (root / "normalized").mkdir(parents=True)
    dates = pd.date_range("2020-02-20", periods=17)
    index = _sample_index(dates)
    thermal = _thermal_fields(index, len(dates))
    np.save(root / "dates.npy", dates.to_numpy())
    np.save(root / "normalized" / "thermal.npy", thermal)
    run = {"run_identity": "synthetic-groups", "config": {
        "splits": {"train": [str(dates[0].date()), str(dates[-1].date())]}}}
    statistics = {"front_threshold_p90": 0.3, "variables": {"mixed_layer_depth": {
        "mean": MLD_MEAN_METERS, "std": MLD_STANDARD_DEVIATION_METERS}}}
    context = AnalysisContext(root, run, statistics, _grid())
    predictions, targets = _forecast_fields()
    return {"context": context, "index": index, "thermal": thermal,
            "predictions": predictions, "targets": targets}


def _evaluator(case: dict) -> GroupedForecastEvaluator:
    return GroupedForecastEvaluator(case["context"], {"mld_threshold_m": MLD_THRESHOLD_METERS})


def _records(case: dict) -> tuple:
    evaluator = _evaluator(case)
    try:
        rows = evaluator.update(case["predictions"], case["targets"], case["index"])
        summary = evaluator.result()
    finally:
        evaluator.close()
    return {(row["group"], row["sample_index"], row["lead_day"]): row for row in rows}, summary


def test_all_origin_lead_group_rows_and_empty_seasons_are_retained(grouped_case):
    rows, summary = _records(grouped_case)
    assert len(rows) == 2 * 3 * 8
    assert len(summary) == 3 * 8
    assert {row["group"] for row in summary} == set(GROUP_NAMES)
    core_count = int((grouped_case["context"].grid["core_mask"] & grouped_case["context"].grid["ocean_mask"]).sum())
    for (group, sample, lead), row in rows.items():
        if group not in ("DJF", "MAM", "JJA", "SON"):
            continue
        expected = "DJF" if sample == 0 and lead == 1 else "MAM"
        assert row["valid_count"] == (core_count if group == expected else 0)
        assert row["sample_count"] == int(group == expected)
        assert (row["rmse"] is None) == (group != expected)
    lookup = {(row["group"], row["lead_day"]): row for row in summary}
    assert [lookup[("MAM", lead)]["sample_count"] for lead in (1, 2, 3)] == [1, 2, 2]
    assert [lookup[("DJF", lead)]["sample_count"] for lead in (1, 2, 3)] == [1, 0, 0]
    assert all("front_distance_km" not in row for row in summary)


def test_front_and_mld_partitions_cover_their_distinct_fixed_domains(grouped_case):
    rows, _ = _records(grouped_case)
    grid = grouped_case["context"].grid
    core_ocean = grid["core_mask"] & grid["ocean_mask"]
    core_gradient = grid["core_mask"] & grid["gradient_mask"]
    weights = grid["area_weights"]
    for sample in (0, 1):
        for lead in (1, 2, 3):
            strong, weak = rows[("strong_front", sample, lead)], rows[("weak_front", sample, lead)]
            shallow, deep = rows[("shallow_mld", sample, lead)], rows[("deep_mld", sample, lead)]
            assert strong["valid_count"] + weak["valid_count"] == core_gradient.sum()
            assert strong["weight_sum"] + weak["weight_sum"] == pytest.approx(weights[core_gradient].sum())
            assert strong["strong_front_count"] == strong["valid_count"]
            assert weak["strong_front_count"] == 0
            assert shallow["valid_count"] + deep["valid_count"] == core_ocean.sum()
            assert shallow["weight_sum"] + deep["weight_sum"] == pytest.approx(weights[core_ocean].sum())
            assert shallow["gradient_count"] + deep["gradient_count"] == core_gradient.sum()


def test_mld_uses_physical_channel_one_at_t_minus_one_and_includes_threshold(grouped_case):
    rows, _ = _records(grouped_case)
    grid = grouped_case["context"].grid
    core = grid["core_mask"] & grid["ocean_mask"]
    expected_shallow = core.copy()
    expected_shallow[:, 4:] = False
    for lead in (1, 2, 3):
        assert rows[("shallow_mld", 0, lead)]["valid_count"] == expected_shallow.sum() == 14
        assert rows[("deep_mld", 0, lead)]["valid_count"] == 10
        assert rows[("shallow_mld", 1, lead)]["valid_count"] == 0
        assert rows[("deep_mld", 1, lead)]["valid_count"] == core.sum()


def test_group_gradient_reductions_preserve_full_field_stencils(grouped_case):
    rows, _ = _records(grouped_case)
    context = grouped_case["context"]
    grid = context.grid
    target = np.where(grid["ocean_mask"], grouped_case["targets"][0, 0], np.nan)
    prediction = np.where(grid["ocean_mask"], grouped_case["predictions"][0, 0], np.nan)
    target_gradient = temperature_gradients(target, grid["latitude"], grid["longitude"])
    predicted_gradient = temperature_gradients(prediction, grid["latitude"], grid["longitude"])
    core_gradient = grid["core_mask"] & grid["gradient_mask"]
    masks = {"strong_front": core_gradient & (target_gradient[2] >= context.front_threshold),
             "weak_front": core_gradient & (target_gradient[2] < context.front_threshold),
             "shallow_mld": core_gradient & (np.indices(core_gradient.shape)[1] <= 3),
             "deep_mld": core_gradient & (np.indices(core_gradient.shape)[1] > 3)}
    error = (np.abs(predicted_gradient[0] - target_gradient[0]) + np.abs(predicted_gradient[1] - target_gradient[1])) / 2
    for group, mask in masks.items():
        weights = grid["area_weights"][mask]
        assert mask.any()
        row = rows[(group, 0, 1)]
        assert row["gradient_count"] == mask.sum()
        assert row["gradient_vector_mae"] == pytest.approx(np.average(error[mask], weights=weights))


def test_front_groups_depend_on_reference_not_prediction(grouped_case):
    original_rows, _ = _records(grouped_case)
    changed = dict(grouped_case)
    changed["predictions"] = grouped_case["targets"] + 10. * np.arange(7)[None, None, None, :]
    changed_rows, _ = _records(changed)
    for sample in (0, 1):
        for lead in (1, 2, 3):
            for group in ("strong_front", "weak_front"):
                key = (group, sample, lead)
                assert changed_rows[key]["valid_count"] == original_rows[key]["valid_count"]
    assert changed_rows[("weak_front", 0, 1)]["valid_count"] > 0


def test_temperature_group_rmse_uses_each_group_area_weights(grouped_case):
    rows, _ = _records(grouped_case)
    context = grouped_case["context"]
    grid = context.grid
    target = np.where(grid["ocean_mask"], grouped_case["targets"][0, 0], np.nan)
    gradient = temperature_gradients(target, grid["latitude"], grid["longitude"])[2]
    core_gradient = grid["core_mask"] & grid["gradient_mask"]
    core_ocean = grid["core_mask"] & grid["ocean_mask"]
    east = np.indices(core_ocean.shape)[1]
    masks = {"strong_front": core_gradient & (gradient >= context.front_threshold),
             "weak_front": core_gradient & (gradient < context.front_threshold),
             "shallow_mld": core_ocean & (east <= 3), "deep_mld": core_ocean & (east > 3)}
    error = grouped_case["predictions"][0, 0] - grouped_case["targets"][0, 0]
    for group, mask in masks.items():
        expected = np.sqrt(np.average(error[mask] ** 2, weights=grid["area_weights"][mask]))
        assert rows[(group, 0, 1)]["rmse"] == pytest.approx(expected)


def test_singleton_temperature_channel_matches_four_dimensional_fields(grouped_case):
    _, expected = _records(grouped_case)
    changed = dict(grouped_case)
    changed["predictions"] = changed["predictions"][:, :, None]
    changed["targets"] = changed["targets"][:, :, None]
    _, actual = _records(changed)
    assert actual == expected


def test_group_results_are_invariant_to_batch_partition(grouped_case):
    _, expected = _records(grouped_case)
    evaluator = _evaluator(grouped_case)
    try:
        for sample in (0, 1):
            evaluator.update(grouped_case["predictions"][sample:sample + 1],
                             grouped_case["targets"][sample:sample + 1],
                             grouped_case["index"].iloc[sample:sample + 1])
        assert evaluator.result() == expected
    finally:
        evaluator.close()


@pytest.mark.parametrize("physical_depth", (0., -1., np.nan, np.inf))
def test_invalid_lagged_mld_fails_and_mapping_can_be_closed(grouped_case, physical_depth):
    thermal = grouped_case["thermal"].copy()
    day = grouped_case["index"].environment_end_index.iloc[0]
    thermal[day, 1, 1, 1] = (physical_depth - MLD_MEAN_METERS) / MLD_STANDARD_DEVIATION_METERS
    np.save(grouped_case["context"].root / "normalized" / "thermal.npy", thermal)
    evaluator = _evaluator(grouped_case)
    mapping = evaluator.thermal._mmap
    try:
        with pytest.raises(DataValidationError, match="finite and positive"):
            evaluator.update(grouped_case["predictions"], grouped_case["targets"], grouped_case["index"])
    finally:
        evaluator.close()
    assert mapping.closed
    evaluator.close()


@pytest.mark.parametrize("bad_index", (-1, 17, 7.5, 8))
def test_environment_indices_must_be_in_range_integer_and_t_minus_one(grouped_case, bad_index):
    index = grouped_case["index"].copy()
    index.loc[0, "environment_end_index"] = bad_index
    evaluator = _evaluator(grouped_case)
    try:
        with pytest.raises(DataValidationError):
            evaluator.update(grouped_case["predictions"], grouped_case["targets"], index)
    finally:
        evaluator.close()


@pytest.mark.parametrize("threshold", (-1., 0., np.nan, np.inf))
def test_mld_threshold_must_be_finite_and_positive(grouped_case, threshold):
    with pytest.raises(DataValidationError, match="finite and positive"):
        GroupedForecastEvaluator(grouped_case["context"], {"mld_threshold_m": threshold})


def test_successful_close_releases_thermal_and_leaves_sources_unchanged(grouped_case):
    path = grouped_case["context"].root / "normalized" / "thermal.npy"
    before = path.read_bytes()
    evaluator = _evaluator(grouped_case)
    mapping = evaluator.thermal._mmap
    try:
        evaluator.update(grouped_case["predictions"], grouped_case["targets"], grouped_case["index"])
    finally:
        evaluator.close()
    assert mapping.closed and evaluator.thermal is None
    assert path.read_bytes() == before
    evaluator.close()


def test_bad_thermal_dtype_releases_constructor_mapping(grouped_case, monkeypatch):
    import sst_analysis.groups as groups_module
    path = grouped_case["context"].root / "normalized" / "thermal.npy"
    np.save(path, grouped_case["thermal"].astype(np.float64))
    original_load = np.load
    mappings = []

    def tracked_load(*arguments, **options):
        values = original_load(*arguments, **options)
        if isinstance(values, np.memmap):
            mappings.append(values._mmap)
        return values

    monkeypatch.setattr(groups_module.np, "load", tracked_load)
    with pytest.raises(DataValidationError, match="shape or dtype"):
        _evaluator(grouped_case)
    assert len(mappings) == 1 and mappings[0].closed


def test_missing_dates_releases_constructor_mapping(grouped_case, monkeypatch):
    import sst_analysis.groups as groups_module
    (grouped_case["context"].root / "dates.npy").unlink()
    original_load = np.load
    mappings = []

    def tracked_load(*arguments, **options):
        values = original_load(*arguments, **options)
        if isinstance(values, np.memmap):
            mappings.append(values._mmap)
        return values

    monkeypatch.setattr(groups_module.np, "load", tracked_load)
    with pytest.raises(FileNotFoundError):
        _evaluator(grouped_case)
    assert len(mappings) == 1 and mappings[0].closed
