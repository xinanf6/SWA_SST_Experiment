"""Known paired uncertainty, missing-day blocks and strict evaluation identities."""

import json

import numpy as np
import pandas as pd
import pytest

from sst_analysis.comparison import (
    STATISTIC_COLUMNS, _bootstrap_counts, _settings, compare_evaluations,
)
from sst_preprocessing.errors import ConfigurationError, DataValidationError


def write_evaluation(root, errors, dates=None):
    root.mkdir()
    dates = pd.date_range("2023-01-01", periods=len(errors)) if dates is None else pd.DatetimeIndex(dates)
    rows = []
    for date, error in zip(dates, errors):
        for lead in (1, 2, 3):
            rows.append({"origin_date": str(date.date()), "target_date": str((date + pd.Timedelta(days=lead)).date()),
                         "lead_day": lead, "weight_sum": 2.0, "weighted_sse": 2 * error ** 2,
                         "weighted_absolute_error": 2 * abs(error), "weighted_error_sum": 2 * error,
                         "strong_front_weight_sum": 1.0, "strong_front_weighted_sse": error ** 2,
                         "strong_front_weighted_absolute_error": abs(error), "gradient_weight_sum": 1.0,
                         "gradient_vector_error_sum": abs(error) / 10,
                         "gradient_magnitude_error_sum": abs(error) / 20})
    pd.DataFrame(rows).to_csv(root / "daily_errors.csv", index=False)
    report = {"status": "complete", "run_identity": "known_run", "split": "test",
              "sample_count": len(errors), "lead_days": [1, 2, 3], "model_name": root.name}
    (root / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def comparison_options(tmp_path, errors=None, dates=None):
    errors = np.arange(1, 41, dtype=float) / 10 if errors is None else np.asarray(errors, dtype=float)
    return {"evaluation_a": write_evaluation(tmp_path / "model_a", errors, dates),
            "evaluation_b": write_evaluation(tmp_path / "model_b", errors + 1, dates),
            "output_root": tmp_path / "comparison", "block_days": 5,
            "repetitions": 80, "seed": 42}


def metric_row(report, metric="rmse", group="all"):
    return next(row for row in report["comparisons"]
                if row["lead_day"] == 1 and row["metric"] == metric and row["group"] == group)


def test_pooled_statistics_are_recomputed_instead_of_averaging_daily_rmse(tmp_path):
    options = comparison_options(tmp_path, [1, 3])
    report = compare_evaluations(options)
    row = metric_row(report)
    assert row["value_a"] == pytest.approx(np.sqrt(5))
    assert row["value_b"] == pytest.approx(np.sqrt(10))
    assert row["mean_difference"] == pytest.approx(np.sqrt(5) - np.sqrt(10))
    assert row["n_pairs"] == 2
    assert row["ci_lower"] == row["ci_upper"] == row["mean_difference"]
    assert report["blocks"]["n_short_runs"] == 1
    assert len(report["comparisons"]) == 21
    assert (options["output_root"] / "comparison.csv").is_file()
    persisted = json.loads((options["output_root"] / "comparison.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "complete"


def test_pairing_uses_the_same_draw_for_models_and_reproduces_with_seed(tmp_path):
    options = comparison_options(tmp_path)
    first = compare_evaluations(options)
    options["output_root"] = tmp_path / "second_comparison"
    second = compare_evaluations(options)
    assert first["comparisons"] == second["comparisons"]
    for name in ("mae", "bias", "strong_front_mae"):
        row = metric_row(first, name)
        assert row["mean_difference"] == pytest.approx(-1.0)
        assert row["ci_lower"] == pytest.approx(-1.0)
        assert row["ci_upper"] == pytest.approx(-1.0)
        assert row["n_valid_replicates"] == options["repetitions"]
    assert metric_row(first, "gradient_vector_mae")["mean_difference"] == pytest.approx(-0.1)


def test_percentile_interval_matches_known_two_origin_rmse_distribution(tmp_path):
    options = {"evaluation_a": write_evaluation(tmp_path / "a", [0, 2]),
               "evaluation_b": write_evaluation(tmp_path / "b", [1, 1]),
               "output_root": tmp_path / "comparison", "block_days": 1,
               "repetitions": 400, "seed": 42}
    row = metric_row(compare_evaluations(options))
    assert row["mean_difference"] == pytest.approx(np.sqrt(2) - 1)
    assert row["ci_lower"] == -1.0
    assert row["ci_upper"] == 1.0
    assert row["n_valid_replicates"] == 400


def test_bias_remains_signed_and_is_distinct_from_error_magnitude(tmp_path):
    options = comparison_options(tmp_path, [-1, -3])
    report = compare_evaluations(options)
    assert metric_row(report, "bias")["mean_difference"] == -1.0
    assert metric_row(report, "mae")["mean_difference"] == 1.0


def test_blocks_do_not_cross_missing_dates_and_preserve_run_lengths(tmp_path):
    dates = pd.DatetimeIndex(list(pd.date_range("2023-01-01", periods=8)) +
                             list(pd.date_range("2023-02-01", periods=3)))
    options = comparison_options(tmp_path, np.arange(11), dates)
    counts, metadata = _bootstrap_counts(dates, _settings(options))
    assert metadata["n_continuous_runs"] == 2
    assert metadata["n_blocks"] == 5
    assert metadata["runs"][0]["effective_block_days"] == 5
    assert metadata["runs"][1]["effective_block_days"] == 3
    assert metadata["n_short_runs"] == 1
    assert np.all(counts[:, :8].sum(axis=1) == 8)
    assert np.all(counts[:, 8:].sum(axis=1) == 3)
    assert np.all(counts[:, 8:] == 1)
    assert (counts[:, :8] != 1).any()


def add_zero_weight_group(root, group="DJF"):
    frame = pd.read_csv(root / "daily_errors.csv")
    frame["group"] = group
    frame.loc[:, STATISTIC_COLUMNS] = 0.0
    frame.to_csv(root / "group_errors.csv", index=False)


def test_zero_weight_groups_retain_calendar_days_and_have_undefined_intervals(tmp_path):
    options = comparison_options(tmp_path)
    for root in (options["evaluation_a"], options["evaluation_b"]):
        add_zero_weight_group(root)
    report = compare_evaluations(options)
    row = metric_row(report, group="DJF")
    assert row["n_pairs"] == 40
    assert row["n_valid_pairs"] == 0
    assert row["n_valid_replicates"] == 0
    assert row["mean_difference"] is None
    assert row["ci_lower"] is None
    assert row["ci_upper"] is None
    assert report["blocks"]["n_continuous_runs"] == 1
    assert len(report["comparisons"]) == 42


def test_zero_weight_origins_stay_in_blocks_when_season_has_observations(tmp_path):
    options = comparison_options(tmp_path)
    for root in (options["evaluation_a"], options["evaluation_b"]):
        frame = pd.read_csv(root / "daily_errors.csv")
        frame["group"] = "DJF"
        frame.loc[frame["origin_date"] > "2023-01-10", STATISTIC_COLUMNS] = 0.0
        frame.to_csv(root / "group_errors.csv", index=False)
    report = compare_evaluations(options)
    row = metric_row(report, "mae", "DJF")
    assert row["n_pairs"] == 40
    assert row["n_valid_pairs"] == 10
    assert row["mean_difference"] == pytest.approx(-1)
    assert row["ci_lower"] == pytest.approx(-1)
    assert row["ci_upper"] == pytest.approx(-1)


@pytest.mark.parametrize("field,value", [("run_identity", "another_run"), ("split", "validation"),
                                        ("sample_count", 1), ("lead_days", [1, 2])])
def test_mismatching_report_identity_is_rejected(tmp_path, field, value):
    options = comparison_options(tmp_path)
    path = options["evaluation_b"] / "metrics.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataValidationError, match="metadata"):
        compare_evaluations(options)
    assert not options["output_root"].exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "group", "date", "weight"])
def test_unpaired_or_inconsistent_daily_rows_are_rejected(tmp_path, change):
    options = comparison_options(tmp_path)
    path = options["evaluation_b"] / "daily_errors.csv"
    frame = pd.read_csv(path)
    if change == "missing":
        frame = frame.iloc[1:]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "group":
        frame["group"] = "other"
    elif change == "date":
        frame.loc[0, "target_date"] = "2024-01-01"
    else:
        frame.loc[0, "weight_sum"] = 1.0
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        compare_evaluations(options)


def test_row_order_does_not_change_strict_date_key_pairing(tmp_path):
    options = comparison_options(tmp_path)
    path = options["evaluation_b"] / "daily_errors.csv"
    frame = pd.read_csv(path)
    frame.iloc[::-1].to_csv(path, index=False)
    assert metric_row(compare_evaluations(options), "mae")["mean_difference"] == pytest.approx(-1)


def test_incomplete_group_grid_is_rejected_even_if_both_inputs_match(tmp_path):
    options = comparison_options(tmp_path)
    for root in (options["evaluation_a"], options["evaluation_b"]):
        path = root / "daily_errors.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Every comparison group"):
        compare_evaluations(options)


@pytest.mark.parametrize("field,value", [("weighted_sse", -1), ("gradient_vector_error_sum", np.inf),
                                        ("weight_sum", 0)])
def test_invalid_sufficient_statistics_are_rejected(tmp_path, field, value):
    options = comparison_options(tmp_path)
    path = options["evaluation_a"] / "daily_errors.csv"
    frame = pd.read_csv(path)
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        compare_evaluations(options)


def test_prepared_output_boundary_and_existing_comparison_are_protected(tmp_path):
    options = comparison_options(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    options["prepared_root"] = prepared
    options["output_root"] = prepared / "comparison"
    with pytest.raises(ConfigurationError):
        compare_evaluations(options)
    options["output_root"] = tmp_path / "comparison"
    compare_evaluations(options)
    with pytest.raises(DataValidationError, match="already exist"):
        compare_evaluations(options)
    options["overwrite"] = True
    assert compare_evaluations(options)["status"] == "complete"


def test_prepared_markers_protect_input_without_explicit_option(tmp_path):
    options = comparison_options(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "READY.json").write_text("{}")
    (prepared / "grid_masks.npz").write_text("")
    options["output_root"] = prepared / "comparison"
    with pytest.raises(DataValidationError, match="outside prepared"):
        compare_evaluations(options)


@pytest.mark.parametrize("field,value", [("block_days", 0), ("repetitions", 1.5), ("seed", -1)])
def test_invalid_resampling_settings_are_rejected(tmp_path, field, value):
    options = comparison_options(tmp_path)
    options[field] = value
    with pytest.raises(DataValidationError):
        compare_evaluations(options)
