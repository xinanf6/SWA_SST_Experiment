"""Offline local-missing-data tests, with no future-field imputation."""

import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst_preprocessing.config import PreprocessConfig
from sst_preprocessing.constants import FEATURE_GROUPS
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.indexing import build_sample_index
from sst_preprocessing.quality import assess_daily_quality, filter_sample_index


def _complete_quality(dates):
    result = pd.DataFrame({"date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d")})
    for group in ("sst", "front", "environment"):
        result[group + "_valid"] = True
        result["missing_" + group] = 0
    return result


def _mark_bad(quality, date, group):
    selected = quality.date == date
    quality.loc[selected, group + "_valid"] = False
    quality.loc[selected, "missing_" + group] = 1


def _index_fixture(tmp_path):
    config = PreprocessConfig(data_root=tmp_path / "raw", output_root=tmp_path / "processed")
    dates = pd.date_range(config.start_date, config.end_date)
    return build_sample_index(dates, config), _complete_quality(dates)


def _physical_fixture():
    dates = pd.date_range("2021-01-01", periods=3)
    variables = {name: (("time", "latitude", "longitude"), np.ones((3, 3, 3), dtype=np.float32))
                 for group in FEATURE_GROUPS.values() for name in group}
    dataset = xr.Dataset(variables, coords={"time": dates, "latitude": [-42, -41, -40], "longitude": [-57, -56, -55]})
    ocean_mask = np.ones((3, 3), dtype=bool)
    ocean_mask[0, 0] = False
    gradient_mask = np.zeros((3, 3), dtype=bool)
    gradient_mask[1, 1] = True
    return dataset, {"ocean_mask": ocean_mask, "gradient_mask": gradient_mask}


def test_daily_missing_counts_use_group_masks_and_sum_across_variables():
    dataset, grid = _physical_fixture()
    dataset["sst"].values[0, 0, 0] = np.nan
    dataset["sst"].values[1, 1, 1] = np.nan
    dataset["gradient_x"].values[1, 0, 1] = np.nan
    dataset["gradient_y"].values[2, 1, 1] = np.inf
    dataset["gradient_magnitude"].values[2, 1, 1] = np.nan
    dataset["u10"].values[1, 1, 1] = np.nan
    dataset["heat_increment"].values[1, 1, 1] = np.nan
    before = dataset.copy(deep=True)

    quality = assess_daily_quality(dataset, grid)

    assert quality.columns.tolist() == ["date", "sst_valid", "front_valid", "environment_valid",
                                        "missing_sst", "missing_front", "missing_environment"]
    assert quality.missing_sst.tolist() == [0, 1, 0]
    assert quality.missing_front.tolist() == [0, 0, 2]
    assert quality.missing_environment.tolist() == [0, 2, 0]
    assert quality.sst_valid.tolist() == [True, False, True]
    assert quality.front_valid.tolist() == [True, True, False]
    assert quality.environment_valid.tolist() == [True, False, True]
    xr.testing.assert_identical(dataset, before)


def test_2021_january_9_environment_gap_excludes_exactly_nine_validation_windows(tmp_path):
    index, quality = _index_fixture(tmp_path)
    _mark_bad(quality, "2021-01-09", "environment")
    accepted, excluded = filter_sample_index(index, quality)
    assert accepted.groupby("split").size().to_dict() == {"train": 3636, "validation": 704, "test": 714}
    assert excluded.split.unique().tolist() == ["validation"]
    assert excluded.origin_date.tolist() == pd.date_range("2021-01-15", "2021-01-23").strftime("%Y-%m-%d").tolist()
    assert excluded.exclusion_reason.tolist() == ["environment_history: 2021-01-09"] * 9
    assert accepted.columns.tolist() == index.columns.tolist()
    assert excluded.columns.tolist() == index.columns.tolist() + ["exclusion_reason"]


def test_environment_gap_during_sst_labels_does_not_exclude_extra_windows(tmp_path):
    index, quality = _index_fixture(tmp_path)
    selected = index.loc[index.origin_date == "2021-01-15"]
    for date in ("2021-01-15", "2021-01-16", "2021-01-17", "2021-01-18"):
        _mark_bad(quality, date, "environment")
    accepted, excluded = filter_sample_index(selected, quality)
    assert len(accepted) == 1
    assert excluded.empty


def test_missing_sst_excludes_consumed_history_and_labels_without_imputation(tmp_path):
    index, quality = _index_fixture(tmp_path)
    _mark_bad(quality, "2021-02-01", "sst")
    before = quality.copy(deep=True)
    selected = index.loc[index.origin_date.isin(["2021-01-29", "2021-02-01", "2021-02-14", "2021-02-15"])]
    accepted, excluded = filter_sample_index(selected, quality)
    assert accepted.origin_date.tolist() == ["2021-02-15"]
    assert excluded.exclusion_reason.tolist() == [
        "sst_target: 2021-02-01", "sst_history: 2021-02-01", "sst_history: 2021-02-01"]
    pd.testing.assert_frame_equal(quality, before)


def test_front_quality_is_required_only_in_input_sst_history(tmp_path):
    index, quality = _index_fixture(tmp_path)
    _mark_bad(quality, "2021-02-01", "front")
    selected = index.loc[index.origin_date.isin(["2021-01-31", "2021-02-01", "2021-02-15"])]
    accepted, excluded = filter_sample_index(selected, quality)
    assert accepted.origin_date.tolist() == ["2021-01-31", "2021-02-15"]
    assert excluded.exclusion_reason.tolist() == ["front_history: 2021-02-01"]


def test_csv_types_and_nonconsecutive_sample_row_labels_are_supported(tmp_path):
    index, quality = _index_fixture(tmp_path)
    index = index.iloc[:2].astype({name: str for name in index.columns if name.endswith("_index")})
    quality = pd.read_csv(io.StringIO(quality.to_csv(index=False)), dtype=str)
    index.index = [10, 99]
    accepted, excluded = filter_sample_index(index, quality)
    pd.testing.assert_frame_equal(accepted, index.reset_index(drop=True))
    assert excluded.empty


@pytest.mark.parametrize("defect", ["gap", "reverse", "duplicate", "row_index", "subdaily"])
def test_quality_requires_continuous_ordered_positional_dates(tmp_path, defect):
    index, quality = _index_fixture(tmp_path)
    if defect == "gap":
        quality = quality.drop(20).reset_index(drop=True)
    elif defect == "reverse":
        quality = quality.iloc[::-1].reset_index(drop=True)
    elif defect == "duplicate":
        quality.loc[20, "date"] = quality.loc[19, "date"]
    elif defect == "row_index":
        quality.index = np.arange(1, len(quality) + 1)
    else:
        quality.loc[20, "date"] += " 12:00:00"
    with pytest.raises(DataValidationError, match="quality"):
        filter_sample_index(index.iloc[:1], quality)


@pytest.mark.parametrize("column", ["origin_date", "target_1_date", "target_3_date"])
def test_sample_dates_must_align_with_daily_quality_positions(tmp_path, column):
    index, quality = _index_fixture(tmp_path)
    index = index.iloc[:1].copy()
    index.loc[index.index[0], column] = "2020-01-01"
    with pytest.raises(DataValidationError, match="does not align"):
        filter_sample_index(index, quality)


def test_quality_rejects_validity_count_disagreement(tmp_path):
    index, quality = _index_fixture(tmp_path)
    quality.loc[0, "missing_sst"] = 1
    with pytest.raises(DataValidationError, match="validity/count mismatch"):
        filter_sample_index(index.iloc[:1], quality)


def test_empty_sample_table_keeps_both_output_schemas(tmp_path):
    index, quality = _index_fixture(tmp_path)
    accepted, excluded = filter_sample_index(index.iloc[:0], quality)
    assert accepted.empty and excluded.empty
    assert accepted.columns.tolist() == index.columns.tolist()
    assert excluded.columns.tolist() == index.columns.tolist() + ["exclusion_reason"]
