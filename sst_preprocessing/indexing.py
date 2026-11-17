"""Leakage-free windows; daily source arrays retain their original dates."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import PreprocessConfig
from .errors import DataValidationError


INDEX_COLUMNS = (
    "split", "origin_date", "origin_index", "sst_start_index", "sst_end_index",
    "environment_start_index", "environment_end_index",
)


def _validate_dates(dates: Iterable) -> pd.DatetimeIndex:
    date_index = pd.DatetimeIndex(dates)
    if date_index.hasnans or date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise DataValidationError("Daily dates must be finite, unique, and increasing")
    if not date_index.equals(date_index.normalize()):
        raise DataValidationError("Daily dates must refer to midnight, without subdaily offsets")
    return date_index


def _window_record(config: PreprocessConfig, dates: pd.DatetimeIndex, origin: int) -> dict:
    record = {
        "origin_date": dates[origin].strftime("%Y-%m-%d"),
        "origin_index": origin,
        "sst_start_index": origin - config.history_days + 1,
        "sst_end_index": origin,
        "environment_start_index": origin - config.history_days + 1 - config.environment_lag_days,
        "environment_end_index": origin - config.environment_lag_days,
    }
    for lead in range(1, config.prediction_days + 1):
        record["target_%d_index" % lead] = origin + lead
        record["target_%d_date" % lead] = dates[origin + lead].strftime("%Y-%m-%d")
    return record


def _has_complete_support(dates: pd.DatetimeIndex, start: int, end: int) -> bool:
    expected_day_count = end - start
    actual_day_count = (dates[end] - dates[start]).days
    # Unique increasing dates make equal span and count equivalent to no missing day.
    return expected_day_count == actual_day_count


def build_sample_index(dates: Iterable, config: PreprocessConfig) -> pd.DataFrame:
    """Require SST, lagged environmental history, and labels inside one split."""
    dates = _validate_dates(dates)
    records = []
    support_history = config.history_days - 1 + config.environment_lag_days
    for split, (split_start, split_end) in config.splits.items():
        start_date, end_date = pd.Timestamp(split_start), pd.Timestamp(split_end)
        for origin in range(support_history, len(dates) - config.prediction_days):
            support_start = origin - support_history
            support_end = origin + config.prediction_days
            is_inside_split = dates[support_start] >= start_date and dates[support_end] <= end_date
            if is_inside_split and _has_complete_support(dates, support_start, support_end):
                record = _window_record(config, dates, origin)
                record["split"] = split
                records.append(record)
    target_columns = [name for lead in range(1, config.prediction_days + 1)
                      for name in ("target_%d_index" % lead, "target_%d_date" % lead)]
    return pd.DataFrame.from_records(records, columns=list(INDEX_COLUMNS) + target_columns)
