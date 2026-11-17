"""Exclude incomplete source windows without filling or moving daily observations."""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .constants import BASE_VARIABLES, FRONT_VARIABLES, THERMAL_VARIABLES
from .errors import DataValidationError


QUALITY_GROUPS = {
    "sst": ("sst",),
    "front": FRONT_VARIABLES,
    "environment": BASE_VARIABLES[1:] + THERMAL_VARIABLES,
}
INDEX_HISTORY_COLUMNS = (
    "origin_index", "sst_start_index", "sst_end_index",
    "environment_start_index", "environment_end_index",
)
QUALITY_COLUMNS = (
    "date", "sst_valid", "front_valid", "environment_valid",
    "missing_sst", "missing_front", "missing_environment",
)
BOOLEAN_TEXT_VALUES = {
    "true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False,
}


def _daily_dates(values) -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as error:
        raise DataValidationError("Daily quality dates cannot be decoded: %s" % error)
    if dates.tz is not None or dates.empty or dates.hasnans:
        raise DataValidationError("Daily quality dates must be nonempty, timezone-free UTC dates")
    if not dates.equals(dates.normalize()):
        raise DataValidationError("Daily quality dates must use midnight labels")
    if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D")):
        raise DataValidationError("Daily quality dates must be ordered, unique, and continuous")
    return dates


def _missing_counts(dataset: xr.Dataset, variables: tuple, mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(dataset.sizes["time"], dtype=np.int64)
    for name in variables:
        if name not in dataset:
            raise DataValidationError("Daily quality input is missing variable: %s" % name)
        values = dataset[name].transpose("time", "latitude", "longitude").values
        if values.shape[1:] != mask.shape:
            raise DataValidationError("Daily quality grid mismatch for %s: %s" % (name, values.shape))
        counts += (~np.isfinite(values[:, mask])).sum(axis=1)
    return counts


def assess_daily_quality(dataset: xr.Dataset, grid: dict) -> pd.DataFrame:
    """Count missing cells on fixed training masks, summing across each group's fields."""
    dates = _daily_dates(dataset.time.values)
    result = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for group, variables in QUALITY_GROUPS.items():
        mask_name = "gradient_mask" if group == "front" else "ocean_mask"
        mask = np.asarray(grid[mask_name], dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise DataValidationError("Daily quality requires a nonempty 2D %s" % mask_name)
        missing = _missing_counts(dataset, variables, mask)
        result[group + "_valid"] = missing == 0
        result["missing_" + group] = missing
    return result.loc[:, QUALITY_COLUMNS]


def _boolean_column(values: pd.Series, name: str) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(BOOLEAN_TEXT_VALUES).all():
        raise DataValidationError("Daily quality %s must contain only bool or 0/1 values" % name)
    return normalized.map(BOOLEAN_TEXT_VALUES).to_numpy(dtype=bool)


def _quality_context(quality: pd.DataFrame) -> dict:
    if not np.array_equal(quality.index.to_numpy(), np.arange(len(quality))):
        raise DataValidationError("Daily quality rows must use consecutive positional indices 0..N-1")
    required = {"date"} | {group + "_valid" for group in QUALITY_GROUPS}
    required |= {"missing_" + group for group in QUALITY_GROUPS}
    if not required.issubset(quality.columns):
        raise DataValidationError("Daily quality columns are incomplete: %s" % sorted(required - set(quality.columns)))
    context = {"dates": _daily_dates(quality["date"]), "validity": {}}
    for group in QUALITY_GROUPS:
        valid = _boolean_column(quality[group + "_valid"], group + "_valid")
        counts = _integer_values(quality["missing_" + group], "missing_" + group)
        if (counts < 0).any() or not np.array_equal(valid, counts == 0):
            raise DataValidationError("Daily quality validity/count mismatch: %s" % group)
        context["validity"][group] = valid
    return context


def _integer_values(values: pd.Series, name: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Invalid numeric sample/quality column %s: %s" % (name, error))
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise DataValidationError("Sample/quality column must contain finite integers: %s" % name)
    return numeric.astype(np.int64)


def _index_context(index: pd.DataFrame, context: dict) -> pd.DataFrame:
    target_leads = sorted(int(match.group(1)) for name in index.columns
                          for match in [re.fullmatch(r"target_(\d+)_index", name)] if match)
    if not target_leads or target_leads != list(range(1, len(target_leads) + 1)):
        raise DataValidationError("Sample index requires consecutive target lead columns starting at 1")
    numeric_columns = list(INDEX_HISTORY_COLUMNS) + ["target_%d_index" % lead for lead in target_leads]
    required = set(numeric_columns) | {"origin_date"} | {"target_%d_date" % lead for lead in target_leads}
    if not required.issubset(index.columns):
        raise DataValidationError("Sample index columns are incomplete: %s" % sorted(required - set(index.columns)))
    numeric = pd.DataFrame({name: _integer_values(index[name], name) for name in numeric_columns})
    if ((numeric < 0) | (numeric >= len(context["dates"]))).to_numpy().any():
        raise DataValidationError("Sample index points outside the daily quality timeline")
    context["target_leads"] = target_leads
    _validate_index_clocks(index, numeric, context)
    return numeric


def _validate_index_clocks(index: pd.DataFrame, numeric: pd.DataFrame, context: dict) -> None:
    origin = numeric["origin_index"]
    is_history_valid = (
        (numeric["sst_end_index"] == origin)
        & (numeric["environment_end_index"] == origin - 1)
        & (numeric["environment_start_index"] == numeric["sst_start_index"] - 1)
        & (numeric["sst_start_index"] <= numeric["sst_end_index"])
    )
    if not is_history_valid.all():
        raise DataValidationError("Sample index violates the original one-day environmental lag")
    _check_date_alignment(index["origin_date"], context["dates"][origin], "origin_date")
    for lead in context["target_leads"]:
        target_indices = numeric["target_%d_index" % lead]
        if not (target_indices == origin + lead).all():
            raise DataValidationError("Sample target index violates its lead: Day %d" % lead)
        name = "target_%d_date" % lead
        _check_date_alignment(index[name], context["dates"][target_indices], name)


def _check_date_alignment(values: pd.Series, expected: pd.DatetimeIndex, name: str) -> None:
    try:
        actual = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as error:
        raise DataValidationError("Invalid sample date %s: %s" % (name, error))
    if not actual.equals(expected):
        raise DataValidationError("Sample %s does not align with the daily quality timeline" % name)


def _bad_dates(context: dict, group: str, indices: np.ndarray) -> str:
    bad_indices = indices[~context["validity"][group][indices]]
    return ",".join(context["dates"][bad_indices].strftime("%Y-%m-%d"))


def _exclusion_reason(row: pd.Series, context: dict) -> str:
    sst_indices = np.arange(row.sst_start_index, row.sst_end_index + 1, dtype=np.int64)
    environment_indices = np.arange(row.environment_start_index, row.environment_end_index + 1, dtype=np.int64)
    target_indices = np.array([row["target_%d_index" % lead] for lead in context["target_leads"]], dtype=np.int64)
    checks = (
        ("sst_history", "sst", sst_indices), ("front_history", "front", sst_indices),
        ("environment_history", "environment", environment_indices), ("sst_target", "sst", target_indices),
    )
    reasons = []
    for category, group, indices in checks:
        bad_dates = _bad_dates(context, group, indices)
        if bad_dates:
            reasons.append("%s: %s" % (category, bad_dates))
    return "; ".join(reasons)


def filter_sample_index(index: pd.DataFrame, quality: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude only windows whose consumed history or SST targets include bad days."""
    context = _quality_context(quality)
    numeric = _index_context(index, context)
    reasons = [_exclusion_reason(row, context) for _, row in numeric.iterrows()]
    is_accepted = np.array([not reason for reason in reasons], dtype=bool)
    accepted = index.loc[is_accepted].copy().reset_index(drop=True)
    excluded = index.loc[~is_accepted].copy().reset_index(drop=True)
    excluded["exclusion_reason"] = np.asarray(reasons, dtype=object)[~is_accepted]
    return accepted, excluded
