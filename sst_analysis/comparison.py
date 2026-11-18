"""Paired calendar-block uncertainty for two supplied model evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sst_preprocessing.artifacts import read_json, write_json_atomic
from sst_preprocessing.errors import DataValidationError

from .config import validate_output_location


DEFAULT_BLOCK_DAYS = 30
DEFAULT_REPETITIONS = 2000
DEFAULT_SEED = 20261117
PAIR_KEYS = ["group", "lead_day", "origin_date"]
STATISTIC_COLUMNS = (
    "weight_sum", "weighted_sse", "weighted_absolute_error", "weighted_error_sum",
    "strong_front_weight_sum", "strong_front_weighted_sse",
    "strong_front_weighted_absolute_error", "gradient_weight_sum",
    "gradient_vector_error_sum", "gradient_magnitude_error_sum",
)
SUPPORT_COLUMNS = ("weight_sum", "strong_front_weight_sum", "gradient_weight_sum")
METRIC_SPECIFICATIONS = {
    "rmse": ("weighted_sse", "weight_sum", True),
    "mae": ("weighted_absolute_error", "weight_sum", False),
    "bias": ("weighted_error_sum", "weight_sum", False),
    "strong_front_rmse": ("strong_front_weighted_sse", "strong_front_weight_sum", True),
    "strong_front_mae": ("strong_front_weighted_absolute_error", "strong_front_weight_sum", False),
    "gradient_vector_mae": ("gradient_vector_error_sum", "gradient_weight_sum", False),
    "gradient_magnitude_mae": ("gradient_magnitude_error_sum", "gradient_weight_sum", False),
}


@dataclass(frozen=True)
class _Settings:
    evaluation_a: Path
    evaluation_b: Path
    output_root: Path
    block_days: int
    repetitions: int
    seed: int


def compare_evaluations(options: dict) -> dict:
    """Write paired model-A minus model-B percentile intervals from daily sums.

    Required options are evaluation_a, evaluation_b and output_root directories.
    Optional block_days/repetitions/seed default to 30/2000/20261117; pass
    prepared_root for an explicit protected-data boundary, and overwrite=True
    to replace existing comparison artifacts. No forecasts are generated.
    """
    settings = _settings(options)
    first, second = _load_evaluation(settings.evaluation_a), _load_evaluation(settings.evaluation_b)
    dates = _validate_pair(first, second)
    counts, blocks = _bootstrap_counts(dates, settings)
    state = {"first": first, "second": second, "counts": counts,
             "settings": settings, "blocks": blocks}
    rows = _compare_groups(state)
    report = _report(state, rows)
    _write_comparison(settings.output_root, report)
    return report


def _settings(options: dict) -> _Settings:
    missing = {"evaluation_a", "evaluation_b", "output_root"} - set(options)
    if missing:
        raise DataValidationError("Comparison options are missing: %s" % sorted(missing))
    output = Path(options["output_root"]).resolve()
    if options.get("prepared_root") is not None:
        validate_output_location(output, Path(options["prepared_root"]))
    for parent in (output,) + tuple(output.parents):
        if (parent / "READY.json").exists() and (parent / "grid_masks.npz").exists():
            raise DataValidationError("Comparison output must be outside prepared data: %s" % parent)
    if not options.get("overwrite", False) and any((output / name).exists() for name in
                                                  ("comparison.csv", "comparison.json")):
        raise DataValidationError("Comparison files already exist; choose a new output or set overwrite=True")
    return _Settings(Path(options["evaluation_a"]).resolve(), Path(options["evaluation_b"]).resolve(),
                     output, _integer(options.get("block_days", DEFAULT_BLOCK_DAYS), "block_days"),
                     _integer(options.get("repetitions", DEFAULT_REPETITIONS), "repetitions"),
                     _integer(options.get("seed", DEFAULT_SEED), "seed", allow_zero=True))


def _integer(value: int, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise DataValidationError("Comparison %s must be an integer" % name) from error
    if isinstance(value, bool) or number != value or number < minimum:
        raise DataValidationError("Comparison %s must be an integer >= %d" % (name, minimum))
    return number


def _load_evaluation(root: Path) -> dict:
    required = (root / "metrics.json", root / "daily_errors.csv")
    for path in required:
        if not path.is_file():
            raise DataValidationError("Comparison input is missing: %s" % path)
    report = read_json(required[0])
    if report.get("status") != "complete":
        raise DataValidationError("Comparison requires a complete evaluation: %s" % root)
    tables = [_read_table(required[1], is_supplement=False)]
    paths = list(required)
    supplementary = root / "group_errors.csv"
    if supplementary.exists():
        tables.append(_read_table(supplementary, is_supplement=True))
        paths.append(supplementary)
    frame = pd.concat(tables, ignore_index=True)
    _validate_table(frame, root)
    return {"root": root, "report": report,
            "frame": frame.set_index(PAIR_KEYS).sort_index(), "paths": paths}


def _read_table(path: Path, is_supplement: bool) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise DataValidationError("Cannot read comparison table %s: %s" % (path, error)) from error
    required = {"origin_date", "lead_day"}.union(STATISTIC_COLUMNS)
    if not required.issubset(frame):
        raise DataValidationError("%s lacks comparison fields: %s" % (path, sorted(required - set(frame))))
    if "group" not in frame:
        if is_supplement:
            raise DataValidationError("Supplementary group_errors.csv must name each group")
        frame["group"] = "all"
    if frame.empty or frame["group"].isna().any() or (frame["group"].astype(str).str.strip() == "").any():
        raise DataValidationError("Comparison table has empty rows/groups: %s" % path)
    frame["group"] = frame["group"].astype(str)
    frame["origin_date"] = _dates(frame["origin_date"], str(path))
    if "target_date" in frame:
        frame["target_date"] = _dates(frame["target_date"], str(path))
    return frame


def _dates(values, context: str) -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True)).tz_localize(None)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Invalid comparison dates in %s" % context) from error
    if dates.hasnans or not dates.equals(dates.normalize()):
        raise DataValidationError("Comparison dates must be finite whole UTC days: %s" % context)
    return dates


def _validate_table(frame: pd.DataFrame, root: Path) -> None:
    if not frame["lead_day"].isin([1, 2, 3]).all():
        raise DataValidationError("Comparison lead_day must be 1, 2 or 3: %s" % root)
    frame["lead_day"] = frame["lead_day"].astype(int)
    if frame.duplicated(PAIR_KEYS).any():
        raise DataValidationError("Duplicate origin/lead/group comparison keys: %s" % root)
    if "target_date" in frame and not np.array_equal(frame["target_date"].values,
            (frame["origin_date"] + pd.to_timedelta(frame["lead_day"], unit="D")).values):
        raise DataValidationError("Comparison target dates disagree with lead days: %s" % root)
    try:
        values = frame.loc[:, STATISTIC_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Comparison sufficient statistics must be numeric: %s" % root) from error
    if not np.isfinite(values).all():
        raise DataValidationError("Comparison sufficient statistics must be finite: %s" % root)
    unsigned = [name for name in STATISTIC_COLUMNS if name != "weighted_error_sum"]
    if (frame.loc[:, unsigned].to_numpy(dtype=np.float64) < 0).any():
        raise DataValidationError("Negative comparison weights or error sums: %s" % root)
    _validate_zero_support(frame, root)


def _validate_zero_support(frame: pd.DataFrame, root: Path) -> None:
    for numerator, denominator, _ in METRIC_SPECIFICATIONS.values():
        invalid = (frame[denominator] == 0) & (frame[numerator] != 0)
        if invalid.any():
            raise DataValidationError("Nonzero %s with zero %s in %s" % (numerator, denominator, root))


def _validate_pair(first: dict, second: dict) -> pd.DatetimeIndex:
    report_a, report_b = first["report"], second["report"]
    for name in ("run_identity", "split", "sample_count", "lead_days"):
        if name not in report_a or name not in report_b or report_a[name] != report_b[name]:
            raise DataValidationError("Comparison evaluation metadata differs or is missing: %s" % name)
    if not report_a["run_identity"] or report_a["split"] not in ("train", "validation", "test"):
        raise DataValidationError("Comparison requires a nonempty run identity and known split")
    if report_a["lead_days"] != [1, 2, 3]:
        raise DataValidationError("Comparison evaluation lead_days must equal [1, 2, 3]")
    for name in ("metric_temperature_units", "front_threshold_C_per_km"):
        if report_a.get(name) != report_b.get(name):
            raise DataValidationError("Comparison evaluation scoring rules differ: %s" % name)
    units = report_a.get("metric_temperature_units", "degree_Celsius")
    if units != "degree_Celsius":
        raise DataValidationError("Comparison requires evaluation metrics in degree_Celsius")
    frame_a, frame_b = first["frame"], second["frame"]
    if not frame_a.index.equals(frame_b.index):
        raise DataValidationError("Comparison origin/lead/group keys must pair exactly")
    dates = pd.DatetimeIndex(frame_a.index.get_level_values("origin_date").unique()).sort_values()
    _validate_complete_keys(frame_a, dates)
    _validate_report_origins((report_a, report_b), dates)
    _validate_matching_truth(frame_a, frame_b)
    return dates


def _validate_complete_keys(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    groups = sorted(frame.index.get_level_values("group").unique())
    expected = pd.MultiIndex.from_product([groups, [1, 2, 3], dates], names=PAIR_KEYS)
    if not frame.index.equals(expected):
        raise DataValidationError("Every comparison group must retain every origin and all three leads, including zero weights")


def _validate_report_origins(reports: tuple, dates: pd.DatetimeIndex) -> None:
    for report in reports:
        if report["sample_count"] != len(dates):
            raise DataValidationError("Comparison sample_count disagrees with paired origin dates")
        if "origin_dates" in report and not _dates(report["origin_dates"], "metrics.json").equals(dates):
            raise DataValidationError("Comparison metrics.json origin_dates disagree with paired dates")


def _validate_matching_truth(first: pd.DataFrame, second: pd.DataFrame) -> None:
    for name in ("target_date",):
        if (name in first) != (name in second):
            raise DataValidationError("Both comparison inputs must supply the same date fields")
        if name in first and not first[name].equals(second[name]):
            raise DataValidationError("Comparison target dates do not match")
    names = list(SUPPORT_COLUMNS)
    names.extend(name for name in ("weighted_target_sum", "weighted_target_square_sum")
                 if name in first and name in second)
    for name in names:
        if not np.allclose(first[name].to_numpy(dtype=np.float64), second[name].to_numpy(dtype=np.float64),
                           rtol=1e-10, atol=1e-10, equal_nan=False):
            raise DataValidationError("Comparison reference support/statistics differ: %s" % name)


def _continuous_runs(dates: pd.DatetimeIndex, block_days: int) -> list:
    boundaries = np.flatnonzero(np.diff(dates.values) != np.timedelta64(1, "D")) + 1
    starts, stops = np.r_[0, boundaries], np.r_[boundaries, len(dates)]
    runs = []
    for start, stop in zip(starts, stops):
        count = int(stop - start)
        length = min(block_days, count)
        runs.append({"start_index": int(start), "end_index_exclusive": int(stop),
                     "start_date": str(dates[start].date()), "end_date": str(dates[stop - 1].date()),
                     "n_origins": count, "effective_block_days": length,
                     "n_moving_blocks": count - length + 1, "is_short_run": count < block_days,
                     "drawn_blocks_per_replicate": int(np.ceil(count / length))})
    return runs


def _bootstrap_counts(dates: pd.DatetimeIndex, settings: _Settings) -> tuple:
    runs = _continuous_runs(dates, settings.block_days)
    generator = np.random.default_rng(settings.seed)
    counts = np.zeros((settings.repetitions, len(dates)), dtype=np.float64)
    for repetition in range(settings.repetitions):
        for run in runs:
            length, count = run["effective_block_days"], run["n_origins"]
            starts = generator.integers(0, run["n_moving_blocks"], run["drawn_blocks_per_replicate"])
            selection = (starts[:, None] + np.arange(length)).reshape(-1)[:count]
            begin, end = run["start_index"], run["end_index_exclusive"]
            counts[repetition, begin:end] = np.bincount(selection, minlength=count)
    return counts, {"runs": runs, "n_continuous_runs": len(runs),
                    "n_blocks": sum(run["n_moving_blocks"] for run in runs),
                    "n_short_runs": sum(run["is_short_run"] for run in runs),
                    "drawn_blocks_per_replicate": sum(run["drawn_blocks_per_replicate"] for run in runs)}


def _compare_groups(state: dict) -> list:
    first = state["first"]["frame"]
    groups = sorted(first.index.get_level_values("group").unique(), key=lambda name: (name != "all", name))
    rows = []
    for group in groups:
        for lead in (1, 2, 3):
            frames = [state[name]["frame"].xs((group, lead), level=("group", "lead_day"))
                      for name in ("first", "second")]
            data = [frame.loc[:, STATISTIC_COLUMNS].to_numpy(dtype=np.float64) for frame in frames]
            summary = {"point": [_metric_values(values.sum(axis=0)) for values in data],
                       "bootstrap": [_metric_values(state["counts"] @ values) for values in data],
                       "data": data, "group": group, "lead_day": lead}
            rows.extend(_metric_rows(state, summary))
    return rows


def _metric_values(sums: np.ndarray) -> dict:
    result = {}
    for name, (numerator, denominator, is_root) in METRIC_SPECIFICATIONS.items():
        first, second = (sums[..., STATISTIC_COLUMNS.index(column)] for column in (numerator, denominator))
        value = np.divide(first, second, out=np.full_like(first, np.nan), where=second > 0)
        result[name] = np.sqrt(value) if is_root else value
    return result


def _metric_rows(state: dict, summary: dict) -> list:
    rows = []
    settings = state["settings"]
    for name, (_, denominator, _) in METRIC_SPECIFICATIONS.items():
        first, second = (values[name] for values in summary["point"])
        differences = summary["bootstrap"][0][name] - summary["bootstrap"][1][name]
        valid = differences[np.isfinite(differences)]
        limits = np.percentile(valid, [2.5, 97.5]) if valid.size else (None, None)
        weights = summary["data"][0][:, STATISTIC_COLUMNS.index(denominator)]
        rows.append({"group": summary["group"], "lead_day": summary["lead_day"], "metric": name,
                     "model_a": _model_name(state["first"]), "model_b": _model_name(state["second"]),
                     "value_a": _finite_or_none(first), "value_b": _finite_or_none(second),
                     "mean_difference": _finite_or_none(first - second),
                     "ci_lower": _finite_or_none(limits[0]), "ci_upper": _finite_or_none(limits[1]),
                     "units": "degree_Celsius km-1" if name.startswith("gradient_") else "degree_Celsius",
                     "n_pairs": int(len(weights)), "n_valid_pairs": int((weights > 0).sum()),
                     "n_valid_replicates": int(valid.size), "n_blocks": state["blocks"]["n_blocks"],
                     "n_continuous_runs": state["blocks"]["n_continuous_runs"],
                     "block_days": settings.block_days, "repetitions": settings.repetitions, "seed": settings.seed})
    return rows


def _finite_or_none(value) -> Optional[float]:
    return float(value) if value is not None and np.isfinite(value) else None


def _model_name(evaluation: dict) -> str:
    return str(evaluation["report"].get("model_name", evaluation["root"].name))


def _report(state: dict, rows: list) -> dict:
    settings, first = state["settings"], state["first"]["report"]
    return {"status": "complete", "run_identity": first["run_identity"], "split": first["split"],
            "sample_count": first["sample_count"], "lead_days": [1, 2, 3],
            "evaluation_a": _input_identity(state["first"]), "evaluation_b": _input_identity(state["second"]),
            "settings": {"block_days": settings.block_days, "repetitions": settings.repetitions,
                         "seed": settings.seed, "confidence_level": 0.95},
            "blocks": state["blocks"], "methods": _methods(), "comparisons": rows,
            "artifacts": ["comparison.csv", "comparison.json"]}


def _input_identity(evaluation: dict) -> dict:
    return {"directory": str(evaluation["root"]), "model_name": _model_name(evaluation),
            "files": {path.name: _sha256(path) for path in evaluation["paths"]}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _methods() -> dict:
    return {"difference": "Pooled statistic from model A minus model B; not mean daily RMSE or mean bootstrap difference.",
            "resampling": "Paired moving calendar blocks within each consecutive-origin run, preserving each run length; no block crosses a missing day.",
            "pairing": "The same origin multiplicities are used for both models, every lead, every group and every metric.",
            "short_run_policy": "Runs shorter than block_days are retained as one complete short block; their contribution is fixed across replicates.",
            "zero_weight_policy": "Keep zero-weight origins in calendar blocks; ratios with zero resampled denominator are undefined and excluded from percentile calculation, with valid replicate counts reported.",
            "interval": "95% percentile interval (2.5 and 97.5 percentiles, NumPy default linear interpolation).",
            "interpretation": "Negative differences in RMSE/MAE/gradient errors mean A is smaller; signed Bias differences describe relative warmth, not universal improvement. No significance or causal labels.",
            "data_scope": "Existing supplied model evaluation sufficient statistics only; no predictions or no-training baselines generated."}


def _write_comparison(root: Path, report: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "comparison.csv.tmp"
    pd.DataFrame(report["comparisons"]).to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(str(temporary), str(root / "comparison.csv"))
    write_json_atomic(root / "comparison.json", report)
