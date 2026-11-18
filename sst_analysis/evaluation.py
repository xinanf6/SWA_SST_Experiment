"""Evaluate supplied model predictions; this module never generates a forecast."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sst_preprocessing.artifacts import write_json_atomic
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.statistics import denormalize_values

from .config import validate_output_location
from .context import AnalysisContext
from .forecasts import ForecastFile
from .groups import GroupedForecastEvaluator
from .metrics import ForecastEvaluator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_forecast_file(context: AnalysisContext, options: dict) -> dict:
    """Side effect: publish metrics for an entire supplied, quality-filtered split."""
    output_root = Path(options["output_root"]).resolve()
    validate_output_location(output_root, context.root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(options["predictions"]).resolve()
    if output_root == predictions_path.parent and predictions_path.name in {
            "metrics.json", "daily_errors.csv", "metrics_summary.csv"}:
        raise DataValidationError("Evaluation output would overwrite its prediction source")
    evaluator = ForecastEvaluator(context.evaluation_grid(), options.get("front_lines", False))
    batch_size = int(options.get("batch_size", 1))
    if batch_size < 1:
        raise DataValidationError("Evaluation batch_size must be positive")
    grouped = GroupedForecastEvaluator(context, options) if "mld_threshold_m" in options else None
    try:
        with ForecastFile(predictions_path, context, options["split"]) as forecasts:
            details = _evaluate_batches(forecasts, evaluator, {"batch_size": batch_size,
                                        "output_root": output_root, "context": context, "grouped": grouped})
            report = _evaluation_report(context, forecasts, options)
        if grouped is not None:
            report["group_metrics"] = grouped.result()
            report["mld_stratification"] = options.get("mld_stratification", {"threshold_m": grouped.threshold})
            _write_csv(output_root / "group_metrics.csv", pd.DataFrame(report["group_metrics"]))
    finally:
        if grouped is not None:
            grouped.close()
    report["metrics"] = evaluator.result()
    report["daily_metric_records"] = details
    _publish_evaluation(output_root, report)
    return report


def _publish_evaluation(output_root: Path, report: dict) -> None:
    _write_csv(output_root / "metrics_summary.csv", pd.DataFrame(report["metrics"]))
    write_json_atomic(output_root / "metrics.json", report)
    write_evaluation_markdown(output_root, report)


def _evaluate_batches(forecasts: ForecastFile, evaluator: ForecastEvaluator, state: dict) -> int:
    path = state["output_root"] / "daily_errors.csv"
    temporary = path.with_name(path.name + ".tmp")
    base = np.load(state["context"].root / "normalized" / "base.npy", mmap_mode="r", allow_pickle=False)
    records_written = 0
    try:
        _validate_truth(base, forecasts)
        records_written = _run_batches(base, forecasts, dict(state, evaluator=evaluator))
    finally:
        base._mmap.close()
    os.replace(str(temporary), str(path))
    if state.get("grouped") is not None:
        group_path = state["output_root"] / "group_errors.csv"
        os.replace(str(group_path.with_suffix(".csv.tmp")), str(group_path))
    return records_written


def _run_batches(base: np.memmap, forecasts: ForecastFile, state: dict) -> int:
    records_written = 0
    for start in range(0, len(forecasts.index), state["batch_size"]):
        stop = min(start + state["batch_size"], len(forecasts.index))
        index = forecasts.index.iloc[start:stop]
        targets = _physical_targets(base, index, state["context"])
        predictions = forecasts.read(slice(start, stop))
        rows = state["evaluator"].update(predictions, targets)
        _append_records(rows, forecasts, {"start": start, "path": state["output_root"] / "daily_errors.csv.tmp"})
        records_written += len(rows)
        if state.get("grouped") is not None:
            grouped_rows = state["grouped"].update(predictions, targets, index)
            _append_records(grouped_rows, forecasts, {"start": start,
                            "path": state["output_root"] / "group_errors.csv.tmp"})
    return records_written


def _append_records(rows: list, forecasts: ForecastFile, options: dict) -> None:
    _attach_dates(rows, forecasts, options["start"])
    pd.DataFrame(rows).to_csv(options["path"], mode="w" if options["start"] == 0 else "a",
                              header=options["start"] == 0, index=False, encoding="utf-8")


def _validate_truth(base: np.memmap, forecasts: ForecastFile) -> None:
    dates = np.load(forecasts.context.root / "dates.npy", allow_pickle=False).astype("datetime64[ns]")
    if base.shape != (len(dates), 7) + forecasts.context.grid["ocean_mask"].shape or base.dtype != np.float32:
        raise DataValidationError("Prepared base array has an unexpected shape or dtype")
    indices = forecasts.index[["target_%d_index" % lead for lead in range(1, 4)]].to_numpy()
    if not np.issubdtype(indices.dtype, np.integer) or ((indices < 0) | (indices >= len(dates))).any():
        raise DataValidationError("Target indices must be integers within the prepared date array")
    if not np.array_equal(dates[indices], forecasts.target_dates):
        raise DataValidationError("Target indices and forecast target dates do not match")


def _physical_targets(base: np.memmap, index: pd.DataFrame,
                       context: AnalysisContext) -> np.ndarray:
    indices = index[["target_%d_index" % lead for lead in range(1, 4)]].to_numpy(dtype=np.int64)
    values = base[indices, 0]
    statistics = context.statistics["variables"]["sst"]
    return denormalize_values(values, statistics["mean"], statistics["std"])


def _attach_dates(rows: list, forecasts: ForecastFile, start: int) -> None:
    for row in rows:
        sample = start + int(row.pop("sample_index"))
        lead = int(row["lead_day"]) - 1
        row["origin_date"] = str(forecasts.origin_dates[sample].astype("datetime64[D]"))
        row["target_date"] = str(forecasts.target_dates[sample, lead].astype("datetime64[D]"))


def _evaluation_report(context: AnalysisContext, forecasts: ForecastFile, options: dict) -> dict:
    return {"status": "complete", "run_identity": context.run_identity,
            "split": options["split"], "model_name": options.get("model_name", forecasts.path.stem),
            "prediction_file": str(forecasts.path), "prediction_sha256": _sha256(forecasts.path),
            "prediction_units": forecasts.units, "metric_temperature_units": "degree_Celsius",
            "front_threshold_C_per_km": context.front_threshold,
            "front_lines_enabled": bool(options.get("front_lines", False)),
            "sample_count": len(forecasts.index), "lead_days": [1, 2, 3],
            "methods": {"temperature_domain": "core_mask AND ocean_mask; cosine-latitude area weights",
                        "gradients": "Full input field; reductions on core_mask AND gradient_mask",
                        "strong_front": "Reference gradient magnitude >= frozen training P90",
                        "aggregation": "Global sufficient statistics, never mean batch RMSE",
                        "r2": "1-SSE/SST about the pooled area-weighted reference mean; undefined for constant truth",
                        "vector_gradient_mae": "Area-weighted mean of (abs(dx_error)+abs(dy_error))/2",
                        "front_lines": "Raw skeleton; great-circle distances; 10 km matching tolerance; no filtering",
                        "reference": "Prepared OSTIA targets; no generated baseline predictions"}}


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _number(value, digits: int = 5) -> str:
    return "undefined" if value is None else ("%." + str(digits) + "f") % value


def write_evaluation_markdown(output_root: Path, report: dict) -> None:
    """Side effect: write a readable table in physical units, without inventing scores."""
    lines = ["# SST forecast evaluation", "", "- Model: %s" % report["model_name"],
             "- Split: %s" % report["split"], "- Origins: %d" % report["sample_count"],
             "- Temperature units: degree_Celsius",
             "- Front threshold: %.8f degree_Celsius/km" % report["front_threshold_C_per_km"], "",
             "| Lead | RMSE | MAE | Bias | Strong-front RMSE | Gradient vector MAE (C/km) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in report["metrics"]:
        values = [row[name] for name in ("rmse", "mae", "bias", "strong_front_rmse", "gradient_vector_mae")]
        lines.append("| %d | %s |" % (row["lead_day"], " | ".join(_number(value) for value in values)))
    lines.extend(["", "Scores use supplied forecasts and canonical prepared targets.",
                  "No forecasts or no-training baselines are generated by this evaluator.", ""])
    temporary = output_root / "evaluation_report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_root / "evaluation_report.md")
