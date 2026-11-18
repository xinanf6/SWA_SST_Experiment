"""CLI for observed training diagnostics and evaluation of supplied model forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sst_preprocessing.errors import DataValidationError, PreprocessingError

from .config import load_analysis_config, validate_output_location
from .context import load_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SWA SST evaluation and observed regional diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("diagnose", "plot-diagnostics", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/analysis.json"))
        command.add_argument("--prepared-root", type=Path)
        command.add_argument("--output-root", type=Path, required=name == "evaluate")
        if name != "plot-diagnostics":
            command.add_argument("--overwrite", action="store_true")
        if name == "evaluate":
            command.add_argument("--predictions", type=Path, required=True)
            command.add_argument("--split", choices=("train", "validation", "test"), default="validation")
            command.add_argument("--model-name")
            command.add_argument("--front-lines", action="store_true")
        if name == "diagnose":
            command.add_argument("--no-plots", action="store_true")
    _add_comparison_parser(commands)
    return parser


def _add_comparison_parser(commands) -> None:
    command = commands.add_parser("compare", help="Paired block-bootstrap comparison of supplied models")
    command.add_argument("--config", type=Path, default=Path("configs/analysis.json"))
    command.add_argument("--prepared-root", type=Path)
    command.add_argument("--evaluation-a", type=Path, required=True)
    command.add_argument("--evaluation-b", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20261117)
    command.add_argument("--overwrite", action="store_true")


def _check_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise DataValidationError("Analysis output is not empty; choose another output or pass --overwrite")


def _diagnose(context, settings, args) -> dict:
    from .diagnostics import run_diagnostics
    from .plots import render_diagnostic_figures
    from .reports import validate_diagnostic_artifacts, write_analysis_manifest, write_diagnostic_report
    from .stratification import fit_training_mld_threshold
    root = (args.output_root or settings.diagnostics_output).resolve()
    validate_output_location(root, context.root)
    if args.command == "diagnose":
        _check_output(root, args.overwrite)
        summary = run_diagnostics(context, root)
        fit_training_mld_threshold(context, root)
    else:
        summary = validate_diagnostic_artifacts(context, root)
    figures = [] if getattr(args, "no_plots", False) else render_diagnostic_figures(
        context, root, {"dpi": settings.figure_dpi})
    write_diagnostic_report(context, root, {"figures": figures})
    artifacts = dict(summary["artifacts"], figures=figures, report="diagnostics_report.md")
    write_analysis_manifest(context, root, {"kind": "training_regional_diagnostics",
                            "settings": {"figure_dpi": settings.figure_dpi,
                                         "train_period": list(context.train_period)}, "artifacts": artifacts})
    return {"status": "complete", "output_root": str(root), "training_days": summary["training_day_count"],
            "training_months": len(summary["source_files"]), "figure_files": figures}


def _evaluate(context, settings, args) -> dict:
    from .evaluation import evaluate_forecast_file
    from .reports import write_analysis_manifest
    from .stratification import fit_training_mld_threshold
    root = args.output_root.resolve()
    validate_output_location(root, context.root)
    _check_output(root, args.overwrite)
    if not args.predictions.is_file():
        raise DataValidationError("Supplied prediction file does not exist: %s" % args.predictions)
    stratification = fit_training_mld_threshold(context, settings.diagnostics_output)
    options = {"predictions": args.predictions, "split": args.split, "output_root": root,
               "front_lines": args.front_lines, "batch_size": settings.batch_size,
               "mld_threshold_m": stratification["threshold_m"], "mld_stratification": stratification}
    if args.model_name is not None:
        options["model_name"] = args.model_name
    report = evaluate_forecast_file(context, options)
    write_analysis_manifest(context, root, {"kind": "supplied_model_forecast_evaluation",
                            "settings": {"split": args.split, "front_lines": args.front_lines},
                            "artifacts": ["metrics.json", "metrics_summary.csv",
                                          "daily_errors.csv", "group_errors.csv", "group_metrics.csv",
                                          "evaluation_report.md"]})
    return report


def _compare(context, settings, args) -> dict:
    from .comparison import compare_evaluations
    root = args.output_root.resolve()
    validate_output_location(root, context.root)
    _check_output(root, args.overwrite)
    return compare_evaluations({"evaluation_a": args.evaluation_a, "evaluation_b": args.evaluation_b,
                               "output_root": root, "prepared_root": context.root, "seed": args.seed,
                               "overwrite": args.overwrite})


def main() -> int:
    args = build_parser().parse_args()
    try:
        settings = load_analysis_config(args.config, args.prepared_root)
        context = load_context(settings.prepared_root)
        handlers = {"evaluate": _evaluate, "compare": _compare,
                    "diagnose": _diagnose, "plot-diagnostics": _diagnose}
        result = handlers[args.command](context, settings, args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except PreprocessingError as error:
        print("Analysis failed: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
