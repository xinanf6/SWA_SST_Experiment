"""Explicit C0 smoke, training and trained-checkpoint prediction commands."""

import argparse
import json
from pathlib import Path
import sys

from sst_preprocessing.errors import PreprocessingError

from .config import load_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C0 training using frozen prepared SST data")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "train", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/training_c0.json"))
        command.add_argument("--output-root", type=Path)
        if name != "predict":
            command.add_argument("--resume", action="store_true")
        if name == "smoke":
            command.add_argument("--epochs", type=int, default=2)
            command.add_argument("--train-samples", type=int, default=4)
            command.add_argument("--validation-samples", type=int, default=2)
        if name == "predict":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--split", choices=("train", "validation", "test"), default="validation")
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser


def _execute(args) -> dict:
    config = load_training_config(args.config, args.output_root)
    if args.command == "predict":
        from .predict import export_predictions
        return export_predictions(config, vars(args))
    from .engine import run_training
    options = {"resume": args.resume}
    if args.command == "smoke":
        if min(args.epochs, args.train_samples, args.validation_samples) < 1:
            raise ValueError("Smoke epochs and sample limits must be positive")
        if args.output_root is None:
            config = load_training_config(args.config, config.output_root.with_name(config.output_root.name + "_smoke"))
        options.update(kind="smoke", epochs=args.epochs, train_limit=args.train_samples,
                       validation_limit=args.validation_samples)
    return run_training(config, options)


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = _execute(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (PreprocessingError, ValueError, TypeError, FloatingPointError, OSError) as error:
        print("Training failed: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
