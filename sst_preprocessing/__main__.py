"""Command-line entry point; offline tests never import root download scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .errors import PreprocessingError
from .pipeline import run_preprocessing
from .sources import build_inventory
from .verification import verify_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original SWA SST preprocessing protocol")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "smoke", "run", "verify", "rebuild"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/preprocess.json"))
        command.add_argument("--output-root", type=Path)
        if name in ("run", "smoke", "rebuild"):
            command.add_argument("--resume", action="store_true")
        if name == "rebuild":
            command.add_argument("--source-root", type=Path, required=True)
        if name == "verify":
            command.add_argument("--quick", action="store_true", help="Skip full physical round-trip checks")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config, output_root=args.output_root)
        if args.command == "inspect":
            inventory = build_inventory(config)
            result = {"selected_files": len(inventory["files"]),
                      "excluded_files": inventory["excluded_files"],
                      "source_fingerprint": inventory["fingerprint"]}
        elif args.command == "verify":
            result = verify_output(config.output_root, deep=not args.quick)
        elif args.command == "smoke":
            if args.output_root is None:
                config = load_config(args.config, output_root=config.output_root.parent / "smoke")
            result = run_preprocessing(config, resume=args.resume, stop_after_months=1)
        elif args.command == "rebuild":
            from .rebuild import rebuild_from_physical
            result = rebuild_from_physical(config, args.source_root, args.resume)
        else:
            result = run_preprocessing(config, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except PreprocessingError as error:
        print("Preprocessing failed: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
