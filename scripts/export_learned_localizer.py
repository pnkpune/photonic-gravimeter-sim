#!/usr/bin/env python3
"""
Export the runtime learned-localizer bundle with optional threshold overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gravnav.ml import RuntimeStudentModel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the learned localizer bundle.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reliability-threshold", type=float, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    model = RuntimeStudentModel.from_npz(Path(args.model).expanduser().resolve())
    if args.reliability_threshold is not None:
        model.reliability_threshold = float(args.reliability_threshold)
    out_path = model.save_npz(Path(args.output).expanduser().resolve())
    summary = {
        "bundle_path": str(out_path),
        "reference_parameter_count": int(model.reference_parameter_count),
        "target_parameter_count": int(model.spec.target_parameter_count),
        "reliability_threshold": float(model.reliability_threshold),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
