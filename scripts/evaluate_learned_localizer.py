#!/usr/bin/env python3
"""
Evaluate the delayed localizer on the tracked corpus.
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

from gravnav.ml import RuntimeStudentModel, evaluate_runtime_student, load_real_ocean_corpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the learned delayed localizer.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional subset of region names to evaluate.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    if args.regions is not None:
        requested = tuple(str(region) for region in args.regions)
        missing = [region for region in requested if region not in corpus.region_names]
        if len(missing) > 0:
            raise SystemExit(f"Unknown regions: {missing}.")
        corpus = corpus.select_regions(include=requested, name="cli_region_subset")
    model = RuntimeStudentModel.from_npz(Path(args.model).expanduser().resolve())
    summary = evaluate_runtime_student(corpus, model)
    summary["model_path"] = str(Path(args.model).expanduser().resolve())
    summary["num_examples"] = int(corpus.query_windows.shape[0])
    summary["region_names"] = list(corpus.region_names)
    summary["region_example_counts"] = corpus.region_example_counts()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
