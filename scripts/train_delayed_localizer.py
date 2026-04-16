#!/usr/bin/env python3
"""
Train the delayed localizer heads on the tracked corpus.
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

from gravnav.ml import (
    DelayedLocalizerTrainingSpec,
    RuntimeStudentModel,
    evaluate_runtime_student,
    load_real_ocean_corpus,
    train_delayed_localizer,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the delayed runtime localizer.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--student-init", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--reliability-threshold", type=float, default=0.65)
    parser.add_argument("--analytic-log-emission-gain", type=float, default=0.35)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    student = RuntimeStudentModel.from_npz(
        Path(args.student_init).expanduser().resolve()
    )
    trained = train_delayed_localizer(
        corpus,
        student,
        spec=DelayedLocalizerTrainingSpec(
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            l2=float(args.l2),
            reliability_threshold=float(args.reliability_threshold),
            analytic_log_emission_gain=float(args.analytic_log_emission_gain),
        ),
    )
    out_path = trained.save_npz(Path(args.output).expanduser().resolve())
    summary = {"model_path": str(out_path)}
    summary.update(evaluate_runtime_student(corpus, trained))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
