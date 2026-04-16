#!/usr/bin/env python3
"""
Train the torch delayed localizer on the tracked corpus.
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

from gravnav.ml import evaluate_runtime_student, load_real_ocean_corpus
from gravnav.ml.torch_models import (
    TorchDelayedLocalizerTrainingSpec,
    train_torch_delayed_localizer,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the torch delayed localizer.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--offset-loss-weight", type=float, default=0.25)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--head-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--reliability-threshold", type=float, default=0.65)
    parser.add_argument("--analytic-log-emission-gain", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    model = train_torch_delayed_localizer(
        corpus,
        spec=TorchDelayedLocalizerTrainingSpec(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            offset_loss_weight=float(args.offset_loss_weight),
            head_epochs=int(args.head_epochs),
            head_learning_rate=float(args.head_learning_rate),
            reliability_threshold=float(args.reliability_threshold),
            analytic_log_emission_gain=float(args.analytic_log_emission_gain),
            device=str(args.device),
        ),
    )
    out_path = model.save_pt(Path(args.output).expanduser().resolve())
    summary = {"model_path": str(out_path)}
    summary.update(evaluate_runtime_student(corpus, model))
    summary["reference_parameter_count"] = int(model.reference_parameter_count)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
