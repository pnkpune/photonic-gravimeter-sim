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
    TorchRuntimeStudentModelSpec,
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
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-validation-examples-per-region", type=int, default=1)
    parser.add_argument("--min-train-examples-per-region", type=int, default=1)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--entropy-regularization-weight", type=float, default=0.05)
    parser.add_argument(
        "--publishable-entropy-target-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--ambiguous-entropy-target-fraction",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--support-expansion-entropy-target-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--head-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--reliability-threshold", type=float, default=0.65)
    parser.add_argument("--analytic-log-emission-gain", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--query-hidden-dim", type=int, default=128)
    parser.add_argument("--candidate-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
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
            label_smoothing=float(args.label_smoothing),
            validation_fraction=float(args.validation_fraction),
            min_validation_examples_per_region=int(
                args.min_validation_examples_per_region
            ),
            min_train_examples_per_region=int(args.min_train_examples_per_region),
            min_epochs=int(args.min_epochs),
            early_stopping_patience=int(args.early_stopping_patience),
            lr_scheduler_patience=int(args.lr_scheduler_patience),
            lr_scheduler_factor=float(args.lr_scheduler_factor),
            min_learning_rate=float(args.min_learning_rate),
            gradient_clip_norm=float(args.gradient_clip_norm),
            entropy_regularization_weight=float(args.entropy_regularization_weight),
            publishable_entropy_target_fraction=float(
                args.publishable_entropy_target_fraction
            ),
            ambiguous_entropy_target_fraction=float(
                args.ambiguous_entropy_target_fraction
            ),
            support_expansion_entropy_target_fraction=float(
                args.support_expansion_entropy_target_fraction
            ),
            head_epochs=int(args.head_epochs),
            head_learning_rate=float(args.head_learning_rate),
            reliability_threshold=float(args.reliability_threshold),
            analytic_log_emission_gain=float(args.analytic_log_emission_gain),
            device=str(args.device),
            random_seed=int(args.seed),
        ),
        model_spec=TorchRuntimeStudentModelSpec(
            candidate_feature_names=tuple(corpus.metadata["candidate_feature_names"]),
            query_feature_names=tuple(corpus.metadata["query_feature_names"]),
            embedding_dim=int(args.embedding_dim),
            query_hidden_dim=int(args.query_hidden_dim),
            candidate_hidden_dim=int(args.candidate_hidden_dim),
            fusion_hidden_dim=int(args.fusion_hidden_dim),
            head_hidden_dim=int(args.head_hidden_dim),
            dropout_prob=float(args.dropout_prob),
        ),
    )
    out_path = model.save_pt(Path(args.output).expanduser().resolve())
    summary = {"model_path": str(out_path)}
    summary.update(evaluate_runtime_student(corpus, model))
    summary["reference_parameter_count"] = int(model.reference_parameter_count)
    summary["seed"] = int(args.seed)
    summary["embedding_dim"] = int(args.embedding_dim)
    summary["query_hidden_dim"] = int(args.query_hidden_dim)
    summary["candidate_hidden_dim"] = int(args.candidate_hidden_dim)
    summary["fusion_hidden_dim"] = int(args.fusion_hidden_dim)
    summary["head_hidden_dim"] = int(args.head_hidden_dim)
    summary["dropout_prob"] = float(args.dropout_prob)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
