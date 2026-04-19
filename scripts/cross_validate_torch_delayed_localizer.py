#!/usr/bin/env python3
"""
Run leave-one-region-out validation for the torch delayed localizer.
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
    aggregate_runtime_student_folds,
    DEFAULT_FIXED_PUBLISHABILITY_THRESHOLD,
    evaluate_runtime_student,
    load_real_ocean_corpus,
)
from gravnav.ml.torch_models import (
    TorchDelayedLocalizerTrainingSpec,
    TorchRuntimeStudentModelSpec,
    train_torch_delayed_localizer,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run held-out region validation for the torch delayed localizer."
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--held-out-regions", nargs="+", default=None)
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
    parser.add_argument("--region-balance-power", type=float, default=1.0)
    parser.add_argument("--label-balance-power", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--query-hidden-dim", type=int, default=128)
    parser.add_argument("--candidate-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--publishability-reporting-mode",
        choices=("single", "dual"),
        default="dual",
    )
    parser.add_argument(
        "--fixed-publishability-threshold",
        type=float,
        default=DEFAULT_FIXED_PUBLISHABILITY_THRESHOLD,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    regions = (
        corpus.region_names
        if args.held_out_regions is None
        else tuple(str(region) for region in args.held_out_regions)
    )
    unknown = [region for region in regions if region not in corpus.region_names]
    if len(unknown) > 0:
        raise SystemExit(f"Unknown held-out regions: {unknown}.")

    train_spec = TorchDelayedLocalizerTrainingSpec(
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
        region_balance_power=float(args.region_balance_power),
        label_balance_power=float(args.label_balance_power),
        device=str(args.device),
        random_seed=int(args.seed),
    )
    model_spec = TorchRuntimeStudentModelSpec(
        candidate_feature_names=tuple(corpus.metadata["candidate_feature_names"]),
        query_feature_names=tuple(corpus.metadata["query_feature_names"]),
        embedding_dim=int(args.embedding_dim),
        query_hidden_dim=int(args.query_hidden_dim),
        candidate_hidden_dim=int(args.candidate_hidden_dim),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
        head_hidden_dim=int(args.head_hidden_dim),
        dropout_prob=float(args.dropout_prob),
    )
    folds = []
    for held_out_region in regions:
        train_corpus = corpus.select_regions(
            exclude=(held_out_region,),
            name=f"torch_train_excluding_{held_out_region}",
        )
        eval_corpus = corpus.select_regions(
            include=(held_out_region,),
            name=f"torch_eval_{held_out_region}",
        )
        model = train_torch_delayed_localizer(
            train_corpus,
            spec=train_spec,
            model_spec=model_spec,
        )
        metrics = evaluate_runtime_student(
            eval_corpus,
            model,
            publishability_reporting_mode=str(args.publishability_reporting_mode),
            fixed_publishability_threshold=float(args.fixed_publishability_threshold),
        )
        folds.append(
            {
                "held_out_region": held_out_region,
                "train_examples": int(train_corpus.num_examples),
                "eval_examples": int(eval_corpus.num_examples),
                "train_region_example_counts": train_corpus.region_example_counts(),
                "eval_region_example_counts": eval_corpus.region_example_counts(),
                "reference_parameter_count": int(model.reference_parameter_count),
                **metrics,
            }
        )

    summary = {
        "num_folds": int(len(folds)),
        "folds": folds,
        "aggregate": aggregate_runtime_student_folds(folds),
        "train_spec": {
            "epochs": int(train_spec.epochs),
            "batch_size": int(train_spec.batch_size),
            "learning_rate": float(train_spec.learning_rate),
            "weight_decay": float(train_spec.weight_decay),
            "offset_loss_weight": float(train_spec.offset_loss_weight),
            "label_smoothing": float(train_spec.label_smoothing),
            "validation_fraction": float(train_spec.validation_fraction),
            "min_validation_examples_per_region": int(
                train_spec.min_validation_examples_per_region
            ),
            "min_train_examples_per_region": int(
                train_spec.min_train_examples_per_region
            ),
            "min_epochs": int(train_spec.min_epochs),
            "early_stopping_patience": int(train_spec.early_stopping_patience),
            "lr_scheduler_patience": int(train_spec.lr_scheduler_patience),
            "lr_scheduler_factor": float(train_spec.lr_scheduler_factor),
            "min_learning_rate": float(train_spec.min_learning_rate),
            "gradient_clip_norm": float(train_spec.gradient_clip_norm),
            "entropy_regularization_weight": float(
                train_spec.entropy_regularization_weight
            ),
            "publishable_entropy_target_fraction": float(
                train_spec.publishable_entropy_target_fraction
            ),
            "ambiguous_entropy_target_fraction": float(
                train_spec.ambiguous_entropy_target_fraction
            ),
            "support_expansion_entropy_target_fraction": float(
                train_spec.support_expansion_entropy_target_fraction
            ),
            "head_epochs": int(train_spec.head_epochs),
            "head_learning_rate": float(train_spec.head_learning_rate),
            "reliability_threshold": float(train_spec.reliability_threshold),
            "analytic_log_emission_gain": float(train_spec.analytic_log_emission_gain),
            "region_balance_power": float(train_spec.region_balance_power),
            "label_balance_power": float(train_spec.label_balance_power),
            "device": str(train_spec.device),
            "random_seed": int(train_spec.random_seed),
        },
        "model_spec": {
            "embedding_dim": int(model_spec.embedding_dim),
            "query_hidden_dim": int(model_spec.query_hidden_dim),
            "candidate_hidden_dim": int(model_spec.candidate_hidden_dim),
            "fusion_hidden_dim": int(model_spec.fusion_hidden_dim),
            "head_hidden_dim": int(model_spec.head_hidden_dim),
            "dropout_prob": float(model_spec.dropout_prob),
        },
        "publishability_reporting_mode": str(args.publishability_reporting_mode),
        "fixed_publishability_threshold": float(args.fixed_publishability_threshold),
        "corpus_path": str(Path(args.corpus).expanduser().resolve()),
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
