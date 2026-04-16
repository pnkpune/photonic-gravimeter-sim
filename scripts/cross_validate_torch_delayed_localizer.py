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

from gravnav.ml import evaluate_runtime_student, load_real_ocean_corpus
from gravnav.ml.torch_models import (
    TorchDelayedLocalizerTrainingSpec,
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
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--head-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--reliability-threshold", type=float, default=0.65)
    parser.add_argument("--analytic-log-emission-gain", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
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
        head_epochs=int(args.head_epochs),
        head_learning_rate=float(args.head_learning_rate),
        reliability_threshold=float(args.reliability_threshold),
        analytic_log_emission_gain=float(args.analytic_log_emission_gain),
        device=str(args.device),
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
        model = train_torch_delayed_localizer(train_corpus, spec=train_spec)
        metrics = evaluate_runtime_student(eval_corpus, model)
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
        "aggregate": {
            "median_top1_accuracy": float(
                sorted(float(f["top1_accuracy"]) for f in folds)[len(folds) // 2]
            ),
            "median_horizontal_error_m": float(
                sorted(float(f["median_horizontal_error_m"]) for f in folds)[
                    len(folds) // 2
                ]
            ),
            "median_p90_horizontal_error_m": float(
                sorted(float(f["p90_horizontal_error_m"]) for f in folds)[
                    len(folds) // 2
                ]
            ),
            "median_mean_publishability_probability": float(
                sorted(float(f["mean_publishability_probability"]) for f in folds)[
                    len(folds) // 2
                ]
            ),
            "median_publishability_positive_fraction": float(
                sorted(float(f["publishability_positive_fraction"]) for f in folds)[
                    len(folds) // 2
                ]
            ),
        },
        "train_spec": {
            "epochs": int(train_spec.epochs),
            "batch_size": int(train_spec.batch_size),
            "learning_rate": float(train_spec.learning_rate),
            "weight_decay": float(train_spec.weight_decay),
            "offset_loss_weight": float(train_spec.offset_loss_weight),
            "head_epochs": int(train_spec.head_epochs),
            "head_learning_rate": float(train_spec.head_learning_rate),
            "reliability_threshold": float(train_spec.reliability_threshold),
            "analytic_log_emission_gain": float(train_spec.analytic_log_emission_gain),
            "device": str(train_spec.device),
        },
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
