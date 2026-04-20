#!/usr/bin/env python3
"""
Train and optionally cross-validate the hybrid sequence-feedback trust model.
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

from gravnav.ml.feedback_trust import (
    SequenceFeedbackEventCorpus,
    SequenceFeedbackTrustModelSpec,
    cross_validate_sequence_feedback_trust_model,
    fit_sequence_feedback_trust_model,
)


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a trust model for hybrid sequence feedback."
    )
    parser.add_argument(
        "--corpus-path",
        required=True,
        help="Input feedback corpus NPZ.",
    )
    parser.add_argument(
        "--model-output-path",
        required=True,
        help="Output NPZ for the trained trust model.",
    )
    parser.add_argument(
        "--summary-output-path",
        default=None,
        help="Optional JSON summary path. Defaults beside the model export.",
    )
    parser.add_argument(
        "--cv-summary-output-path",
        default=None,
        help="Optional JSON path for leave-one-region-out CV summary.",
    )
    parser.add_argument("--l2", type=float, default=1.0e-2)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--trust-threshold", type=float, default=0.5)
    parser.add_argument("--gain-alpha-l2", type=float, default=1.0e-2)
    parser.add_argument(
        "--gain-alpha-reference",
        type=float,
        default=0.25,
        help="Conservative fixed-gain reference used to center learned gain predictions.",
    )
    parser.add_argument(
        "--gain-alpha-prediction-scale",
        type=float,
        default=0.35,
        help="Shrinkage factor applied to learned gain deviations away from the reference.",
    )
    parser.add_argument(
        "--gain-alpha-safe-max",
        type=float,
        default=0.50,
        help="Maximum learned replay gain allowed by the calibrated gain head.",
    )
    parser.add_argument(
        "--covariance-scale-reference-error-m",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--skip-cross-validation",
        action="store_true",
        help="Skip leave-one-region-out cross-validation.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus_path = _resolve_path(args.corpus_path)
    model_output_path = _resolve_path(args.model_output_path)
    summary_output_path = (
        model_output_path.with_name(model_output_path.stem + "_summary.json")
        if args.summary_output_path is None
        else _resolve_path(args.summary_output_path)
    )
    cv_summary_output_path = (
        model_output_path.with_name(model_output_path.stem + "_cv_summary.json")
        if args.cv_summary_output_path is None
        else _resolve_path(args.cv_summary_output_path)
    )

    corpus = SequenceFeedbackEventCorpus.from_npz(corpus_path)
    spec = SequenceFeedbackTrustModelSpec(
        feature_names=corpus.feature_names,
        l2=float(args.l2),
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
        trust_threshold=float(args.trust_threshold),
        gain_alpha_l2=float(args.gain_alpha_l2),
        gain_alpha_reference=float(args.gain_alpha_reference),
        gain_alpha_prediction_scale=float(args.gain_alpha_prediction_scale),
        gain_alpha_safe_max=float(args.gain_alpha_safe_max),
        covariance_scale_reference_error_m=float(
            args.covariance_scale_reference_error_m
        ),
    )

    cv_summary = None
    if not args.skip_cross_validation:
        cv_summary = cross_validate_sequence_feedback_trust_model(corpus, spec=spec)
        cv_summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv_summary_output_path.write_text(
            json.dumps(cv_summary, indent=2) + "\n",
            encoding="utf-8",
        )

    model = fit_sequence_feedback_trust_model(corpus, spec=spec)
    model.save_npz(model_output_path)

    summary = {
        "corpus_path": str(corpus_path),
        "model_output_path": str(model_output_path),
        "num_examples": int(corpus.num_examples),
        "num_regions": int(len(corpus.region_names)),
        "region_example_counts": corpus.region_example_counts(),
        "model_name": model.spec.name,
        "trust_threshold": float(model.spec.trust_threshold),
        "gain_alpha_l2": float(model.spec.gain_alpha_l2),
        "gain_alpha_reference": float(model.spec.gain_alpha_reference),
        "gain_alpha_prediction_scale": float(model.spec.gain_alpha_prediction_scale),
        "gain_alpha_safe_max": float(model.spec.gain_alpha_safe_max),
        "training_metrics": model.metadata.get("training_metrics", {}),
        "gain_histogram_summary": model.metadata.get("gain_histogram_summary", {}),
        "cv_summary_path": (
            None if cv_summary is None else str(cv_summary_output_path)
        ),
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote trust model: {model_output_path}")
    print(f"Wrote summary: {summary_output_path}")
    if cv_summary is not None:
        print(f"Wrote CV summary: {cv_summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
