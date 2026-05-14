#!/usr/bin/env python3
"""
Train and optionally cross-validate the hybrid sequence-feedback trust model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gravnav.ml.feedback_trust import (
    SequenceFeedbackEventCorpus,
    SequenceFeedbackTrustModelSpec,
    cross_validate_sequence_feedback_trust_model,
    fit_sequence_feedback_trust_model,
    load_sequence_feedback_failure_manifest,
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
    parser.add_argument(
        "--hard-negative-manifest",
        default=None,
        help="Optional JSON manifest of region/seed pairs weighted as primary hard negatives.",
    )
    parser.add_argument(
        "--primary-failure-negative-weight",
        type=float,
        default=4.0,
        help="Extra training weight applied to flagged lag_replay hard-negative events.",
    )
    parser.add_argument(
        "--committee-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional global corpus seeds used to train deterministic drop-seed committee members.",
    )
    parser.add_argument(
        "--committee-manifest-output-path",
        default=None,
        help="Optional JSON output path for a trained trust-model committee manifest.",
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
    committee_manifest_output_path = (
        model_output_path.with_name(model_output_path.stem + "_committee_manifest.json")
        if args.committee_manifest_output_path is None
        else _resolve_path(args.committee_manifest_output_path)
    )

    corpus = SequenceFeedbackEventCorpus.from_npz(corpus_path)
    hard_negative_manifest_path = (
        None
        if args.hard_negative_manifest is None
        else _resolve_path(args.hard_negative_manifest)
    )
    hard_negative_pairs = (
        None
        if hard_negative_manifest_path is None
        else load_sequence_feedback_failure_manifest(hard_negative_manifest_path)
    )
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
        primary_failure_negative_weight=float(args.primary_failure_negative_weight),
        covariance_scale_reference_error_m=float(
            args.covariance_scale_reference_error_m
        ),
    )

    cv_summary = None
    if not args.skip_cross_validation:
        cv_summary = cross_validate_sequence_feedback_trust_model(
            corpus,
            spec=spec,
            hard_negative_manifest=hard_negative_pairs,
        )
        cv_summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv_summary_output_path.write_text(
            json.dumps(cv_summary, indent=2) + "\n",
            encoding="utf-8",
        )

    model = fit_sequence_feedback_trust_model(
        corpus,
        spec=spec,
        hard_negative_manifest=hard_negative_pairs,
    )
    model.save_npz(model_output_path)

    committee_members: list[dict[str, object]] = []
    committee_seeds = (
        []
        if args.committee_seeds is None
        else sorted({int(seed) for seed in args.committee_seeds})
    )
    if committee_seeds:
        committee_manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
        for seed in committee_seeds:
            member_mask = np.asarray(corpus.event_seed != int(seed), dtype=bool)
            if not np.any(member_mask):
                raise ValueError(
                    f"Committee member drop_seed_{seed} would train on zero examples."
                )
            member_corpus = corpus.subset(member_mask)
            member_model = fit_sequence_feedback_trust_model(
                member_corpus,
                spec=spec,
                hard_negative_manifest=hard_negative_pairs,
            )
            member_model_path = model_output_path.with_name(
                f"{model_output_path.stem}_drop_seed_{int(seed)}.npz"
            )
            member_summary_path = member_model_path.with_name(
                member_model_path.stem + "_summary.json"
            )
            member_model.save_npz(member_model_path)
            member_summary = {
                "member_name": f"drop_seed_{int(seed)}",
                "drop_seed": int(seed),
                "corpus_path": str(corpus_path),
                "model_output_path": str(member_model_path),
                "num_examples": int(member_corpus.num_examples),
                "num_regions": int(len(member_corpus.region_names)),
                "region_example_counts": member_corpus.region_example_counts(),
                "training_metrics": member_model.metadata.get("training_metrics", {}),
                "gain_histogram_summary": member_model.metadata.get(
                    "gain_histogram_summary",
                    {},
                ),
            }
            member_summary_path.write_text(
                json.dumps(member_summary, indent=2) + "\n",
                encoding="utf-8",
            )
            committee_members.append(
                {
                    "name": f"drop_seed_{int(seed)}",
                    "drop_seed": int(seed),
                    "num_examples": int(member_corpus.num_examples),
                    "model_output_path": str(member_model_path),
                    "summary_output_path": str(member_summary_path),
                }
            )
        committee_manifest_payload = {
            "aggregator": "conservative_unanimity_v1",
            "members": [
                {
                    "name": str(member["name"]),
                    "drop_seed": int(member["drop_seed"]),
                    "model_path": os.path.relpath(
                        str(member["model_output_path"]),
                        start=str(committee_manifest_output_path.parent),
                    ),
                }
                for member in committee_members
            ],
            "metadata": {
                "corpus_path": str(corpus_path),
                "hard_negative_manifest_path": (
                    None
                    if hard_negative_manifest_path is None
                    else str(hard_negative_manifest_path)
                ),
                "committee_seeds": [int(seed) for seed in committee_seeds],
                "primary_failure_negative_weight": float(
                    args.primary_failure_negative_weight
                ),
            },
        }
        committee_manifest_output_path.write_text(
            json.dumps(committee_manifest_payload, indent=2) + "\n",
            encoding="utf-8",
        )

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
        "primary_failure_negative_weight": float(
            model.spec.primary_failure_negative_weight
        ),
        "hard_negative_manifest_path": (
            None
            if hard_negative_manifest_path is None
            else str(hard_negative_manifest_path)
        ),
        "training_metrics": model.metadata.get("training_metrics", {}),
        "gain_histogram_summary": model.metadata.get("gain_histogram_summary", {}),
        "cv_summary_path": (
            None if cv_summary is None else str(cv_summary_output_path)
        ),
        "committee_manifest_output_path": (
            None if not committee_seeds else str(committee_manifest_output_path)
        ),
        "committee_members": committee_members,
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote trust model: {model_output_path}")
    print(f"Wrote summary: {summary_output_path}")
    if cv_summary is not None:
        print(f"Wrote CV summary: {cv_summary_output_path}")
    if committee_seeds:
        print(f"Wrote committee manifest: {committee_manifest_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
