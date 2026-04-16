#!/usr/bin/env python3
"""
Run leave-one-region-out validation for the experimental learned delayed localizer.
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
    TeacherTrainingSpec,
    cross_validate_delayed_localizer,
    load_real_ocean_corpus,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run held-out region validation for the experimental learned delayed localizer."
        )
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--held-out-regions", nargs="+", default=None)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--analytic-log-emission-gain", type=float, default=0.35)
    parser.add_argument("--reliability-threshold", type=float, default=0.65)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    summary = cross_validate_delayed_localizer(
        corpus,
        held_out_regions=(
            None
            if args.held_out_regions is None
            else tuple(str(region) for region in args.held_out_regions)
        ),
        teacher_spec=TeacherTrainingSpec(
            latent_dim=int(args.latent_dim),
            num_latents=int(args.num_latents),
        ),
        train_spec=DelayedLocalizerTrainingSpec(
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            l2=float(args.l2),
            analytic_log_emission_gain=float(args.analytic_log_emission_gain),
            reliability_threshold=float(args.reliability_threshold),
        ),
    )
    summary["corpus_path"] = str(Path(args.corpus).expanduser().resolve())
    text = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
