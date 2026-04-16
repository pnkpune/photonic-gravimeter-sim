#!/usr/bin/env python3
"""
Fit the offline teacher world model from a tracked corpus.
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

from gravnav.ml import TeacherTrainingSpec, load_real_ocean_corpus, train_ocean_teacher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the offline ocean teacher model.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--num-latents", type=int, default=16)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    teacher = train_ocean_teacher(
        corpus,
        spec=TeacherTrainingSpec(
            latent_dim=int(args.latent_dim),
            num_latents=int(args.num_latents),
        ),
    )
    out_path = teacher.save_npz(Path(args.output).expanduser().resolve())
    summary = {
        "teacher_path": str(out_path),
        "latent_dim": int(teacher.spec.latent_dim),
        "num_latents": int(teacher.spec.num_latents),
        "reference_parameter_count": int(teacher.reference_parameter_count),
        "target_parameter_count": int(teacher.spec.target_parameter_count),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
