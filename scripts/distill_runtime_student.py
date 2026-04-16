#!/usr/bin/env python3
"""
Distill the runtime student from the teacher and corpus.
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

from gravnav.ml import OceanTeacherModel, distill_runtime_student, load_real_ocean_corpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill the runtime student model.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analytic-log-emission-gain", type=float, default=0.35)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    teacher = OceanTeacherModel.from_npz(Path(args.teacher).expanduser().resolve())
    student = distill_runtime_student(
        corpus,
        teacher,
        analytic_log_emission_gain=float(args.analytic_log_emission_gain),
    )
    out_path = student.save_npz(Path(args.output).expanduser().resolve())
    summary = {
        "student_init_path": str(out_path),
        "embedding_dim": int(student.spec.embedding_dim),
        "reference_parameter_count": int(student.reference_parameter_count),
        "target_parameter_count": int(student.spec.target_parameter_count),
        "target_reliability_head_parameter_count": int(
            student.spec.target_reliability_head_parameter_count
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
