#!/usr/bin/env python3
"""
Evaluate the delayed localizer on the tracked corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gravnav.ml import RuntimeStudentModel, evaluate_runtime_student, load_real_ocean_corpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the learned delayed localizer.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--held-out-region", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    corpus = load_real_ocean_corpus(Path(args.corpus).expanduser().resolve())
    if args.held_out_region is not None:
        if args.held_out_region not in corpus.region_names:
            raise SystemExit(f"Unknown region {args.held_out_region!r}.")
        region_idx = corpus.region_names.index(args.held_out_region)
        mask = corpus.region_index == region_idx
        corpus = type(corpus)(
            spec=corpus.spec,
            patch_tensors=corpus.patch_tensors[mask],
            patch_summary_features=corpus.patch_summary_features[mask],
            query_windows=corpus.query_windows[mask],
            candidate_features=corpus.candidate_features[mask],
            candidate_offsets_ned_m=corpus.candidate_offsets_ned_m[mask],
            analytic_log_emission=corpus.analytic_log_emission[mask],
            labels=corpus.labels[mask],
            truth_offsets_ned_m=corpus.truth_offsets_ned_m[mask],
            publishability_labels=corpus.publishability_labels[mask],
            covariance_targets=corpus.covariance_targets[mask],
            region_names=(args.held_out_region,),
            region_index=np.zeros(int(np.sum(mask)), dtype=np.int64),
            metadata=corpus.metadata,
        )
    model = RuntimeStudentModel.from_npz(Path(args.model).expanduser().resolve())
    summary = evaluate_runtime_student(corpus, model)
    summary["model_path"] = str(Path(args.model).expanduser().resolve())
    summary["num_examples"] = int(corpus.query_windows.shape[0])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
