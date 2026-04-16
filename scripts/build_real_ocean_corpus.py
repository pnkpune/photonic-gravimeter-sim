#!/usr/bin/env python3
"""
Build a real-data-first Earth-signature corpus from tracked demo packs.
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

from gravnav.estimators.gravity_sequence_match import GravitySequenceMatcherSpec
from gravnav.ml.data import RealOceanCorpusSpec, build_real_ocean_corpus
from gravnav.utils.config import load_config_mapping

DEFAULT_PACKS = [
    PROJECT_ROOT / "data/bathymetry/processed/norwegian_margin_maritime_priority9_emodnet_demo_pack.json",
    PROJECT_ROOT / "data/bathymetry/processed/helgeland_offshore_priority9_emodnet_demo_pack.json",
    PROJECT_ROOT / "data/bathymetry/processed/nordland_offshore_priority9_emodnet_demo_pack.json",
]
DEFAULT_SEQUENCE_PROFILE = PROJECT_ROOT / "configs/sequence_profiles/public_offshore_locked.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/ml/real_ocean_corpus"


def _load_sequence_spec(path: Path) -> GravitySequenceMatcherSpec:
    profile = dict(load_config_mapping(path))
    return GravitySequenceMatcherSpec(
        window_size=int(profile["window_size"]),
        grid_half_span_m=tuple(profile["grid_half_span_m"]),
        grid_spacing_m=tuple(profile["grid_spacing_m"]),
        transition_std_m=tuple(profile["transition_std_m"]),
        center_prior_std_m=tuple(profile["center_prior_std_m"]),
        gravity_meas_std_mps2=float(profile["gravity_meas_std_mps2"]),
        gradient_meas_std_per_s2=float(profile["gradient_meas_std_per_s2"]),
        bathymetry_meas_std_m=float(profile.get("bathymetry_meas_std_m", 12.0)),
        bathymetry_weight=float(profile.get("bathymetry_weight", 1.0)),
        bathymetry_gradient_meas_std_m_per_m=float(
            profile.get("bathymetry_gradient_meas_std_m_per_m", 0.01)
        ),
        bathymetry_gradient_weight=float(
            profile.get("bathymetry_gradient_weight", 1.0)
        ),
        bathymetry_rugosity_meas_std_m=float(
            profile.get("bathymetry_rugosity_meas_std_m", 2.0)
        ),
        bathymetry_rugosity_weight=float(
            profile.get("bathymetry_rugosity_weight", 1.0)
        ),
        magnetic_meas_std_nt=float(profile.get("magnetic_meas_std_nt", 8.0)),
        magnetic_weight=float(profile.get("magnetic_weight", 1.0)),
        magnetic_gradient_meas_std_nt_per_m=float(
            profile.get("magnetic_gradient_meas_std_nt_per_m", 0.02)
        ),
        magnetic_gradient_weight=float(
            profile.get("magnetic_gradient_weight", 0.75)
        ),
        height_std_m=float(profile.get("height_std_m", 2.0)),
        adaptive_grid_enabled=bool(profile.get("adaptive_grid_enabled", True)),
        expanded_grid_half_span_m=profile.get("expanded_grid_half_span_m", None),
        expanded_grid_spacing_m=profile.get("expanded_grid_spacing_m", None),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a tracked real-ocean corpus for the learned localizer."
    )
    parser.add_argument(
        "--demo-pack-manifests",
        nargs="+",
        default=[str(p) for p in DEFAULT_PACKS],
    )
    parser.add_argument(
        "--sequence-profile",
        default=str(DEFAULT_SEQUENCE_PROFILE),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-examples-per-region", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--patch-spacing-m", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_spec = _load_sequence_spec(
        Path(args.sequence_profile).expanduser().resolve()
    )
    corpus = build_real_ocean_corpus(
        [Path(p).expanduser().resolve() for p in args.demo_pack_manifests],
        sequence_spec=sequence_spec,
        corpus_spec=RealOceanCorpusSpec(
            window_size=int(sequence_spec.window_size),
            patch_size=int(args.patch_size),
            patch_spacing_m=float(args.patch_spacing_m),
            max_examples_per_region=int(args.max_examples_per_region),
            random_seed=int(args.seed),
        ),
    )
    corpus_path = corpus.save_npz(output_dir / "real_ocean_corpus.npz")
    summary = {
        "corpus_path": str(corpus_path),
        "num_examples": int(corpus.query_windows.shape[0]),
        "num_regions": int(len(corpus.region_names)),
        "region_names": list(corpus.region_names),
        "query_window_shape": list(corpus.query_windows.shape),
        "candidate_shape": list(corpus.candidate_features.shape),
        "patch_tensor_shape": list(corpus.patch_tensors.shape),
    }
    (output_dir / "real_ocean_corpus_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
