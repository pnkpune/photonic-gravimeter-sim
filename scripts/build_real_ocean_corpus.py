#!/usr/bin/env python3
"""
Build a real-data-first Earth-signature corpus from tracked demo packs.
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

from gravnav.estimators.gravity_sequence_match import GravitySequenceMatcherSpec
from gravnav.ml.data import RealOceanCorpusSpec, build_real_ocean_corpus
from gravnav.ml.experiment_registry import (
    DEFAULT_EXPERIMENT_CONFIG,
    list_corpus_presets,
    list_region_sets,
    resolve_corpus_preset,
    resolve_region_set,
)
from gravnav.utils.config import load_config_mapping

DEFAULT_SEQUENCE_PROFILE = PROJECT_ROOT / "configs/sequence_profiles/public_offshore_locked.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/ml/real_ocean_corpus"
DEFAULT_REGION_SET = "norway4"
DEFAULT_CORPUS_PRESET = "dev"


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
        default=None,
        help="Optional explicit demo-pack manifests. Overrides --region-set when provided.",
    )
    parser.add_argument(
        "--experiment-config",
        default=str(PROJECT_ROOT / DEFAULT_EXPERIMENT_CONFIG),
        help="Path to the region-set and corpus-preset config JSON.",
    )
    parser.add_argument(
        "--region-set",
        default=DEFAULT_REGION_SET,
        help=f"Named region set to resolve from the experiment config. Default: {DEFAULT_REGION_SET}.",
    )
    parser.add_argument(
        "--corpus-preset",
        default=DEFAULT_CORPUS_PRESET,
        help=f"Named corpus preset to load from the experiment config. Default: {DEFAULT_CORPUS_PRESET}.",
    )
    parser.add_argument(
        "--strict-region-set",
        action="store_true",
        help="Fail if the selected region set references manifests that are not present locally.",
    )
    parser.add_argument(
        "--list-region-sets",
        action="store_true",
        help="Print the available named region sets from the experiment config and exit.",
    )
    parser.add_argument(
        "--list-corpus-presets",
        action="store_true",
        help="Print the available named corpus presets from the experiment config and exit.",
    )
    parser.add_argument(
        "--sequence-profile",
        default=str(DEFAULT_SEQUENCE_PROFILE),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-examples-per-region", type=int, default=64)
    parser.add_argument(
        "--num-offset-realizations-per-region",
        type=int,
        default=4,
        help=(
            "How many dead-reckoning drift realizations to generate per tracked region."
        ),
    )
    parser.add_argument(
        "--num-edge-biased-realizations-per-region",
        type=int,
        default=0,
        help=(
            "Additional drift realizations biased toward the nominal candidate-grid "
            "boundary or slightly beyond it."
        ),
    )
    parser.add_argument(
        "--num-route-variants-per-region",
        type=int,
        default=1,
        help=(
            "How many accepted translated route variants to generate per tracked region, "
            "including the original tracked route."
        ),
    )
    parser.add_argument(
        "--route-variant-max-attempts",
        type=int,
        default=24,
        help="Maximum number of translated-route proposals to test per region.",
    )
    parser.add_argument(
        "--route-variant-margin-m",
        type=float,
        default=250.0,
        help="Safety margin from region support bounds when translating routes.",
    )
    parser.add_argument(
        "--route-variant-min-separation-m",
        type=float,
        default=1000.0,
        help="Minimum separation between accepted translated route starts.",
    )
    parser.add_argument(
        "--edge-bias-min-fraction-of-nominal-half-span",
        type=float,
        default=0.8,
        help="Minimum fraction of the nominal grid half-span used for edge-biased priors.",
    )
    parser.add_argument(
        "--edge-bias-max-fraction-of-nominal-half-span",
        type=float,
        default=1.25,
        help="Maximum fraction of the nominal grid half-span used for edge-biased priors.",
    )
    parser.add_argument(
        "--centered-clamp-fraction-of-nominal-half-span",
        type=float,
        default=0.6,
        help=(
            "Clamp centered drift realizations to this fraction of the nominal "
            "grid half-span so the corpus contains genuine in-support cases."
        ),
    )
    parser.add_argument(
        "--disable-expanded-grid-for-edge-biased-realizations",
        action="store_true",
        help="Keep edge-biased examples on the nominal grid instead of using expanded support.",
    )
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--patch-spacing-m", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    experiment_config_path = Path(args.experiment_config).expanduser().resolve()

    if bool(args.list_region_sets):
        print(
            json.dumps(
                {
                    "experiment_config": str(experiment_config_path),
                    "region_sets": list(
                        list_region_sets(experiment_config_path, project_root=PROJECT_ROOT)
                    ),
                },
                indent=2,
            )
        )
        return 0
    if bool(args.list_corpus_presets):
        print(
            json.dumps(
                {
                    "experiment_config": str(experiment_config_path),
                    "corpus_presets": list(
                        list_corpus_presets(
                            experiment_config_path,
                            project_root=PROJECT_ROOT,
                        )
                    ),
                },
                indent=2,
            )
        )
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_spec = _load_sequence_spec(
        Path(args.sequence_profile).expanduser().resolve()
    )
    preset = resolve_corpus_preset(
        str(args.corpus_preset),
        path=experiment_config_path,
        project_root=PROJECT_ROOT,
    )
    if args.demo_pack_manifests is not None:
        manifest_paths = [Path(p).expanduser().resolve() for p in args.demo_pack_manifests]
        missing_manifest_paths: list[Path] = []
    else:
        region_set = resolve_region_set(
            str(args.region_set),
            path=experiment_config_path,
            project_root=PROJECT_ROOT,
        )
        missing_manifest_paths = list(region_set.missing_manifest_paths)
        if bool(args.strict_region_set) and len(missing_manifest_paths) > 0:
            raise FileNotFoundError(
                "Selected region set has missing manifests: "
                + ", ".join(str(path) for path in missing_manifest_paths)
            )
        manifest_paths = list(region_set.manifest_paths)
    if len(manifest_paths) == 0:
        raise ValueError("No demo-pack manifests were resolved for corpus building.")

    corpus = build_real_ocean_corpus(
        manifest_paths,
        sequence_spec=sequence_spec,
        corpus_spec=RealOceanCorpusSpec(
            window_size=int(sequence_spec.window_size),
            patch_size=int(args.patch_size),
            patch_spacing_m=float(args.patch_spacing_m),
            max_examples_per_region=int(
                preset.get("max_examples_per_region", args.max_examples_per_region)
            ),
            num_offset_realizations_per_region=int(
                preset.get(
                    "num_offset_realizations_per_region",
                    args.num_offset_realizations_per_region,
                )
            ),
            num_edge_biased_realizations_per_region=int(
                preset.get(
                    "num_edge_biased_realizations_per_region",
                    args.num_edge_biased_realizations_per_region,
                )
            ),
            num_route_variants_per_region=int(
                preset.get(
                    "num_route_variants_per_region",
                    args.num_route_variants_per_region,
                )
            ),
            route_variant_max_attempts=int(args.route_variant_max_attempts),
            route_variant_margin_m=float(args.route_variant_margin_m),
            route_variant_min_separation_m=float(args.route_variant_min_separation_m),
            centered_clamp_fraction_of_nominal_half_span=float(
                args.centered_clamp_fraction_of_nominal_half_span
            ),
            edge_bias_min_fraction_of_nominal_half_span=float(
                args.edge_bias_min_fraction_of_nominal_half_span
            ),
            edge_bias_max_fraction_of_nominal_half_span=float(
                args.edge_bias_max_fraction_of_nominal_half_span
            ),
            use_expanded_grid_for_edge_biased_realizations=(
                bool(
                    preset.get(
                        "use_expanded_grid_for_edge_biased_realizations",
                        not bool(args.disable_expanded_grid_for_edge_biased_realizations),
                    )
                )
            ),
            random_seed=int(args.seed),
        ),
    )
    corpus_path = corpus.save_npz(output_dir / "real_ocean_corpus.npz")
    region_example_counts = corpus.region_example_counts()
    region_counts = list(region_example_counts.values())
    region_count_median = (
        float(np.median(np.asarray(region_counts, dtype=np.float64)))
        if len(region_counts) > 0
        else 0.0
    )
    summary = {
        "corpus_path": str(corpus_path),
        "num_examples": int(corpus.query_windows.shape[0]),
        "num_regions": int(len(corpus.region_names)),
        "region_names": list(corpus.region_names),
        "input_manifests": [str(path) for path in manifest_paths],
        "missing_manifest_paths": [str(path) for path in missing_manifest_paths],
        "experiment_config": str(experiment_config_path),
        "region_set": None if args.demo_pack_manifests is not None else str(args.region_set),
        "corpus_preset": str(args.corpus_preset),
        "region_example_counts": corpus.region_example_counts(),
        "region_example_count_median": region_count_median,
        "region_example_count_min_ratio_to_median": (
            None
            if region_count_median <= 0.0
            else float(min(region_counts) / region_count_median)
        ),
        "region_example_count_max_ratio_to_median": (
            None
            if region_count_median <= 0.0
            else float(max(region_counts) / region_count_median)
        ),
        "route_variant_counts": dict(corpus.metadata.get("route_variant_counts", {})),
        "training_grid_mode_counts": dict(
            corpus.metadata.get("training_grid_mode_counts", {})
        ),
        "realization_mode_counts_by_region": dict(
            corpus.metadata.get("realization_mode_counts_by_region", {})
        ),
        "skipped_nonfinite_examples_by_region": dict(
            corpus.metadata.get("skipped_nonfinite_examples_by_region", {})
        ),
        "support_expansion_positive_fraction": float(
            corpus.support_expansion_labels.mean()
        ),
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
