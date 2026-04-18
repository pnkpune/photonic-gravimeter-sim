#!/usr/bin/env python3
"""
Run baseline, coarse-sweep, and refine-sweep Torch learned-localizer experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.estimators.gravity_sequence_match import GravitySequenceMatcherSpec
from gravnav.ml import (
    RealOceanCorpusSpec,
    build_real_ocean_corpus,
    evaluate_runtime_student,
    load_real_ocean_corpus,
    resolve_corpus_preset,
    resolve_region_set,
)
from gravnav.ml.torch_models import (
    TorchDelayedLocalizerTrainingSpec,
    TorchRuntimeStudentModelSpec,
    train_torch_delayed_localizer,
)
from gravnav.utils.config import load_config_mapping


DEFAULT_SEQUENCE_PROFILE = PROJECT_ROOT / "configs/sequence_profiles/public_offshore_locked.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/ml/torch_experiments"

COARSE_SWEEP = [
    {
        "analytic_log_emission_gain": analytic_gain,
        "dropout_prob": dropout_prob,
        "entropy_regularization_weight": entropy_weight,
    }
    for analytic_gain in (0.25, 0.35, 0.50)
    for dropout_prob in (0.0, 0.1)
    for entropy_weight in (0.03, 0.05)
]

REFINE_SWEEP = [
    {
        "offset_loss_weight": offset_loss_weight,
        "support_expansion_entropy_target_fraction": support_target,
    }
    for offset_loss_weight in (0.15, 0.25)
    for support_target in (0.65, 0.75)
]


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


def _build_corpus(
    *,
    region_set: str,
    corpus_preset: str,
    output_dir: Path,
    sequence_profile: Path,
    experiment_config: Path,
    seed: int,
) -> Path:
    sequence_spec = _load_sequence_spec(sequence_profile)
    preset = resolve_corpus_preset(
        corpus_preset,
        path=experiment_config,
        project_root=PROJECT_ROOT,
    )
    resolved = resolve_region_set(
        region_set,
        path=experiment_config,
        project_root=PROJECT_ROOT,
    )
    corpus = build_real_ocean_corpus(
        list(resolved.manifest_paths),
        sequence_spec=sequence_spec,
        corpus_spec=RealOceanCorpusSpec(
            window_size=int(sequence_spec.window_size),
            patch_size=9,
            patch_spacing_m=40.0,
            max_examples_per_region=int(preset["max_examples_per_region"]),
            num_offset_realizations_per_region=int(
                preset["num_offset_realizations_per_region"]
            ),
            num_edge_biased_realizations_per_region=int(
                preset["num_edge_biased_realizations_per_region"]
            ),
            num_route_variants_per_region=int(preset["num_route_variants_per_region"]),
            use_expanded_grid_for_edge_biased_realizations=bool(
                preset["use_expanded_grid_for_edge_biased_realizations"]
            ),
            random_seed=int(seed),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "real_ocean_corpus.npz"
    corpus.save_npz(corpus_path)
    (output_dir / "real_ocean_corpus_summary.json").write_text(
        json.dumps(
            {
                "corpus_path": str(corpus_path),
                "region_set": str(region_set),
                "corpus_preset": str(corpus_preset),
                "num_examples": int(corpus.num_examples),
                "region_example_counts": corpus.region_example_counts(),
                "missing_manifest_paths": [
                    str(path) for path in resolved.missing_manifest_paths
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return corpus_path


def _cross_validate_torch(
    *,
    corpus_path: Path,
    train_spec: TorchDelayedLocalizerTrainingSpec,
    model_spec: TorchRuntimeStudentModelSpec,
    held_out_regions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    corpus = load_real_ocean_corpus(corpus_path)
    regions = corpus.region_names if held_out_regions is None else held_out_regions
    folds: list[dict[str, Any]] = []
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
    aggregate = {
        "median_top1_accuracy": float(np.median([float(x["top1_accuracy"]) for x in folds])),
        "median_horizontal_error_m": float(
            np.median([float(x["median_horizontal_error_m"]) for x in folds])
        ),
        "median_publishability_brier_score": float(
            np.median([float(x["publishability_brier_score"]) for x in folds])
        ),
        "median_publishability_positive_fraction": float(
            np.median([float(x["publishability_positive_fraction"]) for x in folds])
        ),
    }
    return {
        "num_folds": int(len(folds)),
        "folds": folds,
        "aggregate": aggregate,
        "train_spec": asdict(train_spec),
        "model_spec": asdict(model_spec),
        "corpus_path": str(corpus_path),
    }


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    agg = dict(result["aggregate"])
    return (
        float(agg["median_horizontal_error_m"]),
        -float(agg["median_top1_accuracy"]),
        float(agg["median_publishability_brier_score"]),
        -int(float(agg["median_publishability_positive_fraction"]) > 0.0),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline, coarse, and refine Torch localizer experiments."
    )
    parser.add_argument(
        "--stage",
        choices=("baseline", "coarse", "refine", "all"),
        default="all",
    )
    parser.add_argument("--region-set", default="all_wave1")
    parser.add_argument("--corpus-preset", default="dev")
    parser.add_argument("--experiment-config", default=str(PROJECT_ROOT / "configs/ml/real_ocean_experiments.json"))
    parser.add_argument("--sequence-profile", default=str(DEFAULT_SEQUENCE_PROFILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--corpus-path", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = Path(args.experiment_config).expanduser().resolve()
    sequence_profile = Path(args.sequence_profile).expanduser().resolve()
    corpus_path = (
        Path(args.corpus_path).expanduser().resolve()
        if str(args.corpus_path).strip()
        else _build_corpus(
            region_set=str(args.region_set),
            corpus_preset=str(args.corpus_preset),
            output_dir=output_dir / "corpus",
            sequence_profile=sequence_profile,
            experiment_config=experiment_config,
            seed=int(args.seed),
        )
    )
    corpus = load_real_ocean_corpus(corpus_path)
    base_model_spec = TorchRuntimeStudentModelSpec(
        candidate_feature_names=tuple(corpus.metadata["candidate_feature_names"]),
        query_feature_names=tuple(corpus.metadata["query_feature_names"]),
    )
    base_train_spec = TorchDelayedLocalizerTrainingSpec(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        head_epochs=int(args.head_epochs),
        device=str(args.device),
        random_seed=int(args.seed),
    )

    payload: dict[str, Any] = {
        "region_set": str(args.region_set),
        "corpus_preset": str(args.corpus_preset),
        "corpus_path": str(corpus_path),
        "stages": {},
    }
    stage = str(args.stage)
    if stage in {"baseline", "all"}:
        baseline = _cross_validate_torch(
            corpus_path=corpus_path,
            train_spec=base_train_spec,
            model_spec=base_model_spec,
        )
        payload["stages"]["baseline"] = baseline
        (output_dir / "baseline_summary.json").write_text(
            json.dumps(baseline, indent=2) + "\n",
            encoding="utf-8",
        )

    coarse_results: list[dict[str, Any]] = []
    if stage in {"coarse", "refine", "all"}:
        for idx, override in enumerate(COARSE_SWEEP):
            train_spec = TorchDelayedLocalizerTrainingSpec(
                **{
                    **asdict(base_train_spec),
                    **override,
                }
            )
            model_spec = TorchRuntimeStudentModelSpec(
                **{
                    **asdict(base_model_spec),
                    "dropout_prob": float(override["dropout_prob"]),
                }
            )
            summary = _cross_validate_torch(
                corpus_path=corpus_path,
                train_spec=train_spec,
                model_spec=model_spec,
            )
            summary["experiment_name"] = f"coarse_{idx:02d}"
            summary["override"] = dict(override)
            coarse_results.append(summary)
        coarse_results.sort(key=_rank_key)
        payload["stages"]["coarse"] = coarse_results
        (output_dir / "coarse_results.json").write_text(
            json.dumps(coarse_results, indent=2) + "\n",
            encoding="utf-8",
        )

    if stage in {"refine", "all"}:
        if len(coarse_results) == 0:
            coarse_results = json.loads(
                (output_dir / "coarse_results.json").read_text(encoding="utf-8")
            )
        top = coarse_results[: max(int(args.top_k), 1)]
        refine_results: list[dict[str, Any]] = []
        for coarse_rank, coarse_result in enumerate(top):
            base_override = dict(coarse_result.get("override", {}))
            for refine_idx, refine_override in enumerate(REFINE_SWEEP):
                override = {**base_override, **refine_override}
                train_spec = TorchDelayedLocalizerTrainingSpec(
                    **{
                        **asdict(base_train_spec),
                        **override,
                    }
                )
                model_spec = TorchRuntimeStudentModelSpec(
                    **{
                        **asdict(base_model_spec),
                        "dropout_prob": float(override.get("dropout_prob", 0.0)),
                    }
                )
                summary = _cross_validate_torch(
                    corpus_path=corpus_path,
                    train_spec=train_spec,
                    model_spec=model_spec,
                )
                summary["experiment_name"] = f"refine_{coarse_rank:02d}_{refine_idx:02d}"
                summary["override"] = override
                refine_results.append(summary)
        refine_results.sort(key=_rank_key)
        payload["stages"]["refine"] = refine_results
        (output_dir / "refine_results.json").write_text(
            json.dumps(refine_results, indent=2) + "\n",
            encoding="utf-8",
        )

    (output_dir / "experiment_index.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
