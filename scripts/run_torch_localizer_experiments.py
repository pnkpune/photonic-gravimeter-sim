#!/usr/bin/env python3
"""
Run the canonical Torch learned-localizer benchmark loop.
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
    aggregate_runtime_student_folds,
    DEFAULT_FIXED_PUBLISHABILITY_THRESHOLD,
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


def _split_override_for_specs(
    override: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_override = dict(override)
    model_override: dict[str, Any] = {}
    if "dropout_prob" in train_override:
        model_override["dropout_prob"] = float(train_override.pop("dropout_prob"))
    return train_override, model_override


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


def _write_corpus_summary(
    *,
    corpus: Any,
    corpus_path: Path,
    output_dir: Path,
    region_set: str,
    corpus_preset: str,
    experiment_config: Path,
    manifest_paths: list[Path],
    missing_manifest_paths: list[Path],
) -> None:
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
        "experiment_config": str(experiment_config),
        "region_set": str(region_set),
        "corpus_preset": str(corpus_preset),
        "region_example_counts": region_example_counts,
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
    manifest_paths = list(resolved.manifest_paths)
    missing_manifest_paths = list(resolved.missing_manifest_paths)
    corpus = build_real_ocean_corpus(
        manifest_paths,
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
    _write_corpus_summary(
        corpus=corpus,
        corpus_path=corpus_path,
        output_dir=output_dir,
        region_set=region_set,
        corpus_preset=corpus_preset,
        experiment_config=experiment_config,
        manifest_paths=manifest_paths,
        missing_manifest_paths=missing_manifest_paths,
    )
    return corpus_path


def _cross_validate_torch(
    *,
    corpus_path: Path,
    train_spec: TorchDelayedLocalizerTrainingSpec,
    model_spec: TorchRuntimeStudentModelSpec,
    publishability_reporting_mode: str,
    fixed_publishability_threshold: float,
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
        metrics = evaluate_runtime_student(
            eval_corpus,
            model,
            publishability_reporting_mode=str(publishability_reporting_mode),
            fixed_publishability_threshold=float(fixed_publishability_threshold),
        )
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
    return {
        "num_folds": int(len(folds)),
        "folds": folds,
        "aggregate": aggregate_runtime_student_folds(folds),
        "train_spec": asdict(train_spec),
        "model_spec": asdict(model_spec),
        "publishability_reporting_mode": str(publishability_reporting_mode),
        "fixed_publishability_threshold": float(fixed_publishability_threshold),
        "corpus_path": str(corpus_path),
    }


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    agg = dict(result["aggregate"])
    views = dict(agg.get("publishability_views", {}))
    fixed_positive = float(
        dict(views.get("fixed_040", {})).get("median_positive_fraction", 0.0)
    )
    calibrated_positive = float(
        dict(views.get("calibrated", {})).get(
            "median_positive_fraction",
            agg.get("median_publishability_positive_fraction", 0.0),
        )
    )
    return (
        float(agg["median_horizontal_error_m"]),
        -float(agg["median_top1_accuracy"]),
        float(agg["median_publishability_brier_score"]),
        -int(fixed_positive > 0.0 and calibrated_positive > 0.0),
    )


def _fold_error_by_region(summary: dict[str, Any]) -> dict[str, float]:
    return {
        str(fold["held_out_region"]): float(fold["median_horizontal_error_m"])
        for fold in summary.get("folds", [])
    }


def _has_degenerate_fold(
    candidate: dict[str, Any],
    baseline_by_region: dict[str, float],
) -> bool:
    for fold in candidate.get("folds", []):
        region_name = str(fold["held_out_region"])
        error = float(fold["median_horizontal_error_m"])
        if not np.isfinite(error):
            return True
        baseline_error = float(baseline_by_region.get(region_name, error))
        allowed_error = max(2.5 * baseline_error, baseline_error + 250.0)
        if error > allowed_error:
            return True
    return False


def _is_meaningful_win(
    candidate: dict[str, Any],
    *,
    baseline: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    baseline_agg = dict(baseline["aggregate"])
    candidate_agg = dict(candidate["aggregate"])
    baseline_error = float(baseline_agg["median_horizontal_error_m"])
    baseline_top1 = float(baseline_agg["median_top1_accuracy"])
    candidate_error = float(candidate_agg["median_horizontal_error_m"])
    candidate_top1 = float(candidate_agg["median_top1_accuracy"])
    error_improvement_fraction = (
        0.0
        if baseline_error <= 0.0
        else float((baseline_error - candidate_error) / baseline_error)
    )
    top1_improvement = float(candidate_top1 - baseline_top1)
    views = dict(candidate_agg.get("publishability_views", {}))
    fixed_positive = float(
        dict(views.get("fixed_040", {})).get("median_positive_fraction", 0.0)
    )
    calibrated_positive = float(
        dict(views.get("calibrated", {})).get(
            "median_positive_fraction",
            candidate_agg.get("median_publishability_positive_fraction", 0.0),
        )
    )
    no_degenerate_fold = not _has_degenerate_fold(
        candidate,
        baseline_by_region=_fold_error_by_region(baseline),
    )
    decision = {
        "error_improvement_fraction": error_improvement_fraction,
        "top1_improvement_absolute": top1_improvement,
        "fixed_040_positive_fraction": fixed_positive,
        "calibrated_positive_fraction": calibrated_positive,
        "no_degenerate_fold": bool(no_degenerate_fold),
    }
    is_win = (
        error_improvement_fraction >= 0.15
        and top1_improvement >= 0.05
        and fixed_positive > 0.0
        and calibrated_positive > 0.0
        and no_degenerate_fold
    )
    return bool(is_win), decision


def _render_comparison_table(
    *,
    norway4_baseline: dict[str, Any] | None,
    main_baseline: dict[str, Any] | None,
    best_coarse: dict[str, Any] | None,
) -> str:
    rows = []
    labels_and_results = [
        ("norway4_baseline", norway4_baseline),
        ("all_wave1_baseline", main_baseline),
        ("best_coarse", best_coarse),
    ]
    for label, result in labels_and_results:
        if result is None:
            continue
        agg = dict(result["aggregate"])
        views = dict(agg.get("publishability_views", {}))
        calibrated = dict(views.get("calibrated", {}))
        fixed = dict(views.get("fixed_040", {}))
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{float(agg['median_horizontal_error_m']):.3f}",
                    f"{float(agg['median_top1_accuracy']):.3f}",
                    f"{float(agg['median_publishability_brier_score']):.3f}",
                    f"{float(calibrated.get('median_positive_fraction', agg.get('median_publishability_positive_fraction', 0.0))):.3f}",
                    f"{float(calibrated.get('median_precision', agg.get('median_publishability_precision', 0.0))):.3f}",
                    f"{float(calibrated.get('median_recall', agg.get('median_publishability_recall', 0.0))):.3f}",
                    f"{float(fixed.get('median_positive_fraction', 0.0)):.3f}",
                    f"{float(fixed.get('median_precision', 0.0)):.3f}",
                    f"{float(fixed.get('median_recall', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    header = [
        "| run | median_error_m | median_top1 | publish_brier | calibrated_pos_frac | calibrated_precision | calibrated_recall | fixed040_pos_frac | fixed040_precision | fixed040_recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    return "\n".join(header + rows) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _checkpoint_ranked_results(path: Path, results: list[dict[str, Any]]) -> None:
    ranked = sorted(results, key=_rank_key)
    _write_json(path, ranked)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical baseline, coarse, and refine Torch benchmark loop."
    )
    parser.add_argument(
        "--stage",
        choices=("baseline", "coarse", "refine", "all"),
        default="all",
    )
    parser.add_argument("--region-set", default="all_wave1")
    parser.add_argument("--corpus-preset", default="dev")
    parser.add_argument(
        "--comparison-baseline-region-set",
        default="norway4",
    )
    parser.add_argument(
        "--comparison-baseline-corpus-preset",
        default="dev",
    )
    parser.add_argument(
        "--refine-corpus-preset",
        default="full",
    )
    parser.add_argument("--experiment-config", default=str(PROJECT_ROOT / "configs/ml/real_ocean_experiments.json"))
    parser.add_argument("--sequence-profile", default=str(DEFAULT_SEQUENCE_PROFILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--corpus-path", default="")
    parser.add_argument("--refine-corpus-path", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--publishability-reporting-mode",
        choices=("single", "dual"),
        default="dual",
    )
    parser.add_argument(
        "--fixed-publishability-threshold",
        type=float,
        default=DEFAULT_FIXED_PUBLISHABILITY_THRESHOLD,
    )
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
            output_dir=output_dir / f"{args.region_set}_{args.corpus_preset}_corpus",
            sequence_profile=sequence_profile,
            experiment_config=experiment_config,
            seed=int(args.seed),
        )
    )
    refine_corpus_path: Path | None = (
        Path(args.refine_corpus_path).expanduser().resolve()
        if str(args.refine_corpus_path).strip()
        else None
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
        "refine_corpus_preset": str(args.refine_corpus_preset),
        "refine_corpus_path": None if refine_corpus_path is None else str(refine_corpus_path),
        "publishability_reporting_mode": str(args.publishability_reporting_mode),
        "fixed_publishability_threshold": float(args.fixed_publishability_threshold),
        "stages": {},
    }
    stage = str(args.stage)

    comparison_baseline: dict[str, Any] | None = None
    if stage in {"baseline", "all"} and str(args.comparison_baseline_region_set).strip():
        comparison_corpus_path = _build_corpus(
            region_set=str(args.comparison_baseline_region_set),
            corpus_preset=str(args.comparison_baseline_corpus_preset),
            output_dir=output_dir
            / f"{args.comparison_baseline_region_set}_{args.comparison_baseline_corpus_preset}_corpus",
            sequence_profile=sequence_profile,
            experiment_config=experiment_config,
            seed=int(args.seed),
        )
        comparison_corpus = load_real_ocean_corpus(comparison_corpus_path)
        comparison_model_spec = TorchRuntimeStudentModelSpec(
            candidate_feature_names=tuple(
                comparison_corpus.metadata["candidate_feature_names"]
            ),
            query_feature_names=tuple(
                comparison_corpus.metadata["query_feature_names"]
            ),
        )
        comparison_baseline = _cross_validate_torch(
            corpus_path=comparison_corpus_path,
            train_spec=base_train_spec,
            model_spec=comparison_model_spec,
            publishability_reporting_mode=str(args.publishability_reporting_mode),
            fixed_publishability_threshold=float(args.fixed_publishability_threshold),
        )
        payload["stages"]["norway4_baseline"] = comparison_baseline
        _write_json(
            output_dir / f"{args.comparison_baseline_region_set}_baseline_summary.json",
            comparison_baseline,
        )

    main_baseline: dict[str, Any] | None = None
    if stage in {"baseline", "all"}:
        main_baseline = _cross_validate_torch(
            corpus_path=corpus_path,
            train_spec=base_train_spec,
            model_spec=base_model_spec,
            publishability_reporting_mode=str(args.publishability_reporting_mode),
            fixed_publishability_threshold=float(args.fixed_publishability_threshold),
        )
        payload["stages"]["all_wave1_baseline"] = main_baseline
        _write_json(output_dir / f"{args.region_set}_baseline_summary.json", main_baseline)
        _write_json(output_dir / "baseline_summary.json", main_baseline)

    coarse_results: list[dict[str, Any]] = []
    if stage in {"coarse", "refine", "all"}:
        coarse_results_path = output_dir / "coarse_results.json"
        for idx, override in enumerate(COARSE_SWEEP):
            train_override, model_override = _split_override_for_specs(override)
            train_spec = TorchDelayedLocalizerTrainingSpec(
                **{
                    **asdict(base_train_spec),
                    **train_override,
                }
            )
            model_spec = TorchRuntimeStudentModelSpec(
                **{
                    **asdict(base_model_spec),
                    **model_override,
                }
            )
            summary = _cross_validate_torch(
                corpus_path=corpus_path,
                train_spec=train_spec,
                model_spec=model_spec,
                publishability_reporting_mode=str(args.publishability_reporting_mode),
                fixed_publishability_threshold=float(args.fixed_publishability_threshold),
            )
            summary["experiment_name"] = f"coarse_{idx:02d}"
            summary["override"] = dict(override)
            coarse_results.append(summary)
            _checkpoint_ranked_results(coarse_results_path, coarse_results)
        coarse_results.sort(key=_rank_key)
        payload["stages"]["coarse"] = coarse_results
        _write_json(coarse_results_path, coarse_results)

    decision_payload: dict[str, Any] = {}
    eligible_coarse_results: list[dict[str, Any]] = []
    if stage in {"refine", "all"}:
        if len(coarse_results) == 0:
            coarse_results = json.loads(
                (output_dir / "coarse_results.json").read_text(encoding="utf-8")
            )
        if main_baseline is None:
            main_baseline = json.loads(
                (output_dir / f"{args.region_set}_baseline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        for coarse_result in coarse_results:
            is_win, decision = _is_meaningful_win(
                coarse_result,
                baseline=main_baseline,
            )
            coarse_result["meaningful_win"] = bool(is_win)
            coarse_result["meaningful_win_decision"] = decision
            if is_win:
                eligible_coarse_results.append(coarse_result)
        decision_payload = {
            "eligible_coarse_count": int(len(eligible_coarse_results)),
            "stopped_after_dev_gate": bool(len(eligible_coarse_results) == 0),
        }
        payload["decision"] = decision_payload
        _write_json(output_dir / "coarse_results.json", coarse_results)

        refine_results: list[dict[str, Any]] = []
        refine_results_path = output_dir / "refine_results.json"
        if len(eligible_coarse_results) > 0:
            if refine_corpus_path is None:
                refine_corpus_path = _build_corpus(
                    region_set=str(args.region_set),
                    corpus_preset=str(args.refine_corpus_preset),
                    output_dir=output_dir
                    / f"{args.region_set}_{args.refine_corpus_preset}_corpus",
                    sequence_profile=sequence_profile,
                    experiment_config=experiment_config,
                    seed=int(args.seed),
                )
                payload["refine_corpus_path"] = str(refine_corpus_path)
            top = eligible_coarse_results[: max(int(args.top_k), 1)]
            refine_corpus = load_real_ocean_corpus(refine_corpus_path)
            refine_model_spec_base = TorchRuntimeStudentModelSpec(
                candidate_feature_names=tuple(
                    refine_corpus.metadata["candidate_feature_names"]
                ),
                query_feature_names=tuple(
                    refine_corpus.metadata["query_feature_names"]
                ),
            )
            for coarse_rank, coarse_result in enumerate(top):
                base_override = dict(coarse_result.get("override", {}))
                for refine_idx, refine_override in enumerate(REFINE_SWEEP):
                    override = {**base_override, **refine_override}
                    train_override, model_override = _split_override_for_specs(
                        override
                    )
                    train_spec = TorchDelayedLocalizerTrainingSpec(
                        **{
                            **asdict(base_train_spec),
                            **train_override,
                        }
                    )
                    model_spec = TorchRuntimeStudentModelSpec(
                        **{
                            **asdict(refine_model_spec_base),
                            **model_override,
                        }
                    )
                    summary = _cross_validate_torch(
                        corpus_path=refine_corpus_path,
                        train_spec=train_spec,
                        model_spec=model_spec,
                        publishability_reporting_mode=str(
                            args.publishability_reporting_mode
                        ),
                        fixed_publishability_threshold=float(
                            args.fixed_publishability_threshold
                        ),
                    )
                    summary["experiment_name"] = (
                        f"refine_{coarse_rank:02d}_{refine_idx:02d}"
                    )
                    summary["override"] = override
                    summary["promoted_from"] = str(coarse_result["experiment_name"])
                    refine_results.append(summary)
                    _checkpoint_ranked_results(refine_results_path, refine_results)
            refine_results.sort(key=_rank_key)
        payload["stages"]["refine"] = refine_results
        _write_json(refine_results_path, refine_results)

    comparison_table = _render_comparison_table(
        norway4_baseline=comparison_baseline,
        main_baseline=main_baseline
        if main_baseline is not None
        else (
            json.loads(
                (output_dir / f"{args.region_set}_baseline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            if (output_dir / f"{args.region_set}_baseline_summary.json").exists()
            else None
        ),
        best_coarse=coarse_results[0] if len(coarse_results) > 0 else None,
    )
    (output_dir / "benchmark_comparison.md").write_text(
        comparison_table,
        encoding="utf-8",
    )

    _write_json(output_dir / "experiment_index.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
