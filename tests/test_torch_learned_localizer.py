from __future__ import annotations

import json
from pathlib import Path
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gravnav.datasets.bathymetry_loader import resolve_regional_demo_pack
from gravnav.estimators.gravity_sequence_match import GravitySequenceMatcherSpec
from gravnav.estimators.map_match_pf import apply_ned_offsets_to_geodetic
from gravnav.ml import (
    aggregate_runtime_student_folds,
    evaluate_runtime_student,
    LearnedLocalizerSpec,
    NeuralEarthSignatureLocalizer,
    RealOceanCorpusSpec,
)
from gravnav.ml.data import _make_state, _measurement_feature_vector, _tide_corrector_from_demo_pack, build_real_ocean_corpus
from gravnav.ml.torch_models import (
    _compute_example_weights,
    _select_binary_threshold,
    TorchDelayedLocalizerTrainingSpec,
    load_runtime_localizer_model,
    train_torch_delayed_localizer,
)
from gravnav.truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from gravnav.utils.config import load_config_mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = (
    ROOT / "data/bathymetry/processed/norwegian_margin_maritime_priority9_emodnet_demo_pack.json"
)


def _small_sequence_spec() -> GravitySequenceMatcherSpec:
    return GravitySequenceMatcherSpec(
        window_size=5,
        grid_half_span_m=(80.0, 80.0),
        grid_spacing_m=(20.0, 20.0),
        transition_std_m=(20.0, 20.0),
        center_prior_std_m=(70.0, 70.0),
        gravity_meas_std_mps2=2.0e-5,
        gradient_meas_std_per_s2=5.0e-8,
        bathymetry_meas_std_m=12.0,
        bathymetry_gradient_meas_std_m_per_m=0.02,
        bathymetry_rugosity_meas_std_m=2.0,
        magnetic_meas_std_nt=8.0,
        magnetic_gradient_meas_std_nt_per_m=0.05,
        adaptive_grid_enabled=True,
    )


def _load_run_experiments_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_torch_localizer_experiments.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_torch_localizer_experiments_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_torch_model_save_load_and_runtime_updates(tmp_path: Path) -> None:
    pack = resolve_regional_demo_pack(DEFAULT_PACK)
    sequence_spec = _small_sequence_spec()
    corpus = build_real_ocean_corpus(
        [DEFAULT_PACK],
        sequence_spec=sequence_spec,
        corpus_spec=RealOceanCorpusSpec(
            window_size=5,
            patch_size=5,
            patch_spacing_m=60.0,
            max_examples_per_region=6,
            num_offset_realizations_per_region=1,
            random_seed=21,
        ),
    )
    model = train_torch_delayed_localizer(
        corpus,
        spec=TorchDelayedLocalizerTrainingSpec(
            epochs=4,
            batch_size=4,
            validation_fraction=0.34,
            min_epochs=2,
            early_stopping_patience=2,
            label_smoothing=0.05,
            head_epochs=2,
            device="cpu",
        ),
    )
    assert int(model.metadata["train_examples"]) > 0
    assert int(model.metadata["validation_examples"]) > 0
    assert int(model.metadata["best_epoch"]) >= 0
    assert float(model.metadata["region_balance_power"]) == 1.0
    assert float(model.metadata["label_balance_power"]) == 1.0
    assert bool(model.metadata["calibrate_reliability_threshold"]) is True
    assert float(model.metadata["reliability_threshold_calibrated"]) <= 0.65
    dual_metrics = evaluate_runtime_student(
        corpus,
        model,
        publishability_reporting_mode="dual",
        fixed_publishability_threshold=0.40,
    )
    assert set(dual_metrics["publishability_views"]) == {"calibrated", "fixed_040"}
    assert float(dual_metrics["publishability_views"]["fixed_040"]["threshold"]) == pytest.approx(0.40)
    model_path = model.save_pt(tmp_path / "runtime_torch_bundle.pt")
    loaded = load_runtime_localizer_model(model_path)

    scores = loaded.predict_candidate_scores(
        query_windows=corpus.query_windows[:2],
        candidate_features=corpus.candidate_features[:2],
        candidate_offsets_ned_m=corpus.candidate_offsets_ned_m[:2],
        analytic_log_emission=corpus.analytic_log_emission[:2],
    )
    assert scores.shape == (2, corpus.candidate_features.shape[1])
    support_prob = loaded.support_expansion_probability_from_features(
        np.zeros((2, 8), dtype=np.float64)
    )
    assert support_prob.shape == (2,)

    localizer = NeuralEarthSignatureLocalizer(
        LearnedLocalizerSpec(
            model_export_path=model_path,
            sequence_spec=sequence_spec,
        ),
        pack.gravity_map,
        bathymetry_map=pack.bathymetry_grid,
        magnetic_map=pack.magnetic_grid,
    )

    tide_corrector = _tide_corrector_from_demo_pack(pack)
    scenario = ScenarioSpec.from_mapping(load_config_mapping(pack.scenario_path))
    truth = build_truth_trajectory_from_scenario(scenario)

    updates = []
    for k in range(sequence_spec.window_size + 1):
        lat_true = float(truth.lat_rad[k])
        lon_true = float(truth.lon_rad[k])
        h_true = float(truth.height_m[k])
        t_now = float(truth.time_s[k])
        v_now = np.asarray(truth.v_ned_mps[k], dtype=np.float64)
        prior_lat, prior_lon, prior_h = apply_ned_offsets_to_geodetic(
            np.array([lat_true], dtype=np.float64),
            np.array([lon_true], dtype=np.float64),
            np.array([h_true], dtype=np.float64),
            np.array([[40.0, -30.0, 0.0]], dtype=np.float64),
        )
        measurement = _measurement_feature_vector(
            pack,
            lat_rad=lat_true,
            lon_rad=lon_true,
            height_m=h_true,
            reference_surface_height_m=0.0,
            time_s=t_now,
            tide_corrector=tide_corrector,
        )
        state = _make_state(
            time_s=t_now,
            lat_rad=float(prior_lat[0]),
            lon_rad=float(prior_lon[0]),
            height_m=float(prior_h[0]),
            v_ned_mps=v_now,
        )
        updates.extend(
            localizer.update(
                float(measurement[0]),
                ins_or_state=state,
                current_track_unit_ned=np.array(
                    [
                        v_now[0] / max(float(np.linalg.norm(v_now[:2])), 1.0e-9),
                        v_now[1] / max(float(np.linalg.norm(v_now[:2])), 1.0e-9),
                        0.0,
                    ],
                    dtype=np.float64,
                ),
                measured_gradient_per_s2=measurement[1:3],
                measured_bathymetry_m=(
                    None if not np.isfinite(measurement[3]) else float(measurement[3])
                ),
                measured_bathymetry_gradient_m_per_m=(
                    None if not np.isfinite(measurement[4]) else float(measurement[4])
                ),
                measured_bathymetry_rugosity_m=(
                    None if not np.isfinite(measurement[6]) else float(measurement[6])
                ),
                measured_magnetic_total_nt=(
                    None if not np.isfinite(measurement[7]) else float(measurement[7])
                ),
                measured_magnetic_gradient_nt_per_m=(
                    None if not np.isfinite(measurement[8]) else float(measurement[8])
                ),
                reference_surface_height_m=0.0,
                time_s=t_now,
            )
        )
    updates.extend(localizer.finalize())

    assert len(updates) >= sequence_spec.window_size
    assert updates[-1].publishability_probability is not None
    assert updates[-1].support_expansion_probability is not None
    assert updates[-1].learned_covariance_scale is not None


def test_torch_example_weights_upweight_minority_regions_and_labels() -> None:
    from gravnav.ml.data import RealOceanCorpus

    imbalanced = RealOceanCorpus(
        spec=RealOceanCorpusSpec(),
        patch_tensors=np.zeros((3, 1, 1, 1), dtype=np.float64),
        patch_summary_features=np.zeros((3, 1), dtype=np.float64),
        query_windows=np.zeros((3, 1, 1), dtype=np.float64),
        candidate_features=np.zeros((3, 1, 1), dtype=np.float64),
        candidate_offsets_ned_m=np.zeros((3, 1, 3), dtype=np.float64),
        analytic_log_emission=np.zeros((3, 1), dtype=np.float64),
        labels=np.zeros(3, dtype=np.int64),
        truth_offsets_ned_m=np.zeros((3, 3), dtype=np.float64),
        publishability_labels=np.zeros(3, dtype=bool),
        support_expansion_labels=np.zeros(3, dtype=bool),
        covariance_targets=np.ones(3, dtype=np.float64),
        region_names=("region_a", "region_b"),
        region_index=np.asarray([0, 0, 1], dtype=np.int64),
        metadata={},
    )

    weights = _compute_example_weights(
        imbalanced,
        region_balance_power=1.0,
        label_balance_power=0.0,
    )

    assert weights.shape == (imbalanced.num_examples,)
    assert float(np.mean(weights)) == pytest.approx(1.0)
    assert float(np.max(weights)) > float(np.min(weights))


def test_select_binary_threshold_improves_over_high_default_cutoff() -> None:
    probabilities = np.asarray([0.39, 0.41, 0.42, 0.44, 0.46], dtype=np.float64)
    labels = np.asarray([False, True, True, True, False], dtype=bool)

    threshold, metadata = _select_binary_threshold(
        probabilities,
        labels,
        default_threshold=0.65,
    )

    assert float(threshold) < 0.65
    assert metadata["source"] == "validation_balanced_accuracy"
    assert float(metadata["balanced_accuracy"]) >= 0.5
    assert float(metadata["positive_fraction"]) > 0.0


def test_aggregate_runtime_student_folds_reports_dual_publishability_views() -> None:
    folds = [
        {
            "top1_accuracy": 0.20,
            "median_horizontal_error_m": 100.0,
            "p90_horizontal_error_m": 180.0,
            "mean_publishability_probability": 0.45,
            "mean_support_expansion_probability": 0.30,
            "publishability_positive_fraction": 0.50,
            "publishability_brier_score": 0.25,
            "publishability_precision": 0.80,
            "publishability_recall": 0.60,
            "publishability_f1": 0.6857142857,
            "support_expansion_positive_fraction": 0.20,
            "support_expansion_brier_score": 0.22,
            "support_expansion_precision": 0.40,
            "support_expansion_recall": 0.50,
            "support_expansion_f1": 0.4444444444,
            "publishability_views": {
                "calibrated": {
                    "threshold": 0.44,
                    "positive_fraction": 0.50,
                    "precision": 0.80,
                    "recall": 0.60,
                    "f1": 0.6857142857,
                },
                "fixed_040": {
                    "threshold": 0.40,
                    "positive_fraction": 0.70,
                    "precision": 0.72,
                    "recall": 0.82,
                    "f1": 0.7667532468,
                },
            },
        },
        {
            "top1_accuracy": 0.30,
            "median_horizontal_error_m": 120.0,
            "p90_horizontal_error_m": 220.0,
            "mean_publishability_probability": 0.40,
            "mean_support_expansion_probability": 0.35,
            "publishability_positive_fraction": 0.40,
            "publishability_brier_score": 0.27,
            "publishability_precision": 0.90,
            "publishability_recall": 0.50,
            "publishability_f1": 0.6428571429,
            "support_expansion_positive_fraction": 0.30,
            "support_expansion_brier_score": 0.24,
            "support_expansion_precision": 0.45,
            "support_expansion_recall": 0.55,
            "support_expansion_f1": 0.495,
            "publishability_views": {
                "calibrated": {
                    "threshold": 0.46,
                    "positive_fraction": 0.40,
                    "precision": 0.90,
                    "recall": 0.50,
                    "f1": 0.6428571429,
                },
                "fixed_040": {
                    "threshold": 0.40,
                    "positive_fraction": 0.80,
                    "precision": 0.68,
                    "recall": 0.90,
                    "f1": 0.7746835443,
                },
            },
        },
    ]

    aggregate = aggregate_runtime_student_folds(folds)

    assert "publishability_views" in aggregate
    assert set(aggregate["publishability_views"]) == {"calibrated", "fixed_040"}
    assert float(
        aggregate["publishability_views"]["fixed_040"]["median_positive_fraction"]
    ) == pytest.approx(0.75)


def test_run_torch_rank_key_prefers_nonzero_publishability_under_both_views() -> None:
    module = _load_run_experiments_module()
    better = {
        "aggregate": {
            "median_horizontal_error_m": 100.0,
            "median_top1_accuracy": 0.25,
            "median_publishability_brier_score": 0.20,
            "publishability_views": {
                "calibrated": {"median_positive_fraction": 0.10},
                "fixed_040": {"median_positive_fraction": 0.20},
            },
        }
    }
    worse = {
        "aggregate": {
            "median_horizontal_error_m": 100.0,
            "median_top1_accuracy": 0.25,
            "median_publishability_brier_score": 0.20,
            "publishability_views": {
                "calibrated": {"median_positive_fraction": 0.10},
                "fixed_040": {"median_positive_fraction": 0.0},
            },
        }
    }

    assert module._rank_key(better) < module._rank_key(worse)


def test_run_torch_checkpoint_ranked_results_sorts_before_writing(
    tmp_path: Path,
) -> None:
    module = _load_run_experiments_module()
    output_path = tmp_path / "coarse_results.json"
    results = [
        {
            "experiment_name": "coarse_bad",
            "aggregate": {
                "median_horizontal_error_m": 120.0,
                "median_top1_accuracy": 0.20,
                "median_publishability_brier_score": 0.21,
                "publishability_views": {
                    "calibrated": {"median_positive_fraction": 0.10},
                    "fixed_040": {"median_positive_fraction": 0.10},
                },
            },
        },
        {
            "experiment_name": "coarse_good",
            "aggregate": {
                "median_horizontal_error_m": 90.0,
                "median_top1_accuracy": 0.25,
                "median_publishability_brier_score": 0.20,
                "publishability_views": {
                    "calibrated": {"median_positive_fraction": 0.10},
                    "fixed_040": {"median_positive_fraction": 0.10},
                },
            },
        },
    ]

    module._checkpoint_ranked_results(output_path, results)

    written = json.loads(output_path.read_text())
    assert [item["experiment_name"] for item in written] == [
        "coarse_good",
        "coarse_bad",
    ]
