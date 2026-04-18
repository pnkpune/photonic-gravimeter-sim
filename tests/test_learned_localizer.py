from __future__ import annotations

from pathlib import Path

import numpy as np

from gravnav.datasets.bathymetry_loader import resolve_regional_demo_pack
from gravnav.estimators.gravity_sequence_match import GravitySequenceMatcherSpec
from gravnav.estimators.map_match_pf import apply_ned_offsets_to_geodetic
from gravnav.ml import (
    DelayedLocalizerTrainingSpec,
    LearnedLocalizerSpec,
    NeuralEarthSignatureLocalizer,
    RealOceanCorpusSpec,
    cross_validate_delayed_localizer,
    distill_runtime_student,
    evaluate_runtime_student,
    train_delayed_localizer,
    train_ocean_teacher,
)
from gravnav.ml.data import _make_state, _measurement_feature_vector, _tide_corrector_from_demo_pack, build_real_ocean_corpus
from gravnav.truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from gravnav.utils.config import load_config_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = (
    ROOT / "data/bathymetry/processed/norwegian_margin_maritime_priority9_emodnet_demo_pack.json"
)
SECOND_PACK = (
    ROOT / "data/bathymetry/processed/helgeland_offshore_priority9_emodnet_demo_pack.json"
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


def test_real_ocean_corpus_and_models_round_trip(tmp_path: Path) -> None:
    corpus = build_real_ocean_corpus(
        [DEFAULT_PACK],
        sequence_spec=_small_sequence_spec(),
        corpus_spec=RealOceanCorpusSpec(
            window_size=5,
            patch_size=5,
            patch_spacing_m=60.0,
            max_examples_per_region=6,
            random_seed=7,
        ),
    )
    teacher = train_ocean_teacher(corpus)
    student = distill_runtime_student(corpus, teacher)
    trained = train_delayed_localizer(
        corpus,
        student,
        spec=DelayedLocalizerTrainingSpec(
            epochs=40,
            learning_rate=0.05,
            l2=1.0e-4,
        ),
    )

    teacher_path = teacher.save_npz(tmp_path / "teacher.npz")
    model_path = trained.save_npz(tmp_path / "student.npz")

    assert teacher_path.exists()
    assert model_path.exists()
    assert teacher.reference_parameter_count > 0
    assert trained.reference_parameter_count > 0

    metrics = evaluate_runtime_student(corpus, trained)
    assert 0.0 <= float(metrics["top1_accuracy"]) <= 1.0
    assert float(metrics["median_horizontal_error_m"]) >= 0.0
    assert 0.0 <= float(metrics["publishability_brier_score"]) <= 1.0
    assert 0.0 <= float(metrics["support_expansion_brier_score"]) <= 1.0


def test_learned_localizer_emits_runtime_updates(tmp_path: Path) -> None:
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
            random_seed=11,
        ),
    )
    student = train_delayed_localizer(
        corpus,
        distill_runtime_student(corpus, train_ocean_teacher(corpus)),
        spec=DelayedLocalizerTrainingSpec(epochs=25, learning_rate=0.05),
    )
    model_path = student.save_npz(tmp_path / "runtime_bundle.npz")
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
    assert updates[-1].learned_covariance_scale is not None
    assert updates[-1].localizer_name == "learned_sequence_localizer"


def test_real_ocean_corpus_region_subset_and_cross_validation() -> None:
    sequence_spec = _small_sequence_spec()
    corpus = build_real_ocean_corpus(
        [DEFAULT_PACK, SECOND_PACK],
        sequence_spec=sequence_spec,
        corpus_spec=RealOceanCorpusSpec(
            window_size=5,
            patch_size=5,
            patch_spacing_m=60.0,
            max_examples_per_region=3,
            num_offset_realizations_per_region=1,
            random_seed=13,
        ),
    )

    norwegian_only = corpus.select_regions(
        include=("norwegian_margin_maritime_demo",),
        name="norwegian_only",
    )
    assert norwegian_only.num_examples > 0
    assert norwegian_only.region_names == ("norwegian_margin_maritime_demo",)

    cv = cross_validate_delayed_localizer(
        corpus,
        teacher_spec=None,
        train_spec=DelayedLocalizerTrainingSpec(
            epochs=5,
            learning_rate=0.05,
            l2=1.0e-4,
        ),
    )
    assert cv["num_folds"] == 2
    assert len(cv["folds"]) == 2
    held_out = {fold["held_out_region"] for fold in cv["folds"]}
    assert held_out == {
        "norwegian_margin_maritime_demo",
        "helgeland_offshore",
    }
    assert cv["aggregate"]["median_horizontal_error_m"] >= 0.0
    assert cv["aggregate"]["median_publishability_brier_score"] >= 0.0
    assert cv["aggregate"]["median_support_expansion_brier_score"] >= 0.0


def test_real_ocean_corpus_route_variants_expand_real_geography() -> None:
    corpus = build_real_ocean_corpus(
        [DEFAULT_PACK],
        sequence_spec=_small_sequence_spec(),
        corpus_spec=RealOceanCorpusSpec(
            window_size=5,
            patch_size=5,
            patch_spacing_m=60.0,
            max_examples_per_region=2,
            num_offset_realizations_per_region=1,
            num_route_variants_per_region=2,
            route_variant_max_attempts=32,
            route_variant_margin_m=150.0,
            route_variant_min_separation_m=500.0,
            random_seed=19,
        ),
    )
    assert corpus.num_examples >= 4
    assert corpus.region_names == ("norwegian_margin_maritime_demo",)
    assert (
        int(corpus.metadata["route_variant_counts"]["norwegian_margin_maritime_demo"]) >= 2
    )


def test_real_ocean_corpus_edge_biased_realizations_use_expanded_grid() -> None:
    sequence_spec = _small_sequence_spec()
    corpus = build_real_ocean_corpus(
        [DEFAULT_PACK],
        sequence_spec=sequence_spec,
        corpus_spec=RealOceanCorpusSpec(
            window_size=5,
            patch_size=5,
            patch_spacing_m=60.0,
            max_examples_per_region=2,
            num_offset_realizations_per_region=1,
            num_edge_biased_realizations_per_region=1,
            random_seed=23,
        ),
    )

    mode_counts = corpus.metadata["training_grid_mode_counts"]
    assert int(mode_counts["nominal"]) > 0
    assert int(mode_counts["expanded"]) > 0
    realization_counts = corpus.metadata["realization_mode_counts_by_region"][
        "norwegian_margin_maritime_demo"
    ]
    assert int(realization_counts["centered"]) == 1
    assert int(realization_counts["edge_biased"]) == 1
    assert corpus.support_expansion_labels.shape[0] == corpus.num_examples
    assert bool(np.any(corpus.support_expansion_labels))
    assert not bool(np.all(corpus.support_expansion_labels))
    assert float(np.mean(corpus.support_expansion_labels.astype(np.float64))) < 1.0
    assert (
        float(corpus.metadata["centered_clamp_fraction_of_nominal_half_span"]) > 0.0
    )
    assert float(np.max(np.abs(corpus.candidate_offsets_ned_m))) > float(
        max(sequence_spec.grid_half_span_m)
    )
