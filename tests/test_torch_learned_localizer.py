from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gravnav.datasets.bathymetry_loader import resolve_regional_demo_pack
from gravnav.estimators.gravity_sequence_match import GravitySequenceMatcherSpec
from gravnav.estimators.map_match_pf import apply_ned_offsets_to_geodetic
from gravnav.ml import LearnedLocalizerSpec, NeuralEarthSignatureLocalizer, RealOceanCorpusSpec
from gravnav.ml.data import _make_state, _measurement_feature_vector, _tide_corrector_from_demo_pack, build_real_ocean_corpus
from gravnav.ml.torch_models import (
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
    model_path = model.save_pt(tmp_path / "runtime_torch_bundle.pt")
    loaded = load_runtime_localizer_model(model_path)

    scores = loaded.predict_candidate_scores(
        query_windows=corpus.query_windows[:2],
        candidate_features=corpus.candidate_features[:2],
        candidate_offsets_ned_m=corpus.candidate_offsets_ned_m[:2],
        analytic_log_emission=corpus.analytic_log_emission[:2],
    )
    assert scores.shape == (2, corpus.candidate_features.shape[1])

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
