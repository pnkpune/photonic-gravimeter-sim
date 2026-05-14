from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from gravnav.estimators.error_state_ins import (
    ErrorStateINSState,
    ErrorStateINSNominalState,
    ErrorStateINSProcessNoise,
)
from gravnav.estimators.gravity_sequence_match import (
    GravitySequenceMatcherSpec,
    SequenceAmbiguityDiagnostics,
    SequenceCandidateHypothesis,
    SequenceMatchEstimate,
    SequenceMatchUpdateResult,
)
from gravnav.sensors.imu import IMUMeasurement
from gravnav.simulation.runner import ScenarioSimulationRunner
from gravnav.truth.trajectory import TruthTrajectory


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "build_sequence_feedback_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "build_sequence_feedback_corpus",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_truth() -> TruthTrajectory:
    time_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    zeros3 = np.zeros((3, 3), dtype=np.float64)
    C_n_b = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0)
    return TruthTrajectory(
        time_s=time_s,
        lat_rad=np.full(3, np.deg2rad(63.0), dtype=np.float64),
        lon_rad=np.full(3, np.deg2rad(10.0), dtype=np.float64),
        height_m=np.zeros(3, dtype=np.float64),
        v_ned_mps=zeros3.copy(),
        v_dot_ned_mps2=zeros3.copy(),
        C_n_b=C_n_b,
        omega_nb_b_radps=zeros3.copy(),
    )


def _make_states(truth: TruthTrajectory) -> list[ErrorStateINSState]:
    states: list[ErrorStateINSState] = []
    for idx, time_s in enumerate(truth.time_s.tolist()):
        nominal = ErrorStateINSNominalState(
            time_s=float(time_s),
            lat_rad=float(truth.lat_rad[idx]),
            lon_rad=float(truth.lon_rad[idx]),
            height_m=float(truth.height_m[idx]),
            v_ned_mps=np.zeros(3, dtype=np.float64),
            C_n_b=np.eye(3, dtype=np.float64),
            gyro_bias_radps=np.zeros(3, dtype=np.float64),
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
        )
        P = np.eye(15, dtype=np.float64)
        P[:3, :3] *= 1.0e-2
        states.append(
            ErrorStateINSState(
                nominal=nominal,
                P=P,
                process_noise=ErrorStateINSProcessNoise.perfect(),
            )
        )
    return states


def _make_imu_samples() -> list[IMUMeasurement]:
    zeros = np.zeros(3, dtype=np.float64)
    return [
        IMUMeasurement(
            time_s=float(idx),
            omega_ib_b_radps=zeros.copy(),
            f_ib_b_mps2=zeros.copy(),
            ideal_omega_ib_b_radps=zeros.copy(),
            ideal_f_ib_b_mps2=zeros.copy(),
            gyro_bias_used_radps=zeros.copy(),
            accel_bias_used_mps2=zeros.copy(),
            gyro_white_noise_radps=zeros.copy(),
            accel_white_noise_mps2=zeros.copy(),
            gyro_saturated=False,
            accel_saturated=False,
        )
        for idx in range(2)
    ]


def _make_update() -> SequenceMatchUpdateResult:
    ambiguity = SequenceAmbiguityDiagnostics(
        posterior_candidate_ess=12.0,
        posterior_candidate_ess_fraction=0.25,
        edge_mass_fraction=0.08,
        support_radius_n_m=22.0,
        support_radius_e_m=18.0,
        horizontal_covariance_eigenvalue_ratio=2.5,
        grid_saturated_north=False,
        grid_saturated_east=False,
        grid_saturated_any=False,
        gravity_predicted_spread_mps2=4.0e-6,
        gravity_information_ratio=2.1,
        bathymetry_predicted_spread_m=5.0,
        bathymetry_information_ratio=1.6,
        magnetic_predicted_spread_nt=12.0,
        magnetic_information_ratio=1.4,
        dominant_failure_mode="informative",
        grid_mode="expanded",
        grid_half_span_m=np.array([60.0, 60.0], dtype=np.float64),
        grid_spacing_m=np.array([10.0, 10.0], dtype=np.float64),
    )
    estimate = SequenceMatchEstimate(
        lat_rad=np.deg2rad(63.0001),
        lon_rad=np.deg2rad(10.0001),
        height_m=0.0,
        covariance_ned_m2=np.diag([36.0, 81.0, 4.0]).astype(np.float64),
        covariance_geodetic=np.diag([1.0e-12, 1.0e-12, 4.0]).astype(np.float64),
        predicted_disturbance_mps2=1.0e-5,
        marginal_peak_probability=0.55,
        predicted_bathymetry_m=100.0,
        predicted_magnetic_total_nt=45000.0,
    )
    return SequenceMatchUpdateResult(
        estimate=estimate,
        global_index=0,
        time_s=0.0,
        window_size_used=7,
        delayed_by_steps=1,
        num_candidates=49,
        posterior_entropy_nats=1.2,
        marginal_peak_probability=0.55,
        predicted_disturbance_mean_mps2=1.0e-5,
        predicted_disturbance_std_mps2=4.0e-6,
        used_gradient=True,
        used_bathymetry=True,
        used_magnetics=True,
        viterbi_log_score=-2.0,
        viterbi_offset_ned_m=np.array([10.0, -8.0, 0.0], dtype=np.float64),
        posterior_mean_offset_ned_m=np.array([8.0, -4.0, 0.0], dtype=np.float64),
        ambiguity_diagnostics=ambiguity,
        candidate_hypotheses=(
            SequenceCandidateHypothesis(
                rank=0,
                candidate_index=3,
                marginal_probability=0.44,
                probability_gap_to_best=0.0,
                lat_rad=np.deg2rad(63.00012),
                lon_rad=np.deg2rad(10.00002),
                height_m=0.0,
                offset_ned_m=np.array([4.0, -2.0, 0.0], dtype=np.float64),
                predicted_disturbance_mps2=1.1e-5,
                predicted_bathymetry_m=101.0,
                predicted_magnetic_total_nt=45010.0,
            ),
        ),
        publishability_probability=0.72,
        support_expansion_probability=0.33,
        learned_covariance_scale=0.9,
    )


def _collect_examples(module, runner, cached_run: dict[str, object]) -> dict[str, list[object]]:
    outputs = {
        "features": [],
        "region_names": [],
        "event_seed": [],
        "current_time_s": [],
        "update_time_s": [],
        "current_step_index": [],
        "target_step_index": [],
        "lag_replay_applied": [],
        "lag_replay_improves_error": [],
        "lag_replay_hmi_safe": [],
        "lag_replay_useful_and_safe": [],
        "lag_replay_error_delta_m": [],
        "lag_replay_best_gain_alpha": [],
        "bias_transfer_applied": [],
        "bias_transfer_improves_error": [],
        "bias_transfer_hmi_safe": [],
        "bias_transfer_useful_and_safe": [],
        "bias_transfer_error_delta_m": [],
        "bias_transfer_best_gain_alpha": [],
    }
    counts = module._append_region_seed_examples(
        runner=runner,
        pack=None,
        region_name="synthetic_region",
        cached_run=cached_run,
        minimum_useful_improvement_m=10.0,
        sequence_feedback_geometry="directional_horizontal",
        sequence_feedback_inflation=6.0,
        sequence_feedback_transfer_rw_std_mps=0.6,
        sequence_feedback_nis_threshold=None,
        gain_alpha_candidates=[0.1, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0],
        **outputs,
    )
    outputs["counts"] = counts
    return outputs


def test_feedback_event_cache_roundtrip_and_reuse(tmp_path: Path) -> None:
    module = _load_module()
    truth = _make_truth()
    ins_states = _make_states(truth)
    imu_samples = _make_imu_samples()
    depth_by_step = [None, None, None]
    velocity_by_step = [None, None, None]
    update = _make_update()
    event_rows = module._build_cached_event_rows(
        truth=truth,
        ins_states=ins_states,
        sequence_updates=[update],
        max_events_per_region_seed=4,
        horizon_s=1.0,
        seed=42,
    )
    sequence_spec = GravitySequenceMatcherSpec(window_size=7)
    runner = ScenarioSimulationRunner(
        module._build_runner_config(
            sequence_spec=sequence_spec,
            use_bathymetry=False,
            use_magnetics=False,
            use_gradiometer=True,
            use_current_correction=False,
        )
    )
    depth_variance_m2 = float(runner.config.depth_aid.measurement_std_m**2)
    velocity_std = np.asarray(
        runner.config.velocity_aid.measurement_std_mps,
        dtype=np.float64,
    ).reshape(3)
    velocity_R = np.diag(velocity_std**2)
    cached_run = {
        "truth": truth,
        "ins_states": ins_states,
        "imu_samples": imu_samples,
        "depth_measurements_by_step": depth_by_step,
        "velocity_measurements_by_step": velocity_by_step,
        "depth_variance_m2": depth_variance_m2,
        "velocity_R": velocity_R,
        "event_rows": event_rows,
        "metadata": {"region_name": "synthetic_region", "seed": 42},
    }

    cache_path = tmp_path / "synthetic_cache.npz"
    module.save_feedback_event_cache(
        cache_path,
        truth=truth,
        ins_states=ins_states,
        imu_samples=imu_samples,
        depth_measurements_by_step=depth_by_step,
        velocity_measurements_by_step=velocity_by_step,
        depth_variance_m2=depth_variance_m2,
        velocity_R=velocity_R,
        event_rows=event_rows,
        metadata=cached_run["metadata"],
    )
    loaded = module.load_feedback_event_cache(cache_path)

    assert loaded["metadata"]["region_name"] == "synthetic_region"
    assert len(loaded["event_rows"]) == 1
    assert np.allclose(loaded["truth"].time_s, truth.time_s)
    assert np.allclose(loaded["velocity_R"], velocity_R)

    direct_outputs = _collect_examples(module, runner, cached_run)
    cached_outputs = _collect_examples(module, runner, loaded)

    assert direct_outputs["counts"] == cached_outputs["counts"]
    assert np.allclose(direct_outputs["features"][0], cached_outputs["features"][0])
    assert direct_outputs["event_seed"] == cached_outputs["event_seed"]
    assert direct_outputs["lag_replay_applied"] == cached_outputs["lag_replay_applied"]
    assert direct_outputs["lag_replay_best_gain_alpha"] == cached_outputs["lag_replay_best_gain_alpha"]
    assert direct_outputs["bias_transfer_best_gain_alpha"] == cached_outputs["bias_transfer_best_gain_alpha"]
    assert loaded["event_rows"][0]["update"]["candidate_hypotheses"][0]["rank"] == 0
