from __future__ import annotations

import numpy as np

from gravnav.estimators.error_state_ins import (
    ErrorStateINSNominalState,
    ErrorStateINSProcessNoise,
    ErrorStateINSState,
)
from gravnav.estimators.gravity_sequence_match import (
    GravitySequenceMatcher,
    GravitySequenceMatcherSpec,
)
from gravnav.estimators.map_match_pf import (
    apply_ned_offsets_to_geodetic,
    geodetic_offsets_to_local_ned,
)
from gravnav.physics.earth import meridian_radius, prime_vertical_radius
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.gravity_gradiometer import GravityGradiometerSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import (
    ins_position_error_metrics_from_result,
    lag_smoothed_position_error_metrics_from_result,
    sequence_position_error_metrics_from_result,
)
from gravnav.simulation.runner import (
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
    _lag_publish_position_covariance_geodetic,
)
from gravnav.truth.scenarios import build_truth_trajectory_from_scenario, get_named_scenario


def _quadratic_map_factory(lat_ref_rad: float, lon_ref_rad: float, height_ref_m: float):
    def map_fn(lat_rad, lon_rad, height_m):
        ned = geodetic_offsets_to_local_ned(
            lat_rad,
            lon_rad,
            height_m,
            lat_ref_rad=lat_ref_rad,
            lon_ref_rad=lon_ref_rad,
            height_ref_m=height_ref_m,
        )
        north = ned[:, 0]
        east = ned[:, 1]
        return (
            2.0e-8 * north
            - 1.5e-8 * east
            + 0.5 * 1.2e-10 * north**2
            + 0.5 * 8.0e-11 * east**2
        )

    return map_fn


def _make_state(
    *,
    time_s: float,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
) -> ErrorStateINSState:
    nominal = ErrorStateINSNominalState(
        time_s=time_s,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        height_m=height_m,
        v_ned_mps=np.zeros(3, dtype=np.float64),
        C_n_b=np.eye(3, dtype=np.float64),
        gyro_bias_radps=np.zeros(3, dtype=np.float64),
        accel_bias_mps2=np.zeros(3, dtype=np.float64),
    )
    return ErrorStateINSState(
        nominal=nominal,
        P=np.eye(15, dtype=np.float64),
        process_noise=ErrorStateINSProcessNoise.perfect(),
    )


def test_sequence_matcher_recovers_constant_ins_offset_on_quadratic_map() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 0.0
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    matcher = GravitySequenceMatcher(
        GravitySequenceMatcherSpec(
            window_size=5,
            grid_half_span_m=(80.0, 80.0),
            grid_spacing_m=(10.0, 10.0),
            transition_std_m=(8.0, 8.0),
            center_prior_std_m=(60.0, 60.0),
            gravity_meas_std_mps2=2.0e-7,
            height_std_m=1.0,
        ),
        map_fn,
    )

    truth_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 10.0, 0.0],
            [40.0, 20.0, 0.0],
            [60.0, 30.0, 0.0],
            [80.0, 40.0, 0.0],
            [100.0, 50.0, 0.0],
            [120.0, 60.0, 0.0],
        ],
        dtype=np.float64,
    )
    ins_bias = np.array([35.0, -25.0, 0.0], dtype=np.float64)

    outputs = []
    for k, offset in enumerate(truth_offsets):
        lat_true, lon_true, h_true = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            offset.reshape(1, 3),
        )
        lat_ins, lon_ins, h_ins = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            (offset + ins_bias).reshape(1, 3),
        )
        g_meas = float(
            map_fn(
                np.array([lat_true[0]], dtype=np.float64),
                np.array([lon_true[0]], dtype=np.float64),
                np.array([h_true[0]], dtype=np.float64),
            )[0]
        )

        ready_results = matcher.update(
            g_meas,
            gravity_meas_std_mps2=2.0e-7,
            ins_or_state=_make_state(
                time_s=float(k),
                lat_rad=float(lat_ins[0]),
                lon_rad=float(lon_ins[0]),
                height_m=float(h_ins[0]),
            ),
            time_s=float(k),
        )
        outputs.extend(ready_results)

    outputs.extend(matcher.finalize())

    assert len(outputs) == len(truth_offsets)

    horizontal_errors = []
    for k, result in enumerate(outputs):
        lat_true, lon_true, h_true = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            truth_offsets[k].reshape(1, 3),
        )
        err_ned = geodetic_offsets_to_local_ned(
            np.array([result.estimate.lat_rad], dtype=np.float64),
            np.array([result.estimate.lon_rad], dtype=np.float64),
            np.array([result.estimate.height_m], dtype=np.float64),
            lat_ref_rad=float(lat_true[0]),
            lon_ref_rad=float(lon_true[0]),
            height_ref_m=float(h_true[0]),
        )[0]
        horizontal_errors.append(float(np.linalg.norm(err_ned[:2])))

    assert float(np.mean(horizontal_errors)) < 15.0
    assert outputs[0].delayed_by_steps >= 0
    assert outputs[-1].window_size_used <= matcher.spec.window_size


def test_sequence_matcher_emits_deterministic_anchor_estimates() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 0.0
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    matcher = GravitySequenceMatcher(
        GravitySequenceMatcherSpec(
            window_size=7,
            grid_half_span_m=(80.0, 80.0),
            grid_spacing_m=(10.0, 10.0),
            transition_std_m=(8.0, 8.0),
            center_prior_std_m=(60.0, 60.0),
            gravity_meas_std_mps2=2.0e-7,
            height_std_m=1.0,
        ),
        map_fn,
    )

    truth_offsets = np.array(
        [[20.0 * k, 8.0 * k, 0.0] for k in range(9)],
        dtype=np.float64,
    )
    ins_bias = np.array([30.0, -18.0, 0.0], dtype=np.float64)
    outputs = []
    for k, offset in enumerate(truth_offsets):
        lat_true, lon_true, h_true = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            offset.reshape(1, 3),
        )
        lat_ins, lon_ins, h_ins = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            (offset + ins_bias).reshape(1, 3),
        )
        g_meas = float(
            map_fn(
                np.array([lat_true[0]], dtype=np.float64),
                np.array([lon_true[0]], dtype=np.float64),
                np.array([h_true[0]], dtype=np.float64),
            )[0]
        )
        outputs.extend(
            matcher.update(
                g_meas,
                gravity_meas_std_mps2=2.0e-7,
                ins_or_state=_make_state(
                    time_s=float(k),
                    lat_rad=float(lat_ins[0]),
                    lon_rad=float(lon_ins[0]),
                    height_m=float(h_ins[0]),
                ),
                time_s=float(k),
            )
        )
    outputs.extend(matcher.finalize())

    mid_result = outputs[4]
    assert len(mid_result.anchor_estimates) == 3
    anchor_times = [float(anchor.time_s) for anchor in mid_result.anchor_estimates]
    assert anchor_times == sorted(anchor_times)
    assert any(abs(float(mid_result.time_s) - t) < 1.0e-12 for t in anchor_times)
    assert all(anchor.covariance_ned_m2.shape == (3, 3) for anchor in mid_result.anchor_estimates)
    assert all(anchor.global_index >= 0 for anchor in mid_result.anchor_estimates)


def test_sequence_matcher_collapsed_posterior_falls_back_without_crashing() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 0.0
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    matcher = GravitySequenceMatcher(
        GravitySequenceMatcherSpec(
            window_size=3,
            grid_half_span_m=(40.0, 40.0),
            grid_spacing_m=(20.0, 20.0),
            transition_std_m=(8.0, 8.0),
            center_prior_std_m=(30.0, 30.0),
            gravity_meas_std_mps2=2.0e-7,
            height_std_m=1.0,
        ),
        map_fn,
    )

    obs = matcher._build_observation(
        measured_disturbance_mps2=0.0,
        gravity_meas_std_mps2=2.0e-7,
        ins_or_state=_make_state(
            time_s=0.0,
            lat_rad=lat0,
            lon_rad=lon0,
            height_m=h0,
        ),
        measured_gradient_per_s2=None,
        gradient_meas_std_per_s2=None,
        measured_bathymetry_m=None,
        bathymetry_meas_std_m=None,
        depth_measurement=None,
        reference_surface_height_m=0.0,
        time_s=0.0,
    )
    window = [obs]
    n = obs.candidate_offsets_ned_m.shape[0]
    alpha = np.full((1, n), np.nan, dtype=np.float64)
    beta = np.full((1, n), np.nan, dtype=np.float64)
    delta = np.zeros((1, n), dtype=np.float64)
    psi = np.zeros((1, n), dtype=np.int64)

    anchor, used_gradient, used_bathymetry, pred_g_mean, pred_g_std, pred_bath = matcher._estimate_for_window_index(
        window,
        alpha,
        beta,
        delta,
        psi,
        0,
    )

    assert np.isfinite(anchor.lat_rad)
    assert np.isfinite(anchor.lon_rad)
    assert anchor.covariance_ned_m2.shape == (3, 3)
    assert used_gradient is False
    assert used_bathymetry is False
    assert np.isfinite(pred_g_mean)
    assert np.isfinite(pred_g_std)
    assert pred_bath is None


def test_runner_sequence_matcher_logs_updates_and_metrics() -> None:
    scenario = get_named_scenario("maritime_baseline")
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=5.0)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    cfg = SimulationRunnerConfig()
    cfg.map_match.matcher = "sequence"
    cfg.map_match.sequence_spec = GravitySequenceMatcherSpec(
        window_size=7,
        grid_half_span_m=(100.0, 100.0),
        grid_spacing_m=(20.0, 20.0),
        transition_std_m=(15.0, 15.0),
        center_prior_std_m=(60.0, 60.0),
        gravity_meas_std_mps2=1.0e-6,
        height_std_m=1.0,
    )
    cfg.map_match.gravity_meas_std_mps2 = 1.0e-6
    cfg.map_match.depth_meas_std_m = 0.1
    cfg.observability.enabled = False

    runner = ScenarioSimulationRunner(cfg)
    result = runner.run_with_specs(
        scenario_or_truth=truth,
        imu_spec=IMUSpec.perfect(),
        gravimeter_spec=GravimeterSpec.perfect_relative(),
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        map_model=map_fn,
        dt_s=5.0,
        seed=777,
    )

    assert len(result.estimators.pf_updates) == 0
    assert len(result.estimators.sequence_updates) > 0

    arrays = result.estimators.sequence_history_arrays()
    assert arrays["sequence_time_s"].shape[0] == len(result.estimators.sequence_updates)
    assert arrays["sequence_covariance_ned_m2"].shape[1:] == (3, 3)

    summary = result.summary()
    assert summary.num_sequence_updates == len(result.estimators.sequence_updates)

    metrics = sequence_position_error_metrics_from_result(result)
    assert metrics is not None
    assert np.isfinite(metrics.horizontal_rmse_m)
    assert metrics.horizontal_rmse_m >= 0.0


def test_runner_sequence_feedback_path_executes_and_logs() -> None:
    scenario = get_named_scenario("maritime_baseline")
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=5.0)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    cfg = SimulationRunnerConfig()
    cfg.map_match.matcher = "sequence"
    cfg.map_match.use_gradiometer = True
    cfg.map_match.use_sequence_feedback = True
    cfg.map_match.sequence_spec = GravitySequenceMatcherSpec(
        window_size=7,
        grid_half_span_m=(100.0, 100.0),
        grid_spacing_m=(20.0, 20.0),
        transition_std_m=(15.0, 15.0),
        center_prior_std_m=(60.0, 60.0),
        gravity_meas_std_mps2=1.0e-6,
        gradient_meas_std_per_s2=1.0e-8,
        height_std_m=1.0,
    )
    cfg.map_match.sequence_feedback_spec.min_peak_probability = 0.05
    cfg.map_match.sequence_feedback_spec.mode = "lag_replay"
    cfg.map_match.sequence_feedback_spec.measurement_geometry = (
        "directional_horizontal"
    )
    cfg.map_match.sequence_feedback_spec.min_horizontal_eigenvalue_ratio = 1.0
    cfg.map_match.sequence_feedback_spec.max_horizontal_std_m = 100.0
    cfg.map_match.sequence_feedback_spec.max_correction_norm_m = 200.0
    cfg.map_match.gravity_meas_std_mps2 = 1.0e-6
    cfg.map_match.gradient_meas_std_per_s2 = 1.0e-8
    cfg.map_match.depth_meas_std_m = 0.1
    cfg.observability.enabled = False

    runner = ScenarioSimulationRunner(cfg)
    result = runner.run_with_specs(
        scenario_or_truth=truth,
        imu_spec=IMUSpec.perfect(),
        gravimeter_spec=GravimeterSpec.perfect_relative(),
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        map_model=map_fn,
        dt_s=5.0,
        seed=777,
        gradiometer_spec=None,
    )

    rows = result.estimators.custom_streams.get("sequence_feedback")
    assert rows is not None
    assert len(rows) > 0
    first = rows[0]
    assert first["mode"] == "lag_replay"
    assert first["measurement_geometry"] == "directional_horizontal"
    assert "feedback_allowed" in first
    assert "horizontal_offset_ned_m" in first
    assert "horizontal_std_m" in first
    assert "horizontal_eigenvalue_ratio" in first
    assert "projected_correction_m" in first
    assert "projected_std_m" in first
    assert "constrained_direction_ned" in first
    assert "target_step_index" in first
    assert "replayed_steps" in first
    assert "matcher_reset" in first


def test_runner_sequence_lag_smoother_improves_output_without_mutating_live_ins() -> None:
    scenario = get_named_scenario("maritime_baseline")
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=5.0)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    def _run(use_lag_smoother: bool):
        cfg = SimulationRunnerConfig()
        cfg.map_match.matcher = "sequence"
        cfg.map_match.use_gradiometer = True
        cfg.map_match.use_sequence_lag_smoother = use_lag_smoother
        cfg.map_match.sequence_spec = GravitySequenceMatcherSpec(
            window_size=7,
            grid_half_span_m=(120.0, 120.0),
            grid_spacing_m=(20.0, 20.0),
            transition_std_m=(18.0, 18.0),
            center_prior_std_m=(70.0, 70.0),
            gravity_meas_std_mps2=5.0e-7,
            gradient_meas_std_per_s2=5.0e-9,
            height_std_m=1.0,
        )
        cfg.map_match.gravity_meas_std_mps2 = 5.0e-7
        cfg.map_match.gradient_meas_std_per_s2 = 5.0e-9
        cfg.map_match.depth_meas_std_m = 0.1
        cfg.map_match.schedule.every_steps = 2
        cfg.map_match.sequence_lag_smoother_spec.measurement_geometry = "directional_horizontal"
        cfg.map_match.sequence_lag_smoother_spec.min_peak_probability = 0.02
        cfg.map_match.sequence_lag_smoother_spec.min_horizontal_eigenvalue_ratio = 1.0
        cfg.map_match.sequence_lag_smoother_spec.max_horizontal_std_m = 160.0
        cfg.map_match.sequence_lag_smoother_spec.max_correction_norm_m = 80.0
        cfg.map_match.sequence_lag_smoother_spec.covariance_inflation = 6.0
        cfg.map_match.sequence_lag_smoother_spec.max_anchor_count = 3
        cfg.observability.enabled = False

        runner = ScenarioSimulationRunner(cfg)
        return runner.run_with_specs(
            scenario_or_truth=truth,
            imu_spec=IMUSpec(
                gyro_fixed_bias_radps=(4.0e-4, -2.0e-4, 3.0e-4),
                accel_fixed_bias_mps2=(1.5e-3, -1.0e-3, 8.0e-4),
                name="biased_test_imu",
            ),
            gravimeter_spec=GravimeterSpec.perfect_relative(),
            depth_spec=DepthSensorSpec.perfect(),
            velocity_aid_spec=VelocityAidSpec.perfect(),
            gradiometer_spec=GravityGradiometerSpec(
                noise_density_per_s2_per_sqrt_hz=0.0,
                bias_random_walk_per_s2_per_sqrt_s=0.0,
                turn_on_bias_std_per_s2=0.0,
                fixed_bias_per_s2=0.0,
            ),
            map_model=map_fn,
            dt_s=5.0,
            seed=222,
        )

    baseline = _run(False)
    smoothed = _run(True)

    baseline_ins = baseline.estimators.ins_history_arrays()
    smoothed_ins = smoothed.estimators.ins_history_arrays()
    assert np.allclose(baseline_ins["ins_lat_rad"], smoothed_ins["ins_lat_rad"])
    assert np.allclose(baseline_ins["ins_lon_rad"], smoothed_ins["ins_lon_rad"])
    assert np.allclose(baseline_ins["ins_height_m"], smoothed_ins["ins_height_m"])

    lag_metrics = lag_smoothed_position_error_metrics_from_result(smoothed)
    live_ins_metrics = ins_position_error_metrics_from_result(baseline)
    assert lag_metrics is not None
    assert live_ins_metrics is not None
    assert len(smoothed.estimators.lag_smoothed_states) > 0
    publish_rows = smoothed.estimators.custom_streams.get("sequence_lag_smoothed_publish")
    assert publish_rows is not None
    assert sum(row["publish_source"] == "sequence_update" for row in publish_rows) > 10
    assert lag_metrics.horizontal_rmse_m < live_ins_metrics.horizontal_rmse_m
    assert lag_metrics.cep95_m <= live_ins_metrics.cep95_m


def test_lag_publish_covariance_envelopes_replay_and_sequence_uncertainty() -> None:
    lat0 = np.deg2rad(63.0)
    lon0 = np.deg2rad(10.0)
    state = _make_state(
        time_s=0.0,
        lat_rad=lat0,
        lon_rad=lon0,
        height_m=0.0,
    )
    # Roughly 20 m / 15 m / 2 m 1-sigma in local NED.
    replay_cov_geo = np.diag(
        [
            (20.0 / float(meridian_radius(lat0))) ** 2,
            (15.0 / (float(prime_vertical_radius(lat0)) * np.cos(lat0))) ** 2,
            2.0**2,
        ]
    )
    state.P[:3, :3] = replay_cov_geo

    # A sharper sequence covariance that should not become the published
    # integrity bound on its own.
    seq_cov_geo = np.diag(
        [
            (4.0 / float(meridian_radius(lat0))) ** 2,
            (3.0 / (float(prime_vertical_radius(lat0)) * np.cos(lat0))) ** 2,
            1.0**2,
        ]
    )

    bound_geo = _lag_publish_position_covariance_geodetic(
        state,
        published_lat_rad=lat0,
        published_height_m=0.0,
        sequence_covariance_geodetic=seq_cov_geo,
    )

    def _geo_to_ned(P_geo: np.ndarray) -> np.ndarray:
        rm = float(meridian_radius(lat0))
        rn = float(prime_vertical_radius(lat0)) * np.cos(lat0)
        J = np.diag([rm, rn, -1.0])
        return J @ P_geo @ J.T

    bound_ned = _geo_to_ned(bound_geo)
    replay_ned = _geo_to_ned(replay_cov_geo)
    seq_ned = _geo_to_ned(seq_cov_geo)

    assert np.all(np.linalg.eigvalsh(bound_ned - replay_ned) >= -1.0e-9)
    assert np.all(np.linalg.eigvalsh(bound_ned - seq_ned) >= -1.0e-9)
