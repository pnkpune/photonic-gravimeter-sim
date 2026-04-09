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
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import sequence_position_error_metrics_from_result
from gravnav.simulation.runner import ScenarioSimulationRunner, SimulationRunnerConfig
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
    assert "feedback_allowed" in first
    assert "horizontal_offset_ned_m" in first
    assert "horizontal_std_m" in first
    assert "target_step_index" in first
    assert "replayed_steps" in first
    assert "matcher_reset" in first
