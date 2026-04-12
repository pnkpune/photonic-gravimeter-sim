from __future__ import annotations

import numpy as np

from gravnav.estimators.error_state_ins import (
    ErrorStateINS,
    ErrorStateINSProcessNoise,
)
from gravnav.estimators.fusion import apply_depth_measurement_height_only
from gravnav.estimators.fusion import apply_velocity_ned_measurement_velocity_only
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.imu import IMUSensor, IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import scenario_metrics_from_result
from gravnav.simulation.runner import ScenarioSimulationRunner, SimulationRunnerConfig
from gravnav.truth.scenarios import build_truth_trajectory_from_scenario, get_named_scenario


def test_depth_height_only_update_preserves_horizontal_state() -> None:
    truth = build_truth_trajectory_from_scenario(
        get_named_scenario("maritime_baseline"),
        dt_s=2.0,
    )
    ins = ErrorStateINS.from_truth_trajectory_start(
        truth,
        process_noise=ErrorStateINSProcessNoise.perfect(),
    )

    lat_before = float(ins.nominal.lat_rad + 1.0e-4)
    lon_before = float(ins.nominal.lon_rad - 2.0e-4)
    height_before = float(ins.nominal.height_m + 12.0)

    ins.state.nominal.lat_rad = lat_before
    ins.state.nominal.lon_rad = lon_before
    ins.state.nominal.height_m = height_before

    # Seed some height cross-covariance to prove the constrained update does not
    # inject horizontal motion through the covariance structure.
    ins.state.P[0, 2] = 1.0e-3
    ins.state.P[2, 0] = 1.0e-3
    ins.state.P[1, 2] = -2.0e-3
    ins.state.P[2, 1] = -2.0e-3
    ins.state.P[2, 2] = 25.0

    result = apply_depth_measurement_height_only(
        ins,
        measured_depth_m=0.0,
        depth_variance_m2=0.25,
        reference_surface_height_m=0.0,
    )

    assert result.accepted
    assert np.isclose(ins.nominal.lat_rad, lat_before)
    assert np.isclose(ins.nominal.lon_rad, lon_before)
    assert abs(ins.nominal.height_m) < abs(height_before)


def test_runner_default_depth_path_stays_stable_with_perfect_velocity_aid() -> None:
    scenario = get_named_scenario("maritime_baseline")

    cfg = SimulationRunnerConfig()
    cfg.map_match.enabled = False

    result = ScenarioSimulationRunner(cfg).run_with_specs(
        scenario,
        imu_spec=IMUSpec.perfect(),
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        gravimeter_spec=None,
        map_model=None,
        seed=123,
        dt_s=2.0,
    )

    pos = scenario_metrics_from_result(result).ins_position_error
    assert pos is not None
    assert pos.horizontal_rmse_m < 100.0
    assert pos.vertical_rmse_m < 200.0


def test_velocity_only_update_preserves_bias_states() -> None:
    truth = build_truth_trajectory_from_scenario(
        get_named_scenario("maritime_baseline"),
        dt_s=2.0,
    )
    ins = ErrorStateINS.from_truth_trajectory_start(
        truth,
        process_noise=ErrorStateINSProcessNoise.perfect(),
    )

    ins.state.nominal.gyro_bias_radps[:] = np.array([1.0, 2.0, 3.0]) * 1.0e-3
    ins.state.nominal.accel_bias_mps2[:] = np.array([1.0, 2.0, 3.0]) * 1.0e-2
    ins.state.P[3:6, 3:6] = np.diag([4.0, 4.0, 4.0])
    ins.state.P[9:12, 3:6] = 1.0
    ins.state.P[3:6, 9:12] = 1.0
    ins.state.P[12:15, 3:6] = 1.0
    ins.state.P[3:6, 12:15] = 1.0

    gyro_before = ins.nominal.gyro_bias_radps.copy()
    accel_before = ins.nominal.accel_bias_mps2.copy()

    result = apply_velocity_ned_measurement_velocity_only(
        ins,
        measured_velocity_ned_mps=np.array([5.0, -1.0, 0.5]),
        R_mps2=np.diag([0.1, 0.1, 0.1]),
    )

    assert result.accepted
    assert np.allclose(ins.nominal.gyro_bias_radps, gyro_before)
    assert np.allclose(ins.nominal.accel_bias_mps2, accel_before)


def test_runner_uses_sensor_turn_on_bias_for_default_bias_prior() -> None:
    truth = build_truth_trajectory_from_scenario(
        get_named_scenario("maritime_baseline"),
        dt_s=2.0,
    )
    imu_spec = IMUSpec(
        gyro_turn_on_bias_std_radps=(2.0e-5, 3.0e-5, 4.0e-5),
        accel_turn_on_bias_std_mps2=(1.0e-4, 2.0e-4, 3.0e-4),
        name="test_imu_with_turn_on_bias",
    )
    runner = ScenarioSimulationRunner(SimulationRunnerConfig())

    ins = runner._build_initial_ins(truth, IMUSensor(imu_spec))

    assert np.all(np.diag(ins.covariance)[9:12] > 0.0)
    assert np.all(np.diag(ins.covariance)[12:15] > 0.0)


def test_runner_nav_grade_with_perfect_aids_stays_bounded() -> None:
    scenario = get_named_scenario("maritime_baseline")
    imu_spec = IMUSpec(
        name="imu_nav_grade",
        gyro_noise_density_radps_per_sqrt_hz=[8.0e-05, 8.0e-05, 8.0e-05],
        accel_noise_density_mps2_per_sqrt_hz=[2.5e-04, 2.5e-04, 2.5e-04],
        gyro_bias_random_walk_radps_per_sqrt_s=[5.0e-07, 5.0e-07, 5.0e-07],
        accel_bias_random_walk_mps2_per_sqrt_s=[2.5e-06, 2.5e-06, 2.5e-06],
        gyro_turn_on_bias_std_radps=[2.0e-05, 2.0e-05, 2.0e-05],
        accel_turn_on_bias_std_mps2=[1.5e-04, 1.5e-04, 1.5e-04],
    )

    cfg = SimulationRunnerConfig()
    cfg.map_match.enabled = False

    result = ScenarioSimulationRunner(cfg).run_with_specs(
        scenario,
        imu_spec=imu_spec,
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        gravimeter_spec=None,
        map_model=None,
        seed=123,
        dt_s=2.0,
    )

    pos = scenario_metrics_from_result(result).ins_position_error
    assert pos is not None
    assert pos.horizontal_rmse_m < 200.0
    assert pos.vertical_rmse_m < 1.0


def test_runner_initial_position_offset_perturbs_nominal_start() -> None:
    scenario = get_named_scenario("maritime_baseline")
    cfg = SimulationRunnerConfig()
    cfg.map_match.enabled = False
    cfg.initial_position_offset_ned_m = (75.0, -40.0, 5.0)

    result = ScenarioSimulationRunner(cfg).run_with_specs(
        scenario,
        imu_spec=IMUSpec.perfect(),
        gravimeter_spec=None,
        depth_spec=None,
        velocity_aid_spec=None,
        map_model=None,
        seed=123,
        dt_s=2.0,
    )

    first_state = result.estimators.ins_states[0].nominal
    truth = result.truth
    assert abs(float(first_state.lat_rad) - float(truth.lat_rad[0])) > 0.0
    assert abs(float(first_state.lon_rad) - float(truth.lon_rad[0])) > 0.0
    assert np.isclose(float(first_state.height_m), float(truth.height_m[0] - 5.0))


def test_optional_gravimeter_does_not_change_live_ins_when_map_match_disabled() -> None:
    scenario = get_named_scenario("maritime_baseline")
    cfg = SimulationRunnerConfig()
    cfg.map_match.enabled = False

    common_kwargs = dict(
        scenario_or_truth=scenario,
        imu_spec=IMUSpec(),
        depth_spec=DepthSensorSpec(),
        velocity_aid_spec=VelocityAidSpec(),
        map_model=None,
        seed=321,
        dt_s=2.0,
    )

    runner = ScenarioSimulationRunner(cfg)
    baseline = runner.run_with_specs(
        gravimeter_spec=None,
        **common_kwargs,
    )
    with_gravimeter = runner.run_with_specs(
        gravimeter_spec=GravimeterSpec(),
        **common_kwargs,
    )

    baseline_hist = baseline.estimators.ins_history_arrays()
    gravimeter_hist = with_gravimeter.estimators.ins_history_arrays()
    for key in (
        "ins_lat_rad",
        "ins_lon_rad",
        "ins_height_m",
        "ins_v_ned_mps",
    ):
        assert np.allclose(baseline_hist[key], gravimeter_hist[key])
