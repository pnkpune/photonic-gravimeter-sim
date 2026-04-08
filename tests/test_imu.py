from __future__ import annotations

import numpy as np

from gravnav.sensors.imu import (
    IMUSpec,
    build_interval_imu_truth_kinematics,
)
from gravnav.simulation.metrics import scenario_metrics_from_result
from gravnav.simulation.runner import ScenarioSimulationRunner, SimulationRunnerConfig
from gravnav.truth.scenarios import build_truth_trajectory_from_scenario, get_named_scenario


def test_interval_imu_truth_kinematics_matches_truth_increment() -> None:
    truth = build_truth_trajectory_from_scenario(
        get_named_scenario("maritime_baseline"),
        dt_s=2.0,
    )

    k = 350
    dt = float(truth.time_s[k] - truth.time_s[k - 1])
    kin = build_interval_imu_truth_kinematics(
        v_ned_prev_mps=truth.v_ned_mps[k - 1],
        v_ned_next_mps=truth.v_ned_mps[k],
        C_n_b_prev=truth.C_n_b[k - 1],
        C_n_b_next=truth.C_n_b[k],
        lat_prev_rad=float(truth.lat_rad[k - 1]),
        height_prev_m=float(truth.height_m[k - 1]),
        dt_s=dt,
    )

    assert kin.f_b_mps2.shape == (3,)
    assert kin.omega_ib_b_radps.shape == (3,)
    assert np.all(np.isfinite(kin.f_b_mps2))
    assert np.all(np.isfinite(kin.omega_ib_b_radps))


def test_runner_perfect_imu_reproduces_truth_without_aiding() -> None:
    scenario = get_named_scenario("maritime_baseline")
    cfg = SimulationRunnerConfig()
    cfg.velocity_aid.enabled = False
    cfg.depth_aid.enabled = False
    cfg.map_match.enabled = False

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

    pos = scenario_metrics_from_result(result).ins_position_error
    assert pos is not None
    assert pos.horizontal_rmse_m < 1.0e-3
    assert pos.vertical_rmse_m < 1.0e-3
