from __future__ import annotations

import numpy as np

from gravnav.analysis.observability import (
    ObservabilityAnalyzer,
    gravity_map_gradient_ned,
)
from gravnav.estimators.feedback_policy import DirectionalFeedbackSpec
from gravnav.estimators.map_match_pf import (
    MapMatchPFSpec,
    apply_ned_offsets_to_geodetic,
    geodetic_offsets_to_local_ned,
)
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.runner import ScenarioSimulationRunner, SimulationRunnerConfig
from gravnav.truth.scenarios import build_truth_trajectory_from_scenario, get_named_scenario


def _affine_map_factory(
    lat_ref_rad: float,
    lon_ref_rad: float,
    height_ref_m: float,
    gradient_ned_mps2_per_m: np.ndarray,
):
    gradient_ned_mps2_per_m = np.asarray(gradient_ned_mps2_per_m, dtype=np.float64).reshape(3)

    def map_fn(lat_rad, lon_rad, height_m):
        ned = geodetic_offsets_to_local_ned(
            lat_rad,
            lon_rad,
            height_m,
            lat_ref_rad=lat_ref_rad,
            lon_ref_rad=lon_ref_rad,
            height_ref_m=height_ref_m,
        )
        return ned @ gradient_ned_mps2_per_m

    return map_fn


def _quadratic_map_factory(
    lat_ref_rad: float,
    lon_ref_rad: float,
    height_ref_m: float,
):
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
            0.5 * 1.0e-10 * north**2
            + 0.5 * 8.0e-11 * east**2
            + 2.0e-11 * north * east
        )

    return map_fn


def test_gravity_map_gradient_ned_matches_affine_truth() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 15.0
    expected_grad = np.array([2.5e-8, -1.5e-8, 4.0e-9], dtype=np.float64)
    map_fn = _affine_map_factory(lat0, lon0, h0, expected_grad)

    lat_t, lon_t, h_t = apply_ned_offsets_to_geodetic(
        np.array([lat0], dtype=np.float64),
        np.array([lon0], dtype=np.float64),
        np.array([h0], dtype=np.float64),
        np.array([[120.0, -80.0, 12.0]], dtype=np.float64),
    )

    grad = gravity_map_gradient_ned(
        map_fn,
        float(lat_t[0]),
        float(lon_t[0]),
        float(h_t[0]),
        delta_north_m=25.0,
        delta_east_m=25.0,
        delta_down_m=4.0,
    )

    assert np.allclose(grad, expected_grad, rtol=0.0, atol=5.0e-13)


def test_observability_analyzer_reaches_rank_two_on_rotating_gradient_map() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 0.0
    map_fn = _quadratic_map_factory(lat0, lon0, h0)
    analyzer = ObservabilityAnalyzer(
        map_fn,
        window_size=8,
        gravity_noise_std_mps2=1.0e-6,
        min_rank_for_feedback=2,
        min_gradient_norm=1.0e-12,
    )

    offsets = np.array(
        [
            [150.0, 0.0, 0.0],
            [0.0, 180.0, 0.0],
            [140.0, 120.0, 0.0],
            [-100.0, 160.0, 0.0],
        ],
        dtype=np.float64,
    )
    lat, lon, h = apply_ned_offsets_to_geodetic(
        np.full(offsets.shape[0], lat0, dtype=np.float64),
        np.full(offsets.shape[0], lon0, dtype=np.float64),
        np.full(offsets.shape[0], h0, dtype=np.float64),
        offsets,
    )

    snapshot = None
    for k in range(offsets.shape[0]):
        snapshot = analyzer.update(
            lat_rad=float(lat[k]),
            lon_rad=float(lon[k]),
            height_m=float(h[k]),
            time_s=float(k),
        )

    assert snapshot is not None
    assert snapshot.observable_rank >= 2
    assert snapshot.feedback_recommended
    assert np.isfinite(snapshot.information_density)
    assert snapshot.gradient_norm_horizontal > 0.0


def test_runner_logs_observability_stream() -> None:
    scenario = get_named_scenario("maritime_baseline")
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=5.0)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    cfg = SimulationRunnerConfig()
    cfg.map_match.pf_spec = MapMatchPFSpec(
        num_particles=64,
        init_position_std_m=(40.0, 40.0, 2.0),
        process_position_rw_std_m_per_sqrt_s=(0.0, 0.0, 0.0),
        rejuvenation_std_m=(0.0, 0.0, 0.0),
        use_ins_position_prior=False,
    )
    cfg.map_match.gravity_meas_std_mps2 = 1.0e-6
    cfg.map_match.depth_meas_std_m = 0.1
    cfg.observability.window_size = 12
    cfg.observability.min_rank_for_feedback = 1
    cfg.observability.min_gradient_norm = 1.0e-14

    runner = ScenarioSimulationRunner(cfg)
    result = runner.run_with_specs(
        scenario_or_truth=truth,
        imu_spec=IMUSpec.perfect(),
        gravimeter_spec=GravimeterSpec.perfect_relative(),
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        map_model=map_fn,
        dt_s=5.0,
        seed=321,
    )

    rows = result.estimators.custom_streams.get("observability")
    assert rows is not None
    assert len(rows) == len(result.estimators.pf_updates)
    assert len(rows) > 0

    first = rows[0]
    assert "gradient_ned_mps2_per_m" in first
    assert "gramian_eigenvalues" in first
    assert "observable_rank" in first
    assert "feedback_recommended" in first
    assert len(first["gradient_ned_mps2_per_m"]) == 3
    assert len(first["gramian_eigenvalues"]) == 3


def test_observability_gate_blocks_directional_feedback_when_rank_is_too_low() -> None:
    scenario = get_named_scenario("maritime_baseline")
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=5.0)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    cfg = SimulationRunnerConfig()
    cfg.map_match.use_directional_feedback = True
    cfg.map_match.pf_spec = MapMatchPFSpec(
        num_particles=64,
        init_position_std_m=(40.0, 40.0, 2.0),
        process_position_rw_std_m_per_sqrt_s=(0.0, 0.0, 0.0),
        rejuvenation_std_m=(0.0, 0.0, 0.0),
        use_ins_position_prior=False,
    )
    cfg.map_match.gravity_meas_std_mps2 = 1.0e-6
    cfg.map_match.depth_meas_std_m = 0.1
    cfg.map_match.directional_feedback_spec = DirectionalFeedbackSpec(
        horizontal_only=True,
        min_eigenvalue_ratio=1.0,
        max_ess_fraction=1.0,
        persistence_count=1,
        max_correction_norm_m=1.0,
        base_inflation=10.0,
        min_observable_rank=3,
    )
    cfg.observability.window_size = 12
    cfg.observability.min_rank_for_feedback = 1
    cfg.observability.min_gradient_norm = 1.0e-14

    runner = ScenarioSimulationRunner(cfg)
    result = runner.run_with_specs(
        scenario_or_truth=truth,
        imu_spec=IMUSpec.perfect(),
        gravimeter_spec=GravimeterSpec.perfect_relative(),
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        map_model=map_fn,
        dt_s=5.0,
        seed=456,
    )

    rows = result.estimators.custom_streams.get("pf_directional_feedback")
    assert rows is not None
    assert len(rows) == len(result.estimators.pf_updates)
    assert len(rows) > 0
    assert sum(1 for row in rows if row.get("applied")) == 0
    assert all(row.get("observable_rank") is not None for row in rows)
    assert all(
        str(row.get("rejection_reason", "")).startswith("observable_rank=")
        for row in rows
    )
