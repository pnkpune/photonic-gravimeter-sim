from __future__ import annotations

import numpy as np

from gravnav.estimators.map_match_pf import (
    GravityMapParticleFilter,
    MapMatchPFSpec,
    apply_ned_offsets_to_geodetic,
    evaluate_gravity_map_horizontal_gradient,
    geodetic_offsets_to_local_ned,
)
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
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


def test_evaluate_gravity_map_horizontal_gradient_matches_quadratic_truth() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 0.0
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    truth_n = 30.0
    truth_e = -20.0
    lat_t, lon_t, h_t = apply_ned_offsets_to_geodetic(
        np.array([lat0], dtype=np.float64),
        np.array([lon0], dtype=np.float64),
        np.array([h0], dtype=np.float64),
        np.array([[truth_n, truth_e, 0.0]], dtype=np.float64),
    )

    grad = evaluate_gravity_map_horizontal_gradient(
        map_fn,
        lat_t,
        lon_t,
        h_t,
        delta_north_m=25.0,
        delta_east_m=25.0,
    ).reshape(2)

    expected = np.array(
        [
            2.0e-8 + 1.2e-10 * truth_n,
            -1.5e-8 + 8.0e-11 * truth_e,
        ],
        dtype=np.float64,
    )
    assert np.allclose(grad, expected, rtol=0.0, atol=5.0e-13)


def test_gradient_likelihood_tightens_pf_posterior_and_improves_estimate() -> None:
    lat0 = np.deg2rad(18.25)
    lon0 = np.deg2rad(72.75)
    h0 = 0.0
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    truth_n = 30.0
    truth_e = -20.0
    lat_t, lon_t, h_t = apply_ned_offsets_to_geodetic(
        np.array([lat0], dtype=np.float64),
        np.array([lon0], dtype=np.float64),
        np.array([h0], dtype=np.float64),
        np.array([[truth_n, truth_e, 0.0]], dtype=np.float64),
    )
    lat_t = float(lat_t[0])
    lon_t = float(lon_t[0])
    h_t = float(h_t[0])

    gravity_true = float(
        map_fn(
            np.array([lat_t], dtype=np.float64),
            np.array([lon_t], dtype=np.float64),
            np.array([h_t], dtype=np.float64),
        )[0]
    )
    gradient_true = evaluate_gravity_map_horizontal_gradient(
        map_fn,
        np.array([lat_t], dtype=np.float64),
        np.array([lon_t], dtype=np.float64),
        np.array([h_t], dtype=np.float64),
    ).reshape(2)

    spec = MapMatchPFSpec(
        num_particles=512,
        init_position_std_m=(80.0, 80.0, 1.0),
        process_position_rw_std_m_per_sqrt_s=(0.0, 0.0, 0.0),
        rejuvenation_std_m=(0.0, 0.0, 0.0),
        gravity_meas_std_mps2=2.0e-7,
        gradient_meas_std_per_s2=5.0e-10,
        use_ins_position_prior=False,
        resample_effective_fraction=0.2,
    )

    pf_scalar = GravityMapParticleFilter(spec, map_fn, rng=np.random.default_rng(1))
    pf_scalar.reset_around_geodetic(lat0, lon0, h0)

    pf_grad = GravityMapParticleFilter(spec, map_fn, rng=np.random.default_rng(1))
    pf_grad.reset_around_geodetic(lat0, lon0, h0)

    scalar_update = pf_scalar.update(
        gravity_true,
        gravity_meas_std_mps2=2.0e-7,
    )
    gradient_update = pf_grad.update(
        gravity_true,
        gravity_meas_std_mps2=2.0e-7,
        measured_gradient_per_s2=gradient_true,
        gradient_meas_std_per_s2=5.0e-10,
    )

    scalar_err_ned = geodetic_offsets_to_local_ned(
        np.array([scalar_update.estimate.lat_rad]),
        np.array([scalar_update.estimate.lon_rad]),
        np.array([scalar_update.estimate.height_m]),
        lat_ref_rad=lat_t,
        lon_ref_rad=lon_t,
        height_ref_m=h_t,
    )[0]
    gradient_err_ned = geodetic_offsets_to_local_ned(
        np.array([gradient_update.estimate.lat_rad]),
        np.array([gradient_update.estimate.lon_rad]),
        np.array([gradient_update.estimate.height_m]),
        lat_ref_rad=lat_t,
        lon_ref_rad=lon_t,
        height_ref_m=h_t,
    )[0]

    scalar_horizontal_err = float(np.linalg.norm(scalar_err_ned[:2]))
    gradient_horizontal_err = float(np.linalg.norm(gradient_err_ned[:2]))

    assert gradient_horizontal_err < scalar_horizontal_err
    assert np.trace(gradient_update.estimate.covariance_ned_m2) < np.trace(
        scalar_update.estimate.covariance_ned_m2
    )
    assert (
        gradient_update.predicted_disturbance_std_mps2
        < scalar_update.predicted_disturbance_std_mps2
    )


def test_runner_pf_history_is_reproducible_for_same_seed() -> None:
    scenario = get_named_scenario("maritime_baseline")
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=2.0)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])
    map_fn = _quadratic_map_factory(lat0, lon0, h0)

    cfg = SimulationRunnerConfig()
    cfg.map_match.pf_spec = MapMatchPFSpec(
        num_particles=64,
        init_position_std_m=(40.0, 40.0, 1.0),
        process_position_rw_std_m_per_sqrt_s=(0.0, 0.0, 0.0),
        rejuvenation_std_m=(0.0, 0.0, 0.0),
        gravity_meas_std_mps2=1.0e-6,
        use_ins_position_prior=False,
        resample_effective_fraction=0.5,
    )
    cfg.map_match.depth_meas_std_m = 0.1

    runner = ScenarioSimulationRunner(cfg)
    common_kwargs = dict(
        scenario_or_truth=truth,
        imu_spec=IMUSpec.perfect(),
        gravimeter_spec=GravimeterSpec.perfect_relative(),
        depth_spec=DepthSensorSpec.perfect(),
        velocity_aid_spec=VelocityAidSpec.perfect(),
        map_model=map_fn,
        dt_s=2.0,
        seed=123,
    )

    result_a = runner.run_with_specs(**common_kwargs)
    result_b = runner.run_with_specs(**common_kwargs)

    pf_a = result_a.estimators.pf_history_arrays()
    pf_b = result_b.estimators.pf_history_arrays()

    assert np.array_equal(pf_a["pf_lat_rad"], pf_b["pf_lat_rad"])
    assert np.array_equal(pf_a["pf_lon_rad"], pf_b["pf_lon_rad"])
    assert np.array_equal(pf_a["pf_height_m"], pf_b["pf_height_m"])
    assert np.array_equal(pf_a["pf_effective_sample_size"], pf_b["pf_effective_sample_size"])
