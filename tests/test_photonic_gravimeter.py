from __future__ import annotations

import numpy as np

from gravnav.sensors.photonic_gravimeter import (
    PhotonicGravimeterSensor,
    PhotonicGravimeterSpec,
)


def test_photonic_reference_load_cancels_static_motion_bias() -> None:
    sensor = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="idealized",
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            specific_force_coupling_b=(0.0, 0.0, 1.0e-5),
            specific_force_reference_b_mps2=(0.0, 0.0, -9.80665),
            angular_rate_coupling_b_mps2_per_radps=(0.0, 0.0, 0.0),
        ),
        rng=np.random.default_rng(0),
    )

    meas = sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=np.array([0.0, 0.0, -9.80665]),
        C_n_b=np.eye(3),
        v_dot_ned_mps2=np.zeros(3),
        v_ned_mps=np.zeros(3),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=1.0,
        body_angular_rate_b_radps=np.zeros(3),
        time_s=0.0,
    )

    assert meas.is_valid
    assert abs(meas.motion_residual_mps2) < 1.0e-12


def test_photonic_mission_mode_applies_warmup_and_cadence() -> None:
    sensor = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="mission",
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            update_period_s=2.0,
            warmup_time_s=3.0,
            specific_force_coupling_b=(0.0, 0.0, 1.0e-5),
            specific_force_reference_b_mps2=(0.0, 0.0, -9.80665),
            angular_rate_coupling_b_mps2_per_radps=(0.0, 0.0, 0.0),
        ),
        rng=np.random.default_rng(1),
    )

    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    values = [
        sensor.measure_disturbance_from_specific_force_body(
            specific_force_body_mps2=np.array([0.0, 0.0, -9.80665]),
            C_n_b=np.eye(3),
            v_dot_ned_mps2=np.zeros(3),
            v_ned_mps=np.zeros(3),
            lat_rad=0.0,
            height_m=0.0,
            dt_s=1.0,
            body_angular_rate_b_radps=np.zeros(3),
            time_s=t,
        )
        for t in times
    ]

    assert [m.warmup_complete for m in values] == [False, False, False, True, True, True]
    assert [m.cadence_emitted for m in values] == [True, True, True, True, False, True]
    assert [m.is_valid for m in values] == [False, False, False, True, False, True]


def test_photonic_idealized_mode_ignores_warmup_and_cadence() -> None:
    sensor = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="idealized",
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            update_period_s=5.0,
            warmup_time_s=120.0,
        ),
        rng=np.random.default_rng(2),
    )

    first = sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=np.array([0.0, 0.0, -9.80665]),
        C_n_b=np.eye(3),
        v_dot_ned_mps2=np.zeros(3),
        v_ned_mps=np.zeros(3),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=0.5,
        body_angular_rate_b_radps=np.zeros(3),
        time_s=0.0,
    )
    second = sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=np.array([0.0, 0.0, -9.80665]),
        C_n_b=np.eye(3),
        v_dot_ned_mps2=np.zeros(3),
        v_ned_mps=np.zeros(3),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=0.5,
        body_angular_rate_b_radps=np.zeros(3),
        time_s=0.5,
    )

    assert first.is_valid
    assert second.is_valid
    assert first.cadence_emitted
    assert second.cadence_emitted
