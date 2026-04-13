from __future__ import annotations

import numpy as np

from gravnav.physics.earth import normal_gravity
from gravnav.sensors.photonic_gravimeter import (
    InterferometerSequenceSpec,
    PhotonicGravimeterSensor,
    PhotonicGravimeterSpec,
    PhotonicSystematicsSpec,
    VibrationCompensationSpec,
)


def _static_specific_force_body(lat_rad: float = 0.0, height_m: float = 0.0) -> np.ndarray:
    return np.array([0.0, 0.0, -float(normal_gravity(lat_rad, height_m))], dtype=np.float64)


def _measure(
    sensor: PhotonicGravimeterSensor,
    *,
    time_s: float,
    dt_s: float,
    specific_force_body_mps2: np.ndarray | None = None,
    C_n_b: np.ndarray | None = None,
    body_angular_rate_b_radps: np.ndarray | None = None,
):
    return sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=_static_specific_force_body() if specific_force_body_mps2 is None else specific_force_body_mps2,
        C_n_b=np.eye(3) if C_n_b is None else C_n_b,
        v_dot_ned_mps2=np.zeros(3),
        v_ned_mps=np.zeros(3),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=dt_s,
        body_angular_rate_b_radps=np.zeros(3) if body_angular_rate_b_radps is None else body_angular_rate_b_radps,
        time_s=time_s,
    )


def test_photonic_reference_load_cancels_static_motion_bias() -> None:
    sensor = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="idealized",
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            fixed_bias_mps2=0.0,
        ),
        rng=np.random.default_rng(0),
    )

    meas = _measure(sensor, time_s=0.0, dt_s=1.0)

    assert meas.is_valid
    assert abs(meas.value_mps2) < 1.0e-15
    assert abs(meas.motion_residual_mps2) < 1.0e-15
    assert meas.telemetry is not None
    assert abs(meas.telemetry.gravity_phase_rad) < 1.0e-15
    assert abs(meas.telemetry.vibration_residual_phase_rad) < 1.0e-15


def test_photonic_mission_mode_applies_warmup_and_cadence() -> None:
    sensor = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="mission",
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            fixed_bias_mps2=0.0,
            update_period_s=2.0,
            warmup_time_s=3.0,
        ),
        rng=np.random.default_rng(1),
    )

    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    values = [_measure(sensor, time_s=t, dt_s=1.0) for t in times]

    assert [m.warmup_complete for m in values] == [False, False, False, True, True, True]
    assert [m.cadence_emitted for m in values] == [True, True, True, True, False, True]
    assert [m.is_valid for m in values] == [False, False, False, True, False, True]
    assert [m.rejection_reason for m in values] == ["warmup", "warmup", "warmup", None, "cadence", None]


def test_photonic_scale_factor_tracks_keff_t_squared() -> None:
    lat_rad = 0.0
    height_m = 0.0
    normal_g = float(normal_gravity(lat_rad, height_m))
    disturbance_mps2 = 1.0e-5
    specific_force_body_mps2 = np.array([0.0, 0.0, -(normal_g + disturbance_mps2)], dtype=np.float64)

    measurements = []
    for interrogation_time_s in (0.08, 0.16):
        spec = PhotonicGravimeterSpec(
            operating_mode="idealized",
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            fixed_bias_mps2=0.0,
            interferometer=InterferometerSequenceSpec(
                interrogation_time_s=interrogation_time_s,
                cycle_time_s=1.0,
            ),
        )
        sensor = PhotonicGravimeterSensor(spec, rng=np.random.default_rng(2))
        measurements.append(
            _measure(
                sensor,
                time_s=0.0,
                dt_s=1.0,
                specific_force_body_mps2=specific_force_body_mps2,
            )
        )

    ratio = measurements[1].telemetry.gravity_phase_rad / measurements[0].telemetry.gravity_phase_rad
    assert np.isclose(ratio, 4.0, rtol=1.0e-4)


def test_vibration_compensation_reduces_residual_phase() -> None:
    normal_g = float(normal_gravity(0.0, 0.0))
    base_kwargs = dict(
        operating_mode="mission",
        warmup_time_s=0.0,
        update_period_s=0.1,
        noise_density_mps2_per_sqrt_hz=0.0,
        bias_random_walk_mps2_per_sqrt_s=0.0,
        turn_on_bias_std_mps2=0.0,
        fixed_bias_mps2=0.0,
        systematics=PhotonicSystematicsSpec(
            body_rotation_residual_gain=0.0,
            wavefront_aberration_coeff_rad_per_m2=0.0,
            quadratic_zeeman_coeff_rad_per_t2=0.0,
            valid_tilt_limit_deg=90.0,
            min_valid_contrast=0.0,
        ),
    )
    uncompensated = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            **base_kwargs,
            vibration_compensation=VibrationCompensationSpec(
                enabled=False,
                accelerometer_noise_density_mps2_per_sqrt_hz=0.0,
                accelerometer_bias_random_walk_mps2_per_sqrt_s=0.0,
                residual_correction_fraction=0.0,
                accelerometer_bandwidth_hz=100.0,
            ),
        ),
        rng=np.random.default_rng(0),
    )
    compensated = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            **base_kwargs,
            vibration_compensation=VibrationCompensationSpec(
                enabled=True,
                accelerometer_noise_density_mps2_per_sqrt_hz=0.0,
                accelerometer_bias_random_walk_mps2_per_sqrt_s=0.0,
                residual_correction_fraction=1.0,
                accelerometer_latency_s=0.0,
                accelerometer_alignment_error_rad=0.0,
                accelerometer_bandwidth_hz=100.0,
            ),
        ),
        rng=np.random.default_rng(0),
    )

    off_residuals = []
    on_residuals = []
    for k in range(40):
        time_s = 0.1 * k
        dynamic_accel_mps2 = 0.02 * np.sin(2.0 * np.pi * 0.5 * time_s)
        specific_force_body_mps2 = np.array([0.0, 0.0, -(normal_g + dynamic_accel_mps2)], dtype=np.float64)
        off_residuals.append(
            abs(
                _measure(
                    uncompensated,
                    time_s=time_s,
                    dt_s=0.1,
                    specific_force_body_mps2=specific_force_body_mps2,
                ).telemetry.vibration_residual_phase_rad
            )
        )
        on_residuals.append(
            abs(
                _measure(
                    compensated,
                    time_s=time_s,
                    dt_s=0.1,
                    specific_force_body_mps2=specific_force_body_mps2,
                ).telemetry.vibration_residual_phase_rad
            )
        )

    assert np.mean(on_residuals[-10:]) < 0.01 * np.mean(off_residuals[-10:])


def test_systematic_terms_are_visible_in_telemetry() -> None:
    spec = PhotonicGravimeterSpec(
        operating_mode="mission",
        warmup_time_s=0.0,
        noise_density_mps2_per_sqrt_hz=0.0,
        bias_random_walk_mps2_per_sqrt_s=0.0,
        turn_on_bias_std_mps2=0.0,
        fixed_bias_mps2=0.0,
        atom_ensemble={
            "detection_noise_std_probability": 0.0,
            "cloud_radius_m": 0.01,
            "base_contrast": 0.8,
            "contrast_floor": 0.2,
        },
        vibration_compensation={
            "accelerometer_noise_density_mps2_per_sqrt_hz": 0.0,
            "accelerometer_bias_random_walk_mps2_per_sqrt_s": 0.0,
        },
        systematics={
            "gravity_gradient_zz_per_s2": 3.0e-6,
            "use_normal_vertical_gradient": False,
            "body_rotation_residual_gain": 0.2,
            "wavefront_aberration_coeff_rad_per_m2": 150.0,
            "quadratic_zeeman_coeff_rad_per_t2": 1.0e6,
            "magnetic_field_bias_t": 2.0e-4,
            "ac_stark_coeff_rad_per_fraction": 0.5,
            "relative_intensity_error": 0.2,
            "chirp_error_fraction": 1.0e-6,
            "valid_tilt_limit_deg": 90.0,
        },
    )
    sensor = PhotonicGravimeterSensor(spec, rng=np.random.default_rng(3))
    meas = _measure(
        sensor,
        time_s=0.0,
        dt_s=1.0,
        body_angular_rate_b_radps=np.array([0.0, 0.1, 0.0]),
    )

    assert meas.telemetry is not None
    assert abs(meas.telemetry.gradient_phase_rad) > 0.0
    assert abs(meas.telemetry.rotation_phase_rad) > 0.0
    assert abs(meas.telemetry.wavefront_phase_rad) > 0.0
    assert abs(meas.telemetry.zeeman_phase_rad) > 0.0
    assert abs(meas.telemetry.lightshift_phase_rad) > 0.0
    assert abs(meas.telemetry.chirp_phase_rad) > 0.0


def test_degraded_mode_reduces_contrast_and_can_invalidate_samples() -> None:
    roll_rad = np.deg2rad(1.0)
    C_n_b = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0.0, np.sin(roll_rad), np.cos(roll_rad)],
        ],
        dtype=np.float64,
    )
    mission = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="mission",
            warmup_time_s=0.0,
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            fixed_bias_mps2=0.0,
        ),
        rng=np.random.default_rng(4),
    )
    degraded = PhotonicGravimeterSensor(
        PhotonicGravimeterSpec(
            operating_mode="degraded",
            warmup_time_s=0.0,
            noise_density_mps2_per_sqrt_hz=0.0,
            bias_random_walk_mps2_per_sqrt_s=0.0,
            turn_on_bias_std_mps2=0.0,
            fixed_bias_mps2=0.0,
        ),
        rng=np.random.default_rng(4),
    )

    mission_meas = _measure(
        mission,
        time_s=0.0,
        dt_s=1.0,
        C_n_b=C_n_b,
        body_angular_rate_b_radps=np.array([0.0, 0.01, 0.0]),
    )
    degraded_meas = _measure(
        degraded,
        time_s=0.0,
        dt_s=1.0,
        C_n_b=C_n_b,
        body_angular_rate_b_radps=np.array([0.0, 0.01, 0.0]),
    )

    assert degraded_meas.telemetry is not None
    assert mission_meas.telemetry is not None
    assert degraded_meas.telemetry.fringe_contrast < mission_meas.telemetry.fringe_contrast
    assert (not degraded_meas.is_valid) or (
        degraded_meas.telemetry.fringe_contrast < mission_meas.telemetry.fringe_contrast
    )


def test_nested_config_blocks_are_coerced_into_dataclasses() -> None:
    spec = PhotonicGravimeterSpec(
        interferometer={"interrogation_time_s": 0.2, "cycle_time_s": 1.5},
        atom_ensemble={"detection_noise_std_probability": 0.02},
        vibration_compensation={"residual_correction_fraction": 0.9},
        systematics={"wavefront_aberration_coeff_rad_per_m2": 10.0},
    )

    assert isinstance(spec.interferometer, InterferometerSequenceSpec)
    assert isinstance(spec.vibration_compensation, VibrationCompensationSpec)
    assert isinstance(spec.systematics, PhotonicSystematicsSpec)
    assert spec.interferometer.interrogation_time_s == 0.2
    assert spec.atom_ensemble.detection_noise_std_probability == 0.02
    assert spec.vibration_compensation.residual_correction_fraction == 0.9
