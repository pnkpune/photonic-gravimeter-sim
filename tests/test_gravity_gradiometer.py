from __future__ import annotations

import numpy as np

from gravnav.sensors.gravity_gradiometer import (
    EOTVOS_PER_INVERSE_S2,
    INVERSE_S2_PER_EOTVOS,
    GravityGradiometerSensor,
    GravityGradiometerSpec,
)


def test_eotvos_conversion_constants_are_reciprocals() -> None:
    assert np.isclose(INVERSE_S2_PER_EOTVOS, 1.0e-9)
    assert np.isclose(EOTVOS_PER_INVERSE_S2, 1.0e9)
    assert np.isclose(INVERSE_S2_PER_EOTVOS * EOTVOS_PER_INVERSE_S2, 1.0)


def test_spec_scalar_inputs_broadcast_to_two_axes() -> None:
    spec = GravityGradiometerSpec(
        noise_density_per_s2_per_sqrt_hz=2.0e-9,
        bias_random_walk_per_s2_per_sqrt_s=3.0e-10,
        turn_on_bias_std_per_s2=4.0e-9,
        fixed_bias_per_s2=5.0e-9,
        scale_factor_error_ppm=6.0,
    )
    assert spec.noise_density_per_s2_per_sqrt_hz.shape == (2,)
    assert np.allclose(spec.noise_density_per_s2_per_sqrt_hz, [2.0e-9, 2.0e-9])
    assert np.allclose(spec.bias_random_walk_per_s2_per_sqrt_s, [3.0e-10, 3.0e-10])
    assert np.allclose(spec.turn_on_bias_std_per_s2, [4.0e-9, 4.0e-9])
    assert np.allclose(spec.fixed_bias_per_s2, [5.0e-9, 5.0e-9])
    assert np.allclose(spec.scale_factor_error_ppm, [6.0, 6.0])


def test_perfect_gradiometer_measurement_is_identity() -> None:
    spec = GravityGradiometerSpec(
        noise_density_per_s2_per_sqrt_hz=0.0,
        bias_random_walk_per_s2_per_sqrt_s=0.0,
        turn_on_bias_std_per_s2=0.0,
        fixed_bias_per_s2=0.0,
        scale_factor_error_ppm=0.0,
    )
    sensor = GravityGradiometerSensor(spec, rng=np.random.default_rng(123))

    ideal = np.array([4.0e-9, -7.0e-9], dtype=np.float64)
    meas = sensor.measure(ideal, dt_s=2.0, time_s=12.5)

    assert meas.time_s == 12.5
    assert np.allclose(meas.ideal_value_per_s2, ideal)
    assert np.allclose(meas.value_per_s2, ideal)
    assert np.allclose(meas.bias_used_per_s2, np.zeros(2))
    assert np.allclose(meas.white_noise_per_s2, np.zeros(2))
    assert np.array_equal(meas.saturated, np.array([False, False]))
    assert np.allclose(meas.value_eotvos, ideal * 1.0e9)


def test_saturation_is_applied_per_axis() -> None:
    spec = GravityGradiometerSpec(
        noise_density_per_s2_per_sqrt_hz=0.0,
        bias_random_walk_per_s2_per_sqrt_s=0.0,
        turn_on_bias_std_per_s2=0.0,
        fixed_bias_per_s2=0.0,
        scale_factor_error_ppm=0.0,
        max_abs_per_s2=5.0e-9,
    )
    sensor = GravityGradiometerSensor(spec, rng=np.random.default_rng(1))
    meas = sensor.measure(np.array([8.0e-9, -2.0e-9]), dt_s=1.0)

    assert np.allclose(meas.value_per_s2, [5.0e-9, -2.0e-9])
    assert np.array_equal(meas.saturated, np.array([True, False]))
