"""
magnetometer.py

Scalar total-field magnetometer model for passive regional magnetic aiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MagnetometerSensorSpec:
    noise_std_nt: float = 3.0
    turn_on_bias_std_nt: float = 15.0
    fixed_bias_nt: float = 0.0
    scale_factor_error_ppm: float = 0.0
    heading_disturbance_amplitude_nt: float = 20.0
    heading_disturbance_harmonic: int = 2
    hard_iron_bias_nt: float = 0.0
    max_abs_nt: float = 100_000.0
    name: str = "magnetometer"

    def __post_init__(self) -> None:
        self.noise_std_nt = float(self.noise_std_nt)
        self.turn_on_bias_std_nt = float(self.turn_on_bias_std_nt)
        self.fixed_bias_nt = float(self.fixed_bias_nt)
        self.scale_factor_error_ppm = float(self.scale_factor_error_ppm)
        self.heading_disturbance_amplitude_nt = float(self.heading_disturbance_amplitude_nt)
        self.heading_disturbance_harmonic = int(self.heading_disturbance_harmonic)
        self.hard_iron_bias_nt = float(self.hard_iron_bias_nt)
        self.max_abs_nt = float(self.max_abs_nt)
        self.name = str(self.name)
        if self.noise_std_nt < 0.0:
            raise ValueError("noise_std_nt must be nonnegative.")
        if self.turn_on_bias_std_nt < 0.0:
            raise ValueError("turn_on_bias_std_nt must be nonnegative.")
        if self.max_abs_nt <= 0.0:
            raise ValueError("max_abs_nt must be positive.")
        if self.heading_disturbance_harmonic < 1:
            raise ValueError("heading_disturbance_harmonic must be positive.")


@dataclass
class MagnetometerMeasurement:
    time_s: Optional[float]
    value_nt: float
    ideal_value_nt: float
    bias_used_nt: float
    white_noise_nt: float
    heading_disturbance_nt: float
    saturated: bool
    kind: str = "total_field"


class ScalarMagnetometerSensor:
    """
    Stateful scalar total-field magnetometer.
    """

    def __init__(
        self,
        spec: MagnetometerSensorSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self._bias_nt = 0.0
        self.reset()

    def reset(self, *, bias_override_nt: Optional[float] = None) -> None:
        if bias_override_nt is not None:
            self._bias_nt = float(bias_override_nt)
            return
        self._bias_nt = float(self.spec.fixed_bias_nt + self.spec.hard_iron_bias_nt)
        if self.spec.turn_on_bias_std_nt > 0.0:
            self._bias_nt += float(self.spec.turn_on_bias_std_nt * self.rng.standard_normal())

    def measure_total_field(
        self,
        ideal_total_field_nt: float,
        *,
        heading_rad: float = 0.0,
        time_s: Optional[float] = None,
    ) -> MagnetometerMeasurement:
        ideal = float(ideal_total_field_nt)
        noise = float(self.spec.noise_std_nt * self.rng.standard_normal())
        heading_term = float(
            self.spec.heading_disturbance_amplitude_nt
            * np.cos(float(self.spec.heading_disturbance_harmonic) * float(heading_rad))
        )
        scale = 1.0 + 1.0e-6 * float(self.spec.scale_factor_error_ppm)
        value = scale * ideal + self._bias_nt + heading_term + noise
        clipped = float(np.clip(value, -self.spec.max_abs_nt, self.spec.max_abs_nt))
        saturated = not np.isclose(clipped, value)
        return MagnetometerMeasurement(
            time_s=None if time_s is None else float(time_s),
            value_nt=clipped,
            ideal_value_nt=ideal,
            bias_used_nt=float(self._bias_nt),
            white_noise_nt=noise,
            heading_disturbance_nt=heading_term,
            saturated=bool(saturated),
        )


__all__ = [
    "MagnetometerMeasurement",
    "MagnetometerSensorSpec",
    "ScalarMagnetometerSensor",
]
