"""
current_profile.py

Simple ADCP-like current-profile observation model for current-aware velocity
correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


def _vec3(x: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


@dataclass
class CurrentProfileSensorSpec:
    noise_std_mps: float = 0.05
    turn_on_bias_std_mps: float = 0.03
    fixed_bias_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_speed_mps: float = 3.0
    name: str = "current_profile_sensor"

    def __post_init__(self) -> None:
        self.noise_std_mps = float(self.noise_std_mps)
        self.turn_on_bias_std_mps = float(self.turn_on_bias_std_mps)
        self.fixed_bias_mps = tuple(float(x) for x in self.fixed_bias_mps)
        self.max_speed_mps = float(self.max_speed_mps)
        self.name = str(self.name)
        if self.noise_std_mps < 0.0:
            raise ValueError("noise_std_mps must be nonnegative.")
        if self.turn_on_bias_std_mps < 0.0:
            raise ValueError("turn_on_bias_std_mps must be nonnegative.")
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive.")


@dataclass
class CurrentProfileMeasurement:
    time_s: Optional[float]
    value_ned_mps: np.ndarray
    ideal_value_ned_mps: np.ndarray
    bias_used_ned_mps: np.ndarray
    white_noise_ned_mps: np.ndarray
    saturated: bool
    kind: str = "current_profile"


class CurrentProfileSensor:
    def __init__(
        self,
        spec: CurrentProfileSensorSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self._bias_ned_mps = np.zeros(3, dtype=np.float64)
        self.reset()

    def reset(self, *, bias_override_ned_mps: Optional[ArrayLike] = None) -> None:
        if bias_override_ned_mps is not None:
            self._bias_ned_mps = _vec3(bias_override_ned_mps, name="bias_override_ned_mps")
            return
        self._bias_ned_mps = np.asarray(self.spec.fixed_bias_mps, dtype=np.float64)
        if self.spec.turn_on_bias_std_mps > 0.0:
            self._bias_ned_mps += float(self.spec.turn_on_bias_std_mps) * self.rng.standard_normal(3)

    def measure_current_profile_ned(
        self,
        ideal_current_ned_mps: ArrayLike,
        *,
        time_s: Optional[float] = None,
    ) -> CurrentProfileMeasurement:
        ideal = _vec3(ideal_current_ned_mps, name="ideal_current_ned_mps")
        noise = float(self.spec.noise_std_mps) * self.rng.standard_normal(3)
        value = ideal + self._bias_ned_mps + noise
        speed = float(np.linalg.norm(value))
        saturated = False
        if speed > self.spec.max_speed_mps:
            value = value * (self.spec.max_speed_mps / max(speed, 1.0e-9))
            saturated = True
        return CurrentProfileMeasurement(
            time_s=None if time_s is None else float(time_s),
            value_ned_mps=np.asarray(value, dtype=np.float64),
            ideal_value_ned_mps=np.asarray(ideal, dtype=np.float64),
            bias_used_ned_mps=np.asarray(self._bias_ned_mps, dtype=np.float64),
            white_noise_ned_mps=np.asarray(noise, dtype=np.float64),
            saturated=bool(saturated),
        )


__all__ = [
    "CurrentProfileMeasurement",
    "CurrentProfileSensor",
    "CurrentProfileSensorSpec",
]
