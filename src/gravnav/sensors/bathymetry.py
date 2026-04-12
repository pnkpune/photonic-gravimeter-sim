"""
bathymetry.py

Simple seabed-clearance sensor model for regional bathymetry aiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BathymetrySensorSpec:
    """
    Echo-sounder style bathymetry sensor specification.

    The measurement is seabed clearance in metres, i.e. water-column depth from
    the platform down to the seafloor.
    """

    noise_std_m: float = 1.5
    turn_on_bias_std_m: float = 0.5
    fixed_bias_m: float = 0.0
    max_range_m: float = 6000.0
    name: str = "bathymetry_sensor"

    def __post_init__(self) -> None:
        self.noise_std_m = float(self.noise_std_m)
        self.turn_on_bias_std_m = float(self.turn_on_bias_std_m)
        self.fixed_bias_m = float(self.fixed_bias_m)
        self.max_range_m = float(self.max_range_m)
        self.name = str(self.name)
        if self.noise_std_m < 0.0:
            raise ValueError("noise_std_m must be nonnegative.")
        if self.turn_on_bias_std_m < 0.0:
            raise ValueError("turn_on_bias_std_m must be nonnegative.")
        if self.max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive.")


@dataclass
class BathymetryMeasurement:
    time_s: Optional[float]
    value_m: float
    ideal_value_m: float
    bias_used_m: float
    white_noise_m: float
    saturated: bool
    reference_surface_height_m: float = 0.0
    kind: str = "seafloor_clearance"


class BathymetrySensor:
    """Stateful scalar seabed-clearance sensor."""

    def __init__(
        self,
        spec: BathymetrySensorSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self._bias_m = float(self.spec.fixed_bias_m)
        self.reset()

    def reset(self, *, bias_override_m: Optional[float] = None) -> None:
        if bias_override_m is not None:
            self._bias_m = float(bias_override_m)
            return
        self._bias_m = float(self.spec.fixed_bias_m)
        if self.spec.turn_on_bias_std_m > 0.0:
            self._bias_m += float(self.spec.turn_on_bias_std_m * self.rng.standard_normal())

    def measure_seafloor_clearance(
        self,
        ideal_clearance_m: float,
        *,
        time_s: Optional[float] = None,
        reference_surface_height_m: float = 0.0,
    ) -> BathymetryMeasurement:
        ideal = float(ideal_clearance_m)
        noise = float(self.spec.noise_std_m * self.rng.standard_normal())
        value = ideal + self._bias_m + noise
        clipped = float(np.clip(value, 0.0, self.spec.max_range_m))
        saturated = not np.isclose(clipped, value)
        return BathymetryMeasurement(
            time_s=None if time_s is None else float(time_s),
            value_m=clipped,
            ideal_value_m=ideal,
            bias_used_m=float(self._bias_m),
            white_noise_m=noise,
            saturated=bool(saturated),
            reference_surface_height_m=float(reference_surface_height_m),
        )


__all__ = [
    "BathymetryMeasurement",
    "BathymetrySensor",
    "BathymetrySensorSpec",
]
