"""
photonic_gravimeter.py

Mission-facing photonic gravimeter wrapper built on top of the existing scalar
gravimeter measurement chain.

This intentionally does not simulate optical-state or atom-interferometer phase
physics directly. It exposes the parts that change navigation behavior:
cadence, warm-up, effective sensitivity, and residual motion fragility after
IMU-assisted rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .gravimeter import GravimeterMeasurement, GravimeterSpec, ScalarGravimeterSensor


class PhotonicOperatingMode(str, Enum):
    IDEALIZED = "idealized"
    MISSION = "mission"
    DEGRADED = "degraded"


@dataclass
class PhotonicGravimeterSpec:
    """
    Hardware-tied photonic gravimeter configuration.

    The internal scalar gravimeter parameters define the baseline measurement
    quality. Operating mode then adjusts cadence, validity, and degradation.
    """

    name: str = "photonic_gravimeter"
    operating_mode: str = PhotonicOperatingMode.MISSION.value
    noise_density_mps2_per_sqrt_hz: float = 4.0e-6
    bias_random_walk_mps2_per_sqrt_s: float = 2.0e-8
    turn_on_bias_std_mps2: float = 1.0e-6
    fixed_bias_mps2: float = 0.0
    scale_factor_error_ppm: float = 0.0
    bandwidth_hz: float = 0.4
    # Residual dynamic coupling is modeled around the nominal stationary
    # 1 g support load in body coordinates, not around zero specific force.
    specific_force_coupling_b: ArrayLike | float = (1.0e-6, 1.0e-6, 1.0e-5)
    specific_force_reference_b_mps2: ArrayLike | float = (0.0, 0.0, -9.80665)
    angular_rate_coupling_b_mps2_per_radps: ArrayLike | float = (
        1.5e-6,
        1.5e-6,
        1.5e-5,
    )
    supports_absolute_mode: bool = True
    update_period_s: float = 2.0
    warmup_time_s: float = 45.0
    valid_duty_cycle: float = 1.0
    degraded_noise_factor: float = 2.5
    degraded_bias_factor: float = 2.0
    degraded_motion_factor: float = 2.0
    max_abs_mps2: float = np.inf

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.operating_mode = str(self.operating_mode).strip().lower()
        valid_modes = {mode.value for mode in PhotonicOperatingMode}
        if self.operating_mode not in valid_modes:
            raise ValueError(
                f"operating_mode must be one of {sorted(valid_modes)}, got {self.operating_mode!r}."
            )
        self.noise_density_mps2_per_sqrt_hz = float(self.noise_density_mps2_per_sqrt_hz)
        self.bias_random_walk_mps2_per_sqrt_s = float(self.bias_random_walk_mps2_per_sqrt_s)
        self.turn_on_bias_std_mps2 = float(self.turn_on_bias_std_mps2)
        self.fixed_bias_mps2 = float(self.fixed_bias_mps2)
        self.scale_factor_error_ppm = float(self.scale_factor_error_ppm)
        self.bandwidth_hz = float(self.bandwidth_hz)
        self.supports_absolute_mode = bool(self.supports_absolute_mode)
        self.update_period_s = float(self.update_period_s)
        self.warmup_time_s = float(self.warmup_time_s)
        self.valid_duty_cycle = float(self.valid_duty_cycle)
        self.degraded_noise_factor = float(self.degraded_noise_factor)
        self.degraded_bias_factor = float(self.degraded_bias_factor)
        self.degraded_motion_factor = float(self.degraded_motion_factor)
        self.max_abs_mps2 = float(self.max_abs_mps2)
        if self.noise_density_mps2_per_sqrt_hz < 0.0:
            raise ValueError("noise_density_mps2_per_sqrt_hz must be nonnegative.")
        if self.bias_random_walk_mps2_per_sqrt_s < 0.0:
            raise ValueError("bias_random_walk_mps2_per_sqrt_s must be nonnegative.")
        if self.turn_on_bias_std_mps2 < 0.0:
            raise ValueError("turn_on_bias_std_mps2 must be nonnegative.")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive.")
        if self.update_period_s <= 0.0:
            raise ValueError("update_period_s must be positive.")
        if self.warmup_time_s < 0.0:
            raise ValueError("warmup_time_s must be nonnegative.")
        if not (0.0 < self.valid_duty_cycle <= 1.0):
            raise ValueError("valid_duty_cycle must lie in (0, 1].")
        if self.degraded_noise_factor <= 0.0:
            raise ValueError("degraded_noise_factor must be positive.")
        if self.degraded_bias_factor <= 0.0:
            raise ValueError("degraded_bias_factor must be positive.")
        if self.degraded_motion_factor <= 0.0:
            raise ValueError("degraded_motion_factor must be positive.")

    def effective_gravimeter_spec(self) -> GravimeterSpec:
        noise = self.noise_density_mps2_per_sqrt_hz
        bias_rw = self.bias_random_walk_mps2_per_sqrt_s
        turn_on_bias = self.turn_on_bias_std_mps2
        force_coupling = self.specific_force_coupling_b
        rate_coupling = self.angular_rate_coupling_b_mps2_per_radps

        if self.operating_mode == PhotonicOperatingMode.DEGRADED.value:
            noise *= self.degraded_noise_factor
            bias_rw *= self.degraded_bias_factor
            turn_on_bias *= self.degraded_bias_factor
            force_coupling = np.asarray(force_coupling, dtype=np.float64) * self.degraded_motion_factor
            rate_coupling = np.asarray(rate_coupling, dtype=np.float64) * self.degraded_motion_factor

        return GravimeterSpec(
            noise_density_mps2_per_sqrt_hz=noise,
            bias_random_walk_mps2_per_sqrt_s=bias_rw,
            turn_on_bias_std_mps2=turn_on_bias,
            fixed_bias_mps2=self.fixed_bias_mps2,
            scale_factor_error_ppm=self.scale_factor_error_ppm,
            max_abs_mps2=self.max_abs_mps2,
            bandwidth_hz=self.bandwidth_hz,
            specific_force_coupling_b=force_coupling,
            specific_force_reference_b_mps2=self.specific_force_reference_b_mps2,
            angular_rate_coupling_b_mps2_per_radps=rate_coupling,
            name=f"{self.name}_{self.operating_mode}",
            supports_absolute_mode=self.supports_absolute_mode,
        )


@dataclass
class PhotonicGravimeterMeasurement(GravimeterMeasurement):
    operating_mode: str = PhotonicOperatingMode.MISSION.value
    warmup_complete: bool = True
    cadence_emitted: bool = True
    effective_noise_std_mps2: float = 0.0
    is_valid: bool = True


class PhotonicGravimeterSensor:
    """Photonic gravimeter wrapper around the scalar gravimeter model."""

    def __init__(
        self,
        spec: PhotonicGravimeterSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self._inner = ScalarGravimeterSensor(
            self.spec.effective_gravimeter_spec(),
            rng=self.rng,
        )
        self._last_sample_time_s: Optional[float] = None

    def reset(self) -> None:
        self._inner = ScalarGravimeterSensor(
            self.spec.effective_gravimeter_spec(),
            rng=self.rng,
        )
        self._last_sample_time_s = None

    def current_bias_state(self):
        return self._inner.current_bias_state()

    def _warmup_complete(self, time_s: Optional[float]) -> bool:
        if self.spec.operating_mode == PhotonicOperatingMode.IDEALIZED.value:
            return True
        if time_s is None:
            return False
        return float(time_s) >= self.spec.warmup_time_s

    def _should_emit(self, time_s: Optional[float]) -> bool:
        if self.spec.operating_mode == PhotonicOperatingMode.IDEALIZED.value:
            return True
        if time_s is None:
            return False
        if self._last_sample_time_s is None:
            return True
        return (float(time_s) - float(self._last_sample_time_s)) >= (self.spec.update_period_s - 1.0e-9)

    def measure_disturbance_from_specific_force_body(
        self,
        *,
        specific_force_body_mps2: ArrayLike,
        C_n_b: ArrayLike,
        v_dot_ned_mps2: ArrayLike,
        v_ned_mps: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> PhotonicGravimeterMeasurement:
        warm = self._warmup_complete(time_s)
        emit = self._should_emit(time_s)
        valid = bool(warm and emit)

        if valid:
            base = self._inner.measure_disturbance_from_specific_force_body(
                specific_force_body_mps2=specific_force_body_mps2,
                C_n_b=C_n_b,
                v_dot_ned_mps2=v_dot_ned_mps2,
                v_ned_mps=v_ned_mps,
                lat_rad=lat_rad,
                height_m=height_m,
                dt_s=dt_s,
                body_angular_rate_b_radps=body_angular_rate_b_radps,
                time_s=time_s,
            )
            self._last_sample_time_s = None if time_s is None else float(time_s)
            white_std = self._inner.spec.white_noise_std(dt_s)
        else:
            base = self._inner.measure_disturbance_from_specific_force_body(
                specific_force_body_mps2=specific_force_body_mps2,
                C_n_b=C_n_b,
                v_dot_ned_mps2=v_dot_ned_mps2,
                v_ned_mps=v_ned_mps,
                lat_rad=lat_rad,
                height_m=height_m,
                dt_s=dt_s,
                body_angular_rate_b_radps=body_angular_rate_b_radps,
                time_s=time_s,
            )
            white_std = self._inner.spec.white_noise_std(dt_s)
            base = GravimeterMeasurement(
                kind=base.kind,
                time_s=base.time_s,
                value_mps2=float("nan"),
                ideal_value_mps2=base.ideal_value_mps2,
                motion_residual_mps2=base.motion_residual_mps2,
                filtered_input_mps2=base.filtered_input_mps2,
                bias_used_mps2=base.bias_used_mps2,
                white_noise_mps2=base.white_noise_mps2,
                saturated=False,
            )

        return PhotonicGravimeterMeasurement(
            kind=base.kind,
            time_s=base.time_s,
            value_mps2=base.value_mps2,
            ideal_value_mps2=base.ideal_value_mps2,
            motion_residual_mps2=base.motion_residual_mps2,
            filtered_input_mps2=base.filtered_input_mps2,
            bias_used_mps2=base.bias_used_mps2,
            white_noise_mps2=base.white_noise_mps2,
            saturated=base.saturated,
            operating_mode=self.spec.operating_mode,
            warmup_complete=bool(warm),
            cadence_emitted=bool(emit),
            effective_noise_std_mps2=float(white_std),
            is_valid=bool(valid),
        )


__all__ = [
    "PhotonicGravimeterMeasurement",
    "PhotonicGravimeterSensor",
    "PhotonicGravimeterSpec",
    "PhotonicOperatingMode",
]
