"""
photonic_gravimeter.py

High-fidelity photonic gravimeter models for the gravity-aided navigation
simulator.

This module preserves the existing navigation-facing interface:

- `PhotonicGravimeterSpec`
- `PhotonicGravimeterMeasurement`
- `PhotonicGravimeterSensor`

but replaces the previous surrogate-only internals with a phase-domain digital
twin for a cold-atom Raman Mach-Zehnder gravimeter. The digital twin is aimed at
navigation-relevant realism on moving maritime platforms rather than laboratory
optical-state simulation.

Modeling philosophy
-------------------
The digital twin combines:
- Raman light-pulse interferometer scale factor `k_eff T^2`
- sensitivity-function-based vibration phase integration
- classical accelerometer-assisted vibration compensation
- moving-platform cadence, warm-up, duty cycle, contrast loss, and dropouts
- gravity-gradient, Coriolis / rotation, wavefront, Zeeman, Stark, chirp, and
  finite-pulse corrections
- phase-to-fringe conversion and phase inversion at mid-fringe

The top-level spec also retains the legacy surrogate parameters so the same
sensor object can still be used in A/B comparisons against the previous model.

Reference model basis
---------------------
This digital twin is a composite navigation-facing model informed primarily by:

- Lellouch et al. (2025), Integration of a high-fidelity model of quantum
  sensors with a map-matching algorithm for ship positioning
- Cheinet et al. (2008), Measurement of the sensitivity function in a
  time-domain atomic interferometer
- Bidel et al. (2018), Absolute marine gravimetry with matter-wave
  interferometry
- Jensen et al. (2025), Airborne gravimetry with quantum technology:
  observations from Iceland and Greenland
- Nobili et al. (2020), related atom-interferometer systematic and wavefront
  error analyses

It intentionally stops short of full optical Bloch, laser-cooling, or vacuum
subsystem simulation because those details are lower-value than phase-domain
accuracy for GNSS-denied navigation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..physics.earth import normal_gravity, normal_gravity_vertical_gradient
from ..physics.frames import earth_rate_ned, project_to_so3
from .gravimeter import (
    GravimeterBiasState,
    GravimeterMeasurement,
    GravimeterSpec,
    ScalarGravimeterSensor,
    recover_gravity_disturbance_from_specific_force_body,
)

_BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
_RB87_MASS_KG = 1.44316060e-25


def _as_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _vec3(x: ArrayLike, *, name: str) -> np.ndarray:
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}.")
    return arr


def _scalar_or_vec3_to_vec3(x: ArrayLike | float, *, name: str) -> np.ndarray:
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}.")
    return arr


def _coerce_dataclass(value, cls, *, name: str):
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"{name} must be a {cls.__name__} or mapping, got {type(value).__name__}.")


def _normalize(v: ArrayLike, *, name: str) -> np.ndarray:
    arr = _vec3(v, name=name)
    n = float(np.linalg.norm(arr))
    if n <= 0.0:
        raise ValueError(f"{name} norm must be positive.")
    return arr / n


def _first_order_lowpass_step(
    previous: Optional[float],
    x: float,
    dt_s: float,
    cutoff_hz: Optional[float],
) -> float:
    if cutoff_hz is None:
        return float(x)
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive when provided.")
    if previous is None:
        return float(x)
    alpha = 1.0 - np.exp(-2.0 * np.pi * float(cutoff_hz) * float(dt_s))
    return float(previous + alpha * (float(x) - float(previous)))


def _temperature_to_velocity_std_mps(temperature_k: float, mass_kg: float) -> float:
    if temperature_k <= 0.0:
        return 0.0
    return float(np.sqrt(_BOLTZMANN_CONSTANT_J_PER_K * temperature_k / mass_kg))


def _unwrap_midfringe_phase(principal_phase_rad: float, reference_phase_rad: float) -> float:
    """
    Unwrap a mid-fringe `arcsin` phase to the branch closest to a reference.

    For `P = P0 - (C/2) sin(phi)`, the principal solution from `arcsin` lies in
    [-pi/2, pi/2]. Equivalent physical phases are:

    - principal + 2*pi*n
    - (pi - principal) + 2*pi*n

    We choose the candidate closest to the model-predicted reference phase.
    """
    candidates: list[float] = []
    for branch in (principal_phase_rad, np.pi - principal_phase_rad):
        for n in range(-6, 7):
            candidates.append(float(branch + 2.0 * np.pi * n))
    return min(candidates, key=lambda cand: abs(cand - reference_phase_rad))


class PhotonicOperatingMode(str, Enum):
    IDEALIZED = "idealized"
    MISSION = "mission"
    DEGRADED = "degraded"


class PhotonicPhysicsModel(str, Enum):
    SURROGATE = "surrogate"
    DIGITAL_TWIN = "digital_twin"


@dataclass
class InterferometerSequenceSpec:
    atom_species: str = "Rb87"
    pulse_sequence: str = "raman_mach_zehnder"
    effective_wavevector_rad_per_m: float = 1.611e7
    pulse_duration_s: float = 12.0e-6
    interrogation_time_s: float = 0.12
    cycle_time_s: float = 1.0
    chirp_rate_rad_per_s2: float = 0.0
    auto_chirp_from_normal_gravity: bool = True
    lmt_order: int = 1
    pulse_efficiency: float = 0.995
    fringe_phase_bias_rad: float = float(np.pi / 2.0)
    fringe_offset_probability: float = 0.5

    def __post_init__(self) -> None:
        self.atom_species = str(self.atom_species)
        self.pulse_sequence = str(self.pulse_sequence)
        self.effective_wavevector_rad_per_m = float(self.effective_wavevector_rad_per_m)
        self.pulse_duration_s = float(self.pulse_duration_s)
        self.interrogation_time_s = float(self.interrogation_time_s)
        self.cycle_time_s = float(self.cycle_time_s)
        self.chirp_rate_rad_per_s2 = float(self.chirp_rate_rad_per_s2)
        self.auto_chirp_from_normal_gravity = bool(self.auto_chirp_from_normal_gravity)
        self.lmt_order = int(self.lmt_order)
        self.pulse_efficiency = float(self.pulse_efficiency)
        self.fringe_phase_bias_rad = float(self.fringe_phase_bias_rad)
        self.fringe_offset_probability = float(self.fringe_offset_probability)
        if self.effective_wavevector_rad_per_m <= 0.0:
            raise ValueError("effective_wavevector_rad_per_m must be positive.")
        if self.pulse_duration_s <= 0.0:
            raise ValueError("pulse_duration_s must be positive.")
        if self.interrogation_time_s <= 0.0:
            raise ValueError("interrogation_time_s must be positive.")
        if self.cycle_time_s <= 0.0:
            raise ValueError("cycle_time_s must be positive.")
        if self.lmt_order <= 0:
            raise ValueError("lmt_order must be positive.")
        if not (0.0 < self.pulse_efficiency <= 1.0):
            raise ValueError("pulse_efficiency must lie in (0, 1].")
        if not (0.0 <= self.fringe_offset_probability <= 1.0):
            raise ValueError("fringe_offset_probability must lie in [0, 1].")


@dataclass
class AtomEnsembleSpec:
    temperature_k: float = 2.0e-6
    transverse_temperature_k: float = 2.0e-6
    cloud_radius_m: float = 2.5e-3
    launch_position_axis_m: float = 0.0
    launch_velocity_axis_mps: float = 0.0
    transverse_velocity_std_mps: float = 0.0
    detection_noise_std_probability: float = 0.01
    dead_time_s: float = 0.15
    base_contrast: float = 0.62
    contrast_floor: float = 0.18

    def __post_init__(self) -> None:
        self.temperature_k = float(self.temperature_k)
        self.transverse_temperature_k = float(self.transverse_temperature_k)
        self.cloud_radius_m = float(self.cloud_radius_m)
        self.launch_position_axis_m = float(self.launch_position_axis_m)
        self.launch_velocity_axis_mps = float(self.launch_velocity_axis_mps)
        self.transverse_velocity_std_mps = float(self.transverse_velocity_std_mps)
        self.detection_noise_std_probability = float(self.detection_noise_std_probability)
        self.dead_time_s = float(self.dead_time_s)
        self.base_contrast = float(self.base_contrast)
        self.contrast_floor = float(self.contrast_floor)
        if self.temperature_k < 0.0:
            raise ValueError("temperature_k must be nonnegative.")
        if self.transverse_temperature_k < 0.0:
            raise ValueError("transverse_temperature_k must be nonnegative.")
        if self.cloud_radius_m < 0.0:
            raise ValueError("cloud_radius_m must be nonnegative.")
        if self.transverse_velocity_std_mps < 0.0:
            raise ValueError("transverse_velocity_std_mps must be nonnegative.")
        if self.detection_noise_std_probability < 0.0:
            raise ValueError("detection_noise_std_probability must be nonnegative.")
        if self.dead_time_s < 0.0:
            raise ValueError("dead_time_s must be nonnegative.")
        if not (0.0 <= self.contrast_floor <= self.base_contrast <= 1.0):
            raise ValueError("Require 0 <= contrast_floor <= base_contrast <= 1.")


@dataclass
class VibrationCompensationSpec:
    enabled: bool = True
    accelerometer_noise_density_mps2_per_sqrt_hz: float = 1.0e-7
    accelerometer_bias_mps2: float = 0.0
    accelerometer_bias_random_walk_mps2_per_sqrt_s: float = 1.0e-8
    accelerometer_scale_factor_error_ppm: float = 40.0
    accelerometer_latency_s: float = 0.015
    accelerometer_alignment_error_rad: float = 1.0e-3
    residual_correction_fraction: float = 0.96
    accelerometer_bandwidth_hz: float = 30.0
    tilt_stage_residual_rad: float = 0.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.accelerometer_noise_density_mps2_per_sqrt_hz = float(
            self.accelerometer_noise_density_mps2_per_sqrt_hz
        )
        self.accelerometer_bias_mps2 = float(self.accelerometer_bias_mps2)
        self.accelerometer_bias_random_walk_mps2_per_sqrt_s = float(
            self.accelerometer_bias_random_walk_mps2_per_sqrt_s
        )
        self.accelerometer_scale_factor_error_ppm = float(self.accelerometer_scale_factor_error_ppm)
        self.accelerometer_latency_s = float(self.accelerometer_latency_s)
        self.accelerometer_alignment_error_rad = float(self.accelerometer_alignment_error_rad)
        self.residual_correction_fraction = float(self.residual_correction_fraction)
        self.accelerometer_bandwidth_hz = float(self.accelerometer_bandwidth_hz)
        self.tilt_stage_residual_rad = float(self.tilt_stage_residual_rad)
        if self.accelerometer_noise_density_mps2_per_sqrt_hz < 0.0:
            raise ValueError("accelerometer_noise_density_mps2_per_sqrt_hz must be nonnegative.")
        if self.accelerometer_bias_random_walk_mps2_per_sqrt_s < 0.0:
            raise ValueError("accelerometer_bias_random_walk_mps2_per_sqrt_s must be nonnegative.")
        if self.accelerometer_latency_s < 0.0:
            raise ValueError("accelerometer_latency_s must be nonnegative.")
        if self.accelerometer_bandwidth_hz <= 0.0:
            raise ValueError("accelerometer_bandwidth_hz must be positive.")
        if not (0.0 <= self.residual_correction_fraction <= 1.0):
            raise ValueError("residual_correction_fraction must lie in [0, 1].")


@dataclass
class PhotonicSystematicsSpec:
    gravity_gradient_zz_per_s2: float = 0.0
    use_normal_vertical_gradient: bool = True
    body_rotation_residual_gain: float = 0.02
    wavefront_aberration_coeff_rad_per_m2: float = 0.0
    quadratic_zeeman_coeff_rad_per_t2: float = 5.0e4
    magnetic_field_bias_t: float = 5.0e-5
    ac_stark_coeff_rad_per_fraction: float = 0.0
    relative_intensity_error: float = 0.0
    chirp_error_fraction: float = 0.0
    finite_pulse_coefficient: float = float(4.0 / np.pi - 1.0)
    readout_phase_bias_rad: float = 0.0
    valid_tilt_limit_deg: float = 3.3
    contrast_tilt_scale_deg: float = 3.3
    contrast_rotation_scale_radps: float = 0.04
    contrast_residual_accel_scale_mps2: float = 0.02
    min_valid_contrast: float = 0.2

    def __post_init__(self) -> None:
        self.gravity_gradient_zz_per_s2 = float(self.gravity_gradient_zz_per_s2)
        self.use_normal_vertical_gradient = bool(self.use_normal_vertical_gradient)
        self.body_rotation_residual_gain = float(self.body_rotation_residual_gain)
        self.wavefront_aberration_coeff_rad_per_m2 = float(self.wavefront_aberration_coeff_rad_per_m2)
        self.quadratic_zeeman_coeff_rad_per_t2 = float(self.quadratic_zeeman_coeff_rad_per_t2)
        self.magnetic_field_bias_t = float(self.magnetic_field_bias_t)
        self.ac_stark_coeff_rad_per_fraction = float(self.ac_stark_coeff_rad_per_fraction)
        self.relative_intensity_error = float(self.relative_intensity_error)
        self.chirp_error_fraction = float(self.chirp_error_fraction)
        self.finite_pulse_coefficient = float(self.finite_pulse_coefficient)
        self.readout_phase_bias_rad = float(self.readout_phase_bias_rad)
        self.valid_tilt_limit_deg = float(self.valid_tilt_limit_deg)
        self.contrast_tilt_scale_deg = float(self.contrast_tilt_scale_deg)
        self.contrast_rotation_scale_radps = float(self.contrast_rotation_scale_radps)
        self.contrast_residual_accel_scale_mps2 = float(self.contrast_residual_accel_scale_mps2)
        self.min_valid_contrast = float(self.min_valid_contrast)
        if self.valid_tilt_limit_deg <= 0.0:
            raise ValueError("valid_tilt_limit_deg must be positive.")
        if self.contrast_tilt_scale_deg <= 0.0:
            raise ValueError("contrast_tilt_scale_deg must be positive.")
        if self.contrast_rotation_scale_radps <= 0.0:
            raise ValueError("contrast_rotation_scale_radps must be positive.")
        if self.contrast_residual_accel_scale_mps2 <= 0.0:
            raise ValueError("contrast_residual_accel_scale_mps2 must be positive.")
        if not (0.0 <= self.min_valid_contrast <= 1.0):
            raise ValueError("min_valid_contrast must lie in [0, 1].")


@dataclass
class PhotonicGravimeterTelemetry:
    gravity_phase_rad: float
    vibration_true_phase_rad: float
    vibration_compensated_phase_rad: float
    vibration_residual_phase_rad: float
    gradient_phase_rad: float
    rotation_phase_rad: float
    wavefront_phase_rad: float
    zeeman_phase_rad: float
    lightshift_phase_rad: float
    chirp_phase_rad: float
    finite_pulse_scale_factor: float
    detection_phase_rad: float
    readout_bias_phase_rad: float
    total_phase_rad: float
    inferred_phase_rad: float
    transition_probability: float
    fringe_contrast: float
    axis_alignment_cosine: float
    tilt_deg: float
    recommended_tilt_exceeded: bool
    effective_scale_factor_rad_per_mps2: float
    estimated_measurement_std_mps2: float
    absolute_gravity_estimate_mps2: float
    disturbance_estimate_mps2: float
    validity_reason: str


@dataclass
class PhotonicTelemetrySummary:
    sample_count: int
    valid_sample_fraction: float
    rejection_reason_counts: dict[str, int]
    dominant_rejection_reason: Optional[str]
    median_fringe_contrast: float
    p95_fringe_contrast: float
    rms_vibration_residual_phase_rad: float
    rms_disturbance_residual_mps2: float
    tilt_exceedance_fraction: float
    median_estimated_measurement_variance_mps4: float


@dataclass
class PhotonicGravimeterSpec:
    """
    Hardware-tied photonic gravimeter configuration.

    The legacy flat fields are retained for compatibility and surrogate A/B
    comparisons. The nested blocks define the digital-twin physics model.
    """

    name: str = "photonic_gravimeter"
    operating_mode: str = PhotonicOperatingMode.MISSION.value
    physics_model: str = PhotonicPhysicsModel.DIGITAL_TWIN.value

    # Legacy / navigation-facing output terms retained for compatibility.
    noise_density_mps2_per_sqrt_hz: float = 4.0e-6
    bias_random_walk_mps2_per_sqrt_s: float = 2.0e-8
    turn_on_bias_std_mps2: float = 1.0e-6
    fixed_bias_mps2: float = 0.0
    scale_factor_error_ppm: float = 0.0
    bandwidth_hz: float = 0.4
    specific_force_coupling_b: ArrayLike | float = (1.0e-6, 1.0e-6, 1.0e-5)
    specific_force_reference_b_mps2: ArrayLike | float = (0.0, 0.0, -9.80665)
    angular_rate_coupling_b_mps2_per_radps: ArrayLike | float = (1.5e-6, 1.5e-6, 1.5e-5)
    supports_absolute_mode: bool = True
    update_period_s: float = 1.0
    warmup_time_s: float = 45.0
    valid_duty_cycle: float = 1.0
    degraded_noise_factor: float = 2.5
    degraded_bias_factor: float = 2.0
    degraded_motion_factor: float = 2.0
    max_abs_mps2: float = np.inf

    raman_axis_body_b: ArrayLike | float = (0.0, 0.0, 1.0)
    interferometer: InterferometerSequenceSpec = field(default_factory=InterferometerSequenceSpec)
    atom_ensemble: AtomEnsembleSpec = field(default_factory=AtomEnsembleSpec)
    vibration_compensation: VibrationCompensationSpec = field(default_factory=VibrationCompensationSpec)
    systematics: PhotonicSystematicsSpec = field(default_factory=PhotonicSystematicsSpec)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.operating_mode = str(self.operating_mode).strip().lower()
        self.physics_model = str(self.physics_model).strip().lower()
        valid_modes = {mode.value for mode in PhotonicOperatingMode}
        valid_models = {model.value for model in PhotonicPhysicsModel}
        if self.operating_mode not in valid_modes:
            raise ValueError(
                f"operating_mode must be one of {sorted(valid_modes)}, got {self.operating_mode!r}."
            )
        if self.physics_model not in valid_models:
            raise ValueError(
                f"physics_model must be one of {sorted(valid_models)}, got {self.physics_model!r}."
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

        self.specific_force_coupling_b = _scalar_or_vec3_to_vec3(
            self.specific_force_coupling_b,
            name="specific_force_coupling_b",
        )
        self.specific_force_reference_b_mps2 = _scalar_or_vec3_to_vec3(
            self.specific_force_reference_b_mps2,
            name="specific_force_reference_b_mps2",
        )
        self.angular_rate_coupling_b_mps2_per_radps = _scalar_or_vec3_to_vec3(
            self.angular_rate_coupling_b_mps2_per_radps,
            name="angular_rate_coupling_b_mps2_per_radps",
        )
        self.raman_axis_body_b = _normalize(self.raman_axis_body_b, name="raman_axis_body_b")

        self.interferometer = _coerce_dataclass(
            self.interferometer,
            InterferometerSequenceSpec,
            name="interferometer",
        )
        self.atom_ensemble = _coerce_dataclass(
            self.atom_ensemble,
            AtomEnsembleSpec,
            name="atom_ensemble",
        )
        self.vibration_compensation = _coerce_dataclass(
            self.vibration_compensation,
            VibrationCompensationSpec,
            name="vibration_compensation",
        )
        self.systematics = _coerce_dataclass(
            self.systematics,
            PhotonicSystematicsSpec,
            name="systematics",
        )

    @property
    def scale_factor(self) -> float:
        return 1.0 + self.scale_factor_error_ppm * 1.0e-6

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
            name=f"{self.name}_{self.operating_mode}_{self.physics_model}",
            supports_absolute_mode=self.supports_absolute_mode,
        )


@dataclass
class PhotonicGravimeterMeasurement(GravimeterMeasurement):
    operating_mode: str = PhotonicOperatingMode.MISSION.value
    physics_model: str = PhotonicPhysicsModel.DIGITAL_TWIN.value
    warmup_complete: bool = True
    cadence_emitted: bool = True
    effective_noise_std_mps2: float = 0.0
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    absolute_gravity_mps2: float = float("nan")
    telemetry: Optional[PhotonicGravimeterTelemetry] = None


class PhotonicGravimeterSensor:
    """Photonic gravimeter with surrogate and phase-domain digital-twin modes."""

    def __init__(
        self,
        spec: PhotonicGravimeterSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self._surrogate_inner = ScalarGravimeterSensor(
            self.spec.effective_gravimeter_spec(),
            rng=self.rng,
        )
        self._last_sample_time_s: Optional[float] = None
        self._measurement_bias_state = GravimeterBiasState(bias_mps2=float(self.spec.fixed_bias_mps2))
        self._compensation_bias_mps2 = float(self.spec.vibration_compensation.accelerometer_bias_mps2)
        self._output_filter_state_mps2: Optional[float] = None
        self._accel_history_time_s: list[float] = []
        self._accel_history_value_mps2: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._surrogate_inner = ScalarGravimeterSensor(
            self.spec.effective_gravimeter_spec(),
            rng=self.rng,
        )
        self._last_sample_time_s = None
        bias = float(self.spec.fixed_bias_mps2)
        if self.spec.turn_on_bias_std_mps2 > 0.0:
            bias += self.spec.turn_on_bias_std_mps2 * float(self.rng.standard_normal())
        self._measurement_bias_state = GravimeterBiasState(bias_mps2=bias)
        self._compensation_bias_mps2 = float(self.spec.vibration_compensation.accelerometer_bias_mps2)
        self._output_filter_state_mps2 = None
        self._accel_history_time_s.clear()
        self._accel_history_value_mps2.clear()

    def current_bias_state(self):
        if self.spec.physics_model == PhotonicPhysicsModel.SURROGATE.value:
            return self._surrogate_inner.current_bias_state()
        return GravimeterBiasState(bias_mps2=float(self._measurement_bias_state.bias_mps2))

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

    def _advance_output_bias_random_walk(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        sigma = self.spec.bias_random_walk_mps2_per_sqrt_s * np.sqrt(dt_s)
        if sigma > 0.0:
            self._measurement_bias_state.bias_mps2 += sigma * float(self.rng.standard_normal())

    def _advance_compensation_bias_random_walk(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        sigma = (
            self.spec.vibration_compensation.accelerometer_bias_random_walk_mps2_per_sqrt_s
            * np.sqrt(dt_s)
        )
        if sigma > 0.0:
            self._compensation_bias_mps2 += sigma * float(self.rng.standard_normal())

    def _current_time(self, *, dt_s: float, time_s: Optional[float]) -> float:
        if time_s is not None:
            return float(time_s)
        if self._last_sample_time_s is None:
            return 0.0
        return float(self._last_sample_time_s + dt_s)

    def _append_dynamic_accel_history(self, current_time_s: float, axis_dynamic_accel_mps2: float) -> None:
        self._accel_history_time_s.append(float(current_time_s))
        self._accel_history_value_mps2.append(float(axis_dynamic_accel_mps2))
        history_horizon_s = max(
            4.0 * self.spec.interferometer.interrogation_time_s
            + self.spec.vibration_compensation.accelerometer_latency_s
            + self.spec.update_period_s
            + self.spec.atom_ensemble.dead_time_s,
            2.0 * self.spec.update_period_s,
        )
        cutoff = current_time_s - history_horizon_s
        while len(self._accel_history_time_s) > 2 and self._accel_history_time_s[1] < cutoff:
            self._accel_history_time_s.pop(0)
            self._accel_history_value_mps2.pop(0)

    def _history_value_hold(self, query_time_s: float) -> float:
        if not self._accel_history_time_s:
            return 0.0
        if query_time_s <= self._accel_history_time_s[0]:
            return float(self._accel_history_value_mps2[0])
        for idx in range(len(self._accel_history_time_s) - 1, -1, -1):
            if self._accel_history_time_s[idx] <= query_time_s:
                return float(self._accel_history_value_mps2[idx])
        return float(self._accel_history_value_mps2[0])

    def _triangular_kernel_integral(
        self,
        *,
        current_time_s: float,
        signal_fn,
        num_points: int = 65,
    ) -> float:
        T = float(self.spec.interferometer.interrogation_time_s)
        if T <= 0.0:
            return 0.0
        if len(self._accel_history_time_s) <= 1:
            current_value = float(signal_fn(current_time_s))
            return float(current_value * T * T)

        ages = np.linspace(0.0, 2.0 * T, num=max(9, int(num_points)))
        query_times = current_time_s - ages
        signal = np.asarray([signal_fn(float(tq)) for tq in query_times], dtype=np.float64)
        weights = np.where(ages <= T, ages, 2.0 * T - ages)
        return float(np.trapezoid(weights * signal, ages))

    def _resolved_sequence(self) -> InterferometerSequenceSpec:
        seq = self.spec.interferometer
        if self.spec.operating_mode != PhotonicOperatingMode.DEGRADED.value:
            return seq
        return InterferometerSequenceSpec(
            atom_species=seq.atom_species,
            pulse_sequence=seq.pulse_sequence,
            effective_wavevector_rad_per_m=seq.effective_wavevector_rad_per_m,
            pulse_duration_s=seq.pulse_duration_s,
            interrogation_time_s=seq.interrogation_time_s,
            cycle_time_s=seq.cycle_time_s,
            chirp_rate_rad_per_s2=seq.chirp_rate_rad_per_s2,
            auto_chirp_from_normal_gravity=seq.auto_chirp_from_normal_gravity,
            lmt_order=seq.lmt_order,
            pulse_efficiency=max(0.7, seq.pulse_efficiency / self.spec.degraded_motion_factor),
            fringe_phase_bias_rad=seq.fringe_phase_bias_rad,
            fringe_offset_probability=seq.fringe_offset_probability,
        )

    def _resolved_atom_ensemble(self) -> AtomEnsembleSpec:
        atom = self.spec.atom_ensemble
        if self.spec.operating_mode == PhotonicOperatingMode.IDEALIZED.value:
            return AtomEnsembleSpec(
                temperature_k=atom.temperature_k,
                transverse_temperature_k=atom.transverse_temperature_k,
                cloud_radius_m=atom.cloud_radius_m,
                launch_position_axis_m=atom.launch_position_axis_m,
                launch_velocity_axis_mps=atom.launch_velocity_axis_mps,
                transverse_velocity_std_mps=atom.transverse_velocity_std_mps,
                detection_noise_std_probability=0.0,
                dead_time_s=0.0,
                base_contrast=1.0,
                contrast_floor=0.95,
            )
        if self.spec.operating_mode != PhotonicOperatingMode.DEGRADED.value:
            return atom
        return AtomEnsembleSpec(
            temperature_k=atom.temperature_k,
            transverse_temperature_k=atom.transverse_temperature_k,
            cloud_radius_m=atom.cloud_radius_m,
            launch_position_axis_m=atom.launch_position_axis_m,
            launch_velocity_axis_mps=atom.launch_velocity_axis_mps,
            transverse_velocity_std_mps=atom.transverse_velocity_std_mps,
            detection_noise_std_probability=atom.detection_noise_std_probability * self.spec.degraded_noise_factor,
            dead_time_s=atom.dead_time_s,
            base_contrast=max(atom.contrast_floor, atom.base_contrast / self.spec.degraded_motion_factor),
            contrast_floor=min(atom.base_contrast, atom.contrast_floor),
        )

    def _resolved_vibration_compensation(self) -> VibrationCompensationSpec:
        vib = self.spec.vibration_compensation
        if self.spec.operating_mode == PhotonicOperatingMode.IDEALIZED.value:
            return VibrationCompensationSpec(
                enabled=vib.enabled,
                accelerometer_noise_density_mps2_per_sqrt_hz=0.0,
                accelerometer_bias_mps2=0.0,
                accelerometer_bias_random_walk_mps2_per_sqrt_s=0.0,
                accelerometer_scale_factor_error_ppm=0.0,
                accelerometer_latency_s=0.0,
                accelerometer_alignment_error_rad=0.0,
                residual_correction_fraction=1.0,
                accelerometer_bandwidth_hz=vib.accelerometer_bandwidth_hz,
                tilt_stage_residual_rad=0.0,
            )
        if self.spec.operating_mode != PhotonicOperatingMode.DEGRADED.value:
            return vib
        return VibrationCompensationSpec(
            enabled=vib.enabled,
            accelerometer_noise_density_mps2_per_sqrt_hz=(
                vib.accelerometer_noise_density_mps2_per_sqrt_hz * self.spec.degraded_noise_factor
            ),
            accelerometer_bias_mps2=vib.accelerometer_bias_mps2,
            accelerometer_bias_random_walk_mps2_per_sqrt_s=(
                vib.accelerometer_bias_random_walk_mps2_per_sqrt_s * self.spec.degraded_bias_factor
            ),
            accelerometer_scale_factor_error_ppm=vib.accelerometer_scale_factor_error_ppm,
            accelerometer_latency_s=vib.accelerometer_latency_s,
            accelerometer_alignment_error_rad=(
                vib.accelerometer_alignment_error_rad * self.spec.degraded_motion_factor
            ),
            residual_correction_fraction=max(
                0.0,
                vib.residual_correction_fraction / self.spec.degraded_motion_factor,
            ),
            accelerometer_bandwidth_hz=vib.accelerometer_bandwidth_hz,
            tilt_stage_residual_rad=vib.tilt_stage_residual_rad * self.spec.degraded_motion_factor,
        )

    def _resolved_systematics(self) -> PhotonicSystematicsSpec:
        sys = self.spec.systematics
        if self.spec.operating_mode == PhotonicOperatingMode.IDEALIZED.value:
            return PhotonicSystematicsSpec(
                gravity_gradient_zz_per_s2=0.0,
                use_normal_vertical_gradient=False,
                body_rotation_residual_gain=0.0,
                wavefront_aberration_coeff_rad_per_m2=0.0,
                quadratic_zeeman_coeff_rad_per_t2=0.0,
                magnetic_field_bias_t=0.0,
                ac_stark_coeff_rad_per_fraction=0.0,
                relative_intensity_error=0.0,
                chirp_error_fraction=0.0,
                finite_pulse_coefficient=sys.finite_pulse_coefficient,
                readout_phase_bias_rad=0.0,
                valid_tilt_limit_deg=1.0e6,
                contrast_tilt_scale_deg=1.0e6,
                contrast_rotation_scale_radps=1.0e6,
                contrast_residual_accel_scale_mps2=1.0e6,
                min_valid_contrast=0.0,
            )
        if self.spec.operating_mode != PhotonicOperatingMode.DEGRADED.value:
            return sys
        return PhotonicSystematicsSpec(
            gravity_gradient_zz_per_s2=sys.gravity_gradient_zz_per_s2,
            use_normal_vertical_gradient=sys.use_normal_vertical_gradient,
            body_rotation_residual_gain=sys.body_rotation_residual_gain * self.spec.degraded_motion_factor,
            wavefront_aberration_coeff_rad_per_m2=(
                sys.wavefront_aberration_coeff_rad_per_m2 * self.spec.degraded_motion_factor
            ),
            quadratic_zeeman_coeff_rad_per_t2=sys.quadratic_zeeman_coeff_rad_per_t2,
            magnetic_field_bias_t=sys.magnetic_field_bias_t,
            ac_stark_coeff_rad_per_fraction=sys.ac_stark_coeff_rad_per_fraction,
            relative_intensity_error=sys.relative_intensity_error,
            chirp_error_fraction=sys.chirp_error_fraction * self.spec.degraded_motion_factor,
            finite_pulse_coefficient=sys.finite_pulse_coefficient,
            readout_phase_bias_rad=sys.readout_phase_bias_rad,
            valid_tilt_limit_deg=max(1.0, sys.valid_tilt_limit_deg / self.spec.degraded_motion_factor),
            contrast_tilt_scale_deg=max(
                1.0,
                sys.contrast_tilt_scale_deg / self.spec.degraded_motion_factor,
            ),
            contrast_rotation_scale_radps=max(
                1.0e-4,
                sys.contrast_rotation_scale_radps / self.spec.degraded_motion_factor,
            ),
            contrast_residual_accel_scale_mps2=max(
                1.0e-4,
                sys.contrast_residual_accel_scale_mps2 / self.spec.degraded_motion_factor,
            ),
            min_valid_contrast=min(1.0, max(sys.min_valid_contrast, 0.3)),
        )

    def _duty_cycle_accept(self) -> bool:
        if self.spec.operating_mode == PhotonicOperatingMode.IDEALIZED.value:
            return True
        return float(self.rng.random()) <= self.spec.valid_duty_cycle

    def _surrogate_measurement(
        self,
        *,
        specific_force_body_mps2: ArrayLike,
        C_n_b: ArrayLike,
        v_dot_ned_mps2: ArrayLike,
        v_ned_mps: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        body_angular_rate_b_radps: Optional[ArrayLike],
        time_s: Optional[float],
    ) -> PhotonicGravimeterMeasurement:
        warm = self._warmup_complete(time_s)
        emit = self._should_emit(time_s)
        valid = bool(warm and emit)

        base = self._surrogate_inner.measure_disturbance_from_specific_force_body(
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
        white_std = self._surrogate_inner.spec.white_noise_std(dt_s)
        if valid:
            self._last_sample_time_s = None if time_s is None else float(time_s)
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
                physics_model=self.spec.physics_model,
                warmup_complete=warm,
                cadence_emitted=emit,
                effective_noise_std_mps2=float(white_std),
                is_valid=True,
                rejection_reason=None,
                absolute_gravity_mps2=float(normal_gravity(lat_rad, height_m) + base.value_mps2),
                telemetry=None,
            )

        invalid_base = GravimeterMeasurement(
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
            kind=invalid_base.kind,
            time_s=invalid_base.time_s,
            value_mps2=invalid_base.value_mps2,
            ideal_value_mps2=invalid_base.ideal_value_mps2,
            motion_residual_mps2=invalid_base.motion_residual_mps2,
            filtered_input_mps2=invalid_base.filtered_input_mps2,
            bias_used_mps2=invalid_base.bias_used_mps2,
            white_noise_mps2=invalid_base.white_noise_mps2,
            saturated=invalid_base.saturated,
            operating_mode=self.spec.operating_mode,
            physics_model=self.spec.physics_model,
            warmup_complete=warm,
            cadence_emitted=emit,
            effective_noise_std_mps2=float(white_std),
            is_valid=False,
            rejection_reason="warmup" if not warm else "cadence",
            absolute_gravity_mps2=float("nan"),
            telemetry=None,
        )

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
        if self.spec.physics_model == PhotonicPhysicsModel.SURROGATE.value:
            return self._surrogate_measurement(
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

        return self._measure_digital_twin(
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

    def _measure_digital_twin(
        self,
        *,
        specific_force_body_mps2: ArrayLike,
        C_n_b: ArrayLike,
        v_dot_ned_mps2: ArrayLike,
        v_ned_mps: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        body_angular_rate_b_radps: Optional[ArrayLike],
        time_s: Optional[float],
    ) -> PhotonicGravimeterMeasurement:
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        current_time_s = self._current_time(dt_s=dt_s, time_s=time_s)
        warm = self._warmup_complete(current_time_s)
        emit = self._should_emit(current_time_s)
        duty = self._duty_cycle_accept()

        C_n_b_arr = project_to_so3(_as_float_array(C_n_b))
        f_b = _vec3(specific_force_body_mps2, name="specific_force_body_mps2")
        w_b = (
            np.zeros(3, dtype=np.float64)
            if body_angular_rate_b_radps is None
            else _vec3(body_angular_rate_b_radps, name="body_angular_rate_b_radps")
        )
        v_n = _vec3(v_ned_mps, name="v_ned_mps")

        seq = self._resolved_sequence()
        atom = self._resolved_atom_ensemble()
        vib = self._resolved_vibration_compensation()
        systematics = self._resolved_systematics()

        ideal_disturbance_mps2 = float(
            recover_gravity_disturbance_from_specific_force_body(
                specific_force_body_mps2=f_b,
                C_n_b=C_n_b_arr,
                v_dot_ned_mps2=v_dot_ned_mps2,
                v_ned_mps=v_n,
                lat_rad=lat_rad,
                height_m=height_m,
            )
        )
        normal_g_mps2 = float(normal_gravity(lat_rad, height_m))
        ideal_absolute_gravity_mps2 = float(normal_g_mps2 + ideal_disturbance_mps2)

        axis_b = self.spec.raman_axis_body_b
        axis_n = C_n_b_arr @ axis_b
        axis_alignment = float(np.clip(np.dot(axis_n, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(np.clip(axis_alignment, -1.0, 1.0))))
        recommended_tilt_exceeded = tilt_deg > systematics.valid_tilt_limit_deg

        auto_chirp_rate = seq.effective_wavevector_rad_per_m * normal_g_mps2
        chirp_rate = auto_chirp_rate if seq.auto_chirp_from_normal_gravity else seq.chirp_rate_rad_per_s2
        chirp_rate *= 1.0 + systematics.chirp_error_fraction

        static_specific_force_b_mps2 = C_n_b_arr.T @ np.array([0.0, 0.0, -normal_g_mps2], dtype=np.float64)
        axis_dynamic_accel_mps2 = float(
            np.dot(axis_b, f_b - static_specific_force_b_mps2)
        )
        self._append_dynamic_accel_history(current_time_s, axis_dynamic_accel_mps2)

        finite_pulse_scale = 1.0 + systematics.finite_pulse_coefficient * (
            seq.pulse_duration_s / max(seq.interrogation_time_s, 1.0e-9)
        )
        finite_pulse_scale = float(max(0.1, finite_pulse_scale))
        effective_scale_factor = (
            seq.effective_wavevector_rad_per_m
            * seq.lmt_order
            * seq.interrogation_time_s
            * seq.interrogation_time_s
            * finite_pulse_scale
        )

        gravity_phase_rad = float(effective_scale_factor * ideal_disturbance_mps2)
        chirp_phase_rad = float(
            seq.interrogation_time_s * seq.interrogation_time_s * (seq.effective_wavevector_rad_per_m * normal_g_mps2 - chirp_rate)
        )

        def true_vibration_signal(query_time_s: float) -> float:
            return float(self._history_value_hold(query_time_s))

        vibration_true_integral = self._triangular_kernel_integral(
            current_time_s=current_time_s,
            signal_fn=true_vibration_signal,
        )
        vibration_true_phase_rad = float(seq.effective_wavevector_rad_per_m * vibration_true_integral)

        compensation_alignment = np.cos(vib.accelerometer_alignment_error_rad)
        compensation_scale = 1.0 + vib.accelerometer_scale_factor_error_ppm * 1.0e-6
        compensation_latency_s = vib.accelerometer_latency_s

        def compensated_signal(query_time_s: float) -> float:
            delayed = self._history_value_hold(query_time_s - compensation_latency_s)
            lowpassed = _first_order_lowpass_step(
                previous=None,
                x=delayed,
                dt_s=max(dt_s, seq.interrogation_time_s / 8.0),
                cutoff_hz=vib.accelerometer_bandwidth_hz,
            )
            return float(compensation_alignment * compensation_scale * lowpassed + self._compensation_bias_mps2)

        compensation_integral = self._triangular_kernel_integral(
            current_time_s=current_time_s,
            signal_fn=compensated_signal,
        )
        compensation_noise_phase_std = 0.0
        if vib.enabled:
            compensation_noise_phase_std = float(
                seq.effective_wavevector_rad_per_m
                * vib.accelerometer_noise_density_mps2_per_sqrt_hz
                * np.sqrt(2.0 * seq.interrogation_time_s ** 3 / 3.0)
            )
        compensation_noise_phase_rad = compensation_noise_phase_std * float(self.rng.standard_normal())
        vibration_compensated_phase_rad = float(
            vib.residual_correction_fraction
            * (seq.effective_wavevector_rad_per_m * compensation_integral + compensation_noise_phase_rad)
        )
        vibration_residual_phase_rad = float(vibration_true_phase_rad - vibration_compensated_phase_rad)

        vertical_gradient_per_m = float(systematics.gravity_gradient_zz_per_s2)
        if systematics.use_normal_vertical_gradient:
            vertical_gradient_per_m += float(normal_gravity_vertical_gradient(lat_rad))
        gravity_gradient_term_m = (
            atom.launch_position_axis_m
            + atom.launch_velocity_axis_mps * seq.interrogation_time_s
            - (7.0 / 12.0) * normal_g_mps2 * seq.interrogation_time_s ** 2
        )
        gradient_phase_rad = float(
            seq.effective_wavevector_rad_per_m
            * vertical_gradient_per_m
            * seq.interrogation_time_s ** 2
            * gravity_gradient_term_m
        )

        transverse_velocity_std_mps = atom.transverse_velocity_std_mps
        if transverse_velocity_std_mps <= 0.0:
            transverse_velocity_std_mps = _temperature_to_velocity_std_mps(
                atom.transverse_temperature_k,
                _RB87_MASS_KG,
            )
        omega_total_n = systematics.body_rotation_residual_gain * (
            earth_rate_ned(lat_rad) + (C_n_b_arr @ w_b)
        )
        transverse_velocity_n = np.array([0.0, transverse_velocity_std_mps, 0.0], dtype=np.float64)
        rotation_accel_eq_mps2 = float(np.dot(axis_n, 2.0 * np.cross(omega_total_n, transverse_velocity_n)))
        rotation_phase_rad = float(effective_scale_factor * rotation_accel_eq_mps2)

        cloud_radius_eff_m = float(
            np.sqrt(atom.cloud_radius_m ** 2 + (transverse_velocity_std_mps * seq.interrogation_time_s) ** 2)
        )
        wavefront_phase_rad = float(
            systematics.wavefront_aberration_coeff_rad_per_m2 * cloud_radius_eff_m ** 2
        )
        zeeman_phase_rad = float(
            systematics.quadratic_zeeman_coeff_rad_per_t2 * systematics.magnetic_field_bias_t ** 2
        )
        lightshift_phase_rad = float(
            systematics.ac_stark_coeff_rad_per_fraction * systematics.relative_intensity_error
        )

        contrast_drop_argument = (
            (tilt_deg / systematics.contrast_tilt_scale_deg) ** 2
            + (float(np.linalg.norm(w_b)) / systematics.contrast_rotation_scale_radps) ** 2
            + (
                abs(vibration_residual_phase_rad / max(effective_scale_factor, 1.0e-12))
                / systematics.contrast_residual_accel_scale_mps2
            )
            ** 2
        )
        fringe_contrast = float(
            atom.contrast_floor + (atom.base_contrast - atom.contrast_floor) * np.exp(-0.5 * contrast_drop_argument)
        )
        fringe_contrast *= seq.pulse_efficiency
        fringe_contrast = float(np.clip(fringe_contrast, 0.0, 1.0))

        detection_phase_std_rad = 0.0
        if fringe_contrast > 1.0e-9 and atom.detection_noise_std_probability > 0.0:
            detection_phase_std_rad = float(2.0 * atom.detection_noise_std_probability / fringe_contrast)
        detection_phase_rad = detection_phase_std_rad * float(self.rng.standard_normal())

        deterministic_phase_rad = float(
            gravity_phase_rad
            + vibration_residual_phase_rad
            + gradient_phase_rad
            + rotation_phase_rad
            + wavefront_phase_rad
            + zeeman_phase_rad
            + lightshift_phase_rad
            + chirp_phase_rad
            + systematics.readout_phase_bias_rad
        )
        total_phase_rad = float(deterministic_phase_rad + detection_phase_rad)

        transition_probability = float(
            seq.fringe_offset_probability
            + 0.5 * fringe_contrast * np.cos(total_phase_rad + seq.fringe_phase_bias_rad)
        )
        transition_probability = float(np.clip(transition_probability, 0.0, 1.0))

        if fringe_contrast <= 1.0e-9:
            inferred_phase_rad = float("nan")
        else:
            normalized = (transition_probability - seq.fringe_offset_probability) / max(
                0.5 * fringe_contrast,
                1.0e-9,
            )
            normalized = float(np.clip(normalized, -1.0, 1.0))
            if np.isclose(np.mod(seq.fringe_phase_bias_rad, 2.0 * np.pi), np.pi / 2.0, atol=1.0e-6):
                inferred_phase_rad = _unwrap_midfringe_phase(
                    float(np.arcsin(-normalized)),
                    reference_phase_rad=deterministic_phase_rad,
                )
            else:
                inferred_phase_rad = float(np.arccos(normalized) - seq.fringe_phase_bias_rad)

        prefilter_measurement_mps2 = float(inferred_phase_rad / max(effective_scale_factor, 1.0e-12))
        filtered_measurement_mps2 = _first_order_lowpass_step(
            self._output_filter_state_mps2,
            prefilter_measurement_mps2,
            dt_s,
            self.spec.bandwidth_hz,
        )
        self._output_filter_state_mps2 = filtered_measurement_mps2

        white_noise_equivalent_mps2 = float(
            np.sqrt(compensation_noise_phase_std ** 2 + detection_phase_std_rad ** 2)
            / max(abs(effective_scale_factor), 1.0e-12)
        )
        estimated_measurement_std_mps2 = white_noise_equivalent_mps2
        deterministic_bias_equivalent_mps2 = float(
            (
                wavefront_phase_rad
                + zeeman_phase_rad
                + lightshift_phase_rad
                + chirp_phase_rad
                + systematics.readout_phase_bias_rad
            )
            / max(effective_scale_factor, 1.0e-12)
        )
        output_value_mps2 = float(
            np.clip(
                self.spec.scale_factor * filtered_measurement_mps2
                + self._measurement_bias_state.bias_mps2,
                -self.spec.max_abs_mps2,
                self.spec.max_abs_mps2,
            )
        )
        absolute_gravity_estimate_mps2 = float(normal_g_mps2 + output_value_mps2)

        valid = bool(warm and emit and duty)
        rejection_reason = None
        if not warm:
            rejection_reason = "warmup"
        elif not emit:
            rejection_reason = "cadence"
        elif not duty:
            rejection_reason = "duty_cycle"
        elif tilt_deg > systematics.valid_tilt_limit_deg:
            valid = False
            rejection_reason = "tilt_limit"
        elif fringe_contrast < systematics.min_valid_contrast:
            valid = False
            rejection_reason = "low_contrast"

        telemetry = PhotonicGravimeterTelemetry(
            gravity_phase_rad=gravity_phase_rad,
            vibration_true_phase_rad=vibration_true_phase_rad,
            vibration_compensated_phase_rad=vibration_compensated_phase_rad,
            vibration_residual_phase_rad=vibration_residual_phase_rad,
            gradient_phase_rad=gradient_phase_rad,
            rotation_phase_rad=rotation_phase_rad,
            wavefront_phase_rad=wavefront_phase_rad,
            zeeman_phase_rad=zeeman_phase_rad,
            lightshift_phase_rad=lightshift_phase_rad,
            chirp_phase_rad=chirp_phase_rad,
            finite_pulse_scale_factor=finite_pulse_scale,
            detection_phase_rad=detection_phase_rad,
            readout_bias_phase_rad=systematics.readout_phase_bias_rad,
            total_phase_rad=total_phase_rad,
            inferred_phase_rad=inferred_phase_rad,
            transition_probability=transition_probability,
            fringe_contrast=fringe_contrast,
            axis_alignment_cosine=axis_alignment,
            tilt_deg=tilt_deg,
            recommended_tilt_exceeded=recommended_tilt_exceeded,
            effective_scale_factor_rad_per_mps2=effective_scale_factor,
            estimated_measurement_std_mps2=estimated_measurement_std_mps2,
            absolute_gravity_estimate_mps2=absolute_gravity_estimate_mps2,
            disturbance_estimate_mps2=output_value_mps2,
            validity_reason="ok" if valid else (rejection_reason or "invalid"),
        )

        measurement_value_mps2 = output_value_mps2 if valid else float("nan")
        absolute_output_mps2 = absolute_gravity_estimate_mps2 if valid else float("nan")
        self._advance_output_bias_random_walk(dt_s)
        self._advance_compensation_bias_random_walk(dt_s)
        if valid:
            self._last_sample_time_s = current_time_s

        return PhotonicGravimeterMeasurement(
            kind="disturbance",
            time_s=current_time_s,
            value_mps2=measurement_value_mps2,
            ideal_value_mps2=ideal_disturbance_mps2,
            motion_residual_mps2=float(vibration_residual_phase_rad / max(effective_scale_factor, 1.0e-12)),
            filtered_input_mps2=filtered_measurement_mps2,
            bias_used_mps2=float(self._measurement_bias_state.bias_mps2 + deterministic_bias_equivalent_mps2),
            white_noise_mps2=float(detection_phase_rad / max(effective_scale_factor, 1.0e-12)),
            saturated=bool(
                valid
                and np.isfinite(measurement_value_mps2)
                and np.isfinite(self.spec.max_abs_mps2)
                and np.isclose(abs(output_value_mps2), self.spec.max_abs_mps2)
            ),
            operating_mode=self.spec.operating_mode,
            physics_model=self.spec.physics_model,
            warmup_complete=warm,
            cadence_emitted=emit,
            effective_noise_std_mps2=estimated_measurement_std_mps2,
            is_valid=valid,
            rejection_reason=rejection_reason,
            absolute_gravity_mps2=absolute_output_mps2,
            telemetry=telemetry,
        )


def _measurement_field(sample: PhotonicGravimeterMeasurement | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _telemetry_field(
    sample: PhotonicGravimeterMeasurement | Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    telemetry = _measurement_field(sample, "telemetry", None)
    if telemetry is None:
        return default
    if isinstance(telemetry, Mapping):
        return telemetry.get(name, default)
    return getattr(telemetry, name, default)


def summarize_photonic_measurements(
    samples: Sequence[PhotonicGravimeterMeasurement | Mapping[str, Any]],
) -> PhotonicTelemetrySummary:
    """
    Aggregate photonic measurement telemetry into one run-level summary.

    The input may be the in-memory dataclass samples emitted by the runner or
    JSON-like dictionaries loaded back from an archive.
    """
    sample_count = len(samples)
    if sample_count == 0:
        return PhotonicTelemetrySummary(
            sample_count=0,
            valid_sample_fraction=float("nan"),
            rejection_reason_counts={},
            dominant_rejection_reason=None,
            median_fringe_contrast=float("nan"),
            p95_fringe_contrast=float("nan"),
            rms_vibration_residual_phase_rad=float("nan"),
            rms_disturbance_residual_mps2=float("nan"),
            tilt_exceedance_fraction=float("nan"),
            median_estimated_measurement_variance_mps4=float("nan"),
        )

    valid_mask = np.asarray(
        [bool(_measurement_field(sample, "is_valid", False)) for sample in samples],
        dtype=bool,
    )

    rejection_counts: dict[str, int] = {}
    for sample in samples:
        if bool(_measurement_field(sample, "is_valid", False)):
            continue
        reason = _measurement_field(sample, "rejection_reason", None)
        if reason is None:
            reason = _telemetry_field(sample, "validity_reason", None)
        reason_key = "unknown" if reason is None else str(reason)
        rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1

    dominant_rejection_reason = None
    if rejection_counts:
        dominant_rejection_reason = max(
            sorted(rejection_counts),
            key=lambda key: rejection_counts[key],
        )

    fringe_contrast = np.asarray(
        [
            float(_telemetry_field(sample, "fringe_contrast", np.nan))
            for sample in samples
        ],
        dtype=np.float64,
    )
    vibration_residual_phase_rad = np.asarray(
        [
            float(_telemetry_field(sample, "vibration_residual_phase_rad", np.nan))
            for sample in samples
        ],
        dtype=np.float64,
    )
    disturbance_residual_mps2 = np.asarray(
        [
            float(_measurement_field(sample, "motion_residual_mps2", np.nan))
            for sample in samples
        ],
        dtype=np.float64,
    )
    tilt_exceeded = np.asarray(
        [
            bool(_telemetry_field(sample, "recommended_tilt_exceeded", False))
            for sample in samples
        ],
        dtype=bool,
    )
    measurement_std_mps2 = np.asarray(
        [
            float(_telemetry_field(sample, "estimated_measurement_std_mps2", np.nan))
            for sample in samples
        ],
        dtype=np.float64,
    )

    finite_contrast = fringe_contrast[np.isfinite(fringe_contrast)]
    finite_vibration = vibration_residual_phase_rad[np.isfinite(vibration_residual_phase_rad)]
    finite_disturbance = disturbance_residual_mps2[np.isfinite(disturbance_residual_mps2)]
    finite_variance = (measurement_std_mps2[np.isfinite(measurement_std_mps2)]) ** 2

    return PhotonicTelemetrySummary(
        sample_count=sample_count,
        valid_sample_fraction=float(np.mean(valid_mask)),
        rejection_reason_counts=rejection_counts,
        dominant_rejection_reason=dominant_rejection_reason,
        median_fringe_contrast=(
            float(np.median(finite_contrast)) if finite_contrast.size > 0 else float("nan")
        ),
        p95_fringe_contrast=(
            float(np.percentile(finite_contrast, 95.0))
            if finite_contrast.size > 0
            else float("nan")
        ),
        rms_vibration_residual_phase_rad=(
            float(np.sqrt(np.mean(finite_vibration**2)))
            if finite_vibration.size > 0
            else float("nan")
        ),
        rms_disturbance_residual_mps2=(
            float(np.sqrt(np.mean(finite_disturbance**2)))
            if finite_disturbance.size > 0
            else float("nan")
        ),
        tilt_exceedance_fraction=float(np.mean(tilt_exceeded)),
        median_estimated_measurement_variance_mps4=(
            float(np.median(finite_variance)) if finite_variance.size > 0 else float("nan")
        ),
    )


__all__ = [
    "AtomEnsembleSpec",
    "InterferometerSequenceSpec",
    "PhotonicGravimeterMeasurement",
    "PhotonicGravimeterSensor",
    "PhotonicGravimeterSpec",
    "PhotonicGravimeterTelemetry",
    "PhotonicTelemetrySummary",
    "PhotonicOperatingMode",
    "PhotonicPhysicsModel",
    "PhotonicSystematicsSpec",
    "VibrationCompensationSpec",
    "summarize_photonic_measurements",
]
