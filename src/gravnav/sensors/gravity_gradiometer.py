"""
Gravity gradiometer sensor model (horizontal, 2-axis).

This module implements a pragmatic 2-axis horizontal gradiometer that measures
the horizontal gradient of the gravity disturbance field in local NED:

    Gamma_N = d(delta_g) / dN     [units: m/s^2 per m, i.e. 1/s^2]
    Gamma_E = d(delta_g) / dE     [units: m/s^2 per m, i.e. 1/s^2]

The "gradiometer" name is used loosely here: a real quantum / photonic
gradiometer measures the full 3x3 gravity gradient tensor T_ij = d g_i / d x_j
in Eotvos units (1 E = 1e-9 1/s^2). In this simulator we are specifically
interested in the *horizontal* gradient of the scalar disturbance field
because that is what breaks the ridge-shaped likelihood of scalar gravity
map matching. Extending to the full tensor is a drop-in change once the
fusion and PF plumbing is validated.

Noise model
-----------
    y_k = [Gamma_N_true + bias_N + n_N,
           Gamma_E_true + bias_E + n_E]

with:
    n ~ N(0, sigma^2 I_2)
    bias = fixed_bias + turn_on_bias + random_walk
    sigma = noise_density_mps2_per_m_per_sqrt_hz / sqrt(dt)

This follows the same architectural pattern as `ScalarGravimeterSensor`:
per-axis noise density, bias random walk, turn-on bias, and deterministic
fixed bias. Motion coupling (residuals from body angular rate / specific
force) is deliberately omitted from the first version; gradiometers are
typically less motion-sensitive than scalar gravimeters because
common-mode accelerations cancel in the gradient difference. Add it if
benchtop data suggests it's needed.

Unit notes
----------
- Gradient units are m/s^2 per m = 1/s^2.
- 1 Eotvos (E) = 1e-9 1/s^2.
- Typical marine gradiometer white-noise performance is on the order of
  1-10 E / sqrt(Hz), i.e. 1e-9 to 1e-8 (1/s^2)/sqrt(Hz).
- Horizontal gravity disturbance gradients over the ocean vary from a
  few Eotvos to tens of Eotvos over horizontal scales of tens of km.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


__all__ = [
    "GravityGradiometerSpec",
    "GravityGradiometerBiasState",
    "GravityGradiometerMeasurement",
    "GravityGradiometerSensor",
    "EOTVOS_PER_INVERSE_S2",
    "INVERSE_S2_PER_EOTVOS",
]


# Unit conversion: 1 Eotvos = 1e-9 1/s^2
INVERSE_S2_PER_EOTVOS: float = 1.0e-9
EOTVOS_PER_INVERSE_S2: float = 1.0e9


def _as_vec2(x: ArrayLike | float, *, name: str) -> FloatArray:
    """Coerce scalar or length-2 to a length-2 float array."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.array([float(arr[0]), float(arr[0])], dtype=np.float64)
    if arr.size != 2:
        raise ValueError(
            f"{name} must be a scalar or length-2 array, got shape {arr.shape}"
        )
    return arr.astype(np.float64)


@dataclass
class GravityGradiometerSpec:
    """
    Horizontal gravity gradiometer specification.

    Parameters
    ----------
    noise_density_per_s2_per_sqrt_hz : scalar or length-2, default=1.0e-9
        White-noise density per horizontal axis in units of
        (1/s^2) / sqrt(Hz). Default is 1 Eotvos / sqrt(Hz), which is an
        aggressive but plausible quantum-gradiometer target.
    bias_random_walk_per_s2_per_sqrt_s : scalar or length-2, default=0.0
        In-run bias random-walk coefficient per axis in (1/s^2) / sqrt(s).
    turn_on_bias_std_per_s2 : scalar or length-2, default=0.0
        1-sigma turn-on bias uncertainty per axis in 1/s^2.
    fixed_bias_per_s2 : scalar or length-2, default=0.0
        Deterministic fixed bias per axis in 1/s^2.
    scale_factor_error_ppm : scalar or length-2, default=0.0
        Scalar scale-factor error per axis in parts per million.
    max_abs_per_s2 : float, default=np.inf
        Saturation magnitude for each axis in 1/s^2.
    name : str, default="gravity_gradiometer"
        Human-readable identifier.
    supports_full_tensor : bool, default=False
        Descriptive flag. Always False in this first version — only the 2
        horizontal components are produced.
    """

    noise_density_per_s2_per_sqrt_hz: ArrayLike | float = 1.0e-9
    bias_random_walk_per_s2_per_sqrt_s: ArrayLike | float = 0.0
    turn_on_bias_std_per_s2: ArrayLike | float = 0.0
    fixed_bias_per_s2: ArrayLike | float = 0.0
    scale_factor_error_ppm: ArrayLike | float = 0.0
    max_abs_per_s2: float = np.inf
    name: str = "gravity_gradiometer"
    supports_full_tensor: bool = False

    def __post_init__(self) -> None:
        self.noise_density_per_s2_per_sqrt_hz = _as_vec2(
            self.noise_density_per_s2_per_sqrt_hz,
            name="noise_density_per_s2_per_sqrt_hz",
        )
        self.bias_random_walk_per_s2_per_sqrt_s = _as_vec2(
            self.bias_random_walk_per_s2_per_sqrt_s,
            name="bias_random_walk_per_s2_per_sqrt_s",
        )
        self.turn_on_bias_std_per_s2 = _as_vec2(
            self.turn_on_bias_std_per_s2,
            name="turn_on_bias_std_per_s2",
        )
        self.fixed_bias_per_s2 = _as_vec2(
            self.fixed_bias_per_s2,
            name="fixed_bias_per_s2",
        )
        self.scale_factor_error_ppm = _as_vec2(
            self.scale_factor_error_ppm,
            name="scale_factor_error_ppm",
        )
        if float(self.max_abs_per_s2) <= 0.0:
            raise ValueError("max_abs_per_s2 must be positive.")
        self.max_abs_per_s2 = float(self.max_abs_per_s2)
        self.name = str(self.name)
        self.supports_full_tensor = bool(self.supports_full_tensor)

    def white_noise_std_per_s2(self, dt_s: float) -> FloatArray:
        """
        Discrete-time per-axis white-noise standard deviation at step dt_s.

        sigma_discrete = rho / sqrt(dt)

        where rho is the continuous-time noise density in (1/s^2) / sqrt(Hz).
        """
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        rho = np.asarray(self.noise_density_per_s2_per_sqrt_hz, dtype=np.float64)
        return rho / np.sqrt(float(dt_s))


@dataclass
class GravityGradiometerBiasState:
    """In-run bias state for a horizontal gradiometer."""

    bias_per_s2: FloatArray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.bias_per_s2 = np.asarray(self.bias_per_s2, dtype=np.float64).reshape(-1)
        if self.bias_per_s2.size != 2:
            raise ValueError("bias_per_s2 must have length 2.")


@dataclass
class GravityGradiometerMeasurement:
    """
    One horizontal-gradient sample.

    Attributes
    ----------
    time_s : float or None
        Timestamp [s].
    value_per_s2 : np.ndarray, shape (2,)
        Final gradient output per horizontal axis [1/s^2]: [Gamma_N, Gamma_E].
    ideal_value_per_s2 : np.ndarray, shape (2,)
        Ideal gradient before bias / noise [1/s^2].
    bias_used_per_s2 : np.ndarray, shape (2,)
        Bias state applied to this sample.
    white_noise_per_s2 : np.ndarray, shape (2,)
        White-noise realization applied to this sample.
    saturated : np.ndarray, shape (2,), dtype=bool
        Per-axis saturation flag.
    """

    time_s: Optional[float]
    value_per_s2: FloatArray
    ideal_value_per_s2: FloatArray
    bias_used_per_s2: FloatArray
    white_noise_per_s2: FloatArray
    saturated: np.ndarray

    @property
    def value_eotvos(self) -> FloatArray:
        return self.value_per_s2 * EOTVOS_PER_INVERSE_S2

    @property
    def ideal_value_eotvos(self) -> FloatArray:
        return self.ideal_value_per_s2 * EOTVOS_PER_INVERSE_S2


class GravityGradiometerSensor:
    """
    Stateful 2-axis horizontal gradiometer simulator.

    Mirrors the design of `ScalarGravimeterSensor`: owns the bias state and
    RNG, and exposes a single `measure(...)` call that takes the ideal
    horizontal-gradient input and a time step and returns a
    `GravityGradiometerMeasurement`.
    """

    def __init__(
        self,
        spec: GravityGradiometerSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self.bias_state = GravityGradiometerBiasState(
            bias_per_s2=np.array(self.spec.fixed_bias_per_s2, dtype=np.float64)
        )
        self.reset()

    def reset(
        self,
        *,
        turn_on_bias_randomized: bool = True,
        bias_override_per_s2: Optional[ArrayLike] = None,
    ) -> None:
        if bias_override_per_s2 is not None:
            bias = np.asarray(bias_override_per_s2, dtype=np.float64).reshape(-1)
            if bias.size != 2:
                raise ValueError("bias_override_per_s2 must have length 2.")
        else:
            bias = np.array(self.spec.fixed_bias_per_s2, dtype=np.float64)
            if turn_on_bias_randomized:
                sigma = np.asarray(self.spec.turn_on_bias_std_per_s2, dtype=np.float64)
                bias = bias + sigma * self.rng.standard_normal(2)
        self.bias_state = GravityGradiometerBiasState(bias_per_s2=bias)

    def _step_bias_random_walk(self, dt_s: float) -> None:
        sigma = np.asarray(
            self.spec.bias_random_walk_per_s2_per_sqrt_s, dtype=np.float64
        )
        if np.all(sigma == 0.0):
            return
        step_std = sigma * np.sqrt(float(dt_s))
        self.bias_state.bias_per_s2 = (
            self.bias_state.bias_per_s2 + step_std * self.rng.standard_normal(2)
        )

    def measure(
        self,
        ideal_gradient_per_s2: ArrayLike,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
    ) -> GravityGradiometerMeasurement:
        """
        Sample the gradiometer.

        Parameters
        ----------
        ideal_gradient_per_s2 : array-like, shape (2,)
            Ideal horizontal gradient [Gamma_N, Gamma_E] at the observation
            point in 1/s^2.
        dt_s : float
            Sample interval [s].
        time_s : float, optional
            Timestamp [s].
        """
        ideal = np.asarray(ideal_gradient_per_s2, dtype=np.float64).reshape(-1)
        if ideal.size != 2:
            raise ValueError("ideal_gradient_per_s2 must have length 2.")

        # Bias evolves across the step
        self._step_bias_random_walk(dt_s)

        # Scale factor
        ppm = np.asarray(self.spec.scale_factor_error_ppm, dtype=np.float64)
        scale = 1.0 + ppm * 1.0e-6

        # White noise
        sigma = self.spec.white_noise_std_per_s2(dt_s)
        n = sigma * self.rng.standard_normal(2)

        # Apply model
        raw = scale * ideal + self.bias_state.bias_per_s2 + n

        # Saturation
        sat = np.abs(raw) > self.spec.max_abs_per_s2
        clipped = np.clip(raw, -self.spec.max_abs_per_s2, self.spec.max_abs_per_s2)

        return GravityGradiometerMeasurement(
            time_s=None if time_s is None else float(time_s),
            value_per_s2=clipped.astype(np.float64),
            ideal_value_per_s2=ideal.astype(np.float64),
            bias_used_per_s2=self.bias_state.bias_per_s2.copy(),
            white_noise_per_s2=n.astype(np.float64),
            saturated=sat.astype(bool),
        )
