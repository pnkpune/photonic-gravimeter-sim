"""
depth.py

Depth-sensor helpers and a simple scalar depth-aiding sensor model for the
gravity-aide

This module is intentionally modest in scope:
- convert between ellipsoidal height and signed depth relative to a reference
  surface
- provide simple hydrostatic pressure/depth conversion helpers
- simulate a scalar depth observation with bias, white noise, optional finite
  bandwidth, scale-factor error, and saturation

Why this file exists
--------------------
The repository already has:
- full truth trajectories carrying ellipsoidal height
- IMU and gravimeter sensor models
- a shared sensor base layer

What is still missing is a simple vertical aiding sensor for underwater or
near-surface scenarios. In the current project structure that role belongs to
`src/gravnav/sens:contentReference[oaicite:4]{index=4}

Conventions
-----------
- NED is the repository-wide local navigation frame.
- Down is positive in NED.
- Ellipsoidal height `h` is positive upward away from the reference ellipsoid.
- Signed depth here is defined relative to a chosen reference surface height
  `h_ref` as:

      d = h_ref - h

  so:
  - `d > 0` means below the reference surface
  - `d = 0` means on the reference surface
  - `d < 0` means above the reference surface

Primary references used here
----------------------------
1) NOAA National Ocean Service, "How does pressure change with ocean depth?"
   https://oceanservice.noaa.gov/facts/pressure.html

   Used for the physical statement that hydrostatic pressure increases with
   depth.

2) NASA Glenn, "Fluid Pressure"
   https://www.grc.nasa.gov/WWW/k-12/WindTunnel/Activities/fluid_pressure.html

   Used for the standard hydrostatic relation:

       p = p0 + rho g h

3) USGS / SEAWAT report discussion of variable-density flow
   https://pubs.usgs.gov/sir/2009/5028/pdf/sir2009-5028_dausman.pdf

   Used for the practical representative seawater density value:
       rho_seawater ≈ 1025 kg/m^3

Design notes
------------
- This is a navigation-facing aiding model, not a detailed commercial pressure
  transducer model.
- The default measurement is signed depth in metres, because that is directly
  useful to the later navigation filter.
- Hydrostatic pressure helpers are included so that later configs / tests can
  still reason about pressure-based sensors cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import (
    SensorSpecBase,
    StatefulSensorBase,
    clip_scalar,
    discrete_random_walk_step_std,
    discrete_white_noise_std_from_density,
    first_order_lpf_step,
    scalar,
)

FloatArray = NDArray[np.float64]


def scalar_white_noise_std_from_density(
    noise_density_m_per_sqrt_hz: float,
    dt_s: float,
) -> float:
    r"""
    Convert scalar white-noise density into discrete-time sample standard deviation.

    Parameters
    ----------
    noise_density_m_per_sqrt_hz : float
        White-noise density in m/sqrt(Hz).
    dt_s : float
        Sample interval [s].

    Returns
    -------
    float
        Per-sample standard deviation in metres.

    Formula
    -------
        sigma_d = sigma / sqrt(dt)

    Notes
    -----
    This scalar wrapper delegates to the generic helper in `sensors.base`.
    """
    sigma = discrete_white_noise_std_from_density(
        noise_density_m_per_sqrt_hz,
        dt_s,
    )
    return float(sigma[0])


def scalar_random_walk_step_std(
    random_walk_m_per_sqrt_s: float,
    dt_s: float,
) -> float:
    r"""
    Convert a scalar random-walk coefficient into the standard deviation of the
    one-step bias increment.

    Parameters
    ----------
    random_walk_m_per_sqrt_s : float
        Bias random-walk coefficient in m/sqrt(s).
    dt_s : float
        Sample interval [s].

    Returns
    -------
    float
        Standard deviation of the one-step increment in metres.

    Formula
    -------
        sigma_step = sigma_rw * sqrt(dt)

    Notes
    -----
    This scalar wrapper delegates to the generic helper in `sensors.base`.
    """
    sigma = discrete_random_walk_step_std(
        random_walk_m_per_sqrt_s,
        dt_s,
    )
    return float(sigma[0])


def depth_from_height(
    height_m: float,
    *,
    reference_surface_height_m: float = 0.0,
) -> float:
    r"""
    Convert ellipsoidal height to signed depth relative to a reference surface.

    Parameters
    ----------
    height_m : float
        Ellipsoidal height [m].
    reference_surface_height_m : float, default=0.0
        Reference surface height [m].

    Returns
    -------
    float
        Signed depth [m].

    Formula
    -------
        d = h_ref - h

    Interpretation
    --------------
    - If `h < h_ref`, the result is positive (below the reference surface).
    - If `h > h_ref`, the result is negative (above the reference surface).

    Notes
    -----
    This sign convention is consistent with NED "Down positive" thinking while
    still starting from the repository's geodetic height state.
    """
    h = float(height_m)
    href = float(reference_surface_height_m)
    return href - h


def height_from_depth(
    depth_m: float,
    *,
    reference_surface_height_m: float = 0.0,
) -> float:
    r"""
    Convert signed depth relative to a reference surface back to ellipsoidal height.

    Parameters
    ----------
    depth_m : float
        Signed depth [m].
    reference_surface_height_m : float, default=0.0
        Reference surface height [m].

    Returns
    -------
    float
        Ellipsoidal height [m].

    Formula
    -------
    Rearranging:

        d = h_ref - h

    gives:

        h = h_ref - d
    """
    d = float(depth_m)
    href = float(reference_surface_height_m)
    return href - d


def hydrostatic_pressure_from_depth(
    depth_m: float,
    *,
    fluid_density_kgpm3: float = 1025.0,
    gravity_mps2: float = 9.80665,
    reference_pressure_pa: float = 101325.0,
) -> float:
    r"""
    Convert depth to absolute hydrostatic pressure using a constant-density fluid.

    Parameters
    ----------
    depth_m : float
        Signed depth [m].
    fluid_density_kgpm3 : float, default=1025.0
        Fluid density [kg/m^3]. Defaults to representative seawater density.
    gravity_mps2 : float, default=9.80665
        Gravitational acceleration [m/s^2].
    reference_pressure_pa : float, default=101325.0
        Pressure at zero depth [Pa], typically atmospheric pressure.

    Returns
    -------
    float
        Absolute pressure [Pa].

    Formula
    -------
    The standard hydrostatic relation is:

        p = p0 + rho g d

    where:
    - `p0` is reference pressure at the surface
    - `rho` is fluid density
    - `g` is gravitational acceleration
    - `d` is positive-down depth

    References
    ----------
    - NOAA notes that hydrostatic pressure increases with ocean depth.
    - NASA Glenn presents the standard hydrostatic pressure relation.
    """
    d = float(depth_m)
    rho = float(fluid_density_kgpm3)
    g = float(gravity_mps2)
    p0 = float(reference_pressure_pa)

    if rho <= 0.0:
        raise ValueError(f"fluid_density_kgpm3 must be positive, got {rho}.")
    if g <= 0.0:
        raise ValueError(f"gravity_mps2 must be positive, got {g}.")
    if p0 < 0.0:
        raise ValueError(f"reference_pressure_pa must be nonnegative, got {p0}.")

    return p0 + rho * g * d


def depth_from_hydrostatic_pressure(
    pressure_pa: float,
    *,
    fluid_density_kgpm3: float = 1025.0,
    gravity_mps2: float = 9.80665,
    reference_pressure_pa: float = 101325.0,
) -> float:
    r"""
    Convert absolute hydrostatic pressure to signed depth using a constant-density
    fluid model.

    Parameters
    ----------
    pressure_pa : float
        Absolute pressure [Pa].
    fluid_density_kgpm3 : float, default=1025.0
        Fluid density [kg/m^3].
    gravity_mps2 : float, default=9.80665
        Gravitational acceleration [m/s^2].
    reference_pressure_pa : float, default=101325.0
        Pressure at zero depth [Pa].

    Returns
    -------
    float
        Signed depth [m].

    Formula
    -------
    Inverting:

        p = p0 + rho g d

    gives:

        d = (p - p0) / (rho g)
    """
    p = float(pressure_pa)
    rho = float(fluid_density_kgpm3)
    g = float(gravity_mps2)
    p0 = float(reference_pressure_pa)

    if rho <= 0.0:
        raise ValueError(f"fluid_density_kgpm3 must be positive, got {rho}.")
    if g <= 0.0:
        raise ValueError(f"gravity_mps2 must be positive, got {g}.")
    return (p - p0) / (rho * g)


@dataclass
class DepthSensorSpec(SensorSpecBase):
    """
    Scalar depth-sensor specification.

    Parameters
    ----------
    noise_density_m_per_sqrt_hz : float, default=0.0
        White-noise density in m/sqrt(Hz).
    bias_random_walk_m_per_sqrt_s : float, default=0.0
        In-run bias random-walk coefficient in m/sqrt(s).
    turn_on_bias_std_m : float, default=0.0
        1-sigma turn-on bias uncertainty [m].
    fixed_bias_m : float, default=0.0
        Deterministic fixed bias [m].
    scale_factor_error_ppm : float, default=0.0
        Scalar scale-factor error in parts per million.
    max_abs_m : float, default=np.inf
        Saturation magnitude for the signed depth output.
    bandwidth_hz : float or None, default=None
        Optional first-order low-pass bandwidth.
    reference_surface_height_m : float, default=0.0
        Height of the reference surface used to define zero depth.
    fluid_density_kgpm3 : float, default=1025.0
        Reference fluid density used by the hydrostatic helper methods.
    reference_pressure_pa : float, default=101325.0
        Pressure corresponding to zero depth.
    gravity_mps2 : float, default=9.80665
        Reference gravity used by the hydrostatic helper methods.
    name : str, default="depth_sensor"
        Human-readable sensor identifier.

    Sensor model
    ------------
    The simulated signed-depth output is:

        y_k = s * LPF(d_k) + b_k + n_k

    where:
    - d_k : ideal signed depth
    - LPF : optional first-order low-pass response
    - s   : scale factor = 1 + ppm * 1e-6
    - b_k : current bias state (fixed + turn-on + random walk)
    - n_k : additive white noise

    Notes
    -----
    This is a navigation-facing aiding model rather than a detailed transducer
    physics model.
    """

    noise_density_m_per_sqrt_hz: float = 0.0
    bias_random_walk_m_per_sqrt_s: float = 0.0
    turn_on_bias_std_m: float = 0.0
    fixed_bias_m: float = 0.0
    scale_factor_error_ppm: float = 0.0
    max_abs_m: float = np.inf
    bandwidth_hz: Optional[float] = None

    reference_surface_height_m: float = 0.0
    fluid_density_kgpm3: float = 1025.0
    reference_pressure_pa: float = 101325.0
    gravity_mps2: float = 9.80665

    name: str = "depth_sensor"

    def __post_init__(self) -> None:
        self.noise_density_m_per_sqrt_hz = float(self.noise_density_m_per_sqrt_hz)
        self.bias_random_walk_m_per_sqrt_s = float(self.bias_random_walk_m_per_sqrt_s)
        self.turn_on_bias_std_m = float(self.turn_on_bias_std_m)
        self.fixed_bias_m = float(self.fixed_bias_m)
        self.scale_factor_error_ppm = float(self.scale_factor_error_ppm)
        self.max_abs_m = float(self.max_abs_m)
        self.reference_surface_height_m = float(self.reference_surface_height_m)
        self.fluid_density_kgpm3 = float(self.fluid_density_kgpm3)
        self.reference_pressure_pa = float(self.reference_pressure_pa)
        self.gravity_mps2 = float(self.gravity_mps2)

        if self.noise_density_m_per_sqrt_hz < 0.0:
            raise ValueError("noise_density_m_per_sqrt_hz must be nonnegative.")
        if self.bias_random_walk_m_per_sqrt_s < 0.0:
            raise ValueError("bias_random_walk_m_per_sqrt_s must be nonnegative.")
        if self.turn_on_bias_std_m < 0.0:
            raise ValueError("turn_on_bias_std_m must be nonnegative.")
        if self.max_abs_m <= 0.0 and not np.isinf(self.max_abs_m):
            raise ValueError("max_abs_m must be positive or inf.")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive when provided.")
        if self.fluid_density_kgpm3 <= 0.0:
            raise ValueError("fluid_density_kgpm3 must be positive.")
        if self.reference_pressure_pa < 0.0:
            raise ValueError("reference_pressure_pa must be nonnegative.")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive.")

    @classmethod
    def perfect(cls, name: str = "perfect_depth_sensor") -> "DepthSensorSpec":
        """
        Return a perfect depth-sensor specification.
        """
        return cls(name=name)

    @property
    def scale_factor(self) -> float:
        """
        Scalar multiplicative scale factor.

        Formula
        -------
            s = 1 + ppm * 1e-6
        """
        return 1.0 + self.scale_factor_error_ppm * 1.0e-6

    def white_noise_std(self, dt_s: float) -> float:
        """Per-sample white-noise standard deviation [m]."""
        return scalar_white_noise_std_from_density(
            self.noise_density_m_per_sqrt_hz,
            dt_s,
        )

    def bias_step_std(self, dt_s: float) -> float:
        """Per-step bias-random-walk standard deviation [m]."""
        return scalar_random_walk_step_std(
            self.bias_random_walk_m_per_sqrt_s,
            dt_s,
        )


@dataclass
class DepthBiasState:
    """
    State of the depth-sensor bias.

    Attributes
    ----------
    bias_m : float
        Current signed-depth bias [m].
    """
    bias_m: float


@dataclass
class DepthMeasurement:
    """
    One scalar depth-sensor sample.

    Attributes
    ----------
    time_s : float or None
        Optional timestamp [s].
    value_m : float
        Final signed-depth measurement [m].
    ideal_depth_m : float
        Ideal signed depth [m].
    filtered_depth_m : float
        Post-bandwidth filtered depth before scale, bias, and white noise [m].
    bias_used_m : float
        Bias applied to this sample [m].
    white_noise_m : float
        White-noise realization applied to this sample [m].
    saturated : bool
        True if clipping occurred.
    reference_surface_height_m : float
        Reference surface used to define zero depth [m].

    Interpretation
    --------------
    Positive values mean below the reference surface. Negative values mean above.
    """

    time_s: Optional[float]
    value_m: float
    ideal_depth_m: float
    filtered_depth_m: float
    bias_used_m: float
    white_noise_m: float
    saturated: bool
    reference_surface_height_m: float

    @property
    def value_pressure_pa(self) -> float:
        """
        A convenience view of the measured signed depth converted to hydrostatic
        pressure using the default seawater/gravity constants.

        Notes
        -----
        This property intentionally uses the simple reference constants defined in
        `DepthSensorSpec`'s defaults, not a dynamic equation of state.
        """
        return hydrostatic_pressure_from_depth(self.value_m)

    @property
    def ideal_pressure_pa(self) -> float:
        """
        Hydrostatic pressure corresponding to the ideal signed depth using the
        same simple default constants.
        """
        return hydrostatic_pressure_from_depth(self.ideal_depth_m)


class DepthSensor(StatefulSensorBase[DepthSensorSpec, DepthBiasState]):
    """
    Stateful scalar depth-sensor simulator.

    This class simulates a depth aiding measurement stream whose ideal input is
    signed depth relative to a chosen reference surface.

    State owned by this object
    --------------------------
    - specification (`DepthSensorSpec`)
    - RNG
    - current scalar bias state
    - optional first-order low-pass filter state

    Design philosophy
    -----------------
    Keep the model transparent:
    - if you want a perfect sensor, use `DepthSensorSpec.perfect()`
    - if you want a realistic one, add white noise, turn-on bias, drift, finite
      bandwidth, and saturation explicitly
    """

    def __init__(
        self,
        spec: DepthSensorSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(spec=spec, rng=rng)
        self.bias_state = DepthBiasState(bias_m=float(self.spec.fixed_bias_m))
        self._filter_state_m: Optional[float] = None
        self.reset()

    def reset(
        self,
        *,
        turn_on_bias_randomized: bool = True,
        bias_override_m: Optional[float] = None,
        filter_state_override_m: Optional[float] = None,
    ) -> None:
        """
        Reset the depth-sensor state.

        Parameters
        ----------
        turn_on_bias_randomized : bool, default=True
            If True, sample a turn-on bias realization.
        bias_override_m : float, optional
            If provided, use this as the complete initial bias state.
        filter_state_override_m : float, optional
            If provided, use this as the initial filter state.

        Notes
        -----
        The low-pass filter state defaults to "uninitialized" so that the first
        sample passes through without an artificial startup transient.
        """
        if bias_override_m is not None:
            bias = float(bias_override_m)
        else:
            bias = float(self.spec.fixed_bias_m)
            if turn_on_bias_randomized:
                bias += self.spec.turn_on_bias_std_m * float(self.rng.standard_normal())

        self.bias_state = DepthBiasState(bias_m=bias)
        self._filter_state_m = (
            None if filter_state_override_m is None else float(filter_state_override_m)
        )

    def current_bias_state(self) -> DepthBiasState:
        """
        Return a copy of the current bias state.
        """
        return DepthBiasState(bias_m=float(self.bias_state.bias_m))

    def _step_bias_random_walk(self, dt_s: float) -> None:
        """
        Evolve the scalar bias state by one step.

        Formula
        -------
            b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
            w_k ~ N(0, 1)
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")
        sigma_step = self.spec.bias_step_std(dt_s)
        self.bias_state.bias_m += sigma_step * float(self.rng.standard_normal())

    def _apply_bandwidth(self, x_m: float, dt_s: float) -> float:
        """
        Apply the optional first-order low-pass response to the scalar depth input.
        """
        y = first_order_lpf_step(
            previous_state=self._filter_state_m,
            input_value=float(x_m),
            dt_s=dt_s,
            cutoff_hz=self.spec.bandwidth_hz,
        )
        self._filter_state_m = float(y)
        return float(y)

    def _measure_depth_core(
        self,
        ideal_depth_m: float,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
        reference_surface_height_m: Optional[float] = None,
    ) -> DepthMeasurement:
        """
        Core scalar measurement routine.

        Parameters
        ----------
        ideal_depth_m : float
            Ideal signed depth [m].
        dt_s : float
            Sample interval [s].
        time_s : float, optional
            Optional timestamp [s].
        reference_surface_height_m : float, optional
            Optional per-call reference surface. If omitted, the spec's default
            reference surface is recorded.

        Returns
        -------
        DepthMeasurement
            Simulated depth sample.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        ideal = float(ideal_depth_m)
        filtered = self._apply_bandwidth(ideal, dt_s)

        white_std = self.spec.white_noise_std(dt_s)
        white_noise = white_std * float(self.rng.standard_normal())
        bias_used = float(self.bias_state.bias_m)

        value = self.spec.scale_factor * filtered + bias_used + white_noise
        clipped_value, saturated = clip_scalar(value, self.spec.max_abs_m)

        meas = DepthMeasurement(
            time_s=None if time_s is None else float(time_s),
            value_m=clipped_value,
            ideal_depth_m=ideal,
            filtered_depth_m=filtered,
            bias_used_m=bias_used,
            white_noise_m=white_noise,
            saturated=saturated,
            reference_surface_height_m=float(
                self.spec.reference_surface_height_m
                if reference_surface_height_m is None
                else reference_surface_height_m
            ),
        )

        self._step_bias_random_walk(dt_s)
        return meas

    def measure_depth(
        self,
        ideal_depth_m: float,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
        reference_surface_height_m: Optional[float] = None,
    ) -> DepthMeasurement:
        """
        Measure an ideal signed depth.

        Parameters
        ----------
        ideal_depth_m : float
            Ideal signed depth [m].
        dt_s : float
            Sample interval [s].
        time_s : float, optional
            Optional timestamp [s].
        reference_surface_height_m : float, optional
            Optional reference surface recorded into the output sample.

        Returns
        -------
        DepthMeasurement
            Simulated depth measurement.
        """
        return self._measure_depth_core(
            ideal_depth_m=float(ideal_depth_m),
            dt_s=dt_s,
            time_s=time_s,
            reference_surface_height_m=reference_surface_height_m,
        )

    def measure_depth_from_height(
        self,
        height_m: float,
        dt_s: float,
        *,
        reference_surface_height_m: Optional[float] = None,
        time_s: Optional[float] = None,
    ) -> DepthMeasurement:
        r"""
        Convenience wrapper:
            ellipsoidal height -> signed depth -> simulated measurement.

        Parameters
        ----------
        height_m : float
            Ellipsoidal height [m].
        dt_s : float
            Sample interval [s].
        reference_surface_height_m : float, optional
            Reference surface used to define depth. If omitted, the spec default
            is used.
        time_s : float, optional
            Optional timestamp [s].

        Returns
        -------
        DepthMeasurement
            Simulated signed-depth measurement.

        Formula
        -------
            d = h_ref - h
        """
        href = (
            self.spec.reference_surface_height_m
            if reference_surface_height_m is None
            else float(reference_surface_height_m)
        )
        ideal_depth = depth_from_height(
            height_m=float(height_m),
            reference_surface_height_m=float(href),
        )
        return self._measure_depth_core(
            ideal_depth_m=ideal_depth,
            dt_s=dt_s,
            time_s=time_s,
            reference_surface_height_m=float(href),
        )


__all__ = [
    "FloatArray",
    "DepthBiasState",
    "DepthMeasurement",
    "DepthSensor",
    "DepthSensorSpec",
    "depth_from_height",
    "depth_from_hydrostatic_pressure",
    "height_from_depth",
    "hydrostatic_pressure_from_depth",
    "scalar_random_walk_step_std",
    "scalar_white_noise_std_from_density",
]