"""
velocity_aid.py

Generic velocity-aiding sensor models and frame-conversion helpers for the
gravity-aided navigation simulator.

This module fills the gap between the already-implemented truth/IMU layers and
later estimator work. In the current repository state, the Earth, frames,
trajectory, IMU, gravimeter, and depth layers exist, while
`src/gravnav/sensors/velocity_aid.py` is still scaffold-only. A velocity-aid
module is the natural next sensor block because later error-state filters will
need a consistent way to simulate external velocity observations before full
map-matching and fusion are built.

What “velocity aid” means here
------------------------------
This file deliberately models a *generic* externally aided velocity sensor
rather than one particular commercial instrument. The same interface can be
used to represent:
- GNSS-like velocity updates in the local navigation frame (NED)
- DVL-like or speed-log-like velocity updates in the body frame
- other aiding sources that ultimately provide a 3-vector velocity observation

The core model is therefore just:
1) take an ideal 3D velocity vector in a chosen frame,
2) optionally apply finite bandwidth,
3) apply deterministic scale/misalignment,
4) add bias and white noise,
5) clip to sensor limits.

Conventions
-----------
- NED is the repository-wide local navigation frame:
      x = North, y = East, z = Down
- Body frame is right-handed and aviation-style:
      x = forward, y = right, z = down
- All velocities are in m/s.
- `C_n_b` is the passive body->NED DCM such that:

      v^n = C_n_b v^b

- Therefore:

      v^b = (C_n_b)^T v^n

Primary references used here
----------------------------
1) INSTINCT / University of Stuttgart,
   "INS/GNSS Loosely-coupled Kalman Filter (local-navigation frame)"
   URL:
   https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

   Used for:
   - the local-navigation-frame state convention in which velocity is naturally
     represented as an Earth-relative vector in the navigation frame
   - motivating the NED-velocity update pathway implemented in this file

2) Nortek DVL Operations and Integration Manual
   URL:
   https://assets.nortekgroup.com/software/N3015-006-DVLOperations.pdf

   Used for the practical point that a DVL measures velocity relative to the
   bottom (Earth) or relative to the water, making a body-frame velocity-aid
   abstraction directly useful for subsea/UUV scenarios.

3) Teledyne Marine DVL overview pages
   URLs:
   https://www.teledynemarine.com/products/product-line/navigation-positioning/doppler-velocity-logs
   https://www.teledynemarine.com/en-us/products/Pages/Pathfinder_DVL.aspx

   Used for the practical point that DVL outputs are commonly used as aiding
   inputs in DVL/INS or DVL/INS/EKF navigation solutions.

4) Kalibr Wiki, "IMU Noise Model"
   URL:
   https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

   Used for the same generic stochastic discretization already used elsewhere in
   this repository:

       sigma_d = sigma / sqrt(dt)
       b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k

   Even though Kalibr discusses IMUs, the mathematics is the same for any aided
   measurement channel modeled with additive white noise and bias random walk.

Design notes
------------
- This file intentionally does *not* decide whether the truth velocity is
  Earth-relative, bottom-relative, or water-relative. That choice belongs to the
  scenario/simulation layer. This sensor only corrupts the ideal velocity vector
  it is given.
- The module supports both NED-frame and body-frame measurements so later
  estimators can choose the update formulation they want.
- The measurement object exposes the filtered input, bias, and white-noise terms
  used for each sample, making Monte Carlo debugging much easier later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import (
    SensorSpecBase,
    StatefulSensorBase,
    axis3,
    clip_vector_per_axis,
    discrete_random_walk_step_std,
    discrete_white_noise_std_from_density,
    first_order_lpf_step,
    nonnegative_axis3,
    positive_or_inf_axis3,
    standard_normal3,
    vec3,
)
from ..physics.frames import project_to_so3

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level validation helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _mat3_or_identity(M: Optional[ArrayLike], *, name: str) -> FloatArray:
    """
    Return identity if `M is None`, else validate and project to SO(3).

    Parameters
    ----------
    M : array-like, shape (3, 3), optional
        Candidate matrix.
    name : str
        Name for error messages.
    """
    if M is None:
        return np.eye(3, dtype=np.float64)
    arr = _as_float_array(M)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return project_to_so3(arr)


# -----------------------------------------------------------------------------
# Generic stochastic scaling wrappers
# -----------------------------------------------------------------------------


def vector_white_noise_std_from_density(
    noise_density_mps_per_sqrt_hz: ArrayLike | float,
    dt_s: float,
) -> FloatArray:
    r"""
    Convert vector white-noise density into discrete-time sample standard deviation.

    Parameters
    ----------
    noise_density_mps_per_sqrt_hz : scalar or shape (3,)
        Continuous-time white-noise density in m/s/sqrt(Hz).
    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Per-sample white-noise standard deviation [m/s].

    Formula
    -------
        sigma_d = sigma / sqrt(dt)

    Notes
    -----
    This is the same engineering discretization already used elsewhere in the
    repository for the IMU, but here applied to a generic velocity-aid vector.
    """
    return discrete_white_noise_std_from_density(
        noise_density_mps_per_sqrt_hz,
        dt_s,
    )


def vector_random_walk_step_std(
    random_walk_mps_per_sqrt_s: ArrayLike | float,
    dt_s: float,
) -> FloatArray:
    r"""
    Convert a vector bias random-walk coefficient into the standard deviation of
    the one-step bias increment.

    Parameters
    ----------
    random_walk_mps_per_sqrt_s : scalar or shape (3,)
        Bias random-walk coefficient in m/s/sqrt(s).
    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        One-step bias increment standard deviation [m/s].

    Formula
    -------
        b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
        w_k ~ N(0, I)

    so:

        sigma_step = sigma_rw * sqrt(dt)
    """
    return discrete_random_walk_step_std(
        random_walk_mps_per_sqrt_s,
        dt_s,
    )


# -----------------------------------------------------------------------------
# Velocity and frame helpers
# -----------------------------------------------------------------------------


def velocity_body_from_ned(
    velocity_ned_mps: ArrayLike,
    C_n_b: ArrayLike,
) -> FloatArray:
    r"""
    Convert a velocity vector from NED coordinates to body coordinates.

    Parameters
    ----------
    velocity_ned_mps : array-like, shape (3,)
        Velocity resolved in NED [m/s].
    C_n_b : array-like, shape (3, 3)
        Passive DCM mapping body -> NED:

            v^n = C_n_b v^b

    Returns
    -------
    np.ndarray, shape (3,)
        Body-frame velocity [m/s].

    Formula
    -------
    Under the repository's passive-transform convention:

        v^b = C_b_n v^n = (C_n_b)^T v^n
    """
    v_n = vec3(velocity_ned_mps, name="velocity_ned_mps")
    C_n_b = _mat3_or_identity(C_n_b, name="C_n_b")
    return C_n_b.T @ v_n


def velocity_ned_from_body(
    velocity_body_mps: ArrayLike,
    C_n_b: ArrayLike,
) -> FloatArray:
    r"""
    Convert a velocity vector from body coordinates to NED coordinates.

    Parameters
    ----------
    velocity_body_mps : array-like, shape (3,)
        Velocity resolved in body coordinates [m/s].
    C_n_b : array-like, shape (3, 3)
        Passive DCM mapping body -> NED.

    Returns
    -------
    np.ndarray, shape (3,)
        NED-frame velocity [m/s].

    Formula
    -------
        v^n = C_n_b v^b
    """
    v_b = vec3(velocity_body_mps, name="velocity_body_mps")
    C_n_b = _mat3_or_identity(C_n_b, name="C_n_b")
    return C_n_b @ v_b


def horizontal_speed_from_velocity_ned(velocity_ned_mps: ArrayLike) -> float:
    r"""
    Return horizontal speed from a NED velocity vector.

    Parameters
    ----------
    velocity_ned_mps : array-like, shape (3,)
        Velocity resolved in NED [m/s].

    Returns
    -------
    float
        Horizontal speed [m/s].

    Formula
    -------
        v_h = sqrt(v_N^2 + v_E^2)
    """
    v_n = vec3(velocity_ned_mps, name="velocity_ned_mps")
    return float(np.linalg.norm(v_n[:2]))


def speed_magnitude_from_velocity(velocity_mps: ArrayLike) -> float:
    r"""
    Return full 3D speed magnitude from a 3-vector velocity.

    Parameters
    ----------
    velocity_mps : array-like, shape (3,)
        Velocity vector [m/s].

    Returns
    -------
    float
        Speed magnitude [m/s].

    Formula
    -------
        |v| = ||v||_2
    """
    v = vec3(velocity_mps, name="velocity_mps")
    return float(np.linalg.norm(v))


def course_over_ground_from_velocity_ned(
    velocity_ned_mps: ArrayLike,
    *,
    min_horizontal_speed_mps: float = 1.0e-9,
) -> float:
    r"""
    Compute course over ground from NED velocity.

    Parameters
    ----------
    velocity_ned_mps : array-like, shape (3,)
        Velocity resolved in NED [m/s].
    min_horizontal_speed_mps : float, default=1e-9
        Minimum horizontal speed below which course is undefined.

    Returns
    -------
    float
        Course over ground [rad], wrapped by `arctan2(v_E, v_N)`.

    Formula
    -------
        chi = atan2(v_E, v_N)

    Raises
    ------
    ValueError
        If horizontal speed is smaller than `min_horizontal_speed_mps`.
    """
    v_n = vec3(velocity_ned_mps, name="velocity_ned_mps")
    vh = float(np.linalg.norm(v_n[:2]))
    if vh < float(min_horizontal_speed_mps):
        raise ValueError(
            "Course over ground is undefined when horizontal speed is near zero."
        )
    return float(np.arctan2(v_n[1], v_n[0]))


@dataclass
class VelocityAidTruth:
    """
    Derived truth quantities useful for velocity-aid debugging.

    Attributes
    ----------
    velocity_ned_mps : np.ndarray, shape (3,)
        Earth-relative velocity resolved in NED.
    velocity_body_mps : np.ndarray, shape (3,)
        The same velocity resolved in the body frame.
    horizontal_speed_mps : float
        Horizontal speed magnitude.
    speed_mps : float
        Full 3D speed magnitude.
    course_over_ground_rad : float or None
        Course over ground when defined, else None.
    """

    velocity_ned_mps: FloatArray
    velocity_body_mps: FloatArray
    horizontal_speed_mps: float
    speed_mps: float
    course_over_ground_rad: Optional[float]


def build_velocity_aid_truth(
    velocity_ned_mps: ArrayLike,
    C_n_b: ArrayLike,
) -> VelocityAidTruth:
    """
    Build a convenient `VelocityAidTruth` container from truth velocity and attitude.

    Parameters
    ----------
    velocity_ned_mps : array-like, shape (3,)
        Truth velocity resolved in NED [m/s].
    C_n_b : array-like, shape (3, 3)
        Passive body->NED DCM.
    """
    v_n = vec3(velocity_ned_mps, name="velocity_ned_mps")
    v_b = velocity_body_from_ned(v_n, C_n_b)
    vh = horizontal_speed_from_velocity_ned(v_n)
    speed = speed_magnitude_from_velocity(v_n)
    cog = None if vh < 1.0e-9 else float(np.arctan2(v_n[1], v_n[0]))
    return VelocityAidTruth(
        velocity_ned_mps=v_n.copy(),
        velocity_body_mps=v_b,
        horizontal_speed_mps=vh,
        speed_mps=speed,
        course_over_ground_rad=cog,
    )


# -----------------------------------------------------------------------------
# Sensor specification and state containers
# -----------------------------------------------------------------------------


@dataclass
class VelocityAidSpec(SensorSpecBase):
    """
    First-order vector velocity-aid sensor specification.

    Parameters
    ----------
    noise_density_mps_per_sqrt_hz : scalar or shape (3,), default=0.0
        White-noise density in m/s/sqrt(Hz).
    bias_random_walk_mps_per_sqrt_s : scalar or shape (3,), default=0.0
        In-run bias random-walk coefficient in m/s/sqrt(s).
    turn_on_bias_std_mps : scalar or shape (3,), default=0.0
        1-sigma turn-on bias uncertainty [m/s].
    fixed_bias_mps : scalar or shape (3,), default=0.0
        Deterministic fixed bias [m/s].
    scale_factor_error_ppm : scalar or shape (3,), default=0.0
        Per-axis scale-factor error in ppm.
    max_abs_mps : scalar or shape (3,), default=np.inf
        Per-axis saturation limits [m/s].
    bandwidth_hz : float or None, default=None
        Optional first-order low-pass bandwidth.
    misalignment_matrix : array-like, shape (3, 3), optional
        Deterministic linear misalignment matrix applied to the filtered velocity.
        If omitted, identity is used.
    name : str, default="velocity_aid"
        Human-readable sensor identifier.

    Sensor model
    ------------
    The simulated vector output is:

        y_k = S M LPF(v_k) + b_k + n_k

    where:
    - v_k : ideal velocity vector in the chosen frame
    - LPF : optional first-order low-pass response
    - M   : misalignment matrix
    - S   : diagonal per-axis scale-factor matrix
    - b_k : current bias state (fixed + turn-on + random walk)
    - n_k : additive white noise

    Notes
    -----
    This class intentionally stays agnostic about whether the ideal velocity is:
    - GNSS-like Earth-relative velocity in NED,
    - DVL-like bottom-track velocity in body,
    - or any other 3D aiding velocity.
    That choice is made by the caller through the measurement method used.
    """

    noise_density_mps_per_sqrt_hz: ArrayLike | float = 0.0
    bias_random_walk_mps_per_sqrt_s: ArrayLike | float = 0.0
    turn_on_bias_std_mps: ArrayLike | float = 0.0
    fixed_bias_mps: ArrayLike | float = 0.0
    scale_factor_error_ppm: ArrayLike | float = 0.0
    max_abs_mps: ArrayLike | float = np.inf
    bandwidth_hz: Optional[float] = None
    misalignment_matrix: Optional[ArrayLike] = None
    name: str = "velocity_aid"

    def __post_init__(self) -> None:
        self.noise_density_mps_per_sqrt_hz = nonnegative_axis3(
            self.noise_density_mps_per_sqrt_hz,
            name="noise_density_mps_per_sqrt_hz",
        )
        self.bias_random_walk_mps_per_sqrt_s = nonnegative_axis3(
            self.bias_random_walk_mps_per_sqrt_s,
            name="bias_random_walk_mps_per_sqrt_s",
        )
        self.turn_on_bias_std_mps = nonnegative_axis3(
            self.turn_on_bias_std_mps,
            name="turn_on_bias_std_mps",
        )
        self.fixed_bias_mps = axis3(self.fixed_bias_mps, name="fixed_bias_mps")
        self.scale_factor_error_ppm = axis3(
            self.scale_factor_error_ppm,
            name="scale_factor_error_ppm",
        )
        self.max_abs_mps = positive_or_inf_axis3(
            self.max_abs_mps,
            name="max_abs_mps",
        )
        self.misalignment_matrix = _mat3_or_identity(
            self.misalignment_matrix,
            name="misalignment_matrix",
        )

        if self.bandwidth_hz is not None and float(self.bandwidth_hz) <= 0.0:
            raise ValueError("bandwidth_hz must be positive when provided.")
        if not np.all(np.isfinite(self.scale_factor_error_ppm)):
            raise ValueError("scale_factor_error_ppm must contain only finite values.")

    @classmethod
    def perfect(cls, name: str = "perfect_velocity_aid") -> "VelocityAidSpec":
        """
        Return a perfect velocity-aid specification.
        """
        return cls(name=name)

    @property
    def scale_factor_vector(self) -> FloatArray:
        """
        Per-axis multiplicative scale factors.

        Formula
        -------
            s_i = 1 + ppm_i * 1e-6
        """
        return 1.0 + 1.0e-6 * self.scale_factor_error_ppm

    @property
    def deterministic_matrix(self) -> FloatArray:
        """
        Combined scale and misalignment matrix.

        Formula
        -------
            D = diag(s) M
        """
        return np.diag(self.scale_factor_vector) @ self.misalignment_matrix

    def white_noise_std(self, dt_s: float) -> FloatArray:
        """Per-sample white-noise standard deviation [m/s]."""
        return vector_white_noise_std_from_density(
            self.noise_density_mps_per_sqrt_hz,
            dt_s,
        )

    def bias_step_std(self, dt_s: float) -> FloatArray:
        """Per-step bias-random-walk standard deviation [m/s]."""
        return vector_random_walk_step_std(
            self.bias_random_walk_mps_per_sqrt_s,
            dt_s,
        )


@dataclass
class VelocityAidBiasState:
    """
    State of the velocity-aid bias.

    Attributes
    ----------
    bias_mps : np.ndarray, shape (3,)
        Current active bias state [m/s].
    """

    bias_mps: FloatArray

    def copy(self) -> "VelocityAidBiasState":
        """Deep copy of the bias state."""
        return VelocityAidBiasState(bias_mps=self.bias_mps.copy())


@dataclass
class VelocityAidMeasurement:
    """
    One vector velocity-aid sample.

    Attributes
    ----------
    kind : str
        Human-readable measurement kind, currently always "velocity".
    frame : str
        Measurement frame label, e.g. "ned" or "body".
    time_s : float or None
        Optional timestamp [s].
    value_mps : np.ndarray, shape (3,)
        Final simulated measurement [m/s].
    ideal_value_mps : np.ndarray, shape (3,)
        Ideal noiseless velocity input [m/s].
    filtered_input_mps : np.ndarray, shape (3,)
        Post-bandwidth filtered velocity before deterministic calibration, bias,
        and white noise [m/s].
    bias_used_mps : np.ndarray, shape (3,)
        Bias state applied to this sample [m/s].
    white_noise_mps : np.ndarray, shape (3,)
        White-noise realization applied to this sample [m/s].
    saturated : bool
        True if clipping occurred on at least one axis.
    """

    kind: str
    frame: str
    time_s: Optional[float]
    value_mps: FloatArray
    ideal_value_mps: FloatArray
    filtered_input_mps: FloatArray
    bias_used_mps: FloatArray
    white_noise_mps: FloatArray
    saturated: bool

    @property
    def horizontal_speed_mps(self) -> float:
        """Horizontal speed magnitude from the final measurement."""
        return float(np.linalg.norm(self.value_mps[:2]))

    @property
    def speed_mps(self) -> float:
        """3D speed magnitude from the final measurement."""
        return float(np.linalg.norm(self.value_mps))


# -----------------------------------------------------------------------------
# Stateful sensor implementation
# -----------------------------------------------------------------------------


class VelocityAidSensor(StatefulSensorBase[VelocityAidSpec, VelocityAidBiasState]):
    """
    Stateful vector velocity-aid simulator.

    This class owns:
    - a specification (`VelocityAidSpec`)
    - a random-number generator
    - a vector bias state
    - an optional vector low-pass filter state

    Design philosophy
    -----------------
    Keep the measurement abstraction simple and estimator-friendly:
    - `measure_velocity_ned(...)` for GNSS-like or navigation-frame aiding
    - `measure_velocity_body(...)` for DVL/log-like or body-frame aiding
    - convenience wrappers from truth velocity and attitude

    More detailed physics such as bottom lock, water-track current modeling,
    beam geometry, or lever-arm rotational velocity can be layered on top later
    without breaking this generic interface.
    """

    def __init__(
        self,
        spec: VelocityAidSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(spec=spec, rng=rng)
        self.bias_state = VelocityAidBiasState(bias_mps=self.spec.fixed_bias_mps.copy())
        self._filter_state_mps: Optional[FloatArray] = None
        self.reset()

    def reset(
        self,
        *,
        turn_on_bias_randomized: bool = True,
        bias_override_mps: Optional[ArrayLike] = None,
        filter_state_override_mps: Optional[ArrayLike] = None,
    ) -> None:
        """
        Reset the velocity-aid state.

        Parameters
        ----------
        turn_on_bias_randomized : bool, default=True
            If True, sample a turn-on bias realization.
        bias_override_mps : array-like, shape (3,), optional
            If provided, use this as the full initial bias state.
        filter_state_override_mps : array-like, shape (3,), optional
            If provided, use this as the initial filter state.
        """
        if bias_override_mps is not None:
            bias = vec3(bias_override_mps, name="bias_override_mps")
        else:
            bias = self.spec.fixed_bias_mps.copy()
            if turn_on_bias_randomized:
                bias = bias + self.spec.turn_on_bias_std_mps * standard_normal3(self.rng)

        self.bias_state = VelocityAidBiasState(bias_mps=bias)
        self._filter_state_mps = (
            None
            if filter_state_override_mps is None
            else vec3(filter_state_override_mps, name="filter_state_override_mps")
        )

    def current_bias_state(self) -> VelocityAidBiasState:
        """
        Return a copy of the current bias state.
        """
        return self.bias_state.copy()

    def _step_bias_random_walk(self, dt_s: float) -> None:
        r"""
        Evolve the in-run bias state by one step.

        Formula
        -------
            b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
            w_k ~ N(0, I)
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")
        self.bias_state.bias_mps = (
            self.bias_state.bias_mps
            + self.spec.bias_step_std(dt_s) * standard_normal3(self.rng)
        )

    def _apply_bandwidth(self, x_mps: ArrayLike, dt_s: float) -> FloatArray:
        """
        Apply the optional first-order low-pass response to a velocity vector.
        """
        y = first_order_lpf_step(
            previous_state=self._filter_state_mps,
            input_value=vec3(x_mps, name="x_mps"),
            dt_s=dt_s,
            cutoff_hz=self.spec.bandwidth_hz,
        )
        y_vec = vec3(y, name="filtered_velocity")
        self._filter_state_mps = y_vec
        return y_vec

    def _measure_velocity(
        self,
        ideal_velocity_mps: ArrayLike,
        dt_s: float,
        *,
        frame: str,
        kind: str = "velocity",
        time_s: Optional[float] = None,
    ) -> VelocityAidMeasurement:
        """
        Core vector measurement routine.

        Parameters
        ----------
        ideal_velocity_mps : array-like, shape (3,)
            Ideal velocity input [m/s].
        dt_s : float
            Sample interval [s].
        frame : str
            Frame label, typically "ned" or "body".
        kind : str, default="velocity"
            Measurement kind label.
        time_s : float, optional
            Optional timestamp [s].
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        ideal = vec3(ideal_velocity_mps, name="ideal_velocity_mps")
        filtered = self._apply_bandwidth(ideal, dt_s)

        white_noise = self.spec.white_noise_std(dt_s) * standard_normal3(self.rng)
        bias_used = self.bias_state.bias_mps.copy()

        value = self.spec.deterministic_matrix @ filtered + bias_used + white_noise
        clipped_value, saturated = clip_vector_per_axis(value, self.spec.max_abs_mps)

        measurement = VelocityAidMeasurement(
            kind=str(kind),
            frame=str(frame),
            time_s=None if time_s is None else float(time_s),
            value_mps=clipped_value,
            ideal_value_mps=ideal,
            filtered_input_mps=filtered,
            bias_used_mps=bias_used,
            white_noise_mps=white_noise,
            saturated=saturated,
        )

        self._step_bias_random_walk(dt_s)
        return measurement

    def measure_velocity_ned(
        self,
        ideal_velocity_ned_mps: ArrayLike,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
    ) -> VelocityAidMeasurement:
        """
        Measure an ideal NED-frame velocity vector.

        This is the most natural interface for GNSS-like aiding or any other
        Earth-relative navigation-frame velocity update.
        """
        return self._measure_velocity(
            ideal_velocity_mps=ideal_velocity_ned_mps,
            dt_s=dt_s,
            frame="ned",
            kind="velocity",
            time_s=time_s,
        )

    def measure_velocity_body(
        self,
        ideal_velocity_body_mps: ArrayLike,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
    ) -> VelocityAidMeasurement:
        """
        Measure an ideal body-frame velocity vector.

        This is the most natural interface for DVL/log-like aiding where the
        measurement is expressed in body axes.
        """
        return self._measure_velocity(
            ideal_velocity_mps=ideal_velocity_body_mps,
            dt_s=dt_s,
            frame="body",
            kind="velocity",
            time_s=time_s,
        )

    def measure_velocity_ned_from_truth(
        self,
        velocity_ned_mps: ArrayLike,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
    ) -> VelocityAidMeasurement:
        """
        Convenience wrapper:
            truth NED velocity -> simulated NED-frame measurement.
        """
        return self.measure_velocity_ned(
            ideal_velocity_ned_mps=velocity_ned_mps,
            dt_s=dt_s,
            time_s=time_s,
        )

    def measure_velocity_body_from_truth(
        self,
        velocity_ned_mps: ArrayLike,
        C_n_b: ArrayLike,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
    ) -> VelocityAidMeasurement:
        """
        Convenience wrapper:
            truth NED velocity + body attitude -> simulated body-frame measurement.
        """
        v_b = velocity_body_from_ned(velocity_ned_mps, C_n_b)
        return self.measure_velocity_body(
            ideal_velocity_body_mps=v_b,
            dt_s=dt_s,
            time_s=time_s,
        )


__all__ = [
    "FloatArray",
    "VelocityAidBiasState",
    "VelocityAidMeasurement",
    "VelocityAidSensor",
    "VelocityAidSpec",
    "VelocityAidTruth",
    "build_velocity_aid_truth",
    "course_over_ground_from_velocity_ned",
    "horizontal_speed_from_velocity_ned",
    "speed_magnitude_from_velocity",
    "vector_random_walk_step_std",
    "vector_white_noise_std_from_density",
    "velocity_body_from_ned",
    "velocity_ned_from_body",
]