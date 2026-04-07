"""
vehicle_models.py

Kinematic vehicle-motion primitives for the gravity-aided navigation simulator.

This module provides the next layer above `truth.trajectory`:
- simple motion profiles expressed in navigation-facing quantities
- coordinated-turn relations that tie bank angle to heading rate
- smooth roll-in / roll-out utilities so truth trajectories do not contain
  unrealistic attitude discontinuities
- convenience builders that return full `TruthTrajectory` objects

Why this file exists
--------------------
The repository now has:
- `physics.earth` for WGS 84 geometry and normal gravity
- `physics.frames` for coordinate transforms and rotations
- `physics.kinematics` for differentiation / integration helpers
- `truth.trajectory` for validated truth-state containers and builders

The missing piece is a compact library of *vehicle motion primitives* that
scenario code can compose without re-deriving turn or climb equations each time.

This file is intentionally kinematic rather than aerodynamic. It is designed to
answer questions like:
- "give me a straight leg at 25 m/s for 60 s"
- "give me a level coordinated turn at 15 deg bank"
- "give me a smooth bank-in / bank-out arc instead of an instantaneous step"

Conventions
-----------
- NED means [North, East, Down].
- Heading / yaw is measured clockwise from North toward East [rad].
- Flight-path angle is positive for climb [rad].
- Roll angle is positive right-wing-down [rad].
- The body x-axis is assumed aligned with the velocity vector for these simple
  kinematic models (zero sideslip, zero angle-of-attack model).
- Attitude is represented by the passive body->NED DCM `C_n_b`.

Primary references used here
----------------------------
1) NASA Glenn Research Center, "Banking Turns"
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/banking-turns/

   Used for the standard coordinated-turn force balance idea that the lateral
   component of lift supplies the centripetal acceleration in a banked turn.

2) MIT OpenCourseWare, 16.333 Aircraft Stability and Control, Lecture 12
   https://ocw.mit.edu/courses/16-333-aircraft-stability-and-control-fall-2004/03fb92311b62c652e222e14efb49573b_lecture_12.pdf

   Used for the small set of practical coordinated-turn relations connecting
   bank angle, speed, and heading rate. The lecture notes summarize the steady
   coordinated-turn relation in the familiar form:

       psi_dot = g * tan(phi) / U

   where `phi` is bank angle and `U` is speed.

3) University of Stuttgart INSTINCT notes, local-navigation-frame position
   equations
   https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

   Used indirectly through `truth.trajectory` for the geodetic propagation from
   NED velocity to `(lat, lon, h)`.

4) NumPy documentation for the trapezoidal rule
   https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html

   Used conceptually for cumulative time integration of heading schedules and
   other sampled profiles. The implementation here reuses the repository's
   `physics.kinematics.cumulative_trapezoid(...)` helper.

Design philosophy
-----------------
- Keep the motion primitives transparent and testable.
- Prefer smooth profiles over discontinuous attitude jumps.
- Expose both profile-level builders and direct `TruthTrajectory` builders.
- Keep later scenario files declarative: they should say *what* motion to fly,
  not re-implement the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import normal_gravity
from ..physics.kinematics import cumulative_trapezoid
from .trajectory import (
    TruthTrajectory,
    build_truth_trajectory_from_velocity_attitude_history,
    dcm_history_body_to_ned_from_ypr,
)

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _time_vector(times_s: ArrayLike) -> FloatArray:
    """Validate and return a strictly increasing time vector."""
    t = _as_float_array(times_s).reshape(-1)
    if t.ndim != 1:
        raise ValueError("times_s must be one-dimensional.")
    if t.size < 2:
        raise ValueError("times_s must contain at least two samples.")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    return t


def _history_1d(x: ArrayLike, n: int, *, name: str) -> FloatArray:
    """Validate and return a length-N scalar history."""
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}.")
    return arr


def _wrap_angle_pi(angle_rad: ArrayLike) -> FloatArray | float:
    """Wrap angle(s) to [-pi, pi)."""
    angle = _as_float_array(angle_rad)
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float64)


def _require_nonnegative_scalar(x: float, *, name: str) -> float:
    """Require a scalar to be nonnegative."""
    val = float(x)
    if val < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {val}.")
    return val


# -----------------------------------------------------------------------------
# Fundamental kinematic relations
# -----------------------------------------------------------------------------


def velocity_ned_from_speed_heading_flight_path(
    speed_mps: ArrayLike,
    heading_rad: ArrayLike,
    flight_path_angle_rad: ArrayLike,
) -> FloatArray:
    r"""
    Convert speed magnitude, heading, and flight-path angle into NED velocity.

    Parameters
    ----------
    speed_mps : array-like, shape (...) or scalar
        Speed magnitude [m/s].
    heading_rad : array-like, shape (...) or scalar
        Heading / course angle [rad], measured clockwise from North toward East.
    flight_path_angle_rad : array-like, shape (...) or scalar
        Flight-path angle [rad]. Positive means climbing.

    Returns
    -------
    np.ndarray, shape (..., 3)
        NED velocity `[v_N, v_E, v_D]` [m/s].

    Formula
    -------
    With speed magnitude `V`, heading `psi`, and flight-path angle `gamma`:

        v_N = V cos(gamma) cos(psi)
        v_E = V cos(gamma) sin(psi)
        v_D = -V sin(gamma)

    Notes
    -----
    In NED, Down is positive, which is why a climb (`gamma > 0`) produces
    `v_D < 0`.
    """
    speed, heading, gamma = np.broadcast_arrays(
        _as_float_array(speed_mps),
        _as_float_array(heading_rad),
        _as_float_array(flight_path_angle_rad),
    )
    if np.any(speed < 0.0):
        raise ValueError("speed_mps must be nonnegative.")

    v_n = speed * np.cos(gamma) * np.cos(heading)
    v_e = speed * np.cos(gamma) * np.sin(heading)
    v_d = -speed * np.sin(gamma)
    return np.stack((v_n, v_e, v_d), axis=-1).astype(np.float64)


def heading_rate_from_coordinated_turn_bank(
    speed_mps: ArrayLike,
    bank_angle_rad: ArrayLike,
    *,
    gravity_mps2: float = 9.80665,
) -> FloatArray | float:
    r"""
    Compute steady coordinated-turn heading rate from speed and bank angle.

    Parameters
    ----------
    speed_mps : array-like or scalar
        Speed magnitude [m/s].
    bank_angle_rad : array-like or scalar
        Bank / roll angle [rad]. Positive values turn right.
    gravity_mps2 : float, default=9.80665
        Reference gravitational acceleration [m/s^2].

    Returns
    -------
    float or np.ndarray
        Heading rate [rad/s]. Positive means right turn.

    Formula
    -------
    For a steady coordinated turn, the standard relation is:

        psi_dot = g * tan(phi) / V

    where:
    - `psi_dot` is heading rate
    - `phi` is bank angle
    - `V` is speed magnitude

    References
    ----------
    - NASA Glenn explains the force-balance idea behind a banked coordinated turn.
    - MIT 16.333 Lecture 12 summarizes the relation between bank angle and
      heading rate for coordinated-turn motion.
    """
    g = float(gravity_mps2)
    if g <= 0.0:
        raise ValueError(f"gravity_mps2 must be positive, got {g}.")

    speed, bank = np.broadcast_arrays(_as_float_array(speed_mps), _as_float_array(bank_angle_rad))
    if np.any(speed <= 0.0):
        raise ValueError("speed_mps must be strictly positive.")

    out = g * np.tan(bank) / speed
    if out.ndim == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


def bank_angle_for_coordinated_turn_rate(
    speed_mps: ArrayLike,
    heading_rate_radps: ArrayLike,
    *,
    gravity_mps2: float = 9.80665,
) -> FloatArray | float:
    r"""
    Invert the coordinated-turn relation to obtain bank angle from speed and
    heading rate.

    Parameters
    ----------
    speed_mps : array-like or scalar
        Speed magnitude [m/s].
    heading_rate_radps : array-like or scalar
        Heading rate [rad/s].
    gravity_mps2 : float, default=9.80665
        Reference gravitational acceleration [m/s^2].

    Returns
    -------
    float or np.ndarray
        Bank angle [rad]. Positive corresponds to right turn.

    Formula
    -------
    Rearranging the steady coordinated-turn relation gives:

        phi = atan( V * psi_dot / g )
    """
    g = float(gravity_mps2)
    if g <= 0.0:
        raise ValueError(f"gravity_mps2 must be positive, got {g}.")

    speed, yaw_rate = np.broadcast_arrays(
        _as_float_array(speed_mps), _as_float_array(heading_rate_radps)
    )
    if np.any(speed < 0.0):
        raise ValueError("speed_mps must be nonnegative.")

    out = np.arctan2(speed * yaw_rate, g)
    if out.ndim == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


def turn_radius_from_speed_and_heading_rate(
    speed_mps: ArrayLike,
    heading_rate_radps: ArrayLike,
    *,
    flight_path_angle_rad: ArrayLike | float = 0.0,
) -> FloatArray | float:
    r"""
    Compute horizontal turn radius from speed and heading rate.

    Parameters
    ----------
    speed_mps : array-like or scalar
        Speed magnitude [m/s].
    heading_rate_radps : array-like or scalar
        Heading rate [rad/s].
    flight_path_angle_rad : array-like or scalar, default=0.0
        Flight-path angle [rad]. Positive means climbing.

    Returns
    -------
    float or np.ndarray
        Signed horizontal turn radius [m]. Positive and negative values indicate
        turn direction through the sign of the heading rate.

    Formula
    -------
    The horizontal speed magnitude is:

        V_h = V cos(gamma)

    so the horizontal turn radius is:

        R = V_h / psi_dot = V cos(gamma) / psi_dot

    Notes
    -----
    If `heading_rate_radps == 0`, the radius is undefined (straight motion) and
    this function raises `ValueError`.
    """
    speed, yaw_rate, gamma = np.broadcast_arrays(
        _as_float_array(speed_mps),
        _as_float_array(heading_rate_radps),
        _as_float_array(flight_path_angle_rad),
    )
    if np.any(speed < 0.0):
        raise ValueError("speed_mps must be nonnegative.")
    if np.any(yaw_rate == 0.0):
        raise ValueError("heading_rate_radps must be nonzero to define a turn radius.")

    out = speed * np.cos(gamma) / yaw_rate
    if out.ndim == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


def turn_radius_from_bank_angle(
    speed_mps: ArrayLike,
    bank_angle_rad: ArrayLike,
    *,
    gravity_mps2: float = 9.80665,
    flight_path_angle_rad: ArrayLike | float = 0.0,
) -> FloatArray | float:
    r"""
    Compute horizontal turn radius from speed and bank angle.

    Parameters
    ----------
    speed_mps : array-like or scalar
        Speed magnitude [m/s].
    bank_angle_rad : array-like or scalar
        Bank / roll angle [rad].
    gravity_mps2 : float, default=9.80665
        Reference gravitational acceleration [m/s^2].
    flight_path_angle_rad : array-like or scalar, default=0.0
        Flight-path angle [rad]. Positive means climbing.

    Returns
    -------
    float or np.ndarray
        Signed horizontal turn radius [m].

    Formula
    -------
    Combining:

        psi_dot = g tan(phi) / V
        R = V cos(gamma) / psi_dot

    gives:

        R = V^2 cos(gamma) / (g tan(phi))

    Notes
    -----
    The sign is inherited from `tan(phi)`, so positive bank gives positive
    right-turn radius in the same sign convention as `heading_rate_from_coordinated_turn_bank(...)`.
    """
    yaw_rate = heading_rate_from_coordinated_turn_bank(
        speed_mps=speed_mps,
        bank_angle_rad=bank_angle_rad,
        gravity_mps2=gravity_mps2,
    )
    return turn_radius_from_speed_and_heading_rate(
        speed_mps=speed_mps,
        heading_rate_radps=yaw_rate,
        flight_path_angle_rad=flight_path_angle_rad,
    )


# -----------------------------------------------------------------------------
# Smooth profile utilities
# -----------------------------------------------------------------------------


def uniform_time_grid(
    duration_s: float,
    dt_s: float,
    *,
    include_endpoint: bool = True,
) -> FloatArray:
    """
    Construct a uniform time grid starting at zero.

    Parameters
    ----------
    duration_s : float
        Segment duration [s]. Must be positive.
    dt_s : float
        Sample interval [s]. Must be positive.
    include_endpoint : bool, default=True
        If True, include the final point at `duration_s`.

    Returns
    -------
    np.ndarray, shape (N,)
        Uniform sample times [s].
    """
    duration = float(duration_s)
    dt = float(dt_s)
    if duration <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration}.")
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")

    if include_endpoint:
        n = int(np.round(duration / dt))
        t = np.linspace(0.0, duration, n + 1, dtype=np.float64)
    else:
        n = int(np.ceil(duration / dt))
        t = np.arange(n, dtype=np.float64) * dt
        if t.size < 2:
            raise ValueError("Time grid must contain at least two samples.")
    return t


def raised_cosine_transition(
    times_s: ArrayLike,
    start_value: float,
    end_value: float,
) -> FloatArray:
    r"""
    Smoothly transition between two scalar values over a supplied time base.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Time vector. Only relative position within the interval matters.
    start_value : float
        Initial value.
    end_value : float
        Final value.

    Returns
    -------
    np.ndarray, shape (N,)
        Smooth transition history.

    Formula
    -------
    Let:

        s = (t - t_0) / (t_f - t_0)

    for `s in [0, 1]`. The raised-cosine profile is:

        y(s) = y_0 + (y_f - y_0) * 0.5 * (1 - cos(pi s))

    Properties
    ----------
    - zero slope at the endpoints
    - no discontinuity in the value history
    - useful for roll-in / roll-out schedules and other motion primitives
    """
    t = _as_float_array(times_s).reshape(-1)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times_s must be one-dimensional with at least two samples.")
    if not np.all(np.diff(t) >= 0.0):
        raise ValueError("times_s must be nondecreasing.")

    s = (t - t[0]) / (t[-1] - t[0])
    alpha = 0.5 * (1.0 - np.cos(np.pi * s))
    return float(start_value) + (float(end_value) - float(start_value)) * alpha


# -----------------------------------------------------------------------------
# Profile container
# -----------------------------------------------------------------------------


@dataclass
class VehicleKinematicProfile:
    """
    Sampled motion profile expressed in vehicle-kinematic quantities.

    Attributes
    ----------
    time_s : np.ndarray, shape (N,)
        Sample times [s].
    speed_mps : np.ndarray, shape (N,)
        Speed magnitude history [m/s].
    heading_rad : np.ndarray, shape (N,)
        Heading / course history [rad].
    flight_path_angle_rad : np.ndarray, shape (N,)
        Flight-path angle history [rad]. Positive means climbing.
    roll_rad : np.ndarray, shape (N,)
        Roll / bank history [rad]. Positive means right-wing-down.

    Interpretation
    --------------
    These profiles assume a simple "body x-axis aligned with the velocity"
    kinematic model. That is deliberately appropriate for the simulator's first
    stage because the main objective is to create dynamically consistent truth
    motion for IMU and gravimeter testing rather than a high-fidelity aerodynamic
    state model.
    """

    time_s: FloatArray
    speed_mps: FloatArray
    heading_rad: FloatArray
    flight_path_angle_rad: FloatArray
    roll_rad: FloatArray

    def __post_init__(self) -> None:
        self.time_s = _time_vector(self.time_s)
        n = self.time_s.size
        self.speed_mps = _history_1d(self.speed_mps, n, name="speed_mps")
        self.heading_rad = _history_1d(self.heading_rad, n, name="heading_rad")
        self.flight_path_angle_rad = _history_1d(
            self.flight_path_angle_rad,
            n,
            name="flight_path_angle_rad",
        )
        self.roll_rad = _history_1d(self.roll_rad, n, name="roll_rad")

        if np.any(self.speed_mps < 0.0):
            raise ValueError("speed_mps must be nonnegative.")
        self.heading_rad = np.asarray(_wrap_angle_pi(self.heading_rad), dtype=np.float64)

    def __len__(self) -> int:
        """Number of samples in the profile."""
        return int(self.time_s.size)

    def copy(self) -> "VehicleKinematicProfile":
        """Return a deep copy of the profile."""
        return VehicleKinematicProfile(
            time_s=self.time_s.copy(),
            speed_mps=self.speed_mps.copy(),
            heading_rad=self.heading_rad.copy(),
            flight_path_angle_rad=self.flight_path_angle_rad.copy(),
            roll_rad=self.roll_rad.copy(),
        )

    @property
    def velocity_ned_mps(self) -> FloatArray:
        """NED velocity history [m/s]."""
        return velocity_ned_from_speed_heading_flight_path(
            speed_mps=self.speed_mps,
            heading_rad=self.heading_rad,
            flight_path_angle_rad=self.flight_path_angle_rad,
        )

    @property
    def yaw_pitch_roll_rad(self) -> FloatArray:
        """
        Yaw-pitch-roll history [rad].

        For this simple kinematic model we identify:
        - yaw   = heading
        - pitch = flight-path angle
        - roll  = bank angle history
        """
        return np.column_stack(
            [
                self.heading_rad,
                self.flight_path_angle_rad,
                self.roll_rad,
            ]
        ).astype(np.float64)

    @property
    def C_n_b(self) -> FloatArray:
        """Passive body->NED DCM history."""
        return dcm_history_body_to_ned_from_ypr(
            yaw_rad=self.heading_rad,
            pitch_rad=self.flight_path_angle_rad,
            roll_rad=self.roll_rad,
        )


# -----------------------------------------------------------------------------
# Primitive profile builders
# -----------------------------------------------------------------------------


def make_straight_profile(
    duration_s: float,
    dt_s: float,
    *,
    speed_mps: float,
    heading_rad: float,
    flight_path_angle_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> VehicleKinematicProfile:
    """
    Construct a constant-speed straight-line kinematic profile.

    Parameters
    ----------
    duration_s : float
        Segment duration [s].
    dt_s : float
        Sample interval [s].
    speed_mps : float
        Constant speed [m/s].
    heading_rad : float
        Constant heading [rad].
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad]. Positive means climbing.
    roll_rad : float, default=0.0
        Constant roll angle [rad].

    Returns
    -------
    VehicleKinematicProfile
        Straight-motion profile.
    """
    speed = _require_nonnegative_scalar(speed_mps, name="speed_mps")
    t = uniform_time_grid(duration_s, dt_s)
    n = t.size
    return VehicleKinematicProfile(
        time_s=t,
        speed_mps=np.full(n, speed, dtype=np.float64),
        heading_rad=np.full(n, float(heading_rad), dtype=np.float64),
        flight_path_angle_rad=np.full(n, float(flight_path_angle_rad), dtype=np.float64),
        roll_rad=np.full(n, float(roll_rad), dtype=np.float64),
    )


def make_constant_rate_turn_profile(
    duration_s: float,
    dt_s: float,
    *,
    speed_mps: float,
    initial_heading_rad: float,
    heading_rate_radps: float,
    flight_path_angle_rad: float = 0.0,
    roll_rad: float | None = None,
    gravity_mps2: float = 9.80665,
) -> VehicleKinematicProfile:
    r"""
    Construct a constant-speed, constant-heading-rate turn profile.

    Parameters
    ----------
    duration_s : float
        Segment duration [s].
    dt_s : float
        Sample interval [s].
    speed_mps : float
        Constant speed [m/s].
    initial_heading_rad : float
        Initial heading at the first sample [rad].
    heading_rate_radps : float
        Constant heading rate [rad/s]. Positive means right turn.
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad].
    roll_rad : float, optional
        Constant roll angle [rad]. If omitted, the steady coordinated-turn bank
        angle corresponding to `heading_rate_radps` is used.
    gravity_mps2 : float, default=9.80665
        Reference gravitational acceleration used if `roll_rad` is omitted.

    Returns
    -------
    VehicleKinematicProfile
        Constant-rate turn profile.

    Notes
    -----
    This builder is useful when you know the turn-rate requirement directly,
    such as standard-rate turns or scripted scenario commands.
    """
    speed = float(speed_mps)
    if speed <= 0.0:
        raise ValueError(f"speed_mps must be positive, got {speed}.")

    t = uniform_time_grid(duration_s, dt_s)
    heading = float(initial_heading_rad) + float(heading_rate_radps) * t
    if roll_rad is None:
        roll = bank_angle_for_coordinated_turn_rate(
            speed_mps=speed,
            heading_rate_radps=float(heading_rate_radps),
            gravity_mps2=gravity_mps2,
        )
    else:
        roll = float(roll_rad)

    n = t.size
    return VehicleKinematicProfile(
        time_s=t,
        speed_mps=np.full(n, speed, dtype=np.float64),
        heading_rad=np.asarray(_wrap_angle_pi(heading), dtype=np.float64),
        flight_path_angle_rad=np.full(n, float(flight_path_angle_rad), dtype=np.float64),
        roll_rad=np.full(n, float(roll), dtype=np.float64),
    )


def make_coordinated_turn_profile(
    duration_s: float,
    dt_s: float,
    *,
    speed_mps: float,
    initial_heading_rad: float,
    bank_angle_rad: float,
    flight_path_angle_rad: float = 0.0,
    gravity_mps2: float = 9.80665,
) -> VehicleKinematicProfile:
    r"""
    Construct a steady coordinated-turn profile from speed and bank angle.

    Parameters
    ----------
    duration_s : float
        Segment duration [s].
    dt_s : float
        Sample interval [s].
    speed_mps : float
        Constant speed [m/s].
    initial_heading_rad : float
        Heading at the first sample [rad].
    bank_angle_rad : float
        Constant bank / roll angle [rad]. Positive values turn right.
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad]. Positive means climbing.
    gravity_mps2 : float, default=9.80665
        Reference gravitational acceleration [m/s^2].

    Returns
    -------
    VehicleKinematicProfile
        Coordinated-turn profile.

    Formula
    -------
    The constant heading rate is computed from the steady coordinated-turn
    relation:

        psi_dot = g * tan(phi) / V
    """
    yaw_rate = heading_rate_from_coordinated_turn_bank(
        speed_mps=float(speed_mps),
        bank_angle_rad=float(bank_angle_rad),
        gravity_mps2=gravity_mps2,
    )
    return make_constant_rate_turn_profile(
        duration_s=duration_s,
        dt_s=dt_s,
        speed_mps=float(speed_mps),
        initial_heading_rad=float(initial_heading_rad),
        heading_rate_radps=float(yaw_rate),
        flight_path_angle_rad=float(flight_path_angle_rad),
        roll_rad=float(bank_angle_rad),
        gravity_mps2=gravity_mps2,
    )


def make_smooth_coordinated_turn_profile(
    duration_s: float,
    dt_s: float,
    *,
    speed_mps: float,
    initial_heading_rad: float,
    target_bank_angle_rad: float,
    flight_path_angle_rad: float = 0.0,
    gravity_mps2: float = 9.80665,
    roll_in_duration_s: float = 0.0,
    roll_out_duration_s: float = 0.0,
) -> VehicleKinematicProfile:
    r"""
    Construct a constant-speed coordinated-turn profile with optional smooth
    roll-in and roll-out transitions.

    Parameters
    ----------
    duration_s : float
        Total segment duration [s].
    dt_s : float
        Sample interval [s].
    speed_mps : float
        Constant speed [m/s].
    initial_heading_rad : float
        Initial heading [rad].
    target_bank_angle_rad : float
        Target peak bank angle [rad].
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad].
    gravity_mps2 : float, default=9.80665
        Reference gravitational acceleration [m/s^2].
    roll_in_duration_s : float, default=0.0
        Duration of the smooth roll-in [s].
    roll_out_duration_s : float, default=0.0
        Duration of the smooth roll-out [s].

    Returns
    -------
    VehicleKinematicProfile
        Smooth turn profile.

    Method
    ------
    1. Build a roll-angle history consisting of:
       - raised-cosine roll-in from 0 to `target_bank_angle_rad`
       - constant-bank hold
       - raised-cosine roll-out back to 0
    2. Convert instantaneous bank angle to heading rate using:

           psi_dot(t) = g tan(phi(t)) / V

    3. Integrate heading rate over time using the cumulative trapezoidal rule.

    Notes
    -----
    This is often a better truth primitive than an instantaneous bank step,
    because it avoids unrealistic attitude discontinuities and infinite angular
    acceleration at turn entry/exit.
    """
    speed = float(speed_mps)
    if speed <= 0.0:
        raise ValueError(f"speed_mps must be positive, got {speed}.")

    duration = float(duration_s)
    roll_in = _require_nonnegative_scalar(roll_in_duration_s, name="roll_in_duration_s")
    roll_out = _require_nonnegative_scalar(roll_out_duration_s, name="roll_out_duration_s")
    if roll_in + roll_out > duration + 1e-12:
        raise ValueError(
            "roll_in_duration_s + roll_out_duration_s must not exceed duration_s."
        )

    t = uniform_time_grid(duration, dt_s)
    roll = np.zeros_like(t, dtype=np.float64)
    bank = float(target_bank_angle_rad)

    t_hold_start = roll_in
    t_hold_end = duration - roll_out

    if roll_in > 0.0:
        mask_in = t <= t_hold_start + 1e-15
        roll[mask_in] = raised_cosine_transition(
            t[mask_in],
            0.0,
            bank,
        )
    else:
        mask_in = np.zeros_like(t, dtype=bool)

    mask_hold = (t > t_hold_start) & (t < t_hold_end)
    roll[mask_hold] = bank

    if roll_out > 0.0:
        mask_out = t >= t_hold_end - 1e-15
        roll[mask_out] = raised_cosine_transition(
            t[mask_out],
            bank,
            0.0,
        )
    else:
        mask_out = np.zeros_like(t, dtype=bool)

    if roll_in == 0.0 and roll_out == 0.0:
        roll[:] = bank
    else:
        remaining_mask = ~(mask_in | mask_hold | mask_out)
        roll[remaining_mask] = bank

    yaw_rate = np.asarray(
        heading_rate_from_coordinated_turn_bank(
            speed_mps=speed,
            bank_angle_rad=roll,
            gravity_mps2=gravity_mps2,
        ),
        dtype=np.float64,
    )
    heading = float(initial_heading_rad) + cumulative_trapezoid(t, yaw_rate, initial=0.0)

    n = t.size
    return VehicleKinematicProfile(
        time_s=t,
        speed_mps=np.full(n, speed, dtype=np.float64),
        heading_rad=np.asarray(_wrap_angle_pi(heading), dtype=np.float64),
        flight_path_angle_rad=np.full(n, float(flight_path_angle_rad), dtype=np.float64),
        roll_rad=roll,
    )


# -----------------------------------------------------------------------------
# Profile composition and trajectory conversion
# -----------------------------------------------------------------------------


def concatenate_profiles(
    profiles: Sequence[VehicleKinematicProfile],
    *,
    time_gap_s: float = 0.0,
) -> VehicleKinematicProfile:
    """
    Concatenate multiple profiles into one continuous time history.

    Parameters
    ----------
    profiles : sequence of VehicleKinematicProfile
        Profiles to concatenate in order.
    time_gap_s : float, default=0.0
        Optional extra gap inserted between consecutive segments [s].

    Returns
    -------
    VehicleKinematicProfile
        Concatenated profile.

    Notes
    -----
    The first sample of every segment after the first is dropped to avoid a
    duplicated timestamp at the join.
    """
    if not profiles:
        raise ValueError("profiles must contain at least one profile.")

    gap = float(time_gap_s)
    if gap < 0.0:
        raise ValueError("time_gap_s must be nonnegative.")

    t_parts: list[FloatArray] = []
    speed_parts: list[FloatArray] = []
    heading_parts: list[FloatArray] = []
    gamma_parts: list[FloatArray] = []
    roll_parts: list[FloatArray] = []

    time_offset = 0.0
    for i, prof in enumerate(profiles):
        p = prof.copy()
        if i == 0:
            t_local = p.time_s.copy()
            speed_local = p.speed_mps.copy()
            heading_local = p.heading_rad.copy()
            gamma_local = p.flight_path_angle_rad.copy()
            roll_local = p.roll_rad.copy()
        else:
            t_local = p.time_s[1:].copy()
            speed_local = p.speed_mps[1:].copy()
            heading_local = p.heading_rad[1:].copy()
            gamma_local = p.flight_path_angle_rad[1:].copy()
            roll_local = p.roll_rad[1:].copy()

        t_parts.append(t_local + time_offset)
        speed_parts.append(speed_local)
        heading_parts.append(heading_local)
        gamma_parts.append(gamma_local)
        roll_parts.append(roll_local)

        time_offset = float(t_parts[-1][-1] + gap)

    return VehicleKinematicProfile(
        time_s=np.concatenate(t_parts),
        speed_mps=np.concatenate(speed_parts),
        heading_rad=np.concatenate(heading_parts),
        flight_path_angle_rad=np.concatenate(gamma_parts),
        roll_rad=np.concatenate(roll_parts),
    )


def build_truth_trajectory_from_profile(
    profile: VehicleKinematicProfile,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
) -> TruthTrajectory:
    """
    Convert a kinematic profile into a full `TruthTrajectory`.

    Parameters
    ----------
    profile : VehicleKinematicProfile
        Input sampled motion profile.
    lat0_rad : float
        Initial geodetic latitude [rad].
    lon0_rad : float
        Initial longitude [rad].
    height0_m : float
        Initial ellipsoidal height [m].

    Returns
    -------
    TruthTrajectory
        Fully derived truth trajectory with geodetic position, NED velocity,
        attitude, velocity derivative, and body-rate history.
    """
    return build_truth_trajectory_from_velocity_attitude_history(
        times_s=profile.time_s,
        v_ned_mps=profile.velocity_ned_mps,
        C_n_b=profile.C_n_b,
        lat0_rad=float(lat0_rad),
        lon0_rad=float(lon0_rad),
        height0_m=float(height0_m),
    )


def make_straight_trajectory(
    duration_s: float,
    dt_s: float,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
    speed_mps: float,
    heading_rad: float,
    flight_path_angle_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> TruthTrajectory:
    """
    Convenience wrapper: straight profile -> `TruthTrajectory`.
    """
    profile = make_straight_profile(
        duration_s=duration_s,
        dt_s=dt_s,
        speed_mps=speed_mps,
        heading_rad=heading_rad,
        flight_path_angle_rad=flight_path_angle_rad,
        roll_rad=roll_rad,
    )
    return build_truth_trajectory_from_profile(
        profile,
        lat0_rad=lat0_rad,
        lon0_rad=lon0_rad,
        height0_m=height0_m,
    )


def make_coordinated_turn_trajectory(
    duration_s: float,
    dt_s: float,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
    speed_mps: float,
    initial_heading_rad: float,
    bank_angle_rad: float,
    flight_path_angle_rad: float = 0.0,
    gravity_mps2: float | None = None,
) -> TruthTrajectory:
    """
    Convenience wrapper: steady coordinated turn -> `TruthTrajectory`.

    Parameters
    ----------
    gravity_mps2 : float, optional
        Reference gravity used by the coordinated-turn relation. If omitted,
        WGS 84 normal gravity at the initial position is used.
    """
    g_ref = (
        float(normal_gravity(lat0_rad, height0_m))
        if gravity_mps2 is None
        else float(gravity_mps2)
    )
    profile = make_coordinated_turn_profile(
        duration_s=duration_s,
        dt_s=dt_s,
        speed_mps=speed_mps,
        initial_heading_rad=initial_heading_rad,
        bank_angle_rad=bank_angle_rad,
        flight_path_angle_rad=flight_path_angle_rad,
        gravity_mps2=g_ref,
    )
    return build_truth_trajectory_from_profile(
        profile,
        lat0_rad=lat0_rad,
        lon0_rad=lon0_rad,
        height0_m=height0_m,
    )


def make_smooth_coordinated_turn_trajectory(
    duration_s: float,
    dt_s: float,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
    speed_mps: float,
    initial_heading_rad: float,
    target_bank_angle_rad: float,
    flight_path_angle_rad: float = 0.0,
    gravity_mps2: float | None = None,
    roll_in_duration_s: float = 0.0,
    roll_out_duration_s: float = 0.0,
) -> TruthTrajectory:
    """
    Convenience wrapper: smooth bank-in / bank-out coordinated turn ->
    `TruthTrajectory`.
    """
    g_ref = (
        float(normal_gravity(lat0_rad, height0_m))
        if gravity_mps2 is None
        else float(gravity_mps2)
    )
    profile = make_smooth_coordinated_turn_profile(
        duration_s=duration_s,
        dt_s=dt_s,
        speed_mps=speed_mps,
        initial_heading_rad=initial_heading_rad,
        target_bank_angle_rad=target_bank_angle_rad,
        flight_path_angle_rad=flight_path_angle_rad,
        gravity_mps2=g_ref,
        roll_in_duration_s=roll_in_duration_s,
        roll_out_duration_s=roll_out_duration_s,
    )
    return build_truth_trajectory_from_profile(
        profile,
        lat0_rad=lat0_rad,
        lon0_rad=lon0_rad,
        height0_m=height0_m,
    )


__all__ = [
    "FloatArray",
    "VehicleKinematicProfile",
    "bank_angle_for_coordinated_turn_rate",
    "build_truth_trajectory_from_profile",
    "concatenate_profiles",
    "heading_rate_from_coordinated_turn_bank",
    "make_constant_rate_turn_profile",
    "make_coordinated_turn_profile",
    "make_coordinated_turn_trajectory",
    "make_smooth_coordinated_turn_profile",
    "make_smooth_coordinated_turn_trajectory",
    "make_straight_profile",
    "make_straight_trajectory",
    "raised_cosine_transition",
    "turn_radius_from_bank_angle",
    "turn_radius_from_speed_and_heading_rate",
    "uniform_time_grid",
    "velocity_ned_from_speed_heading_flight_path",
]