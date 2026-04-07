"""
kinematics.py

Reusable kinematic and numerical-trajectory helpers for the gravity-aided
navigation simulator.

This module is the bridge between the low-level frame/rotation utilities in
`gravnav.physics.frames` and the higher-level truth/scenario code that will
construct platform motion histories.

Why this file exists
--------------------
The repository already has:
- `earth.py` for WGS 84 geometry and normal gravity
- `frames.py` for rotations, local-level frame transforms, and quaternion/DCM math
- `imu.py` / `gravimeter.py` for truth-to-sensor measurement models

The missing middle layer is a clean set of reusable motion helpers for:
- differentiating sampled trajectories into velocity/acceleration-like signals
- integrating sampled rates into position-like signals
- propagating attitude from body angular-rate commands
- extracting simple navigation-relevant scalars such as speed, course, and
  flight-path angle

This file intentionally stays generic and deterministic. It does NOT implement a
full vehicle model, scenario language, or estimator. Those belong in later
modules.

Conventions
-----------
- Time is in seconds.
- Vectors are NumPy arrays with final dimension 3 where applicable.
- NED means [North, East, Down].
- Angles are in radians.
- DCMs follow the repository-wide passive convention documented in
  `gravnav.physics.frames`:

      v^beta = C^beta_alpha v^alpha

- For body attitude with respect to NED, we use `C_n_b`, the passive DCM mapping
  body-resolved vectors into NED coordinates.

Primary references used here
----------------------------
1) AHRS documentation, "Attitude from angular rate"
   URL:
   https://ahrs.readthedocs.io/en/latest/filters/angular.html

   Used for:
   - the quaternion angular-rate kinematics idea
   - the practical interpretation that attitude can be updated by integrating a
     local angular-rate signal over time

2) Carlo Tomasi, "Vector Representation of Rotations"
   URL:
   https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf

   Used for:
   - the exponential-map / Rodrigues relation between a rotation vector and a
     DCM increment
   - converting a small rotation vector `omega * dt` into an attitude update

3) NumPy documentation for `numpy.gradient`
   URL:
   https://numpy.org/doc/stable/reference/generated/numpy.gradient.html

   Used for:
   - a practical, well-tested implementation of sampled first derivatives using
     second-order accurate central differences in the interior and one-sided
     differences near the boundaries

4) NumPy documentation for `numpy.trapezoid`
   URL:
   https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html

   Used for:
   - the composite trapezoidal rule interpretation for integrating sampled data
     over time

Design notes
------------
- The derivative/integral helpers are deliberately written for the common case of
  time histories stored as arrays of shape `(N, ...)`, where axis 0 is time.
- The attitude propagators use the same passive-DCM convention as `frames.py`.
- The rate propagation uses a zero-order-hold assumption over each integration
  step. That is appropriate for simulation truth generation at sufficiently small
  time steps.
- This file prefers transparent, easily testable helpers over clever abstractions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frames import (
    dcm_from_rotvec,
    dcm_to_quaternion,
    project_to_so3,
    quaternion_to_dcm,
    rotvec_from_dcm,
    wrap_angle_pi,
)

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 3-vector."""
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}.")
    return arr


def _mat3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 3x3 matrix."""
    arr = _as_float_array(x)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be shape (3, 3), got {arr.shape}.")
    return arr


def _time_vector(times_s: ArrayLike) -> FloatArray:
    """
    Validate and return a strictly increasing 1D time vector.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    """
    t = _as_float_array(times_s).reshape(-1)
    if t.ndim != 1:
        raise ValueError("times_s must be one-dimensional.")
    if t.size < 2:
        raise ValueError("times_s must contain at least two samples.")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    return t


def _time_aligned_samples(
    times_s: ArrayLike,
    samples: ArrayLike,
    *,
    name: str,
) -> tuple[FloatArray, FloatArray]:
    """
    Validate a time history stored with time along axis 0.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    samples : array-like, shape (N, ...)
        Sampled data aligned with `times_s`.
    name : str
        Name used in error messages.
    """
    t = _time_vector(times_s)
    y = _as_float_array(samples)
    if y.shape[0] != t.shape[0]:
        raise ValueError(
            f"{name} must have shape (N, ...) with N=len(times_s)={t.shape[0]}, "
            f"got shape {y.shape}."
        )
    return t, y


def mean_sample_period(times_s: ArrayLike) -> float:
    """
    Return the mean sample interval of a strictly increasing time vector.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].

    Returns
    -------
    float
        Mean sample interval [s].
    """
    t = _time_vector(times_s)
    return float(np.mean(np.diff(t)))


def require_nearly_uniform_sampling(
    times_s: ArrayLike,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> float:
    r"""
    Validate that a time vector is nearly uniformly sampled and return the mean
    sample interval.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    rtol : float, default=1e-6
        Relative tolerance on step-to-step variation.
    atol : float, default=1e-12
        Absolute tolerance on step-to-step variation [s].

    Returns
    -------
    float
        Mean sample interval [s].

    Formula
    -------
    Let:

        dt_k = t_{k+1} - t_k
        dt_bar = mean(dt_k)

    Then this function requires:

        |dt_k - dt_bar| <= atol + rtol * |dt_bar|

    for all k.

    Notes
    -----
    Many truth-generation utilities will naturally use fixed-step propagation.
    This helper is useful when a later algorithm should fail fast rather than
    silently assuming uniform spacing.
    """
    t = _time_vector(times_s)
    dt = np.diff(t)
    dt_bar = float(np.mean(dt))
    if not np.allclose(dt, dt_bar, rtol=rtol, atol=atol):
        raise ValueError(
            "times_s is not nearly uniform at the requested tolerance: "
            f"mean dt={dt_bar:.9g} s, min dt={dt.min():.9g} s, max dt={dt.max():.9g} s."
        )
    return dt_bar


def first_derivative(times_s: ArrayLike, samples: ArrayLike) -> FloatArray:
    r"""
    Estimate the first time derivative of a sampled signal.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    samples : array-like, shape (N, ...)
        Sampled signal values.

    Returns
    -------
    np.ndarray, shape (N, ...)
        Estimated time derivative of the signal.

    Formula
    -------
    For uniformly sampled data, the interior samples follow the familiar
    second-order central-difference form:

        ydot_k ≈ (y_{k+1} - y_{k-1}) / (2 Δt)

    with one-sided differences used near the boundaries.

    Implementation note
    -------------------
    This function delegates the differentiation stencil construction to
    `numpy.gradient`, which uses second-order accurate central differences in the
    interior and one-sided differences at the edges.

    Reference
    ---------
    NumPy documentation for `numpy.gradient`:
    https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    """
    t, y = _time_aligned_samples(times_s, samples, name="samples")
    edge_order = 2 if t.size >= 3 else 1
    return np.asarray(np.gradient(y, t, axis=0, edge_order=edge_order), dtype=np.float64)


# Common alias used in trajectory / signal-processing code.
finite_difference = first_derivative


def second_derivative(times_s: ArrayLike, samples: ArrayLike) -> FloatArray:
    r"""
    Estimate the second time derivative of a sampled signal.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    samples : array-like, shape (N, ...)
        Sampled signal values.

    Returns
    -------
    np.ndarray, shape (N, ...)
        Estimated second time derivative.

    Formula
    -------
    For uniformly sampled data, the familiar interior stencil is:

        yddot_k ≈ (y_{k+1} - 2 y_k + y_{k-1}) / (Δt^2)

    This implementation computes the second derivative by applying the same
    sampled-gradient operator twice.

    Reference
    ---------
    The underlying first-derivative operator follows the `numpy.gradient`
    finite-difference scheme documented at:
    https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    """
    return first_derivative(times_s, first_derivative(times_s, samples))


def cumulative_trapezoid(
    times_s: ArrayLike,
    samples: ArrayLike,
    *,
    initial: ArrayLike | float = 0.0,
) -> FloatArray:
    r"""
    Cumulatively integrate a sampled signal using the composite trapezoidal rule.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    samples : array-like, shape (N, ...)
        Sampled signal values.
    initial : scalar or array-like, default=0.0
        Initial value of the integral at `times_s[0]`.
        Its shape must match `samples.shape[1:]` for non-scalar signals.

    Returns
    -------
    np.ndarray, shape (N, ...)
        Cumulative integral with the same leading time dimension as `samples`.

    Formula
    -------
    Over each interval `[t_k, t_{k+1}]`, the trapezoidal-rule increment is:

        ΔI_k ≈ 0.5 * (y_k + y_{k+1}) * (t_{k+1} - t_k)

    Therefore:

        I_0 = I_initial
        I_k = I_0 + Σ_{i=0}^{k-1} ΔI_i

    Reference
    ---------
    NumPy documentation for the composite trapezoidal rule:
    https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html
    """
    t, y = _time_aligned_samples(times_s, samples, name="samples")
    dt = np.diff(t)

    scale_shape = (dt.shape[0],) + (1,) * (y.ndim - 1)
    increments = 0.5 * (y[1:] + y[:-1]) * dt.reshape(scale_shape)

    out = np.empty_like(y, dtype=np.float64)

    init_arr = _as_float_array(initial)
    if init_arr.ndim == 0:
        out[0] = float(init_arr)
    else:
        if init_arr.shape != y.shape[1:]:
            raise ValueError(
                f"initial must be scalar or shape {y.shape[1:]}, got {init_arr.shape}."
            )
        out[0] = init_arr

    out[1:] = out[0] + np.cumsum(increments, axis=0)
    return out


def integrate_velocity_ned(
    times_s: ArrayLike,
    velocity_ned_mps: ArrayLike,
    *,
    initial_position_ned_m: ArrayLike | float = 0.0,
) -> FloatArray:
    r"""
    Integrate an NED velocity history into an NED displacement history.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    velocity_ned_mps : array-like, shape (N, 3)
        NED velocity history [m/s].
    initial_position_ned_m : scalar or array-like, default=0.0
        Initial NED displacement at `times_s[0]` [m].

    Returns
    -------
    np.ndarray, shape (N, 3)
        Integrated NED displacement history [m].

    Formula
    -------
        p^n(t) = p^n(t_0) + ∫_{t_0}^{t} v^n(τ) dτ

    with the integral approximated by the composite trapezoidal rule.
    """
    v_n = _as_float_array(velocity_ned_mps)
    if v_n.shape[-1] != 3:
        raise ValueError(
            f"velocity_ned_mps must have trailing shape 3, got {v_n.shape}."
        )
    return cumulative_trapezoid(
        times_s,
        v_n,
        initial=initial_position_ned_m,
    )


def speed_from_velocity_ned(v_ned_mps: ArrayLike) -> FloatArray | float:
    r"""
    Compute total speed from an NED velocity vector.

    Parameters
    ----------
    v_ned_mps : array-like, shape (..., 3)
        NED velocity [m/s].

    Returns
    -------
    float or np.ndarray
        Speed magnitude [m/s].

    Formula
    -------
        v = ||v^n||_2 = sqrt(v_N^2 + v_E^2 + v_D^2)
    """
    v = _as_float_array(v_ned_mps)
    if v.shape[-1] != 3:
        raise ValueError(f"v_ned_mps must have trailing shape 3, got {v.shape}.")
    speed = np.linalg.norm(v, axis=-1)
    if speed.ndim == 0:
        return float(speed)
    return np.asarray(speed, dtype=np.float64)


def horizontal_speed_from_velocity_ned(v_ned_mps: ArrayLike) -> FloatArray | float:
    r"""
    Compute horizontal speed from an NED velocity vector.

    Parameters
    ----------
    v_ned_mps : array-like, shape (..., 3)
        NED velocity [m/s].

    Returns
    -------
    float or np.ndarray
        Horizontal speed [m/s].

    Formula
    -------
        v_h = sqrt(v_N^2 + v_E^2)
    """
    v = _as_float_array(v_ned_mps)
    if v.shape[-1] != 3:
        raise ValueError(f"v_ned_mps must have trailing shape 3, got {v.shape}.")
    speed_h = np.hypot(v[..., 0], v[..., 1])
    if speed_h.ndim == 0:
        return float(speed_h)
    return np.asarray(speed_h, dtype=np.float64)


def course_from_velocity_ned(
    v_ned_mps: ArrayLike,
    *,
    undefined_value: float = np.nan,
) -> FloatArray | float:
    r"""
    Compute ground-track course angle from an NED velocity vector.

    Parameters
    ----------
    v_ned_mps : array-like, shape (..., 3)
        NED velocity [m/s].
    undefined_value : float, default=np.nan
        Value returned where horizontal speed is zero.

    Returns
    -------
    float or np.ndarray
        Course angle [rad], wrapped to [-pi, pi).

    Formula
    -------
        chi = atan2(v_E, v_N)

    Notes
    -----
    This is a ground-track / velocity-direction quantity, not necessarily the
    vehicle heading.
    """
    v = _as_float_array(v_ned_mps)
    if v.shape[-1] != 3:
        raise ValueError(f"v_ned_mps must have trailing shape 3, got {v.shape}.")

    north = v[..., 0]
    east = v[..., 1]
    vh = np.hypot(north, east)
    chi = wrap_angle_pi(np.arctan2(east, north))
    chi = np.where(vh > 0.0, chi, undefined_value)

    if np.asarray(chi).ndim == 0:
        return float(np.asarray(chi))
    return np.asarray(chi, dtype=np.float64)


def flight_path_angle_from_velocity_ned(
    v_ned_mps: ArrayLike,
    *,
    undefined_value: float = np.nan,
) -> FloatArray | float:
    r"""
    Compute flight-path angle from an NED velocity vector.

    Parameters
    ----------
    v_ned_mps : array-like, shape (..., 3)
        NED velocity [m/s].
    undefined_value : float, default=np.nan
        Value returned where total speed is zero.

    Returns
    -------
    float or np.ndarray
        Flight-path angle [rad]. Positive means climbing.

    Formula
    -------
    In NED coordinates, Down is positive, so climb rate is `-v_D`. Therefore:

        gamma = atan2(-v_D, sqrt(v_N^2 + v_E^2))

    Notes
    -----
    This sign convention is easy to get wrong in NED. The explicit `-v_D` is
    what makes positive angle correspond to upward flight.
    """
    v = _as_float_array(v_ned_mps)
    if v.shape[-1] != 3:
        raise ValueError(f"v_ned_mps must have trailing shape 3, got {v.shape}.")

    vh = np.hypot(v[..., 0], v[..., 1])
    speed = np.linalg.norm(v, axis=-1)
    gamma = np.arctan2(-v[..., 2], vh)
    gamma = np.where(speed > 0.0, gamma, undefined_value)

    if np.asarray(gamma).ndim == 0:
        return float(np.asarray(gamma))
    return np.asarray(gamma, dtype=np.float64)


def yaw_rate_from_turn_radius(speed_mps: ArrayLike, radius_m: ArrayLike):
    r"""
    Compute planar yaw/course rate from speed and signed turn radius.

    Parameters
    ----------
    speed_mps : array-like or scalar
        Tangential speed [m/s].
    radius_m : array-like or scalar
        Signed turn radius [m]. Positive and negative values encode turn
        direction.

    Returns
    -------
    float or np.ndarray
        Yaw/course rate [rad/s].

    Formula
    -------
    For planar circular motion:

        r = v / R

    where:
    - `r` is yaw rate [rad/s]
    - `v` is speed [m/s]
    - `R` is signed turn radius [m]
    """
    speed, radius = np.broadcast_arrays(_as_float_array(speed_mps), _as_float_array(radius_m))
    if np.any(radius == 0.0):
        raise ValueError("radius_m must be nonzero.")
    out = speed / radius
    if out.ndim == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


def lateral_acceleration_from_yaw_rate(speed_mps: ArrayLike, yaw_rate_radps: ArrayLike):
    r"""
    Compute planar lateral acceleration from speed and yaw/course rate.

    Parameters
    ----------
    speed_mps : array-like or scalar
        Tangential speed [m/s].
    yaw_rate_radps : array-like or scalar
        Yaw/course rate [rad/s].

    Returns
    -------
    float or np.ndarray
        Signed lateral acceleration [m/s^2].

    Formula
    -------
    For planar circular motion:

        a_lat = v * r

    which is equivalent to:

        a_lat = v^2 / R

    when `r = v / R`.
    """
    speed, yaw_rate = np.broadcast_arrays(
        _as_float_array(speed_mps), _as_float_array(yaw_rate_radps)
    )
    out = speed * yaw_rate
    if out.ndim == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


def body_rate_from_consecutive_dcms(
    C_n_b_prev: ArrayLike,
    C_n_b_next: ArrayLike,
    dt_s: float,
) -> FloatArray:
    r"""
    Estimate body angular rate with respect to NED from two successive body->NED DCMs.

    Parameters
    ----------
    C_n_b_prev : array-like, shape (3, 3)
        Body->NED DCM at time k.
    C_n_b_next : array-like, shape (3, 3)
        Body->NED DCM at time k+1.
    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Approximate body angular rate with respect to NED, resolved in body
        coordinates [rad/s].

    Formula
    -------
    Under the passive body->nav attitude convention:

        C_n_b(t + Δt) ≈ C_n_b(t) Exp([ω_nb^b Δt]_x)

    Therefore the body-frame incremental rotation is:

        ΔC_b ≈ C_n_b(t)^T C_n_b(t + Δt)

    and its rotation vector satisfies:

        rotvec(ΔC_b) ≈ ω_nb^b Δt

    so:

        ω_nb^b ≈ rotvec(ΔC_b) / Δt

    Reference
    ---------
    The matrix exponential / Rodrigues rotation-vector relation follows:
    Carlo Tomasi, "Vector Representation of Rotations"
    https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")

    C0 = project_to_so3(_mat3(C_n_b_prev, name="C_n_b_prev"))
    C1 = project_to_so3(_mat3(C_n_b_next, name="C_n_b_next"))
    delta_C_b = C0.T @ C1
    return rotvec_from_dcm(delta_C_b) / dt_s


def propagate_dcm_body_rate(
    C_n_b: ArrayLike,
    omega_nb_b_radps: ArrayLike,
    dt_s: float,
) -> FloatArray:
    r"""
    Propagate a body->NED DCM forward one step using a body-frame angular rate.

    Parameters
    ----------
    C_n_b : array-like, shape (3, 3)
        Current passive body->NED DCM.
    omega_nb_b_radps : array-like, shape (3,)
        Angular rate of body with respect to NED, resolved in body coordinates
        [rad/s].
    dt_s : float
        Time step [s].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Updated passive body->NED DCM.

    Formula
    -------
    Assuming piecewise-constant body rate over the interval, the rotation-vector
    increment is:

        δθ^b = ω_nb^b Δt

    and the passive attitude update is:

        C_n_b(t + Δt) = C_n_b(t) Exp([δθ^b]_x)

    where `Exp([δθ]_x)` is computed via Rodrigues' formula.

    References
    ----------
    - AHRS documentation, "Attitude from angular rate":
      https://ahrs.readthedocs.io/en/latest/filters/angular.html
    - Rodrigues / exponential-map relation from Tomasi:
      https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")

    C = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    omega_b = _vec3(omega_nb_b_radps, name="omega_nb_b_radps")
    delta_C = dcm_from_rotvec(omega_b * dt_s)
    return project_to_so3(C @ delta_C)


def propagate_quaternion_body_rate(
    q_n_b: ArrayLike,
    omega_nb_b_radps: ArrayLike,
    dt_s: float,
) -> FloatArray:
    r"""
    Propagate a body->NED attitude quaternion forward one step using a body-frame
    angular rate.

    Parameters
    ----------
    q_n_b : array-like, shape (4,)
        Scalar-first quaternion representing the same passive body->NED transform
        as `C_n_b` in the rest of the repository.
    omega_nb_b_radps : array-like, shape (3,)
        Angular rate of body with respect to NED, resolved in body coordinates
        [rad/s].
    dt_s : float
        Time step [s].

    Returns
    -------
    np.ndarray, shape (4,)
        Updated scalar-first quaternion.

    Method
    ------
    To avoid any ambiguity about quaternion multiplication order, this function
    performs the update through the repository's DCM convention:

        C_k     = C(q_k)
        C_{k+1} = propagate_dcm_body_rate(C_k, ω_nb^b, Δt)
        q_{k+1} = dcm_to_quaternion(C_{k+1})

    Reference
    ---------
    The underlying angular-rate propagation idea follows the AHRS documentation:
    https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """
    q = _as_float_array(q_n_b).reshape(-1)
    if q.shape != (4,):
        raise ValueError(f"q_n_b must be shape (4,), got {q.shape}.")
    C = quaternion_to_dcm(q)
    C_next = propagate_dcm_body_rate(C, omega_nb_b_radps, dt_s)
    return dcm_to_quaternion(C_next)


def propagate_dcm_sequence(
    initial_C_n_b: ArrayLike,
    omega_nb_b_radps: ArrayLike,
    times_s: ArrayLike,
) -> FloatArray:
    r"""
    Propagate a body->NED DCM sequence from sampled body rates.

    Parameters
    ----------
    initial_C_n_b : array-like, shape (3, 3)
        Initial body->NED DCM at `times_s[0]`.
    omega_nb_b_radps : array-like, shape (N-1, 3) or (N, 3)
        Sampled body angular-rate history [rad/s].
        If shape is `(N-1, 3)`, rate k is applied over `[t_k, t_{k+1}]`.
        If shape is `(N, 3)`, the final sample is ignored and the same rule is
        applied using the first `N-1` samples.
    times_s : array-like, shape (N,)
        Sample times [s].

    Returns
    -------
    np.ndarray, shape (N, 3, 3)
        Propagated body->NED DCM history.

    Formula
    -------
    For each interval:

        C_{k+1} = C_k Exp([ω_k Δt_k]_x)
        Δt_k = t_{k+1} - t_k

    Notes
    -----
    This is a zero-order-hold integration of the angular rate.
    """
    t = _time_vector(times_s)
    C0 = project_to_so3(_mat3(initial_C_n_b, name="initial_C_n_b"))
    omega = _as_float_array(omega_nb_b_radps)

    if omega.ndim != 2 or omega.shape[1] != 3:
        raise ValueError(
            f"omega_nb_b_radps must have shape (N-1, 3) or (N, 3), got {omega.shape}."
        )

    if omega.shape[0] == t.size:
        omega_int = omega[:-1]
    elif omega.shape[0] == t.size - 1:
        omega_int = omega
    else:
        raise ValueError(
            f"omega_nb_b_radps must have {t.size - 1} or {t.size} rows, got {omega.shape[0]}."
        )

    out = np.empty((t.size, 3, 3), dtype=np.float64)
    out[0] = C0

    for k in range(t.size - 1):
        dt = float(t[k + 1] - t[k])
        out[k + 1] = propagate_dcm_body_rate(out[k], omega_int[k], dt)

    return out


def propagate_quaternion_sequence(
    initial_q_n_b: ArrayLike,
    omega_nb_b_radps: ArrayLike,
    times_s: ArrayLike,
) -> FloatArray:
    r"""
    Propagate a body->NED quaternion sequence from sampled body rates.

    Parameters
    ----------
    initial_q_n_b : array-like, shape (4,)
        Initial body->NED scalar-first quaternion at `times_s[0]`.
    omega_nb_b_radps : array-like, shape (N-1, 3) or (N, 3)
        Sampled body angular-rate history [rad/s].
    times_s : array-like, shape (N,)
        Sample times [s].

    Returns
    -------
    np.ndarray, shape (N, 4)
        Propagated quaternion history.

    Method
    ------
    This function propagates the sequence through the DCM representation to stay
    perfectly aligned with the repository's passive-transform convention.
    """
    q0 = _as_float_array(initial_q_n_b).reshape(-1)
    if q0.shape != (4,):
        raise ValueError(f"initial_q_n_b must be shape (4,), got {q0.shape}.")

    C_hist = propagate_dcm_sequence(quaternion_to_dcm(q0), omega_nb_b_radps, times_s)
    q_hist = np.empty((C_hist.shape[0], 4), dtype=np.float64)
    for k in range(C_hist.shape[0]):
        q_hist[k] = dcm_to_quaternion(C_hist[k])
    return q_hist


__all__ = [
    "FloatArray",
    "body_rate_from_consecutive_dcms",
    "course_from_velocity_ned",
    "cumulative_trapezoid",
    "finite_difference",
    "first_derivative",
    "flight_path_angle_from_velocity_ned",
    "horizontal_speed_from_velocity_ned",
    "integrate_velocity_ned",
    "lateral_acceleration_from_yaw_rate",
    "mean_sample_period",
    "propagate_dcm_body_rate",
    "propagate_dcm_sequence",
    "propagate_quaternion_body_rate",
    "propagate_quaternion_sequence",
    "require_nearly_uniform_sampling",
    "second_derivative",
    "speed_from_velocity_ned",
    "yaw_rate_from_turn_radius",
]