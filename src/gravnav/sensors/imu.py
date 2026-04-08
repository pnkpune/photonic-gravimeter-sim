"""
imu.py

IMU sensor models, ideal inertial measurement generation from truth kinematics,
and discrete-time stochastic corruption models for the gravity-aided navigation
simulator.

This module is deliberately split into two layers:

1) "Ideal measurement" layer
   Converts truth trajectory / truth attitude / truth kinematics into the
   quantities a strapdown IMU would measure in the body frame:
   - gyroscope output: angular rate of body with respect to inertial space,
     resolved in body coordinates, ω_ib^b
   - accelerometer output: specific force resolved in body coordinates, f_ib^b

2) "Sensor corruption" layer
   Applies deterministic calibration residuals (scale / misalignment), fixed
   biases, turn-on bias uncertainty, additive white noise, and in-run bias
   random walk to produce a realistic simulated IMU measurement stream.

Conventions
-----------
- Frames follow the repository-wide conventions defined in `gravnav.physics.frames`.
- NED is the local navigation frame:
      x = North, y = East, z = Down
- Body frame is assumed right-handed and aviation-style:
      x = forward, y = right, z = down
- Angular rates are in rad/s.
- Specific force and acceleration are in m/s^2.
- Noise densities are interpreted using the standard "ideal anti-alias / ideal
  decimation" discrete-time scaling used by Kalibr:
      sigma_discrete = sigma_density / sqrt(dt)
- Bias random walk parameters in THIS FILE are specified in explicit discrete-step
  units:
      [bias units] / sqrt(s)
  so that:
      b_{k+1} = b_k + sigma_rw * sqrt(dt) * w_k
  which is numerically equivalent to the Kalibr Brownian-motion discretization.

Primary references used here
----------------------------
1) INSTINCT: IMU Simulator
   https://unistuttgart-ins.github.io/INSTINCT/ImuSimulator.html

   Used for the local-navigation-frame velocity equation and the accelerometer
   specific-force relation:
       v_dot^n = f^n - (2 ω_ie^n + ω_en^n) × v^n + g^n
   therefore:
       f^n = v_dot^n + (2 ω_ie^n + ω_en^n) × v^n - g^n

2) INSTINCT: IMU Integrator (local-navigation frame)
   https://unistuttgart-ins.github.io/INSTINCT/ImuIntegrator_n.html

   Used for:
   - gyroscope relation:
         ω_nb^b = ω_ib^b - C_n^b (ω_ie^n + ω_en^n)
     rearranged here to:
         ω_ib^b = ω_nb^b + C_n^b (ω_ie^n + ω_en^n)
   - local-level mechanization consistency with the rest of the repository

3) AHRS documentation, "Attitude from angular rate"
   https://ahrs.readthedocs.io/en/latest/filters/angular.html

   Used for the quaternion angular-rate propagation relation:
       q_dot = 0.5 * Ω(ω) q
   and the corresponding constant-rate discrete-time propagation idea.
   In this file we do not perform full navigation-frame quaternion mechanization,
   but we do use this relation to document the meaning of gyroscope truth rates
   and to support helper functions that estimate body rate from successive
   attitudes.

4) Kalibr Wiki, "IMU Noise Model"
   https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

   Used for the standard stochastic IMU model:
       y(t) = y_true(t) + b(t) + n(t)
   with:
       n(t): additive white Gaussian noise
       b(t): Brownian / random-walk bias
   and the practical discrete-time simulation rules:
       sigma_d = sigma / sqrt(dt)
       b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k

5) gravnav.physics.earth
   This repository's `earth.py` implements WGS 84 normal gravity and documents
   the underlying NGA WGS 84 formulas. In this module, the local NED gravity vector
   is built from that normal-gravity magnitude as:
       g_l^n = [0, 0, +g]^T
   because Down is positive in NED.

Design notes
------------
- This module intentionally uses WGS 84 *normal gravity* by default, not a local
  geological anomaly field. The gravity anomaly will later be added separately in
  the gravimeter / map layers of the simulator.
- The accelerometer model here generates *specific force*, not total acceleration.
- The gyroscope model here generates inertial angular rate ω_ib^b, not body rate
  relative to the local NED frame alone.
- The stochastic model is deliberately transparent and conservative. Temperature
  drift, colored noise, vibration rectification error, g-sensitivity of gyros,
  and clip/recovery transients can be layered in later if needed.

What this file is for
---------------------
This file is the correct next layer after:
- `gravnav.physics.earth`
- `gravnav.physics.frames`

and before:
- truth trajectory generation
- gravimeter simulation
- full INS mechanization
- gravity-map matching

because every later module needs a consistent answer to:
- "What should the ideal IMU read for this truth motion?"
- "How do we convert IMU datasheet-style parameters into simulated samples?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import normal_gravity
from ..physics.frames import (
    earth_rate_ned,
    project_to_so3,
    rotvec_from_dcm,
    transport_rate_ned,
)
from ..physics.kinematics import body_rate_from_consecutive_dcms

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a 3-vector of shape (3,).

    Parameters
    ----------
    x : array-like
        Input vector.
    name : str
        Name used in error messages.
    """
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}.")
    return arr


def _mat3(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a 3x3 matrix.
    """
    arr = _as_float_array(x)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be shape (3, 3), got {arr.shape}.")
    return arr


def _axis3(
    x: ArrayLike | float,
    *,
    name: str,
) -> FloatArray:
    """
    Convert a scalar or 3-vector into a 3-vector.

    This is convenient for sensor specs because many datasheets or first-order
    simulation studies use one scalar value per sensor type, while later stages
    may need per-axis values.
    """
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}.")
    return arr


def _nonnegative_axis3(
    x: ArrayLike | float,
    *,
    name: str,
) -> FloatArray:
    """
    Like `_axis3(...)`, but require all entries to be nonnegative.
    """
    arr = _axis3(x, name=name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative, got {arr}.")
    return arr


def _max_abs_axis3(
    x: ArrayLike | float,
    *,
    name: str,
) -> FloatArray:
    """
    Convert input into a per-axis maximum-absolute-value vector.

    Accepts:
    - scalar
    - shape (3,)
    - np.inf
    """
    arr = _axis3(x, name=name)
    if np.any(arr <= 0.0) and not np.all(np.isinf(arr)):
        raise ValueError(f"{name} must contain positive values or inf, got {arr}.")
    return arr


def _maybe_identity_matrix(M: Optional[ArrayLike], *, name: str) -> FloatArray:
    """
    Return identity if M is None, else validate as a 3x3 matrix.
    """
    if M is None:
        return np.eye(3, dtype=np.float64)
    return _mat3(M, name=name)


def _clip_per_axis(v: ArrayLike, max_abs: ArrayLike) -> FloatArray:
    """
    Clip a 3-vector independently on each axis to +/- max_abs.
    """
    vec = _vec3(v, name="v")
    lim = _axis3(max_abs, name="max_abs")
    return np.clip(vec, -lim, lim)


def _randn3(rng: np.random.Generator) -> FloatArray:
    """
    Standard normal 3-vector.
    """
    return rng.standard_normal(3, dtype=np.float64)


def discrete_white_noise_std_from_density(
    noise_density_per_sqrt_hz: ArrayLike | float,
    dt_s: float,
) -> FloatArray:
    r"""
    Convert continuous-time white-noise density to discrete-time sample standard deviation.

    Parameters
    ----------
    noise_density_per_sqrt_hz : scalar or shape (3,)
        Continuous-time white-noise density.
        Examples:
        - gyro:  rad/s / sqrt(Hz)
        - accel: m/s^2 / sqrt(Hz)
    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Discrete-time per-sample standard deviation in the same physical units as
        the underlying measurement (rad/s for gyro, m/s^2 for accel).

    Formula
    -------
    The standard Kalibr discrete-time approximation is:

        sigma_d = sigma / sqrt(dt)

    assuming ideal anti-alias filtering / ideal decimation before sampling.

    Reference
    ---------
    Kalibr Wiki, "IMU Noise Model":
    https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

    See the "Additive White Noise" section and the discrete-time implementation:
        sigma_d = sigma / sqrt(Δt)

    Important
    ---------
    This scaling is correct only under the same assumption explicitly called out
    by Kalibr: the sensor stream has been properly low-pass filtered before
    decimation. If you later simulate sub-sampling without anti-alias filtering,
    do NOT reuse this scaling blindly.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")

    sigma = _nonnegative_axis3(
        noise_density_per_sqrt_hz,
        name="noise_density_per_sqrt_hz",
    )
    return sigma / np.sqrt(dt_s)


def discrete_random_walk_step_std(
    random_walk_per_sqrt_s: ArrayLike | float,
    dt_s: float,
) -> FloatArray:
    r"""
    Convert a bias random-walk coefficient into the standard deviation of the
    discrete-time bias increment over one step.

    Parameters
    ----------
    random_walk_per_sqrt_s : scalar or shape (3,)
        Bias random-walk coefficient in explicit discrete-step units:
        [bias units] / sqrt(s)

        Examples:
        - gyro bias random walk:  rad/s / sqrt(s)
        - accel bias random walk: m/s^2 / sqrt(s)

    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Standard deviation of the bias increment over one step, in bias units.

    Formula
    -------
    The Brownian / Wiener random-walk discretization is:

        b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
        w_k ~ N(0, I)

    so the one-step bias increment standard deviation is:

        sigma_step = sigma_rw * sqrt(dt)

    Reference
    ---------
    Kalibr Wiki, "IMU Noise Model":
    https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

    See the "Bias" section and the discrete-time implementation.

    Note on units
    -------------
    Some texts/toolboxes parameterize the continuous-time driving-noise intensity
    instead. Here we deliberately choose the unambiguous practical simulation unit
    "[bias units]/sqrt(s)" so that the discrete implementation is obvious and
    dimensionally transparent.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")

    sigma = _nonnegative_axis3(
        random_walk_per_sqrt_s,
        name="random_walk_per_sqrt_s",
    )
    return sigma * np.sqrt(dt_s)


def gravity_vector_ned(
    lat_rad: float,
    height_m: float,
    gravity_override_mps2: Optional[float] = None,
) -> FloatArray:
    r"""
    Return the local gravity vector resolved in NED coordinates.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    gravity_override_mps2 : float, optional
        If provided, use this scalar gravity magnitude instead of WGS 84 normal
        gravity.

    Returns
    -------
    np.ndarray, shape (3,)
        Gravity vector in NED:
            g_l^n = [0, 0, +g]^T

    Formula
    -------
    In local NED coordinates, Down is positive. Therefore a scalar gravity
    magnitude g is represented as:

        g_l^n = [0, 0, +g]^T

    In this repository, the default magnitude g is the WGS 84 normal gravity
    returned by `gravnav.physics.earth.normal_gravity(...)`.

    Reference
    ---------
    The local-navigation-frame velocity equation used by INSTINCT writes the
    gravity term as +g^n in the NED frame:
    https://unistuttgart-ins.github.io/INSTINCT/ImuSimulator.html

    The magnitude used here comes from the repository's WGS 84 normal-gravity
    implementation in `earth.py`, which is based on NGA.STND.0036_1.0.0_WGS84.
    """
    g = float(normal_gravity(lat_rad, height_m) if gravity_override_mps2 is None else gravity_override_mps2)
    return np.array([0.0, 0.0, g], dtype=np.float64)


def coriolis_transport_acceleration_ned(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> FloatArray:
    r"""
    Return the Coriolis + transport acceleration term in local NED coordinates.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : array-like, shape (3,)
        Velocity with respect to Earth resolved in NED [m/s].

    Returns
    -------
    np.ndarray, shape (3,)
        The quantity:
            (2 ω_ie^n + ω_en^n) × v^n

    Formula
    -------
    The standard local-level navigation velocity equation is:

        v_dot^n = f^n - (2 ω_ie^n + ω_en^n) × v^n + g^n

    Therefore the Coriolis/transport term that appears with a minus sign in the
    mechanization is:

        c^n = (2 ω_ie^n + ω_en^n) × v^n

    This function returns exactly c^n.

    Reference
    ---------
    INSTINCT: IMU Simulator
    https://unistuttgart-ins.github.io/INSTINCT/ImuSimulator.html

    See the local-navigation-frame velocity equation.
    """
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    omega_ie_n = earth_rate_ned(lat_rad)
    omega_en_n = transport_rate_ned(lat_rad, height_m, v_n)
    return np.cross(2.0 * omega_ie_n + omega_en_n, v_n)


def ideal_specific_force_ned(
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
    gravity_override_mps2: Optional[float] = None,
) -> FloatArray:
    r"""
    Compute the ideal accelerometer specific-force output resolved in NED.

    Parameters
    ----------
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity with respect to Earth, resolved in NED [m/s^2].
        This is the left-hand side of the standard local-level velocity equation.
    v_ned_mps : array-like, shape (3,)
        Velocity with respect to Earth resolved in NED [m/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    gravity_override_mps2 : float, optional
        If provided, use this scalar gravity magnitude instead of WGS 84 normal
        gravity.

    Returns
    -------
    np.ndarray, shape (3,)
        Ideal accelerometer specific force resolved in NED [m/s^2].

    Formula
    -------
    Starting from the standard local-level navigation equation:

        v_dot^n = f^n - (2 ω_ie^n + ω_en^n) × v^n + g^n

    rearrange to obtain the ideal specific force:

        f^n = v_dot^n + (2 ω_ie^n + ω_en^n) × v^n - g^n

    This is exactly what an ideal accelerometer triad would report, after the
    result is resolved in the local NED frame.

    Physical interpretation
    -----------------------
    Specific force is not total acceleration. It is the non-gravitational
    acceleration per unit mass, i.e. the proper acceleration measured by the
    accelerometers. Example:
    - stationary, level, aligned NED/body, zero velocity:
          v_dot^n = 0, v^n = 0, g^n = [0,0,+g]
      then:
          f^n = [0,0,-g]
      which is the familiar "accelerometer reads -g on the down axis" result in
      an NED/down-positive convention.

    Reference
    ---------
    INSTINCT: IMU Simulator
    https://unistuttgart-ins.github.io/INSTINCT/ImuSimulator.html

    See the NED velocity equation and the identification of f^n as the specific
    force measured by the accelerometers.
    """
    v_dot_n = _vec3(v_dot_ned_mps2, name="v_dot_ned_mps2")
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    g_n = gravity_vector_ned(lat_rad, height_m, gravity_override_mps2)
    c_n = coriolis_transport_acceleration_ned(lat_rad, height_m, v_n)
    return v_dot_n + c_n - g_n


def ideal_specific_force_body(
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    C_n_b: ArrayLike,
    lat_rad: float,
    height_m: float,
    gravity_override_mps2: Optional[float] = None,
) -> FloatArray:
    r"""
    Compute the ideal accelerometer specific-force output resolved in body coordinates.

    Parameters
    ----------
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity with respect to Earth, resolved in NED [m/s^2].
    v_ned_mps : array-like, shape (3,)
        Velocity with respect to Earth resolved in NED [m/s].
    C_n_b : array-like, shape (3, 3)
        Passive DCM that maps body-resolved vectors into NED:
            v^n = C_n_b v^b
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    gravity_override_mps2 : float, optional
        If provided, use this scalar gravity magnitude instead of WGS 84 normal
        gravity.

    Returns
    -------
    np.ndarray, shape (3,)
        Ideal accelerometer specific force resolved in body frame [m/s^2].

    Formula
    -------
    First compute f^n using:

        f^n = v_dot^n + (2 ω_ie^n + ω_en^n) × v^n - g^n

    Then transform to body coordinates:

        f^b = C_b_n f^n = (C_n_b)^T f^n

    Reference
    ---------
    The specific-force relation is from:
    https://unistuttgart-ins.github.io/INSTINCT/ImuSimulator.html

    The frame-transform convention is the repository's passive-transform
    convention documented in `gravnav.physics.frames`.
    """
    C_n_b = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    C_b_n = C_n_b.T
    f_n = ideal_specific_force_ned(
        v_dot_ned_mps2=v_dot_ned_mps2,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        height_m=height_m,
        gravity_override_mps2=gravity_override_mps2,
    )
    return C_b_n @ f_n


def ideal_gyro_rate_body(
    omega_nb_b_radps: ArrayLike,
    C_n_b: ArrayLike,
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> FloatArray:
    r"""
    Compute the ideal gyroscope output ω_ib^b from truth body rate and navigation-frame motion.

    Parameters
    ----------
    omega_nb_b_radps : array-like, shape (3,)
        Angular rate of the body frame with respect to the local navigation frame,
        resolved in body coordinates [rad/s].
    C_n_b : array-like, shape (3, 3)
        Passive DCM mapping body -> NED:
            v^n = C_n_b v^b
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : array-like, shape (3,)
        Velocity with respect to Earth resolved in NED [m/s].

    Returns
    -------
    np.ndarray, shape (3,)
        Ideal gyroscope measurement ω_ib^b [rad/s].

    Formula
    -------
    INSTINCT gives:

        ω_nb^b = ω_ib^b - C_n^b (ω_ie^n + ω_en^n)

    Rearranging:

        ω_ib^b = ω_nb^b + C_n^b (ω_ie^n + ω_en^n)

    In the notation of this repository:
    - C_n_b maps body -> nav (NED)
    - therefore C_b_n = (C_n_b)^T maps nav -> body

    So numerically we compute:

        ω_ib^b = ω_nb^b + C_b_n (ω_ie^n + ω_en^n)

    Reference
    ---------
    INSTINCT: IMU Integrator (local-navigation frame)
    https://unistuttgart-ins.github.io/INSTINCT/ImuIntegrator_n.html

    See the relation:
        ω_nb^b = ω_ib^b - C_n^b [ω_ie^n + ω_en^n]
    """
    omega_nb_b = _vec3(omega_nb_b_radps, name="omega_nb_b_radps")
    C_n_b = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    C_b_n = C_n_b.T
    v_n = _vec3(v_ned_mps, name="v_ned_mps")

    omega_ie_n = earth_rate_ned(lat_rad)
    omega_en_n = transport_rate_ned(lat_rad, height_m, v_n)
    omega_in_n = omega_ie_n + omega_en_n

    return omega_nb_b + C_b_n @ omega_in_n


def body_rate_from_consecutive_attitudes(
    C_n_b_prev: ArrayLike,
    C_n_b_next: ArrayLike,
    dt_s: float,
) -> FloatArray:
    r"""
    Estimate the body angular rate with respect to the navigation frame,
    resolved in body coordinates, from two successive body->navigation DCMs.

    Parameters
    ----------
    C_n_b_prev : array-like, shape (3, 3)
        Body->NED DCM at time k.
    C_n_b_next : array-like, shape (3, 3)
        Body->NED DCM at time k+1.
    dt_s : float
        Time step [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Approximate constant body rate over the interval:
            ω_nb^b [rad/s]

    Formula
    -------
    Under the standard passive DCM attitude update:

        C_n_b(t + Δt) ≈ C_n_b(t) Exp([ω_nb^b Δt]_x)

    Therefore the body-frame incremental rotation matrix is:

        ΔC_b ≈ C_n_b(t)^T C_n_b(t + Δt)

    and the corresponding rotation vector is:

        r = log(ΔC_b) ≈ ω_nb^b Δt

    so:

        ω_nb^b ≈ r / Δt

    Reference
    ---------
    The exponential/logarithm rotation-vector machinery is implemented in
    `gravnav.physics.frames.rotvec_from_dcm(...)`, whose docstring cites the
    Rodrigues / SO(3) formulas used.

    This helper is extremely useful when your truth model provides attitude
    samples but not angular-rate truth directly.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")

    C0 = project_to_so3(_mat3(C_n_b_prev, name="C_n_b_prev"))
    C1 = project_to_so3(_mat3(C_n_b_next, name="C_n_b_next"))

    delta_C_body = C0.T @ C1
    rotvec = rotvec_from_dcm(delta_C_body)
    return rotvec / dt_s


@dataclass
class IMUTruthKinematics:
    """
    Container for the ideal inertial quantities implied by truth kinematics.

    Attributes
    ----------
    f_n_mps2 : np.ndarray, shape (3,)
        Ideal accelerometer specific force in NED.
    f_b_mps2 : np.ndarray, shape (3,)
        Ideal accelerometer specific force in body coordinates.
    omega_ie_n_radps : np.ndarray, shape (3,)
        Earth rotation resolved in NED.
    omega_en_n_radps : np.ndarray, shape (3,)
        Transport rate resolved in NED.
    omega_in_n_radps : np.ndarray, shape (3,)
        Total navigation-frame rate with respect to inertial space, resolved in NED.
    omega_nb_b_radps : np.ndarray, shape (3,)
        Body rate with respect to navigation frame, resolved in body coordinates.
    omega_ib_b_radps : np.ndarray, shape (3,)
        Gyro truth measurement: body rate with respect to inertial frame, resolved in body.
    gravity_ned_mps2 : np.ndarray, shape (3,)
        Gravity vector in NED.
    coriolis_transport_ned_mps2 : np.ndarray, shape (3,)
        The acceleration term:
            (2 ω_ie^n + ω_en^n) × v^n
    """

    f_n_mps2: FloatArray
    f_b_mps2: FloatArray
    omega_ie_n_radps: FloatArray
    omega_en_n_radps: FloatArray
    omega_in_n_radps: FloatArray
    omega_nb_b_radps: FloatArray
    omega_ib_b_radps: FloatArray
    gravity_ned_mps2: FloatArray
    coriolis_transport_ned_mps2: FloatArray


def build_imu_truth_kinematics(
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    C_n_b: ArrayLike,
    omega_nb_b_radps: ArrayLike,
    lat_rad: float,
    height_m: float,
    gravity_override_mps2: Optional[float] = None,
) -> IMUTruthKinematics:
    r"""
    Build the full set of ideal IMU quantities implied by truth kinematics.

    This is the main "truth -> ideal IMU" helper for simulation.

    Parameters
    ----------
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity with respect to Earth [m/s^2].
    v_ned_mps : array-like, shape (3,)
        NED velocity with respect to Earth [m/s].
    C_n_b : array-like, shape (3, 3)
        Passive DCM mapping body -> NED:
            v^n = C_n_b v^b
    omega_nb_b_radps : array-like, shape (3,)
        Body angular rate with respect to NED, resolved in body [rad/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    gravity_override_mps2 : float, optional
        Optional scalar gravity magnitude override.

    Returns
    -------
    IMUTruthKinematics
        Structured ideal IMU truth quantities.

    Notes
    -----
    This function is intentionally verbose and explicit so that later notebooks
    and test cases can inspect intermediate physical terms:
    - gravity
    - Coriolis/transport coupling
    - body-vs-inertial angular rate separation
    """
    v_dot_n = _vec3(v_dot_ned_mps2, name="v_dot_ned_mps2")
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    C_n_b = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    omega_nb_b = _vec3(omega_nb_b_radps, name="omega_nb_b_radps")

    omega_ie_n = earth_rate_ned(lat_rad)
    omega_en_n = transport_rate_ned(lat_rad, height_m, v_n)
    omega_in_n = omega_ie_n + omega_en_n
    gravity_n = gravity_vector_ned(lat_rad, height_m, gravity_override_mps2)
    coriolis_transport_n = np.cross(2.0 * omega_ie_n + omega_en_n, v_n)

    f_n = v_dot_n + coriolis_transport_n - gravity_n
    f_b = C_n_b.T @ f_n
    omega_ib_b = omega_nb_b + C_n_b.T @ omega_in_n

    return IMUTruthKinematics(
        f_n_mps2=f_n,
        f_b_mps2=f_b,
        omega_ie_n_radps=omega_ie_n,
        omega_en_n_radps=omega_en_n,
        omega_in_n_radps=omega_in_n,
        omega_nb_b_radps=omega_nb_b,
        omega_ib_b_radps=omega_ib_b,
        gravity_ned_mps2=gravity_n,
        coriolis_transport_ned_mps2=coriolis_transport_n,
    )


def build_interval_imu_truth_kinematics(
    v_ned_prev_mps: ArrayLike,
    v_ned_next_mps: ArrayLike,
    C_n_b_prev: ArrayLike,
    C_n_b_next: ArrayLike,
    lat_prev_rad: float,
    height_prev_m: float,
    dt_s: float,
    gravity_override_mps2: Optional[float] = None,
) -> IMUTruthKinematics:
    r"""
    Build ideal IMU quantities over one propagation interval.

    Why this exists
    ---------------
    `build_imu_truth_kinematics(...)` is a pointwise helper: it assumes sampled
    truth kinematics are already known at one instant. The INS runner, however,
    performs *interval* propagation from sample `k-1` to sample `k`. For that
    use case, an interval-consistent IMU construction is numerically much more
    stable because it derives:

    - body rate from the consecutive attitudes over the interval
    - velocity derivative from the consecutive velocities over the interval
    - Earth-rate / transport / gravity terms from the interval start state

    This makes the synthetic IMU stream consistent with the repository's
    discrete prediction step.

    Parameters
    ----------
    v_ned_prev_mps, v_ned_next_mps : array-like, shape (3,)
        Consecutive NED velocities at the start and end of the interval [m/s].
    C_n_b_prev, C_n_b_next : array-like, shape (3, 3)
        Consecutive body->NED DCMs.
    lat_prev_rad : float
        Geodetic latitude at the interval start [rad].
    height_prev_m : float
        Ellipsoidal height at the interval start [m].
    dt_s : float
        Interval duration [s].
    gravity_override_mps2 : float, optional
        Optional scalar gravity magnitude override.

    Returns
    -------
    IMUTruthKinematics
        Interval-consistent ideal IMU quantities resolved at the interval start.
    """
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")

    v_prev = _vec3(v_ned_prev_mps, name="v_ned_prev_mps")
    v_next = _vec3(v_ned_next_mps, name="v_ned_next_mps")
    C_prev = project_to_so3(_mat3(C_n_b_prev, name="C_n_b_prev"))
    C_next = project_to_so3(_mat3(C_n_b_next, name="C_n_b_next"))

    omega_ie_n = earth_rate_ned(lat_prev_rad)
    omega_en_n = transport_rate_ned(lat_prev_rad, height_prev_m, v_prev)
    omega_in_n = omega_ie_n + omega_en_n

    omega_nb_b = body_rate_from_consecutive_dcms(
        C_n_b_prev=C_prev,
        C_n_b_next=C_next,
        dt_s=dt,
    )
    gravity_n = gravity_vector_ned(
        lat_prev_rad,
        height_prev_m,
        gravity_override_mps2,
    )
    coriolis_transport_n = np.cross(2.0 * omega_ie_n + omega_en_n, v_prev)
    v_dot_n = (v_next - v_prev) / dt

    f_n = v_dot_n + coriolis_transport_n - gravity_n
    f_b = C_prev.T @ f_n
    omega_ib_b = omega_nb_b + C_prev.T @ omega_in_n

    return IMUTruthKinematics(
        f_n_mps2=f_n.astype(np.float64),
        f_b_mps2=f_b.astype(np.float64),
        omega_ie_n_radps=omega_ie_n.astype(np.float64),
        omega_en_n_radps=omega_en_n.astype(np.float64),
        omega_in_n_radps=omega_in_n.astype(np.float64),
        omega_nb_b_radps=omega_nb_b.astype(np.float64),
        omega_ib_b_radps=omega_ib_b.astype(np.float64),
        gravity_ned_mps2=gravity_n.astype(np.float64),
        coriolis_transport_ned_mps2=coriolis_transport_n.astype(np.float64),
    )


@dataclass
class IMUSpec:
    """
    First-order IMU specification for simulation.

    This dataclass intentionally keeps the model simple, explicit, and easy to
    map from either a datasheet or an Allan-variance fit.

    Parameters
    ----------
    gyro_noise_density_radps_per_sqrt_hz : scalar or shape (3,), default=0
        Gyroscope white-noise density.
    accel_noise_density_mps2_per_sqrt_hz : scalar or shape (3,), default=0
        Accelerometer white-noise density.
    gyro_bias_random_walk_radps_per_sqrt_s : scalar or shape (3,), default=0
        Gyroscope in-run bias random walk coefficient, using the explicit discrete
        bias-step unit:
            [rad/s] / sqrt(s)
    accel_bias_random_walk_mps2_per_sqrt_s : scalar or shape (3,), default=0
        Accelerometer in-run bias random walk coefficient, using:
            [m/s^2] / sqrt(s)
    gyro_turn_on_bias_std_radps : scalar or shape (3,), default=0
        1-sigma turn-on bias uncertainty applied when the IMU is reset.
    accel_turn_on_bias_std_mps2 : scalar or shape (3,), default=0
        1-sigma turn-on bias uncertainty applied when the IMU is reset.
    gyro_fixed_bias_radps : scalar or shape (3,), default=0
        Fixed gyroscope bias term added to all measurements.
    accel_fixed_bias_mps2 : scalar or shape (3,), default=0
        Fixed accelerometer bias term added to all measurements.
    gyro_scale_misalignment_matrix : array-like, shape (3,3), optional
        Linear input matrix applied to the ideal gyro signal before biases/noise.
        Default is identity.
    accel_scale_misalignment_matrix : array-like, shape (3,3), optional
        Linear input matrix applied to the ideal accelerometer signal before
        biases/noise. Default is identity.
    gyro_max_abs_radps : scalar or shape (3,), default=np.inf
        Per-axis saturation limit for gyro output.
    accel_max_abs_mps2 : scalar or shape (3,), default=np.inf
        Per-axis saturation limit for accelerometer output.
    name : str, default="imu"
        Human-readable identifier.

    Stochastic model
    ----------------
    Each sensor channel is modeled as:

        y_k = M y_true,k + b_fixed + b_turnon + b_rw,k + n_k

    where:
    - M is the scale/misalignment matrix
    - b_fixed is a constant bias
    - b_turnon is sampled once per reset
    - b_rw,k is a random-walk bias state
    - n_k is additive white Gaussian noise

    References
    ----------
    - Kalibr IMU noise model for the white-noise + bias-random-walk structure:
      https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
    - The scale/misalignment matrix is a standard first-order deterministic
      sensor-error model used in calibration and simulation practice.
    """

    gyro_noise_density_radps_per_sqrt_hz: ArrayLike | float = 0.0
    accel_noise_density_mps2_per_sqrt_hz: ArrayLike | float = 0.0

    gyro_bias_random_walk_radps_per_sqrt_s: ArrayLike | float = 0.0
    accel_bias_random_walk_mps2_per_sqrt_s: ArrayLike | float = 0.0

    gyro_turn_on_bias_std_radps: ArrayLike | float = 0.0
    accel_turn_on_bias_std_mps2: ArrayLike | float = 0.0

    gyro_fixed_bias_radps: ArrayLike | float = 0.0
    accel_fixed_bias_mps2: ArrayLike | float = 0.0

    gyro_scale_misalignment_matrix: Optional[ArrayLike] = None
    accel_scale_misalignment_matrix: Optional[ArrayLike] = None

    gyro_max_abs_radps: ArrayLike | float = np.inf
    accel_max_abs_mps2: ArrayLike | float = np.inf

    name: str = "imu"

    def __post_init__(self) -> None:
        self.gyro_noise_density_radps_per_sqrt_hz = _nonnegative_axis3(
            self.gyro_noise_density_radps_per_sqrt_hz,
            name="gyro_noise_density_radps_per_sqrt_hz",
        )
        self.accel_noise_density_mps2_per_sqrt_hz = _nonnegative_axis3(
            self.accel_noise_density_mps2_per_sqrt_hz,
            name="accel_noise_density_mps2_per_sqrt_hz",
        )

        self.gyro_bias_random_walk_radps_per_sqrt_s = _nonnegative_axis3(
            self.gyro_bias_random_walk_radps_per_sqrt_s,
            name="gyro_bias_random_walk_radps_per_sqrt_s",
        )
        self.accel_bias_random_walk_mps2_per_sqrt_s = _nonnegative_axis3(
            self.accel_bias_random_walk_mps2_per_sqrt_s,
            name="accel_bias_random_walk_mps2_per_sqrt_s",
        )

        self.gyro_turn_on_bias_std_radps = _nonnegative_axis3(
            self.gyro_turn_on_bias_std_radps,
            name="gyro_turn_on_bias_std_radps",
        )
        self.accel_turn_on_bias_std_mps2 = _nonnegative_axis3(
            self.accel_turn_on_bias_std_mps2,
            name="accel_turn_on_bias_std_mps2",
        )

        self.gyro_fixed_bias_radps = _axis3(
            self.gyro_fixed_bias_radps,
            name="gyro_fixed_bias_radps",
        )
        self.accel_fixed_bias_mps2 = _axis3(
            self.accel_fixed_bias_mps2,
            name="accel_fixed_bias_mps2",
        )

        self.gyro_scale_misalignment_matrix = _maybe_identity_matrix(
            self.gyro_scale_misalignment_matrix,
            name="gyro_scale_misalignment_matrix",
        )
        self.accel_scale_misalignment_matrix = _maybe_identity_matrix(
            self.accel_scale_misalignment_matrix,
            name="accel_scale_misalignment_matrix",
        )

        self.gyro_max_abs_radps = _max_abs_axis3(
            self.gyro_max_abs_radps,
            name="gyro_max_abs_radps",
        )
        self.accel_max_abs_mps2 = _max_abs_axis3(
            self.accel_max_abs_mps2,
            name="accel_max_abs_mps2",
        )

    @classmethod
    def perfect(cls, name: str = "perfect_imu") -> "IMUSpec":
        """
        Return a perfect noise-free, bias-free, unbounded IMU specification.
        """
        return cls(name=name)

    def gyro_white_noise_std(self, dt_s: float) -> FloatArray:
        """
        Per-sample gyroscope white-noise standard deviation [rad/s].
        """
        return discrete_white_noise_std_from_density(
            self.gyro_noise_density_radps_per_sqrt_hz,
            dt_s,
        )

    def accel_white_noise_std(self, dt_s: float) -> FloatArray:
        """
        Per-sample accelerometer white-noise standard deviation [m/s^2].
        """
        return discrete_white_noise_std_from_density(
            self.accel_noise_density_mps2_per_sqrt_hz,
            dt_s,
        )

    def gyro_bias_step_std(self, dt_s: float) -> FloatArray:
        """
        One-step gyroscope bias random-walk increment standard deviation [rad/s].
        """
        return discrete_random_walk_step_std(
            self.gyro_bias_random_walk_radps_per_sqrt_s,
            dt_s,
        )

    def accel_bias_step_std(self, dt_s: float) -> FloatArray:
        """
        One-step accelerometer bias random-walk increment standard deviation [m/s^2].
        """
        return discrete_random_walk_step_std(
            self.accel_bias_random_walk_mps2_per_sqrt_s,
            dt_s,
        )


@dataclass
class IMUBiasState:
    """
    State of the time-varying IMU bias terms.

    Attributes
    ----------
    gyro_bias_radps : np.ndarray, shape (3,)
        Current gyroscope bias state.
    accel_bias_mps2 : np.ndarray, shape (3,)
        Current accelerometer bias state.

    Interpretation
    --------------
    This stores the combined bias contribution used *during the run*:
    - fixed bias
    - turn-on bias realization
    - accumulated random-walk component

    Keeping it explicit makes it easier to:
    - inspect bias growth in notebooks
    - compare truth vs estimated filter states later
    - reset / reseed turn-on conditions reproducibly
    """

    gyro_bias_radps: FloatArray
    accel_bias_mps2: FloatArray

    def copy(self) -> "IMUBiasState":
        """Deep copy of the bias state."""
        return IMUBiasState(
            gyro_bias_radps=self.gyro_bias_radps.copy(),
            accel_bias_mps2=self.accel_bias_mps2.copy(),
        )


@dataclass
class IMUMeasurement:
    """
    Container for a single simulated IMU sample.

    Attributes
    ----------
    time_s : float or None
        Optional sample timestamp [s].
    omega_ib_b_radps : np.ndarray, shape (3,)
        Simulated gyroscope output.
    f_ib_b_mps2 : np.ndarray, shape (3,)
        Simulated accelerometer output (specific force).
    ideal_omega_ib_b_radps : np.ndarray, shape (3,)
        Ideal noiseless gyroscope signal before corruption.
    ideal_f_ib_b_mps2 : np.ndarray, shape (3,)
        Ideal noiseless accelerometer signal before corruption.
    gyro_bias_used_radps : np.ndarray, shape (3,)
        Bias state applied to this sample.
    accel_bias_used_mps2 : np.ndarray, shape (3,)
        Bias state applied to this sample.
    gyro_white_noise_radps : np.ndarray, shape (3,)
        Additive gyro white-noise realization for this sample.
    accel_white_noise_mps2 : np.ndarray, shape (3,)
        Additive accelerometer white-noise realization for this sample.
    gyro_saturated : bool
        True if gyro clipping occurred on at least one axis.
    accel_saturated : bool
        True if accelerometer clipping occurred on at least one axis.
    """

    time_s: Optional[float]
    omega_ib_b_radps: FloatArray
    f_ib_b_mps2: FloatArray
    ideal_omega_ib_b_radps: FloatArray
    ideal_f_ib_b_mps2: FloatArray
    gyro_bias_used_radps: FloatArray
    accel_bias_used_mps2: FloatArray
    gyro_white_noise_radps: FloatArray
    accel_white_noise_mps2: FloatArray
    gyro_saturated: bool
    accel_saturated: bool


class IMUSensor:
    """
    Stateful IMU simulator.

    This class owns:
    - a specification (`IMUSpec`)
    - a random-number generator
    - the current in-run bias state

    Typical usage
    -------------
    1) Create the IMU with a spec and RNG seed.
    2) Call `reset()` once per simulated power cycle / mission start.
    3) At each step:
       - generate ideal truth IMU quantities from the trajectory
       - call `measure_from_truth(...)` or `measure_from_ideal(...)`

    Notes
    -----
    - Bias random walk is updated *after* generating the current sample, so the
      measurement uses the bias state as it existed at the start of the interval.
      This is a standard, simple Euler-Maruyama style convention.
    - If you later want time-correlated colored noise or temperature-driven bias,
      subclass or extend this class instead of modifying the physics equations.
    """

    def __init__(
        self,
        spec: IMUSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng
        self.bias_state = IMUBiasState(
            gyro_bias_radps=np.zeros(3, dtype=np.float64),
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
        )
        self.reset()

    def reset(
        self,
        *,
        turn_on_bias_randomized: bool = True,
        gyro_bias_override_radps: Optional[ArrayLike] = None,
        accel_bias_override_mps2: Optional[ArrayLike] = None,
    ) -> None:
        """
        Reset the IMU bias state.

        Parameters
        ----------
        turn_on_bias_randomized : bool, default=True
            If True, sample a new turn-on bias realization using the turn-on bias
            standard deviations in the spec.
        gyro_bias_override_radps : array-like, optional
            If provided, use this as the complete initial gyroscope bias state.
        accel_bias_override_mps2 : array-like, optional
            If provided, use this as the complete initial accelerometer bias state.

        Behavior
        --------
        If overrides are provided, they take precedence over the standard
        fixed-bias + turn-on-bias initialization.
        """
        if gyro_bias_override_radps is not None:
            gyro_bias = _vec3(gyro_bias_override_radps, name="gyro_bias_override_radps")
        else:
            gyro_bias = self.spec.gyro_fixed_bias_radps.copy()
            if turn_on_bias_randomized:
                gyro_bias += self.spec.gyro_turn_on_bias_std_radps * _randn3(self.rng)

        if accel_bias_override_mps2 is not None:
            accel_bias = _vec3(accel_bias_override_mps2, name="accel_bias_override_mps2")
        else:
            accel_bias = self.spec.accel_fixed_bias_mps2.copy()
            if turn_on_bias_randomized:
                accel_bias += self.spec.accel_turn_on_bias_std_mps2 * _randn3(self.rng)

        self.bias_state = IMUBiasState(
            gyro_bias_radps=gyro_bias,
            accel_bias_mps2=accel_bias,
        )

    def _step_bias_random_walk(self, dt_s: float) -> None:
        """
        Evolve the in-run gyro and accelerometer biases by one time step.

        Formula
        -------
        Using the standard Brownian / random-walk model:

            b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
            w_k ~ N(0, I)

        Reference
        ---------
        Kalibr Wiki, "IMU Noise Model":
        https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        self.bias_state.gyro_bias_radps = (
            self.bias_state.gyro_bias_radps
            + self.spec.gyro_bias_step_std(dt_s) * _randn3(self.rng)
        )
        self.bias_state.accel_bias_mps2 = (
            self.bias_state.accel_bias_mps2
            + self.spec.accel_bias_step_std(dt_s) * _randn3(self.rng)
        )

    def measure_from_ideal(
        self,
        ideal_omega_ib_b_radps: ArrayLike,
        ideal_f_ib_b_mps2: ArrayLike,
        dt_s: float,
        *,
        time_s: Optional[float] = None,
    ) -> IMUMeasurement:
        r"""
        Corrupt ideal IMU inputs into a simulated IMU sample.

        Parameters
        ----------
        ideal_omega_ib_b_radps : array-like, shape (3,)
            Ideal gyroscope input ω_ib^b [rad/s].
        ideal_f_ib_b_mps2 : array-like, shape (3,)
            Ideal accelerometer input f_ib^b [m/s^2].
        dt_s : float
            Sample interval [s].
        time_s : float, optional
            Sample timestamp [s].

        Returns
        -------
        IMUMeasurement
            Simulated sample containing both corrupted outputs and the latent
            ideal/noise/bias terms used.

        Model
        -----
        Gyro:
            y_g = M_g ω_ib^b + b_g + n_g

        Accel:
            y_a = M_a f_ib^b + b_a + n_a

        where:
        - M_g, M_a are deterministic scale/misalignment matrices
        - b_g, b_a are the current bias states
        - n_g, n_a are zero-mean white Gaussian noise realizations

        White noise
        -----------
        Per-sample white-noise standard deviation is computed from the specified
        continuous-time noise density using:

            sigma_d = sigma / sqrt(dt)

        Bias evolution
        --------------
        After generating the sample, the in-run bias state is advanced by:

            b_k = b_{k-1} + sigma_rw sqrt(dt) w_k

        Reference
        ---------
        Kalibr Wiki, "IMU Noise Model":
        https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        ideal_omega = _vec3(ideal_omega_ib_b_radps, name="ideal_omega_ib_b_radps")
        ideal_f = _vec3(ideal_f_ib_b_mps2, name="ideal_f_ib_b_mps2")

        gyro_white_std = self.spec.gyro_white_noise_std(dt_s)
        accel_white_std = self.spec.accel_white_noise_std(dt_s)

        gyro_white = gyro_white_std * _randn3(self.rng)
        accel_white = accel_white_std * _randn3(self.rng)

        gyro_bias_used = self.bias_state.gyro_bias_radps.copy()
        accel_bias_used = self.bias_state.accel_bias_mps2.copy()

        omega_meas = (
            self.spec.gyro_scale_misalignment_matrix @ ideal_omega
            + gyro_bias_used
            + gyro_white
        )
        accel_meas = (
            self.spec.accel_scale_misalignment_matrix @ ideal_f
            + accel_bias_used
            + accel_white
        )

        omega_clipped = _clip_per_axis(omega_meas, self.spec.gyro_max_abs_radps)
        accel_clipped = _clip_per_axis(accel_meas, self.spec.accel_max_abs_mps2)

        gyro_saturated = not np.allclose(omega_clipped, omega_meas)
        accel_saturated = not np.allclose(accel_clipped, accel_meas)

        sample = IMUMeasurement(
            time_s=time_s,
            omega_ib_b_radps=omega_clipped,
            f_ib_b_mps2=accel_clipped,
            ideal_omega_ib_b_radps=ideal_omega,
            ideal_f_ib_b_mps2=ideal_f,
            gyro_bias_used_radps=gyro_bias_used,
            accel_bias_used_mps2=accel_bias_used,
            gyro_white_noise_radps=gyro_white,
            accel_white_noise_mps2=accel_white,
            gyro_saturated=gyro_saturated,
            accel_saturated=accel_saturated,
        )

        self._step_bias_random_walk(dt_s)
        return sample

    def measure_from_truth(
        self,
        v_dot_ned_mps2: ArrayLike,
        v_ned_mps: ArrayLike,
        C_n_b: ArrayLike,
        omega_nb_b_radps: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        *,
        gravity_override_mps2: Optional[float] = None,
        time_s: Optional[float] = None,
    ) -> IMUMeasurement:
        """
        High-level convenience function:
        truth kinematics -> ideal IMU -> corrupted IMU sample.

        Parameters
        ----------
        v_dot_ned_mps2 : array-like, shape (3,)
            Time derivative of NED velocity with respect to Earth [m/s^2].
        v_ned_mps : array-like, shape (3,)
            NED velocity with respect to Earth [m/s].
        C_n_b : array-like, shape (3, 3)
            Passive body->NED DCM.
        omega_nb_b_radps : array-like, shape (3,)
            Body angular rate with respect to NED, resolved in body [rad/s].
        lat_rad : float
            Geodetic latitude [rad].
        height_m : float
            Ellipsoidal height [m].
        dt_s : float
            Sample interval [s].
        gravity_override_mps2 : float, optional
            Optional scalar gravity magnitude override.
        time_s : float, optional
            Sample timestamp [s].

        Returns
        -------
        IMUMeasurement
            Corrupted IMU sample.

        Notes
        -----
        This is the main entry point most scenario runners will use once they
        have truth trajectory and attitude available.
        """
        truth = build_imu_truth_kinematics(
            v_dot_ned_mps2=v_dot_ned_mps2,
            v_ned_mps=v_ned_mps,
            C_n_b=C_n_b,
            omega_nb_b_radps=omega_nb_b_radps,
            lat_rad=lat_rad,
            height_m=height_m,
            gravity_override_mps2=gravity_override_mps2,
        )

        return self.measure_from_ideal(
            ideal_omega_ib_b_radps=truth.omega_ib_b_radps,
            ideal_f_ib_b_mps2=truth.f_b_mps2,
            dt_s=dt_s,
            time_s=time_s,
        )

    def current_bias_state(self) -> IMUBiasState:
        """
        Return a copy of the current IMU bias state.
        """
        return self.bias_state.copy()


__all__ = [
    "FloatArray",
    "IMUBiasState",
    "IMUMeasurement",
    "IMUSensor",
    "IMUSpec",
    "IMUTruthKinematics",
    "body_rate_from_consecutive_attitudes",
    "build_interval_imu_truth_kinematics",
    "build_imu_truth_kinematics",
    "coriolis_transport_acceleration_ned",
    "discrete_random_walk_step_std",
    "discrete_white_noise_std_from_density",
    "gravity_vector_ned",
    "ideal_gyro_rate_body",
    "ideal_specific_force_body",
    "ideal_specific_force_ned",
]
