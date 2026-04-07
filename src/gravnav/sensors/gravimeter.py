"""
gravimeter.py

Scalar gravimeter physics and sensor models for the gravity-aided navigation
simulator.

This module is intentionally centered on *navigation-useful* gravimeter outputs,
not on pitch-deck-friendly but physically ambiguous "g-sensor" abstractions.

It provides three related layers:

1) Geodetic/gravity quantities
   - normal gravity at the measurement point
   - scalar gravity disturbance:
         δg(P) = g(P) - γ(P)
   - height / free-air style corrections between measurement levels

2) Moving-base gravimetry relations
   - exact local-level (NED) recovery of vertical gravity and gravity disturbance
     from specific force, vertical acceleration, and the Coriolis / transport term
   - a classical Harlan-style Eötvös approximation for comparison / validation

3) Scalar sensor simulation
   - white noise
   - bias random walk
   - turn-on bias
   - fixed bias
   - scale factor error
   - optional finite bandwidth
   - optional motion-coupling residuals driven by body specific force and body
     angular rate

Why this file is written this way
---------------------------------
For gravity-aided navigation, the quantity that is usually most useful in the
simulator is not "absolute g on the ellipsoid" and not a geoid-reduced anomaly
that requires downward continuation, but the *same-point* measurement-level
gravity disturbance:

    δg(P) = g(P) - γ(P)

where:
- g(P) is the measured scalar gravity magnitude at the observation point P
- γ(P) is the normal gravity magnitude at the same point P

This follows the geodetic definition of scalar gravity disturbance and avoids
injecting unnecessary downward-continuation assumptions into the simulator.

For moving-base gravimetry, a classical up-positive reduction formula often
appears as:

    δg = f_U - vdot_U + δa_E - γ

where:
- f_U     : specific force along local Up (or instrument vertical)
- vdot_U  : vertical acceleration
- δa_E    : Eötvös correction
- γ       : normal gravity

In this repository we use NED with Down positive. The exact local-level velocity
equation is:

    vdot^n = f^n - (2 ω_ie^n + ω_en^n) × v^n + g^n

Therefore, taking the Down component gives:

    g_D = vdot_D - f_D + c_D
    c_D = ((2 ω_ie^n + ω_en^n) × v^n)_D

and the recovered gravity disturbance becomes:

    δg = g_D - γ = vdot_D - f_D + c_D - γ

This is the sign-consistent NED form used below.

Primary references used here
----------------------------
1) Hackney, R. I., and Featherstone, W. E. (2003),
   "Geodetic versus geophysical perspectives of the 'gravity anomaly'"
   Geophysical Journal International, 154(1), 35-43.
   URL:
   https://academic.oup.com/gji/article/154/1/35/604237

   Used for:
   - distinction between gravity anomaly and gravity disturbance
   - scalar gravity disturbance defined at the observation point:
         δg(P) = g(P) - γ(P)
   - the point that γ(P) should be evaluated at the same point using ellipsoidal
     height and the normal-gravity vertical gradient ("free-air correction")

2) Harmonica / Fatiando a Terra documentation, "Gravity Disturbance"
   URL:
   https://www.fatiando.org/harmonica/latest/user_guide/gravity_disturbance.html

   Used as a concise readable statement of:
       δg(p) = g(p) - γ(p)
   where both are evaluated at the same point.

3) Shi et al. (2021),
   "Experimental study on improving the accuracy of marine gravimetry by
   combining moving-base gravimeters with GNSS antenna array"
   Earth, Planets and Space, 73, 177.
   URL:
   https://link.springer.com/article/10.1186/s40623-021-01498-x

   Used for:
   - classical moving-base gravity anomaly formula
   - Harlan-style Eötvös correction formula
   - practical reminder that Eötvös, vertical acceleration, and free-air
     corrections dominate moving-platform reduction

   Key equations quoted in the paper:
       δg = f_U - vdot_U + δa_E - γ
       δa_E = (1 + h/a)(2 ω v_E cosφ + v^2/a)
              - (f/a)(v^2 - cos^2φ (3 v^2 - 2 v_E^2))

4) WGS 84 / NGA standard
   Implemented in this repository in `gravnav.physics.earth`.
   The current file uses:
   - normal_gravity(lat, h)
   - the WGS 84 ellipsoid constants
   - exact measurement-level normal gravity rather than a crude linear 0.3086 mGal/m
     unless a comparison helper is explicitly requested.

5) Jensen et al. (2025),
   "Airborne gravimetry with quantum technology: observations from Iceland and Greenland"
   Earth System Science Data, 17, 1667-1684.
   URL:
   https://essd.copernicus.org/articles/17/1667/2025/

   Used for the modern practical distinction:
   - classical airborne gravimeters measure relative gravity variation and need
     drift handling
   - quantum gravimeters can provide direct absolute gravity measurements
   In this module, both are simulated by the same sensor class via different
   bias/drift settings.

Important modeling choices
--------------------------
- Internal units are always SI:
    gravity, acceleration, specific force: m/s^2
- mGal conversion is provided for reporting / debugging only.
- The default "navigation observation" in this module is the scalar gravity
  disturbance at the measurement point, NOT a geoid-reduced anomaly.
- The moving-base exact recovery uses the local-level NED mechanization relation.
- The Harlan Eötvös approximation is included for comparison and sanity checks,
  but the exact NED kinematic recovery should be preferred inside the simulator.
- Motion-coupling residuals are modeled explicitly as optional additive terms
  driven by body specific force and body angular rate; this is where you can
  later inject more realistic photonic / mechanical cross-coupling.

What this file is for
---------------------
This is the correct next layer after:
- `gravnav.physics.earth`
- `gravnav.physics.frames`
- `gravnav.sensors.imu`

because the later simulation stack needs a clean answer to:
- what is the ideal gravity disturbance at the observation point?
- how do I recover gravity from moving-platform kinematics?
- how do I inject realistic sensor noise / drift / motion residuals?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import WGS84, mgal_to_mps2, mps2_to_mgal, normal_gravity
from ..physics.frames import earth_rate_ned, project_to_so3, transport_rate_ned

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _scalar(x: ArrayLike, *, name: str) -> float:
    """
    Validate and return a scalar float.

    Parameters
    ----------
    x : array-like or scalar
        Input to validate.
    name : str
        Name for error messages.
    """
    arr = _as_float_array(x)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}.")
    return float(arr)


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a 3-vector.
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


def _scalar_or_vec3_to_vec3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Convert a scalar or 3-vector input into a 3-vector.

    This is useful for per-axis motion-coupling parameters:
    - scalar -> broadcast to all 3 axes
    - 3-vector -> keep as-is
    """
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}.")
    return arr


def scalar_white_noise_std_from_density(
    noise_density_per_sqrt_hz: float,
    dt_s: float,
) -> float:
    r"""
    Convert a continuous-time white-noise density into a discrete-time scalar
    sample standard deviation.

    Parameters
    ----------
    noise_density_per_sqrt_hz : float
        Continuous-time white-noise density in measurement units / sqrt(Hz).
        For a gravimeter this is usually m/s^2 / sqrt(Hz), though some
        specifications are often reported in μGal/√Hz or mGal/√Hz.
    dt_s : float
        Sample interval [s].

    Returns
    -------
    float
        Per-sample discrete-time white-noise standard deviation in the same
        physical units as the measurement itself.

    Formula
    -------
    Following the standard discrete-time approximation used in IMU modeling:

        sigma_d = sigma / sqrt(dt)

    This assumes ideal anti-alias filtering / ideal decimation prior to sampling.

    Reference
    ---------
    This is the same engineering discretization used in Kalibr-style sensor
    simulation and is consistent with the repository's IMU noise model.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")
    sigma = float(noise_density_per_sqrt_hz)
    if sigma < 0.0:
        raise ValueError(
            f"noise_density_per_sqrt_hz must be nonnegative, got {sigma}."
        )
    return sigma / np.sqrt(dt_s)


def scalar_random_walk_step_std(
    random_walk_per_sqrt_s: float,
    dt_s: float,
) -> float:
    r"""
    Convert a scalar bias random-walk coefficient into the standard deviation of
    the one-step bias increment.

    Parameters
    ----------
    random_walk_per_sqrt_s : float
        Bias random-walk coefficient in:
            [measurement units] / sqrt(s)
    dt_s : float
        Sample interval [s].

    Returns
    -------
    float
        Standard deviation of the one-step bias increment.

    Formula
    -------
    Using the usual Brownian / Wiener bias model:

        b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
        w_k ~ N(0, 1)

    so the one-step standard deviation is:

        sigma_step = sigma_rw * sqrt(dt)

    Reference
    ---------
    Standard stochastic sensor discretization; identical in spirit to the
    Kalibr bias-random-walk model used in `imu.py`.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}.")
    sigma = float(random_walk_per_sqrt_s)
    if sigma < 0.0:
        raise ValueError(
            f"random_walk_per_sqrt_s must be nonnegative, got {sigma}."
        )
    return sigma * np.sqrt(dt_s)


def gravity_magnitude_from_vector_ned(total_gravity_ned_mps2: ArrayLike) -> float:
    r"""
    Return the scalar gravity magnitude from a gravity vector resolved in NED.

    Parameters
    ----------
    total_gravity_ned_mps2 : array-like, shape (3,)
        True gravity vector resolved in NED [m/s^2].

    Returns
    -------
    float
        Scalar gravity magnitude |g| [m/s^2].

    Formula
    -------
        g = ||g^n||_2

    Notes
    -----
    This should be used only when the provided vector is the actual gravity vector
    (gravitational + centrifugal field), not specific force and not total vehicle
    acceleration.
    """
    g_n = _vec3(total_gravity_ned_mps2, name="total_gravity_ned_mps2")
    return float(np.linalg.norm(g_n))


def gravity_disturbance_from_scalar_gravity(
    measured_gravity_mps2: float,
    lat_rad: float,
    height_m: float,
) -> float:
    r"""
    Compute the scalar gravity disturbance at the observation point.

    Parameters
    ----------
    measured_gravity_mps2 : float
        Scalar measured gravity magnitude g(P) [m/s^2].
    lat_rad : float
        Geodetic latitude of the observation point [rad].
    height_m : float
        Ellipsoidal height of the observation point [m].

    Returns
    -------
    float
        Scalar gravity disturbance δg(P) [m/s^2].

    Formula
    -------
        δg(P) = g(P) - γ(P)

    where:
    - g(P) is the measured scalar gravity magnitude at point P
    - γ(P) is the normal gravity magnitude evaluated at the same point P

    References
    ----------
    1) Hackney & Featherstone (2003), GJI 154(1), Section 2.2:
       scalar gravity disturbance is defined using measured gravity and normal
       gravity at the same point, with γ(P) computed at the observation point
       using ellipsoidal height.
    2) Harmonica documentation:
       https://www.fatiando.org/harmonica/latest/user_guide/gravity_disturbance.html
    """
    g = float(measured_gravity_mps2)
    gamma_p = float(normal_gravity(lat_rad, height_m))
    return g - gamma_p


def gravity_disturbance_from_total_gravity_vector_ned(
    total_gravity_ned_mps2: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> float:
    r"""
    Compute the scalar gravity disturbance from the full gravity vector resolved
    in NED coordinates.

    Parameters
    ----------
    total_gravity_ned_mps2 : array-like, shape (3,)
        True gravity vector resolved in NED [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].

    Returns
    -------
    float
        Scalar gravity disturbance δg(P) [m/s^2].

    Formula
    -------
        δg(P) = ||g^n(P)|| - γ(P)

    where γ(P) is evaluated at the same point using WGS 84 normal gravity.

    References
    ----------
    Same-point scalar gravity disturbance definition:
    - Hackney & Featherstone (2003), Section 2.2
    - Harmonica documentation
    """
    g = gravity_magnitude_from_vector_ned(total_gravity_ned_mps2)
    return gravity_disturbance_from_scalar_gravity(g, lat_rad, height_m)


def vertical_gravity_disturbance_approx_from_vector_ned(
    total_gravity_ned_mps2: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> float:
    r"""
    Approximate the gravity disturbance using the Down component of the gravity
    vector in NED.

    Parameters
    ----------
    total_gravity_ned_mps2 : array-like, shape (3,)
        True gravity vector resolved in NED [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].

    Returns
    -------
    float
        Approximate vertical gravity disturbance [m/s^2].

    Formula
    -------
        δg_D ≈ g_D(P) - γ(P)

    where:
    - g_D is the Down component of the gravity vector in NED
    - γ(P) is the scalar normal gravity at the same point

    Notes
    -----
    This is a useful approximation for local-level scalar gravimetry when the
    horizontal gravity components are tiny compared with the Down component.
    It is not mathematically identical to the scalar gravity disturbance based
    on the vector magnitude.

    This approximation is often practically fine for navigation simulation,
    but the exact scalar disturbance should be preferred when the full gravity
    vector is available.
    """
    g_n = _vec3(total_gravity_ned_mps2, name="total_gravity_ned_mps2")
    gamma_p = float(normal_gravity(lat_rad, height_m))
    return float(g_n[2] - gamma_p)


def normal_gravity_change_between_heights(
    lat_rad: float,
    height_from_m: float,
    height_to_m: float,
) -> float:
    r"""
    Change in WGS 84 normal gravity between two ellipsoidal heights.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_from_m : float
        Initial ellipsoidal height [m].
    height_to_m : float
        Final ellipsoidal height [m].

    Returns
    -------
    float
        γ(lat, h_to) - γ(lat, h_from) [m/s^2].

    Formula
    -------
        Δγ = γ(φ, h_to) - γ(φ, h_from)

    Notes
    -----
    This function is the cleanest way to compute the exact measurement-level
    normal-gravity difference between two heights using the same WGS 84 model
    as the rest of the repository.

    It is preferable to a crude constant 0.3086 mGal/m approximation when you
    already have a WGS 84 normal-gravity implementation available.
    """
    gamma_from = float(normal_gravity(lat_rad, height_from_m))
    gamma_to = float(normal_gravity(lat_rad, height_to_m))
    return gamma_to - gamma_from


def free_air_correction_to_reference_height(
    lat_rad: float,
    measurement_height_m: float,
    reference_height_m: float,
) -> float:
    r"""
    Compute the normal-gravity free-air style correction needed to compare a
    measurement made at one ellipsoidal height to a reference height.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    measurement_height_m : float
        Ellipsoidal height of the observation point [m].
    reference_height_m : float
        Ellipsoidal height of the reference level [m].

    Returns
    -------
    float
        Correction Δγ_ref [m/s^2] such that:

            γ(reference) = γ(measurement) + Δγ_ref

    Formula
    -------
        Δγ_ref = γ(φ, h_ref) - γ(φ, h_meas)

    Interpretation
    --------------
    If the reference is lower than the measurement point, this correction is
    usually positive because normal gravity is stronger at lower height.

    References
    ----------
    Hackney & Featherstone (2003), Section 3.2.1:
    the free-air correction is fundamentally the vertical gradient of normal
    gravity used to move the theoretical normal gravity between levels.
    """
    return normal_gravity_change_between_heights(
        lat_rad=lat_rad,
        height_from_m=measurement_height_m,
        height_to_m=reference_height_m,
    )


def free_air_linear_approx_correction_to_reference_height(
    measurement_height_m: float,
    reference_height_m: float,
) -> float:
    r"""
    First-order free-air correction using the textbook 0.3086 mGal/m gradient.

    Parameters
    ----------
    measurement_height_m : float
        Measurement height [m].
    reference_height_m : float
        Reference height [m].

    Returns
    -------
    float
        Approximate free-air correction [m/s^2] that should be ADDED to the
        gravity value at the measurement level to refer it to the reference level.

    Formula
    -------
    The classical first-order approximation is:

        Δg_FA ≈ 0.3086 * (h_meas - h_ref)   [mGal]

    converted internally to m/s^2.

    Reference
    ---------
    The 0.3086 mGal/m linear free-air gradient is the standard textbook
    first-order approximation referenced throughout applied gravimetry.
    The present repository still prefers the exact WGS 84 height-dependent
    normal gravity from `earth.py` for actual simulation work.
    """
    delta_h = float(measurement_height_m - reference_height_m)
    return float(mgal_to_mps2(0.3086 * delta_h))


def eotvos_correction_exact_ned(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> float:
    r"""
    Exact Down-axis Eötvös / Coriolis-transport correction from the local-level
    NED kinematic equation.

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
    float
        Down component of:
            (2 ω_ie^n + ω_en^n) × v^n
        in m/s^2.

    Formula
    -------
    From the standard local-level velocity equation:

        vdot^n = f^n - (2 ω_ie^n + ω_en^n) × v^n + g^n

    define:

        c^n = (2 ω_ie^n + ω_en^n) × v^n

    Then the Down component c_D is exactly the NED-sign-convention analogue of
    the classical vertical Eötvös correction term used in moving-base gravimetry.

    Why this is preferred
    ---------------------
    This exact NED expression is the physically clean form to use inside the
    simulator because it is consistent with the same Earth / transport-rate
    model used by the IMU mechanization.

    Reference
    ---------
    The classical moving-base reduction formula with an Eötvös term is discussed
    in Shi et al. (2021). The exact NED form used here follows directly from the
    local-level inertial navigation equation already used in `imu.py`.
    """
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    omega_ie_n = earth_rate_ned(lat_rad)
    omega_en_n = transport_rate_ned(lat_rad, height_m, v_n)
    correction_n = np.cross(2.0 * omega_ie_n + omega_en_n, v_n)
    return float(correction_n[2])


def eotvos_correction_harlan_approx(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> float:
    r"""
    Classical Harlan-style Eötvös correction approximation.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : array-like, shape (3,)
        Platform velocity resolved in NED [m/s].

    Returns
    -------
    float
        Approximate Eötvös correction [m/s^2].

    Formula
    -------
    Shi et al. (2021) quote the following expression:

        δa_E = (1 + h/a) (2 ω v_E cosφ + v^2/a)
               - (f/a) (v^2 - cos^2φ (3 v^2 - 2 v_E^2))

    where:
    - v   is the speed magnitude
    - v_E is eastward velocity
    - ω   is Earth rotation rate
    - f   is ellipsoid flattening
    - a   is ellipsoid semi-major axis

    Implementation notes
    --------------------
    - This approximation is included mainly for comparison with classical marine/
      airborne gravimetry formulas.
    - The exact local-level NED correction from `eotvos_correction_exact_ned(...)`
      should be preferred for actual simulation work.
    - The literature often applies this formula under assumptions of near-level
      motion and conventional survey geometries.

    Reference
    ---------
    Shi et al. (2021), Eq. (3):
    https://link.springer.com/article/10.1186/s40623-021-01498-x
    """
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    speed = float(np.linalg.norm(v_n))
    v_e = float(v_n[1])
    a = float(WGS84.a)
    f = float(WGS84.f)
    omega = float(WGS84.omega)
    cphi = float(np.cos(lat_rad))
    h = float(height_m)

    correction = (
        (1.0 + h / a) * (2.0 * omega * v_e * cphi + (speed ** 2) / a)
        - (f / a) * ((speed ** 2) - (cphi ** 2) * (3.0 * (speed ** 2) - 2.0 * (v_e ** 2)))
    )
    return float(correction)


def recover_total_gravity_down_from_specific_force_ned(
    specific_force_ned_mps2: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> float:
    r"""
    Recover the Down component of total gravity from specific force and local-level
    kinematics.

    Parameters
    ----------
    specific_force_ned_mps2 : array-like, shape (3,)
        Specific force resolved in NED [m/s^2].
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity with respect to Earth [m/s^2].
    v_ned_mps : array-like, shape (3,)
        NED velocity with respect to Earth [m/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].

    Returns
    -------
    float
        Down component of total gravity g_D [m/s^2].

    Formula
    -------
    From the standard NED velocity equation:

        vdot^n = f^n - (2 ω_ie^n + ω_en^n) × v^n + g^n

    Rearranging:

        g^n = vdot^n - f^n + (2 ω_ie^n + ω_en^n) × v^n

    so the Down component is:

        g_D = vdot_D - f_D + c_D

    where:
        c_D = ((2 ω_ie^n + ω_en^n) × v^n)_D

    Consistency check
    -----------------
    If the platform is stationary and level:
    - v = 0
    - vdot = 0
    - f_D = -g
    then:
        g_D = 0 - (-g) + 0 = g

    References
    ----------
    - Standard local-level INS equation used in the repository IMU model
    - Classical moving-base gravity reduction formula in Shi et al. (2021),
      which is the same physical relation written in Up-positive form
    """
    f_n = _vec3(specific_force_ned_mps2, name="specific_force_ned_mps2")
    v_dot_n = _vec3(v_dot_ned_mps2, name="v_dot_ned_mps2")
    c_d = eotvos_correction_exact_ned(lat_rad, height_m, v_ned_mps)
    return float(v_dot_n[2] - f_n[2] + c_d)


def recover_gravity_disturbance_from_specific_force_ned(
    specific_force_ned_mps2: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> float:
    r"""
    Recover the measurement-level gravity disturbance from NED specific force and
    local-level kinematics.

    Parameters
    ----------
    specific_force_ned_mps2 : array-like, shape (3,)
        Specific force resolved in NED [m/s^2].
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity with respect to Earth [m/s^2].
    v_ned_mps : array-like, shape (3,)
        Velocity with respect to Earth resolved in NED [m/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].

    Returns
    -------
    float
        Gravity disturbance δg(P) [m/s^2] at the observation point.

    Formula
    -------
    First recover the Down component of total gravity:

        g_D = vdot_D - f_D + c_D

    Then subtract normal gravity at the same point:

        δg = g_D - γ(φ, h)

    Relation to the classical up-positive formula
    ---------------------------------------------
    Shi et al. (2021) write:

        δg = f_U - vdot_U + δa_E - γ

    In NED with Down positive:
    - f_U    = -f_D
    - vdot_U = -vdot_D
    - δa_E   = c_D

    hence:

        δg = vdot_D - f_D + c_D - γ

    exactly as used here.

    References
    ----------
    - Shi et al. (2021), Eq. (1)
    - same-point gravity disturbance definition from Hackney & Featherstone (2003)
    """
    g_d = recover_total_gravity_down_from_specific_force_ned(
        specific_force_ned_mps2=specific_force_ned_mps2,
        v_dot_ned_mps2=v_dot_ned_mps2,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        height_m=height_m,
    )
    gamma_p = float(normal_gravity(lat_rad, height_m))
    return float(g_d - gamma_p)


def recover_gravity_disturbance_from_specific_force_body(
    specific_force_body_mps2: ArrayLike,
    C_n_b: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> float:
    r"""
    Recover the measurement-level gravity disturbance from body-frame specific
    force, given the body->NED attitude.

    Parameters
    ----------
    specific_force_body_mps2 : array-like, shape (3,)
        Specific force resolved in body coordinates [m/s^2].
    C_n_b : array-like, shape (3, 3)
        Passive DCM mapping body -> NED:
            v^n = C_n_b v^b
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity [m/s^2].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].

    Returns
    -------
    float
        Gravity disturbance δg(P) [m/s^2].

    Formula
    -------
    First rotate body specific force into NED:

        f^n = C_n_b f^b

    then apply:

        δg = vdot_D - f_D + c_D - γ(φ, h)

    Notes
    -----
    This is the most convenient interface when you already have ideal IMU truth
    in the body frame from `imu.py`.
    """
    f_b = _vec3(specific_force_body_mps2, name="specific_force_body_mps2")
    C_n_b = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    f_n = C_n_b @ f_b
    return recover_gravity_disturbance_from_specific_force_ned(
        specific_force_ned_mps2=f_n,
        v_dot_ned_mps2=v_dot_ned_mps2,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        height_m=height_m,
    )


@dataclass
class ScalarGravityTruth:
    """
    Basic truth quantities derived from a true gravity vector at the observation point.

    Attributes
    ----------
    total_gravity_magnitude_mps2 : float
        Scalar gravity magnitude |g(P)|.
    normal_gravity_mps2 : float
        Normal gravity γ(P) at the same point.
    scalar_disturbance_mps2 : float
        Scalar gravity disturbance:
            δg(P) = |g(P)| - γ(P)
    vertical_disturbance_approx_mps2 : float
        Approximate disturbance using the Down component:
            g_D(P) - γ(P)

    Notes
    -----
    This dataclass is convenient for debugging synthetic gravity fields before
    any moving-platform correction logic is introduced.
    """

    total_gravity_magnitude_mps2: float
    normal_gravity_mps2: float
    scalar_disturbance_mps2: float
    vertical_disturbance_approx_mps2: float

    @property
    def scalar_disturbance_mgal(self) -> float:
        """Scalar gravity disturbance in mGal."""
        return float(mps2_to_mgal(self.scalar_disturbance_mps2))

    @property
    def vertical_disturbance_approx_mgal(self) -> float:
        """Vertical disturbance approximation in mGal."""
        return float(mps2_to_mgal(self.vertical_disturbance_approx_mps2))


def build_scalar_gravity_truth_from_total_vector_ned(
    total_gravity_ned_mps2: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> ScalarGravityTruth:
    """
    Build `ScalarGravityTruth` from a true gravity vector resolved in NED.

    Parameters
    ----------
    total_gravity_ned_mps2 : array-like, shape (3,)
        True gravity vector in NED [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    """
    g_mag = gravity_magnitude_from_vector_ned(total_gravity_ned_mps2)
    gamma_p = float(normal_gravity(lat_rad, height_m))
    scalar_dist = gravity_disturbance_from_scalar_gravity(g_mag, lat_rad, height_m)
    vertical_dist = vertical_gravity_disturbance_approx_from_vector_ned(
        total_gravity_ned_mps2, lat_rad, height_m
    )
    return ScalarGravityTruth(
        total_gravity_magnitude_mps2=g_mag,
        normal_gravity_mps2=gamma_p,
        scalar_disturbance_mps2=scalar_dist,
        vertical_disturbance_approx_mps2=vertical_dist,
    )


@dataclass
class MovingBaseGravimetryTruth:
    """
    Truth quantities for moving-base gravimetry reduction.

    Attributes
    ----------
    specific_force_down_mps2 : float
        Down component of the specific force in NED.
    vertical_acceleration_down_mps2 : float
        Down component of NED velocity derivative.
    eotvos_exact_mps2 : float
        Exact NED Down-axis Eötvös / Coriolis-transport correction.
    eotvos_harlan_approx_mps2 : float
        Harlan-style approximation for comparison.
    recovered_total_gravity_down_mps2 : float
        Recovered Down component of total gravity from exact NED kinematics.
    normal_gravity_mps2 : float
        WGS 84 normal gravity at the same point.
    recovered_disturbance_mps2 : float
        Recovered gravity disturbance:
            δg = g_D - γ

    Notes
    -----
    This container is helpful for unit tests and for debugging the reduction
    chain before the full navigation filter exists.
    """

    specific_force_down_mps2: float
    vertical_acceleration_down_mps2: float
    eotvos_exact_mps2: float
    eotvos_harlan_approx_mps2: float
    recovered_total_gravity_down_mps2: float
    normal_gravity_mps2: float
    recovered_disturbance_mps2: float

    @property
    def recovered_disturbance_mgal(self) -> float:
        """Recovered gravity disturbance in mGal."""
        return float(mps2_to_mgal(self.recovered_disturbance_mps2))

    @property
    def eotvos_exact_mgal(self) -> float:
        """Exact Eötvös term in mGal."""
        return float(mps2_to_mgal(self.eotvos_exact_mps2))

    @property
    def eotvos_harlan_approx_mgal(self) -> float:
        """Approximate Harlan Eötvös term in mGal."""
        return float(mps2_to_mgal(self.eotvos_harlan_approx_mps2))


def build_moving_base_gravimetry_truth_from_ned_kinematics(
    specific_force_ned_mps2: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
) -> MovingBaseGravimetryTruth:
    """
    Build `MovingBaseGravimetryTruth` from local-level kinematics and specific force.

    Parameters
    ----------
    specific_force_ned_mps2 : array-like, shape (3,)
        Specific force in NED [m/s^2].
    v_dot_ned_mps2 : array-like, shape (3,)
        NED velocity derivative [m/s^2].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    """
    f_n = _vec3(specific_force_ned_mps2, name="specific_force_ned_mps2")
    v_dot_n = _vec3(v_dot_ned_mps2, name="v_dot_ned_mps2")
    c_d = eotvos_correction_exact_ned(lat_rad, height_m, v_ned_mps)
    c_h = eotvos_correction_harlan_approx(lat_rad, height_m, v_ned_mps)
    g_d = recover_total_gravity_down_from_specific_force_ned(
        specific_force_ned_mps2=f_n,
        v_dot_ned_mps2=v_dot_n,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        height_m=height_m,
    )
    gamma_p = float(normal_gravity(lat_rad, height_m))
    delta_g = g_d - gamma_p

    return MovingBaseGravimetryTruth(
        specific_force_down_mps2=float(f_n[2]),
        vertical_acceleration_down_mps2=float(v_dot_n[2]),
        eotvos_exact_mps2=c_d,
        eotvos_harlan_approx_mps2=c_h,
        recovered_total_gravity_down_mps2=g_d,
        normal_gravity_mps2=gamma_p,
        recovered_disturbance_mps2=delta_g,
    )


@dataclass
class GravimeterSpec:
    """
    Scalar gravimeter specification.

    Parameters
    ----------
    noise_density_mps2_per_sqrt_hz : float, default=0.0
        White-noise density in m/s^2 / sqrt(Hz).
    bias_random_walk_mps2_per_sqrt_s : float, default=0.0
        In-run bias random-walk coefficient in m/s^2 / sqrt(s).
    turn_on_bias_std_mps2 : float, default=0.0
        1-sigma turn-on bias uncertainty in m/s^2.
    fixed_bias_mps2 : float, default=0.0
        Deterministic fixed bias in m/s^2.
    scale_factor_error_ppm : float, default=0.0
        Scalar scale-factor error in parts per million.
    max_abs_mps2 : float, default=np.inf
        Saturation magnitude for the output.
    bandwidth_hz : float or None, default=None
        Optional first-order low-pass bandwidth. If None, no bandwidth limit is applied.
    specific_force_coupling_b : scalar or shape (3,), default=0.0
        Additive motion-coupling coefficient applied to body specific force:

            residual_f = k_f^T (f_b - f_ref)

        Units are dimensionless because the output units are m/s^2 and the input
        units are m/s^2.
    specific_force_reference_b_mps2 : scalar or shape (3,), default=0.0
        Reference body specific force about which the linear coupling is defined.
        This avoids baking a stationary 1g offset into the coupling model if you
        do not want it.
    angular_rate_coupling_b_mps2_per_radps : scalar or shape (3,), default=0.0
        Additive motion-coupling coefficient applied to body angular rate:

            residual_ω = k_ω^T ω_b

        Units: m/s^2 per rad/s
    name : str, default="gravimeter"
        Human-readable identifier.
    supports_absolute_mode : bool, default=False
        Descriptive flag only. Set True for a configuration intended to represent
        an absolute / quantum gravimeter; the actual physics still comes from the
        chosen noise / bias parameters.

    Sensor model
    ------------
    The simulated scalar output is:

        y_k = s * LPF( x_k + r_k ) + b_k + n_k

    where:
    - x_k : ideal scalar input (absolute gravity or disturbance)
    - r_k : motion-coupling residual
    - LPF : optional first-order low-pass response
    - s   : scale factor = 1 + ppm * 1e-6
    - b_k : current bias state (fixed + turn-on + random walk)
    - n_k : additive white noise

    Notes
    -----
    This model is deliberately honest about what the current repository knows:
    we are not yet claiming a detailed photonic transduction model. Instead we
    expose the dominant navigation-facing error knobs explicitly so that later
    benchtop data can replace them with measured fits.
    """

    noise_density_mps2_per_sqrt_hz: float = 0.0
    bias_random_walk_mps2_per_sqrt_s: float = 0.0
    turn_on_bias_std_mps2: float = 0.0
    fixed_bias_mps2: float = 0.0
    scale_factor_error_ppm: float = 0.0
    max_abs_mps2: float = np.inf
    bandwidth_hz: Optional[float] = None

    specific_force_coupling_b: ArrayLike | float = 0.0
    specific_force_reference_b_mps2: ArrayLike | float = 0.0
    angular_rate_coupling_b_mps2_per_radps: ArrayLike | float = 0.0

    name: str = "gravimeter"
    supports_absolute_mode: bool = False

    def __post_init__(self) -> None:
        self.noise_density_mps2_per_sqrt_hz = float(
            self.noise_density_mps2_per_sqrt_hz
        )
        self.bias_random_walk_mps2_per_sqrt_s = float(
            self.bias_random_walk_mps2_per_sqrt_s
        )
        self.turn_on_bias_std_mps2 = float(self.turn_on_bias_std_mps2)
        self.fixed_bias_mps2 = float(self.fixed_bias_mps2)
        self.scale_factor_error_ppm = float(self.scale_factor_error_ppm)
        self.max_abs_mps2 = float(self.max_abs_mps2)

        if self.noise_density_mps2_per_sqrt_hz < 0.0:
            raise ValueError(
                "noise_density_mps2_per_sqrt_hz must be nonnegative."
            )
        if self.bias_random_walk_mps2_per_sqrt_s < 0.0:
            raise ValueError(
                "bias_random_walk_mps2_per_sqrt_s must be nonnegative."
            )
        if self.turn_on_bias_std_mps2 < 0.0:
            raise ValueError("turn_on_bias_std_mps2 must be nonnegative.")
        if self.max_abs_mps2 <= 0.0 and not np.isinf(self.max_abs_mps2):
            raise ValueError("max_abs_mps2 must be positive or inf.")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive when provided.")

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

    @classmethod
    def perfect_relative(cls, name: str = "perfect_relative_gravimeter") -> "GravimeterSpec":
        """
        Return a perfect gravimeter specification intended for disturbance /
        relative-gravity simulation.
        """
        return cls(name=name, supports_absolute_mode=False)

    @classmethod
    def perfect_absolute(cls, name: str = "perfect_absolute_gravimeter") -> "GravimeterSpec":
        """
        Return a perfect gravimeter specification intended for absolute-gravity
        simulation.
        """
        return cls(name=name, supports_absolute_mode=True)

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
        """Per-sample scalar white-noise standard deviation."""
        return scalar_white_noise_std_from_density(
            self.noise_density_mps2_per_sqrt_hz,
            dt_s,
        )

    def bias_step_std(self, dt_s: float) -> float:
        """Per-step scalar bias-random-walk standard deviation."""
        return scalar_random_walk_step_std(
            self.bias_random_walk_mps2_per_sqrt_s,
            dt_s,
        )


@dataclass
class GravimeterBiasState:
    """
    State of the gravimeter bias.

    Attributes
    ----------
    bias_mps2 : float
        Current scalar bias state in m/s^2.

    Interpretation
    --------------
    This stores the total active bias used by the sensor during the run:
    - fixed bias
    - turn-on bias realization
    - accumulated random-walk drift
    """

    bias_mps2: float

    @property
    def bias_mgal(self) -> float:
        """Bias in mGal."""
        return float(mps2_to_mgal(self.bias_mps2))


@dataclass
class GravimeterMeasurement:
    """
    One scalar gravimeter sample.

    Attributes
    ----------
    kind : str
        Human-readable label, e.g. "disturbance" or "absolute_gravity".
    time_s : float or None
        Optional timestamp [s].
    value_mps2 : float
        Final scalar sensor output [m/s^2].
    ideal_value_mps2 : float
        Ideal scalar input before motion residuals, filtering, scale, bias, and noise.
    motion_residual_mps2 : float
        Additive motion-coupling residual applied at the physical-input level.
    filtered_input_mps2 : float
        Post-bandwidth filtered scalar input before scale, bias, and white noise.
    bias_used_mps2 : float
        Bias state applied to this sample.
    white_noise_mps2 : float
        White-noise realization applied to this sample.
    saturated : bool
        True if saturation clipping occurred.

    Notes
    -----
    This struct intentionally exposes the latent terms so that you can later
    inspect:
    - how much the motion residual hurt
    - how much of the output is bias vs white noise
    - how much the bandwidth smoothed the input
    """

    kind: str
    time_s: Optional[float]
    value_mps2: float
    ideal_value_mps2: float
    motion_residual_mps2: float
    filtered_input_mps2: float
    bias_used_mps2: float
    white_noise_mps2: float
    saturated: bool

    @property
    def value_mgal(self) -> float:
        """Final scalar output in mGal."""
        return float(mps2_to_mgal(self.value_mps2))

    @property
    def ideal_value_mgal(self) -> float:
        """Ideal scalar input in mGal."""
        return float(mps2_to_mgal(self.ideal_value_mps2))

    @property
    def motion_residual_mgal(self) -> float:
        """Motion residual in mGal."""
        return float(mps2_to_mgal(self.motion_residual_mps2))

    @property
    def bias_used_mgal(self) -> float:
        """Bias used in this sample in mGal."""
        return float(mps2_to_mgal(self.bias_used_mps2))

    @property
    def white_noise_mgal(self) -> float:
        """White-noise realization in mGal."""
        return float(mps2_to_mgal(self.white_noise_mps2))


class ScalarGravimeterSensor:
    """
    Stateful scalar gravimeter simulator.

    This class can simulate either:
    - a processed gravity-disturbance channel for navigation map matching, or
    - an absolute-gravity channel

    depending on which measurement method you call.

    State owned by this object
    --------------------------
    - specification (`GravimeterSpec`)
    - RNG
    - current bias state
    - optional first-order low-pass state

    Design philosophy
    -----------------
    This class is intentionally honest and modular:
    - if you want to emulate a classical relative gravimeter, give it nonzero
      drift / random walk / calibration bias
    - if you want to emulate a quantum/absolute gravimeter, drive those terms
      low or zero
    - if you want to model motion fragility, use the explicit coupling vectors
      instead of burying "mysterious degradation" in the white-noise term
    """

    def __init__(
        self,
        spec: GravimeterSpec,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = np.random.default_rng() if rng is None else rng

        self.bias_state = GravimeterBiasState(
            bias_mps2=float(self.spec.fixed_bias_mps2)
        )
        self._filter_state_mps2: Optional[float] = None
        self.reset()

    def reset(
        self,
        *,
        turn_on_bias_randomized: bool = True,
        bias_override_mps2: Optional[float] = None,
        filter_state_override_mps2: Optional[float] = None,
    ) -> None:
        """
        Reset the gravimeter state.

        Parameters
        ----------
        turn_on_bias_randomized : bool, default=True
            If True, sample a turn-on bias realization.
        bias_override_mps2 : float, optional
            If provided, use this as the complete initial bias state.
        filter_state_override_mps2 : float, optional
            If provided, use this as the initial low-pass filter state.

        Notes
        -----
        The low-pass filter state defaults to "uninitialized" so that the first
        sample sets it directly to the first input, avoiding an artificial filter
        startup transient unless you explicitly want one.
        """
        if bias_override_mps2 is not None:
            bias = float(bias_override_mps2)
        else:
            bias = float(self.spec.fixed_bias_mps2)
            if turn_on_bias_randomized:
                bias += self.spec.turn_on_bias_std_mps2 * float(
                    self.rng.standard_normal()
                )

        self.bias_state = GravimeterBiasState(bias_mps2=bias)
        self._filter_state_mps2 = (
            None if filter_state_override_mps2 is None else float(filter_state_override_mps2)
        )

    def current_bias_state(self) -> GravimeterBiasState:
        """
        Return a copy of the current bias state.
        """
        return GravimeterBiasState(bias_mps2=float(self.bias_state.bias_mps2))

    def _step_bias_random_walk(self, dt_s: float) -> None:
        """
        Evolve the in-run bias state by one step.

        Formula
        -------
            b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
            w_k ~ N(0, 1)
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")
        sigma_step = self.spec.bias_step_std(dt_s)
        self.bias_state.bias_mps2 += sigma_step * float(self.rng.standard_normal())

    def _apply_bandwidth(self, x_mps2: float, dt_s: float) -> float:
        r"""
        Apply an optional first-order low-pass response to the scalar input.

        Model
        -----
        If no bandwidth is specified:
            y = x

        Otherwise:
            y_k = y_{k-1} + α (x_k - y_{k-1})

        with the exact first-order zero-order-hold coefficient:

            α = 1 - exp(-2π f_c dt)

        where f_c is the 3 dB bandwidth.

        Notes
        -----
        This is a simple engineering bandwidth model. It is not intended to be a
        full internal transfer function for a specific commercial gravimeter.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        if self.spec.bandwidth_hz is None:
            return float(x_mps2)

        if self._filter_state_mps2 is None:
            self._filter_state_mps2 = float(x_mps2)
            return float(self._filter_state_mps2)

        alpha = 1.0 - np.exp(-2.0 * np.pi * self.spec.bandwidth_hz * dt_s)
        self._filter_state_mps2 = float(
            self._filter_state_mps2 + alpha * (x_mps2 - self._filter_state_mps2)
        )
        return float(self._filter_state_mps2)

    def _motion_residual(
        self,
        body_specific_force_b_mps2: Optional[ArrayLike],
        body_angular_rate_b_radps: Optional[ArrayLike],
    ) -> float:
        r"""
        Compute the optional additive motion-coupling residual.

        Formula
        -------
        The current model is linear in body specific force and body angular rate:

            r = k_f^T (f_b - f_ref) + k_ω^T ω_b

        where:
        - k_f has units [m/s^2] / [m/s^2] = dimensionless
        - k_ω has units [m/s^2] / [rad/s]

        Interpretation
        --------------
        This is a deliberately transparent placeholder for real dynamic-coupling
        effects such as:
        - residual tilt leakage
        - imperfect common-mode rejection
        - motion-dependent readout coupling
        - photonic/packaging sensitivity to body rotation or vibration

        If you do not want motion residuals, leave the coupling vectors at zero.
        """
        residual = 0.0

        if body_specific_force_b_mps2 is not None:
            f_b = _vec3(body_specific_force_b_mps2, name="body_specific_force_b_mps2")
            residual += float(
                np.dot(
                    self.spec.specific_force_coupling_b,
                    f_b - self.spec.specific_force_reference_b_mps2,
                )
            )

        if body_angular_rate_b_radps is not None:
            w_b = _vec3(body_angular_rate_b_radps, name="body_angular_rate_b_radps")
            residual += float(
                np.dot(
                    self.spec.angular_rate_coupling_b_mps2_per_radps,
                    w_b,
                )
            )

        return float(residual)

    def _measure_scalar(
        self,
        ideal_value_mps2: float,
        dt_s: float,
        *,
        kind: str,
        body_specific_force_b_mps2: Optional[ArrayLike] = None,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Core scalar measurement routine.

        Parameters
        ----------
        ideal_value_mps2 : float
            Ideal scalar input (either disturbance or absolute gravity) [m/s^2].
        dt_s : float
            Sample interval [s].
        kind : str
            Measurement label.
        body_specific_force_b_mps2 : array-like, optional
            Body specific force for motion-residual modeling.
        body_angular_rate_b_radps : array-like, optional
            Body angular rate for motion-residual modeling.
        time_s : float, optional
            Optional timestamp [s].

        Returns
        -------
        GravimeterMeasurement
            Simulated measurement sample.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}.")

        ideal = float(ideal_value_mps2)
        motion_residual = self._motion_residual(
            body_specific_force_b_mps2=body_specific_force_b_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
        )

        physical_input = ideal + motion_residual
        filtered_input = self._apply_bandwidth(physical_input, dt_s)

        white_std = self.spec.white_noise_std(dt_s)
        white_noise = white_std * float(self.rng.standard_normal())
        bias_used = float(self.bias_state.bias_mps2)

        value = self.spec.scale_factor * filtered_input + bias_used + white_noise
        clipped_value = float(np.clip(value, -self.spec.max_abs_mps2, self.spec.max_abs_mps2))
        saturated = not np.isclose(clipped_value, value)

        measurement = GravimeterMeasurement(
            kind=str(kind),
            time_s=None if time_s is None else float(time_s),
            value_mps2=clipped_value,
            ideal_value_mps2=ideal,
            motion_residual_mps2=motion_residual,
            filtered_input_mps2=filtered_input,
            bias_used_mps2=bias_used,
            white_noise_mps2=white_noise,
            saturated=saturated,
        )

        self._step_bias_random_walk(dt_s)
        return measurement

    def measure_disturbance(
        self,
        ideal_disturbance_mps2: float,
        dt_s: float,
        *,
        body_specific_force_b_mps2: Optional[ArrayLike] = None,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Measure an ideal scalar gravity disturbance.

        Parameters
        ----------
        ideal_disturbance_mps2 : float
            Ideal scalar gravity disturbance at the observation point [m/s^2].
        dt_s : float
            Sample interval [s].
        body_specific_force_b_mps2 : array-like, optional
            Optional body specific force for motion-residual modeling.
        body_angular_rate_b_radps : array-like, optional
            Optional body angular rate for motion-residual modeling.
        time_s : float, optional
            Optional timestamp [s].
        """
        return self._measure_scalar(
            ideal_value_mps2=float(ideal_disturbance_mps2),
            dt_s=dt_s,
            kind="disturbance",
            body_specific_force_b_mps2=body_specific_force_b_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
            time_s=time_s,
        )

    def measure_absolute_gravity(
        self,
        ideal_absolute_gravity_mps2: float,
        dt_s: float,
        *,
        body_specific_force_b_mps2: Optional[ArrayLike] = None,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Measure an ideal scalar absolute gravity input.

        Parameters
        ----------
        ideal_absolute_gravity_mps2 : float
            Ideal scalar gravity magnitude [m/s^2].
        dt_s : float
            Sample interval [s].
        body_specific_force_b_mps2 : array-like, optional
            Optional body specific force for motion-residual modeling.
        body_angular_rate_b_radps : array-like, optional
            Optional body angular rate for motion-residual modeling.
        time_s : float, optional
            Optional timestamp [s].
        """
        return self._measure_scalar(
            ideal_value_mps2=float(ideal_absolute_gravity_mps2),
            dt_s=dt_s,
            kind="absolute_gravity",
            body_specific_force_b_mps2=body_specific_force_b_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
            time_s=time_s,
        )

    def measure_disturbance_from_total_gravity_vector_ned(
        self,
        total_gravity_ned_mps2: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        *,
        body_specific_force_b_mps2: Optional[ArrayLike] = None,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Convenience wrapper:
            true gravity vector -> scalar disturbance -> simulated measurement.
        """
        ideal_disturbance = gravity_disturbance_from_total_gravity_vector_ned(
            total_gravity_ned_mps2=total_gravity_ned_mps2,
            lat_rad=lat_rad,
            height_m=height_m,
        )
        return self.measure_disturbance(
            ideal_disturbance_mps2=ideal_disturbance,
            dt_s=dt_s,
            body_specific_force_b_mps2=body_specific_force_b_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
            time_s=time_s,
        )

    def measure_absolute_gravity_from_total_gravity_vector_ned(
        self,
        total_gravity_ned_mps2: ArrayLike,
        dt_s: float,
        *,
        body_specific_force_b_mps2: Optional[ArrayLike] = None,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Convenience wrapper:
            true gravity vector -> scalar gravity magnitude -> simulated measurement.
        """
        ideal_absolute_gravity = gravity_magnitude_from_vector_ned(total_gravity_ned_mps2)
        return self.measure_absolute_gravity(
            ideal_absolute_gravity_mps2=ideal_absolute_gravity,
            dt_s=dt_s,
            body_specific_force_b_mps2=body_specific_force_b_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
            time_s=time_s,
        )

    def measure_disturbance_from_specific_force_ned(
        self,
        specific_force_ned_mps2: ArrayLike,
        v_dot_ned_mps2: ArrayLike,
        v_ned_mps: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        *,
        body_specific_force_b_mps2: Optional[ArrayLike] = None,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Convenience wrapper:
            local-level moving-base reduction -> scalar disturbance -> simulated measurement.

        This is the most useful high-level entry point when you already have:
        - truth specific force
        - truth velocity derivative
        - truth velocity
        - platform latitude / height

        and want the gravimeter output that will later be map-matched.
        """
        ideal_disturbance = recover_gravity_disturbance_from_specific_force_ned(
            specific_force_ned_mps2=specific_force_ned_mps2,
            v_dot_ned_mps2=v_dot_ned_mps2,
            v_ned_mps=v_ned_mps,
            lat_rad=lat_rad,
            height_m=height_m,
        )
        return self.measure_disturbance(
            ideal_disturbance_mps2=ideal_disturbance,
            dt_s=dt_s,
            body_specific_force_b_mps2=body_specific_force_b_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
            time_s=time_s,
        )

    def measure_disturbance_from_specific_force_body(
        self,
        specific_force_body_mps2: ArrayLike,
        C_n_b: ArrayLike,
        v_dot_ned_mps2: ArrayLike,
        v_ned_mps: ArrayLike,
        lat_rad: float,
        height_m: float,
        dt_s: float,
        *,
        body_angular_rate_b_radps: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> GravimeterMeasurement:
        """
        Convenience wrapper:
            body specific force + attitude + local-level kinematics ->
            scalar disturbance -> simulated measurement.

        Parameters
        ----------
        specific_force_body_mps2 : array-like, shape (3,)
            Body specific force [m/s^2].
        C_n_b : array-like, shape (3, 3)
            Passive body->NED DCM.
        v_dot_ned_mps2 : array-like, shape (3,)
            NED velocity derivative [m/s^2].
        v_ned_mps : array-like, shape (3,)
            NED velocity [m/s].
        lat_rad : float
            Geodetic latitude [rad].
        height_m : float
            Ellipsoidal height [m].
        dt_s : float
            Sample interval [s].
        body_angular_rate_b_radps : array-like, optional
            Body angular rate [rad/s] for motion residual modeling.
        time_s : float, optional
            Optional timestamp [s].
        """
        ideal_disturbance = recover_gravity_disturbance_from_specific_force_body(
            specific_force_body_mps2=specific_force_body_mps2,
            C_n_b=C_n_b,
            v_dot_ned_mps2=v_dot_ned_mps2,
            v_ned_mps=v_ned_mps,
            lat_rad=lat_rad,
            height_m=height_m,
        )
        return self.measure_disturbance(
            ideal_disturbance_mps2=ideal_disturbance,
            dt_s=dt_s,
            body_specific_force_b_mps2=specific_force_body_mps2,
            body_angular_rate_b_radps=body_angular_rate_b_radps,
            time_s=time_s,
        )


__all__ = [
    "FloatArray",
    "GravimeterBiasState",
    "GravimeterMeasurement",
    "GravimeterSpec",
    "MovingBaseGravimetryTruth",
    "ScalarGravityTruth",
    "ScalarGravimeterSensor",
    "build_moving_base_gravimetry_truth_from_ned_kinematics",
    "build_scalar_gravity_truth_from_total_vector_ned",
    "eotvos_correction_exact_ned",
    "eotvos_correction_harlan_approx",
    "free_air_correction_to_reference_height",
    "free_air_linear_approx_correction_to_reference_height",
    "gravity_disturbance_from_scalar_gravity",
    "gravity_disturbance_from_total_gravity_vector_ned",
    "gravity_magnitude_from_vector_ned",
    "normal_gravity_change_between_heights",
    "recover_gravity_disturbance_from_specific_force_body",
    "recover_gravity_disturbance_from_specific_force_ned",
    "recover_total_gravity_down_from_specific_force_ned",
    "scalar_random_walk_step_std",
    "scalar_white_noise_std_from_density",
    "vertical_gravity_disturbance_approx_from_vector_ned",
]