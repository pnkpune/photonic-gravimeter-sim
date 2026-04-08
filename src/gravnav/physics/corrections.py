"""
corrections.py

Gravity-reduction and correction helpers for the gravity-aided navigation
simulator.

Why this file exists
--------------------
The repository already contains:
- WGS 84 normal gravity and geodetic geometry in `earth.py`
- local-level frame and Earth-rate math in `frames.py`
- scalar gravimeter physics in `sensors/gravimeter.py`
- a particle filter that expects map-predicted *gravity disturbance* values

The gap this file fills is not "another gravimeter model", but a reusable
correction/reduction layer that can be used by:
- future map-generation scripts
- preprocessing pipelines
- benchmark / station-style reduction workflows
- moving-base reduction utilities
- later simulation runners that need both navigation-first and classical
  gravimetry-style products

What this file provides
-----------------------
This module organizes corrections into three buckets:

1) Navigation-first reduction
   - measurement-level scalar gravity disturbance:
         delta_g(P) = g(P) - gamma(P)

   This is the repository's preferred quantity because it avoids unnecessary
   downward continuation through topography.

2) Classical anomaly-style reductions
   - atmospheric correction
   - free-air correction
   - Bouguer slab correction
   - terrain-corrected Bouguer anomaly from user-supplied terrain correction

3) Moving-base reduction helpers
   - exact NED Eotvos/Coriolis-transport correction
   - Harlan-style Eotvos approximation
   - gravity recovery from specific force and local-level kinematics
   - reduction of moving-base recovered gravity into disturbance / free-air /
     Bouguer-style products

Conventions
-----------
- Internal angles are radians.
- Heights are ellipsoidal and positive upward [m].
- NED uses Down positive.
- Gravity values, accelerations, and corrections are in m/s^2 internally.
- mGal is used only for reporting/debugging convenience.
- The default preferred product is the *measurement-level gravity disturbance*,
  not a geoid-referenced anomaly.

Important modeling choice
-------------------------
This file intentionally distinguishes between:

A) same-point disturbance
       delta_g(P) = g(P) - gamma(P)

and

B) classical free-air / Bouguer anomaly products
   that refer observations to a reference level through approximate correction
   models.

That distinction matters because Hackney & Featherstone show that gravity
disturbance is often the cleaner, more stable quantity to work with when the
observation level is known, while classical anomaly reduction requires
up/downward continuation approximations.

Primary references used here
----------------------------
1) Hackney, R. I., and Featherstone, W. E. (2003),
   "Geodetic versus geophysical perspectives of the 'gravity anomaly'"
   Geophysical Journal International, 154(1), 35-43.
   URL:
   https://academic.oup.com/gji/article/154/1/35/604237

   Used for:
   - same-point scalar gravity disturbance
   - why measurement-level disturbance is preferable to unnecessary downward
     continuation
   - free-air correction as the vertical gradient of normal gravity

2) National Imagery and Mapping Agency / USGS gravity computations notes
   URL:
   https://pubs.usgs.gov/of/2006/1204/Gravity/computations.pdf

   Used for:
   - atmospheric correction approximation
   - Bouguer slab attraction formula:
         delta_g_B = 2 pi G rho h
   - standard anomaly-reduction bookkeeping

3) New Mexico Bureau of Geology gravity-method notes
   URL:
   https://geoinfo.nmt.edu/geoscience/projects/astronauts/gravity_method.html

   Used as a concise readable statement of the common textbook formulas:
       g_fa = g_obs - g_n + 0.3086 h
       g_b  = g_obs - g_n + 0.3086 h - 0.04193 rho h
   in mGal with height in metres and density in g/cm^3.

4) Shi et al. (2021),
   "Experimental study on improving the accuracy of marine gravimetry by
   combining moving-base gravimeters with GNSS antenna array"
   Earth, Planets and Space, 73, 174.
   URL:
   https://link.springer.com/article/10.1186/s40623-021-01498-x

   Used for:
   - the moving-base relation
         delta_g = f_U - vdot_U + delta_a_E - gamma
   - the Harlan-style Eotvos approximation
   - practical reminder that Eotvos / vertical acceleration / free-air
     corrections dominate moving-platform reduction

5) Repository alignment
   This file is intentionally aligned with:
   - `gravnav.sensors.gravimeter`
   - `gravnav.estimators.map_match_pf`
   - `gravnav.physics.earth`
   - `gravnav.physics.frames`

Design notes
------------
- This module is a correction/reduction layer, not a full terrain-modeling or
  full geophysical continuation package.
- Terrain correction is accepted as an explicit additive input because the
  repository does not yet contain a DEM/topography engine.
- Bouguer correction is implemented as the simple infinite-slab model, which is
  appropriate as a first-order reduction helper.
- For navigation work, prefer the disturbance products in this file. Use the
  classical anomaly products only when you explicitly want those conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .earth import WGS84, mgal_to_mps2, mps2_to_mgal, normal_gravity
from .frames import earth_rate_ned, project_to_so3, transport_rate_ned

FloatArray = NDArray[np.float64]

# CODATA 2018 exact value commonly used in SI computations.
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11  # [m^3 / (kg s^2)]

# Common first-order reduction defaults.
STANDARD_BOUGUER_DENSITY_KGPM3 = 2670.0
SEAWATER_DENSITY_KGPM3 = 1025.0


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _scalar(x: ArrayLike, *, name: str) -> float:
    """Validate and return a scalar float."""
    arr = _as_float_array(x)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}.")
    return float(arr)


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 3-vector."""
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _mat3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 3x3 matrix."""
    arr = _as_float_array(x)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


def _nonnegative_scalar(x: float, *, name: str) -> float:
    """Require a nonnegative scalar."""
    value = float(x)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")
    return value


def _maybe_scalar(value: FloatArray, *refs: ArrayLike):
    """
    Return a Python float when all references are scalar-like, else return array.
    """
    scalar_like = all(np.asarray(r).ndim == 0 for r in refs)
    if scalar_like and np.asarray(value).ndim == 0:
        return float(np.asarray(value))
    return np.asarray(value, dtype=np.float64)


# -----------------------------------------------------------------------------
# Basic correction terms
# -----------------------------------------------------------------------------


def atmospheric_correction(height_m: ArrayLike):
    r"""
    Atmospheric gravity correction.

    Parameters
    ----------
    height_m : array-like or scalar
        Elevation/height proxy [m].

    Returns
    -------
    float or np.ndarray
        Atmospheric correction to be ADDED to observed gravity [m/s^2].

    Formula
    -------
    The NIMA/USGS notes give the approximation:

        delta_g_A = 0.87 * exp(-0.116 * (h / 1000)^1.047)   [mGal],  h >= 0
        delta_g_A = 0.87                                     [mGal],  h < 0

    Notes
    -----
    The original reference describes `h` as elevation relative to sea level.
    In this repository, callers often have ellipsoidal height rather than
    orthometric height. This function therefore treats the provided height as a
    pragmatic elevation proxy, which is acceptable for first-order simulation and
    sensitivity studies but should not be mistaken for a rigorous geoid-based
    atmospheric reduction workflow.
    """
    h = _as_float_array(height_m)
    corr_mgal = np.where(
        h >= 0.0,
        0.87 * np.exp(-0.116 * (np.maximum(h, 0.0) / 1000.0) ** 1.047),
        0.87,
    )
    corr = np.asarray(mgal_to_mps2(corr_mgal), dtype=np.float64)
    return _maybe_scalar(corr, height_m)


def bouguer_slab_attraction(
    thickness_m: ArrayLike,
    density_kgpm3: ArrayLike | float = STANDARD_BOUGUER_DENSITY_KGPM3,
):
    r"""
    Infinite-slab Bouguer attraction.

    Parameters
    ----------
    thickness_m : array-like or scalar
        Slab thickness [m]. Positive thickness means excess mass above the
        reference level; negative thickness means mass deficiency below it.
    density_kgpm3 : array-like or scalar, default=2670
        Slab density [kg/m^3].

    Returns
    -------
    float or np.ndarray
        Bouguer slab attraction [m/s^2].

    Formula
    -------
        delta_g_B = 2 pi G rho h

    where:
    - G   is the gravitational constant
    - rho is slab density
    - h   is slab thickness

    Notes
    -----
    In classical Bouguer reduction, this is the amount typically SUBTRACTED from
    the free-air-referenced observation when the observation lies above the
    reference level.
    """
    h = _as_float_array(thickness_m)
    rho = _as_float_array(density_kgpm3)
    if np.any(rho < 0.0):
        raise ValueError("density_kgpm3 must be nonnegative.")
    h_b, rho_b = np.broadcast_arrays(h, rho)
    corr = 2.0 * np.pi * GRAVITATIONAL_CONSTANT_SI * rho_b * h_b
    return _maybe_scalar(np.asarray(corr, dtype=np.float64), thickness_m, density_kgpm3)


def bouguer_slab_vertical_gradient(
    density_kgpm3: ArrayLike | float = STANDARD_BOUGUER_DENSITY_KGPM3,
):
    r"""
    Vertical gradient implied by the infinite Bouguer slab model.

    Parameters
    ----------
    density_kgpm3 : array-like or scalar, default=2670
        Density [kg/m^3].

    Returns
    -------
    float or np.ndarray
        Vertical gradient [m/s^2 per m].

    Formula
    -------
        d(delta_g_B) / dh = 2 pi G rho
    """
    rho = _as_float_array(density_kgpm3)
    if np.any(rho < 0.0):
        raise ValueError("density_kgpm3 must be nonnegative.")
    grad = 2.0 * np.pi * GRAVITATIONAL_CONSTANT_SI * rho
    return _maybe_scalar(np.asarray(grad, dtype=np.float64), density_kgpm3)


def free_air_correction_to_reference_height_exact(
    lat_rad: ArrayLike,
    measurement_height_m: ArrayLike,
    reference_height_m: ArrayLike,
):
    r"""
    Exact normal-gravity transfer between two heights using the repository's WGS 84
    model.

    Parameters
    ----------
    lat_rad : array-like or scalar
        Geodetic latitude [rad].
    measurement_height_m : array-like or scalar
        Height of the observation point [m].
    reference_height_m : array-like or scalar
        Height of the reference level [m].

    Returns
    -------
    float or np.ndarray
        Correction to be ADDED to gravity at the measurement level to refer it to
        the reference level [m/s^2].

    Formula
    -------
        delta_g_FA,exact = gamma(phi, h_ref) - gamma(phi, h_meas)

    Notes
    -----
    This is not full gravity-field continuation. It is the exact transfer of the
    *normal-gravity* term between two heights using the same WGS 84 model as the
    rest of the repository.
    """
    lat = _as_float_array(lat_rad)
    h_meas = _as_float_array(measurement_height_m)
    h_ref = _as_float_array(reference_height_m)
    lat_b, h_meas_b, h_ref_b = np.broadcast_arrays(lat, h_meas, h_ref)
    corr = _as_float_array(normal_gravity(lat_b, h_ref_b)) - _as_float_array(
        normal_gravity(lat_b, h_meas_b)
    )
    return _maybe_scalar(np.asarray(corr, dtype=np.float64), lat_rad, measurement_height_m, reference_height_m)


def free_air_correction_to_reference_height_linear(
    measurement_height_m: ArrayLike,
    reference_height_m: ArrayLike,
):
    r"""
    First-order free-air correction using the common 0.3086 mGal/m approximation.

    Parameters
    ----------
    measurement_height_m : array-like or scalar
        Height of the observation point [m].
    reference_height_m : array-like or scalar
        Reference height [m].

    Returns
    -------
    float or np.ndarray
        Correction to be ADDED to gravity at the measurement level to refer it to
        the reference level [m/s^2].

    Formula
    -------
        delta_g_FA,lin ≈ 0.3086 * (h_meas - h_ref)   [mGal]
    """
    h_meas = _as_float_array(measurement_height_m)
    h_ref = _as_float_array(reference_height_m)
    delta_h = h_meas - h_ref
    corr = np.asarray(mgal_to_mps2(0.3086 * delta_h), dtype=np.float64)
    return _maybe_scalar(corr, measurement_height_m, reference_height_m)


def disturbance_from_observed_gravity(
    observed_gravity_mps2: float,
    lat_rad: float,
    height_m: float,
    *,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> float:
    r"""
    Convert an observed scalar gravity value into the measurement-level scalar
    gravity disturbance.

    Parameters
    ----------
    observed_gravity_mps2 : float
        Observed scalar gravity magnitude [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Measurement height [m].
    apply_atmospheric_correction : bool, default=False
        If True, add the simple atmospheric correction before differencing.
    supplemental_correction_mps2 : float, default=0.0
        Any additional additive correction to the observed gravity, e.g. a known
        instrument tie or drift term.

    Returns
    -------
    float
        Measurement-level scalar gravity disturbance [m/s^2].

    Formula
    -------
        delta_g(P) = g_corr(P) - gamma(P)

    where:

        g_corr(P) = g_obs(P) + delta_g_A + delta_g_suppl

    Notes
    -----
    This is the repository-preferred navigation-facing quantity.
    """
    g_obs = float(observed_gravity_mps2)
    phi = float(lat_rad)
    h = float(height_m)
    delta_atm = float(atmospheric_correction(h)) if apply_atmospheric_correction else 0.0
    g_corr = g_obs + delta_atm + float(supplemental_correction_mps2)
    return float(g_corr - float(normal_gravity(phi, h)))


def free_air_anomaly_from_observed_gravity(
    observed_gravity_mps2: float,
    lat_rad: float,
    measurement_height_m: float,
    *,
    reference_height_m: float = 0.0,
    exact_free_air: bool = True,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> float:
    r"""
    Compute a free-air-referenced gravity anomaly style quantity.

    Parameters
    ----------
    observed_gravity_mps2 : float
        Observed scalar gravity [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    measurement_height_m : float
        Height of the observation point [m].
    reference_height_m : float, default=0.0
        Reference height [m].
    exact_free_air : bool, default=True
        If True, use the repository's exact WGS 84 normal-gravity height
        difference. If False, use the common linear 0.3086 mGal/m approximation.
    apply_atmospheric_correction : bool, default=False
        Whether to add the simple atmospheric correction.
    supplemental_correction_mps2 : float, default=0.0
        Additional additive correction term.

    Returns
    -------
    float
        Free-air style anomaly [m/s^2].

    Formula
    -------
        g_FA = g_corr + delta_g_FA - gamma(phi, h_ref)

    Notes
    -----
    When `exact_free_air=True`, this quantity becomes numerically identical to
    the measurement-level disturbance if the correction is defined purely through
    the normal-gravity height transfer. That is consistent with the repository's
    preference for disturbance-first processing.
    """
    g_obs = float(observed_gravity_mps2)
    phi = float(lat_rad)
    h_meas = float(measurement_height_m)
    h_ref = float(reference_height_m)

    delta_atm = float(atmospheric_correction(h_meas)) if apply_atmospheric_correction else 0.0
    g_corr = g_obs + delta_atm + float(supplemental_correction_mps2)

    if exact_free_air:
        delta_fa = float(
            free_air_correction_to_reference_height_exact(phi, h_meas, h_ref)
        )
    else:
        delta_fa = float(
            free_air_correction_to_reference_height_linear(h_meas, h_ref)
        )

    gamma_ref = float(normal_gravity(phi, h_ref))
    return float(g_corr + delta_fa - gamma_ref)


def bouguer_anomaly_from_observed_gravity(
    observed_gravity_mps2: float,
    lat_rad: float,
    measurement_height_m: float,
    *,
    reference_height_m: float = 0.0,
    density_kgpm3: float = STANDARD_BOUGUER_DENSITY_KGPM3,
    terrain_correction_mps2: float = 0.0,
    exact_free_air: bool = True,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> float:
    r"""
    Compute a simple terrain-corrected Bouguer anomaly style quantity.

    Parameters
    ----------
    observed_gravity_mps2 : float
        Observed scalar gravity [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    measurement_height_m : float
        Observation height [m].
    reference_height_m : float, default=0.0
        Reference level [m].
    density_kgpm3 : float, default=2670
        Bouguer slab density [kg/m^3].
    terrain_correction_mps2 : float, default=0.0
        Precomputed terrain correction to ADD [m/s^2].
    exact_free_air : bool, default=True
        Whether to use the exact normal-gravity free-air transfer.
    apply_atmospheric_correction : bool, default=False
        Whether to include the simple atmospheric correction.
    supplemental_correction_mps2 : float, default=0.0
        Additional additive correction term.

    Returns
    -------
    float
        Terrain-corrected Bouguer anomaly style value [m/s^2].

    Formula
    -------
    First compute the free-air style quantity:

        g_FA = g_corr + delta_g_FA - gamma_ref

    Then apply the Bouguer and terrain terms:

        g_B = g_FA - delta_g_B + TC

    where:

        delta_g_B = 2 pi G rho (h_meas - h_ref)

    Notes
    -----
    This is the simple infinite-slab classical reduction, not a full terrain/DEM
    workflow.
    """
    density = _nonnegative_scalar(density_kgpm3, name="density_kgpm3")
    tc = float(terrain_correction_mps2)

    g_fa = free_air_anomaly_from_observed_gravity(
        observed_gravity_mps2=observed_gravity_mps2,
        lat_rad=lat_rad,
        measurement_height_m=measurement_height_m,
        reference_height_m=reference_height_m,
        exact_free_air=exact_free_air,
        apply_atmospheric_correction=apply_atmospheric_correction,
        supplemental_correction_mps2=supplemental_correction_mps2,
    )
    delta_b = float(
        bouguer_slab_attraction(
            float(measurement_height_m) - float(reference_height_m),
            density,
        )
    )
    return float(g_fa - delta_b + tc)


def disturbance_height_transfer_linear(
    disturbance_mps2: float,
    height_from_m: float,
    height_to_m: float,
    *,
    vertical_gradient_mps2_per_m: float,
) -> float:
    r"""
    Transfer a disturbance between heights using a supplied linear vertical gradient.

    Parameters
    ----------
    disturbance_mps2 : float
        Disturbance at the original height [m/s^2].
    height_from_m : float
        Original height [m].
    height_to_m : float
        Target height [m].
    vertical_gradient_mps2_per_m : float
        Assumed disturbance vertical gradient [m/s^2 per m].

    Returns
    -------
    float
        Disturbance at the target height [m/s^2].

    Formula
    -------
        delta_g(h_to) ≈ delta_g(h_from) + (d delta_g / dh) * (h_to - h_from)
    """
    return float(
        float(disturbance_mps2)
        + float(vertical_gradient_mps2_per_m) * (float(height_to_m) - float(height_from_m))
    )


# -----------------------------------------------------------------------------
# Moving-base reduction helpers
# -----------------------------------------------------------------------------


def eotvos_correction_exact_ned(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> float:
    r"""
    Exact Down-axis Eotvos / Coriolis-transport correction from the local NED
    velocity equation.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Height [m].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s].

    Returns
    -------
    float
        Down component of:

            (2 omega_ie^n + omega_en^n) x v^n

        in m/s^2.

    Formula
    -------
    From the standard local-level velocity equation:

        vdot^n = f^n - (2 omega_ie^n + omega_en^n) x v^n + g^n

    define:

        c^n = (2 omega_ie^n + omega_en^n) x v^n

    Then `c_D` is the exact NED-sign-convention moving-base correction term.
    """
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    omega_ie_n = earth_rate_ned(float(lat_rad))
    omega_en_n = transport_rate_ned(float(lat_rad), float(height_m), v_n)
    correction_n = np.cross(2.0 * omega_ie_n + omega_en_n, v_n)
    return float(correction_n[2])


def eotvos_correction_harlan_approx(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> float:
    r"""
    Harlan-style Eotvos correction approximation.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Height [m].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s].

    Returns
    -------
    float
        Approximate Eotvos correction [m/s^2].

    Formula
    -------
    Following the expression quoted by Shi et al. (2021):

        delta_a_E = (1 + h/a) (2 omega v_E cos(phi) + v^2/a)
                    - (f/a) (v^2 - cos^2(phi) (3 v^2 - 2 v_E^2))

    Notes
    -----
    This is included mainly for comparison with the exact NED expression above.
    """
    v_n = _vec3(v_ned_mps, name="v_ned_mps")
    speed = float(np.linalg.norm(v_n))
    v_e = float(v_n[1])
    a = float(WGS84.a)
    f = float(WGS84.f)
    omega = float(WGS84.omega)
    cphi = float(np.cos(float(lat_rad)))
    h = float(height_m)

    correction = (
        (1.0 + h / a) * (2.0 * omega * v_e * cphi + (speed**2) / a)
        - (f / a)
        * ((speed**2) - (cphi**2) * (3.0 * (speed**2) - 2.0 * (v_e**2)))
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
        Specific force in NED [m/s^2].
    v_dot_ned_mps2 : array-like, shape (3,)
        Time derivative of NED velocity [m/s^2].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s].
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Height [m].

    Returns
    -------
    float
        Down component of total gravity [m/s^2].

    Formula
    -------
        g_D = vdot_D - f_D + c_D

    where `c_D` is the exact NED Eotvos/Coriolis-transport correction.
    """
    f_n = _vec3(specific_force_ned_mps2, name="specific_force_ned_mps2")
    v_dot_n = _vec3(v_dot_ned_mps2, name="v_dot_ned_mps2")
    c_d = eotvos_correction_exact_ned(lat_rad, height_m, v_ned_mps)
    return float(v_dot_n[2] - f_n[2] + c_d)


def recover_measurement_level_disturbance_from_specific_force_ned(
    specific_force_ned_mps2: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
    *,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> float:
    r"""
    Recover the measurement-level gravity disturbance from local-level kinematics.

    Parameters
    ----------
    specific_force_ned_mps2, v_dot_ned_mps2, v_ned_mps
        Moving-base kinematic inputs.
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Height [m].
    apply_atmospheric_correction : bool, default=False
        Whether to include the simple atmospheric correction after recovering total
        gravity.
    supplemental_correction_mps2 : float, default=0.0
        Additional additive correction term.

    Returns
    -------
    float
        Measurement-level gravity disturbance [m/s^2].

    Formula
    -------
        delta_g = g_D - gamma(phi, h)

    with optional additive corrections applied to the recovered gravity term.
    """
    g_d = recover_total_gravity_down_from_specific_force_ned(
        specific_force_ned_mps2=specific_force_ned_mps2,
        v_dot_ned_mps2=v_dot_ned_mps2,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        height_m=height_m,
    )
    delta_atm = float(atmospheric_correction(height_m)) if apply_atmospheric_correction else 0.0
    g_corr = g_d + delta_atm + float(supplemental_correction_mps2)
    return float(g_corr - float(normal_gravity(lat_rad, height_m)))


def recover_measurement_level_disturbance_from_specific_force_body(
    specific_force_body_mps2: ArrayLike,
    C_n_b: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    height_m: float,
    *,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> float:
    r"""
    Body-frame variant of
    :func:`recover_measurement_level_disturbance_from_specific_force_ned`.

    Parameters
    ----------
    specific_force_body_mps2 : array-like, shape (3,)
        Body specific force [m/s^2].
    C_n_b : array-like, shape (3, 3)
        Passive body->NED DCM.
    """
    f_b = _vec3(specific_force_body_mps2, name="specific_force_body_mps2")
    C_nb = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    f_n = C_nb @ f_b
    return recover_measurement_level_disturbance_from_specific_force_ned(
        specific_force_ned_mps2=f_n,
        v_dot_ned_mps2=v_dot_ned_mps2,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        height_m=height_m,
        apply_atmospheric_correction=apply_atmospheric_correction,
        supplemental_correction_mps2=supplemental_correction_mps2,
    )


# -----------------------------------------------------------------------------
# Reduction result containers
# -----------------------------------------------------------------------------


@dataclass
class StationaryGravityReduction:
    """
    Bundle of scalar gravity-reduction products for a stationary or already-reduced
    observed gravity sample.

    Attributes
    ----------
    observed_gravity_mps2 : float
        Input observed gravity [m/s^2].
    corrected_observed_gravity_mps2 : float
        Observed gravity after atmospheric/supplemental corrections [m/s^2].
    atmospheric_correction_mps2 : float
        Atmospheric correction applied [m/s^2].
    supplemental_correction_mps2 : float
        Additional additive correction term [m/s^2].
    normal_gravity_measurement_mps2 : float
        WGS 84 normal gravity at the measurement point [m/s^2].
    normal_gravity_reference_mps2 : float
        WGS 84 normal gravity at the reference height [m/s^2].
    measurement_level_disturbance_mps2 : float
        Same-point gravity disturbance [m/s^2].
    free_air_correction_mps2 : float
        Free-air correction to the reference height [m/s^2].
    free_air_anomaly_mps2 : float
        Free-air-referenced anomaly style quantity [m/s^2].
    bouguer_slab_correction_mps2 : float
        Infinite-slab Bouguer correction magnitude [m/s^2].
    terrain_correction_mps2 : float
        Terrain correction applied additively [m/s^2].
    bouguer_anomaly_mps2 : float
        Bouguer anomaly without terrain correction [m/s^2].
    terrain_corrected_bouguer_anomaly_mps2 : float
        Terrain-corrected Bouguer anomaly [m/s^2].
    """

    observed_gravity_mps2: float
    corrected_observed_gravity_mps2: float
    atmospheric_correction_mps2: float
    supplemental_correction_mps2: float

    normal_gravity_measurement_mps2: float
    normal_gravity_reference_mps2: float

    measurement_level_disturbance_mps2: float
    free_air_correction_mps2: float
    free_air_anomaly_mps2: float

    bouguer_slab_correction_mps2: float
    terrain_correction_mps2: float
    bouguer_anomaly_mps2: float
    terrain_corrected_bouguer_anomaly_mps2: float

    @property
    def measurement_level_disturbance_mgal(self) -> float:
        """Measurement-level disturbance in mGal."""
        return float(mps2_to_mgal(self.measurement_level_disturbance_mps2))

    @property
    def free_air_anomaly_mgal(self) -> float:
        """Free-air anomaly style value in mGal."""
        return float(mps2_to_mgal(self.free_air_anomaly_mps2))

    @property
    def bouguer_anomaly_mgal(self) -> float:
        """Bouguer anomaly in mGal."""
        return float(mps2_to_mgal(self.bouguer_anomaly_mps2))

    @property
    def terrain_corrected_bouguer_anomaly_mgal(self) -> float:
        """Terrain-corrected Bouguer anomaly in mGal."""
        return float(mps2_to_mgal(self.terrain_corrected_bouguer_anomaly_mps2))


@dataclass
class MovingBaseGravityReduction:
    """
    Bundle of moving-base gravity-reduction products derived from local-level
    kinematics.

    Attributes
    ----------
    specific_force_down_mps2 : float
        Down component of specific force [m/s^2].
    vertical_acceleration_down_mps2 : float
        Down component of NED velocity derivative [m/s^2].
    recovered_total_gravity_down_mps2 : float
        Recovered Down component of total gravity [m/s^2].
    corrected_recovered_gravity_down_mps2 : float
        Recovered gravity after atmospheric/supplemental corrections [m/s^2].
    eotvos_exact_mps2 : float
        Exact NED Eotvos correction [m/s^2].
    eotvos_harlan_approx_mps2 : float
        Harlan-style Eotvos approximation [m/s^2].
    atmospheric_correction_mps2 : float
        Atmospheric correction applied [m/s^2].
    supplemental_correction_mps2 : float
        Additional additive correction term [m/s^2].
    normal_gravity_measurement_mps2 : float
        WGS 84 normal gravity at the measurement height [m/s^2].
    normal_gravity_reference_mps2 : float
        WGS 84 normal gravity at the reference height [m/s^2].
    measurement_level_disturbance_mps2 : float
        Same-point disturbance [m/s^2].
    free_air_correction_mps2 : float
        Free-air correction to the reference height [m/s^2].
    free_air_anomaly_mps2 : float
        Free-air-referenced anomaly style quantity [m/s^2].
    bouguer_slab_correction_mps2 : float
        Bouguer slab correction [m/s^2].
    terrain_correction_mps2 : float
        Terrain correction applied additively [m/s^2].
    bouguer_anomaly_mps2 : float
        Bouguer anomaly [m/s^2].
    terrain_corrected_bouguer_anomaly_mps2 : float
        Terrain-corrected Bouguer anomaly [m/s^2].
    """

    specific_force_down_mps2: float
    vertical_acceleration_down_mps2: float

    recovered_total_gravity_down_mps2: float
    corrected_recovered_gravity_down_mps2: float

    eotvos_exact_mps2: float
    eotvos_harlan_approx_mps2: float

    atmospheric_correction_mps2: float
    supplemental_correction_mps2: float

    normal_gravity_measurement_mps2: float
    normal_gravity_reference_mps2: float

    measurement_level_disturbance_mps2: float
    free_air_correction_mps2: float
    free_air_anomaly_mps2: float

    bouguer_slab_correction_mps2: float
    terrain_correction_mps2: float
    bouguer_anomaly_mps2: float
    terrain_corrected_bouguer_anomaly_mps2: float

    @property
    def measurement_level_disturbance_mgal(self) -> float:
        """Measurement-level disturbance in mGal."""
        return float(mps2_to_mgal(self.measurement_level_disturbance_mps2))

    @property
    def eotvos_exact_mgal(self) -> float:
        """Exact Eotvos correction in mGal."""
        return float(mps2_to_mgal(self.eotvos_exact_mps2))

    @property
    def eotvos_harlan_approx_mgal(self) -> float:
        """Approximate Eotvos correction in mGal."""
        return float(mps2_to_mgal(self.eotvos_harlan_approx_mps2))

    @property
    def free_air_anomaly_mgal(self) -> float:
        """Free-air anomaly style value in mGal."""
        return float(mps2_to_mgal(self.free_air_anomaly_mps2))

    @property
    def terrain_corrected_bouguer_anomaly_mgal(self) -> float:
        """Terrain-corrected Bouguer anomaly in mGal."""
        return float(mps2_to_mgal(self.terrain_corrected_bouguer_anomaly_mps2))


# -----------------------------------------------------------------------------
# High-level bundle builders
# -----------------------------------------------------------------------------


def build_stationary_gravity_reduction(
    observed_gravity_mps2: float,
    lat_rad: float,
    measurement_height_m: float,
    *,
    reference_height_m: float = 0.0,
    density_kgpm3: float = STANDARD_BOUGUER_DENSITY_KGPM3,
    terrain_correction_mps2: float = 0.0,
    exact_free_air: bool = True,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> StationaryGravityReduction:
    """
    Build a complete reduction bundle for a scalar gravity observation.

    Parameters
    ----------
    observed_gravity_mps2 : float
        Observed scalar gravity [m/s^2].
    lat_rad : float
        Geodetic latitude [rad].
    measurement_height_m : float
        Observation height [m].
    reference_height_m : float, default=0.0
        Reference height [m].
    density_kgpm3 : float, default=2670
        Bouguer slab density [kg/m^3].
    terrain_correction_mps2 : float, default=0.0
        Terrain correction to add [m/s^2].
    exact_free_air : bool, default=True
        Whether to use exact WGS 84 normal-gravity transfer instead of the
        textbook linear 0.3086 mGal/m approximation.
    apply_atmospheric_correction : bool, default=False
        Whether to apply the simple atmospheric correction.
    supplemental_correction_mps2 : float, default=0.0
        Additional additive correction term.

    Returns
    -------
    StationaryGravityReduction
        Structured reduction result.
    """
    g_obs = float(observed_gravity_mps2)
    phi = float(lat_rad)
    h_meas = float(measurement_height_m)
    h_ref = float(reference_height_m)
    density = _nonnegative_scalar(density_kgpm3, name="density_kgpm3")
    tc = float(terrain_correction_mps2)
    delta_atm = float(atmospheric_correction(h_meas)) if apply_atmospheric_correction else 0.0

    g_corr = g_obs + delta_atm + float(supplemental_correction_mps2)
    gamma_meas = float(normal_gravity(phi, h_meas))
    gamma_ref = float(normal_gravity(phi, h_ref))
    disturbance = g_corr - gamma_meas

    if exact_free_air:
        delta_fa = float(free_air_correction_to_reference_height_exact(phi, h_meas, h_ref))
    else:
        delta_fa = float(free_air_correction_to_reference_height_linear(h_meas, h_ref))

    g_fa = g_corr + delta_fa - gamma_ref
    delta_b = float(bouguer_slab_attraction(h_meas - h_ref, density))
    g_b = g_fa - delta_b
    g_bt = g_b + tc

    return StationaryGravityReduction(
        observed_gravity_mps2=g_obs,
        corrected_observed_gravity_mps2=g_corr,
        atmospheric_correction_mps2=delta_atm,
        supplemental_correction_mps2=float(supplemental_correction_mps2),
        normal_gravity_measurement_mps2=gamma_meas,
        normal_gravity_reference_mps2=gamma_ref,
        measurement_level_disturbance_mps2=disturbance,
        free_air_correction_mps2=delta_fa,
        free_air_anomaly_mps2=g_fa,
        bouguer_slab_correction_mps2=delta_b,
        terrain_correction_mps2=tc,
        bouguer_anomaly_mps2=g_b,
        terrain_corrected_bouguer_anomaly_mps2=g_bt,
    )


def build_moving_base_gravity_reduction_from_specific_force_ned(
    specific_force_ned_mps2: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    measurement_height_m: float,
    *,
    reference_height_m: float = 0.0,
    density_kgpm3: float = STANDARD_BOUGUER_DENSITY_KGPM3,
    terrain_correction_mps2: float = 0.0,
    exact_free_air: bool = True,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> MovingBaseGravityReduction:
    """
    Build a complete moving-base reduction bundle from NED kinematics.

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
    measurement_height_m : float
        Measurement height [m].
    reference_height_m : float, default=0.0
        Reference height [m].
    density_kgpm3 : float, default=2670
        Bouguer slab density [kg/m^3].
    terrain_correction_mps2 : float, default=0.0
        Terrain correction to add [m/s^2].
    exact_free_air : bool, default=True
        Whether to use exact normal-gravity height transfer.
    apply_atmospheric_correction : bool, default=False
        Whether to apply the simple atmospheric correction.
    supplemental_correction_mps2 : float, default=0.0
        Additional additive correction term.

    Returns
    -------
    MovingBaseGravityReduction
        Structured reduction result.
    """
    f_n = _vec3(specific_force_ned_mps2, name="specific_force_ned_mps2")
    v_dot_n = _vec3(v_dot_ned_mps2, name="v_dot_ned_mps2")
    _vec3(v_ned_mps, name="v_ned_mps")  # validation

    phi = float(lat_rad)
    h_meas = float(measurement_height_m)
    h_ref = float(reference_height_m)
    density = _nonnegative_scalar(density_kgpm3, name="density_kgpm3")
    tc = float(terrain_correction_mps2)

    eotvos_exact = eotvos_correction_exact_ned(phi, h_meas, v_ned_mps)
    eotvos_harlan = eotvos_correction_harlan_approx(phi, h_meas, v_ned_mps)
    g_d = recover_total_gravity_down_from_specific_force_ned(
        specific_force_ned_mps2=f_n,
        v_dot_ned_mps2=v_dot_n,
        v_ned_mps=v_ned_mps,
        lat_rad=phi,
        height_m=h_meas,
    )

    delta_atm = float(atmospheric_correction(h_meas)) if apply_atmospheric_correction else 0.0
    g_corr = g_d + delta_atm + float(supplemental_correction_mps2)

    gamma_meas = float(normal_gravity(phi, h_meas))
    gamma_ref = float(normal_gravity(phi, h_ref))
    disturbance = g_corr - gamma_meas

    if exact_free_air:
        delta_fa = float(free_air_correction_to_reference_height_exact(phi, h_meas, h_ref))
    else:
        delta_fa = float(free_air_correction_to_reference_height_linear(h_meas, h_ref))

    g_fa = g_corr + delta_fa - gamma_ref
    delta_b = float(bouguer_slab_attraction(h_meas - h_ref, density))
    g_b = g_fa - delta_b
    g_bt = g_b + tc

    return MovingBaseGravityReduction(
        specific_force_down_mps2=float(f_n[2]),
        vertical_acceleration_down_mps2=float(v_dot_n[2]),
        recovered_total_gravity_down_mps2=g_d,
        corrected_recovered_gravity_down_mps2=g_corr,
        eotvos_exact_mps2=eotvos_exact,
        eotvos_harlan_approx_mps2=eotvos_harlan,
        atmospheric_correction_mps2=delta_atm,
        supplemental_correction_mps2=float(supplemental_correction_mps2),
        normal_gravity_measurement_mps2=gamma_meas,
        normal_gravity_reference_mps2=gamma_ref,
        measurement_level_disturbance_mps2=disturbance,
        free_air_correction_mps2=delta_fa,
        free_air_anomaly_mps2=g_fa,
        bouguer_slab_correction_mps2=delta_b,
        terrain_correction_mps2=tc,
        bouguer_anomaly_mps2=g_b,
        terrain_corrected_bouguer_anomaly_mps2=g_bt,
    )


def build_moving_base_gravity_reduction_from_specific_force_body(
    specific_force_body_mps2: ArrayLike,
    C_n_b: ArrayLike,
    v_dot_ned_mps2: ArrayLike,
    v_ned_mps: ArrayLike,
    lat_rad: float,
    measurement_height_m: float,
    *,
    reference_height_m: float = 0.0,
    density_kgpm3: float = STANDARD_BOUGUER_DENSITY_KGPM3,
    terrain_correction_mps2: float = 0.0,
    exact_free_air: bool = True,
    apply_atmospheric_correction: bool = False,
    supplemental_correction_mps2: float = 0.0,
) -> MovingBaseGravityReduction:
    """
    Body-frame convenience wrapper for
    :func:`build_moving_base_gravity_reduction_from_specific_force_ned`.
    """
    f_b = _vec3(specific_force_body_mps2, name="specific_force_body_mps2")
    C_nb = project_to_so3(_mat3(C_n_b, name="C_n_b"))
    f_n = C_nb @ f_b
    return build_moving_base_gravity_reduction_from_specific_force_ned(
        specific_force_ned_mps2=f_n,
        v_dot_ned_mps2=v_dot_ned_mps2,
        v_ned_mps=v_ned_mps,
        lat_rad=lat_rad,
        measurement_height_m=measurement_height_m,
        reference_height_m=reference_height_m,
        density_kgpm3=density_kgpm3,
        terrain_correction_mps2=terrain_correction_mps2,
        exact_free_air=exact_free_air,
        apply_atmospheric_correction=apply_atmospheric_correction,
        supplemental_correction_mps2=supplemental_correction_mps2,
    )


__all__ = [
    "FloatArray",
    "GRAVITATIONAL_CONSTANT_SI",
    "SEAWATER_DENSITY_KGPM3",
    "STANDARD_BOUGUER_DENSITY_KGPM3",
    "MovingBaseGravityReduction",
    "StationaryGravityReduction",
    "atmospheric_correction",
    "bouguer_anomaly_from_observed_gravity",
    "bouguer_slab_attraction",
    "bouguer_slab_vertical_gradient",
    "build_moving_base_gravity_reduction_from_specific_force_body",
    "build_moving_base_gravity_reduction_from_specific_force_ned",
    "build_stationary_gravity_reduction",
    "disturbance_from_observed_gravity",
    "disturbance_height_transfer_linear",
    "eotvos_correction_exact_ned",
    "eotvos_correction_harlan_approx",
    "free_air_anomaly_from_observed_gravity",
    "free_air_correction_to_reference_height_exact",
    "free_air_correction_to_reference_height_linear",
    "recover_measurement_level_disturbance_from_specific_force_body",
    "recover_measurement_level_disturbance_from_specific_force_ned",
    "recover_total_gravity_down_from_specific_force_ned",
]