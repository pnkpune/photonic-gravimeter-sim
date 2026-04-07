"""
frames.py

Reference-frame, rotation, quaternion, and local-level navigation helpers for the
gravity-aided navigation simulator.

This module sits directly on top of `earth.py`. Its job is to provide one and only
one consistent set of conventions for:
- Earth-Centered Earth-Fixed (ECEF) frame
- local East-North-Up (ENU) frame
- local North-East-Down (NED) frame
- passive direction-cosine matrices (DCMs)
- quaternion utilities that are consistent with those DCMs
- Earth-rate and transport-rate terms used by local-level inertial navigation

Conventions
-----------
Frames
~~~~~~
- ECEF:
    right-handed Cartesian frame fixed to the Earth
    x-axis: intersection of equator and Greenwich meridian
    y-axis: 90 deg east in equatorial plane
    z-axis: Earth rotation axis toward north pole

- ENU:
    local tangent frame with axes East, North, Up

- NED:
    local tangent frame with axes North, East, Down

Rotation matrices / DCMs
~~~~~~~~~~~~~~~~~~~~~~~~
This module uses passive frame transforms in the standard navigation sense:

    v^beta = C^beta_alpha v^alpha

meaning:
- the physical vector is unchanged
- only the coordinates are changed from frame alpha to frame beta

Example:
    v_n = C_n_e @ v_e

means "resolve the same physical vector from ECEF into NED coordinates".

This convention matches the frame-transform notation used in standard navigation
texts and in the cited geodesy/navigation references below.

Quaternions
~~~~~~~~~~~
Quaternions are scalar-first Hamilton quaternions:

    q = [q_w, q_x, q_y, q_z]

and `quaternion_to_dcm(q)` returns the DCM representing the SAME coordinate
transform convention as the DCM functions in this file:

    v_target = C_target_source(q) @ v_source

Thus, quaternions in this file are not a separate convention. They are simply
another representation of the same passive coordinate transform matrix.

Primary references used here
----------------------------
1) NGA.STND.0036_1.0.0_WGS84
   "Department of Defense World Geodetic System 1984, Its Definition and
   Relationships With Local Geodetic Systems"
   URL:
   https://ia801409.us.archive.org/35/items/nga.-stnd.-0036-1.0.0-wgs-84/NGA.STND.0036_1.0.0_WGS84.pdf

   Used for:
   - WGS 84 Earth rotation rate (through `earth.WGS84.omega`)
   - consistency with the ellipsoidal/geodetic framework in `earth.py`

2) Navipedia (ESA / UPC), "Transformations between ECEF and ENU coordinates"
   URL:
   https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates

   Used for:
   - exact ECEF <-> ENU rotation matrix formulas
   - local east, north, up unit vectors expressed in ECEF

3) Enkhmurun Bayasgalan, "Frame Transformations"
   URL:
   https://murundb.github.io/navigation/geodesy/frame_transformations/

   Used for:
   - ECEF <-> NED DCM expressions
   - ENU <-> NED fixed transform matrix

4) Enkhmurun Bayasgalan, "Rotation of Earth"
   URL:
   https://murundb.github.io/navigation/geodesy/rotation_of_earth/

   Used for:
   - Earth rotation vector in ECEF and in local navigation frame

5) INSTINCT / University of Stuttgart,
   "INS/GNSS Loosely-coupled Kalman Filter (local-navigation frame)"
   URL:
   https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

   Used for:
   - local-navigation-frame transport-rate vector:
         omega_en^n = [ v_E/(R_E+h), -v_N/(R_N+h), -v_E tan(phi)/(R_E+h) ]^T
     where R_E is the prime vertical radius and R_N is the meridian radius
   - curvilinear position rates:
         dot(phi)    = v_N / (R_N + h)
         dot(lambda) = v_E / ((R_E + h) cos(phi))
         dot(h)      = -v_D

6) AHRS documentation, quaternion/DCM relations
   URLs:
   - https://ahrs.readthedocs.io/en/latest/special/Chiaverini.html
   - https://ahrs.readthedocs.io/en/latest/filters/quest.html

   Used for:
   - scalar-first Hamilton quaternion convention
   - DCM <-> quaternion formulas

7) Carlo Tomasi, "Vector Representation of Rotations"
   URL:
   https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf

   Used for:
   - Rodrigues formula / exponential map from rotation vector to SO(3)
   - inverse rotation-vector extraction from a DCM

Design note
-----------
This module intentionally avoids premature "full INS mechanization" logic.
It provides only the geometry/kinematics pieces that the later IMU and INS
modules need in a trustworthy, testable form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .earth import (
    WGS84,
    ecef_to_geodetic,
    geodetic_to_ecef,
    meridian_radius,
    prime_vertical_radius,
)

FloatArray = NDArray[np.float64]


# Fixed transform between local ENU and local NED coordinates.
#
# Reference
# ---------
# MurunDB "Frame Transformations", ENU and NED section:
#   R^NED_ENU = [[0,1,0],[1,0,0],[0,0,-1]]
#
# It maps:
#   [E, N, U]^T -> [N, E, D]^T
#
R_NED_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

R_ENU_NED = R_NED_ENU.T


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _check_vector3(v: ArrayLike, *, name: str = "vector") -> FloatArray:
    """
    Validate and return a 3-vector as shape (3,) float64 NumPy array.
    """
    arr = _as_float_array(v).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {arr.shape}.")
    return arr


def _check_matrix33(C: ArrayLike, *, name: str = "matrix") -> FloatArray:
    """
    Validate and return a 3x3 matrix as float64 NumPy array.
    """
    arr = _as_float_array(C)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be shape (3, 3), got shape {arr.shape}.")
    return arr


def _normalize(v: ArrayLike, *, eps: float = 1e-15, name: str = "vector") -> FloatArray:
    """
    Normalize a 3-vector.

    Raises
    ------
    ValueError
        If the vector norm is too small.
    """
    vec = _check_vector3(v, name=name)
    n = np.linalg.norm(vec)
    if n < eps:
        raise ValueError(f"{name} norm is too small to normalize: {n:.3e}")
    return vec / n


def wrap_angle_pi(angle_rad: ArrayLike):
    """
    Wrap angle(s) to [-pi, pi).

    This is a utility function, not a geodesy formula. It is used to keep
    longitudes and yaw-like variables numerically well behaved.
    """
    angle = _as_float_array(angle_rad)
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if np.asarray(angle_rad).ndim == 0:
        return float(wrapped)
    return wrapped


def skew(v: ArrayLike) -> FloatArray:
    r"""
    Return the skew-symmetric cross-product matrix [v]_x.

    For v = [v1, v2, v3]^T:

        [v]_x =
        [[  0, -v3,  v2],
         [ v3,   0, -v1],
         [-v2,  v1,   0]]

    such that for any 3-vector x:

        [v]_x x = v x x

    Reference
    ---------
    This is the standard antisymmetric matrix representation used throughout
    rigid-body kinematics, navigation, and the AHRS quaternion/DCM formulas.

    See:
    - AHRS QUEST notes:
      https://ahrs.readthedocs.io/en/latest/filters/quest.html
    - AHRS Chiaverini notes:
      https://ahrs.readthedocs.io/en/latest/special/Chiaverini.html
    """
    v = _check_vector3(v, name="v")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def unskew(S: ArrayLike) -> FloatArray:
    """
    Inverse of `skew(...)` for an ideal skew-symmetric matrix.

    Parameters
    ----------
    S : array-like, shape (3, 3)

    Returns
    -------
    np.ndarray, shape (3,)
        Vector v such that skew(v) == S for a perfect skew matrix.
    """
    S = _check_matrix33(S, name="S")
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def project_to_so3(C: ArrayLike) -> FloatArray:
    r"""
    Project a near-rotation matrix onto SO(3) using SVD.

    Given a nearly orthonormal matrix M, the closest proper rotation matrix in
    Frobenius norm is:

        C = U diag(1, 1, det(U V^T)) V^T

    where M = U Sigma V^T is the singular value decomposition.

    This is not a navigation-specific formula, but it is an important numerical
    safeguard when repeated floating-point updates cause a DCM to drift away from
    orthonormality.
    """
    M = _check_matrix33(C, name="C")
    U, _, Vt = np.linalg.svd(M)
    Cproj = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return Cproj


def is_rotation_matrix(C: ArrayLike, *, atol: float = 1e-10) -> bool:
    """
    Check whether a matrix is a proper rotation matrix.

    Conditions checked
    ------------------
    - C^T C ~= I
    - det(C) ~= +1
    """
    C = _check_matrix33(C, name="C")
    return np.allclose(C.T @ C, np.eye(3), atol=atol) and np.isclose(
        np.linalg.det(C), 1.0, atol=atol
    )


def rot_x(angle_rad: float) -> FloatArray:
    r"""
    Principal passive rotation matrix about x-axis.

    Formula
    -------
    Using the standard principal rotation matrix convention:

        R1(phi) =
        [[1,      0,       0],
         [0, cos(phi), sin(phi)],
         [0,-sin(phi), cos(phi)]]

    Reference
    ---------
    MurunDB "Rotation Matrix", principal rotation matrices:
    https://murundb.github.io/kinematics/rotations/rotation_matrix/
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def rot_y(angle_rad: float) -> FloatArray:
    r"""
    Principal passive rotation matrix about y-axis.

    Formula
    -------
        R2(theta) =
        [[ cos(theta), 0, -sin(theta)],
         [          0, 1,           0],
         [ sin(theta), 0,  cos(theta)]]

    Reference
    ---------
    MurunDB "Rotation Matrix", principal rotation matrices:
    https://murundb.github.io/kinematics/rotations/rotation_matrix/
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rot_z(angle_rad: float) -> FloatArray:
    r"""
    Principal passive rotation matrix about z-axis.

    Formula
    -------
        R3(psi) =
        [[ cos(psi), sin(psi), 0],
         [-sin(psi), cos(psi), 0],
         [        0,        0, 1]]

    Reference
    ---------
    MurunDB "Rotation Matrix", principal rotation matrices:
    https://murundb.github.io/kinematics/rotations/rotation_matrix/
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def dcm_ecef_to_enu(lat_rad: float, lon_rad: float) -> FloatArray:
    r"""
    DCM that transforms an ECEF-resolved vector into local ENU coordinates.

    Convention
    ----------
    For a vector v:

        v_enu = C_enu_ecef @ v_ecef

    Formula
    -------
    From the Navipedia ECEF/ENU derivation, the local East, North, Up unit vectors
    expressed in ECEF are:

        e_hat = (-sin(lambda),  cos(lambda), 0)
        n_hat = (-cos(lambda) sin(phi), -sin(lambda) sin(phi), cos(phi))
        u_hat = ( cos(lambda) cos(phi),  sin(lambda) cos(phi), sin(phi))

    Therefore the ECEF -> ENU DCM is:

        C_enu_ecef =
        [[-sin(lambda),              cos(lambda),             0],
         [-cos(lambda) sin(phi), -sin(lambda) sin(phi), cos(phi)],
         [ cos(lambda) cos(phi),  sin(lambda) cos(phi), sin(phi)]]

    where:
    - phi    = geodetic latitude
    - lambda = longitude

    Reference
    ---------
    Navipedia:
    https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    See equations (3), (4), and the transpose relation for ECEF -> ENU.
    """
    sphi = np.sin(lat_rad)
    cphi = np.cos(lat_rad)
    slam = np.sin(lon_rad)
    clam = np.cos(lon_rad)

    return np.array(
        [
            [-slam, clam, 0.0],
            [-clam * sphi, -slam * sphi, cphi],
            [clam * cphi, slam * cphi, sphi],
        ],
        dtype=np.float64,
    )


def dcm_enu_to_ecef(lat_rad: float, lon_rad: float) -> FloatArray:
    """
    DCM that transforms a local ENU-resolved vector into ECEF coordinates.

    This is the transpose / inverse of `dcm_ecef_to_enu(...)`:

        C_ecef_enu = C_enu_ecef^T

    Reference
    ---------
    Navipedia:
    https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    """
    return dcm_ecef_to_enu(lat_rad, lon_rad).T


def dcm_ecef_to_ned(lat_rad: float, lon_rad: float) -> FloatArray:
    r"""
    DCM that transforms an ECEF-resolved vector into local NED coordinates.

    Convention
    ----------
    For a vector v:

        v_ned = C_ned_ecef @ v_ecef

    Formula
    -------
    One way to derive NED from ECEF is:

        C_ned_ecef = C_ned_enu @ C_enu_ecef

    Using:
        C_ned_enu =
        [[0,1,0],
         [1,0,0],
         [0,0,-1]]

    and the standard ECEF -> ENU DCM gives:

        C_ned_ecef =
        [[-sin(phi) cos(lambda), -sin(phi) sin(lambda),  cos(phi)],
         [         -sin(lambda),           cos(lambda),        0],
         [-cos(phi) cos(lambda), -cos(phi) sin(lambda), -sin(phi)]]

    Reference
    ---------
    - MurunDB "Frame Transformations":
      https://murundb.github.io/navigation/geodesy/frame_transformations/
      See the ECEF <-> NED expressions and ENU <-> NED relation.
    - Consistent with Navipedia ECEF <-> ENU expressions.
    """
    return R_NED_ENU @ dcm_ecef_to_enu(lat_rad, lon_rad)


def dcm_ned_to_ecef(lat_rad: float, lon_rad: float) -> FloatArray:
    """
    DCM that transforms a local NED-resolved vector into ECEF coordinates.

    This is the transpose / inverse of `dcm_ecef_to_ned(...)`:

        C_ecef_ned = C_ned_ecef^T
    """
    return dcm_ecef_to_ned(lat_rad, lon_rad).T


def dcm_ecef_to_ned_from_ecef_position(
    x_m: float,
    y_m: float,
    z_m: float,
) -> FloatArray:
    """
    Convenience helper: infer geodetic latitude/longitude from an ECEF position
    and return the corresponding ECEF -> NED DCM.

    This is useful when a later module has an ECEF state but wants the local
    navigation frame without explicitly carrying geodetic coordinates around.
    """
    lat, lon, _ = ecef_to_geodetic(x_m, y_m, z_m)
    return dcm_ecef_to_ned(lat, lon)


def enu_to_ned(v_enu: ArrayLike) -> FloatArray:
    """
    Convert a local ENU-resolved vector into NED coordinates.

    Formula
    -------
        v_ned = R_ned_enu @ v_enu
    """
    v = _check_vector3(v_enu, name="v_enu")
    return R_NED_ENU @ v


def ned_to_enu(v_ned: ArrayLike) -> FloatArray:
    """
    Convert a local NED-resolved vector into ENU coordinates.

    Formula
    -------
        v_enu = R_enu_ned @ v_ned
    """
    v = _check_vector3(v_ned, name="v_ned")
    return R_ENU_NED @ v


def ecef_vector_to_enu(v_ecef: ArrayLike, lat_rad: float, lon_rad: float) -> FloatArray:
    """
    Resolve a vector from ECEF coordinates into local ENU coordinates.
    """
    v = _check_vector3(v_ecef, name="v_ecef")
    return dcm_ecef_to_enu(lat_rad, lon_rad) @ v


def enu_vector_to_ecef(v_enu: ArrayLike, lat_rad: float, lon_rad: float) -> FloatArray:
    """
    Resolve a vector from ENU coordinates into ECEF coordinates.
    """
    v = _check_vector3(v_enu, name="v_enu")
    return dcm_enu_to_ecef(lat_rad, lon_rad) @ v


def ecef_vector_to_ned(v_ecef: ArrayLike, lat_rad: float, lon_rad: float) -> FloatArray:
    """
    Resolve a vector from ECEF coordinates into local NED coordinates.
    """
    v = _check_vector3(v_ecef, name="v_ecef")
    return dcm_ecef_to_ned(lat_rad, lon_rad) @ v


def ned_vector_to_ecef(v_ned: ArrayLike, lat_rad: float, lon_rad: float) -> FloatArray:
    """
    Resolve a vector from local NED coordinates into ECEF coordinates.
    """
    v = _check_vector3(v_ned, name="v_ned")
    return dcm_ned_to_ecef(lat_rad, lon_rad) @ v


def ecef_to_ned_position(
    p_ecef_m: ArrayLike,
    ref_lat_rad: float,
    ref_lon_rad: float,
    ref_height_m: float,
) -> FloatArray:
    r"""
    Convert an ECEF position to a local NED position with respect to a geodetic reference origin.

    Formula
    -------
    Let:
    - p_e be the ECEF position of the point
    - p_ref,e be the ECEF position of the local tangent-plane origin
    - C_n_e be the ECEF -> NED DCM at the reference origin

    Then:

        p_n = C_n_e (p_e - p_ref,e)

    This is the standard local-tangent-plane position relation.

    Reference
    ---------
    MurunDB "Frame Transformations":
    https://murundb.github.io/navigation/geodesy/frame_transformations/
    See:
        r_lb^l = R_e^l (r_eb^e - r_el^e)
    with notation adapted here to NED.
    """
    p_e = _check_vector3(p_ecef_m, name="p_ecef_m")
    p_ref = np.array(geodetic_to_ecef(ref_lat_rad, ref_lon_rad, ref_height_m), dtype=np.float64)
    C_n_e = dcm_ecef_to_ned(ref_lat_rad, ref_lon_rad)
    return C_n_e @ (p_e - p_ref)


def ned_to_ecef_position(
    p_ned_m: ArrayLike,
    ref_lat_rad: float,
    ref_lon_rad: float,
    ref_height_m: float,
) -> FloatArray:
    r"""
    Convert a local NED position to ECEF coordinates with respect to a geodetic reference origin.

    Formula
    -------
    With the same notation as `ecef_to_ned_position(...)`:

        p_e = p_ref,e + C_e_n p_n

    where C_e_n = C_n_e^T.

    Reference
    ---------
    Same local-tangent-plane position relation as in MurunDB:
    https://murundb.github.io/navigation/geodesy/frame_transformations/
    """
    p_n = _check_vector3(p_ned_m, name="p_ned_m")
    p_ref = np.array(geodetic_to_ecef(ref_lat_rad, ref_lon_rad, ref_height_m), dtype=np.float64)
    C_e_n = dcm_ned_to_ecef(ref_lat_rad, ref_lon_rad)
    return p_ref + C_e_n @ p_n


def earth_rotation_rate_radps() -> float:
    """
    Return the WGS 84 Earth rotation rate [rad/s].

    Reference
    ---------
    WGS 84 defining parameter omega through `earth.WGS84.omega`.
    Also summarized in:
    https://murundb.github.io/navigation/geodesy/rotation_of_earth/
    """
    return float(WGS84.omega)


def earth_rate_ecef() -> FloatArray:
    r"""
    Earth rotation vector resolved in ECEF coordinates [rad/s].

    Formula
    -------
        omega_ie^e = [0, 0, omega_ie]^T

    Reference
    ---------
    MurunDB "Rotation of Earth":
    https://murundb.github.io/navigation/geodesy/rotation_of_earth/
    """
    return np.array([0.0, 0.0, WGS84.omega], dtype=np.float64)


def earth_rate_ned(lat_rad: float) -> FloatArray:
    r"""
    Earth rotation vector resolved in local NED coordinates [rad/s].

    Formula
    -------
        omega_ie^n = [omega_ie cos(phi), 0, -omega_ie sin(phi)]^T

    where phi is geodetic latitude.

    Reference
    ---------
    MurunDB "Rotation of Earth":
    https://murundb.github.io/navigation/geodesy/rotation_of_earth/
    """
    return np.array(
        [
            WGS84.omega * np.cos(lat_rad),
            0.0,
            -WGS84.omega * np.sin(lat_rad),
        ],
        dtype=np.float64,
    )


def transport_rate_ned(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> FloatArray:
    r"""
    Transport rate of the local NED frame with respect to the Earth, resolved in NED [rad/s].

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s] = [v_N, v_E, v_D].

    Formula
    -------
    The standard local-navigation-frame transport rate is:

        omega_en^n =
        [ v_E / (R_E + h),
         -v_N / (R_N + h),
         -v_E tan(phi) / (R_E + h) ]^T

    In this implementation:
    - R_E = prime vertical radius N(phi)
    - R_N = meridian radius M(phi)

    so numerically:

        omega_en^n =
        [ v_E / (N + h),
         -v_N / (M + h),
         -v_E tan(phi) / (N + h) ]^T

    Notes
    -----
    Unfortunately the literature is inconsistent in symbols:
    some authors use R_E / R_N,
    others use R_N / R_M,
    and some write N(phi) / M(phi).
    This file uses the unambiguous geodetic names:
    - prime vertical radius = N(phi)
    - meridian radius       = M(phi)

    Reference
    ---------
    INSTINCT / University of Stuttgart:
    https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

    See the transport-rate formula:
        omega_en^n = [ v_E/(R_E+h), -v_N/(R_N+h), -v_E tan(phi)/(R_E+h) ]^T
    """
    v = _check_vector3(v_ned_mps, name="v_ned_mps")
    v_n, v_e, _ = v

    N = float(prime_vertical_radius(lat_rad))
    M = float(meridian_radius(lat_rad))

    return np.array(
        [
            v_e / (N + height_m),
            -v_n / (M + height_m),
            -v_e * np.tan(lat_rad) / (N + height_m),
        ],
        dtype=np.float64,
    )


def navigation_frame_rate_ned(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> FloatArray:
    r"""
    Total local navigation-frame rate with respect to inertial space, resolved in NED [rad/s].

    Formula
    -------
        omega_in^n = omega_ie^n + omega_en^n

    where:
    - omega_ie^n is Earth rotation resolved in NED
    - omega_en^n is transport rate of the local frame with respect to Earth

    This quantity appears in local-level INS attitude and velocity mechanization.

    References
    ----------
    - MurunDB local tangent-plane navigation equations:
      https://murundb.github.io/navigation/inertial_navigation/inertial_navigation_local_tangent_plane/
    - INSTINCT / University of Stuttgart:
      https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html
    """
    return earth_rate_ned(lat_rad) + transport_rate_ned(lat_rad, height_m, v_ned_mps)


def geodetic_rates_from_ned_velocity(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> FloatArray:
    r"""
    Convert NED velocity into geodetic coordinate rates.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : array-like, shape (3,)
        NED velocity [m/s] = [v_N, v_E, v_D].

    Returns
    -------
    np.ndarray, shape (3,)
        [lat_dot, lon_dot, h_dot] with units [rad/s, rad/s, m/s].

    Formula
    -------
        dot(phi)    = v_N / (M + h)
        dot(lambda) = v_E / ((N + h) cos(phi))
        dot(h)      = -v_D

    where:
    - M = meridian radius of curvature
    - N = prime vertical radius of curvature

    Reference
    ---------
    INSTINCT / University of Stuttgart:
    https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

    See the local-navigation-frame position-rate equations.
    """
    v = _check_vector3(v_ned_mps, name="v_ned_mps")
    v_n, v_e, v_d = v

    N = float(prime_vertical_radius(lat_rad))
    M = float(meridian_radius(lat_rad))

    lat_dot = v_n / (M + height_m)
    lon_dot = v_e / ((N + height_m) * np.cos(lat_rad))
    h_dot = -v_d

    return np.array([lat_dot, lon_dot, h_dot], dtype=np.float64)


def geodetic_step_from_ned_velocity(
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
    dt_s: float,
) -> Tuple[float, float, float]:
    """
    First-order curvilinear position update from NED velocity.

    Formula
    -------
    Uses:

        [lat_dot, lon_dot, h_dot] = geodetic_rates_from_ned_velocity(...)

    then:

        lat_{k+1} = lat_k + lat_dot * dt
        lon_{k+1} = lon_k + lon_dot * dt
        h_{k+1}   = h_k   + h_dot * dt

    This is the simplest geodetic propagation needed for early simulation and
    truth-model modules. Higher-order integration can be added later in the
    INS module if needed.
    """
    rates = geodetic_rates_from_ned_velocity(lat_rad, height_m, v_ned_mps)
    lat_new = lat_rad + rates[0] * dt_s
    lon_new = wrap_angle_pi(lon_rad + rates[1] * dt_s)
    h_new = height_m + rates[2] * dt_s
    return float(lat_new), float(lon_new), float(h_new)


def dcm_from_rotvec(rotvec_rad: ArrayLike) -> FloatArray:
    r"""
    Convert a rotation vector to a DCM using Rodrigues' formula.

    Parameters
    ----------
    rotvec_rad : array-like, shape (3,)
        Rotation vector r = theta * u, where:
        - theta = ||r|| is rotation angle [rad]
        - u is unit rotation axis

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix C.

    Formula
    -------
    Let:
        theta = ||r||
        u = r / theta
        U = [u]_x

    Then Rodrigues' formula gives:

        C = I cos(theta) + (1 - cos(theta)) u u^T + U sin(theta)

    Equivalently, in exponential-map form:

        C = exp([r]_x)

    Small-angle implementation
    --------------------------
    For numerical stability, this implementation evaluates:

        A(theta) = sin(theta) / theta
        B(theta) = (1 - cos(theta)) / theta^2

    and then:

        C = I + A [r]_x + B [r]_x^2

    with series expansions when theta is very small.

    Reference
    ---------
    Carlo Tomasi, "Vector Representation of Rotations":
    https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    """
    r = _check_vector3(rotvec_rad, name="rotvec_rad")
    theta = np.linalg.norm(r)
    K = skew(r)

    if theta < 1e-8:
        # Series:
        # sin(theta)/theta       = 1 - theta^2/6 + theta^4/120 + ...
        # (1-cos(theta))/theta^2 = 1/2 - theta^2/24 + theta^4/720 + ...
        theta2 = theta * theta
        A = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        B = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)

    C = np.eye(3) + A * K + B * (K @ K)
    return project_to_so3(C)


def rotvec_from_dcm(C: ArrayLike) -> FloatArray:
    r"""
    Convert a DCM to a rotation vector.

    Parameters
    ----------
    C : array-like, shape (3, 3)
        Proper rotation matrix.

    Returns
    -------
    np.ndarray, shape (3,)
        Rotation vector r = theta * u with ||r|| in [0, pi].

    Formula
    -------
    Following the Rodrigues inverse relation, define:

        A = (C - C^T) / 2
        rho = [A_32, A_13, A_21]^T
        s = ||rho||
        c = (trace(C) - 1) / 2

    Then:
    - if s = 0 and c = 1, rotation is zero
    - otherwise theta = atan2(s, c)
    - u = rho / s
    - r = theta * u

    For the theta = pi singular case, use a nonzero column of (C + I) to recover
    the axis.

    Reference
    ---------
    Carlo Tomasi, "Vector Representation of Rotations":
    https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    """
    R = project_to_so3(C)
    A = 0.5 * (R - R.T)
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=np.float64)

    s = np.linalg.norm(rho)
    c = 0.5 * (np.trace(R) - 1.0)
    c = np.clip(c, -1.0, 1.0)

    if s < 1e-12 and c > 0.0:
        return np.zeros(3, dtype=np.float64)

    if s < 1e-12 and c <= 0.0:
        # theta = pi case
        RpI = R + np.eye(3)
        axis = None
        for i in range(3):
            col = RpI[:, i]
            if np.linalg.norm(col) > 1e-10:
                axis = col / np.linalg.norm(col)
                break
        if axis is None:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.pi * axis

    theta = np.arctan2(s, c)
    axis = rho / s
    return theta * axis


def quaternion_normalize(q: ArrayLike) -> FloatArray:
    """
    Normalize a scalar-first quaternion.

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion [w, x, y, z].
    """
    q = _as_float_array(q).reshape(-1)
    if q.shape != (4,):
        raise ValueError(f"q must be shape (4,), got shape {q.shape}.")
    n = np.linalg.norm(q)
    if n < 1e-15:
        raise ValueError("Quaternion norm is too small to normalize.")
    return q / n


def quaternion_conjugate(q: ArrayLike) -> FloatArray:
    """
    Quaternion conjugate for scalar-first Hamilton quaternion.

    For q = [w, x, y, z], the conjugate is:

        q* = [w, -x, -y, -z]

    For unit quaternions, this is also the inverse.
    """
    q = quaternion_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> FloatArray:
    r"""
    Hamilton product of two scalar-first quaternions.

    Formula
    -------
    For q = [q_w, q_v] and p = [p_w, p_v]:

        q ⊗ p =
        [ q_w p_w - q_v^T p_v,
          q_w p_v + p_w q_v + q_v x p_v ]

    This is the Hamilton product.

    Reference
    ---------
    AHRS quaternion documentation:
    https://ahrs.readthedocs.io/en/latest/quaternion/quaternion.html
    """
    q = quaternion_normalize(q1)
    p = quaternion_normalize(q2)

    qw, qx, qy, qz = q
    pw, px, py, pz = p

    out = np.array(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ],
        dtype=np.float64,
    )
    return quaternion_normalize(out)


def quaternion_from_rotvec(rotvec_rad: ArrayLike) -> FloatArray:
    r"""
    Convert a rotation vector to a scalar-first Hamilton quaternion.

    Parameters
    ----------
    rotvec_rad : array-like, shape (3,)
        Rotation vector r = theta * u.

    Formula
    -------
    Let:
        theta = ||r||
        u = r / theta

    Then the unit quaternion is:

        q_w = cos(theta / 2)
        q_v = u sin(theta / 2)

    so:

        q = [cos(theta/2), u sin(theta/2)]

    Small-angle implementation
    --------------------------
    For small theta, the vector factor sin(theta/2)/theta is evaluated using
    a stable series expansion.

    References
    ----------
    AHRS Chiaverini notes (quaternion from axis-angle form):
    https://ahrs.readthedocs.io/en/latest/special/Chiaverini.html
    """
    r = _check_vector3(rotvec_rad, name="rotvec_rad")
    theta = np.linalg.norm(r)

    if theta < 1e-8:
        # sin(theta/2)/theta = 1/2 - theta^2/48 + theta^4/3840 + ...
        theta2 = theta * theta
        s_over_theta = 0.5 - theta2 / 48.0 + theta2 * theta2 / 3840.0
        q = np.array([1.0, *(s_over_theta * r)], dtype=np.float64)
    else:
        half = 0.5 * theta
        q = np.empty(4, dtype=np.float64)
        q[0] = np.cos(half)
        q[1:] = (np.sin(half) / theta) * r

    return quaternion_normalize(q)


def quaternion_to_dcm(q: ArrayLike) -> FloatArray:
    r"""
    Convert a scalar-first Hamilton quaternion to a DCM.

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion [q_w, q_x, q_y, q_z].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Direction cosine matrix C.

    Formula
    -------
    Let q = [q_w, q_v], where q_v is the 3-vector part. Then:

        C(q) = (q_w^2 - q_v^T q_v) I
               + 2 q_v q_v^T
               + 2 q_w [q_v]_x

    This implementation interprets the resulting matrix as the same passive
    coordinate transform used everywhere else in this module.

    References
    ----------
    - AHRS QUEST notes:
      https://ahrs.readthedocs.io/en/latest/filters/quest.html
    - AHRS Chiaverini notes:
      https://ahrs.readthedocs.io/en/latest/special/Chiaverini.html
    """
    q = quaternion_normalize(q)
    qw = q[0]
    qv = q[1:]

    C = (qw * qw - qv @ qv) * np.eye(3) + 2.0 * np.outer(qv, qv) + 2.0 * qw * skew(qv)
    return project_to_so3(C)


def dcm_to_quaternion(C: ArrayLike) -> FloatArray:
    r"""
    Convert a DCM to a scalar-first Hamilton quaternion.

    Parameters
    ----------
    C : array-like, shape (3, 3)

    Returns
    -------
    np.ndarray, shape (4,)
        Quaternion [q_w, q_x, q_y, q_z].

    Formula
    -------
    This uses the standard stable branch-based extraction consistent with the
    quaternion/DCM relation:

        C(q) = (q_w^2 - q_v^T q_v) I + 2 q_v q_v^T + 2 q_w [q_v]_x

    The corresponding direct formulas are summarized in the AHRS Chiaverini notes.

    Reference
    ---------
    AHRS Chiaverini notes:
    https://ahrs.readthedocs.io/en/latest/special/Chiaverini.html
    """
    R = project_to_so3(C)
    tr = np.trace(R)

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)

    # Canonicalize sign to reduce discontinuities:
    # q and -q represent the same rotation. Force nonnegative scalar part.
    q = quaternion_normalize(q)
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_rotate_vector(q: ArrayLike, v: ArrayLike) -> FloatArray:
    """
    Rotate / transform a 3-vector using the DCM corresponding to quaternion q.

    Convention
    ----------
    This function is intentionally simple and unambiguous:

        v_out = quaternion_to_dcm(q) @ v

    Therefore q is interpreted with the SAME passive transform convention as
    all DCMs in this module.

    Parameters
    ----------
    q : array-like, shape (4,)
        Scalar-first Hamilton quaternion.
    v : array-like, shape (3,)
        Source-frame vector coordinates.
    """
    C = quaternion_to_dcm(q)
    vec = _check_vector3(v, name="v")
    return C @ vec


def dcm_from_rpy_321(roll_rad: float, pitch_rad: float, yaw_rad: float) -> FloatArray:
    r"""
    Construct a 3-2-1 (yaw-pitch-roll) passive DCM.

    Convention
    ----------
    This returns the matrix:

        C = R1(roll) @ R2(pitch) @ R3(yaw)

    using the passive principal-rotation matrices defined in this module.

    This helper is mainly for simulation/debug convenience. In the INS itself,
    quaternions or incremental rotation vectors are usually preferred.

    Important
    ---------
    Euler-angle conventions vary widely across aerospace, robotics, and graphics.
    This file defines the convention explicitly through the underlying principal
    passive rotation matrices, rather than relying on overloaded words like
    "yaw-pitch-roll" alone.
    """
    return rot_x(roll_rad) @ rot_y(pitch_rad) @ rot_z(yaw_rad)


def orthonormal_basis_from_down(down_ned: ArrayLike) -> FloatArray:
    """
    Build a right-handed orthonormal basis whose third axis is the provided
    'down' direction.

    This is a generic geometry helper used in some truth-model construction
    and debugging utilities. It is not tied to any specific Earth model.
    """
    z = _normalize(down_ned, name="down_ned")

    # Choose a helper axis not parallel to z
    if abs(z[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x = np.cross(helper, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    return np.column_stack((x, y, z))


@dataclass(frozen=True)
class LocalLevelFrame:
    """
    Convenience container for a geodetic local-level frame.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad]
    lon_rad : float
        Longitude [rad]
    height_m : float
        Ellipsoidal height [m]

    This class is not strictly necessary, but it helps keep later modules cleaner:
    the same reference origin can carry its DCMs, origin ECEF position, and local
    rate helpers together.
    """

    lat_rad: float
    lon_rad: float
    height_m: float

    @property
    def origin_ecef_m(self) -> FloatArray:
        """ECEF position of the local tangent-plane origin [m]."""
        return np.array(
            geodetic_to_ecef(self.lat_rad, self.lon_rad, self.height_m),
            dtype=np.float64,
        )

    @property
    def C_n_e(self) -> FloatArray:
        """ECEF -> NED DCM at the local origin."""
        return dcm_ecef_to_ned(self.lat_rad, self.lon_rad)

    @property
    def C_e_n(self) -> FloatArray:
        """NED -> ECEF DCM at the local origin."""
        return dcm_ned_to_ecef(self.lat_rad, self.lon_rad)

    @property
    def C_enu_e(self) -> FloatArray:
        """ECEF -> ENU DCM at the local origin."""
        return dcm_ecef_to_enu(self.lat_rad, self.lon_rad)

    @property
    def C_e_enu(self) -> FloatArray:
        """ENU -> ECEF DCM at the local origin."""
        return dcm_enu_to_ecef(self.lat_rad, self.lon_rad)

    def vector_ecef_to_ned(self, v_ecef: ArrayLike) -> FloatArray:
        """Resolve an ECEF vector into this frame's NED coordinates."""
        return self.C_n_e @ _check_vector3(v_ecef, name="v_ecef")

    def vector_ned_to_ecef(self, v_ned: ArrayLike) -> FloatArray:
        """Resolve a NED vector into ECEF coordinates."""
        return self.C_e_n @ _check_vector3(v_ned, name="v_ned")

    def position_ecef_to_ned(self, p_ecef_m: ArrayLike) -> FloatArray:
        """Convert an ECEF position to local NED coordinates."""
        p = _check_vector3(p_ecef_m, name="p_ecef_m")
        return self.C_n_e @ (p - self.origin_ecef_m)

    def position_ned_to_ecef(self, p_ned_m: ArrayLike) -> FloatArray:
        """Convert a local NED position to ECEF coordinates."""
        p = _check_vector3(p_ned_m, name="p_ned_m")
        return self.origin_ecef_m + self.C_e_n @ p

    def earth_rate_ned(self) -> FloatArray:
        """Earth rotation vector resolved in this NED frame [rad/s]."""
        return earth_rate_ned(self.lat_rad)

    def transport_rate_ned(self, v_ned_mps: ArrayLike) -> FloatArray:
        """Transport rate resolved in this NED frame [rad/s]."""
        return transport_rate_ned(self.lat_rad, self.height_m, v_ned_mps)

    def navigation_frame_rate_ned(self, v_ned_mps: ArrayLike) -> FloatArray:
        """Total navigation-frame rate resolved in this NED frame [rad/s]."""
        return navigation_frame_rate_ned(self.lat_rad, self.height_m, v_ned_mps)

    def geodetic_rates_from_ned_velocity(self, v_ned_mps: ArrayLike) -> FloatArray:
        """[lat_dot, lon_dot, h_dot] induced by local NED velocity."""
        return geodetic_rates_from_ned_velocity(self.lat_rad, self.height_m, v_ned_mps)

    @classmethod
    def from_ecef_position(cls, x_m: float, y_m: float, z_m: float) -> "LocalLevelFrame":
        """
        Build a local-level frame centered at the geodetic location corresponding
        to the provided ECEF point.
        """
        lat, lon, h = ecef_to_geodetic(x_m, y_m, z_m)
        return cls(float(lat), float(lon), float(h))


__all__ = [
    "FloatArray",
    "LocalLevelFrame",
    "R_ENU_NED",
    "R_NED_ENU",
    "dcm_ecef_to_enu",
    "dcm_ecef_to_ned",
    "dcm_ecef_to_ned_from_ecef_position",
    "dcm_enu_to_ecef",
    "dcm_from_rotvec",
    "dcm_from_rpy_321",
    "dcm_ned_to_ecef",
    "dcm_to_quaternion",
    "earth_rate_ecef",
    "earth_rate_ned",
    "earth_rotation_rate_radps",
    "ecef_to_ned_position",
    "ecef_vector_to_enu",
    "ecef_vector_to_ned",
    "enu_to_ned",
    "enu_vector_to_ecef",
    "geodetic_rates_from_ned_velocity",
    "geodetic_step_from_ned_velocity",
    "is_rotation_matrix",
    "navigation_frame_rate_ned",
    "ned_to_ecef_position",
    "ned_to_enu",
    "ned_vector_to_ecef",
    "orthonormal_basis_from_down",
    "project_to_so3",
    "quaternion_conjugate",
    "quaternion_from_rotvec",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_rotate_vector",
    "quaternion_to_dcm",
    "rot_x",
    "rot_y",
    "rot_z",
    "rotvec_from_dcm",
    "skew",
    "transport_rate_ned",
    "unskew",
    "wrap_angle_pi",
]