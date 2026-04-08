"""
earth.py

Core WGS 84 ellipsoid, geodetic conversion, and normal-gravity utilities for the
gravity-aided navigation simulator.

This module is intentionally foundational. Every later component in the repository
(trajectory generation, IMU mechanization, gravity preprocessing, map matching,
and navigation performance evaluation) should depend on this file for Earth constants
and geodetic conventions instead of redefining its own approximations.

Conventions
-----------
- Angles are in radians internally.
- Geodetic latitude is used (not geocentric latitude).
- Longitudes are east-positive.
- Heights are ellipsoidal heights above the WGS 84 reference ellipsoid.
- Gravity is returned in m/s^2 unless explicitly converted to mGal.

Primary references used by this module
--------------------------------------
1) NGA.STND.0036_1.0.0_WGS84 (2014-07-08)
   "Department of Defense World Geodetic System 1984, Its Definition and
   Relationships With Local Geodetic Systems"
   URL:
   https://ia801409.us.archive.org/35/items/nga.-stnd.-0036-1.0.0-wgs-84/NGA.STND.0036_1.0.0_WGS84.pdf

   Relevant parts:
   - Table 3.1: defining parameters a, 1/f, GM, omega
   - Chapter 4: ellipsoidal gravity formula
   - Section 4.2: Somigliana normal gravity on the ellipsoid
   - Section 4.3: truncated Taylor expansion above the ellipsoid

2) James R. Clynch, "Geodetic Coordinate Conversions", Naval Postgraduate School, 2002
   URL:
   https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf

   Relevant parts:
   - Prime vertical radius of curvature
   - Geodetic <-> ECEF conversion formulas
   - Practical iterative ECEF -> geodetic conversion

3) AHRS geodesy documentation, "World Geodetic System (1984)"
   URL:
   https://ahrs.readthedocs.io/en/latest/geodesy/wgs84.html

   This is not the primary standard, but it is a useful readable summary of the
   exact helper quantities q0, q0', m, gamma_e, gamma_p, Somigliana's formula,
   and the NGA height-Taylor expression, all tied back to WGS 84.

Notes on model fidelity
-----------------------
- The normal gravity model here is the WGS 84 normal gravity field, not the true
  gravity field. That means it includes the latitude dependence of the reference
  ellipsoid but does NOT include local gravity anomalies from geology.
- The function `normal_gravity(...)` uses the standard truncated Taylor expansion
  above the ellipsoid. This is appropriate for low-altitude navigation simulation
  (surface, maritime, low-altitude air, modest UUV depth corrections expressed as
  height offsets with sign convention handled externally), but it is not intended
  for high-altitude orbital dynamics.
- Later repository modules should model local gravity anomaly separately as:
      g_total = g_normal + gravity_anomaly + sensor errors + motion-coupling residuals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
MPS2_TO_MGAL = 1.0e5
MGAL_TO_MPS2 = 1.0e-5


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array without copying unless needed."""
    return np.asarray(x, dtype=np.float64)


def _is_scalar_like(x: ArrayLike) -> bool:
    """Return True if x behaves like a scalar input."""
    arr = np.asarray(x)
    return arr.ndim == 0


def _maybe_scalar(result: FloatArray, *original_inputs: ArrayLike):
    """
    Return a Python float when all original inputs were scalar-like; otherwise
    return a NumPy array.

    This keeps the public API convenient for both scalar and vectorized use.
    """
    if all(_is_scalar_like(x) for x in original_inputs):
        return float(np.asarray(result))
    return np.asarray(result, dtype=np.float64)


def _maybe_scalar_tuple(
    result: Tuple[FloatArray, ...], *original_inputs: ArrayLike
):
    """
    Tuple version of `_maybe_scalar`.

    Returns a tuple of Python floats if all inputs were scalars; otherwise returns
    a tuple of NumPy arrays.
    """
    if all(_is_scalar_like(x) for x in original_inputs):
        return tuple(float(np.asarray(r)) for r in result)
    return tuple(np.asarray(r, dtype=np.float64) for r in result)


def mps2_to_mgal(g_mps2: ArrayLike):
    """
    Convert acceleration from m/s^2 to mGal.

    Formula
    -------
    1 m/s^2 = 10^5 mGal

    Reference
    ---------
    NGA.STND.0036_1.0.0_WGS84, Chapter 4 notes that gravity in SI units can be
    converted using:
        1 m/s^2 = 10^5 mGal
    """
    g = _as_float_array(g_mps2) * MPS2_TO_MGAL
    return _maybe_scalar(g, g_mps2)


def mgal_to_mps2(g_mgal: ArrayLike):
    """
    Convert acceleration from mGal to m/s^2.

    Formula
    -------
    1 mGal = 10^-5 m/s^2
    """
    g = _as_float_array(g_mgal) * MGAL_TO_MPS2
    return _maybe_scalar(g, g_mgal)


@dataclass(frozen=True)
class ReferenceEllipsoid:
    """
    Reference ellipsoid with the four WGS-84-style defining parameters.

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    inv_f : float
        Reciprocal flattening 1/f [-].
    gm : float
        Geocentric gravitational constant GM [m^3/s^2].
    omega : float
        Nominal mean angular velocity [rad/s].

    Background
    ----------
    WGS 84 is fully defined by four independent parameters:
    - semi-major axis a
    - reciprocal flattening 1/f
    - geocentric gravitational constant GM
    - nominal mean angular velocity omega

    This follows NGA.STND.0036_1.0.0_WGS84, Chapter 3 and Table 3.1.

    The associated normal gravity field is then computed from these parameters.
    In particular, the helper quantities below follow the standard geodetic
    construction summarized in the WGS 84 documentation and AHRS geodesy notes.

    References
    ----------
    1) NGA.STND.0036_1.0.0_WGS84, Table 3.1 and Chapter 4
    2) AHRS WGS84 notes:
       https://ahrs.readthedocs.io/en/latest/geodesy/wgs84.html
    """

    a: float
    inv_f: float
    gm: float
    omega: float

    @property
    def f(self) -> float:
        """Flattening f = 1 / inv_f."""
        return 1.0 / self.inv_f

    @property
    def b(self) -> float:
        """
        Semi-minor axis [m].

        Formula
        -------
        b = a * (1 - f)

        Reference
        ---------
        Standard ellipsoid geometry; directly implied by the WGS 84 defining
        parameters a and f.
        """
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        """
        First eccentricity squared.

        Formula
        -------
        e^2 = 1 - b^2 / a^2 = f * (2 - f)

        Reference
        ---------
        Standard ellipsoid geometry.
        """
        return self.f * (2.0 - self.f)

    @property
    def ep2(self) -> float:
        """
        Second eccentricity squared.

        Formula
        -------
        e'^2 = (a^2 - b^2) / b^2 = e^2 / (1 - e^2)

        Reference
        ---------
        Standard ellipsoid geometry.
        """
        return self.e2 / (1.0 - self.e2)

    @property
    def linear_eccentricity(self) -> float:
        """
        Linear eccentricity E [m].

        Formula
        -------
        E = sqrt(a^2 - b^2)

        Reference
        ---------
        AHRS WGS84 notes, geodetic potential setup, following standard
        physical geodesy notation.
        """
        return np.sqrt(self.a * self.a - self.b * self.b)

    @property
    def m(self) -> float:
        r"""
        Helper rotation parameter used in WGS 84 normal gravity.

        Formula
        -------
        m = omega^2 * a^2 * b / GM

        Reference
        ---------
        AHRS WGS84 notes, "Normal Gravity on the Surface", equation defining m.
        This is consistent with the WGS 84 normal-gravity formulation.
        """
        return (self.omega ** 2) * (self.a ** 2) * self.b / self.gm

    @property
    def q0(self) -> float:
        r"""
        Helper quantity q0 used in the exact WGS 84 surface normal gravity formulas.

        Formula
        -------
        Let E = sqrt(a^2 - b^2). Then:

        q0 = 0.5 * [ (1 + 3 b^2 / E^2) * arctan(E / b) - 3 b / E ]

        Reference
        ---------
        AHRS WGS84 notes, "Earth's Gravity Field".
        """
        E = self.linear_eccentricity
        return 0.5 * (
            (1.0 + 3.0 * (self.b ** 2) / (E ** 2)) * np.arctan(E / self.b)
            - 3.0 * self.b / E
        )

    @property
    def q0_prime(self) -> float:
        r"""
        Helper quantity q0' used in the exact WGS 84 surface normal gravity formulas.

        Formula
        -------
        Let e' = sqrt(ep2). Then:

        q0' = 3 * [ (1 + 1/e'^2) * (1 - arctan(e') / e') ] - 1

        Reference
        ---------
        AHRS WGS84 notes, "Earth's Gravity Field".
        """
        e_prime = np.sqrt(self.ep2)
        return 3.0 * (
            (1.0 + 1.0 / (e_prime ** 2))
            * (1.0 - np.arctan(e_prime) / e_prime)
        ) - 1.0

    @property
    def gamma_e(self) -> float:
        r"""
        Normal gravity at the equator [m/s^2].

        Formula
        -------
        gamma_e = GM / (a b) * [1 - m - (m e' q0') / (6 q0)]

        Reference
        ---------
        AHRS WGS84 notes, "Normal Gravity on the Surface", equation for g_e.
        Equivalent to the WGS 84 normal gravity construction.
        """
        e_prime = np.sqrt(self.ep2)
        return (self.gm / (self.a * self.b)) * (
            1.0 - self.m - (self.m * e_prime * self.q0_prime) / (6.0 * self.q0)
        )

    @property
    def gamma_p(self) -> float:
        r"""
        Normal gravity at the pole [m/s^2].

        Formula
        -------
        gamma_p = GM / a^2 * [1 + (m e' q0') / (3 q0)]

        Reference
        ---------
        AHRS WGS84 notes, "Normal Gravity on the Surface", equation for g_p.
        Equivalent to the WGS 84 normal gravity construction.
        """
        e_prime = np.sqrt(self.ep2)
        return (self.gm / (self.a ** 2)) * (
            1.0 + (self.m * e_prime * self.q0_prime) / (3.0 * self.q0)
        )

    @property
    def somigliana_k(self) -> float:
        r"""
        Somigliana helper constant k.

        Formula
        -------
        k = (b * gamma_p) / (a * gamma_e) - 1

        Reference
        ---------
        NGA.STND.0036_1.0.0_WGS84, Section 4.2 and standard Somigliana form.
        """
        return (self.b * self.gamma_p) / (self.a * self.gamma_e) - 1.0

    def prime_vertical_radius(self, lat_rad: ArrayLike):
        r"""
        Radius of curvature in the prime vertical, N(phi) [m].

        Formula
        -------
        N(phi) = a / sqrt(1 - e^2 sin^2(phi))

        This is the standard prime-vertical radius used in geodetic <-> ECEF
        conversion and local Earth geometry.

        Parameters
        ----------
        lat_rad : ArrayLike
            Geodetic latitude [rad].

        Reference
        ---------
        Clynch (2002), "Geodetic Coordinate Conversions", definition of RN.
        """
        lat = _as_float_array(lat_rad)
        s = np.sin(lat)
        N = self.a / np.sqrt(1.0 - self.e2 * s * s)
        return _maybe_scalar(N, lat_rad)

    def meridian_radius(self, lat_rad: ArrayLike):
        r"""
        Meridian radius of curvature, M(phi) [m].

        Formula
        -------
        M(phi) = a (1 - e^2) / (1 - e^2 sin^2(phi))^(3/2)

        This is the north-south radius of curvature of the ellipsoid and is
        needed for local navigation kinematics and latitude propagation.

        Parameters
        ----------
        lat_rad : ArrayLike
            Geodetic latitude [rad].

        Reference
        ---------
        Standard ellipsoid differential geometry; consistent with geodesy texts
        and the Naval Postgraduate School geodetic notes.
        """
        lat = _as_float_array(lat_rad)
        s = np.sin(lat)
        denom = (1.0 - self.e2 * s * s) ** 1.5
        M = self.a * (1.0 - self.e2) / denom
        return _maybe_scalar(M, lat_rad)

    def geocentric_radius(self, lat_rad: ArrayLike):
        r"""
        Geocentric radius of the ellipsoid surface at geodetic latitude [m].

        Formula
        -------
        For a point on the reference ellipsoid (h = 0), the distance from the
        Earth center to the surface point is:

            r(phi) = sqrt(
                ((a^2 cos(phi))^2 + (b^2 sin(phi))^2) /
                ((a cos(phi))^2 + (b sin(phi))^2)
            )

        This is useful for diagnostics and validation, but note that navigation
        mechanization should use N(phi) and M(phi), not this radius.

        Reference
        ---------
        Standard ellipsoid geometry.
        """
        lat = _as_float_array(lat_rad)
        c = np.cos(lat)
        s = np.sin(lat)

        num = (self.a * self.a * c) ** 2 + (self.b * self.b * s) ** 2
        den = (self.a * c) ** 2 + (self.b * s) ** 2
        r = np.sqrt(num / den)
        return _maybe_scalar(r, lat_rad)

    def normal_gravity_surface(self, lat_rad: ArrayLike):
        r"""
        Normal gravity on the reference ellipsoid surface [m/s^2].

        Formula
        -------
        Somigliana's formula:

            gamma(phi) = gamma_e * (1 + k sin^2(phi)) / sqrt(1 - e^2 sin^2(phi))

        where:
            k = (b gamma_p) / (a gamma_e) - 1

        Parameters
        ----------
        lat_rad : ArrayLike
            Geodetic latitude [rad].

        Returns
        -------
        float or np.ndarray
            Surface normal gravity [m/s^2].

        Reference
        ---------
        NGA.STND.0036_1.0.0_WGS84, Section 4.2 (Somigliana formula);
        AHRS WGS84 notes provide a readable equivalent form.
        """
        lat = _as_float_array(lat_rad)
        s2 = np.sin(lat) ** 2
        gamma = self.gamma_e * (1.0 + self.somigliana_k * s2) / np.sqrt(
            1.0 - self.e2 * s2
        )
        return _maybe_scalar(gamma, lat_rad)

    def normal_gravity(self, lat_rad: ArrayLike, height_m: ArrayLike = 0.0):
        r"""
        Normal gravity above the ellipsoid using the WGS 84 truncated Taylor expansion [m/s^2].

        Formula
        -------
        For small ellipsoidal heights h above the reference ellipsoid:

            gamma(phi, h) = gamma(phi) * [
                1
                - (2 / a) * (1 + f + m - 2 f sin^2(phi)) * h
                + (3 / a^2) * h^2
            ]

        where:
        - gamma(phi) is surface normal gravity from Somigliana's formula
        - f is ellipsoid flattening
        - m = omega^2 * a^2 * b / GM

        Parameters
        ----------
        lat_rad : ArrayLike
            Geodetic latitude [rad].
        height_m : ArrayLike, default=0.0
            Ellipsoidal height above the reference ellipsoid [m].

        Returns
        -------
        float or np.ndarray
            Normal gravity [m/s^2].

        Important
        ---------
        This is the standard practical WGS 84 approximation for modest heights
        used in geodesy and navigation engineering. It should NOT be mistaken for
        the true gravity field or for a full spherical-harmonic Earth model.

        Reference
        ---------
        NGA.STND.0036_1.0.0_WGS84, Section 4.3, Eq. (4-3).
        The same formula is shown explicitly in the AHRS WGS84 notes.
        """
        lat, h = np.broadcast_arrays(_as_float_array(lat_rad), _as_float_array(height_m))
        gamma0 = _as_float_array(self.normal_gravity_surface(lat))
        s2 = np.sin(lat) ** 2

        correction = (
            1.0
            - (2.0 / self.a) * (1.0 + self.f + self.m - 2.0 * self.f * s2) * h
            + (3.0 / (self.a ** 2)) * (h ** 2)
        )
        gamma = gamma0 * correction
        return _maybe_scalar(gamma, lat_rad, height_m)

    def normal_gravity_vertical_gradient(self, lat_rad: ArrayLike):
        r"""
        Vertical derivative of WGS 84 normal gravity at h = 0 [m/s^2 per m].

        Derived from the linear term of the standard height Taylor expansion:

            d gamma / d h |_(h=0)
            = - gamma(phi) * (2 / a) * (1 + f + m - 2 f sin^2(phi))

        This is useful for quick-order sensitivity checks, height-to-gravity
        coupling estimates, and validating altitude/depth error budgets.

        Parameters
        ----------
        lat_rad : ArrayLike
            Geodetic latitude [rad].

        Returns
        -------
        float or np.ndarray
            Vertical gradient [m/s^2 / m]. This value is negative above the
            ellipsoid because gravity decreases with height.

        Reference
        ---------
        Direct derivative of the WGS 84 Eq. (4-3) truncated Taylor expression
        in NGA.STND.0036_1.0.0_WGS84 Section 4.3.
        """
        lat = _as_float_array(lat_rad)
        gamma0 = _as_float_array(self.normal_gravity_surface(lat))
        s2 = np.sin(lat) ** 2
        grad = -gamma0 * (2.0 / self.a) * (1.0 + self.f + self.m - 2.0 * self.f * s2)
        return _maybe_scalar(grad, lat_rad)

    def geodetic_to_ecef(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        height_m: ArrayLike,
    ):
        r"""
        Convert geodetic coordinates to ECEF coordinates.

        Formula
        -------
        Let N(phi) be the prime vertical radius of curvature:

            N(phi) = a / sqrt(1 - e^2 sin^2(phi))

        Then:

            x = (N + h) cos(phi) cos(lambda)
            y = (N + h) cos(phi) sin(lambda)
            z = ((1 - e^2) N + h) sin(phi)

        Parameters
        ----------
        lat_rad : ArrayLike
            Geodetic latitude [rad].
        lon_rad : ArrayLike
            Longitude [rad], east-positive.
        height_m : ArrayLike
            Ellipsoidal height [m].

        Returns
        -------
        tuple
            (x, y, z) in ECEF coordinates [m].

        Reference
        ---------
        Clynch (2002), "Geodetic Coordinate Conversions", section:
        "Latitude, Longitude and Height to/from ECEF (x,y,z)".
        """
        lat, lon, h = np.broadcast_arrays(
            _as_float_array(lat_rad),
            _as_float_array(lon_rad),
            _as_float_array(height_m),
        )

        N = _as_float_array(self.prime_vertical_radius(lat))
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        x = (N + h) * cos_lat * cos_lon
        y = (N + h) * cos_lat * sin_lon
        z = ((1.0 - self.e2) * N + h) * sin_lat
        return _maybe_scalar_tuple((x, y, z), lat_rad, lon_rad, height_m)

    def ecef_to_geodetic(
        self,
        x_m: ArrayLike,
        y_m: ArrayLike,
        z_m: ArrayLike,
        max_iter: int = 10,
        tol: float = 1e-13,
    ):
        r"""
        Convert ECEF coordinates to geodetic coordinates using a practical
        iterative scheme.

        Method
        ------
        This implementation follows the standard practical geodetic iteration
        described by Clynch (2002):

        1. p = sqrt(x^2 + y^2)
        2. lambda = atan2(y, x)
        3. initial latitude guess from geocentric latitude
        4. iterate:
               N = a / sqrt(1 - e^2 sin^2(phi))
               h = p / cos(phi) - N
               phi_next = atan2(z, p * (1 - e^2 * N / (N + h)))
        5. final h = p / cos(phi) - N

        Parameters
        ----------
        x_m, y_m, z_m : ArrayLike
            ECEF coordinates [m].
        max_iter : int, default=10
            Maximum number of fixed-point iterations.
        tol : float, default=1e-13
            Convergence threshold in radians on latitude update.

        Returns
        -------
        tuple
            (lat_rad, lon_rad, height_m)

        Special cases
        -------------
        For points on the polar axis (x ~= 0 and y ~= 0), the iteration is
        bypassed and the result is set analytically.

        Reference
        ---------
        Clynch (2002), "Geodetic Coordinate Conversions", section:
        "ECEF xyz to Latitude, Longitude, Height".
        """
        x_b, y_b, z_b = np.broadcast_arrays(
            _as_float_array(x_m),
            _as_float_array(y_m),
            _as_float_array(z_m),
        )
        shape = x_b.shape

        # Flatten for uniform masked assignment. This avoids 0-D scalar arrays,
        # which do not support boolean item assignment.
        x = np.asarray(x_b, dtype=np.float64).reshape(-1)
        y = np.asarray(y_b, dtype=np.float64).reshape(-1)
        z = np.asarray(z_b, dtype=np.float64).reshape(-1)

        lon = np.arctan2(y, x)
        p = np.hypot(x, y)

        lat = np.arctan2(z, p * (1.0 - self.e2))
        h = np.zeros_like(lat)

        pole_mask = p < 1e-12
        general_mask = ~pole_mask

        if np.any(general_mask):
            lat_g = lat[general_mask]
            p_g = p[general_mask]
            z_g = z[general_mask]

            for _ in range(max_iter):
                sin_lat = np.sin(lat_g)
                N = self.a / np.sqrt(1.0 - self.e2 * sin_lat * sin_lat)
                h_g = p_g / np.cos(lat_g) - N
                lat_next = np.arctan2(
                    z_g, p_g * (1.0 - self.e2 * N / (N + h_g))
                )

                if np.max(np.abs(lat_next - lat_g)) < tol:
                    lat_g = lat_next
                    break
                lat_g = lat_next

            sin_lat = np.sin(lat_g)
            N = self.a / np.sqrt(1.0 - self.e2 * sin_lat * sin_lat)
            h_g = p_g / np.cos(lat_g) - N

            lat[general_mask] = lat_g
            h[general_mask] = h_g

        if np.any(pole_mask):
            lat[pole_mask] = np.sign(z[pole_mask]) * (0.5 * np.pi)
            h[pole_mask] = np.abs(z[pole_mask]) - self.b
            lon[pole_mask] = 0.0

        return _maybe_scalar_tuple(
            (
                lat.reshape(shape),
                lon.reshape(shape),
                h.reshape(shape),
            ),
            x_m,
            y_m,
            z_m,
        )


WGS84 = ReferenceEllipsoid(
    a=6378137.0,
    inv_f=298.257223563,
    gm=3.986004418e14,
    omega=7.292115e-5,
)


def prime_vertical_radius(lat_rad: ArrayLike):
    """
    Convenience wrapper around WGS84.prime_vertical_radius(...).
    """
    return WGS84.prime_vertical_radius(lat_rad)


def meridian_radius(lat_rad: ArrayLike):
    """
    Convenience wrapper around WGS84.meridian_radius(...).
    """
    return WGS84.meridian_radius(lat_rad)


def geocentric_radius(lat_rad: ArrayLike):
    """
    Convenience wrapper around WGS84.geocentric_radius(...).
    """
    return WGS84.geocentric_radius(lat_rad)


def normal_gravity_surface(lat_rad: ArrayLike):
    """
    Convenience wrapper around WGS84.normal_gravity_surface(...).
    """
    return WGS84.normal_gravity_surface(lat_rad)


def normal_gravity(lat_rad: ArrayLike, height_m: ArrayLike = 0.0):
    """
    Convenience wrapper around WGS84.normal_gravity(...).
    """
    return WGS84.normal_gravity(lat_rad, height_m)


def normal_gravity_vertical_gradient(lat_rad: ArrayLike):
    """
    Convenience wrapper around WGS84.normal_gravity_vertical_gradient(...).
    """
    return WGS84.normal_gravity_vertical_gradient(lat_rad)


def geodetic_to_ecef(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
):
    """
    Convenience wrapper around WGS84.geodetic_to_ecef(...).
    """
    return WGS84.geodetic_to_ecef(lat_rad, lon_rad, height_m)


def ecef_to_geodetic(
    x_m: ArrayLike,
    y_m: ArrayLike,
    z_m: ArrayLike,
    max_iter: int = 10,
    tol: float = 1e-13,
):
    """
    Convenience wrapper around WGS84.ecef_to_geodetic(...).
    """
    return WGS84.ecef_to_geodetic(x_m, y_m, z_m, max_iter=max_iter, tol=tol)


__all__ = [
    "FloatArray",
    "MGAL_TO_MPS2",
    "MPS2_TO_MGAL",
    "ReferenceEllipsoid",
    "WGS84",
    "ecef_to_geodetic",
    "geocentric_radius",
    "geodetic_to_ecef",
    "mgal_to_mps2",
    "meridian_radius",
    "mps2_to_mgal",
    "normal_gravity",
    "normal_gravity_surface",
    "normal_gravity_vertical_gradient",
    "prime_vertical_radius",
]
