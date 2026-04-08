"""
units.py

Shared unit-conversion constants and helpers for the gravity-aided navigation
simulator.

This module is intentionally simple, dependency-light, and reusable across:
- gravimetry (`mGal`, `µGal`, `Gal`, `m/s²`)
- IMU datasheet conversions (`deg/s`, `deg/h`, `rad/s`, `g`, `mg`, `µg`)
- navigation geometry (`deg`, `arcmin`, `arcsec`, `rad`)
- maritime / aviation speed and distance (`knot`, `nautical mile`, `m/s`)
- scale-factor style quantities (`ppm`, `ppb`, `ppt`)

Why this file exists
--------------------
The repository already includes physics and sensor modules that either:
- directly use geodetic / gravimetric units, or
- will soon need to parse datasheet-style values from configuration files.

Rather than letting each module hard-code its own factors, this file provides
one canonical place for unit conversions. That keeps the codebase:
- consistent,
- testable,
- easier to review against published specs.

Scope
-----
This module is not a full symbolic unit system. It is a compact set of explicit
conversion helpers for the unit families that matter in this project.

Primary references used here
----------------------------
1) BIPM / NIST SI Brochure:
   The degree, minute, and second are non-SI units accepted for use with the SI,
   with:

       1 degree = (pi / 180) rad

   and minute / second as decimal subdivisions of the degree.

   Sources:
   - BIPM SI Brochure:
     https://www.bipm.org/en/publications/si-brochure/
   - NIST SP 330:
     https://www.nist.gov/pml/special-publication-330

2) NIST CODATA value for standard acceleration of gravity:
   The standard acceleration of gravity is exact:

       g_n = 9.80665 m s^-2

   Source:
   https://physics.nist.gov/cgi-bin/cuu/Value?gn=

3) NIST SP 330, Section 4:
   The gal is a non-SI unit of acceleration used in geodesy / geophysics:

       1 Gal = 1 cm s^-2 = 1e-2 m s^-2

   Source:
   https://www.nist.gov/pml/special-publication-330/sp-330-section-4

4) NIST Guide to the SI, footnotes:
   The international nautical mile is exact:

       1 nmi = 1852 m

   Therefore:

       1 knot = 1 nmi / h = 1852 / 3600 m s^-1

   Source:
   https://www.nist.gov/pml/special-publication-811/nist-guide-si-footnotes

5) SI prefixes:
   milli = 1e-3
   micro = 1e-6
   nano  = 1e-9
   pico  = 1e-12

   Sources:
   - BIPM SI Brochure
   - NIST Metric (SI) Prefixes:
     https://www.nist.gov/pml/owm/metric-si-prefixes

Design notes
------------
- Functions accept either scalars or NumPy-compatible arrays.
- Scalar inputs return floats; array inputs return NumPy arrays.
- Constants are named in an explicit "FROM_to_TO" style where appropriate.
- The helpers here stay purely numeric; they do not depend on other project
  modules, which avoids circular imports.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]

PI = float(np.pi)

SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
HOURS_PER_DAY = 24.0
SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR

DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

ARCMIN_TO_DEG = 1.0 / 60.0
DEG_TO_ARCMIN = 60.0

ARCSEC_TO_DEG = 1.0 / 3600.0
DEG_TO_ARCSEC = 3600.0

ARCMIN_TO_RAD = DEG_TO_RAD * ARCMIN_TO_DEG
RAD_TO_ARCMIN = RAD_TO_DEG * DEG_TO_ARCMIN

ARCSEC_TO_RAD = DEG_TO_RAD * ARCSEC_TO_DEG
RAD_TO_ARCSEC = RAD_TO_DEG * DEG_TO_ARCSEC

STANDARD_GRAVITY_MPS2 = 9.80665

GAL_TO_MPS2 = 1.0e-2
MPS2_TO_GAL = 1.0 / GAL_TO_MPS2

MGAL_TO_MPS2 = 1.0e-5
MPS2_TO_MGAL = 1.0 / MGAL_TO_MPS2

UGAL_TO_MPS2 = 1.0e-8
MPS2_TO_UGAL = 1.0 / UGAL_TO_MPS2

G_TO_MPS2 = STANDARD_GRAVITY_MPS2
MPS2_TO_G = 1.0 / G_TO_MPS2

MILLI_G_TO_MPS2 = 1.0e-3 * STANDARD_GRAVITY_MPS2
MPS2_TO_MILLI_G = 1.0 / MILLI_G_TO_MPS2

MICRO_G_TO_MPS2 = 1.0e-6 * STANDARD_GRAVITY_MPS2
MPS2_TO_MICRO_G = 1.0 / MICRO_G_TO_MPS2

NMI_TO_M = 1852.0
M_TO_NMI = 1.0 / NMI_TO_M

KNOT_TO_MPS = NMI_TO_M / SECONDS_PER_HOUR
MPS_TO_KNOT = 1.0 / KNOT_TO_MPS

KMH_TO_MPS = 1000.0 / SECONDS_PER_HOUR
MPS_TO_KMH = 1.0 / KMH_TO_MPS

PPM_TO_FRACTION = 1.0e-6
FRACTION_TO_PPM = 1.0 / PPM_TO_FRACTION

PPB_TO_FRACTION = 1.0e-9
FRACTION_TO_PPB = 1.0 / PPB_TO_FRACTION

PPT_TO_FRACTION = 1.0e-12
FRACTION_TO_PPT = 1.0 / PPT_TO_FRACTION

DEG_PER_SEC_TO_RAD_PER_SEC = DEG_TO_RAD
RAD_PER_SEC_TO_DEG_PER_SEC = RAD_TO_DEG

DEG_PER_HOUR_TO_RAD_PER_SEC = DEG_TO_RAD / SECONDS_PER_HOUR
RAD_PER_SEC_TO_DEG_PER_HOUR = SECONDS_PER_HOUR * RAD_TO_DEG

DEG_PER_SQRT_HOUR_TO_RAD_PER_SQRT_SEC = DEG_TO_RAD / np.sqrt(SECONDS_PER_HOUR)
RAD_PER_SQRT_SEC_TO_DEG_PER_SQRT_HOUR = np.sqrt(SECONDS_PER_HOUR) * RAD_TO_DEG

MGAL_PER_SQRT_HZ_TO_MPS2_PER_SQRT_HZ = MGAL_TO_MPS2
MPS2_PER_SQRT_HZ_TO_MGAL_PER_SQRT_HZ = MPS2_TO_MGAL

UGAL_PER_SQRT_HZ_TO_MPS2_PER_SQRT_HZ = UGAL_TO_MPS2
MPS2_PER_SQRT_HZ_TO_UGAL_PER_SQRT_HZ = MPS2_TO_UGAL


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _scalar_or_array(x: FloatArray):
    """
    Return a Python float for scalar inputs, else a NumPy array.
    """
    if x.ndim == 0:
        return float(x)
    return np.asarray(x, dtype=np.float64)


def _multiply(x: ArrayLike, factor: float):
    """Multiply scalar/array input by a factor while preserving scalar outputs."""
    arr = _as_float_array(x)
    return _scalar_or_array(arr * float(factor))


def deg_to_rad(angle_deg: ArrayLike):
    r"""
    Convert angle from degrees to radians.

    Formula
    -------
        rad = deg * (pi / 180)
    """
    return _multiply(angle_deg, DEG_TO_RAD)


def rad_to_deg(angle_rad: ArrayLike):
    r"""
    Convert angle from radians to degrees.

    Formula
    -------
        deg = rad * (180 / pi)
    """
    return _multiply(angle_rad, RAD_TO_DEG)


def arcmin_to_rad(angle_arcmin: ArrayLike):
    r"""
    Convert angle from arcminutes to radians.

    Formula
    -------
        rad = arcmin * (pi / (180 * 60))
    """
    return _multiply(angle_arcmin, ARCMIN_TO_RAD)


def rad_to_arcmin(angle_rad: ArrayLike):
    r"""
    Convert angle from radians to arcminutes.
    """
    return _multiply(angle_rad, RAD_TO_ARCMIN)


def arcsec_to_rad(angle_arcsec: ArrayLike):
    r"""
    Convert angle from arcseconds to radians.

    Formula
    -------
        rad = arcsec * (pi / (180 * 3600))
    """
    return _multiply(angle_arcsec, ARCSEC_TO_RAD)


def rad_to_arcsec(angle_rad: ArrayLike):
    r"""
    Convert angle from radians to arcseconds.
    """
    return _multiply(angle_rad, RAD_TO_ARCSEC)


def degps_to_radps(rate_degps: ArrayLike):
    r"""
    Convert angular rate from degrees per second to radians per second.
    """
    return _multiply(rate_degps, DEG_PER_SEC_TO_RAD_PER_SEC)


def radps_to_degps(rate_radps: ArrayLike):
    r"""
    Convert angular rate from radians per second to degrees per second.
    """
    return _multiply(rate_radps, RAD_PER_SEC_TO_DEG_PER_SEC)


def dph_to_radps(rate_dph: ArrayLike):
    r"""
    Convert angular rate from degrees per hour to radians per second.

    Formula
    -------
        rad/s = (deg/h) * (pi / 180) / 3600
    """
    return _multiply(rate_dph, DEG_PER_HOUR_TO_RAD_PER_SEC)


def radps_to_dph(rate_radps: ArrayLike):
    r"""
    Convert angular rate from radians per second to degrees per hour.
    """
    return _multiply(rate_radps, RAD_PER_SEC_TO_DEG_PER_HOUR)


def deg_per_sqrt_hour_to_rad_per_sqrt_sec(noise_deg_per_sqrt_hour: ArrayLike):
    r"""
    Convert noise density from degrees per sqrt(hour) to radians per sqrt(second).

    Formula
    -------
        rad/√s = (deg/√h) * (pi/180) / 60
    """
    return _multiply(noise_deg_per_sqrt_hour, DEG_PER_SQRT_HOUR_TO_RAD_PER_SQRT_SEC)


def rad_per_sqrt_sec_to_deg_per_sqrt_hour(noise_rad_per_sqrt_sec: ArrayLike):
    r"""
    Convert noise density from radians per sqrt(second) to degrees per sqrt(hour).
    """
    return _multiply(noise_rad_per_sqrt_sec, RAD_PER_SQRT_SEC_TO_DEG_PER_SQRT_HOUR)


def gal_to_mps2(accel_gal: ArrayLike):
    r"""
    Convert acceleration from Gal to m/s².

    Formula
    -------
        1 Gal = 1 cm/s² = 1e-2 m/s²
    """
    return _multiply(accel_gal, GAL_TO_MPS2)


def mps2_to_gal(accel_mps2: ArrayLike):
    r"""
    Convert acceleration from m/s² to Gal.
    """
    return _multiply(accel_mps2, MPS2_TO_GAL)


def mgal_to_mps2(accel_mgal: ArrayLike):
    r"""
    Convert acceleration from mGal to m/s².

    Formula
    -------
        1 mGal = 1e-5 m/s²
    """
    return _multiply(accel_mgal, MGAL_TO_MPS2)


def mps2_to_mgal(accel_mps2: ArrayLike):
    r"""
    Convert acceleration from m/s² to mGal.
    """
    return _multiply(accel_mps2, MPS2_TO_MGAL)


def ugal_to_mps2(accel_ugal: ArrayLike):
    r"""
    Convert acceleration from µGal to m/s².

    Formula
    -------
        1 µGal = 1e-8 m/s²
    """
    return _multiply(accel_ugal, UGAL_TO_MPS2)


def mps2_to_ugal(accel_mps2: ArrayLike):
    r"""
    Convert acceleration from m/s² to µGal.
    """
    return _multiply(accel_mps2, MPS2_TO_UGAL)


def g_to_mps2(accel_g: ArrayLike):
    r"""
    Convert acceleration from standard gravities `g` to m/s².

    Formula
    -------
        1 g_n = 9.80665 m/s²
    """
    return _multiply(accel_g, G_TO_MPS2)


def mps2_to_g(accel_mps2: ArrayLike):
    r"""
    Convert acceleration from m/s² to standard gravities `g`.
    """
    return _multiply(accel_mps2, MPS2_TO_G)


def milli_g_to_mps2(accel_mg: ArrayLike):
    r"""
    Convert acceleration from milli-g to m/s².
    """
    return _multiply(accel_mg, MILLI_G_TO_MPS2)


def mps2_to_milli_g(accel_mps2: ArrayLike):
    r"""
    Convert acceleration from m/s² to milli-g.
    """
    return _multiply(accel_mps2, MPS2_TO_MILLI_G)


def micro_g_to_mps2(accel_ug: ArrayLike):
    r"""
    Convert acceleration from micro-g to m/s².
    """
    return _multiply(accel_ug, MICRO_G_TO_MPS2)


def mps2_to_micro_g(accel_mps2: ArrayLike):
    r"""
    Convert acceleration from m/s² to micro-g.
    """
    return _multiply(accel_mps2, MPS2_TO_MICRO_G)


def mgal_per_sqrt_hz_to_mps2_per_sqrt_hz(noise_mgal_per_sqrt_hz: ArrayLike):
    r"""
    Convert noise density from mGal/√Hz to m/s²/√Hz.
    """
    return _multiply(noise_mgal_per_sqrt_hz, MGAL_PER_SQRT_HZ_TO_MPS2_PER_SQRT_HZ)


def mps2_per_sqrt_hz_to_mgal_per_sqrt_hz(noise_mps2_per_sqrt_hz: ArrayLike):
    r"""
    Convert noise density from m/s²/√Hz to mGal/√Hz.
    """
    return _multiply(noise_mps2_per_sqrt_hz, MPS2_PER_SQRT_HZ_TO_MGAL_PER_SQRT_HZ)


def ugal_per_sqrt_hz_to_mps2_per_sqrt_hz(noise_ugal_per_sqrt_hz: ArrayLike):
    r"""
    Convert noise density from µGal/√Hz to m/s²/√Hz.
    """
    return _multiply(noise_ugal_per_sqrt_hz, UGAL_PER_SQRT_HZ_TO_MPS2_PER_SQRT_HZ)


def mps2_per_sqrt_hz_to_ugal_per_sqrt_hz(noise_mps2_per_sqrt_hz: ArrayLike):
    r"""
    Convert noise density from m/s²/√Hz to µGal/√Hz.
    """
    return _multiply(noise_mps2_per_sqrt_hz, MPS2_PER_SQRT_HZ_TO_UGAL_PER_SQRT_HZ)


def nautical_mile_to_m(length_nmi: ArrayLike):
    r"""
    Convert distance from nautical miles to metres.

    Formula
    -------
        1 nmi = 1852 m
    """
    return _multiply(length_nmi, NMI_TO_M)


def m_to_nautical_mile(length_m: ArrayLike):
    r"""
    Convert distance from metres to nautical miles.
    """
    return _multiply(length_m, M_TO_NMI)


def knot_to_mps(speed_knot: ArrayLike):
    r"""
    Convert speed from knots to m/s.

    Formula
    -------
        1 knot = 1852 / 3600 m/s
    """
    return _multiply(speed_knot, KNOT_TO_MPS)


def mps_to_knot(speed_mps: ArrayLike):
    r"""
    Convert speed from m/s to knots.
    """
    return _multiply(speed_mps, MPS_TO_KNOT)


def kmh_to_mps(speed_kmh: ArrayLike):
    r"""
    Convert speed from km/h to m/s.

    Formula
    -------
        1 km/h = 1000 / 3600 m/s
    """
    return _multiply(speed_kmh, KMH_TO_MPS)


def mps_to_kmh(speed_mps: ArrayLike):
    r"""
    Convert speed from m/s to km/h.
    """
    return _multiply(speed_mps, MPS_TO_KMH)


def ppm_to_fraction(value_ppm: ArrayLike):
    r"""
    Convert parts per million to a dimensionless fraction.
    """
    return _multiply(value_ppm, PPM_TO_FRACTION)


def fraction_to_ppm(value_fraction: ArrayLike):
    r"""
    Convert a dimensionless fraction to ppm.
    """
    return _multiply(value_fraction, FRACTION_TO_PPM)


def ppb_to_fraction(value_ppb: ArrayLike):
    r"""
    Convert parts per billion to a dimensionless fraction.
    """
    return _multiply(value_ppb, PPB_TO_FRACTION)


def fraction_to_ppb(value_fraction: ArrayLike):
    r"""
    Convert a dimensionless fraction to ppb.
    """
    return _multiply(value_fraction, FRACTION_TO_PPB)


def ppt_to_fraction(value_ppt: ArrayLike):
    r"""
    Convert parts per trillion to a dimensionless fraction.
    """
    return _multiply(value_ppt, PPT_TO_FRACTION)


def fraction_to_ppt(value_fraction: ArrayLike):
    r"""
    Convert a dimensionless fraction to ppt.
    """
    return _multiply(value_fraction, FRACTION_TO_PPT)


__all__ = [
    "FloatArray",
    "PI",
    "SECONDS_PER_MINUTE",
    "MINUTES_PER_HOUR",
    "SECONDS_PER_HOUR",
    "HOURS_PER_DAY",
    "SECONDS_PER_DAY",
    "DEG_TO_RAD",
    "RAD_TO_DEG",
    "ARCMIN_TO_DEG",
    "DEG_TO_ARCMIN",
    "ARCSEC_TO_DEG",
    "DEG_TO_ARCSEC",
    "ARCMIN_TO_RAD",
    "RAD_TO_ARCMIN",
    "ARCSEC_TO_RAD",
    "RAD_TO_ARCSEC",
    "STANDARD_GRAVITY_MPS2",
    "GAL_TO_MPS2",
    "MPS2_TO_GAL",
    "MGAL_TO_MPS2",
    "MPS2_TO_MGAL",
    "UGAL_TO_MPS2",
    "MPS2_TO_UGAL",
    "G_TO_MPS2",
    "MPS2_TO_G",
    "MILLI_G_TO_MPS2",
    "MPS2_TO_MILLI_G",
    "MICRO_G_TO_MPS2",
    "MPS2_TO_MICRO_G",
    "NMI_TO_M",
    "M_TO_NMI",
    "KNOT_TO_MPS",
    "MPS_TO_KNOT",
    "KMH_TO_MPS",
    "MPS_TO_KMH",
    "PPM_TO_FRACTION",
    "FRACTION_TO_PPM",
    "PPB_TO_FRACTION",
    "FRACTION_TO_PPB",
    "PPT_TO_FRACTION",
    "FRACTION_TO_PPT",
    "DEG_PER_SEC_TO_RAD_PER_SEC",
    "RAD_PER_SEC_TO_DEG_PER_SEC",
    "DEG_PER_HOUR_TO_RAD_PER_SEC",
    "RAD_PER_SEC_TO_DEG_PER_HOUR",
    "DEG_PER_SQRT_HOUR_TO_RAD_PER_SQRT_SEC",
    "RAD_PER_SQRT_SEC_TO_DEG_PER_SQRT_HOUR",
    "MGAL_PER_SQRT_HZ_TO_MPS2_PER_SQRT_HZ",
    "MPS2_PER_SQRT_HZ_TO_MGAL_PER_SQRT_HZ",
    "UGAL_PER_SQRT_HZ_TO_MPS2_PER_SQRT_HZ",
    "MPS2_PER_SQRT_HZ_TO_UGAL_PER_SQRT_HZ",
    "arcmin_to_rad",
    "arcsec_to_rad",
    "deg_per_sqrt_hour_to_rad_per_sqrt_sec",
    "deg_to_rad",
    "degps_to_radps",
    "dph_to_radps",
    "fraction_to_ppb",
    "fraction_to_ppm",
    "fraction_to_ppt",
    "g_to_mps2",
    "gal_to_mps2",
    "kmh_to_mps",
    "knot_to_mps",
    "m_to_nautical_mile",
    "micro_g_to_mps2",
    "mgal_per_sqrt_hz_to_mps2_per_sqrt_hz",
    "mgal_to_mps2",
    "milli_g_to_mps2",
    "mps2_per_sqrt_hz_to_mgal_per_sqrt_hz",
    "mps2_per_sqrt_hz_to_ugal_per_sqrt_hz",
    "mps2_to_g",
    "mps2_to_gal",
    "mps2_to_mgal",
    "mps2_to_micro_g",
    "mps2_to_milli_g",
    "mps2_to_ugal",
    "mps_to_kmh",
    "mps_to_knot",
    "nautical_mile_to_m",
    "ppb_to_fraction",
    "ppm_to_fraction",
    "ppt_to_fraction",
    "rad_per_sqrt_sec_to_deg_per_sqrt_hour",
    "rad_to_arcmin",
    "rad_to_arcsec",
    "rad_to_deg",
    "radps_to_degps",
    "radps_to_dph",
    "ugal_per_sqrt_hz_to_mps2_per_sqrt_hz",
    "ugal_to_mps2",
]