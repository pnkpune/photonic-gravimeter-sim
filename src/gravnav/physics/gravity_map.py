"""
gravity_map.py

Regular-grid gravity-disturbance map models and synthetic-map generation helpers
for the gravity-aided navigation simulator.

Why this file exists
--------------------
The repository already contains:
- WGS 84 geodesy and local-frame math
- scalar gravimeter models whose navigation-facing observation is the
  measurement-level gravity disturbance
- a particle-filter map matcher that is intentionally written against a
  gravity-map *interface* rather than a concrete backend

What is still missing is the concrete gravity-map layer itself. This file fills
that gap with five pieces:

1) a regular lat/lon grid map container for scalar gravity disturbance
2) bilinear and nearest-neighbour interpolation on rectilinear grids
3) optional reference-height and vertical-gradient handling
4) synthetic map generation from analytic anomaly components
5) lightweight NPZ persistence helpers for map assets

Conventions
-----------
- Internal angles are radians.
- Latitude is geodetic latitude [rad].
- Longitude is east-positive [rad].
- Height is ellipsoidal and positive upward [m].
- Gravity-map values are scalar gravity disturbance values evaluated at the
  observation point [m/s^2], consistent with ``gravnav.sensors.gravimeter``.
- Disturbance values may be reported in mGal for debugging, but are stored in SI.

Map-height convention
---------------------
A regular grid in this file stores disturbance at a *reference height*
``h_ref``. Queries at a different height ``h`` are handled through the optional
vertical disturbance gradient:

    delta_g(lat, lon, h)
        = delta_g_ref(lat, lon)
        + d(delta_g)/dh * (h - h_ref)

This is intentionally modest. It does **not** claim to perform full geophysical
upward/downward continuation. It simply gives the simulator a clean way to:
- represent a survey map made at one height
- query it at nearby heights
- later replace the simplistic vertical model with a better correction layer

Interpolation reference
-----------------------
For linear interpolation on a rectilinear 2D grid, this file uses standard
bilinear interpolation:

    f(x, y) =
        (1-tx)(1-ty) f11 + tx(1-ty) f21 + (1-tx)ty f12 + tx ty f22

where ``tx`` and ``ty`` are the normalized local coordinates in the enclosing
cell. This is the same repeated-linear-interpolation construction described in
standard bilinear interpolation references.

Primary references used here
----------------------------
1) Hackney, R. I., and Featherstone, W. E. (2003),
   "Geodetic versus geophysical perspectives of the 'gravity anomaly'"
   Geophysical Journal International, 154(1), 35-43.
   URL:
   https://academic.oup.com/gji/article/154/1/35/604237

   Used for:
   - the distinction between gravity anomaly and gravity disturbance
   - the same-point scalar gravity disturbance convention used throughout this
     repository and therefore by the map values in this file

2) Harmonica / Fatiando a Terra documentation, "Gravity Disturbance"
   URL:
   https://www.fatiando.org/harmonica/latest/user_guide/gravity_disturbance.html

   Used as a concise statement of the same-point disturbance concept:
       delta_g(P) = g(P) - gamma(P)

3) Bilinear interpolation reference (repeated linear interpolation form)
   URL:
   https://en.wikipedia.org/wiki/Bilinear_interpolation

   Used for:
   - the explicit bilinear interpolation formula implemented here
   - the reminder that the method is naturally suited to values sampled on a
     2D rectilinear grid

4) SciPy documentation for ``RegularGridInterpolator``
   URL:
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

   Used for the practical rectilinear-grid framing of the interpolation problem.
   This file does **not** depend on SciPy; it implements the minimal needed
   interpolation itself using NumPy so the repository keeps its current light
   dependency footprint.

Design notes
------------
- The default concrete backend here is a regular lat/lon grid because it is a
  practical fit for synthetic-map generation, simple persistence, and the later
  particle filter.
- The interface deliberately exposes several aliases:
      sample_disturbance(...)
      interpolate_disturbance(...)
      evaluate_disturbance(...)
      lookup_disturbance(...)
      disturbance(...)
      __call__(...)
  so that ``estimators.map_match_pf`` can work with it immediately.
- The synthetic-map layer is intentionally transparent rather than pretending to
  be a geologically faithful regional model. It is meant to generate structured,
  controllable test maps for estimator development and Monte Carlo work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .earth import meridian_radius, mgal_to_mps2, mps2_to_mgal, prime_vertical_radius
from .frames import wrap_angle_pi

FloatArray = NDArray[np.float64]


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


def _vec1_strictly_increasing(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a one-dimensional strictly increasing coordinate vector.
    """
    arr = _as_float_array(x).reshape(-1)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least 2 values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def _grid2(x: ArrayLike, *, shape: tuple[int, int], name: str) -> FloatArray:
    """
    Validate and return a 2D grid with the expected shape.
    """
    arr = _as_float_array(x)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _scalar_or_grid(
    x: ArrayLike | float,
    *,
    shape: tuple[int, int],
    name: str,
) -> float | FloatArray:
    """
    Accept either a scalar or a 2D grid matching ``shape``.
    """
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return float(arr)
    return _grid2(arr, shape=shape, name=name)


def _broadcast_query_inputs(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, tuple[int, ...], bool]:
    """
    Broadcast geodetic query inputs to a common shape.

    Returns
    -------
    lat, lon, h, shape, scalar_output
    """
    lat = _as_float_array(lat_rad)
    lon = _as_float_array(lon_rad)
    h = _as_float_array(height_m)
    lat_b, lon_b, h_b = np.broadcast_arrays(lat, lon, h)
    scalar_output = lat_b.ndim == 0
    shape = lat_b.shape
    return (
        np.asarray(lat_b, dtype=np.float64),
        np.asarray(lon_b, dtype=np.float64),
        np.asarray(h_b, dtype=np.float64),
        shape,
        scalar_output,
    )


def _restore_shape(x: FloatArray, shape: tuple[int, ...], *, scalar_output: bool):
    """
    Restore either scalar or array output form after flattened computation.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(shape)
    if scalar_output:
        return float(arr)
    return arr


# -----------------------------------------------------------------------------
# Interpolation helpers
# -----------------------------------------------------------------------------


def _in_bounds_mask(
    x_axis: FloatArray,
    y_axis: FloatArray,
    xq: FloatArray,
    yq: FloatArray,
) -> NDArray[np.bool_]:
    """
    Return a boolean mask of query points lying inside the closed grid bounds.
    """
    return (
        (xq >= x_axis[0])
        & (xq <= x_axis[-1])
        & (yq >= y_axis[0])
        & (yq <= y_axis[-1])
    )


def bilinear_interpolate_rectilinear(
    x_axis: ArrayLike,
    y_axis: ArrayLike,
    values: ArrayLike,
    xq: ArrayLike,
    yq: ArrayLike,
    *,
    bounds_error: bool = True,
    fill_value: float = np.nan,
    method: str = "linear",
) -> FloatArray:
    r"""
    Interpolate a scalar field tabulated on a 2D rectilinear grid.

    Parameters
    ----------
    x_axis : array-like, shape (Nx,)
        First coordinate axis, strictly increasing.
    y_axis : array-like, shape (Ny,)
        Second coordinate axis, strictly increasing.
    values : array-like, shape (Nx, Ny)
        Grid values on ``(x_axis, y_axis)``.
    xq : array-like
        Query coordinate(s) along the first axis.
    yq : array-like
        Query coordinate(s) along the second axis.
    bounds_error : bool, default=True
        If True, raise when any query lies outside the grid extent.
    fill_value : float, default=np.nan
        Value used for out-of-bounds queries when ``bounds_error=False``.
    method : {"linear", "nearest"}, default="linear"
        Interpolation method.

    Returns
    -------
    np.ndarray
        Interpolated values with the broadcasted query shape.

    Formula
    -------
    For ``method='linear'``, the interpolant inside one cell is the standard
    bilinear form:

        f(x, y) =
            (1-tx)(1-ty) f11 + tx(1-ty) f21 + (1-tx)ty f12 + tx ty f22

    where:

        tx = (x - x1) / (x2 - x1)
        ty = (y - y1) / (y2 - y1)

    For ``method='nearest'``, the nearest sample in the rectilinear grid is
    returned independently along each axis.
    """
    x = _vec1_strictly_increasing(x_axis, name="x_axis")
    y = _vec1_strictly_increasing(y_axis, name="y_axis")
    v = _grid2(values, shape=(x.size, y.size), name="values")

    xq_arr = _as_float_array(xq)
    yq_arr = _as_float_array(yq)
    xq_b, yq_b = np.broadcast_arrays(xq_arr, yq_arr)
    out = np.full(xq_b.shape, float(fill_value), dtype=np.float64)

    mask = _in_bounds_mask(x, y, xq_b, yq_b)
    if bounds_error and not np.all(mask):
        raise ValueError("One or more query points lie outside the grid bounds.")

    if not np.any(mask):
        return out

    method_norm = str(method).strip().lower()
    if method_norm not in {"linear", "nearest"}:
        raise ValueError(f"Unsupported interpolation method {method!r}.")

    x_valid = xq_b[mask]
    y_valid = yq_b[mask]

    if method_norm == "nearest":
        ix_r = np.searchsorted(x, x_valid, side="left")
        ix_l = np.clip(ix_r - 1, 0, x.size - 1)
        ix_r = np.clip(ix_r, 0, x.size - 1)
        choose_r = np.abs(x[ix_r] - x_valid) < np.abs(x_valid - x[ix_l])
        ix = np.where(choose_r, ix_r, ix_l)

        iy_r = np.searchsorted(y, y_valid, side="left")
        iy_l = np.clip(iy_r - 1, 0, y.size - 1)
        iy_r = np.clip(iy_r, 0, y.size - 1)
        choose_r = np.abs(y[iy_r] - y_valid) < np.abs(y_valid - y[iy_l])
        iy = np.where(choose_r, iy_r, iy_l)

        out[mask] = v[ix, iy]
        return out

    ix = np.searchsorted(x, x_valid, side="right") - 1
    iy = np.searchsorted(y, y_valid, side="right") - 1
    ix = np.clip(ix, 0, x.size - 2)
    iy = np.clip(iy, 0, y.size - 2)

    x1 = x[ix]
    x2 = x[ix + 1]
    y1 = y[iy]
    y2 = y[iy + 1]

    tx = (x_valid - x1) / (x2 - x1)
    ty = (y_valid - y1) / (y2 - y1)

    f11 = v[ix, iy]
    f21 = v[ix + 1, iy]
    f12 = v[ix, iy + 1]
    f22 = v[ix + 1, iy + 1]

    out[mask] = (
        (1.0 - tx) * (1.0 - ty) * f11
        + tx * (1.0 - ty) * f21
        + (1.0 - tx) * ty * f12
        + tx * ty * f22
    )
    return out


# -----------------------------------------------------------------------------
# Local geodetic/metric helpers
# -----------------------------------------------------------------------------


def local_north_east_offsets_from_reference(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    *,
    lat_ref_rad: float,
    lon_ref_rad: float,
    height_ref_m: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    r"""
    Convert geodetic offsets near a reference point into local North/East offsets.

    Parameters
    ----------
    lat_rad : array-like
        Query latitude(s) [rad].
    lon_rad : array-like
        Query longitude(s) [rad].
    lat_ref_rad : float
        Reference latitude [rad].
    lon_ref_rad : float
        Reference longitude [rad].
    height_ref_m : float, default=0.0
        Reference ellipsoidal height [m].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        North and East offsets [m].

    Formula
    -------
    Using the standard small-area local-level metric approximation:

        dN ≈ (R_M + h) * dphi
        dE ≈ (R_N + h) cos(phi) * dlambda

    where ``R_M`` is the meridian radius and ``R_N`` the prime-vertical radius
    at the reference latitude.

    Notes
    -----
    This is the same curvilinear small-angle approximation underlying the local
    NED position-rate relations used elsewhere in the repository. It is accurate
    enough for synthetic local map patches and estimator-scale map matching.
    """
    lat = _as_float_array(lat_rad)
    lon = _as_float_array(lon_rad)
    lat_b, lon_b = np.broadcast_arrays(lat, lon)

    phi0 = float(lat_ref_rad)
    lam0 = float(lon_ref_rad)
    h0 = float(height_ref_m)
    rm = float(meridian_radius(phi0)) + h0
    rn = float(prime_vertical_radius(phi0)) + h0

    dphi = lat_b - phi0
    dlam = wrap_angle_pi(lon_b - lam0)

    d_north = rm * dphi
    d_east = rn * np.cos(phi0) * dlam
    return (
        np.asarray(d_north, dtype=np.float64),
        np.asarray(d_east, dtype=np.float64),
    )


def geodetic_axes_from_local_extent(
    *,
    lat_ref_rad: float,
    lon_ref_rad: float,
    north_min_m: float,
    north_max_m: float,
    east_min_m: float,
    east_max_m: float,
    num_lat: int,
    num_lon: int,
    height_ref_m: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    r"""
    Build regular latitude/longitude axes for a local map patch specified in metres.

    Parameters
    ----------
    lat_ref_rad, lon_ref_rad : float
        Patch centre/reference geodetic coordinates [rad].
    north_min_m, north_max_m : float
        North offset limits of the patch [m].
    east_min_m, east_max_m : float
        East offset limits of the patch [m].
    num_lat, num_lon : int
        Number of samples along latitude and longitude.
    height_ref_m : float, default=0.0
        Reference height used in the local metric conversion [m].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lat_axis_rad, lon_axis_rad)``.

    Formula
    -------
    Inverting the local metric relations gives:

        phi    ≈ phi0 + dN / (R_M + h0)
        lambda ≈ lambda0 + dE / ((R_N + h0) cos(phi0))
    """
    num_lat = int(num_lat)
    num_lon = int(num_lon)
    if num_lat < 2 or num_lon < 2:
        raise ValueError("num_lat and num_lon must both be at least 2.")

    phi0 = float(lat_ref_rad)
    lam0 = float(lon_ref_rad)
    h0 = float(height_ref_m)
    rm = float(meridian_radius(phi0)) + h0
    rn = float(prime_vertical_radius(phi0)) + h0
    cos_phi0 = float(np.cos(phi0))
    if np.isclose(cos_phi0, 0.0):
        raise ValueError(
            "geodetic_axes_from_local_extent is not well-conditioned exactly at the pole."
        )

    north_axis_m = np.linspace(float(north_min_m), float(north_max_m), num_lat)
    east_axis_m = np.linspace(float(east_min_m), float(east_max_m), num_lon)

    lat_axis = phi0 + north_axis_m / rm
    lon_axis = lam0 + east_axis_m / (rn * cos_phi0)
    return (
        np.asarray(lat_axis, dtype=np.float64),
        np.asarray(lon_axis, dtype=np.float64),
    )


# -----------------------------------------------------------------------------
# Synthetic anomaly components
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianAnomalySource:
    r"""
    Elliptical 2D Gaussian disturbance component defined in the local North/East plane.

    Parameters
    ----------
    center_lat_rad : float
        Geodetic latitude of the anomaly centre [rad].
    center_lon_rad : float
        Geodetic longitude of the anomaly centre [rad].
    amplitude_mps2 : float
        Peak disturbance amplitude [m/s^2]. Positive and negative values are both
        allowed.
    sigma_north_m : float
        1-sigma spread along the anomaly's local major axis when ``heading_rad=0`` [m].
    sigma_east_m : float
        1-sigma spread along the local cross axis when ``heading_rad=0`` [m].
    heading_rad : float, default=0.0
        Clockwise rotation of the anomaly's ``sigma_north_m`` axis from North
        toward East [rad].
    reference_height_m : float, default=0.0
        Reference height used for the local metric linearization [m].

    Formula
    -------
    After rotating the local North/East offsets into anomaly axes ``u`` and ``v``:

        g(u, v) = A * exp(-0.5 * ((u/sigma_u)^2 + (v/sigma_v)^2))

    Notes
    -----
    This is not meant to be a geophysical source inversion model. It is a clean,
    controllable building block for synthetic maps whose spatial scales and peak
    amplitudes can be tuned explicitly.
    """

    center_lat_rad: float
    center_lon_rad: float
    amplitude_mps2: float
    sigma_north_m: float
    sigma_east_m: float
    heading_rad: float = 0.0
    reference_height_m: float = 0.0

    def __post_init__(self) -> None:
        if float(self.sigma_north_m) <= 0.0:
            raise ValueError("sigma_north_m must be positive.")
        if float(self.sigma_east_m) <= 0.0:
            raise ValueError("sigma_east_m must be positive.")

    @classmethod
    def from_mgal(
        cls,
        *,
        center_lat_rad: float,
        center_lon_rad: float,
        amplitude_mgal: float,
        sigma_north_m: float,
        sigma_east_m: float,
        heading_rad: float = 0.0,
        reference_height_m: float = 0.0,
    ) -> "GaussianAnomalySource":
        """Convenience constructor using mGal amplitude input."""
        return cls(
            center_lat_rad=float(center_lat_rad),
            center_lon_rad=float(center_lon_rad),
            amplitude_mps2=float(mgal_to_mps2(amplitude_mgal)),
            sigma_north_m=float(sigma_north_m),
            sigma_east_m=float(sigma_east_m),
            heading_rad=float(heading_rad),
            reference_height_m=float(reference_height_m),
        )

    @property
    def amplitude_mgal(self) -> float:
        """Peak amplitude in mGal."""
        return float(mps2_to_mgal(self.amplitude_mps2))

    def evaluate(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
    ) -> FloatArray:
        """
        Evaluate the Gaussian anomaly at one or more geodetic coordinates.
        """
        d_n, d_e = local_north_east_offsets_from_reference(
            lat_rad,
            lon_rad,
            lat_ref_rad=float(self.center_lat_rad),
            lon_ref_rad=float(self.center_lon_rad),
            height_ref_m=float(self.reference_height_m),
        )

        c = float(np.cos(self.heading_rad))
        s = float(np.sin(self.heading_rad))
        u = c * d_n + s * d_e
        v = -s * d_n + c * d_e

        exponent = -0.5 * (
            (u / float(self.sigma_north_m)) ** 2
            + (v / float(self.sigma_east_m)) ** 2
        )
        return float(self.amplitude_mps2) * np.exp(exponent)


@dataclass(frozen=True)
class SinusoidalAnomalySource:
    r"""
    Plane-wave disturbance component in the local North/East plane.

    Parameters
    ----------
    origin_lat_rad : float
        Reference latitude for the local metric [rad].
    origin_lon_rad : float
        Reference longitude for the local metric [rad].
    amplitude_mps2 : float
        Disturbance amplitude [m/s^2].
    wavelength_m : float
        Spatial wavelength [m].
    heading_rad : float, default=0.0
        Wave propagation direction measured clockwise from North [rad].
    phase_rad : float, default=0.0
        Phase offset [rad].
    reference_height_m : float, default=0.0
        Reference height used for the local metric linearization [m].

    Formula
    -------
    Let ``s`` be the signed distance along the selected heading. Then:

        g(s) = A * sin(2*pi*s / lambda + phase)

    Notes
    -----
    This is useful for injecting long-wavelength structure into synthetic maps so
    that the particle filter sees both local peaks and broader regional texture.
    """

    origin_lat_rad: float
    origin_lon_rad: float
    amplitude_mps2: float
    wavelength_m: float
    heading_rad: float = 0.0
    phase_rad: float = 0.0
    reference_height_m: float = 0.0

    def __post_init__(self) -> None:
        if float(self.wavelength_m) <= 0.0:
            raise ValueError("wavelength_m must be positive.")

    @classmethod
    def from_mgal(
        cls,
        *,
        origin_lat_rad: float,
        origin_lon_rad: float,
        amplitude_mgal: float,
        wavelength_m: float,
        heading_rad: float = 0.0,
        phase_rad: float = 0.0,
        reference_height_m: float = 0.0,
    ) -> "SinusoidalAnomalySource":
        """Convenience constructor using mGal amplitude input."""
        return cls(
            origin_lat_rad=float(origin_lat_rad),
            origin_lon_rad=float(origin_lon_rad),
            amplitude_mps2=float(mgal_to_mps2(amplitude_mgal)),
            wavelength_m=float(wavelength_m),
            heading_rad=float(heading_rad),
            phase_rad=float(phase_rad),
            reference_height_m=float(reference_height_m),
        )

    @property
    def amplitude_mgal(self) -> float:
        """Amplitude in mGal."""
        return float(mps2_to_mgal(self.amplitude_mps2))

    def evaluate(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
    ) -> FloatArray:
        """
        Evaluate the sinusoidal anomaly at one or more geodetic coordinates.
        """
        d_n, d_e = local_north_east_offsets_from_reference(
            lat_rad,
            lon_rad,
            lat_ref_rad=float(self.origin_lat_rad),
            lon_ref_rad=float(self.origin_lon_rad),
            height_ref_m=float(self.reference_height_m),
        )
        s = np.cos(float(self.heading_rad)) * d_n + np.sin(float(self.heading_rad)) * d_e
        arg = (2.0 * np.pi / float(self.wavelength_m)) * s + float(self.phase_rad)
        return float(self.amplitude_mps2) * np.sin(arg)


# -----------------------------------------------------------------------------
# Main map container
# -----------------------------------------------------------------------------


@dataclass
class GravityGridMap:
    """
    Regular latitude/longitude gravity-disturbance grid.

    Parameters
    ----------
    lat_axis_rad : array-like, shape (Nlat,)
        Latitude grid nodes [rad], strictly increasing.
    lon_axis_rad : array-like, shape (Nlon,)
        Longitude grid nodes [rad], strictly increasing.
    disturbance_grid_mps2 : array-like, shape (Nlat, Nlon)
        Disturbance grid stored at ``reference_height_m`` [m/s^2].
    reference_height_m : float, default=0.0
        Reference ellipsoidal height at which ``disturbance_grid_mps2`` is defined [m].
    vertical_gradient_mps2_per_m : scalar or array-like, default=0.0
        Optional disturbance vertical gradient ``d(delta_g)/dh`` [m/s^2 per m].
        May be a scalar or a grid of the same shape as ``disturbance_grid_mps2``.
    default_method : {"linear", "nearest"}, default="linear"
        Default interpolation method.
    bounds_error : bool, default=True
        If True, out-of-bounds queries raise an error.
    fill_value_mps2 : float, default=np.nan
        Output value for out-of-bounds queries when ``bounds_error=False``.
    name : str, default="gravity_grid_map"
        Human-readable identifier.
    metadata : dict[str, Any], default={}
        Optional free-form metadata.

    Stored field model
    ------------------
    The map stores disturbance at the reference height and evaluates query values as:

        delta_g(lat, lon, h)
            = delta_g_ref(lat, lon)
            + grad_h(lat, lon) * (h - h_ref)

    This is a pragmatic simulator interface, not a claim of full geophysical
    continuation physics.
    """

    lat_axis_rad: ArrayLike
    lon_axis_rad: ArrayLike
    disturbance_grid_mps2: ArrayLike
    reference_height_m: float = 0.0
    vertical_gradient_mps2_per_m: ArrayLike | float = 0.0
    default_method: str = "linear"
    bounds_error: bool = True
    fill_value_mps2: float = np.nan
    name: str = "gravity_grid_map"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lat_axis_rad = _vec1_strictly_increasing(self.lat_axis_rad, name="lat_axis_rad")
        self.lon_axis_rad = _vec1_strictly_increasing(self.lon_axis_rad, name="lon_axis_rad")
        shape = (self.lat_axis_rad.size, self.lon_axis_rad.size)
        self.disturbance_grid_mps2 = _grid2(
            self.disturbance_grid_mps2,
            shape=shape,
            name="disturbance_grid_mps2",
        )
        self.reference_height_m = float(self.reference_height_m)
        self.vertical_gradient_mps2_per_m = _scalar_or_grid(
            self.vertical_gradient_mps2_per_m,
            shape=shape,
            name="vertical_gradient_mps2_per_m",
        )
        self.default_method = str(self.default_method).strip().lower()
        if self.default_method not in {"linear", "nearest"}:
            raise ValueError(
                f"default_method must be 'linear' or 'nearest', got {self.default_method!r}."
            )
        self.bounds_error = bool(self.bounds_error)
        self.fill_value_mps2 = float(self.fill_value_mps2)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @classmethod
    def from_degrees(
        cls,
        *,
        lat_axis_deg: ArrayLike,
        lon_axis_deg: ArrayLike,
        disturbance_grid_mps2: ArrayLike,
        reference_height_m: float = 0.0,
        vertical_gradient_mps2_per_m: ArrayLike | float = 0.0,
        default_method: str = "linear",
        bounds_error: bool = True,
        fill_value_mps2: float = np.nan,
        name: str = "gravity_grid_map",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "GravityGridMap":
        """
        Construct a map from degree-based axes.
        """
        return cls(
            lat_axis_rad=np.deg2rad(_as_float_array(lat_axis_deg)),
            lon_axis_rad=np.deg2rad(_as_float_array(lon_axis_deg)),
            disturbance_grid_mps2=disturbance_grid_mps2,
            reference_height_m=reference_height_m,
            vertical_gradient_mps2_per_m=vertical_gradient_mps2_per_m,
            default_method=default_method,
            bounds_error=bounds_error,
            fill_value_mps2=fill_value_mps2,
            name=name,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_mgal_grid(
        cls,
        *,
        lat_axis_rad: ArrayLike,
        lon_axis_rad: ArrayLike,
        disturbance_grid_mgal: ArrayLike,
        reference_height_m: float = 0.0,
        vertical_gradient_mgal_per_m: ArrayLike | float = 0.0,
        default_method: str = "linear",
        bounds_error: bool = True,
        fill_value_mgal: float = np.nan,
        name: str = "gravity_grid_map",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "GravityGridMap":
        """
        Construct a map from disturbance values expressed in mGal.
        """
        fill_value_mps2 = (
            float(fill_value_mgal)
            if np.isnan(float(fill_value_mgal))
            else float(mgal_to_mps2(fill_value_mgal))
        )

        if np.asarray(vertical_gradient_mgal_per_m).ndim == 0:
            grad = float(mgal_to_mps2(vertical_gradient_mgal_per_m))
        else:
            grad = mgal_to_mps2(_as_float_array(vertical_gradient_mgal_per_m))

        return cls(
            lat_axis_rad=lat_axis_rad,
            lon_axis_rad=lon_axis_rad,
            disturbance_grid_mps2=mgal_to_mps2(_as_float_array(disturbance_grid_mgal)),
            reference_height_m=reference_height_m,
            vertical_gradient_mps2_per_m=grad,
            default_method=default_method,
            bounds_error=bounds_error,
            fill_value_mps2=fill_value_mps2,
            name=name,
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(Nlat, Nlon)``."""
        return (self.lat_axis_rad.size, self.lon_axis_rad.size)

    @property
    def lat_axis_deg(self) -> FloatArray:
        """Latitude axis in degrees."""
        return np.rad2deg(self.lat_axis_rad)

    @property
    def lon_axis_deg(self) -> FloatArray:
        """Longitude axis in degrees."""
        return np.rad2deg(self.lon_axis_rad)

    @property
    def disturbance_grid_mgal(self) -> FloatArray:
        """Disturbance grid in mGal."""
        return np.asarray(mps2_to_mgal(self.disturbance_grid_mps2), dtype=np.float64)

    def contains(self, lat_rad: ArrayLike, lon_rad: ArrayLike):
        """
        Return a boolean mask indicating whether query points lie within the map.
        """
        lat = _as_float_array(lat_rad)
        lon = _as_float_array(lon_rad)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        mask = _in_bounds_mask(self.lat_axis_rad, self.lon_axis_rad, lat_b, lon_b)
        if mask.ndim == 0:
            return bool(mask)
        return np.asarray(mask, dtype=bool)

    def _interp_base(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        *,
        values: FloatArray,
        method: Optional[str] = None,
        bounds_error: Optional[bool] = None,
        fill_value_mps2: Optional[float] = None,
    ) -> FloatArray:
        """
        Interpolate a 2D grid field over the map's lat/lon axes.
        """
        method_use = self.default_method if method is None else str(method).strip().lower()
        bounds_use = self.bounds_error if bounds_error is None else bool(bounds_error)
        fill_use = self.fill_value_mps2 if fill_value_mps2 is None else float(fill_value_mps2)
        return bilinear_interpolate_rectilinear(
            self.lat_axis_rad,
            self.lon_axis_rad,
            values,
            lat_rad,
            lon_rad,
            bounds_error=bounds_use,
            fill_value=fill_use,
            method=method_use,
        )

    def _vertical_gradient_eval(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        *,
        method: Optional[str] = None,
        bounds_error: Optional[bool] = None,
        fill_value_mps2_per_m: Optional[float] = None,
    ) -> FloatArray:
        """
        Evaluate the optional vertical gradient field at query coordinates.
        """
        if isinstance(self.vertical_gradient_mps2_per_m, float):
            lat = _as_float_array(lat_rad)
            lon = _as_float_array(lon_rad)
            lat_b, lon_b = np.broadcast_arrays(lat, lon)
            return np.full(lat_b.shape, float(self.vertical_gradient_mps2_per_m), dtype=np.float64)

        fill_use = (
            self.fill_value_mps2 if fill_value_mps2_per_m is None else float(fill_value_mps2_per_m)
        )
        return self._interp_base(
            lat_rad,
            lon_rad,
            values=self.vertical_gradient_mps2_per_m,
            method=method,
            bounds_error=bounds_error,
            fill_value_mps2=fill_use,
        )

    def sample_disturbance(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        height_m: ArrayLike = 0.0,
        *,
        method: Optional[str] = None,
        bounds_error: Optional[bool] = None,
        fill_value_mps2: Optional[float] = None,
    ):
        r"""
        Evaluate the map disturbance at one or more query points.

        Parameters
        ----------
        lat_rad, lon_rad, height_m : array-like or scalar
            Geodetic query coordinates.
        method : {"linear", "nearest"}, optional
            Interpolation method. Defaults to ``self.default_method``.
        bounds_error : bool, optional
            If True, raise for out-of-bounds queries. Defaults to ``self.bounds_error``.
        fill_value_mps2 : float, optional
            Fill value for out-of-bounds queries when bounds errors are disabled.

        Returns
        -------
        float or np.ndarray
            Disturbance value(s) [m/s^2].
        """
        lat_b, lon_b, h_b, shape, scalar_output = _broadcast_query_inputs(
            lat_rad,
            lon_rad,
            height_m,
        )
        base = self._interp_base(
            lat_b,
            lon_b,
            values=self.disturbance_grid_mps2,
            method=method,
            bounds_error=bounds_error,
            fill_value_mps2=fill_value_mps2,
        )
        grad_h = self._vertical_gradient_eval(
            lat_b,
            lon_b,
            method=method,
            bounds_error=bounds_error,
            fill_value_mps2_per_m=0.0,
        )
        out = base + grad_h * (h_b - float(self.reference_height_m))
        return _restore_shape(out, shape, scalar_output=scalar_output)

    interpolate_disturbance = sample_disturbance
    evaluate_disturbance = sample_disturbance
    lookup_disturbance = sample_disturbance
    disturbance = sample_disturbance

    def __call__(self, lat_rad: ArrayLike, lon_rad: ArrayLike, height_m: ArrayLike = 0.0):
        """Callable alias for ``sample_disturbance(...)``."""
        return self.sample_disturbance(lat_rad, lon_rad, height_m)

    def disturbance_gradient_lat_lon(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        *,
        method: Optional[str] = None,
        bounds_error: Optional[bool] = None,
        fill_value: float = np.nan,
    ) -> tuple[FloatArray, FloatArray]:
        r"""
        Evaluate the horizontal disturbance gradient with respect to latitude and longitude.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(d(delta_g)/dlat, d(delta_g)/dlon)`` in units of ``[m/s^2]/rad``.

        Notes
        -----
        For bilinear interpolation the gradient inside each cell is piecewise
        linear/constant as implied by the bilinear polynomial coefficients.
        For nearest-neighbour interpolation the gradient is reported as zero.
        """
        lat = _as_float_array(lat_rad)
        lon = _as_float_array(lon_rad)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        out_lat = np.full(lat_b.shape, float(fill_value), dtype=np.float64)
        out_lon = np.full(lat_b.shape, float(fill_value), dtype=np.float64)

        method_use = self.default_method if method is None else str(method).strip().lower()
        bounds_use = self.bounds_error if bounds_error is None else bool(bounds_error)
        if method_use not in {"linear", "nearest"}:
            raise ValueError(f"Unsupported interpolation method {method_use!r}.")

        mask = _in_bounds_mask(self.lat_axis_rad, self.lon_axis_rad, lat_b, lon_b)
        if bounds_use and not np.all(mask):
            raise ValueError("One or more query points lie outside the grid bounds.")
        if not np.any(mask):
            return out_lat, out_lon

        if method_use == "nearest":
            out_lat[mask] = 0.0
            out_lon[mask] = 0.0
            return out_lat, out_lon

        lat_valid = lat_b[mask]
        lon_valid = lon_b[mask]

        ix = np.searchsorted(self.lat_axis_rad, lat_valid, side="right") - 1
        iy = np.searchsorted(self.lon_axis_rad, lon_valid, side="right") - 1
        ix = np.clip(ix, 0, self.lat_axis_rad.size - 2)
        iy = np.clip(iy, 0, self.lon_axis_rad.size - 2)

        lat1 = self.lat_axis_rad[ix]
        lat2 = self.lat_axis_rad[ix + 1]
        lon1 = self.lon_axis_rad[iy]
        lon2 = self.lon_axis_rad[iy + 1]
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        tx = (lat_valid - lat1) / dlat
        ty = (lon_valid - lon1) / dlon

        f11 = self.disturbance_grid_mps2[ix, iy]
        f21 = self.disturbance_grid_mps2[ix + 1, iy]
        f12 = self.disturbance_grid_mps2[ix, iy + 1]
        f22 = self.disturbance_grid_mps2[ix + 1, iy + 1]

        out_lat[mask] = ((1.0 - ty) * (f21 - f11) + ty * (f22 - f12)) / dlat
        out_lon[mask] = ((1.0 - tx) * (f12 - f11) + tx * (f22 - f21)) / dlon
        return out_lat, out_lon

    def disturbance_gradient_ned(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        height_m: ArrayLike = 0.0,
        *,
        method: Optional[str] = None,
        bounds_error: Optional[bool] = None,
        fill_value: float = np.nan,
    ) -> FloatArray:
        r"""
        Evaluate the disturbance gradient resolved in local NED coordinates.

        Parameters
        ----------
        lat_rad, lon_rad, height_m : array-like or scalar
            Query coordinates.

        Returns
        -------
        np.ndarray
            Array of shape ``(..., 3)`` containing:

                [d(delta_g)/dN, d(delta_g)/dE, d(delta_g)/dD]

            in units of ``[m/s^2]/m``.

        Formula
        -------
        Using the local curvilinear metric:

            d/dN = (1 / (R_M + h)) d/dphi
            d/dE = (1 / ((R_N + h) cos(phi))) d/dlambda
            d/dD = - d/dh

        where Down-positive implies the minus sign in the vertical derivative.
        """
        lat_b, lon_b, h_b, shape, scalar_output = _broadcast_query_inputs(lat_rad, lon_rad, height_m)
        dgdphi, dgdlambda = self.disturbance_gradient_lat_lon(
            lat_b,
            lon_b,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

        rm = np.asarray(meridian_radius(lat_b), dtype=np.float64) + h_b
        rn = np.asarray(prime_vertical_radius(lat_b), dtype=np.float64) + h_b
        cos_phi = np.cos(lat_b)

        grad_n = dgdphi / rm
        grad_e = dgdlambda / (rn * cos_phi)
        grad_h = self._vertical_gradient_eval(
            lat_b,
            lon_b,
            method=method,
            bounds_error=bounds_error,
            fill_value_mps2_per_m=fill_value,
        )
        grad_d = -grad_h

        out = np.stack([grad_n, grad_e, grad_d], axis=-1)
        if scalar_output:
            return np.asarray(out.reshape(3), dtype=np.float64)
        return np.asarray(out.reshape(shape + (3,)), dtype=np.float64)

    def crop(
        self,
        *,
        lat_min_rad: float,
        lat_max_rad: float,
        lon_min_rad: float,
        lon_max_rad: float,
        name: Optional[str] = None,
    ) -> "GravityGridMap":
        """
        Return a cropped sub-map.
        """
        lat_min = float(lat_min_rad)
        lat_max = float(lat_max_rad)
        lon_min = float(lon_min_rad)
        lon_max = float(lon_max_rad)
        lat_mask = (self.lat_axis_rad >= lat_min) & (self.lat_axis_rad <= lat_max)
        lon_mask = (self.lon_axis_rad >= lon_min) & (self.lon_axis_rad <= lon_max)
        if np.count_nonzero(lat_mask) < 2 or np.count_nonzero(lon_mask) < 2:
            raise ValueError("Crop bounds must retain at least 2 samples on each axis.")

        grad = self.vertical_gradient_mps2_per_m
        if isinstance(grad, float):
            grad_new: float | FloatArray = grad
        else:
            grad_new = grad[np.ix_(lat_mask, lon_mask)]

        return GravityGridMap(
            lat_axis_rad=self.lat_axis_rad[lat_mask],
            lon_axis_rad=self.lon_axis_rad[lon_mask],
            disturbance_grid_mps2=self.disturbance_grid_mps2[np.ix_(lat_mask, lon_mask)],
            reference_height_m=self.reference_height_m,
            vertical_gradient_mps2_per_m=grad_new,
            default_method=self.default_method,
            bounds_error=self.bounds_error,
            fill_value_mps2=self.fill_value_mps2,
            name=self.name if name is None else str(name),
            metadata=dict(self.metadata),
        )

    def to_npz(self, path: str | Path) -> Path:
        """
        Save the map to an ``.npz`` file.

        Notes
        -----
        ``metadata`` is serialized through JSON so it should contain only JSON-
        serializable content if exact round-tripping matters.
        """
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        grad = self.vertical_gradient_mps2_per_m
        grad_is_scalar = isinstance(grad, float)
        np.savez_compressed(
            p,
            lat_axis_rad=self.lat_axis_rad,
            lon_axis_rad=self.lon_axis_rad,
            disturbance_grid_mps2=self.disturbance_grid_mps2,
            reference_height_m=np.array(self.reference_height_m, dtype=np.float64),
            vertical_gradient_is_scalar=np.array(1 if grad_is_scalar else 0, dtype=np.int64),
            vertical_gradient_mps2_per_m=(
                np.array(float(grad), dtype=np.float64) if grad_is_scalar else np.asarray(grad, dtype=np.float64)
            ),
            default_method=np.array(self.default_method),
            bounds_error=np.array(1 if self.bounds_error else 0, dtype=np.int64),
            fill_value_mps2=np.array(self.fill_value_mps2, dtype=np.float64),
            name=np.array(self.name),
            metadata_json=np.array(json.dumps(self.metadata)),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "GravityGridMap":
        """
        Load a map from an ``.npz`` file produced by :meth:`to_npz`.
        """
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            grad_is_scalar = bool(int(data["vertical_gradient_is_scalar"]))
            grad_arr = np.asarray(data["vertical_gradient_mps2_per_m"], dtype=np.float64)
            grad: float | FloatArray = float(grad_arr) if grad_is_scalar else grad_arr
            metadata_raw = str(np.asarray(data["metadata_json"]).item())
            metadata = json.loads(metadata_raw) if metadata_raw else {}
            return cls(
                lat_axis_rad=np.asarray(data["lat_axis_rad"], dtype=np.float64),
                lon_axis_rad=np.asarray(data["lon_axis_rad"], dtype=np.float64),
                disturbance_grid_mps2=np.asarray(data["disturbance_grid_mps2"], dtype=np.float64),
                reference_height_m=float(np.asarray(data["reference_height_m"], dtype=np.float64)),
                vertical_gradient_mps2_per_m=grad,
                default_method=str(np.asarray(data["default_method"]).item()),
                bounds_error=bool(int(np.asarray(data["bounds_error"]).item())),
                fill_value_mps2=float(np.asarray(data["fill_value_mps2"], dtype=np.float64)),
                name=str(np.asarray(data["name"]).item()),
                metadata=metadata,
            )


# -----------------------------------------------------------------------------
# Synthetic map builders
# -----------------------------------------------------------------------------


def build_synthetic_disturbance_grid(
    lat_axis_rad: ArrayLike,
    lon_axis_rad: ArrayLike,
    *,
    gaussian_sources: Sequence[GaussianAnomalySource] = (),
    sinusoid_sources: Sequence[SinusoidalAnomalySource] = (),
    constant_bias_mps2: float = 0.0,
    planar_north_gradient_mps2_per_m: float = 0.0,
    planar_east_gradient_mps2_per_m: float = 0.0,
    planar_reference_lat_rad: Optional[float] = None,
    planar_reference_lon_rad: Optional[float] = None,
    planar_reference_height_m: float = 0.0,
    white_noise_std_mps2: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> FloatArray:
    r"""
    Build a synthetic disturbance grid from analytic components.

    Parameters
    ----------
    lat_axis_rad, lon_axis_rad : array-like
        Target grid axes.
    gaussian_sources : sequence, default=()
        Gaussian anomaly components.
    sinusoid_sources : sequence, default=()
        Sinusoidal anomaly components.
    constant_bias_mps2 : float, default=0.0
        Constant background offset [m/s^2].
    planar_north_gradient_mps2_per_m : float, default=0.0
        Optional linear North gradient [m/s^2 per m].
    planar_east_gradient_mps2_per_m : float, default=0.0
        Optional linear East gradient [m/s^2 per m].
    planar_reference_lat_rad, planar_reference_lon_rad : float, optional
        Reference origin for the planar trend. If omitted, the grid centre is used.
    planar_reference_height_m : float, default=0.0
        Reference height used for the planar local metric [m].
    white_noise_std_mps2 : float, default=0.0
        Optional iid white texture added to each cell [m/s^2].
    rng : numpy.random.Generator, optional
        RNG used when ``white_noise_std_mps2 > 0``.

    Returns
    -------
    np.ndarray, shape (Nlat, Nlon)
        Synthetic disturbance grid [m/s^2].

    Notes
    -----
    This is a deterministic structured-field generator with optional additive
    white texture. It is meant for simulator development, not for geophysical
    interpretation.
    """
    lat_axis = _vec1_strictly_increasing(lat_axis_rad, name="lat_axis_rad")
    lon_axis = _vec1_strictly_increasing(lon_axis_rad, name="lon_axis_rad")

    lat_mesh, lon_mesh = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    grid = np.full(lat_mesh.shape, float(constant_bias_mps2), dtype=np.float64)

    if planar_reference_lat_rad is None:
        lat0 = float(0.5 * (lat_axis[0] + lat_axis[-1]))
    else:
        lat0 = float(planar_reference_lat_rad)
    if planar_reference_lon_rad is None:
        lon0 = float(0.5 * (lon_axis[0] + lon_axis[-1]))
    else:
        lon0 = float(planar_reference_lon_rad)

    if planar_north_gradient_mps2_per_m != 0.0 or planar_east_gradient_mps2_per_m != 0.0:
        d_n, d_e = local_north_east_offsets_from_reference(
            lat_mesh,
            lon_mesh,
            lat_ref_rad=lat0,
            lon_ref_rad=lon0,
            height_ref_m=float(planar_reference_height_m),
        )
        grid += float(planar_north_gradient_mps2_per_m) * d_n
        grid += float(planar_east_gradient_mps2_per_m) * d_e

    for src in gaussian_sources:
        grid += np.asarray(src.evaluate(lat_mesh, lon_mesh), dtype=np.float64)

    for src in sinusoid_sources:
        grid += np.asarray(src.evaluate(lat_mesh, lon_mesh), dtype=np.float64)

    white_std = float(white_noise_std_mps2)
    if white_std < 0.0:
        raise ValueError("white_noise_std_mps2 must be nonnegative.")
    if white_std > 0.0:
        rng_use = np.random.default_rng() if rng is None else rng
        grid += white_std * rng_use.standard_normal(grid.shape)

    return np.asarray(grid, dtype=np.float64)


def build_synthetic_gravity_map(
    *,
    lat_axis_rad: ArrayLike,
    lon_axis_rad: ArrayLike,
    gaussian_sources: Sequence[GaussianAnomalySource] = (),
    sinusoid_sources: Sequence[SinusoidalAnomalySource] = (),
    constant_bias_mps2: float = 0.0,
    planar_north_gradient_mps2_per_m: float = 0.0,
    planar_east_gradient_mps2_per_m: float = 0.0,
    planar_reference_lat_rad: Optional[float] = None,
    planar_reference_lon_rad: Optional[float] = None,
    planar_reference_height_m: float = 0.0,
    reference_height_m: float = 0.0,
    vertical_gradient_mps2_per_m: ArrayLike | float = 0.0,
    white_noise_std_mps2: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    default_method: str = "linear",
    bounds_error: bool = True,
    fill_value_mps2: float = np.nan,
    name: str = "synthetic_gravity_map",
    metadata: Optional[dict[str, Any]] = None,
) -> GravityGridMap:
    """
    Build a :class:`GravityGridMap` from analytic synthetic components.
    """
    disturbance = build_synthetic_disturbance_grid(
        lat_axis_rad=lat_axis_rad,
        lon_axis_rad=lon_axis_rad,
        gaussian_sources=gaussian_sources,
        sinusoid_sources=sinusoid_sources,
        constant_bias_mps2=constant_bias_mps2,
        planar_north_gradient_mps2_per_m=planar_north_gradient_mps2_per_m,
        planar_east_gradient_mps2_per_m=planar_east_gradient_mps2_per_m,
        planar_reference_lat_rad=planar_reference_lat_rad,
        planar_reference_lon_rad=planar_reference_lon_rad,
        planar_reference_height_m=planar_reference_height_m,
        white_noise_std_mps2=white_noise_std_mps2,
        rng=rng,
    )
    meta = {} if metadata is None else dict(metadata)
    meta.setdefault("synthetic", True)
    meta.setdefault("num_gaussian_sources", int(len(gaussian_sources)))
    meta.setdefault("num_sinusoid_sources", int(len(sinusoid_sources)))

    return GravityGridMap(
        lat_axis_rad=lat_axis_rad,
        lon_axis_rad=lon_axis_rad,
        disturbance_grid_mps2=disturbance,
        reference_height_m=reference_height_m,
        vertical_gradient_mps2_per_m=vertical_gradient_mps2_per_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_mps2=fill_value_mps2,
        name=name,
        metadata=meta,
    )


__all__ = [
    "FloatArray",
    "GaussianAnomalySource",
    "GravityGridMap",
    "SinusoidalAnomalySource",
    "bilinear_interpolate_rectilinear",
    "build_synthetic_disturbance_grid",
    "build_synthetic_gravity_map",
    "geodetic_axes_from_local_extent",
    "local_north_east_offsets_from_reference",
]