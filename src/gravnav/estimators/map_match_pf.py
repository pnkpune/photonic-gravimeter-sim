"""
map_match_pf.py

Particle-filter gravity map matching for the gravity-aided navigation simulator.

This module provides a practical sequential map-matching layer on top of the
repository's IMU / INS / fusion foundation. Its job is to estimate platform
position from:
- a scalar gravity-disturbance measurement stream
- a prior trajectory from the INS
- a geo-referenced gravity map or gravity-map-like callable

Why this file exists
--------------------
The repository now has:
- Earth / frames / geodesy foundations
- truth trajectory generation
- IMU, gravimeter, depth, and velocity-aid sensor models
- an error-state INS propagator
- a generic fusion layer for linearized aiding updates

The next estimator layer in the project roadmap is gravity map matching. This
module implements that layer as a particle filter (PF) because gravity-based map
matching is naturally:
- nonlinear
- multi-modal
- ambiguous in low-feature map regions
- sequential in time

and therefore does not fit cleanly into a purely Gaussian single-hypothesis
filter on its own.

Conventions
-----------
- Navigation frame is NED:
      x = North, y = East, z = Down
- Geodetic state is:
      latitude  [rad]
      longitude [rad], east-positive
      height    [m], ellipsoidal and positive upward
- Gravity map values in this module are assumed to be scalar gravity
  disturbance values at the measurement point in m/s^2, consistent with the
  repository gravimeter model.
- The PF state in this file is intentionally *position-only*:
      [lat, lon, h]
  The INS already carries velocity, attitude, and IMU biases. The PF uses the
  INS trajectory as a proposal / motion prior rather than re-estimating the full
  navigation state.

Map interface design
--------------------
`gravity_map.py` is still scaffold-only in the current repository, so this file
does not hard-depend on a concrete map class. Instead it accepts any map object
that supports one of the following call signatures:

1) callable:
       map_model(lat_rad, lon_rad, height_m) -> disturbance_mps2

2) method-based:
       map_model.sample_disturbance(lat_rad, lon_rad, height_m)
       map_model.interpolate_disturbance(lat_rad, lon_rad, height_m)
       map_model.evaluate_disturbance(lat_rad, lon_rad, height_m)
       map_model.lookup_disturbance(lat_rad, lon_rad, height_m)
       map_model.disturbance(lat_rad, lon_rad, height_m)

The implementation first tries vectorized evaluation. If the map backend only
supports scalar calls, it falls back to elementwise evaluation.

Statistical model
-----------------
At each update time k:

    x_k^(i) ~ p(x_k | x_{k-1}^(i), INS motion prior)

and the gravity measurement model is:

    z_k = g_map(x_k) + eps_k
    eps_k ~ N(0, sigma_g^2)

Optionally, the update can also include:
- a depth likelihood:
      d_k = h_ref - h_k + eps_d
- a soft INS position prior likelihood in local NED coordinates

Resampling
----------
This file uses:
- effective sample size (ESS) to monitor degeneracy
- systematic resampling when ESS falls below a chosen fraction of particle count
- optional post-resample rejuvenation jitter to retain diversity

Primary references used here
----------------------------
1) Arulampalam, S., Maskell, S., Gordon, N., and Clapp, T. (2002),
   "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking"
   IEEE Transactions on Signal Processing, 50(2), 174-188.
   URL:
   https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/Arulampalam_etal_2002.pdf

   Used for:
   - PF motivation for nonlinear / non-Gaussian sequential estimation
   - weight degeneracy and effective sample size
   - systematic resampling as a simple practical default

2) Li, W., Gilliam, C., Wang, X., Kealy, A., Greentree, A. D., and Moran, B. (2024),
   "Gravity-aided navigation using Viterbi map matching algorithm"
   The Journal of Navigation, 77(3), 307-321.
   URL:
   https://doi.org/10.1017/S0373463324000250

   Used for:
   - framing gravity map matching as a sequential estimation problem
   - explicit statement that the matching algorithm must account for:
       * sensor noise
       * spatial uncertainty
       * map ambiguity
   - the practical reminder that INS velocity information is naturally part of
     the motion model around the map-matching layer

3) Repository modules:
   - `gravnav.sensors.gravimeter`
   - `gravnav.sensors.depth`
   - `gravnav.estimators.error_state_ins`
   - `gravnav.estimators.fusion`

   This file is intentionally aligned with the repository's conventions for:
   - scalar gravity disturbance
   - signed depth
   - NED/local-level navigation
   - geodetic state representation

Design notes
------------
- This is a practical bootstrap/SIR-style particle filter for map matching.
- It is intentionally conservative and transparent.
- The PF is position-only by design so that it complements, rather than replaces,
  the error-state INS.
- The INS prior is modeled as a motion proposal plus an optional soft position
  likelihood. This gives a clean path to later integration with `fusion.py`.
- Once `gravity_map.py` exists, this file should work with it without major
  changes if that module exposes one of the accepted map-evaluation APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import meridian_radius, prime_vertical_radius
from ..physics.frames import wrap_angle_pi
from ..sensors.depth import DepthMeasurement
from ..sensors.gravimeter import GravimeterMeasurement
from .error_state_ins import ErrorStateINS, ErrorStateINSState

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


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a shape-(3,) vector."""
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Convert a scalar or 3-vector into a 3-vector.
    """
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}.")
    return arr


def _nonnegative_axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Like `_axis3(...)`, but require all entries to be nonnegative.
    """
    arr = _axis3(x, name=name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative, got {arr}.")
    return arr


def _normalize_weights(weights: ArrayLike) -> FloatArray:
    """
    Normalize nonnegative particle weights.

    Raises
    ------
    ValueError
        If weights are negative or sum to zero.
    """
    w = _as_float_array(weights).reshape(-1)
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative.")
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError("weights must have a finite positive sum.")
    return (w / s).astype(np.float64)


def _normalize_log_weights(log_weights: ArrayLike) -> FloatArray:
    """
    Normalize log-weights in a numerically stable way.

    Returns
    -------
    np.ndarray, shape (N,)
        Normalized linear weights.
    """
    lw = _as_float_array(log_weights).reshape(-1)
    if lw.size == 0:
        raise ValueError("log_weights must be non-empty.")
    lw_max = float(np.max(lw))
    w = np.exp(lw - lw_max)
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        # Robust fallback rather than hard failure.
        return np.full(lw.size, 1.0 / lw.size, dtype=np.float64)
    return (w / s).astype(np.float64)


def effective_sample_size(weights: ArrayLike) -> float:
    r"""
    Compute the effective sample size (ESS).

    Formula
    -------
        N_eff = 1 / sum_i w_i^2

    where weights are assumed normalized.
    """
    w = _normalize_weights(weights)
    return float(1.0 / np.sum(w**2))


def systematic_resample(
    weights: ArrayLike,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    r"""
    Systematic resampling indices.

    Parameters
    ----------
    weights : array-like, shape (N,)
        Normalized or unnormalized particle weights.
    rng : numpy.random.Generator
        Random-number generator.

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
        Parent indices after resampling.

    Method
    ------
    This is the standard systematic-resampling construction based on one random
    offset and evenly spaced cumulative-probability targets.
    """
    w = _normalize_weights(weights)
    n = w.size
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    cdf = np.cumsum(w)
    idx = np.searchsorted(cdf, positions, side="right")
    idx = np.clip(idx, 0, n - 1)
    return idx.astype(np.int64)


def _state_from_filter_or_state(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> ErrorStateINSState:
    """
    Return `ErrorStateINSState` whether the caller passed the filter or the state.
    """
    if isinstance(ins_or_state, ErrorStateINS):
        return ins_or_state.state
    if isinstance(ins_or_state, ErrorStateINSState):
        return ins_or_state
    raise TypeError(
        "Expected ErrorStateINS or ErrorStateINSState, "
        f"got {type(ins_or_state).__name__}."
    )


# -----------------------------------------------------------------------------
# Gravity map interface abstraction
# -----------------------------------------------------------------------------


@runtime_checkable
class GravityMapCallable(Protocol):
    """
    Protocol for a generic gravity-map backend.

    The backend may be a callable or expose one of several method names. This is
    intentionally permissive because `physics/gravity_map.py` is not yet fixed.
    """

    def __call__(
        self,
        lat_rad: ArrayLike,
        lon_rad: ArrayLike,
        height_m: ArrayLike,
    ) -> ArrayLike:
        ...


def _evaluate_map_scalar_method(
    fn: Callable[[float, float, float], Any],
    lat_rad: FloatArray,
    lon_rad: FloatArray,
    height_m: FloatArray,
) -> FloatArray:
    """
    Evaluate a scalar-only map function elementwise over particle arrays.
    """
    out = np.empty(lat_rad.size, dtype=np.float64)
    for k in range(lat_rad.size):
        out[k] = float(fn(float(lat_rad[k]), float(lon_rad[k]), float(height_m[k])))
    return out


def evaluate_gravity_map_disturbance(
    map_model: Any,
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
) -> FloatArray:
    """
    Evaluate map-predicted scalar gravity disturbance for one or many positions.

    Parameters
    ----------
    map_model : object
        Gravity map backend. Accepted APIs:
        - callable(lat, lon, h)
        - .sample_disturbance(...)
        - .interpolate_disturbance(...)
        - .evaluate_disturbance(...)
        - .lookup_disturbance(...)
        - .disturbance(...)
    lat_rad, lon_rad, height_m : array-like
        Position coordinates. Must be broadcast-compatible and ultimately reduce
        to the same flattened shape.

    Returns
    -------
    np.ndarray, shape (N,)
        Disturbance values [m/s^2].

    Notes
    -----
    The function first tries vectorized evaluation. If that fails, it falls back
    to elementwise scalar calls.
    """
    lat = _as_float_array(lat_rad).reshape(-1)
    lon = _as_float_array(lon_rad).reshape(-1)
    h = _as_float_array(height_m).reshape(-1)

    if not (lat.shape == lon.shape == h.shape):
        raise ValueError(
            f"lat_rad, lon_rad, and height_m must share one shape, got "
            f"{lat.shape}, {lon.shape}, {h.shape}."
        )

    candidate_fns: list[Callable[..., Any]] = []

    if callable(map_model):
        candidate_fns.append(map_model)

    for name in (
        "sample_disturbance",
        "interpolate_disturbance",
        "evaluate_disturbance",
        "lookup_disturbance",
        "disturbance",
    ):
        fn = getattr(map_model, name, None)
        if callable(fn):
            candidate_fns.append(fn)

    if len(candidate_fns) == 0:
        raise TypeError(
            "map_model must be callable or provide one of: "
            "sample_disturbance, interpolate_disturbance, evaluate_disturbance, "
            "lookup_disturbance, disturbance."
        )

    last_exception: Exception | None = None

    for fn in candidate_fns:
        # Try vectorized call first.
        try:
            out = _as_float_array(fn(lat, lon, h)).reshape(-1)
            if out.shape == lat.shape:
                return out.astype(np.float64)
            if out.size == 1 and lat.size > 1:
                return np.full(lat.size, float(out[0]), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - fallback path
            last_exception = exc

        # Fall back to scalar evaluation.
        try:
            return _evaluate_map_scalar_method(fn, lat, lon, h)
        except Exception as exc:  # pragma: no cover - fallback path
            last_exception = exc

    raise RuntimeError(
        "Failed to evaluate gravity map model with any supported API."
    ) from last_exception


# -----------------------------------------------------------------------------
# Local geodetic/NED geometry helpers
# -----------------------------------------------------------------------------


def _meridian_radius_array(lat_rad: ArrayLike) -> FloatArray:
    """
    Evaluate meridian radius for scalar or array latitude input.
    """
    lat = _as_float_array(lat_rad).reshape(-1)
    try:
        out = _as_float_array(meridian_radius(lat)).reshape(-1)
        if out.shape == lat.shape:
            return out
    except Exception:
        pass
    return np.array([float(meridian_radius(float(phi))) for phi in lat], dtype=np.float64)


def _prime_vertical_radius_array(lat_rad: ArrayLike) -> FloatArray:
    """
    Evaluate prime-vertical radius for scalar or array latitude input.
    """
    lat = _as_float_array(lat_rad).reshape(-1)
    try:
        out = _as_float_array(prime_vertical_radius(lat)).reshape(-1)
        if out.shape == lat.shape:
            return out
    except Exception:
        pass
    return np.array(
        [float(prime_vertical_radius(float(phi))) for phi in lat],
        dtype=np.float64,
    )


def _safe_cos_lat(lat_rad: ArrayLike) -> FloatArray:
    """
    Cosine of latitude with a small floor to avoid singular behavior very near poles.

    This repository is not targeting polar scenarios in its early-stage maritime /
    UAV use cases, but this floor keeps the numerics from exploding if particles
    wander into pathological latitudes.
    """
    c = np.cos(_as_float_array(lat_rad).reshape(-1))
    c_abs = np.maximum(np.abs(c), 1.0e-8)
    return np.sign(c) * c_abs


def apply_ned_offsets_to_geodetic(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
    ned_offsets_m: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Apply local NED offsets to geodetic positions with a first-order local model.

    Parameters
    ----------
    lat_rad, lon_rad, height_m : array-like, shape (N,)
        Current geodetic particle positions.
    ned_offsets_m : array-like, shape (N, 3)
        Local NED offsets [m] to apply.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated `(lat, lon, h)` each shape `(N,)`.

    Formula
    -------
    Using the standard small-displacement local curvilinear approximation:

        d_lat = dN / (R_M + h)
        d_lon = dE / ((R_N + h) cos(lat))
        d_h   = -dD

    because Down is positive in NED while ellipsoidal height is positive upward.
    """
    lat = _as_float_array(lat_rad).reshape(-1)
    lon = _as_float_array(lon_rad).reshape(-1)
    h = _as_float_array(height_m).reshape(-1)
    ned = _as_float_array(ned_offsets_m)

    if ned.shape != (lat.size, 3):
        raise ValueError(
            f"ned_offsets_m must have shape ({lat.size}, 3), got {ned.shape}."
        )
    if not (lat.shape == lon.shape == h.shape):
        raise ValueError(
            f"lat_rad, lon_rad, height_m must share one shape, got "
            f"{lat.shape}, {lon.shape}, {h.shape}."
        )

    M = _meridian_radius_array(lat)
    N = _prime_vertical_radius_array(lat)
    cos_phi = _safe_cos_lat(lat)

    dlat = ned[:, 0] / (M + h)
    dlon = ned[:, 1] / ((N + h) * cos_phi)
    dh = -ned[:, 2]

    lat_new = lat + dlat
    lon_new = np.array([float(wrap_angle_pi(v)) for v in lon + dlon], dtype=np.float64)
    h_new = h + dh
    return lat_new, lon_new, h_new


def geodetic_offsets_to_local_ned(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
    *,
    lat_ref_rad: float,
    lon_ref_rad: float,
    height_ref_m: float,
) -> FloatArray:
    r"""
    Convert geodetic position offsets to a local NED approximation about a reference.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Local offsets `[dN, dE, dD]` [m] from the reference state.

    Formula
    -------
    Around a reference point `(phi_ref, lambda_ref, h_ref)`:

        dN ≈ (phi - phi_ref) (R_M + h_ref)
        dE ≈ (lambda - lambda_ref) (R_N + h_ref) cos(phi_ref)
        dD = -(h - h_ref)
    """
    lat = _as_float_array(lat_rad).reshape(-1)
    lon = _as_float_array(lon_rad).reshape(-1)
    h = _as_float_array(height_m).reshape(-1)

    if not (lat.shape == lon.shape == h.shape):
        raise ValueError(
            f"lat_rad, lon_rad, height_m must share one shape, got "
            f"{lat.shape}, {lon.shape}, {h.shape}."
        )

    phi_ref = float(lat_ref_rad)
    lam_ref = float(lon_ref_rad)
    h_ref = float(height_ref_m)

    M_ref = float(meridian_radius(phi_ref))
    N_ref = float(prime_vertical_radius(phi_ref))
    cos_ref = max(abs(float(np.cos(phi_ref))), 1.0e-8) * np.sign(np.cos(phi_ref) if np.cos(phi_ref) != 0 else 1.0)

    dlat = lat - phi_ref
    dlon = np.array([float(wrap_angle_pi(v - lam_ref)) for v in lon], dtype=np.float64)
    dh = h - h_ref

    dN = dlat * (M_ref + h_ref)
    dE = dlon * (N_ref + h_ref) * cos_ref
    dD = -dh

    return np.column_stack([dN, dE, dD]).astype(np.float64)


def geodetic_covariance_from_ned_covariance(
    lat_ref_rad: float,
    height_ref_m: float,
    ned_cov_m2: ArrayLike,
) -> FloatArray:
    r"""
    Convert a small local NED covariance into geodetic `[lat, lon, h]` covariance.

    Parameters
    ----------
    lat_ref_rad : float
        Reference latitude [rad].
    height_ref_m : float
        Reference ellipsoidal height [m].
    ned_cov_m2 : array-like, shape (3, 3)
        Covariance of `[dN, dE, dD]` in metres.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Approximate covariance of `[d_lat, d_lon, d_h]`.

    Linear mapping
    --------------
        d_lat = dN / (R_M + h)
        d_lon = dE / ((R_N + h) cos(lat))
        d_h   = -dD
    """
    P_ned = _as_float_array(ned_cov_m2)
    if P_ned.shape != (3, 3):
        raise ValueError(f"ned_cov_m2 must have shape (3, 3), got {P_ned.shape}.")

    phi = float(lat_ref_rad)
    h = float(height_ref_m)
    M = float(meridian_radius(phi))
    N = float(prime_vertical_radius(phi))
    cos_phi = max(abs(float(np.cos(phi))), 1.0e-8) * np.sign(np.cos(phi) if np.cos(phi) != 0 else 1.0)

    J = np.array(
        [
            [1.0 / (M + h), 0.0, 0.0],
            [0.0, 1.0 / ((N + h) * cos_phi), 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return (J @ _symmetrize(P_ned) @ J.T).astype(np.float64)


def _symmetrize(M: ArrayLike) -> FloatArray:
    """Return the symmetric part of a square matrix."""
    A = _as_float_array(M)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}.")
    return 0.5 * (A + A.T)


def weighted_geodetic_mean(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
    weights: ArrayLike,
) -> tuple[float, float, float]:
    """
    Weighted mean geodetic state.

    Longitude is averaged on the circle using weighted sine/cosine moments.
    """
    lat = _as_float_array(lat_rad).reshape(-1)
    lon = _as_float_array(lon_rad).reshape(-1)
    h = _as_float_array(height_m).reshape(-1)
    w = _normalize_weights(weights)

    if not (lat.shape == lon.shape == h.shape == w.shape):
        raise ValueError(
            f"lat_rad, lon_rad, height_m, and weights must share one shape, got "
            f"{lat.shape}, {lon.shape}, {h.shape}, {w.shape}."
        )

    lat_mean = float(np.sum(w * lat))
    h_mean = float(np.sum(w * h))

    s = float(np.sum(w * np.sin(lon)))
    c = float(np.sum(w * np.cos(lon)))
    lon_mean = float(np.arctan2(s, c))
    lon_mean = float(wrap_angle_pi(lon_mean))

    return lat_mean, lon_mean, h_mean


def weighted_covariance(
    samples: ArrayLike,
    weights: ArrayLike,
) -> FloatArray:
    """
    Weighted covariance of sample rows.

    Parameters
    ----------
    samples : array-like, shape (N, D)
        Sample rows.
    weights : array-like, shape (N,)
        Normalized or unnormalized nonnegative weights.

    Returns
    -------
    np.ndarray, shape (D, D)
        Weighted covariance using the normalized second central moment.
    """
    X = _as_float_array(samples)
    if X.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {X.shape}.")
    w = _normalize_weights(weights)
    if X.shape[0] != w.size:
        raise ValueError(
            f"samples and weights must agree on N, got {X.shape[0]} and {w.size}."
        )

    mean = np.sum(X * w[:, None], axis=0)
    dX = X - mean[None, :]
    P = (dX * w[:, None]).T @ dX
    return _symmetrize(P)


# -----------------------------------------------------------------------------
# Likelihood helpers
# -----------------------------------------------------------------------------


def gaussian_log_likelihood_scalar(
    residual: ArrayLike,
    sigma: float,
) -> FloatArray:
    r"""
    Scalar Gaussian log-likelihood up to the full normalizing constant.

    Formula
    -------
        log p(r) = -0.5 * (r^2 / sigma^2 + log(2 pi sigma^2))
    """
    s = float(sigma)
    if s <= 0.0:
        raise ValueError(f"sigma must be positive, got {s}.")
    r = _as_float_array(residual)
    var = s * s
    return (-0.5 * ((r * r) / var + np.log(2.0 * np.pi * var))).astype(np.float64)


def diagonal_gaussian_log_likelihood(
    residuals: ArrayLike,
    sigmas: ArrayLike | float,
) -> FloatArray:
    r"""
    Diagonal-Gaussian log-likelihood for sample rows.

    Parameters
    ----------
    residuals : array-like, shape (N, D)
        Residual rows.
    sigmas : scalar or array-like, shape (D,)
        Standard deviations.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-row log-likelihood.

    Formula
    -------
        log p(r) = -0.5 * sum_j ( r_j^2 / sigma_j^2 + log(2 pi sigma_j^2) )
    """
    R = _as_float_array(residuals)
    if R.ndim != 2:
        raise ValueError(f"residuals must be 2D, got shape {R.shape}.")
    sigma = _axis3(sigmas, name="sigmas") if R.shape[1] == 3 else _as_float_array(sigmas).reshape(-1)

    if sigma.shape != (R.shape[1],):
        raise ValueError(
            f"sigmas must have shape ({R.shape[1]},), got {sigma.shape}."
        )
    if np.any(sigma <= 0.0):
        raise ValueError(f"sigmas must be positive, got {sigma}.")

    var = sigma**2
    return (
        -0.5
        * np.sum((R**2) / var[None, :] + np.log(2.0 * np.pi * var[None, :]), axis=1)
    ).astype(np.float64)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class MapMatchPFSpec:
    """
    Particle-filter specification for gravity map matching.

    Parameters
    ----------
    num_particles : int, default=512
        Number of particles.
    init_position_std_m : scalar or shape (3,), default=(250, 250, 20)
        Initial particle spread around the starting INS position in local NED
        coordinates [m].
    process_position_rw_std_m_per_sqrt_s : scalar or shape (3,), default=(10, 10, 0.5)
        Position random-walk diffusion coefficient in local NED [m/sqrt(s)].
        Over one time step dt, the process-noise standard deviation is:
            sigma_step = sigma_rw * sqrt(dt)
    rejuvenation_std_m : scalar or shape (3,), default=(20, 20, 1)
        Post-resample jitter in local NED [m].
    gravity_meas_std_mps2 : float, default=1e-5
        Default scalar gravity-disturbance measurement standard deviation [m/s^2].
        1e-5 m/s^2 corresponds to 1 mGal.
    depth_meas_std_m : float or None, default=None
        Optional default depth-measurement standard deviation [m].
    ins_position_prior_std_m : scalar or shape (3,), default=(150, 150, 15)
        Standard deviation of the optional soft INS position prior in local NED [m].
    use_ins_position_prior : bool, default=True
        Whether to include the soft INS position prior likelihood when an INS
        state is supplied to `update(...)`.
    resample_effective_fraction : float, default=0.5
        Trigger resampling when:
            ESS < resample_effective_fraction * num_particles
    name : str, default="gravity_map_pf"
        Human-readable identifier.
    """

    num_particles: int = 512
    init_position_std_m: ArrayLike | float = (250.0, 250.0, 20.0)
    process_position_rw_std_m_per_sqrt_s: ArrayLike | float = (10.0, 10.0, 0.5)
    rejuvenation_std_m: ArrayLike | float = (20.0, 20.0, 1.0)
    gravity_meas_std_mps2: float = 1.0e-5
    depth_meas_std_m: Optional[float] = None
    ins_position_prior_std_m: ArrayLike | float = (150.0, 150.0, 15.0)
    use_ins_position_prior: bool = True
    resample_effective_fraction: float = 0.5
    name: str = "gravity_map_pf"

    def __post_init__(self) -> None:
        self.num_particles = int(self.num_particles)
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least 2.")

        self.init_position_std_m = _nonnegative_axis3(
            self.init_position_std_m,
            name="init_position_std_m",
        )
        self.process_position_rw_std_m_per_sqrt_s = _nonnegative_axis3(
            self.process_position_rw_std_m_per_sqrt_s,
            name="process_position_rw_std_m_per_sqrt_s",
        )
        self.rejuvenation_std_m = _nonnegative_axis3(
            self.rejuvenation_std_m,
            name="rejuvenation_std_m",
        )
        self.ins_position_prior_std_m = _nonnegative_axis3(
            self.ins_position_prior_std_m,
            name="ins_position_prior_std_m",
        )

        self.gravity_meas_std_mps2 = float(self.gravity_meas_std_mps2)
        if self.gravity_meas_std_mps2 <= 0.0:
            raise ValueError("gravity_meas_std_mps2 must be positive.")

        if self.depth_meas_std_m is not None:
            self.depth_meas_std_m = float(self.depth_meas_std_m)
            if self.depth_meas_std_m <= 0.0:
                raise ValueError("depth_meas_std_m must be positive when provided.")

        self.resample_effective_fraction = float(self.resample_effective_fraction)
        if not (0.0 < self.resample_effective_fraction <= 1.0):
            raise ValueError(
                "resample_effective_fraction must lie in (0, 1]."
            )


@dataclass
class ParticleCloudEstimate:
    """
    Weighted particle-cloud position estimate.

    Attributes
    ----------
    lat_rad : float
        Estimated geodetic latitude [rad].
    lon_rad : float
        Estimated longitude [rad].
    height_m : float
        Estimated ellipsoidal height [m].
    covariance_ned_m2 : np.ndarray, shape (3, 3)
        Weighted local NED covariance about the estimate [m^2].
    covariance_geodetic : np.ndarray, shape (3, 3)
        Approximate geodetic covariance for `[lat, lon, h]`.
    effective_sample_size : float
        Effective sample size after the most recent normalization.
    predicted_disturbance_mps2 : float or None
        Weighted mean map-predicted gravity disturbance if available.
    """

    lat_rad: float
    lon_rad: float
    height_m: float
    covariance_ned_m2: FloatArray
    covariance_geodetic: FloatArray
    effective_sample_size: float
    predicted_disturbance_mps2: Optional[float] = None

    @property
    def geodetic_vector(self) -> FloatArray:
        """Return `[lat, lon, h]`."""
        return np.array(
            [self.lat_rad, self.lon_rad, self.height_m],
            dtype=np.float64,
        )


@dataclass
class MapMatchPFUpdateResult:
    """
    Diagnostics/result of one PF update.

    Attributes
    ----------
    estimate : ParticleCloudEstimate
        Posterior position estimate.
    effective_sample_size_before : float
        ESS before possible resampling.
    effective_sample_size_after : float
        ESS after update/resampling.
    resampled : bool
        True if resampling was performed.
    predicted_disturbance_mean_mps2 : float
        Weighted mean predicted map disturbance before update result packaging.
    predicted_disturbance_std_mps2 : float
        Weighted standard deviation of predicted map disturbance.
    time_s : float or None
        Optional timestamp.
    """

    estimate: ParticleCloudEstimate
    effective_sample_size_before: float
    effective_sample_size_after: float
    resampled: bool
    predicted_disturbance_mean_mps2: float
    predicted_disturbance_std_mps2: float
    time_s: Optional[float] = None


# -----------------------------------------------------------------------------
# PF implementation
# -----------------------------------------------------------------------------


class GravityMapParticleFilter:
    """
    Position-only gravity map-matching particle filter.

    State carried by the PF
    -----------------------
    - particle geodetic positions:
          lat_i, lon_i, h_i
    - normalized particle weights
    - optional last map-predicted gravity values for diagnostics

    Intended use
    ------------
    1) Initialize around the starting INS state.
    2) At each step:
       - propagate particles using INS-derived NED displacement + process noise
       - update weights using gravity disturbance measurements
       - optionally include depth and/or soft INS position prior
    3) Extract a weighted position estimate and, if desired, feed that back to
       the INS/fusion layer as a pseudo-position measurement.

    Important design choice
    -----------------------
    This filter estimates *position only*. The motion proposal comes from the INS,
    which keeps the particle state compact and makes the module easy to slot into
    the repository's later fusion and simulation code.
    """

    def __init__(
        self,
        spec: MapMatchPFSpec,
        map_model: Any,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.map_model = map_model
        self.rng = np.random.default_rng() if rng is None else rng

        self.lat_particles_rad = np.empty(self.spec.num_particles, dtype=np.float64)
        self.lon_particles_rad = np.empty(self.spec.num_particles, dtype=np.float64)
        self.height_particles_m = np.empty(self.spec.num_particles, dtype=np.float64)
        self.weights = np.full(
            self.spec.num_particles,
            1.0 / self.spec.num_particles,
            dtype=np.float64,
        )
        self._last_predicted_disturbance_mps2: Optional[FloatArray] = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Return True if the particle cloud has been initialized."""
        return bool(self._initialized)

    def reset_around_geodetic(
        self,
        lat_rad: float,
        lon_rad: float,
        height_m: float,
    ) -> None:
        """
        Initialize particles around a geodetic reference state using the configured
        local-NED Gaussian spread.
        """
        lat0 = float(lat_rad)
        lon0 = float(wrap_angle_pi(lon_rad))
        h0 = float(height_m)

        jitter_ned = (
            self.rng.standard_normal((self.spec.num_particles, 3), dtype=np.float64)
            * self.spec.init_position_std_m[None, :]
        )

        lat = np.full(self.spec.num_particles, lat0, dtype=np.float64)
        lon = np.full(self.spec.num_particles, lon0, dtype=np.float64)
        h = np.full(self.spec.num_particles, h0, dtype=np.float64)

        lat, lon, h = apply_ned_offsets_to_geodetic(
            lat,
            lon,
            h,
            jitter_ned,
        )

        self.lat_particles_rad = lat
        self.lon_particles_rad = lon
        self.height_particles_m = h
        self.weights.fill(1.0 / self.spec.num_particles)
        self._last_predicted_disturbance_mps2 = None
        self._initialized = True

    def reset_from_ins(
        self,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
    ) -> None:
        """
        Initialize particles around the current nominal INS state.
        """
        state = _state_from_filter_or_state(ins_or_state)
        self.reset_around_geodetic(
            lat_rad=state.nominal.lat_rad,
            lon_rad=state.nominal.lon_rad,
            height_m=state.nominal.height_m,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Particle filter is not initialized. Call reset_around_geodetic(...) "
                "or reset_from_ins(...) first."
            )

    def predict(
        self,
        delta_ned_m: ArrayLike,
        dt_s: float,
    ) -> None:
        r"""
        Propagate particles using an INS-derived local NED displacement plus
        Gaussian diffusion noise.

        Parameters
        ----------
        delta_ned_m : array-like, shape (3,)
            Mean local displacement `[dN, dE, dD]` [m] over the interval.
        dt_s : float
            Step duration [s].

        Model
        -----
        Each particle is propagated by:

            delta_i = delta_nominal + eta_i
            eta_i ~ N(0, diag((sigma_rw * sqrt(dt))^2))

        followed by first-order local curvilinear conversion to geodetic state.
        """
        self._require_initialized()

        dt = float(dt_s)
        if dt <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt}.")

        delta_nom = _vec3(delta_ned_m, name="delta_ned_m")
        sigma_step = self.spec.process_position_rw_std_m_per_sqrt_s * np.sqrt(dt)

        noise = (
            self.rng.standard_normal((self.spec.num_particles, 3), dtype=np.float64)
            * sigma_step[None, :]
        )
        delta = delta_nom[None, :] + noise

        self.lat_particles_rad, self.lon_particles_rad, self.height_particles_m = apply_ned_offsets_to_geodetic(
            self.lat_particles_rad,
            self.lon_particles_rad,
            self.height_particles_m,
            delta,
        )

        # Propagation does not change normalized weights.
        self._last_predicted_disturbance_mps2 = None

    def predict_from_ins(
        self,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        dt_s: float,
    ) -> None:
        """
        Propagate particles using the nominal INS velocity over one step:

            delta_ned ≈ v_ned * dt
        """
        state = _state_from_filter_or_state(ins_or_state)
        delta = state.nominal.v_ned_mps * float(dt_s)
        self.predict(delta_ned_m=delta, dt_s=dt_s)

    def _apply_soft_ins_position_prior(
        self,
        log_weights: FloatArray,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
    ) -> FloatArray:
        """
        Add an optional soft INS position-prior likelihood in local NED coordinates.
        """
        state = _state_from_filter_or_state(ins_or_state)
        d_ned = geodetic_offsets_to_local_ned(
            self.lat_particles_rad,
            self.lon_particles_rad,
            self.height_particles_m,
            lat_ref_rad=state.nominal.lat_rad,
            lon_ref_rad=state.nominal.lon_rad,
            height_ref_m=state.nominal.height_m,
        )
        return (
            log_weights
            + diagonal_gaussian_log_likelihood(
                d_ned,
                self.spec.ins_position_prior_std_m,
            )
        )

    def _resample_if_needed(self) -> tuple[float, float, bool]:
        """
        Resample the particle cloud when ESS falls below threshold.

        Returns
        -------
        tuple
            `(ess_before, ess_after, resampled)`
        """
        ess_before = effective_sample_size(self.weights)
        threshold = self.spec.resample_effective_fraction * self.spec.num_particles

        if ess_before >= threshold:
            return ess_before, ess_before, False

        idx = systematic_resample(self.weights, self.rng)
        self.lat_particles_rad = self.lat_particles_rad[idx]
        self.lon_particles_rad = self.lon_particles_rad[idx]
        self.height_particles_m = self.height_particles_m[idx]

        self.weights.fill(1.0 / self.spec.num_particles)

        # Optional rejuvenation to reduce sample impoverishment.
        if np.any(self.spec.rejuvenation_std_m > 0.0):
            jitter = (
                self.rng.standard_normal((self.spec.num_particles, 3), dtype=np.float64)
                * self.spec.rejuvenation_std_m[None, :]
            )
            (
                self.lat_particles_rad,
                self.lon_particles_rad,
                self.height_particles_m,
            ) = apply_ned_offsets_to_geodetic(
                self.lat_particles_rad,
                self.lon_particles_rad,
                self.height_particles_m,
                jitter,
            )

        if self._last_predicted_disturbance_mps2 is not None:
            self._last_predicted_disturbance_mps2 = self._last_predicted_disturbance_mps2[idx]

        ess_after = effective_sample_size(self.weights)
        return ess_before, ess_after, True

    def estimate(self) -> ParticleCloudEstimate:
        """
        Return the current weighted particle-cloud estimate.
        """
        self._require_initialized()

        lat_hat, lon_hat, h_hat = weighted_geodetic_mean(
            self.lat_particles_rad,
            self.lon_particles_rad,
            self.height_particles_m,
            self.weights,
        )

        d_ned = geodetic_offsets_to_local_ned(
            self.lat_particles_rad,
            self.lon_particles_rad,
            self.height_particles_m,
            lat_ref_rad=lat_hat,
            lon_ref_rad=lon_hat,
            height_ref_m=h_hat,
        )
        P_ned = weighted_covariance(d_ned, self.weights)
        P_geo = geodetic_covariance_from_ned_covariance(
            lat_ref_rad=lat_hat,
            height_ref_m=h_hat,
            ned_cov_m2=P_ned,
        )

        pred_g = None
        if self._last_predicted_disturbance_mps2 is not None:
            pred_g = float(np.sum(self.weights * self._last_predicted_disturbance_mps2))

        return ParticleCloudEstimate(
            lat_rad=float(lat_hat),
            lon_rad=float(lon_hat),
            height_m=float(h_hat),
            covariance_ned_m2=P_ned,
            covariance_geodetic=P_geo,
            effective_sample_size=effective_sample_size(self.weights),
            predicted_disturbance_mps2=pred_g,
        )

    def as_geodetic_pseudo_measurement(
        self,
        *,
        min_std_geodetic: ArrayLike | float = (0.0, 0.0, 0.0),
        covariance_inflation: float = 1.0,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Return the current PF estimate as a geodetic pseudo-measurement `(z, R)`.

        Parameters
        ----------
        min_std_geodetic : scalar or shape (3,), default=(0, 0, 0)
            Optional floor on the standard deviation of `[lat, lon, h]`.
        covariance_inflation : float, default=1.0
            Multiplicative covariance inflation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `z = [lat, lon, h]`, `R` geodetic covariance.

        Notes
        -----
        This is the intended bridge from the PF output back into the INS/fusion
        layer when you later want to treat the PF estimate as a pseudo-position
        update.
        """
        est = self.estimate()
        z = est.geodetic_vector.copy()
        R = _symmetrize(est.covariance_geodetic * float(covariance_inflation))

        floor_std = _axis3(min_std_geodetic, name="min_std_geodetic")
        floor_var = floor_std**2
        R[np.diag_indices(3)] = np.maximum(R[np.diag_indices(3)], floor_var)
        return z, R

    def update(
        self,
        measured_disturbance_mps2: float,
        *,
        gravity_meas_std_mps2: Optional[float] = None,
        ins_or_state: Optional[ErrorStateINS | ErrorStateINSState] = None,
        measured_depth_m: Optional[float] = None,
        depth_meas_std_m: Optional[float] = None,
        reference_surface_height_m: float = 0.0,
        time_s: Optional[float] = None,
    ) -> MapMatchPFUpdateResult:
        r"""
        Perform one gravity-map update.

        Parameters
        ----------
        measured_disturbance_mps2 : float
            Scalar gravity-disturbance measurement [m/s^2].
        gravity_meas_std_mps2 : float, optional
            Measurement standard deviation [m/s^2]. If omitted, the spec default
            is used.
        ins_or_state : ErrorStateINS or ErrorStateINSState, optional
            If provided and `use_ins_position_prior=True`, apply a soft INS
            position prior likelihood in local NED coordinates.
        measured_depth_m : float, optional
            Optional scalar signed-depth measurement [m].
        depth_meas_std_m : float, optional
            Optional depth standard deviation [m]. If omitted, `spec.depth_meas_std_m`
            is used. Required only when `measured_depth_m` is supplied.
        reference_surface_height_m : float, default=0.0
            Reference surface used for signed-depth interpretation:
                depth = h_ref - h
        time_s : float, optional
            Optional timestamp.

        Measurement model
        -----------------
        Gravity:
            z_g = g_map(lat, lon, h) + eps_g

        Optional depth:
            z_d = h_ref - h + eps_d

        Optional INS prior:
            p(x_k | INS_k) approximated as a diagonal Gaussian in local NED.

        Returns
        -------
        MapMatchPFUpdateResult
            Posterior summary and diagnostics.
        """
        self._require_initialized()

        sigma_g = self.spec.gravity_meas_std_mps2 if gravity_meas_std_mps2 is None else float(gravity_meas_std_mps2)
        if sigma_g <= 0.0:
            raise ValueError("gravity_meas_std_mps2 must be positive.")

        g_pred = evaluate_gravity_map_disturbance(
            self.map_model,
            self.lat_particles_rad,
            self.lon_particles_rad,
            self.height_particles_m,
        )
        self._last_predicted_disturbance_mps2 = g_pred.copy()

        logw = np.log(np.maximum(self.weights, 1.0e-300))
        logw += gaussian_log_likelihood_scalar(
            float(measured_disturbance_mps2) - g_pred,
            sigma=sigma_g,
        )

        if measured_depth_m is not None:
            sigma_d = (
                self.spec.depth_meas_std_m
                if depth_meas_std_m is None
                else float(depth_meas_std_m)
            )
            if sigma_d is None or sigma_d <= 0.0:
                raise ValueError(
                    "A positive depth standard deviation is required when "
                    "measured_depth_m is provided."
                )

            depth_pred = float(reference_surface_height_m) - self.height_particles_m
            logw += gaussian_log_likelihood_scalar(
                float(measured_depth_m) - depth_pred,
                sigma=sigma_d,
            )

        if self.spec.use_ins_position_prior and ins_or_state is not None:
            logw = self._apply_soft_ins_position_prior(logw, ins_or_state)

        self.weights = _normalize_log_weights(logw)
        ess_before, ess_after, resampled = self._resample_if_needed()

        estimate = self.estimate()

        g_mean = float(np.sum(self.weights * self._last_predicted_disturbance_mps2))
        g_var = float(np.sum(self.weights * (self._last_predicted_disturbance_mps2 - g_mean) ** 2))
        g_std = float(np.sqrt(max(g_var, 0.0)))

        return MapMatchPFUpdateResult(
            estimate=estimate,
            effective_sample_size_before=float(ess_before),
            effective_sample_size_after=float(ess_after),
            resampled=bool(resampled),
            predicted_disturbance_mean_mps2=g_mean,
            predicted_disturbance_std_mps2=g_std,
            time_s=None if time_s is None else float(time_s),
        )

    def update_from_gravimeter_measurement(
        self,
        measurement: GravimeterMeasurement,
        *,
        gravity_meas_std_mps2: Optional[float] = None,
        ins_or_state: Optional[ErrorStateINS | ErrorStateINSState] = None,
        depth_measurement: Optional[DepthMeasurement] = None,
        depth_meas_std_m: Optional[float] = None,
        reference_surface_height_m: Optional[float] = None,
    ) -> MapMatchPFUpdateResult:
        """
        Convenience wrapper for a gravimeter sensor-layer measurement.

        Requirements
        ------------
        The map-matching PF in this file expects a *disturbance* measurement,
        not absolute gravity. Therefore `measurement.kind` must be `"disturbance"`.
        """
        if measurement.kind != "disturbance":
            raise ValueError(
                "update_from_gravimeter_measurement(...) requires a disturbance "
                f"measurement, got kind={measurement.kind!r}."
            )

        depth_value = None
        href = 0.0
        time_s = measurement.time_s

        if depth_measurement is not None:
            depth_value = float(depth_measurement.value_m)
            href = (
                depth_measurement.reference_surface_height_m
                if reference_surface_height_m is None
                else float(reference_surface_height_m)
            )
            if time_s is None:
                time_s = depth_measurement.time_s
        else:
            href = 0.0 if reference_surface_height_m is None else float(reference_surface_height_m)

        return self.update(
            measured_disturbance_mps2=measurement.value_mps2,
            gravity_meas_std_mps2=gravity_meas_std_mps2,
            ins_or_state=ins_or_state,
            measured_depth_m=depth_value,
            depth_meas_std_m=depth_meas_std_m,
            reference_surface_height_m=href,
            time_s=time_s,
        )


__all__ = [
    "FloatArray",
    "GravityMapCallable",
    "GravityMapParticleFilter",
    "MapMatchPFSpec",
    "MapMatchPFUpdateResult",
    "ParticleCloudEstimate",
    "apply_ned_offsets_to_geodetic",
    "diagonal_gaussian_log_likelihood",
    "effective_sample_size",
    "evaluate_gravity_map_disturbance",
    "gaussian_log_likelihood_scalar",
    "geodetic_covariance_from_ned_covariance",
    "geodetic_offsets_to_local_ned",
    "systematic_resample",
    "weighted_covariance",
    "weighted_geodetic_mean",
]