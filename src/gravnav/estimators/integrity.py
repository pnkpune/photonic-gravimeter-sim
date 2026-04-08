"""
integrity.py

Integrity, consistency, and protection-level utilities for the gravity-aided
navigation simulator.

This module is the estimator-quality layer that sits on top of:
- `estimators.error_state_ins`
- `estimators.fusion`
- later map-matching outputs

Why this file exists
--------------------
The repository now has:
- nominal / error-state INS propagation
- measurement fusion helpers
- particle-filter gravity map matching

The next missing estimator concern is *trustworthiness of the navigation output*:
- Is the covariance consistent with the observed residuals?
- Is the estimated uncertainty small enough to satisfy an alert limit?
- Is the current filter behaving over-optimistically?
- If truth is available in simulation, are the actual errors covered by the
  predicted uncertainty?

This file answers those questions with a compact set of reusable tools:
1) NIS / NEES consistency metrics
2) chi-square consistency bounds
3) horizontal / vertical protection levels (HPL / VPL style)
4) simple alert-limit checks
5) compact monitor classes for logging integrity snapshots over time

Conventions
-----------
- Navigation frame is local NED:
      x = North, y = East, z = Down
- Geodetic state is:
      [lat, lon, h] = [rad, rad, m]
- Height is ellipsoidal height, positive upward.
- Local NED Down is positive, so:

      dD = -(h - h_ref)

- Position covariance can be represented either in:
  - geodetic coordinates: [lat, lon, h]
  - local NED coordinates: [dN, dE, dD]

Integrity concepts used here
----------------------------
This file intentionally uses a lightweight, simulation-friendly version of the
integrity concepts common in navigation:

- Alert Limit (AL):
    maximum tolerable position error before an alert should be raised
- Protection Level (PL):
    a conservative uncertainty bound around the reported navigation solution
- Horizontal Protection Level (HPL):
    horizontal bound derived from the covariance
- Vertical Protection Level (VPL):
    vertical bound derived from the covariance

This module does *not* claim to implement certified aviation RAIM/ARAIM logic.
Instead, it provides repository-friendly protection-level calculations that are
useful for:
- Monte Carlo studies
- comparing estimator configurations
- checking when a covariance-derived bound is tighter or looser than a chosen
  operational alert limit
- flagging overconfident filters in simulation

Consistency concepts used here
------------------------------
For filter consistency this file provides:
- NIS (Normalized Innovation Squared)
- NEES (Normalized Estimation Error Squared)

The intended usage is:
- NIS:
    when you have a residual and innovation covariance
- NEES:
    when you additionally have truth and can compare estimation error to the
    estimated covariance

Primary references used here
----------------------------
1) ESA Navipedia, "Integrity"
   Used for the operational integrity vocabulary:
   - Alert Limit
   - Integrity Risk
   - Protection Level
   - Time to Alert

2) FAA / WAAS-GPS terminology
   Used for the plain-language definition of vertical protection level as a
   bound around the indicated vertical position.

3) NASA filter-tuning references and common estimator practice
   Used for the role of chi-square statistics in evaluating covariance realism
   and filter consistency.

4) OpenVINS / standard estimator literature
   Used for the practical interpretation of NEES as a normalized state-error
   consistency metric.

Design notes
------------
- This module avoids hard-coding aviation certification constants.
- The user chooses the sigma-scaling / risk scaling used for protection levels.
- The chi-square quantiles are computed with a light-weight Wilson-Hilferty
  approximation so the file stays NumPy-only and does not require SciPy.
- The protection-level logic is intentionally explicit and auditable, making it
  suitable for debugging and Monte Carlo analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import meridian_radius, prime_vertical_radius
from ..physics.frames import wrap_angle_pi
from .error_state_ins import ERROR_STATE_SIZE, ErrorStateINS, ErrorStateINSState

FloatArray = NDArray[np.float64]

_POSITION_SLICE = slice(0, 3)


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


def _vec(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 1D vector."""
    arr = _as_float_array(x).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    return arr


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a shape-(3,) vector."""
    arr = _vec(x, name=name)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _mat(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 2D matrix."""
    arr = _as_float_array(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    return arr


def _square_mat(x: ArrayLike, *, name: str, n: Optional[int] = None) -> FloatArray:
    """Validate and return a square matrix."""
    arr = _mat(x, name=name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square, got shape {arr.shape}.")
    if n is not None and arr.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {arr.shape}.")
    return _symmetrize(arr)


def _symmetrize(M: ArrayLike) -> FloatArray:
    """Return the symmetric part of a square matrix."""
    A = _as_float_array(M)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}.")
    return 0.5 * (A + A.T)


def _state_from_filter_or_state(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> ErrorStateINSState:
    """Return an `ErrorStateINSState` regardless of wrapper type."""
    if isinstance(ins_or_state, ErrorStateINS):
        return ins_or_state.state
    if isinstance(ins_or_state, ErrorStateINSState):
        return ins_or_state
    raise TypeError(
        "Expected ErrorStateINS or ErrorStateINSState, "
        f"got {type(ins_or_state).__name__}."
    )


def _safe_cos_lat(lat_rad: float) -> float:
    """Cosine latitude with a floor away from zero for local linearization."""
    c = float(np.cos(lat_rad))
    if abs(c) < 1.0e-8:
        return 1.0e-8 if c >= 0.0 else -1.0e-8
    return c


# -----------------------------------------------------------------------------
# Chi-square / consistency helpers
# -----------------------------------------------------------------------------


def chi_square_quantile_approx(
    probability: float,
    dof: int,
) -> float:
    r"""
    Approximate the chi-square inverse CDF using the Wilson-Hilferty transform.

    Parameters
    ----------
    probability : float
        Probability in (0, 1).
    dof : int
        Degrees of freedom, must be positive.

    Returns
    -------
    float
        Approximate chi-square quantile.

    Approximation
    -------------
    For X ~ chi^2_k, the Wilson-Hilferty transform gives:

        X ≈ k * ( 1 - 2/(9k) + z * sqrt(2/(9k)) )^3

    where z is the standard-normal quantile at the requested probability.

    Notes
    -----
    This is sufficiently accurate for integrity-monitoring thresholds and
    simulation diagnostics, while keeping the module NumPy-only.
    """
    p = float(probability)
    k = int(dof)

    if not (0.0 < p < 1.0):
        raise ValueError(f"probability must lie in (0, 1), got {p}.")
    if k <= 0:
        raise ValueError(f"dof must be positive, got {k}.")

    z = NormalDist().inv_cdf(p)
    a = 1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))
    return float(max(k * (a**3), 0.0))


def chi_square_two_sided_bounds(
    dof: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    r"""
    Return approximate two-sided chi-square acceptance bounds.

    Parameters
    ----------
    dof : int
        Degrees of freedom.
    confidence : float, default=0.95
        Central confidence level in (0, 1).

    Returns
    -------
    tuple[float, float]
        `(lower, upper)` acceptance bounds.

    Notes
    -----
    For a two-sided consistency test with central confidence `c`:
        alpha = 1 - c
        lower = chi2_ppf(alpha / 2, dof)
        upper = chi2_ppf(1 - alpha / 2, dof)
    """
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError(f"confidence must lie in (0, 1), got {c}.")
    alpha = 1.0 - c
    lower = chi_square_quantile_approx(alpha / 2.0, dof)
    upper = chi_square_quantile_approx(1.0 - alpha / 2.0, dof)
    return lower, upper


def mahalanobis_squared(
    error: ArrayLike,
    covariance: ArrayLike,
) -> float:
    r"""
    Compute Mahalanobis squared distance.

    Formula
    -------
        d^2 = e^T P^{-1} e
    """
    e = _vec(error, name="error")
    P = _square_mat(covariance, name="covariance", n=e.size)
    return float(e.T @ np.linalg.inv(P) @ e)


def normalized_innovation_squared(
    residual: ArrayLike,
    innovation_covariance: ArrayLike,
) -> float:
    r"""
    Compute normalized innovation squared (NIS).

    Formula
    -------
        NIS = r^T S^{-1} r
    """
    r = _vec(residual, name="residual")
    S = _square_mat(innovation_covariance, name="innovation_covariance", n=r.size)
    return float(r.T @ np.linalg.inv(S) @ r)


def normalized_estimation_error_squared(
    estimation_error: ArrayLike,
    estimation_covariance: ArrayLike,
) -> float:
    r"""
    Compute normalized estimation error squared (NEES).

    Formula
    -------
        NEES = e^T P^{-1} e
    """
    e = _vec(estimation_error, name="estimation_error")
    P = _square_mat(estimation_covariance, name="estimation_covariance", n=e.size)
    return float(e.T @ np.linalg.inv(P) @ e)


# -----------------------------------------------------------------------------
# Position / geometry helpers
# -----------------------------------------------------------------------------


def geodetic_position_error_ned(
    estimated_lat_rad: float,
    estimated_lon_rad: float,
    estimated_height_m: float,
    true_lat_rad: float,
    true_lon_rad: float,
    true_height_m: float,
    *,
    reference_lat_rad: Optional[float] = None,
    reference_height_m: Optional[float] = None,
) -> FloatArray:
    r"""
    Convert geodetic position error into a local NED approximation.

    Parameters
    ----------
    estimated_lat_rad, estimated_lon_rad, estimated_height_m : float
        Estimated geodetic state.
    true_lat_rad, true_lon_rad, true_height_m : float
        Truth geodetic state.
    reference_lat_rad : float, optional
        Reference latitude for local linearization. Defaults to truth latitude.
    reference_height_m : float, optional
        Reference height for local linearization. Defaults to truth height.

    Returns
    -------
    np.ndarray, shape (3,)
        `[dN, dE, dD]` error in metres.

    Formula
    -------
    Using the standard small local linearization about `(phi_ref, h_ref)`:

        dN ≈ (phi_est - phi_true) (R_M + h_ref)
        dE ≈ (lambda_est - lambda_true) (R_N + h_ref) cos(phi_ref)
        dD = -(h_est - h_true)
    """
    phi_ref = float(true_lat_rad if reference_lat_rad is None else reference_lat_rad)
    h_ref = float(true_height_m if reference_height_m is None else reference_height_m)

    M = float(meridian_radius(phi_ref))
    N = float(prime_vertical_radius(phi_ref))
    cphi = _safe_cos_lat(phi_ref)

    dphi = float(estimated_lat_rad) - float(true_lat_rad)
    dlam = float(wrap_angle_pi(float(estimated_lon_rad) - float(true_lon_rad)))
    dh = float(estimated_height_m) - float(true_height_m)

    return np.array(
        [
            dphi * (M + h_ref),
            dlam * (N + h_ref) * cphi,
            -dh,
        ],
        dtype=np.float64,
    )


def geodetic_covariance_to_ned(
    geodetic_covariance: ArrayLike,
    *,
    lat_ref_rad: float,
    height_ref_m: float,
) -> FloatArray:
    r"""
    Convert a small geodetic covariance `[lat, lon, h]` into local NED covariance.

    Parameters
    ----------
    geodetic_covariance : array-like, shape (3, 3)
        Covariance of `[d_lat, d_lon, d_h]`.
    lat_ref_rad : float
        Reference latitude [rad].
    height_ref_m : float
        Reference height [m].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Covariance of `[dN, dE, dD]` in metres.

    Linear mapping
    --------------
        dN = (R_M + h) d_lat
        dE = (R_N + h) cos(lat) d_lon
        dD = -d_h
    """
    P_geo = _square_mat(geodetic_covariance, name="geodetic_covariance", n=3)

    phi = float(lat_ref_rad)
    h = float(height_ref_m)
    M = float(meridian_radius(phi))
    N = float(prime_vertical_radius(phi))
    cphi = _safe_cos_lat(phi)

    J = np.array(
        [
            [M + h, 0.0, 0.0],
            [0.0, (N + h) * cphi, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return _symmetrize(J @ P_geo @ J.T)


def ins_position_covariance_geodetic(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> FloatArray:
    """
    Extract the position covariance block `[lat, lon, h]` from the INS state.
    """
    state = _state_from_filter_or_state(ins_or_state)
    return _square_mat(
        state.P[_POSITION_SLICE, _POSITION_SLICE],
        name="position_covariance_geodetic",
        n=3,
    )


def ins_position_covariance_ned(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> FloatArray:
    """
    Extract the INS position covariance as local NED covariance in metres.
    """
    state = _state_from_filter_or_state(ins_or_state)
    P_geo = ins_position_covariance_geodetic(state)
    return geodetic_covariance_to_ned(
        P_geo,
        lat_ref_rad=state.nominal.lat_rad,
        height_ref_m=state.nominal.height_m,
    )


def horizontal_position_error(error_ned_m: ArrayLike) -> float:
    """
    Horizontal position error magnitude from a NED error vector.
    """
    e = _vec3(error_ned_m, name="error_ned_m")
    return float(np.linalg.norm(e[:2]))


def vertical_position_error(error_ned_m: ArrayLike) -> float:
    """
    Vertical position error magnitude from a NED error vector.

    Returns
    -------
    float
        Absolute vertical error in metres, using the Down component magnitude.
    """
    e = _vec3(error_ned_m, name="error_ned_m")
    return float(abs(e[2]))


# -----------------------------------------------------------------------------
# Protection levels
# -----------------------------------------------------------------------------


def horizontal_protection_level_from_covariance_ned(
    covariance_ned_m2: ArrayLike,
    *,
    k_sigma: float = 6.0,
) -> float:
    r"""
    Compute a conservative horizontal protection level (HPL) from NED covariance.

    Parameters
    ----------
    covariance_ned_m2 : array-like, shape (3, 3)
        Local NED covariance [m^2].
    k_sigma : float, default=6.0
        Sigma multiplier used to convert the 1-sigma horizontal bound into a
        protection level.

    Returns
    -------
    float
        Horizontal protection level [m].

    Model
    -----
    Let `P_h` be the 2x2 horizontal covariance over `[N, E]`. This function uses
    the worst-case 1-sigma horizontal axis:

        sigma_h = sqrt(lambda_max(P_h))
        HPL = k_sigma * sigma_h

    Notes
    -----
    This is a conservative simulation-friendly bound, not a certified aviation
    RAIM formula.
    """
    P = _square_mat(covariance_ned_m2, name="covariance_ned_m2", n=3)
    k = float(k_sigma)
    if k < 0.0:
        raise ValueError(f"k_sigma must be nonnegative, got {k}.")

    Ph = P[:2, :2]
    lam_max = float(np.max(np.linalg.eigvalsh(_symmetrize(Ph))))
    lam_max = max(lam_max, 0.0)
    sigma_h = float(np.sqrt(lam_max))
    return float(k * sigma_h)


def vertical_protection_level_from_covariance_ned(
    covariance_ned_m2: ArrayLike,
    *,
    k_sigma: float = 6.0,
) -> float:
    r"""
    Compute a vertical protection level (VPL) from NED covariance.

    Parameters
    ----------
    covariance_ned_m2 : array-like, shape (3, 3)
        Local NED covariance [m^2].
    k_sigma : float, default=6.0
        Sigma multiplier.

    Returns
    -------
    float
        Vertical protection level [m].

    Model
    -----
        sigma_v = sqrt(P_DD)
        VPL = k_sigma * sigma_v
    """
    P = _square_mat(covariance_ned_m2, name="covariance_ned_m2", n=3)
    k = float(k_sigma)
    if k < 0.0:
        raise ValueError(f"k_sigma must be nonnegative, got {k}.")
    sigma_v = float(np.sqrt(max(float(P[2, 2]), 0.0)))
    return float(k * sigma_v)


def radial_protection_level_from_covariance_ned(
    covariance_ned_m2: ArrayLike,
    *,
    k_sigma: float = 6.0,
) -> float:
    r"""
    Compute a 3D radial protection level from NED covariance.

    Model
    -----
    Uses the worst principal axis of the full 3x3 covariance:

        sigma_r = sqrt(lambda_max(P))
        RPL = k_sigma * sigma_r
    """
    P = _square_mat(covariance_ned_m2, name="covariance_ned_m2", n=3)
    k = float(k_sigma)
    if k < 0.0:
        raise ValueError(f"k_sigma must be nonnegative, got {k}.")
    lam_max = float(np.max(np.linalg.eigvalsh(P)))
    sigma_r = float(np.sqrt(max(lam_max, 0.0)))
    return float(k * sigma_r)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ChiSquareConsistencyResult:
    """
    Result of a chi-square consistency test.

    Attributes
    ----------
    name : str
        Human-readable statistic name, e.g. "nis" or "nees".
    statistic : float
        Observed test statistic.
    dof : int
        Degrees of freedom.
    confidence : float
        Central confidence level used for the two-sided bounds.
    lower_bound : float
        Lower acceptance bound.
    upper_bound : float
        Upper acceptance bound.
    passed : bool
        True if the statistic falls inside `[lower_bound, upper_bound]`.
    """

    name: str
    statistic: float
    dof: int
    confidence: float
    lower_bound: float
    upper_bound: float
    passed: bool


@dataclass
class ProtectionLevels:
    """
    Protection-level summary.

    Attributes
    ----------
    horizontal_m : float
        Horizontal protection level [m].
    vertical_m : float
        Vertical protection level [m].
    radial_3d_m : float
        3D radial protection level [m].
    horizontal_alert_limit_m : float or None
        Horizontal alert limit [m], if provided.
    vertical_alert_limit_m : float or None
        Vertical alert limit [m], if provided.
    horizontal_within_alert_limit : bool or None
        True if HPL <= HAL when HAL is supplied.
    vertical_within_alert_limit : bool or None
        True if VPL <= VAL when VAL is supplied.
    """

    horizontal_m: float
    vertical_m: float
    radial_3d_m: float
    horizontal_alert_limit_m: Optional[float] = None
    vertical_alert_limit_m: Optional[float] = None
    horizontal_within_alert_limit: Optional[bool] = None
    vertical_within_alert_limit: Optional[bool] = None


@dataclass
class IntegritySnapshot:
    """
    One integrity-monitor snapshot.

    Attributes
    ----------
    time_s : float or None
        Optional timestamp [s].
    protection_levels : ProtectionLevels
        Covariance-derived protection levels.
    position_error_ned_m : np.ndarray, shape (3,), optional
        Truth-based local NED position error if truth is available.
    horizontal_error_m : float or None
        Truth-based horizontal error magnitude if available.
    vertical_error_m : float or None
        Truth-based vertical error magnitude if available.
    nis_result : ChiSquareConsistencyResult or None
        Innovation consistency result if evaluated.
    nees_result : ChiSquareConsistencyResult or None
        Estimation consistency result if evaluated.
    hazardously_misleading_horizontal : bool or None
        True when actual horizontal error exceeds HAL while HPL <= HAL.
    hazardously_misleading_vertical : bool or None
        True when actual vertical error exceeds VAL while VPL <= VAL.
    """

    time_s: Optional[float]
    protection_levels: ProtectionLevels
    position_error_ned_m: Optional[FloatArray] = None
    horizontal_error_m: Optional[float] = None
    vertical_error_m: Optional[float] = None
    nis_result: Optional[ChiSquareConsistencyResult] = None
    nees_result: Optional[ChiSquareConsistencyResult] = None
    hazardously_misleading_horizontal: Optional[bool] = None
    hazardously_misleading_vertical: Optional[bool] = None


@dataclass
class IntegrityHistory:
    """
    Collected integrity snapshots.

    This is a light-weight container that later simulation/plotting modules can
    use directly.
    """

    snapshots: list[IntegritySnapshot] = field(default_factory=list)

    def append(self, snapshot: IntegritySnapshot) -> None:
        """Append one snapshot."""
        self.snapshots.append(snapshot)

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def times_s(self) -> FloatArray:
        """Return available snapshot times."""
        times = [s.time_s for s in self.snapshots if s.time_s is not None]
        return np.asarray(times, dtype=np.float64)

    @property
    def horizontal_protection_levels_m(self) -> FloatArray:
        """Return the HPL history."""
        return np.asarray(
            [s.protection_levels.horizontal_m for s in self.snapshots],
            dtype=np.float64,
        )

    @property
    def vertical_protection_levels_m(self) -> FloatArray:
        """Return the VPL history."""
        return np.asarray(
            [s.protection_levels.vertical_m for s in self.snapshots],
            dtype=np.float64,
        )

    @property
    def horizontal_errors_m(self) -> FloatArray:
        """Return the available horizontal-error history."""
        vals = [s.horizontal_error_m for s in self.snapshots if s.horizontal_error_m is not None]
        return np.asarray(vals, dtype=np.float64)

    @property
    def vertical_errors_m(self) -> FloatArray:
        """Return the available vertical-error history."""
        vals = [s.vertical_error_m for s in self.snapshots if s.vertical_error_m is not None]
        return np.asarray(vals, dtype=np.float64)

    @property
    def nis_values(self) -> FloatArray:
        """Return available NIS history."""
        vals = [s.nis_result.statistic for s in self.snapshots if s.nis_result is not None]
        return np.asarray(vals, dtype=np.float64)

    @property
    def nees_values(self) -> FloatArray:
        """Return available NEES history."""
        vals = [s.nees_result.statistic for s in self.snapshots if s.nees_result is not None]
        return np.asarray(vals, dtype=np.float64)


# -----------------------------------------------------------------------------
# Consistency result builders
# -----------------------------------------------------------------------------


def chi_square_consistency_result(
    statistic: float,
    dof: int,
    *,
    confidence: float = 0.95,
    name: str = "chi_square_statistic",
) -> ChiSquareConsistencyResult:
    """
    Build a `ChiSquareConsistencyResult` from a scalar statistic.
    """
    stat = float(statistic)
    k = int(dof)
    if k <= 0:
        raise ValueError(f"dof must be positive, got {k}.")

    lower, upper = chi_square_two_sided_bounds(k, confidence=confidence)
    passed = bool(lower <= stat <= upper)
    return ChiSquareConsistencyResult(
        name=str(name),
        statistic=stat,
        dof=k,
        confidence=float(confidence),
        lower_bound=float(lower),
        upper_bound=float(upper),
        passed=passed,
    )


def nis_consistency_result(
    residual: ArrayLike,
    innovation_covariance: ArrayLike,
    *,
    confidence: float = 0.95,
    name: str = "nis",
) -> ChiSquareConsistencyResult:
    """
    Evaluate innovation consistency using NIS.
    """
    r = _vec(residual, name="residual")
    S = _square_mat(innovation_covariance, name="innovation_covariance", n=r.size)
    stat = normalized_innovation_squared(r, S)
    return chi_square_consistency_result(
        stat,
        r.size,
        confidence=confidence,
        name=name,
    )


def nees_consistency_result(
    estimation_error: ArrayLike,
    estimation_covariance: ArrayLike,
    *,
    confidence: float = 0.95,
    name: str = "nees",
) -> ChiSquareConsistencyResult:
    """
    Evaluate state-estimation consistency using NEES.
    """
    e = _vec(estimation_error, name="estimation_error")
    P = _square_mat(estimation_covariance, name="estimation_covariance", n=e.size)
    stat = normalized_estimation_error_squared(e, P)
    return chi_square_consistency_result(
        stat,
        e.size,
        confidence=confidence,
        name=name,
    )


# -----------------------------------------------------------------------------
# High-level integrity evaluation helpers
# -----------------------------------------------------------------------------


def protection_levels_from_covariance_ned(
    covariance_ned_m2: ArrayLike,
    *,
    horizontal_k_sigma: float = 6.0,
    vertical_k_sigma: float = 6.0,
    radial_k_sigma: float = 6.0,
    horizontal_alert_limit_m: Optional[float] = None,
    vertical_alert_limit_m: Optional[float] = None,
) -> ProtectionLevels:
    """
    Build a protection-level summary from local NED covariance.
    """
    P = _square_mat(covariance_ned_m2, name="covariance_ned_m2", n=3)

    hpl = horizontal_protection_level_from_covariance_ned(
        P,
        k_sigma=horizontal_k_sigma,
    )
    vpl = vertical_protection_level_from_covariance_ned(
        P,
        k_sigma=vertical_k_sigma,
    )
    rpl = radial_protection_level_from_covariance_ned(
        P,
        k_sigma=radial_k_sigma,
    )

    h_ok = None
    v_ok = None
    if horizontal_alert_limit_m is not None:
        hal = float(horizontal_alert_limit_m)
        if hal < 0.0:
            raise ValueError("horizontal_alert_limit_m must be nonnegative.")
        h_ok = bool(hpl <= hal)
    if vertical_alert_limit_m is not None:
        val = float(vertical_alert_limit_m)
        if val < 0.0:
            raise ValueError("vertical_alert_limit_m must be nonnegative.")
        v_ok = bool(vpl <= val)

    return ProtectionLevels(
        horizontal_m=hpl,
        vertical_m=vpl,
        radial_3d_m=rpl,
        horizontal_alert_limit_m=horizontal_alert_limit_m,
        vertical_alert_limit_m=vertical_alert_limit_m,
        horizontal_within_alert_limit=h_ok,
        vertical_within_alert_limit=v_ok,
    )


def integrity_snapshot_from_covariance_ned(
    covariance_ned_m2: ArrayLike,
    *,
    position_error_ned_m: Optional[ArrayLike] = None,
    horizontal_alert_limit_m: Optional[float] = None,
    vertical_alert_limit_m: Optional[float] = None,
    horizontal_k_sigma: float = 6.0,
    vertical_k_sigma: float = 6.0,
    radial_k_sigma: float = 6.0,
    nis_result: Optional[ChiSquareConsistencyResult] = None,
    nees_result: Optional[ChiSquareConsistencyResult] = None,
    time_s: Optional[float] = None,
) -> IntegritySnapshot:
    """
    Build an integrity snapshot from a local NED covariance, optionally with
    truth-based position error and consistency test results.
    """
    P_ned = _square_mat(covariance_ned_m2, name="covariance_ned_m2", n=3)
    pls = protection_levels_from_covariance_ned(
        P_ned,
        horizontal_k_sigma=horizontal_k_sigma,
        vertical_k_sigma=vertical_k_sigma,
        radial_k_sigma=radial_k_sigma,
        horizontal_alert_limit_m=horizontal_alert_limit_m,
        vertical_alert_limit_m=vertical_alert_limit_m,
    )

    err = None
    h_err = None
    v_err = None
    hmi_h = None
    hmi_v = None

    if position_error_ned_m is not None:
        err = _vec3(position_error_ned_m, name="position_error_ned_m")
        h_err = horizontal_position_error(err)
        v_err = vertical_position_error(err)

        if horizontal_alert_limit_m is not None:
            hal = float(horizontal_alert_limit_m)
            hmi_h = bool((h_err > hal) and (pls.horizontal_m <= hal))

        if vertical_alert_limit_m is not None:
            val = float(vertical_alert_limit_m)
            hmi_v = bool((v_err > val) and (pls.vertical_m <= val))

    return IntegritySnapshot(
        time_s=None if time_s is None else float(time_s),
        protection_levels=pls,
        position_error_ned_m=err,
        horizontal_error_m=h_err,
        vertical_error_m=v_err,
        nis_result=nis_result,
        nees_result=nees_result,
        hazardously_misleading_horizontal=hmi_h,
        hazardously_misleading_vertical=hmi_v,
    )


def integrity_snapshot_from_ins(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    *,
    true_lat_rad: Optional[float] = None,
    true_lon_rad: Optional[float] = None,
    true_height_m: Optional[float] = None,
    horizontal_alert_limit_m: Optional[float] = None,
    vertical_alert_limit_m: Optional[float] = None,
    horizontal_k_sigma: float = 6.0,
    vertical_k_sigma: float = 6.0,
    radial_k_sigma: float = 6.0,
    nis_residual: Optional[ArrayLike] = None,
    innovation_covariance: Optional[ArrayLike] = None,
    nees_error_override: Optional[ArrayLike] = None,
    nees_covariance_override: Optional[ArrayLike] = None,
    consistency_confidence: float = 0.95,
    time_s: Optional[float] = None,
) -> IntegritySnapshot:
    """
    Build an integrity snapshot from an INS state.

    Parameters
    ----------
    ins_or_state : ErrorStateINS or ErrorStateINSState
        INS filter or state.
    true_lat_rad, true_lon_rad, true_height_m : float, optional
        Truth geodetic position. If all three are supplied, the snapshot includes
        truth-based local NED position error.
    horizontal_alert_limit_m, vertical_alert_limit_m : float, optional
        Optional alert limits.
    horizontal_k_sigma, vertical_k_sigma, radial_k_sigma : float, default=6.0
        Protection-level sigma multipliers.
    nis_residual : array-like, optional
        Residual vector for an optional NIS test.
    innovation_covariance : array-like, optional
        Innovation covariance for the NIS test.
    nees_error_override : array-like, optional
        Optional state-error vector for NEES testing. If omitted and truth
        position is available, a position-only NEES is computed from the 3D
        local NED position error.
    nees_covariance_override : array-like, optional
        Optional covariance matrix matching `nees_error_override`. If omitted and
        truth position is available, the INS position covariance in local NED is
        used for a position-only NEES.
    consistency_confidence : float, default=0.95
        Confidence level for NIS/NEES chi-square bounds.
    time_s : float, optional
        Optional timestamp.

    Returns
    -------
    IntegritySnapshot
        Snapshot containing protection levels and optional consistency results.
    """
    state = _state_from_filter_or_state(ins_or_state)
    P_ned = ins_position_covariance_ned(state)

    pos_err_ned = None
    if (
        true_lat_rad is not None
        and true_lon_rad is not None
        and true_height_m is not None
    ):
        pos_err_ned = geodetic_position_error_ned(
            estimated_lat_rad=state.nominal.lat_rad,
            estimated_lon_rad=state.nominal.lon_rad,
            estimated_height_m=state.nominal.height_m,
            true_lat_rad=float(true_lat_rad),
            true_lon_rad=float(true_lon_rad),
            true_height_m=float(true_height_m),
        )

    nis_res = None
    if (nis_residual is None) ^ (innovation_covariance is None):
        raise ValueError(
            "nis_residual and innovation_covariance must either both be provided "
            "or both be omitted."
        )
    if nis_residual is not None and innovation_covariance is not None:
        nis_res = nis_consistency_result(
            nis_residual,
            innovation_covariance,
            confidence=consistency_confidence,
            name="nis",
        )

    nees_res = None
    if (nees_error_override is None) ^ (nees_covariance_override is None):
        raise ValueError(
            "nees_error_override and nees_covariance_override must either both be "
            "provided or both be omitted."
        )

    if nees_error_override is not None and nees_covariance_override is not None:
        nees_res = nees_consistency_result(
            nees_error_override,
            nees_covariance_override,
            confidence=consistency_confidence,
            name="nees",
        )
    elif pos_err_ned is not None:
        nees_res = nees_consistency_result(
            pos_err_ned,
            P_ned,
            confidence=consistency_confidence,
            name="position_nees",
        )

    return integrity_snapshot_from_covariance_ned(
        P_ned,
        position_error_ned_m=pos_err_ned,
        horizontal_alert_limit_m=horizontal_alert_limit_m,
        vertical_alert_limit_m=vertical_alert_limit_m,
        horizontal_k_sigma=horizontal_k_sigma,
        vertical_k_sigma=vertical_k_sigma,
        radial_k_sigma=radial_k_sigma,
        nis_result=nis_res,
        nees_result=nees_res,
        time_s=time_s,
    )


# -----------------------------------------------------------------------------
# Monitor class
# -----------------------------------------------------------------------------


class IntegrityMonitor:
    """
    Lightweight stateful integrity monitor.

    This class is meant for simulation loops and Monte Carlo runs where you want
    to accumulate protection levels and consistency metrics over time.

    Typical usage
    -------------
    1) Construct the monitor with chosen alert limits and sigma multipliers.
    2) After each update/step, call:
         - `snapshot_from_ins(...)`, or
         - `snapshot_from_covariance_ned(...)`
    3) Use the returned snapshot immediately and/or keep the internal history for
       plotting and summary metrics later.
    """

    def __init__(
        self,
        *,
        horizontal_alert_limit_m: Optional[float] = None,
        vertical_alert_limit_m: Optional[float] = None,
        horizontal_k_sigma: float = 6.0,
        vertical_k_sigma: float = 6.0,
        radial_k_sigma: float = 6.0,
        consistency_confidence: float = 0.95,
    ) -> None:
        if horizontal_alert_limit_m is not None and float(horizontal_alert_limit_m) < 0.0:
            raise ValueError("horizontal_alert_limit_m must be nonnegative.")
        if vertical_alert_limit_m is not None and float(vertical_alert_limit_m) < 0.0:
            raise ValueError("vertical_alert_limit_m must be nonnegative.")
        if horizontal_k_sigma < 0.0 or vertical_k_sigma < 0.0 or radial_k_sigma < 0.0:
            raise ValueError("sigma multipliers must be nonnegative.")
        if not (0.0 < float(consistency_confidence) < 1.0):
            raise ValueError("consistency_confidence must lie in (0, 1).")

        self.horizontal_alert_limit_m = horizontal_alert_limit_m
        self.vertical_alert_limit_m = vertical_alert_limit_m
        self.horizontal_k_sigma = float(horizontal_k_sigma)
        self.vertical_k_sigma = float(vertical_k_sigma)
        self.radial_k_sigma = float(radial_k_sigma)
        self.consistency_confidence = float(consistency_confidence)
        self.history = IntegrityHistory()

    def reset(self) -> None:
        """Clear the accumulated history."""
        self.history = IntegrityHistory()

    def snapshot_from_covariance_ned(
        self,
        covariance_ned_m2: ArrayLike,
        *,
        position_error_ned_m: Optional[ArrayLike] = None,
        nis_result: Optional[ChiSquareConsistencyResult] = None,
        nees_result: Optional[ChiSquareConsistencyResult] = None,
        time_s: Optional[float] = None,
    ) -> IntegritySnapshot:
        """
        Build, store, and return one snapshot from local NED covariance.
        """
        snap = integrity_snapshot_from_covariance_ned(
            covariance_ned_m2,
            position_error_ned_m=position_error_ned_m,
            horizontal_alert_limit_m=self.horizontal_alert_limit_m,
            vertical_alert_limit_m=self.vertical_alert_limit_m,
            horizontal_k_sigma=self.horizontal_k_sigma,
            vertical_k_sigma=self.vertical_k_sigma,
            radial_k_sigma=self.radial_k_sigma,
            nis_result=nis_result,
            nees_result=nees_result,
            time_s=time_s,
        )
        self.history.append(snap)
        return snap

    def snapshot_from_ins(
        self,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        *,
        true_lat_rad: Optional[float] = None,
        true_lon_rad: Optional[float] = None,
        true_height_m: Optional[float] = None,
        nis_residual: Optional[ArrayLike] = None,
        innovation_covariance: Optional[ArrayLike] = None,
        nees_error_override: Optional[ArrayLike] = None,
        nees_covariance_override: Optional[ArrayLike] = None,
        time_s: Optional[float] = None,
    ) -> IntegritySnapshot:
        """
        Build, store, and return one snapshot from an INS state.
        """
        snap = integrity_snapshot_from_ins(
            ins_or_state,
            true_lat_rad=true_lat_rad,
            true_lon_rad=true_lon_rad,
            true_height_m=true_height_m,
            horizontal_alert_limit_m=self.horizontal_alert_limit_m,
            vertical_alert_limit_m=self.vertical_alert_limit_m,
            horizontal_k_sigma=self.horizontal_k_sigma,
            vertical_k_sigma=self.vertical_k_sigma,
            radial_k_sigma=self.radial_k_sigma,
            nis_residual=nis_residual,
            innovation_covariance=innovation_covariance,
            nees_error_override=nees_error_override,
            nees_covariance_override=nees_covariance_override,
            consistency_confidence=self.consistency_confidence,
            time_s=time_s,
        )
        self.history.append(snap)
        return snap


__all__ = [
    "FloatArray",
    "ChiSquareConsistencyResult",
    "IntegrityHistory",
    "IntegrityMonitor",
    "IntegritySnapshot",
    "ProtectionLevels",
    "chi_square_consistency_result",
    "chi_square_quantile_approx",
    "chi_square_two_sided_bounds",
    "geodetic_covariance_to_ned",
    "geodetic_position_error_ned",
    "horizontal_position_error",
    "horizontal_protection_level_from_covariance_ned",
    "ins_position_covariance_geodetic",
    "ins_position_covariance_ned",
    "integrity_snapshot_from_covariance_ned",
    "integrity_snapshot_from_ins",
    "mahalanobis_squared",
    "nees_consistency_result",
    "nis_consistency_result",
    "normalized_estimation_error_squared",
    "normalized_innovation_squared",
    "protection_levels_from_covariance_ned",
    "radial_protection_level_from_covariance_ned",
    "vertical_position_error",
    "vertical_protection_level_from_covariance_ned",
]