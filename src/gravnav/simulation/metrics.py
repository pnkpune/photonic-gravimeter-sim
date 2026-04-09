"""
metrics.py

Simulation metrics and estimator-performance summaries for the gravity-aided
navigation simulator.

Why this file exists
--------------------
The repository already contains:
- truth trajectories
- sensor models that expose ideal and corrupted measurements
- INS, fusion, map-matching, and integrity layers
- a simulation results container intended to aggregate one full run

What is still missing is the "how good was the run?" layer.

This module therefore provides:
1) reusable scalar/vector error metrics
2) NED position-error metrics derived from truth vs estimator histories
3) integrity-history summaries (NIS/NEES pass rates, protection levels, HMI flags)
4) one top-level scenario-summary container for a full simulation run

Conventions
-----------
- Angles are radians.
- Distances are metres.
- Velocities are m/s.
- Accelerations and gravity are m/s^2.
- Position-error vectors are in local NED coordinates:
      [dN, dE, dD]
- Horizontal error means sqrt(dN^2 + dE^2).
- Vertical error means |dD|.
- Radial error means sqrt(dN^2 + dE^2 + dD^2).

Design notes
------------
- This file intentionally stays NumPy-only.
- Metric builders are tolerant of NaNs and will ignore non-finite values where
  that is mathematically reasonable.
- The top-level `ScenarioMetricsSummary` is meant for reporting, benchmarking,
  and Monte Carlo aggregation. It is not a plotting layer and does not own any
  visualization logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.error_state_ins import ErrorStateINSState
from ..estimators.gravity_sequence_match import SequenceMatchUpdateResult
from ..estimators.integrity import IntegritySnapshot, geodetic_position_error_ned
from ..estimators.map_match_pf import MapMatchPFUpdateResult
from ..truth.trajectory import TruthTrajectory
from .results import ScenarioSimulationResult

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _vec(x: ArrayLike, *, name: str) -> FloatArray:
    """Return a flattened vector."""
    arr = _as_float_array(x).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be vector-like.")
    return arr


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a 3-vector."""
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _mat_nxm(x: ArrayLike, *, ncol: int, name: str) -> FloatArray:
    """Validate and return a 2D array with `ncol` columns."""
    arr = _as_float_array(x)
    if arr.ndim != 2 or arr.shape[1] != ncol:
        raise ValueError(
            f"{name} must have shape (N, {ncol}), got {arr.shape}."
        )
    return arr


def _finite_mask(x: ArrayLike) -> NDArray[np.bool_]:
    """Return a finite-value mask."""
    return np.isfinite(_as_float_array(x))


def _finite_values_1d(x: ArrayLike) -> FloatArray:
    """Return only finite values from a 1D array-like input."""
    arr = _vec(x, name="x")
    return arr[np.isfinite(arr)]


def _nanmean(x: ArrayLike) -> float:
    """Mean ignoring NaNs, or NaN if no finite values exist."""
    vals = _finite_values_1d(x)
    return float(np.mean(vals)) if vals.size > 0 else float(np.nan)


def _nanstd(x: ArrayLike) -> float:
    """Standard deviation ignoring NaNs, or NaN if no finite values exist."""
    vals = _finite_values_1d(x)
    return float(np.std(vals)) if vals.size > 0 else float(np.nan)


def _nanpercentile(x: ArrayLike, q: float) -> float:
    """Percentile ignoring NaNs, or NaN if no finite values exist."""
    vals = _finite_values_1d(x)
    return float(np.percentile(vals, q)) if vals.size > 0 else float(np.nan)


def _count_finite(x: ArrayLike) -> int:
    """Count finite entries in a 1D array-like input."""
    return int(_finite_values_1d(x).size)


def _fraction_true(x: Sequence[bool | None]) -> float:
    """
    Fraction of True values over the subset that is not None.

    Returns NaN if the input contains no non-None values.
    """
    filtered = [bool(v) for v in x if v is not None]
    if len(filtered) == 0:
        return float(np.nan)
    return float(np.mean(filtered))


def _error_series(observed: ArrayLike, reference: ArrayLike) -> FloatArray:
    """
    Compute `observed - reference` with broadcasting preserved.
    """
    obs = _as_float_array(observed)
    ref = _as_float_array(reference)
    return np.asarray(obs - ref, dtype=np.float64)


# -----------------------------------------------------------------------------
# Basic scalar/vector metrics
# -----------------------------------------------------------------------------


def mean_error(error: ArrayLike) -> float:
    r"""
    Mean signed error.

    Formula
    -------
        mean(e) = (1/N) sum_i e_i
    """
    return _nanmean(error)


def root_mean_square_error(error: ArrayLike) -> float:
    r"""
    Root-mean-square error (RMSE).

    Formula
    -------
        RMSE = sqrt( mean( e^2 ) )

    Notes
    -----
    The output has the same physical units as the error itself.
    """
    vals = _finite_values_1d(error)
    if vals.size == 0:
        return float(np.nan)
    return float(np.sqrt(np.mean(vals ** 2)))


def mean_absolute_error(error: ArrayLike) -> float:
    r"""
    Mean absolute error (MAE).

    Formula
    -------
        MAE = mean( |e| )
    """
    vals = _finite_values_1d(error)
    if vals.size == 0:
        return float(np.nan)
    return float(np.mean(np.abs(vals)))


def median_absolute_error(error: ArrayLike) -> float:
    r"""
    Median absolute error.

    Formula
    -------
        MedAE = median( |e| )
    """
    vals = _finite_values_1d(error)
    if vals.size == 0:
        return float(np.nan)
    return float(np.median(np.abs(vals)))


def max_absolute_error(error: ArrayLike) -> float:
    r"""
    Maximum absolute error.

    Formula
    -------
        max_i |e_i|
    """
    vals = _finite_values_1d(error)
    if vals.size == 0:
        return float(np.nan)
    return float(np.max(np.abs(vals)))


def percentile_absolute_error(
    error: ArrayLike,
    q: float = 95.0,
) -> float:
    r"""
    Percentile of absolute error.

    Formula
    -------
        Q_q( |e| )

    where `Q_q` denotes the `q`-th percentile.
    """
    qf = float(q)
    if not (0.0 <= qf <= 100.0):
        raise ValueError(f"q must lie in [0, 100], got {qf}.")
    vals = _finite_values_1d(error)
    if vals.size == 0:
        return float(np.nan)
    return float(np.percentile(np.abs(vals), qf))


def fraction_within_threshold(
    error_magnitude: ArrayLike,
    threshold: float,
) -> float:
    r"""
    Fraction of error magnitudes that do not exceed a threshold.

    Formula
    -------
        frac = (1/N) sum_i 1[ |e_i| <= tau ]

    Returns
    -------
    float
        Fraction in [0, 1], or NaN if no finite values exist.
    """
    tau = float(threshold)
    if tau < 0.0:
        raise ValueError(f"threshold must be nonnegative, got {tau}.")
    vals = _finite_values_1d(error_magnitude)
    if vals.size == 0:
        return float(np.nan)
    return float(np.mean(np.abs(vals) <= tau))


def first_threshold_exceedance_time(
    time_s: ArrayLike,
    error_magnitude: ArrayLike,
    threshold: float,
) -> float:
    """
    Return the first time at which the error magnitude exceeds a threshold.

    Parameters
    ----------
    time_s : array-like
        Sample times [s].
    error_magnitude : array-like
        Error magnitude at each time.
    threshold : float
        Threshold value.

    Returns
    -------
    float
        First exceedance time [s], or NaN if no exceedance occurs.
    """
    tau = float(threshold)
    if tau < 0.0:
        raise ValueError(f"threshold must be nonnegative, got {tau}.")

    t = _vec(time_s, name="time_s")
    e = _vec(error_magnitude, name="error_magnitude")
    if t.shape != e.shape:
        raise ValueError(
            f"time_s and error_magnitude must have the same shape, got "
            f"{t.shape} and {e.shape}."
        )

    mask = np.isfinite(t) & np.isfinite(e) & (np.abs(e) > tau)
    if not np.any(mask):
        return float(np.nan)
    return float(t[np.argmax(mask)])


def horizontal_error_series(position_error_ned_m: ArrayLike) -> FloatArray:
    r"""
    Horizontal position error magnitude from NED error vectors.

    Parameters
    ----------
    position_error_ned_m : array-like, shape (N, 3)
        Position error history in NED [m].

    Returns
    -------
    np.ndarray, shape (N,)
        Horizontal error magnitude [m].

    Formula
    -------
        e_h = sqrt(dN^2 + dE^2)
    """
    err = _mat_nxm(position_error_ned_m, ncol=3, name="position_error_ned_m")
    return np.sqrt(np.sum(err[:, :2] ** 2, axis=1))


def vertical_error_series(position_error_ned_m: ArrayLike) -> FloatArray:
    r"""
    Vertical position error magnitude from NED error vectors.

    Formula
    -------
        e_v = |dD|
    """
    err = _mat_nxm(position_error_ned_m, ncol=3, name="position_error_ned_m")
    return np.abs(err[:, 2])


def radial_error_series(position_error_ned_m: ArrayLike) -> FloatArray:
    r"""
    Full 3D radial position error magnitude from NED error vectors.

    Formula
    -------
        e_r = sqrt(dN^2 + dE^2 + dD^2)
    """
    err = _mat_nxm(position_error_ned_m, ncol=3, name="position_error_ned_m")
    return np.sqrt(np.sum(err ** 2, axis=1))


def horizontal_radius_percentile(
    position_error_ned_m: ArrayLike,
    q: float = 50.0,
) -> float:
    r"""
    Empirical horizontal-radius percentile from NED position errors.

    Parameters
    ----------
    position_error_ned_m : array-like, shape (N, 3)
        NED position error history [m].
    q : float, default=50
        Requested percentile.

    Returns
    -------
    float
        Horizontal radius percentile [m].

    Notes
    -----
    For `q=50`, this is often used in a CEP50-style reporting role.
    For `q=95`, it behaves like a practical horizontal 95% error radius summary.
    """
    qf = float(q)
    if not (0.0 <= qf <= 100.0):
        raise ValueError(f"q must lie in [0, 100], got {qf}.")
    horiz = horizontal_error_series(position_error_ned_m)
    vals = _finite_values_1d(horiz)
    if vals.size == 0:
        return float(np.nan)
    return float(np.percentile(vals, qf))


# -----------------------------------------------------------------------------
# Dataclass metric summaries
# -----------------------------------------------------------------------------


@dataclass
class ScalarErrorMetrics:
    """
    Summary statistics for a scalar error series.

    Attributes
    ----------
    count : int
        Number of finite samples included.
    mean_error : float
        Mean signed error.
    std_error : float
        Standard deviation of signed error.
    rmse : float
        Root-mean-square error.
    mae : float
        Mean absolute error.
    median_abs_error : float
        Median absolute error.
    p95_abs_error : float
        95th percentile of absolute error.
    max_abs_error : float
        Maximum absolute error.
    """

    count: int
    mean_error: float
    std_error: float
    rmse: float
    mae: float
    median_abs_error: float
    p95_abs_error: float
    max_abs_error: float

    @classmethod
    def from_error_series(cls, error: ArrayLike) -> "ScalarErrorMetrics":
        """
        Build a scalar metric summary from one error series.
        """
        return cls(
            count=_count_finite(error),
            mean_error=mean_error(error),
            std_error=_nanstd(error),
            rmse=root_mean_square_error(error),
            mae=mean_absolute_error(error),
            median_abs_error=median_absolute_error(error),
            p95_abs_error=percentile_absolute_error(error, q=95.0),
            max_abs_error=max_absolute_error(error),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return asdict(self)


@dataclass
class VectorErrorMetrics:
    """
    Summary statistics for a 3-vector error history.

    Attributes
    ----------
    component_labels : tuple[str, str, str]
        Human-readable axis labels.
    x, y, z : ScalarErrorMetrics
        Per-axis scalar summaries.
    norm_rmse : float
        RMSE of the vector norm.
    norm_mean : float
        Mean vector-error norm.
    norm_p95 : float
        95th percentile of vector-error norm.
    norm_max : float
        Maximum vector-error norm.
    """

    component_labels: tuple[str, str, str]
    x: ScalarErrorMetrics
    y: ScalarErrorMetrics
    z: ScalarErrorMetrics
    norm_rmse: float
    norm_mean: float
    norm_p95: float
    norm_max: float

    @classmethod
    def from_error_series(
        cls,
        error_xyz: ArrayLike,
        *,
        component_labels: tuple[str, str, str] = ("x", "y", "z"),
    ) -> "VectorErrorMetrics":
        """
        Build a vector metric summary from a shape-(N, 3) error history.
        """
        err = _mat_nxm(error_xyz, ncol=3, name="error_xyz")
        norms = np.linalg.norm(err, axis=1)
        return cls(
            component_labels=tuple(component_labels),
            x=ScalarErrorMetrics.from_error_series(err[:, 0]),
            y=ScalarErrorMetrics.from_error_series(err[:, 1]),
            z=ScalarErrorMetrics.from_error_series(err[:, 2]),
            norm_rmse=root_mean_square_error(norms),
            norm_mean=_nanmean(norms),
            norm_p95=_nanpercentile(norms, 95.0),
            norm_max=max_absolute_error(norms),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return asdict(self)


@dataclass
class PositionErrorMetrics:
    """
    Summary statistics for local NED position errors.

    Attributes
    ----------
    north, east, down : ScalarErrorMetrics
        Per-axis error summaries in metres.
    horizontal_rmse_m : float
        Horizontal RMSE.
    horizontal_mean_m : float
        Mean horizontal error.
    horizontal_p95_m : float
        95th percentile horizontal error.
    horizontal_max_m : float
        Maximum horizontal error.
    vertical_rmse_m : float
        Vertical RMSE using |dD|.
    vertical_mean_m : float
        Mean vertical error using |dD|.
    vertical_p95_m : float
        95th percentile vertical error.
    vertical_max_m : float
        Maximum vertical error.
    radial_rmse_m : float
        3D radial RMSE.
    radial_mean_m : float
        Mean 3D radial error.
    radial_p95_m : float
        95th percentile 3D radial error.
    radial_max_m : float
        Maximum 3D radial error.
    cep50_m : float
        Empirical 50th percentile horizontal radius.
    cep95_m : float
        Empirical 95th percentile horizontal radius.
    """

    north: ScalarErrorMetrics
    east: ScalarErrorMetrics
    down: ScalarErrorMetrics

    horizontal_rmse_m: float
    horizontal_mean_m: float
    horizontal_p95_m: float
    horizontal_max_m: float

    vertical_rmse_m: float
    vertical_mean_m: float
    vertical_p95_m: float
    vertical_max_m: float

    radial_rmse_m: float
    radial_mean_m: float
    radial_p95_m: float
    radial_max_m: float

    cep50_m: float
    cep95_m: float

    @classmethod
    def from_error_series(
        cls,
        position_error_ned_m: ArrayLike,
    ) -> "PositionErrorMetrics":
        """
        Build a position-error summary from a shape-(N, 3) NED error history.
        """
        err = _mat_nxm(position_error_ned_m, ncol=3, name="position_error_ned_m")
        horiz = horizontal_error_series(err)
        vert = vertical_error_series(err)
        radial = radial_error_series(err)

        return cls(
            north=ScalarErrorMetrics.from_error_series(err[:, 0]),
            east=ScalarErrorMetrics.from_error_series(err[:, 1]),
            down=ScalarErrorMetrics.from_error_series(err[:, 2]),
            horizontal_rmse_m=root_mean_square_error(horiz),
            horizontal_mean_m=_nanmean(horiz),
            horizontal_p95_m=_nanpercentile(horiz, 95.0),
            horizontal_max_m=max_absolute_error(horiz),
            vertical_rmse_m=root_mean_square_error(vert),
            vertical_mean_m=_nanmean(vert),
            vertical_p95_m=_nanpercentile(vert, 95.0),
            vertical_max_m=max_absolute_error(vert),
            radial_rmse_m=root_mean_square_error(radial),
            radial_mean_m=_nanmean(radial),
            radial_p95_m=_nanpercentile(radial, 95.0),
            radial_max_m=max_absolute_error(radial),
            cep50_m=horizontal_radius_percentile(err, q=50.0),
            cep95_m=horizontal_radius_percentile(err, q=95.0),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return asdict(self)


@dataclass
class IntegrityMetricsSummary:
    """
    Summary of an integrity-monitor history.

    Attributes
    ----------
    count : int
        Number of snapshots.
    fraction_nis_available : float
        Fraction of snapshots containing a NIS result.
    fraction_nis_passed : float
        Fraction of available NIS results that passed.
    fraction_nees_available : float
        Fraction of snapshots containing a NEES result.
    fraction_nees_passed : float
        Fraction of available NEES results that passed.
    mean_horizontal_protection_m : float
        Mean horizontal protection level.
    mean_vertical_protection_m : float
        Mean vertical protection level.
    mean_radial_protection_m : float
        Mean 3D radial protection level.
    max_horizontal_protection_m : float
        Maximum horizontal protection level.
    max_vertical_protection_m : float
        Maximum vertical protection level.
    max_radial_protection_m : float
        Maximum 3D radial protection level.
    fraction_horizontal_alert_satisfied : float
        Fraction of snapshots where HPL <= HAL, over snapshots with a HAL.
    fraction_vertical_alert_satisfied : float
        Fraction of snapshots where VPL <= VAL, over snapshots with a VAL.
    fraction_hazardously_misleading_horizontal : float
        Fraction of snapshots flagged as horizontally HMI, over snapshots where
        the field is available.
    fraction_hazardously_misleading_vertical : float
        Fraction of snapshots flagged as vertically HMI, over snapshots where
        the field is available.
    """

    count: int

    fraction_nis_available: float
    fraction_nis_passed: float
    fraction_nees_available: float
    fraction_nees_passed: float

    mean_horizontal_protection_m: float
    mean_vertical_protection_m: float
    mean_radial_protection_m: float

    max_horizontal_protection_m: float
    max_vertical_protection_m: float
    max_radial_protection_m: float

    fraction_horizontal_alert_satisfied: float
    fraction_vertical_alert_satisfied: float

    fraction_hazardously_misleading_horizontal: float
    fraction_hazardously_misleading_vertical: float

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Sequence[IntegritySnapshot],
    ) -> "IntegrityMetricsSummary":
        """
        Build an integrity summary from a sequence of snapshots.
        """
        n = len(snapshots)

        hpl = np.asarray(
            [s.protection_levels.horizontal_m for s in snapshots],
            dtype=np.float64,
        )
        vpl = np.asarray(
            [s.protection_levels.vertical_m for s in snapshots],
            dtype=np.float64,
        )
        rpl = np.asarray(
            [s.protection_levels.radial_3d_m for s in snapshots],
            dtype=np.float64,
        )

        nis_available = [s.nis_result is not None for s in snapshots]
        nis_passed = [s.nis_result.passed for s in snapshots if s.nis_result is not None]

        nees_available = [s.nees_result is not None for s in snapshots]
        nees_passed = [
            s.nees_result.passed for s in snapshots if s.nees_result is not None
        ]

        h_alert_ok = [
            s.protection_levels.horizontal_within_alert_limit
            for s in snapshots
            if s.protection_levels.horizontal_within_alert_limit is not None
        ]
        v_alert_ok = [
            s.protection_levels.vertical_within_alert_limit
            for s in snapshots
            if s.protection_levels.vertical_within_alert_limit is not None
        ]

        hmi_h = [
            s.hazardously_misleading_horizontal
            for s in snapshots
            if s.hazardously_misleading_horizontal is not None
        ]
        hmi_v = [
            s.hazardously_misleading_vertical
            for s in snapshots
            if s.hazardously_misleading_vertical is not None
        ]

        return cls(
            count=n,
            fraction_nis_available=float(np.mean(nis_available)) if n > 0 else float(np.nan),
            fraction_nis_passed=float(np.mean(nis_passed)) if len(nis_passed) > 0 else float(np.nan),
            fraction_nees_available=float(np.mean(nees_available)) if n > 0 else float(np.nan),
            fraction_nees_passed=float(np.mean(nees_passed)) if len(nees_passed) > 0 else float(np.nan),
            mean_horizontal_protection_m=_nanmean(hpl),
            mean_vertical_protection_m=_nanmean(vpl),
            mean_radial_protection_m=_nanmean(rpl),
            max_horizontal_protection_m=max_absolute_error(hpl),
            max_vertical_protection_m=max_absolute_error(vpl),
            max_radial_protection_m=max_absolute_error(rpl),
            fraction_horizontal_alert_satisfied=float(np.mean(h_alert_ok)) if len(h_alert_ok) > 0 else float(np.nan),
            fraction_vertical_alert_satisfied=float(np.mean(v_alert_ok)) if len(v_alert_ok) > 0 else float(np.nan),
            fraction_hazardously_misleading_horizontal=float(np.mean(hmi_h)) if len(hmi_h) > 0 else float(np.nan),
            fraction_hazardously_misleading_vertical=float(np.mean(hmi_v)) if len(hmi_v) > 0 else float(np.nan),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return asdict(self)


@dataclass
class ScenarioMetricsSummary:
    """
    Top-level metric summary for one scenario simulation.

    Attributes
    ----------
    duration_s : float
        Truth duration [s].
    num_truth_samples : int
        Number of truth samples.
    num_ins_states : int
        Number of INS states logged.
    num_pf_updates : int
        Number of PF updates logged.
    num_sequence_updates : int
        Number of sequence-matcher updates logged.
    num_integrity_snapshots : int
        Number of integrity snapshots logged.
    gravimeter_error : ScalarErrorMetrics or None
        Gravimeter measurement-minus-ideal summary.
    depth_error : ScalarErrorMetrics or None
        Depth measurement-minus-ideal summary.
    velocity_aid_error : VectorErrorMetrics or None
        Velocity-aid measurement-minus-ideal summary.
    imu_gyro_error : VectorErrorMetrics or None
        IMU gyro measurement-minus-ideal summary.
    imu_accel_error : VectorErrorMetrics or None
        IMU accelerometer measurement-minus-ideal summary.
    ins_position_error : PositionErrorMetrics or None
        INS position error summary against truth.
    pf_position_error : PositionErrorMetrics or None
        PF position error summary against truth.
    sequence_position_error : PositionErrorMetrics or None
        Sequence-matcher position error summary against truth.
    integrity : IntegrityMetricsSummary or None
        Integrity-history summary.
    """

    duration_s: float
    num_truth_samples: int
    num_ins_states: int
    num_pf_updates: int
    num_sequence_updates: int
    num_integrity_snapshots: int

    gravimeter_error: Optional[ScalarErrorMetrics] = None
    depth_error: Optional[ScalarErrorMetrics] = None
    velocity_aid_error: Optional[VectorErrorMetrics] = None
    imu_gyro_error: Optional[VectorErrorMetrics] = None
    imu_accel_error: Optional[VectorErrorMetrics] = None
    ins_position_error: Optional[PositionErrorMetrics] = None
    pf_position_error: Optional[PositionErrorMetrics] = None
    sequence_position_error: Optional[PositionErrorMetrics] = None
    integrity: Optional[IntegrityMetricsSummary] = None

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Truth/estimator alignment helpers
# -----------------------------------------------------------------------------


def interpolate_truth_geodetic(
    truth: TruthTrajectory,
    query_time_s: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Interpolate truth geodetic position to arbitrary query times.

    Parameters
    ----------
    truth : TruthTrajectory
        Canonical truth trajectory.
    query_time_s : array-like
        Query times [s].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Interpolated `(lat_rad, lon_rad, height_m)`.

    Notes
    -----
    This uses 1D linear interpolation independently on each geodetic component.
    For the current repository's scenario scales and smooth truth trajectories,
    that is a practical and sufficient choice for estimator metric evaluation.
    """
    tq = _vec(query_time_s, name="query_time_s")
    tt = _vec(truth.time_s, name="truth.time_s")

    lat = np.interp(tq, tt, _vec(truth.lat_rad, name="truth.lat_rad"))
    lon = np.interp(tq, tt, _vec(truth.lon_rad, name="truth.lon_rad"))
    h = np.interp(tq, tt, _vec(truth.height_m, name="truth.height_m"))

    return (
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )


def ins_state_times(ins_states: Sequence[ErrorStateINSState]) -> FloatArray:
    """
    Extract nominal-state timestamps from a sequence of INS states.

    Missing timestamps are recorded as NaN.
    """
    out = np.full(len(ins_states), np.nan, dtype=np.float64)
    for k, state in enumerate(ins_states):
        time_s = getattr(state.nominal, "time_s", None)
        if time_s is not None:
            out[k] = float(time_s)
    return out


def pf_update_times(pf_updates: Sequence[MapMatchPFUpdateResult]) -> FloatArray:
    """
    Extract timestamps from a sequence of PF update results.

    Missing timestamps are recorded as NaN.
    """
    out = np.full(len(pf_updates), np.nan, dtype=np.float64)
    for k, update in enumerate(pf_updates):
        time_s = getattr(update, "time_s", None)
        if time_s is not None:
            out[k] = float(time_s)
    return out


def sequence_update_times(
    sequence_updates: Sequence[SequenceMatchUpdateResult],
) -> FloatArray:
    """
    Extract timestamps from a sequence of sequence-matcher update results.

    Missing timestamps are recorded as NaN.
    """
    out = np.full(len(sequence_updates), np.nan, dtype=np.float64)
    for k, update in enumerate(sequence_updates):
        time_s = getattr(update, "time_s", None)
        if time_s is not None:
            out[k] = float(time_s)
    return out


def ins_position_error_history_from_truth(
    truth: TruthTrajectory,
    ins_states: Sequence[ErrorStateINSState],
) -> FloatArray:
    """
    Build local NED INS position error history against truth.

    Parameters
    ----------
    truth : TruthTrajectory
        Truth trajectory.
    ins_states : sequence[ErrorStateINSState]
        Logged INS states.

    Returns
    -------
    np.ndarray, shape (N, 3)
        NED position errors `[dN, dE, dD]` [m].

    Notes
    -----
    Truth is first interpolated to the INS-state timestamps, then each INS
    geodetic state is differenced against that interpolated truth position using
    the repository's `geodetic_position_error_ned(...)` helper.
    """
    n = len(ins_states)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    t_ins = ins_state_times(ins_states)
    if np.any(~np.isfinite(t_ins)):
        raise ValueError(
            "All INS states must have finite nominal.time_s for truth comparison."
        )

    true_lat, true_lon, true_h = interpolate_truth_geodetic(truth, t_ins)
    err = np.empty((n, 3), dtype=np.float64)

    for k, state in enumerate(ins_states):
        err[k] = _vec3(
            geodetic_position_error_ned(
                estimated_lat_rad=float(state.nominal.lat_rad),
                estimated_lon_rad=float(state.nominal.lon_rad),
                estimated_height_m=float(state.nominal.height_m),
                true_lat_rad=float(true_lat[k]),
                true_lon_rad=float(true_lon[k]),
                true_height_m=float(true_h[k]),
            ),
            name="position_error_ned_m",
        )

    return err


def pf_position_error_history_from_truth(
    truth: TruthTrajectory,
    pf_updates: Sequence[MapMatchPFUpdateResult],
) -> FloatArray:
    """
    Build local NED PF position error history against truth.

    Parameters
    ----------
    truth : TruthTrajectory
        Truth trajectory.
    pf_updates : sequence[MapMatchPFUpdateResult]
        Logged PF update results.

    Returns
    -------
    np.ndarray, shape (N, 3)
        NED position errors `[dN, dE, dD]` [m].
    """
    n = len(pf_updates)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    t_pf = pf_update_times(pf_updates)
    if np.any(~np.isfinite(t_pf)):
        raise ValueError(
            "All PF updates must have finite time_s for truth comparison."
        )

    true_lat, true_lon, true_h = interpolate_truth_geodetic(truth, t_pf)
    err = np.empty((n, 3), dtype=np.float64)

    for k, update in enumerate(pf_updates):
        est = update.estimate
        err[k] = _vec3(
            geodetic_position_error_ned(
                estimated_lat_rad=float(est.lat_rad),
                estimated_lon_rad=float(est.lon_rad),
                estimated_height_m=float(est.height_m),
                true_lat_rad=float(true_lat[k]),
                true_lon_rad=float(true_lon[k]),
                true_height_m=float(true_h[k]),
            ),
            name="pf_position_error_ned_m",
        )

    return err


def sequence_position_error_history_from_truth(
    truth: TruthTrajectory,
    sequence_updates: Sequence[SequenceMatchUpdateResult],
) -> FloatArray:
    """
    Build local NED sequence-matcher position error history against truth.

    Parameters
    ----------
    truth : TruthTrajectory
        Truth trajectory.
    sequence_updates : sequence[SequenceMatchUpdateResult]
        Logged delayed sequence-matcher updates.

    Returns
    -------
    np.ndarray, shape (N, 3)
        NED position errors `[dN, dE, dD]` [m].
    """
    n = len(sequence_updates)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    t_seq = sequence_update_times(sequence_updates)
    if np.any(~np.isfinite(t_seq)):
        raise ValueError(
            "All sequence updates must have finite time_s for truth comparison."
        )

    true_lat, true_lon, true_h = interpolate_truth_geodetic(truth, t_seq)
    err = np.empty((n, 3), dtype=np.float64)

    for k, update in enumerate(sequence_updates):
        est = update.estimate
        err[k] = _vec3(
            geodetic_position_error_ned(
                estimated_lat_rad=float(est.lat_rad),
                estimated_lon_rad=float(est.lon_rad),
                estimated_height_m=float(est.height_m),
                true_lat_rad=float(true_lat[k]),
                true_lon_rad=float(true_lon[k]),
                true_height_m=float(true_h[k]),
            ),
            name="sequence_position_error_ned_m",
        )

    return err


# -----------------------------------------------------------------------------
# High-level builders
# -----------------------------------------------------------------------------


def gravimeter_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[ScalarErrorMetrics]:
    """
    Build gravimeter error metrics from one scenario result.

    The error series is:
        measured - ideal
    """
    if len(result.sensors.gravimeter_samples) == 0:
        return None
    arrays = result.sensors.gravimeter_history_arrays()
    err = arrays["gravimeter_value_mps2"] - arrays["gravimeter_ideal_value_mps2"]
    return ScalarErrorMetrics.from_error_series(err)


def depth_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[ScalarErrorMetrics]:
    """
    Build depth-sensor error metrics from one scenario result.

    The error series is:
        measured - ideal
    """
    if len(result.sensors.depth_samples) == 0:
        return None
    arrays = result.sensors.depth_history_arrays()
    err = arrays["depth_value_m"] - arrays["depth_ideal_depth_m"]
    return ScalarErrorMetrics.from_error_series(err)


def velocity_aid_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[VectorErrorMetrics]:
    """
    Build velocity-aid error metrics from one scenario result.

    The error series is:
        measured - ideal
    """
    if len(result.sensors.velocity_aid_samples) == 0:
        return None
    arrays = result.sensors.velocity_aid_history_arrays()
    err = arrays["velocity_aid_value_mps"] - arrays["velocity_aid_ideal_value_mps"]
    return VectorErrorMetrics.from_error_series(
        err,
        component_labels=("forward_or_north", "right_or_east", "down"),
    )


def imu_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> tuple[Optional[VectorErrorMetrics], Optional[VectorErrorMetrics]]:
    """
    Build IMU gyro and accelerometer error metrics from one scenario result.

    Returns
    -------
    tuple
        `(gyro_metrics, accel_metrics)` or `(None, None)` when no IMU history exists.
    """
    if len(result.sensors.imu_samples) == 0:
        return None, None

    arrays = result.sensors.imu_history_arrays()
    gyro_err = arrays["imu_omega_ib_b_radps"] - arrays["imu_ideal_omega_ib_b_radps"]
    accel_err = arrays["imu_f_ib_b_mps2"] - arrays["imu_ideal_f_ib_b_mps2"]

    gyro_metrics = VectorErrorMetrics.from_error_series(
        gyro_err,
        component_labels=("p", "q", "r"),
    )
    accel_metrics = VectorErrorMetrics.from_error_series(
        accel_err,
        component_labels=("fx", "fy", "fz"),
    )
    return gyro_metrics, accel_metrics


def ins_position_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[PositionErrorMetrics]:
    """
    Build INS position error metrics against truth.

    Returns
    -------
    PositionErrorMetrics or None
        None when no INS history exists.
    """
    if len(result.estimators.ins_states) == 0:
        return None
    err = ins_position_error_history_from_truth(
        result.truth,
        result.estimators.ins_states,
    )
    return PositionErrorMetrics.from_error_series(err)


def pf_position_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[PositionErrorMetrics]:
    """
    Build PF position error metrics against truth.

    Returns
    -------
    PositionErrorMetrics or None
        None when no PF history exists.
    """
    if len(result.estimators.pf_updates) == 0:
        return None
    err = pf_position_error_history_from_truth(
        result.truth,
        result.estimators.pf_updates,
    )
    return PositionErrorMetrics.from_error_series(err)


def sequence_position_error_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[PositionErrorMetrics]:
    """
    Build sequence-matcher position error metrics against truth.

    Returns
    -------
    PositionErrorMetrics or None
        None when no sequence history exists.
    """
    if len(result.estimators.sequence_updates) == 0:
        return None
    err = sequence_position_error_history_from_truth(
        result.truth,
        result.estimators.sequence_updates,
    )
    return PositionErrorMetrics.from_error_series(err)


def integrity_metrics_from_result(
    result: ScenarioSimulationResult,
) -> Optional[IntegrityMetricsSummary]:
    """
    Build integrity-history metrics from one scenario result.
    """
    if len(result.estimators.integrity_snapshots) == 0:
        return None
    return IntegrityMetricsSummary.from_snapshots(
        result.estimators.integrity_snapshots
    )


def scenario_metrics_from_result(
    result: ScenarioSimulationResult,
) -> ScenarioMetricsSummary:
    """
    Build a top-level metric summary from one scenario result.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Full simulation result.

    Returns
    -------
    ScenarioMetricsSummary
        Nested metric summary suitable for reporting and Monte Carlo aggregation.
    """
    gyro_metrics, accel_metrics = imu_error_metrics_from_result(result)

    return ScenarioMetricsSummary(
        duration_s=float(result.duration_s),
        num_truth_samples=len(result.truth),
        num_ins_states=len(result.estimators.ins_states),
        num_pf_updates=len(result.estimators.pf_updates),
        num_sequence_updates=len(result.estimators.sequence_updates),
        num_integrity_snapshots=len(result.estimators.integrity_snapshots),
        gravimeter_error=gravimeter_error_metrics_from_result(result),
        depth_error=depth_error_metrics_from_result(result),
        velocity_aid_error=velocity_aid_error_metrics_from_result(result),
        imu_gyro_error=gyro_metrics,
        imu_accel_error=accel_metrics,
        ins_position_error=ins_position_error_metrics_from_result(result),
        pf_position_error=pf_position_error_metrics_from_result(result),
        sequence_position_error=sequence_position_error_metrics_from_result(result),
        integrity=integrity_metrics_from_result(result),
    )


__all__ = [
    "FloatArray",
    "IntegrityMetricsSummary",
    "PositionErrorMetrics",
    "ScalarErrorMetrics",
    "ScenarioMetricsSummary",
    "VectorErrorMetrics",
    "depth_error_metrics_from_result",
    "first_threshold_exceedance_time",
    "fraction_within_threshold",
    "gravimeter_error_metrics_from_result",
    "horizontal_error_series",
    "horizontal_radius_percentile",
    "ins_position_error_history_from_truth",
    "ins_position_error_metrics_from_result",
    "ins_state_times",
    "integrity_metrics_from_result",
    "interpolate_truth_geodetic",
    "imu_error_metrics_from_result",
    "max_absolute_error",
    "mean_absolute_error",
    "mean_error",
    "median_absolute_error",
    "percentile_absolute_error",
    "pf_position_error_history_from_truth",
    "pf_position_error_metrics_from_result",
    "pf_update_times",
    "radial_error_series",
    "root_mean_square_error",
    "scenario_metrics_from_result",
    "sequence_position_error_history_from_truth",
    "sequence_position_error_metrics_from_result",
    "sequence_update_times",
    "velocity_aid_error_metrics_from_result",
    "vertical_error_series",
]
