"""
fusion.py

Reusable measurement-packaging, gating, and update helpers for the
gravity-aided navigation simulator.

This module sits directly on top of `estimators.error_state_ins` and is the
estimator-side orchestration layer between:
- nominal/error-state INS propagation, and
- external aiding measurements such as:
    - NED velocity aiding
    - body-frame velocity aiding (for DVL-like sensors)
    - geodetic position aiding
    - signed depth aiding
    - later custom scalar/vector measurements such as gravity-map residuals

Why this file exists
--------------------
The repository now has the ingredients to propagate a navigation solution:
- truth and sensor layers
- a velocity-aid sensor abstraction
- a local-level error-state INS core

What is still missing is the estimator-side “glue” that turns measurements into
consistent linearized updates. That glue belongs here, not inside each sensor
class and not inside the low-level INS propagator.

This file therefore provides:
1) generic linear-measurement containers,
2) innovation covariance / NIS helpers,
3) optional innovation gating,
4) batch stacking for independent measurements,
5) convenience wrappers for common aiding measurements.

Conventions
-----------
- Navigation frame is local NED:
      x = North, y = East, z = Down
- Body frame is right-handed:
      x = forward, y = right, z = down
- `C_n_b` is the passive body->NED DCM:

      v^n = C_n_b v^b

- All velocities are in m/s.
- Position measurements are geodetic:
      [lat, lon, h] = [rad, rad, m]
- Depth is signed relative to a reference surface:
      d = h_ref - h

Relationship to `error_state_ins.py`
------------------------------------
This file assumes the 15-state closed-loop error formulation implemented in the
INS core:

    delta_x =
    [ d_lat, d_lon, d_h,
      d_v_N, d_v_E, d_v_D,
      d_theta_N, d_theta_E, d_theta_D,
      d_bg_x, d_bg_y, d_bg_z,
      d_ba_x, d_ba_y, d_ba_z ]^T

and uses the generic linear update:

    r = z - h(x_nom)
    S = H P H^T + R
    K = P H^T S^{-1}
    delta_x = K r

with covariance update performed inside `ErrorStateINS.linear_update(...)`.

Primary references used here
----------------------------
1) INSTINCT / University of Stuttgart,
   "INS/GNSS Loosely-coupled Kalman Filter (local-navigation frame)"
   URL:
   https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

   Used for:
   - the standard local-level INS / KF separation into:
       * continuous error-state dynamics F
       * noise-input matrix G
       * transition matrix Phi
       * process-noise covariance Q
   - reinforcing that measurement fusion should be layered on top of that
     propagation model rather than embedded inside the mechanization itself

2) Welch & Bishop,
   "An Introduction to the Kalman Filter"
   URL:
   https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

   Used for the standard linear/discrete Kalman update structure:

       K = P H^T (H P H^T + R)^{-1}
       x^+ = x^- + K (z - h(x^-))
       P^+ = (I - K H) P^-     [or Joseph form in implementation]

   In this repository the actual state correction is handled by the closed-loop
   injection logic in `error_state_ins.py`, but the same measurement-update
   structure is used.

3) Repository `estimators.error_state_ins`
   This module relies on:
   - the nominal-state convention
   - the closed-loop small-angle attitude correction
   - the generic linear update API
   - the existing direct NED velocity / geodetic position / depth models

Design notes
------------
- This is intentionally a practical estimator utility layer, not a full
  graph-optimization or factor-graph framework.
- The module keeps measurement packaging explicit and transparent.
- Innovation gating is optional and threshold-based. It computes NIS, but the
  policy choice of what threshold to use is left to the caller.
- Later gravity-map matching code can build custom linearized measurements here
  without rewriting Kalman-update plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.frames import skew, wrap_angle_pi
from ..sensors.depth import DepthMeasurement
from ..sensors.velocity_aid import VelocityAidMeasurement
from .error_state_ins import (
    ERROR_STATE_SIZE,
    ERR_ATT,
    ERR_POS,
    ERR_VEL,
    ErrorStateINS,
    ErrorStateINSState,
    LinearizedMeasurementUpdate,
    depth_measurement_jacobian,
    depth_measurement_model,
    depth_measurement_residual,
    position_measurement_jacobian_geodetic,
    position_measurement_model_geodetic,
    position_measurement_residual_geodetic,
    velocity_measurement_jacobian_ned,
    velocity_measurement_model_ned,
    velocity_measurement_residual_ned,
)

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


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


def _row(x: ArrayLike, *, name: str) -> FloatArray:
    """Validate and return a shape-(15,) Jacobian row."""
    arr = _vec(x, name=name)
    if arr.shape != (ERROR_STATE_SIZE,):
        raise ValueError(
            f"{name} must have shape ({ERROR_STATE_SIZE},), got {arr.shape}."
        )
    return arr


def _mat(x: ArrayLike, *, name: str, rows: Optional[int] = None, cols: Optional[int] = None) -> FloatArray:
    """Validate and return a matrix with optional shape checks."""
    arr = _as_float_array(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {arr.shape}.")
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"{name} must have {cols} cols, got {arr.shape}.")
    return arr


def _symmetrize(M: ArrayLike) -> FloatArray:
    """Return the symmetric part of a square matrix."""
    A = _as_float_array(M)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}.")
    return 0.5 * (A + A.T)


def _state_from_filter_or_state(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> ErrorStateINSState:
    """
    Return `ErrorStateINSState` regardless of whether the caller passed a filter
    object or the state directly.
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
# Covariance helpers
# -----------------------------------------------------------------------------


def diagonal_covariance_from_std(
    std: ArrayLike | float,
) -> FloatArray:
    r"""
    Build a diagonal covariance matrix from scalar or vector standard deviations.

    Parameters
    ----------
    std : scalar or array-like, shape (N,)
        Standard deviation(s).

    Returns
    -------
    np.ndarray, shape (N, N)
        Diagonal covariance matrix.

    Formula
    -------
        R = diag(sigma_i^2)
    """
    arr = _as_float_array(std)
    if arr.ndim == 0:
        sigma = np.array([float(arr)], dtype=np.float64)
    else:
        sigma = arr.reshape(-1).astype(np.float64)

    if np.any(sigma < 0.0):
        raise ValueError(f"Standard deviations must be nonnegative, got {sigma}.")
    return np.diag(sigma ** 2)


def diagonal_covariance_from_variance(
    variance: ArrayLike | float,
) -> FloatArray:
    r"""
    Build a diagonal covariance matrix from scalar or vector variances.

    Parameters
    ----------
    variance : scalar or array-like, shape (N,)
        Variance(s).

    Returns
    -------
    np.ndarray, shape (N, N)
        Diagonal covariance matrix.
    """
    arr = _as_float_array(variance)
    if arr.ndim == 0:
        var = np.array([float(arr)], dtype=np.float64)
    else:
        var = arr.reshape(-1).astype(np.float64)

    if np.any(var < 0.0):
        raise ValueError(f"Variances must be nonnegative, got {var}.")
    return np.diag(var)


def block_diag(*matrices: ArrayLike) -> FloatArray:
    """
    Construct a block-diagonal matrix from square input matrices.

    Parameters
    ----------
    *matrices : array-like
        Square blocks to place on the diagonal.

    Returns
    -------
    np.ndarray
        Block-diagonal matrix.

    Notes
    -----
    This avoids depending on SciPy just for `scipy.linalg.block_diag`.
    """
    if len(matrices) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    mats = [_as_float_array(M) for M in matrices]
    for k, M in enumerate(mats):
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Block {k} must be square, got shape {M.shape}.")

    total = sum(M.shape[0] for M in mats)
    out = np.zeros((total, total), dtype=np.float64)

    cursor = 0
    for M in mats:
        n = M.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = M
        cursor += n
    return out


# -----------------------------------------------------------------------------
# Generic linear measurement containers
# -----------------------------------------------------------------------------


@dataclass
class LinearMeasurement:
    """
    Generic linearized measurement container.

    Attributes
    ----------
    label : str
        Human-readable measurement label.
    z : np.ndarray, shape (m,)
        Actual measurement vector.
    h : np.ndarray, shape (m,)
        Predicted measurement vector from the nominal state.
    H : np.ndarray, shape (m, 15)
        Linearized measurement Jacobian with respect to the error state.
    R : np.ndarray, shape (m, m)
        Measurement covariance.
    time_s : float or None
        Optional timestamp.

    Residual convention
    -------------------
    Residual is always formed as:

        r = z - h

    which matches the closed-loop update convention in `error_state_ins.py`.
    """

    label: str
    z: FloatArray
    h: FloatArray
    H: FloatArray
    R: FloatArray
    time_s: Optional[float] = None

    def __post_init__(self) -> None:
        self.label = str(self.label)
        self.z = _vec(self.z, name="z")
        self.h = _vec(self.h, name="h")
        if self.z.shape != self.h.shape:
            raise ValueError(f"z and h must have same shape, got {self.z.shape} and {self.h.shape}.")

        m = self.z.shape[0]
        self.H = _mat(self.H, name="H", rows=m, cols=ERROR_STATE_SIZE)
        self.R = _mat(self.R, name="R", rows=m, cols=m)
        self.R = _symmetrize(self.R)

        if self.time_s is not None:
            self.time_s = float(self.time_s)

    @property
    def residual(self) -> FloatArray:
        """Return `z - h`."""
        return (self.z - self.h).astype(np.float64)

    @property
    def dimension(self) -> int:
        """Measurement dimension."""
        return int(self.z.shape[0])


@dataclass
class InnovationGateResult:
    """
    Innovation gating result.

    Attributes
    ----------
    nis : float
        Normalized innovation squared:
            nis = r^T S^{-1} r
    threshold : float or None
        Gating threshold used, if any.
    accepted : bool
        True if the measurement passed the gate.
    """

    nis: float
    threshold: Optional[float]
    accepted: bool


@dataclass
class FusionUpdateResult:
    """
    Result of attempting one fused measurement update.

    Attributes
    ----------
    label : str
        Measurement label.
    time_s : float or None
        Optional timestamp.
    accepted : bool
        True if the update was applied.
    gate : InnovationGateResult
        Innovation gating result.
    update : LinearizedMeasurementUpdate or None
        The actual Kalman update result if accepted, else None.
    """

    label: str
    time_s: Optional[float]
    accepted: bool
    gate: InnovationGateResult
    update: Optional[LinearizedMeasurementUpdate]

    @property
    def residual(self) -> FloatArray:
        """Residual vector whether or not the update was accepted."""
        if self.update is not None:
            return self.update.residual
        raise AttributeError("Residual is only available when an update was applied.")

    @property
    def nis(self) -> float:
        """Normalized innovation squared."""
        return float(self.gate.nis)


# -----------------------------------------------------------------------------
# Innovation / gating helpers
# -----------------------------------------------------------------------------


def innovation_covariance(
    P: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> FloatArray:
    r"""
    Compute innovation covariance.

    Formula
    -------
        S = H P H^T + R
    """
    Pm = _mat(P, name="P", rows=ERROR_STATE_SIZE, cols=ERROR_STATE_SIZE)
    Hm = _as_float_array(H)
    Rm = _as_float_array(R)

    if Hm.ndim != 2 or Hm.shape[1] != ERROR_STATE_SIZE:
        raise ValueError(
            f"H must have shape (m, {ERROR_STATE_SIZE}), got {Hm.shape}."
        )
    m = Hm.shape[0]
    if Rm.shape != (m, m):
        raise ValueError(f"R must have shape ({m}, {m}), got {Rm.shape}.")

    return _symmetrize(Hm @ Pm @ Hm.T + Rm)


def normalized_innovation_squared(
    residual: ArrayLike,
    innovation_covariance_matrix: ArrayLike,
) -> float:
    r"""
    Compute normalized innovation squared (NIS).

    Formula
    -------
        nis = r^T S^{-1} r
    """
    r = _vec(residual, name="residual")
    S = _as_float_array(innovation_covariance_matrix)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != r.shape[0]:
        raise ValueError(
            f"S must be square with side length {r.shape[0]}, got {S.shape}."
        )
    return float(r.T @ np.linalg.inv(S) @ r)


def gate_measurement(
    measurement: LinearMeasurement,
    P: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
) -> InnovationGateResult:
    """
    Evaluate optional NIS-based innovation gating.

    Parameters
    ----------
    measurement : LinearMeasurement
        Candidate measurement.
    P : array-like, shape (15, 15)
        Current covariance matrix.
    nis_threshold : float, optional
        If provided, reject the measurement when `nis > nis_threshold`.
        If omitted, the gate always accepts.

    Returns
    -------
    InnovationGateResult
        Gating result.

    Notes
    -----
    This function computes NIS whether or not a threshold is provided so the
    caller can log it for diagnostics.
    """
    S = innovation_covariance(P, measurement.H, measurement.R)
    nis = normalized_innovation_squared(measurement.residual, S)

    if nis_threshold is None:
        return InnovationGateResult(
            nis=nis,
            threshold=None,
            accepted=True,
        )

    threshold = float(nis_threshold)
    if threshold < 0.0:
        raise ValueError(f"nis_threshold must be nonnegative, got {threshold}.")
    return InnovationGateResult(
        nis=nis,
        threshold=threshold,
        accepted=bool(nis <= threshold),
    )


# -----------------------------------------------------------------------------
# Generic measurement application
# -----------------------------------------------------------------------------


def make_linear_measurement(
    z: ArrayLike,
    h: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    *,
    label: str,
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Convenience constructor for `LinearMeasurement`.
    """
    return LinearMeasurement(
        label=label,
        z=_vec(z, name="z"),
        h=_vec(h, name="h"),
        H=_as_float_array(H),
        R=_as_float_array(R),
        time_s=time_s,
    )


def apply_linear_measurement(
    ins: ErrorStateINS,
    measurement: LinearMeasurement,
    *,
    nis_threshold: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply one generic linearized measurement to the filter.

    Parameters
    ----------
    ins : ErrorStateINS
        Target filter.
    measurement : LinearMeasurement
        Measurement package to apply.
    nis_threshold : float, optional
        Optional NIS threshold for innovation gating.

    Returns
    -------
    FusionUpdateResult
        Update result.
    """
    gate = gate_measurement(
        measurement,
        ins.covariance,
        nis_threshold=nis_threshold,
    )

    if not gate.accepted:
        return FusionUpdateResult(
            label=measurement.label,
            time_s=measurement.time_s,
            accepted=False,
            gate=gate,
            update=None,
        )

    update = ins.linear_update(
        residual=measurement.residual,
        H=measurement.H,
        R=measurement.R,
    )
    return FusionUpdateResult(
        label=measurement.label,
        time_s=measurement.time_s,
        accepted=True,
        gate=gate,
        update=update,
    )


def stack_independent_measurements(
    measurements: Sequence[LinearMeasurement],
    *,
    label: Optional[str] = None,
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Stack multiple independent measurements into one larger linear measurement.

    Parameters
    ----------
    measurements : sequence of LinearMeasurement
        Measurements to combine.
    label : str, optional
        Optional combined label. If omitted, labels are joined with `+`.
    time_s : float, optional
        Optional combined timestamp.

    Returns
    -------
    LinearMeasurement
        Stacked measurement with block-diagonal covariance.

    Assumption
    ----------
    The input measurements are assumed mutually independent, so:

        R_stacked = block_diag(R_1, ..., R_k)
    """
    if len(measurements) == 0:
        raise ValueError("measurements must contain at least one element.")

    z = np.concatenate([m.z for m in measurements]).astype(np.float64)
    h = np.concatenate([m.h for m in measurements]).astype(np.float64)
    H = np.vstack([m.H for m in measurements]).astype(np.float64)
    R = block_diag(*[m.R for m in measurements])

    if label is None:
        label = "+".join(m.label for m in measurements)
    if time_s is None:
        times = [m.time_s for m in measurements if m.time_s is not None]
        if len(times) > 0:
            time_s = float(times[0])

    return LinearMeasurement(
        label=label,
        z=z,
        h=h,
        H=H,
        R=R,
        time_s=time_s,
    )


# -----------------------------------------------------------------------------
# Body-frame velocity aiding
# -----------------------------------------------------------------------------


def velocity_measurement_model_body(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> FloatArray:
    r"""
    Predicted body-frame Earth-relative velocity.

    Parameters
    ----------
    ins_or_state : ErrorStateINS or ErrorStateINSState
        Filter or state.

    Returns
    -------
    np.ndarray, shape (3,)
        Predicted body-frame velocity [m/s].

    Formula
    -------
    With passive body->NED DCM `C_n_b`,

        v^b = C_b_n v^n = (C_n_b)^T v^n
    """
    state = _state_from_filter_or_state(ins_or_state)
    C_b_n = state.nominal.C_n_b.T
    return (C_b_n @ state.nominal.v_ned_mps).astype(np.float64)


def velocity_measurement_jacobian_body(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> FloatArray:
    r"""
    Jacobian for a direct body-frame velocity measurement.

    Predicted measurement
    ---------------------
        h(x) = C_b_n v^n

    Linearization
    -------------
    Under the same small navigation-frame attitude error convention used by the
    INS core, the first-order perturbation is:

        delta h ≈ C_b_n delta v^n - C_b_n [v^n]_x delta theta^n

    Therefore:
    - velocity block      = C_b_n
    - attitude block      = -C_b_n [v^n]_x
    """
    state = _state_from_filter_or_state(ins_or_state)
    C_b_n = state.nominal.C_n_b.T
    v_n = state.nominal.v_ned_mps

    H = np.zeros((3, ERROR_STATE_SIZE), dtype=np.float64)
    H[:, ERR_VEL] = C_b_n
    H[:, ERR_ATT] = -C_b_n @ skew(v_n)
    return H


def velocity_measurement_residual_body(
    measured_velocity_body_mps: ArrayLike,
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> FloatArray:
    """
    Residual for a direct body-frame Earth-relative velocity measurement.
    """
    z = _vec3(measured_velocity_body_mps, name="measured_velocity_body_mps")
    h = velocity_measurement_model_body(ins_or_state)
    return (z - h).astype(np.float64)


# -----------------------------------------------------------------------------
# Common measurement builders
# -----------------------------------------------------------------------------


def make_velocity_ned_measurement(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    measured_velocity_ned_mps: ArrayLike,
    R_mps2: ArrayLike,
    *,
    label: str = "velocity_ned",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Build a direct NED-velocity linear measurement.
    """
    state = _state_from_filter_or_state(ins_or_state)
    z = _vec3(measured_velocity_ned_mps, name="measured_velocity_ned_mps")
    h = velocity_measurement_model_ned(state)
    H = velocity_measurement_jacobian_ned()
    R = _mat(R_mps2, name="R_mps2", rows=3, cols=3)
    return LinearMeasurement(label=label, z=z, h=h, H=H, R=R, time_s=time_s)


def make_velocity_body_measurement(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    measured_velocity_body_mps: ArrayLike,
    R_mps2: ArrayLike,
    *,
    label: str = "velocity_body",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Build a direct body-frame Earth-relative velocity linear measurement.
    """
    state = _state_from_filter_or_state(ins_or_state)
    z = _vec3(measured_velocity_body_mps, name="measured_velocity_body_mps")
    h = velocity_measurement_model_body(state)
    H = velocity_measurement_jacobian_body(state)
    R = _mat(R_mps2, name="R_mps2", rows=3, cols=3)
    return LinearMeasurement(label=label, z=z, h=h, H=H, R=R, time_s=time_s)


def make_geodetic_position_measurement(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    measured_lat_rad: float,
    measured_lon_rad: float,
    measured_height_m: float,
    R: ArrayLike,
    *,
    label: str = "position_geodetic",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Build a direct geodetic-position linear measurement.
    """
    state = _state_from_filter_or_state(ins_or_state)
    z = np.array(
        [
            float(measured_lat_rad),
            float(measured_lon_rad),
            float(measured_height_m),
        ],
        dtype=np.float64,
    )
    h = position_measurement_model_geodetic(state)
    # Keep predicted and measured longitude consistent only through the residual,
    # not by forcibly wrapping z itself.
    residual = position_measurement_residual_geodetic(
        measured_lat_rad=measured_lat_rad,
        measured_lon_rad=measured_lon_rad,
        measured_height_m=measured_height_m,
        state=state,
    )
    H = position_measurement_jacobian_geodetic()
    Rm = _mat(R, name="R", rows=3, cols=3)
    return LinearMeasurement(
        label=label,
        z=residual,
        h=np.zeros(3, dtype=np.float64),
        H=H,
        R=Rm,
        time_s=time_s,
    )


def make_depth_measurement(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    measured_depth_m: float,
    depth_variance_m2: float,
    *,
    reference_surface_height_m: float = 0.0,
    label: str = "depth",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Build a scalar signed-depth linear measurement.
    """
    state = _state_from_filter_or_state(ins_or_state)
    if depth_variance_m2 < 0.0:
        raise ValueError("depth_variance_m2 must be nonnegative.")

    residual = depth_measurement_residual(
        measured_depth_m=measured_depth_m,
        state=state,
        reference_surface_height_m=reference_surface_height_m,
    )
    H = depth_measurement_jacobian()
    R = np.array([[float(depth_variance_m2)]], dtype=np.float64)

    return LinearMeasurement(
        label=label,
        z=residual,
        h=np.zeros(1, dtype=np.float64),
        H=H,
        R=R,
        time_s=time_s,
    )


def make_custom_scalar_measurement(
    measured_value: float,
    predicted_value: float,
    H_row: ArrayLike,
    variance: float,
    *,
    label: str = "custom_scalar",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    r"""
    Build a custom scalar linear measurement.

    This is the hook later gravity-map matching code can use when it has:
    - a scalar measurement `z`
    - a scalar predicted value `h`
    - a 1x15 Jacobian row
    - a scalar variance

    Formula
    -------
        r = z - h
    """
    if variance < 0.0:
        raise ValueError("variance must be nonnegative.")

    return LinearMeasurement(
        label=label,
        z=np.array([float(measured_value)], dtype=np.float64),
        h=np.array([float(predicted_value)], dtype=np.float64),
        H=_row(H_row, name="H_row").reshape(1, -1),
        R=np.array([[float(variance)]], dtype=np.float64),
        time_s=time_s,
    )


def make_custom_vector_measurement(
    measured_vector: ArrayLike,
    predicted_vector: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    *,
    label: str = "custom_vector",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    """
    Build a custom vector linear measurement.
    """
    return LinearMeasurement(
        label=label,
        z=_vec(measured_vector, name="measured_vector"),
        h=_vec(predicted_vector, name="predicted_vector"),
        H=_as_float_array(H),
        R=_as_float_array(R),
        time_s=time_s,
    )


def make_horizontal_ned_position_measurement(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    measured_horizontal_offset_ned_m: ArrayLike,
    R_horizontal_m2: ArrayLike,
    *,
    label: str = "position_ned_horizontal",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    r"""
    Build a horizontal local-NED position measurement.

    The measurement model is:

        z = [dN, dE]^T
        h = 0
        z = H delta_x + eps

    where the error-state position block is geodetic `[d_lat, d_lon, d_h]` and
    the local horizontal mapping is:

        dN = (R_M + h) d_lat
        dE = (R_N + h) cos(lat) d_lon
    """
    from ..physics.earth import meridian_radius, prime_vertical_radius

    state = _state_from_filter_or_state(ins_or_state)
    z = _as_float_array(measured_horizontal_offset_ned_m).reshape(-1)
    if z.shape != (2,):
        raise ValueError(
            "measured_horizontal_offset_ned_m must have shape (2,), "
            f"got {z.shape}."
        )

    Rm = _mat(R_horizontal_m2, name="R_horizontal_m2", rows=2, cols=2)

    phi = float(state.nominal.lat_rad)
    h = float(state.nominal.height_m)
    M = float(meridian_radius(phi))
    N_pv = float(prime_vertical_radius(phi))
    cos_phi = max(abs(float(np.cos(phi))), 1.0e-8)

    H = np.zeros((2, ERROR_STATE_SIZE), dtype=np.float64)
    H[0, ERR_POS.start + 0] = M + h
    H[1, ERR_POS.start + 1] = (N_pv + h) * cos_phi

    return LinearMeasurement(
        label=label,
        z=z.astype(np.float64),
        h=np.zeros(2, dtype=np.float64),
        H=H,
        R=Rm,
        time_s=time_s,
    )


# -----------------------------------------------------------------------------
# Common measurement application wrappers
# -----------------------------------------------------------------------------


def apply_velocity_ned_measurement(
    ins: ErrorStateINS,
    measured_velocity_ned_mps: ArrayLike,
    R_mps2: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: str = "velocity_ned",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply a direct NED velocity update.
    """
    meas = make_velocity_ned_measurement(
        ins,
        measured_velocity_ned_mps,
        R_mps2,
        label=label,
        time_s=time_s,
    )
    return apply_linear_measurement(ins, meas, nis_threshold=nis_threshold)


def apply_velocity_ned_measurement_velocity_only(
    ins: ErrorStateINS,
    measured_velocity_ned_mps: ArrayLike,
    R_mps2: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: str = "velocity_ned_velocity_only",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply a direct NED velocity update while only correcting the velocity state.

    This is a conservative stabilization path for the current repository: it
    prevents velocity measurements from forcing large bias/attitude corrections
    through imperfect cross-covariances in the early-stage INS error model.
    """
    meas = make_velocity_ned_measurement(
        ins,
        measured_velocity_ned_mps,
        R_mps2,
        label=label,
        time_s=time_s,
    )
    gate = gate_measurement(
        meas,
        ins.covariance,
        nis_threshold=nis_threshold,
    )

    if not gate.accepted:
        return FusionUpdateResult(
            label=meas.label,
            time_s=meas.time_s,
            accepted=False,
            gate=gate,
            update=None,
        )

    P = ins.state.P
    R = np.asarray(meas.R, dtype=np.float64)
    residual = np.asarray(meas.residual, dtype=np.float64)

    P_vv = P[ERR_VEL, ERR_VEL]
    S = _symmetrize(P_vv + R)
    K_vv = P_vv @ np.linalg.inv(S)

    K = np.zeros((ERROR_STATE_SIZE, 3), dtype=np.float64)
    K[ERR_VEL, :] = K_vv

    delta_x = np.zeros(ERROR_STATE_SIZE, dtype=np.float64)
    delta_x[ERR_VEL] = K_vv @ residual
    ins.inject_error_state(delta_x)

    I = np.eye(ERROR_STATE_SIZE, dtype=np.float64)
    KH = K @ meas.H
    P_new = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
    ins.state.P = _symmetrize(P_new)

    return FusionUpdateResult(
        label=meas.label,
        time_s=meas.time_s,
        accepted=True,
        gate=gate,
        update=LinearizedMeasurementUpdate(
            residual=residual,
            innovation_covariance=S,
            kalman_gain=K,
            delta_x=delta_x,
        ),
    )


def apply_velocity_body_measurement(
    ins: ErrorStateINS,
    measured_velocity_body_mps: ArrayLike,
    R_mps2: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: str = "velocity_body",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply a direct body-frame Earth-relative velocity update.
    """
    meas = make_velocity_body_measurement(
        ins,
        measured_velocity_body_mps,
        R_mps2,
        label=label,
        time_s=time_s,
    )
    return apply_linear_measurement(ins, meas, nis_threshold=nis_threshold)


def apply_velocity_body_measurement_velocity_only(
    ins: ErrorStateINS,
    measured_velocity_body_mps: ArrayLike,
    R_mps2: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: str = "velocity_body_velocity_only",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Conservative body-frame velocity aid variant.

    The measured body-frame velocity is rotated into NED using the current
    nominal attitude, then fused as a velocity-only NED update.
    """
    z_body = _vec3(measured_velocity_body_mps, name="measured_velocity_body_mps")
    z_ned = ins.nominal.C_n_b @ z_body
    return apply_velocity_ned_measurement_velocity_only(
        ins,
        z_ned,
        R_mps2,
        nis_threshold=nis_threshold,
        label=label,
        time_s=time_s,
    )


def apply_velocity_aid_measurement(
    ins: ErrorStateINS,
    measurement: VelocityAidMeasurement,
    R_mps2: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: Optional[str] = None,
) -> FusionUpdateResult:
    """
    Apply a `VelocityAidMeasurement` from the sensor layer.

    Parameters
    ----------
    ins : ErrorStateINS
        Target filter.
    measurement : VelocityAidMeasurement
        Sensor-layer measurement.
    R_mps2 : array-like, shape (3, 3)
        Measurement covariance [m^2/s^2].
    nis_threshold : float, optional
        Optional NIS gate threshold.
    label : str, optional
        Optional override label.

    Behavior
    --------
    Dispatches on `measurement.frame`:
    - `"ned"`  -> direct NED velocity update
    - `"body"` -> direct body-frame Earth-relative velocity update
    """
    resolved_label = measurement.frame if label is None else str(label)

    if measurement.frame == "ned":
        return apply_velocity_ned_measurement(
            ins,
            measurement.value_mps,
            R_mps2,
            nis_threshold=nis_threshold,
            label=resolved_label,
            time_s=measurement.time_s,
        )

    if measurement.frame == "body":
        return apply_velocity_body_measurement(
            ins,
            measurement.value_mps,
            R_mps2,
            nis_threshold=nis_threshold,
            label=resolved_label,
            time_s=measurement.time_s,
        )

    raise ValueError(
        f"Unsupported VelocityAidMeasurement.frame={measurement.frame!r}. "
        "Expected 'ned' or 'body'."
    )


def apply_velocity_aid_measurement_velocity_only(
    ins: ErrorStateINS,
    measurement: VelocityAidMeasurement,
    R_mps2: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: Optional[str] = None,
) -> FusionUpdateResult:
    """
    Velocity-only constrained variant of :func:`apply_velocity_aid_measurement`.
    """
    resolved_label = measurement.frame if label is None else str(label)

    if measurement.frame == "ned":
        return apply_velocity_ned_measurement_velocity_only(
            ins,
            measurement.value_mps,
            R_mps2,
            nis_threshold=nis_threshold,
            label=resolved_label,
            time_s=measurement.time_s,
        )

    if measurement.frame == "body":
        return apply_velocity_body_measurement_velocity_only(
            ins,
            measurement.value_mps,
            R_mps2,
            nis_threshold=nis_threshold,
            label=resolved_label,
            time_s=measurement.time_s,
        )

    raise ValueError(
        f"Unsupported VelocityAidMeasurement.frame={measurement.frame!r}. "
        "Expected 'ned' or 'body'."
    )


def apply_geodetic_position_measurement(
    ins: ErrorStateINS,
    measured_lat_rad: float,
    measured_lon_rad: float,
    measured_height_m: float,
    R: ArrayLike,
    *,
    nis_threshold: Optional[float] = None,
    label: str = "position_geodetic",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply a direct geodetic position update.
    """
    meas = make_geodetic_position_measurement(
        ins,
        measured_lat_rad,
        measured_lon_rad,
        measured_height_m,
        R,
        label=label,
        time_s=time_s,
    )
    return apply_linear_measurement(ins, meas, nis_threshold=nis_threshold)


def apply_depth_measurement(
    ins: ErrorStateINS,
    measured_depth_m: float,
    depth_variance_m2: float,
    *,
    reference_surface_height_m: float = 0.0,
    nis_threshold: Optional[float] = None,
    label: str = "depth",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply a scalar signed-depth update.
    """
    meas = make_depth_measurement(
        ins,
        measured_depth_m,
        depth_variance_m2,
        reference_surface_height_m=reference_surface_height_m,
        label=label,
        time_s=time_s,
    )
    return apply_linear_measurement(ins, meas, nis_threshold=nis_threshold)


def apply_depth_measurement_height_only(
    ins: ErrorStateINS,
    measured_depth_m: float,
    depth_variance_m2: float,
    *,
    reference_surface_height_m: float = 0.0,
    nis_threshold: Optional[float] = None,
    label: str = "depth_height_only",
    time_s: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Apply a scalar depth update while only correcting the height channel.

    This is a pragmatic safeguard for the current early-stage INS model. Depth
    is a vertical aid; constraining the correction to the height state prevents
    a single scalar depth residual from pulling the horizontal states around via
    imperfect cross-covariances.
    """
    if depth_variance_m2 < 0.0:
        raise ValueError("depth_variance_m2 must be nonnegative.")

    meas = make_depth_measurement(
        ins,
        measured_depth_m,
        depth_variance_m2,
        reference_surface_height_m=reference_surface_height_m,
        label=label,
        time_s=time_s,
    )
    gate = gate_measurement(
        meas,
        ins.covariance,
        nis_threshold=nis_threshold,
    )

    if not gate.accepted:
        return FusionUpdateResult(
            label=meas.label,
            time_s=meas.time_s,
            accepted=False,
            gate=gate,
            update=None,
        )

    P = ins.state.P
    residual = float(meas.residual[0])
    R = float(meas.R[0, 0])

    # Signed depth uses d = h_ref - h, so the measurement Jacobian with respect
    # to the height error state is H_h = -1.
    p_hh = float(P[2, 2])
    S = float(p_hh + R)
    k_h = -p_hh / S

    K = np.zeros((ERROR_STATE_SIZE, 1), dtype=np.float64)
    K[2, 0] = k_h

    delta_x = np.zeros(ERROR_STATE_SIZE, dtype=np.float64)
    delta_x[2] = k_h * residual
    ins.inject_error_state(delta_x)

    I = np.eye(ERROR_STATE_SIZE, dtype=np.float64)
    KH = K @ meas.H
    P_new = (I - KH) @ P @ (I - KH).T + K @ meas.R @ K.T
    ins.state.P = 0.5 * (P_new + P_new.T)

    return FusionUpdateResult(
        label=meas.label,
        time_s=meas.time_s,
        accepted=True,
        gate=gate,
        update=LinearizedMeasurementUpdate(
            residual=meas.residual.copy(),
            innovation_covariance=np.array([[S]], dtype=np.float64),
            kalman_gain=K,
            delta_x=delta_x,
        ),
    )


def apply_depth_sensor_measurement(
    ins: ErrorStateINS,
    measurement: DepthMeasurement,
    depth_variance_m2: float,
    *,
    reference_surface_height_m: Optional[float] = None,
    nis_threshold: Optional[float] = None,
    label: str = "depth",
) -> FusionUpdateResult:
    """
    Apply a `DepthMeasurement` from the sensor layer.
    """
    href = (
        measurement.reference_surface_height_m
        if reference_surface_height_m is None
        else float(reference_surface_height_m)
    )
    return apply_depth_measurement(
        ins,
        measured_depth_m=measurement.value_m,
        depth_variance_m2=depth_variance_m2,
        reference_surface_height_m=href,
        nis_threshold=nis_threshold,
        label=label,
        time_s=measurement.time_s,
    )


def apply_depth_sensor_measurement_height_only(
    ins: ErrorStateINS,
    measurement: DepthMeasurement,
    depth_variance_m2: float,
    *,
    reference_surface_height_m: Optional[float] = None,
    nis_threshold: Optional[float] = None,
    label: str = "depth_height_only",
) -> FusionUpdateResult:
    """
    Height-only variant of :func:`apply_depth_sensor_measurement`.
    """
    href = (
        measurement.reference_surface_height_m
        if reference_surface_height_m is None
        else float(reference_surface_height_m)
    )
    return apply_depth_measurement_height_only(
        ins,
        measured_depth_m=measurement.value_m,
        depth_variance_m2=depth_variance_m2,
        reference_surface_height_m=href,
        nis_threshold=nis_threshold,
        label=label,
        time_s=measurement.time_s,
    )


def apply_custom_linear_measurement(
    ins: ErrorStateINS,
    measurement: LinearMeasurement,
    *,
    nis_threshold: Optional[float] = None,
) -> FusionUpdateResult:
    """
    Alias for `apply_linear_measurement(...)` for semantic clarity in later code.
    """
    return apply_linear_measurement(ins, measurement, nis_threshold=nis_threshold)


# -----------------------------------------------------------------------------
# Lightweight measurement summaries for logging/debugging
# -----------------------------------------------------------------------------


def summarize_measurement(measurement: LinearMeasurement) -> dict[str, object]:
    """
    Return a lightweight serializable summary of a measurement package.
    """
    return {
        "label": measurement.label,
        "time_s": measurement.time_s,
        "dimension": measurement.dimension,
        "z": measurement.z.copy(),
        "h": measurement.h.copy(),
        "residual": measurement.residual.copy(),
        "H": measurement.H.copy(),
        "R": measurement.R.copy(),
    }


def summarize_update_result(result: FusionUpdateResult) -> dict[str, object]:
    """
    Return a lightweight serializable summary of a fusion result.
    """
    out: dict[str, object] = {
        "label": result.label,
        "time_s": result.time_s,
        "accepted": result.accepted,
        "nis": result.gate.nis,
        "nis_threshold": result.gate.threshold,
    }

    if result.update is not None:
        out["residual"] = result.update.residual.copy()
        out["innovation_covariance"] = result.update.innovation_covariance.copy()
        out["kalman_gain"] = result.update.kalman_gain.copy()
        out["delta_x"] = result.update.delta_x.copy()

    return out


# -----------------------------------------------------------------------------
# Directional pseudo-measurement from PF posterior
# -----------------------------------------------------------------------------


def make_directional_position_measurement(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
    direction_ned: ArrayLike,
    projected_offset_m: float,
    projected_variance_m2: float,
    *,
    label: str = "pf_directional",
    time_s: Optional[float] = None,
) -> LinearMeasurement:
    r"""
    Build a rank-1 directional position measurement for PF-to-INS feedback.

    Instead of injecting the full 3D PF posterior as a pseudo-position
    measurement (which fails when the posterior is ridge-shaped), this
    injects a scalar measurement along the well-constrained direction only.

    Parameters
    ----------
    ins_or_state : ErrorStateINS or ErrorStateINSState
        Current INS state.
    direction_ned : array-like, shape (3,)
        Unit vector in NED indicating the well-constrained direction
        (typically the smallest-eigenvalue eigenvector of the PF posterior
        NED covariance).
    projected_offset_m : float
        Scalar offset = direction^T @ (PF_mean_ned - INS_pos_ned).
        Positive means the PF thinks the true position is displaced in the
        positive direction.
    projected_variance_m2 : float
        PF posterior variance along this direction [m^2], typically the
        smallest eigenvalue of the PF NED covariance times an inflation
        factor.
    label : str
        Human-readable label.
    time_s : float, optional
        Timestamp.

    Returns
    -------
    LinearMeasurement
        A scalar measurement suitable for `apply_linear_measurement(...)`.

    Measurement model
    -----------------
    The measurement is:

        z = e^T @ (true_pos_ned - INS_pos_ned) + eps
        h = 0  (predicted under zero error assumption)
        H[ERR_POS] = e^T @ J_{ned->geo}  (maps NED direction to geodetic pos block)
        R = [[projected_variance_m2]]

    where e is the direction vector and J_{ned->geo} maps geodetic position
    errors to NED offsets.
    """
    from ..physics.earth import meridian_radius, prime_vertical_radius

    state = _state_from_filter_or_state(ins_or_state)
    e = _vec3(direction_ned, name="direction_ned")
    e_norm = float(np.linalg.norm(e))
    if e_norm < 1.0e-12:
        raise ValueError("direction_ned must be a non-zero vector.")
    e = e / e_norm

    phi = state.nominal.lat_rad
    h = state.nominal.height_m
    M = float(meridian_radius(phi))
    N_pv = float(prime_vertical_radius(phi))
    cos_phi = max(abs(float(np.cos(phi))), 1.0e-8)

    # Jacobian mapping geodetic pos errors to NED:
    #   dN = d_lat * (M + h)
    #   dE = d_lon * (N + h) * cos(phi)
    #   dD = -d_h
    J_ned_geo = np.array([
        [M + h, 0.0, 0.0],
        [0.0, (N_pv + h) * cos_phi, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    # H row: z = e^T @ J_ned_geo @ delta_pos_geo
    H_row = np.zeros(ERROR_STATE_SIZE, dtype=np.float64)
    H_row[ERR_POS] = e @ J_ned_geo

    z = np.array([float(projected_offset_m)], dtype=np.float64)
    h_pred = np.zeros(1, dtype=np.float64)
    H = H_row.reshape(1, ERROR_STATE_SIZE)
    R = np.array([[float(projected_variance_m2)]], dtype=np.float64)

    return LinearMeasurement(
        label=label,
        z=z,
        h=h_pred,
        H=H,
        R=R,
        time_s=time_s,
    )


__all__ = [
    "FloatArray",
    "FusionUpdateResult",
    "InnovationGateResult",
    "LinearMeasurement",
    "apply_custom_linear_measurement",
    "apply_depth_measurement",
    "apply_depth_measurement_height_only",
    "apply_depth_sensor_measurement",
    "apply_depth_sensor_measurement_height_only",
    "apply_geodetic_position_measurement",
    "apply_linear_measurement",
    "apply_velocity_aid_measurement",
    "apply_velocity_aid_measurement_velocity_only",
    "apply_velocity_body_measurement",
    "apply_velocity_body_measurement_velocity_only",
    "apply_velocity_ned_measurement",
    "apply_velocity_ned_measurement_velocity_only",
    "block_diag",
    "diagonal_covariance_from_std",
    "diagonal_covariance_from_variance",
    "gate_measurement",
    "innovation_covariance",
    "make_custom_scalar_measurement",
    "make_custom_vector_measurement",
    "make_depth_measurement",
    "make_horizontal_ned_position_measurement",
    "make_directional_position_measurement",
    "make_geodetic_position_measurement",
    "make_linear_measurement",
    "make_velocity_body_measurement",
    "make_velocity_ned_measurement",
    "normalized_innovation_squared",
    "stack_independent_measurements",
    "summarize_measurement",
    "summarize_update_result",
    "velocity_measurement_jacobian_body",
    "velocity_measurement_model_body",
    "velocity_measurement_residual_body",
]
