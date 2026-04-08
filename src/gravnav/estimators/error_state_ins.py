"""
error_state_ins.py

Nominal local-level INS propagation and a compact 15-state error-state model
for the gravity-aided navigation simulator.

This module is the bridge between:
- the already-implemented Earth / frame / kinematics / truth / IMU layers, and
- later estimator fusion layers such as:
    - external velocity aiding
    - depth aiding
    - gravity map matching
    - integrity monitoring

Why this file exists
--------------------
The repository now has:
- trustworthy WGS 84 geodesy and normal gravity
- consistent ECEF / NED / body frame transforms
- truth trajectories containing:
    lat, lon, h, v^n, vdot^n, C_n_b, omega_nb^b
- IMU sensor models that generate omega_ib^b and f_ib^b

The next missing estimator layer is therefore:
1) propagate a nominal strapdown INS state in local NED coordinates
2) propagate the covariance of a small linearized error state
3) provide reusable measurement-model/Jacobian helpers for later fusion

Conventions
-----------
Frames
~~~~~~
- Navigation frame is local NED:
      x = North, y = East, z = Down
- Body frame is right-handed aviation-style:
      x = forward, y = right, z = down
- `C_n_b` is the passive body->NED DCM such that:

      v^n = C_n_b v^b

Coordinates
~~~~~~~~~~~
The nominal state uses geodetic coordinates:
- latitude  phi [rad]
- longitude lambda [rad]
- ellipsoidal height h [m], positive upward

Velocities are always Earth-relative NED velocities [m/s].

Nominal INS state
-----------------
The propagated nominal state is:

    x_nom =
    [ lat, lon, h, v_N, v_E, v_D, C_n_b, b_g^b, b_a^b ]

where:
- b_g^b is gyro bias resolved in body [rad/s]
- b_a^b is accelerometer bias resolved in body [m/s^2]

The nominal propagation uses the standard local-navigation-frame equations:

    dot(v)^n = f^n - (2 omega_ie^n + omega_en^n) x v^n + g^n

with:
    f^n = C_n_b f^b

and geodetic position rates:

    dot(phi)    = v_N / (M + h)
    dot(lambda) = v_E / ((N + h) cos(phi))
    dot(h)      = -v_D

The body rate with respect to the navigation frame is:

    omega_nb^b = omega_ib^b - C_b_n (omega_ie^n + omega_en^n)

and the body->NED attitude update is approximated by a right-multiplied
incremental body rotation:

    C_n_b(k+1) ≈ C_n_b(k) Exp([omega_nb^b dt]_x)

Small-angle error state
-----------------------
This file uses a compact 15-state error model:

    delta_x =
    [ d_lat, d_lon, d_h,
      d_v_N, d_v_E, d_v_D,
      d_theta_N, d_theta_E, d_theta_D,
      d_bg_x, d_bg_y, d_bg_z,
      d_ba_x, d_ba_y, d_ba_z ]^T

Interpretation
--------------
- `d_theta_n` is a small navigation-frame attitude error used in a
  left-multiplicative correction:
      C_n_b <- Exp(-[d_theta_n]_x) C_n_b
- The linearized Jacobian implemented here is intentionally pragmatic:
  it captures the dominant couplings needed for an early repository-wide INS /
  fusion stack, while omitting some higher-order curvature and Schuler terms
  that can be added later.

Process noise model
-------------------
The continuous driving-noise vector is:

    w =
    [ n_g, n_a, n_bg, n_ba ]^T

where:
- n_g  : gyro white noise, body frame [rad/s / sqrt(Hz)]
- n_a  : accel white noise, body frame [m/s^2 / sqrt(Hz)]
- n_bg : gyro bias random walk [rad/s / sqrt(s)]
- n_ba : accel bias random walk [m/s^2 / sqrt(s)]

Primary references used here
----------------------------
1) INSTINCT / University of Stuttgart,
   "INS/GNSS Loosely-coupled Kalman Filter (local-navigation frame)"
   URL:
   https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

   Used for:
   - local-navigation-frame velocity equation
   - geodetic position-rate equations
   - transport-rate notation and local-level conventions

2) INSTINCT / University of Stuttgart,
   "IMU Integrator (local-navigation frame)"
   URL:
   https://unistuttgart-ins.github.io/INSTINCT/ImuIntegrator_n.html

   Used for:
   - local-level attitude-rate equation
   - relation:
         omega_nb^b = omega_ib^b - C_b_n (omega_ie^n + omega_en^n)

3) Groves / classical INS error-state practice
   The exact full F-matrix varies across formulations, but the central idea is
   the same: propagate a nominal nonlinear INS and a linearized small error state
   through the navigation equations. This module implements a compact, estimator-
   friendly form of that idea suitable for the repository's current maturity.

4) Kalibr Wiki, "IMU Noise Model"
   URL:
   https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

   Used for the practical white-noise and bias-random-walk discretization
   conventions already used elsewhere in the repository.

Design notes
------------
- This is not yet a production-grade full Schuler-tuned navigation package.
- The emphasis is consistency with the existing repository conventions.
- The nominal propagation is physically meaningful and directly reusable now.
- The error-state Jacobian is intentionally transparent and compact rather than
  maximally elaborate.
- Later files (`fusion.py`, `map_match_pf.py`) can build on the generic linear
  update helpers defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import (
    meridian_radius,
    normal_gravity,
    normal_gravity_vertical_gradient,
    prime_vertical_radius,
)
from ..physics.frames import (
    dcm_from_rotvec,
    earth_rate_ned,
    geodetic_rates_from_ned_velocity,
    navigation_frame_rate_ned,
    project_to_so3,
    skew,
    transport_rate_ned,
    wrap_angle_pi,
)
from ..truth.trajectory import TruthTrajectory

FloatArray = NDArray[np.float64]

# Error-state indexing.
ERR_POS = slice(0, 3)
ERR_VEL = slice(3, 6)
ERR_ATT = slice(6, 9)
ERR_BG = slice(9, 12)
ERR_BA = slice(12, 15)

ERROR_STATE_SIZE = 15
PROCESS_NOISE_SIZE = 12


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


def _axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Convert a scalar or 3-vector into a 3-vector.

    This is convenient for process-noise specifications because many specs are
    given as one scalar repeated across all three axes.
    """
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}.")
    return arr


def _nonnegative_axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """Like `_axis3(...)`, but require all entries to be nonnegative."""
    arr = _axis3(x, name=name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative, got {arr}.")
    return arr


def _covariance(P: ArrayLike, *, name: str, n: int) -> FloatArray:
    """Validate a square covariance matrix."""
    arr = _as_float_array(P)
    if arr.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {arr.shape}.")
    return _symmetrize(arr)


def _symmetrize(M: ArrayLike) -> FloatArray:
    """Return the symmetric part of a square matrix."""
    A = _as_float_array(M)
    return 0.5 * (A + A.T)


def _check_latitude(lat_rad: float, *, name: str = "lat_rad") -> float:
    """Validate geodetic latitude."""
    lat = float(lat_rad)
    if abs(lat) > 0.5 * np.pi + 1e-12:
        raise ValueError(f"{name} must lie in [-pi/2, pi/2], got {lat}.")
    return lat


def _gravity_vector_ned(lat_rad: float, height_m: float) -> FloatArray:
    """
    Return normal gravity vector resolved in NED.

    In NED, Down is positive, so:

        g^n = [0, 0, +g]^T
    """
    return np.array(
        [0.0, 0.0, float(normal_gravity(lat_rad, height_m))],
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ErrorStateINSNominalState:
    """
    Nominal local-level INS state.

    Attributes
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    lon_rad : float
        Longitude [rad], east-positive.
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : np.ndarray, shape (3,)
        Earth-relative NED velocity [m/s].
    C_n_b : np.ndarray, shape (3, 3)
        Passive body->NED DCM.
    gyro_bias_radps : np.ndarray, shape (3,)
        Gyroscope bias resolved in body coordinates [rad/s].
    accel_bias_mps2 : np.ndarray, shape (3,)
        Accelerometer bias resolved in body coordinates [m/s^2].
    """

    lat_rad: float
    lon_rad: float
    height_m: float
    v_ned_mps: FloatArray
    C_n_b: FloatArray
    gyro_bias_radps: FloatArray
    accel_bias_mps2: FloatArray

    def __post_init__(self) -> None:
        self.lat_rad = _check_latitude(self.lat_rad)
        self.lon_rad = float(wrap_angle_pi(self.lon_rad))
        self.height_m = float(self.height_m)
        self.v_ned_mps = _vec3(self.v_ned_mps, name="v_ned_mps")
        self.C_n_b = project_to_so3(_mat3(self.C_n_b, name="C_n_b"))
        self.gyro_bias_radps = _vec3(self.gyro_bias_radps, name="gyro_bias_radps")
        self.accel_bias_mps2 = _vec3(self.accel_bias_mps2, name="accel_bias_mps2")

    def copy(self) -> "ErrorStateINSNominalState":
        """Deep copy of the nominal state."""
        return ErrorStateINSNominalState(
            lat_rad=float(self.lat_rad),
            lon_rad=float(self.lon_rad),
            height_m=float(self.height_m),
            v_ned_mps=self.v_ned_mps.copy(),
            C_n_b=self.C_n_b.copy(),
            gyro_bias_radps=self.gyro_bias_radps.copy(),
            accel_bias_mps2=self.accel_bias_mps2.copy(),
        )

    @classmethod
    def from_truth_trajectory_start(
        cls,
        truth: TruthTrajectory,
        *,
        gyro_bias_radps: ArrayLike | float = 0.0,
        accel_bias_mps2: ArrayLike | float = 0.0,
    ) -> "ErrorStateINSNominalState":
        """
        Build the nominal state from the first sample of a truth trajectory.
        """
        if len(truth) < 1:
            raise ValueError("truth trajectory must contain at least one sample.")
        return cls(
            lat_rad=float(truth.lat_rad[0]),
            lon_rad=float(truth.lon_rad[0]),
            height_m=float(truth.height_m[0]),
            v_ned_mps=truth.v_ned_mps[0],
            C_n_b=truth.C_n_b[0],
            gyro_bias_radps=_axis3(gyro_bias_radps, name="gyro_bias_radps"),
            accel_bias_mps2=_axis3(accel_bias_mps2, name="accel_bias_mps2"),
        )


@dataclass
class ErrorStateINSProcessNoise:
    """
    Continuous-time process-noise specification for the error-state INS.

    Parameters
    ----------
    gyro_white_noise_radps_per_sqrt_hz : scalar or shape (3,), default=0.0
        Gyroscope white-noise density [rad/s / sqrt(Hz)].
    accel_white_noise_mps2_per_sqrt_hz : scalar or shape (3,), default=0.0
        Accelerometer white-noise density [m/s^2 / sqrt(Hz)].
    gyro_bias_random_walk_radps_per_sqrt_s : scalar or shape (3,), default=0.0
        Gyroscope bias random-walk coefficient [rad/s / sqrt(s)].
    accel_bias_random_walk_mps2_per_sqrt_s : scalar or shape (3,), default=0.0
        Accelerometer bias random-walk coefficient [m/s^2 / sqrt(s)].
    name : str, default="ins_process_noise"
        Human-readable identifier.
    """

    gyro_white_noise_radps_per_sqrt_hz: ArrayLike | float = 0.0
    accel_white_noise_mps2_per_sqrt_hz: ArrayLike | float = 0.0
    gyro_bias_random_walk_radps_per_sqrt_s: ArrayLike | float = 0.0
    accel_bias_random_walk_mps2_per_sqrt_s: ArrayLike | float = 0.0
    name: str = "ins_process_noise"

    def __post_init__(self) -> None:
        self.gyro_white_noise_radps_per_sqrt_hz = _nonnegative_axis3(
            self.gyro_white_noise_radps_per_sqrt_hz,
            name="gyro_white_noise_radps_per_sqrt_hz",
        )
        self.accel_white_noise_mps2_per_sqrt_hz = _nonnegative_axis3(
            self.accel_white_noise_mps2_per_sqrt_hz,
            name="accel_white_noise_mps2_per_sqrt_hz",
        )
        self.gyro_bias_random_walk_radps_per_sqrt_s = _nonnegative_axis3(
            self.gyro_bias_random_walk_radps_per_sqrt_s,
            name="gyro_bias_random_walk_radps_per_sqrt_s",
        )
        self.accel_bias_random_walk_mps2_per_sqrt_s = _nonnegative_axis3(
            self.accel_bias_random_walk_mps2_per_sqrt_s,
            name="accel_bias_random_walk_mps2_per_sqrt_s",
        )

    @classmethod
    def perfect(cls, name: str = "perfect_ins_process_noise") -> "ErrorStateINSProcessNoise":
        """Return a zero-noise process model."""
        return cls(name=name)

    @property
    def Qc(self) -> FloatArray:
        """
        Continuous driving-noise covariance for:

            w = [n_g, n_a, n_bg, n_ba]

        Shape
        -----
        (12, 12)
        """
        q = np.concatenate(
            [
                self.gyro_white_noise_radps_per_sqrt_hz ** 2,
                self.accel_white_noise_mps2_per_sqrt_hz ** 2,
                self.gyro_bias_random_walk_radps_per_sqrt_s ** 2,
                self.accel_bias_random_walk_mps2_per_sqrt_s ** 2,
            ]
        )
        return np.diag(q.astype(np.float64))


@dataclass
class NominalStateDerivative:
    """
    Instantaneous nominal-state derivative and intermediate navigation quantities.

    Attributes
    ----------
    lat_dot_radps : float
        Latitude rate [rad/s].
    lon_dot_radps : float
        Longitude rate [rad/s].
    height_dot_mps : float
        Height rate [m/s].
    v_dot_ned_mps2 : np.ndarray, shape (3,)
        NED velocity derivative [m/s^2].
    omega_nb_b_radps : np.ndarray, shape (3,)
        Body rate with respect to navigation frame, resolved in body [rad/s].
    corrected_omega_ib_b_radps : np.ndarray, shape (3,)
        Bias-corrected body inertial rate input [rad/s].
    corrected_f_ib_b_mps2 : np.ndarray, shape (3,)
        Bias-corrected body specific-force input [m/s^2].
    corrected_f_n_mps2 : np.ndarray, shape (3,)
        Bias-corrected specific force resolved in NED [m/s^2].
    gravity_ned_mps2 : np.ndarray, shape (3,)
        Normal gravity vector in NED [m/s^2].
    coriolis_transport_ned_mps2 : np.ndarray, shape (3,)
        Coriolis + transport term:
            (2 omega_ie^n + omega_en^n) x v^n
    """

    lat_dot_radps: float
    lon_dot_radps: float
    height_dot_mps: float
    v_dot_ned_mps2: FloatArray
    omega_nb_b_radps: FloatArray
    corrected_omega_ib_b_radps: FloatArray
    corrected_f_ib_b_mps2: FloatArray
    corrected_f_n_mps2: FloatArray
    gravity_ned_mps2: FloatArray
    coriolis_transport_ned_mps2: FloatArray


@dataclass
class ErrorStatePropagationMatrices:
    """
    Continuous and discrete linearized error-state propagation matrices.

    Attributes
    ----------
    F : np.ndarray, shape (15, 15)
        Continuous-time error-state Jacobian.
    G : np.ndarray, shape (15, 12)
        Continuous-time noise-input matrix.
    Qc : np.ndarray, shape (12, 12)
        Continuous driving-noise covariance.
    Phi : np.ndarray, shape (15, 15)
        Discrete transition matrix.
    Qd : np.ndarray, shape (15, 15)
        Discrete process-noise covariance.
    """

    F: FloatArray
    G: FloatArray
    Qc: FloatArray
    Phi: FloatArray
    Qd: FloatArray


@dataclass
class ErrorStateINSState:
    """
    Full filter state: nominal state + covariance + process-noise model.
    """

    nominal: ErrorStateINSNominalState
    P: FloatArray
    process_noise: ErrorStateINSProcessNoise

    def __post_init__(self) -> None:
        self.P = _covariance(self.P, name="P", n=ERROR_STATE_SIZE)

    def copy(self) -> "ErrorStateINSState":
        """Deep copy of the full estimator state."""
        return ErrorStateINSState(
            nominal=self.nominal.copy(),
            P=self.P.copy(),
            process_noise=self.process_noise,
        )


@dataclass
class LinearizedMeasurementUpdate:
    """
    Result of one linearized Kalman update.

    Attributes
    ----------
    residual : np.ndarray, shape (m,)
        Measurement residual r = z - h(x_nom).
    innovation_covariance : np.ndarray, shape (m, m)
        Innovation covariance S.
    kalman_gain : np.ndarray, shape (15, m)
        Kalman gain K.
    delta_x : np.ndarray, shape (15,)
        Closed-loop state correction applied to the nominal state.
    """

    residual: FloatArray
    innovation_covariance: FloatArray
    kalman_gain: FloatArray
    delta_x: FloatArray


# -----------------------------------------------------------------------------
# Core INS mechanization helpers
# -----------------------------------------------------------------------------


def corrected_imu_inputs(
    omega_ib_b_meas_radps: ArrayLike,
    f_ib_b_meas_mps2: ArrayLike,
    gyro_bias_radps: ArrayLike,
    accel_bias_mps2: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """
    Bias-correct IMU inputs.

    Parameters
    ----------
    omega_ib_b_meas_radps : array-like, shape (3,)
        Measured gyroscope input [rad/s].
    f_ib_b_meas_mps2 : array-like, shape (3,)
        Measured accelerometer specific force [m/s^2].
    gyro_bias_radps : array-like, shape (3,)
        Estimated gyro bias [rad/s].
    accel_bias_mps2 : array-like, shape (3,)
        Estimated accelerometer bias [m/s^2].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(omega_corr, f_corr)` both shape (3,).
    """
    omega = _vec3(omega_ib_b_meas_radps, name="omega_ib_b_meas_radps")
    f = _vec3(f_ib_b_meas_mps2, name="f_ib_b_meas_mps2")
    bg = _vec3(gyro_bias_radps, name="gyro_bias_radps")
    ba = _vec3(accel_bias_mps2, name="accel_bias_mps2")
    return omega - bg, f - ba


def nominal_state_derivative(
    state: ErrorStateINSNominalState,
    omega_ib_b_meas_radps: ArrayLike,
    f_ib_b_meas_mps2: ArrayLike,
) -> NominalStateDerivative:
    r"""
    Evaluate the nominal local-level INS derivative at the current state.

    Equations
    ---------
    Position:
        dot(phi)    = v_N / (M + h)
        dot(lambda) = v_E / ((N + h) cos(phi))
        dot(h)      = -v_D

    Velocity:
        dot(v)^n = f^n - (2 omega_ie^n + omega_en^n) x v^n + g^n

    Attitude input relation:
        omega_nb^b = omega_ib^b - C_b_n (omega_ie^n + omega_en^n)

    Parameters
    ----------
    state : ErrorStateINSNominalState
        Current nominal state.
    omega_ib_b_meas_radps : array-like, shape (3,)
        Gyroscope measurement [rad/s].
    f_ib_b_meas_mps2 : array-like, shape (3,)
        Accelerometer measurement [m/s^2].

    Returns
    -------
    NominalStateDerivative
        Derivative and useful intermediate terms.
    """
    omega_corr_b, f_corr_b = corrected_imu_inputs(
        omega_ib_b_meas_radps=omega_ib_b_meas_radps,
        f_ib_b_meas_mps2=f_ib_b_meas_mps2,
        gyro_bias_radps=state.gyro_bias_radps,
        accel_bias_mps2=state.accel_bias_mps2,
    )

    C_n_b = state.C_n_b
    C_b_n = C_n_b.T

    omega_ie_n = earth_rate_ned(state.lat_rad)
    omega_en_n = transport_rate_ned(
        state.lat_rad,
        state.height_m,
        state.v_ned_mps,
    )
    omega_in_n = omega_ie_n + omega_en_n

    omega_nb_b = omega_corr_b - C_b_n @ omega_in_n
    f_n = C_n_b @ f_corr_b
    g_n = _gravity_vector_ned(state.lat_rad, state.height_m)
    coriolis_transport_n = np.cross(2.0 * omega_ie_n + omega_en_n, state.v_ned_mps)
    v_dot_n = f_n - coriolis_transport_n + g_n

    lat_dot, lon_dot, h_dot = geodetic_rates_from_ned_velocity(
        state.lat_rad,
        state.height_m,
        state.v_ned_mps,
    )

    return NominalStateDerivative(
        lat_dot_radps=float(lat_dot),
        lon_dot_radps=float(lon_dot),
        height_dot_mps=float(h_dot),
        v_dot_ned_mps2=v_dot_n.astype(np.float64),
        omega_nb_b_radps=omega_nb_b.astype(np.float64),
        corrected_omega_ib_b_radps=omega_corr_b.astype(np.float64),
        corrected_f_ib_b_mps2=f_corr_b.astype(np.float64),
        corrected_f_n_mps2=f_n.astype(np.float64),
        gravity_ned_mps2=g_n.astype(np.float64),
        coriolis_transport_ned_mps2=coriolis_transport_n.astype(np.float64),
    )


def propagate_nominal_state(
    state: ErrorStateINSNominalState,
    omega_ib_b_meas_radps: ArrayLike,
    f_ib_b_meas_mps2: ArrayLike,
    dt_s: float,
) -> ErrorStateINSNominalState:
    r"""
    Propagate the nominal state by one time step.

    Integration method
    ------------------
    - attitude: right-multiplied body-frame increment from `omega_nb^b * dt`
    - velocity: forward Euler
    - geodetic position: midpoint velocity for reduced drift

    Parameters
    ----------
    state : ErrorStateINSNominalState
        Current nominal state.
    omega_ib_b_meas_radps : array-like, shape (3,)
        Gyroscope measurement [rad/s].
    f_ib_b_meas_mps2 : array-like, shape (3,)
        Accelerometer measurement [m/s^2].
    dt_s : float
        Time step [s].

    Returns
    -------
    ErrorStateINSNominalState
        Propagated nominal state.
    """
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")

    deriv = nominal_state_derivative(
        state=state,
        omega_ib_b_meas_radps=omega_ib_b_meas_radps,
        f_ib_b_meas_mps2=f_ib_b_meas_mps2,
    )

    # Velocity update.
    v_new = state.v_ned_mps + dt * deriv.v_dot_ned_mps2
    v_mid = 0.5 * (state.v_ned_mps + v_new)

    # Geodetic update using midpoint velocity.
    lat_dot_mid, lon_dot_mid, h_dot_mid = geodetic_rates_from_ned_velocity(
        state.lat_rad,
        state.height_m,
        v_mid,
    )

    lat_new = state.lat_rad + dt * float(lat_dot_mid)
    lon_new = float(wrap_angle_pi(state.lon_rad + dt * float(lon_dot_mid)))
    h_new = state.height_m + dt * float(h_dot_mid)

    # Attitude update:
    #   C_n_b(k+1) ≈ C_n_b(k) Exp([omega_nb^b dt]_x)
    C_new = project_to_so3(
        state.C_n_b @ dcm_from_rotvec(deriv.omega_nb_b_radps * dt)
    )

    return ErrorStateINSNominalState(
        lat_rad=float(lat_new),
        lon_rad=float(lon_new),
        height_m=float(h_new),
        v_ned_mps=v_new.astype(np.float64),
        C_n_b=C_new.astype(np.float64),
        gyro_bias_radps=state.gyro_bias_radps.copy(),
        accel_bias_mps2=state.accel_bias_mps2.copy(),
    )


# -----------------------------------------------------------------------------
# Linearized error-state propagation
# -----------------------------------------------------------------------------


def gravity_gradient_ned_height_only(
    lat_rad: float,
    height_m: float,
) -> FloatArray:
    r"""
    Return a simple local gravity-gradient approximation in NED.

    Model
    -----
    This repository currently uses a compact approximation that keeps only the
    dominant height sensitivity of the Down gravity component:

        g^n = [0, 0, gamma(phi, h)]^T

    so:

        d g^n / d [lat, lon, h] ≈
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, d gamma / d h]]

    where `d gamma / d h` is negative above the ellipsoid.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Mapping from `[d_lat, d_lon, d_h]` to gravity-vector perturbation.
    """
    dg_dh = float(normal_gravity_vertical_gradient(lat_rad))
    Gg = np.zeros((3, 3), dtype=np.float64)
    Gg[2, 2] = dg_dh
    return Gg


def continuous_error_state_jacobian(
    state: ErrorStateINSNominalState,
    omega_ib_b_meas_radps: ArrayLike,
    f_ib_b_meas_mps2: ArrayLike,
) -> FloatArray:
    r"""
    Build a compact continuous-time error-state Jacobian `F`.

    State ordering
    --------------
        [d_pos, d_vel, d_att, d_bg, d_ba]

    Dominant couplings included
    ---------------------------
    - position <- velocity
    - velocity <- velocity via Coriolis/transport
    - velocity <- attitude via specific-force misprojection
    - velocity <- accel bias
    - velocity <- height-to-gravity coupling
    - attitude <- attitude via local navigation-frame rotation
    - attitude <- gyro bias

    Important note
    --------------
    This is intentionally a compact Jacobian suitable for the repository's
    current stage. A later higher-fidelity version can add:
    - more complete curvature terms
    - full transport-rate partials
    - more detailed gravity partials
    - lever-arm coupling terms
    """
    lat = state.lat_rad
    h = state.height_m
    v_n = state.v_ned_mps
    C_n_b = state.C_n_b

    phi = lat
    cos_phi = float(np.cos(phi))
    if abs(cos_phi) < 1e-12:
        raise ValueError("continuous_error_state_jacobian is singular near the poles.")

    M = float(meridian_radius(phi))
    N = float(prime_vertical_radius(phi))

    deriv = nominal_state_derivative(
        state=state,
        omega_ib_b_meas_radps=omega_ib_b_meas_radps,
        f_ib_b_meas_mps2=f_ib_b_meas_mps2,
    )

    omega_ie_n = earth_rate_ned(phi)
    omega_en_n = transport_rate_ned(phi, h, v_n)
    omega_in_n = omega_ie_n + omega_en_n
    omega_total_n = 2.0 * omega_ie_n + omega_en_n
    f_n = deriv.corrected_f_n_mps2

    F = np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE), dtype=np.float64)

    # Position error dynamics:
    #   d(dot(phi)) / d(v_N)    = 1 / (M + h)
    #   d(dot(lambda))/d(v_E)   = 1 / ((N + h) cos(phi))
    #   d(dot(h)) / d(v_D)      = -1
    F[ERR_POS, ERR_VEL] = np.array(
        [
            [1.0 / (M + h), 0.0, 0.0],
            [0.0, 1.0 / ((N + h) * cos_phi), 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    # Small curvature couplings in the geodetic rates.
    F[0, 2] = -v_n[0] / ((M + h) ** 2)
    F[1, 0] = v_n[1] * np.tan(phi) / ((N + h) * cos_phi)
    F[1, 2] = -v_n[1] / (((N + h) ** 2) * cos_phi)

    # Velocity error dynamics:
    #   d(dv)/d(dv)   ≈ -[2 omega_ie^n + omega_en^n]_x
    #   d(dv)/d(dth)  ≈ -[f^n]_x
    #   d(dv)/d(dba)  ≈ -C_n_b
    F[ERR_VEL, ERR_VEL] = -skew(omega_total_n)
    F[ERR_VEL, ERR_ATT] = -skew(f_n)
    F[ERR_VEL, ERR_BA] = -C_n_b

    # Simple gravity-height coupling.
    F[ERR_VEL, ERR_POS] = gravity_gradient_ned_height_only(phi, h)

    # Attitude error dynamics:
    #   d(dtheta)/d(dtheta) ≈ -[omega_in^n]_x
    #   d(dtheta)/d(dbg)    ≈ -C_n_b
    F[ERR_ATT, ERR_ATT] = -skew(omega_in_n)
    F[ERR_ATT, ERR_BG] = -C_n_b

    # Bias states are random walks, so deterministic subblocks stay zero.
    return F


def continuous_noise_input_matrix(
    state: ErrorStateINSNominalState,
) -> FloatArray:
    r"""
    Build the continuous-time noise-input matrix `G`.

    Noise ordering
    --------------
        w = [n_g, n_a, n_bg, n_ba]

    Model
    -----
    - gyro white noise enters attitude dynamics
    - accel white noise enters velocity dynamics
    - gyro-bias random walk enters gyro-bias states
    - accel-bias random walk enters accel-bias states
    """
    C_n_b = state.C_n_b

    G = np.zeros((ERROR_STATE_SIZE, PROCESS_NOISE_SIZE), dtype=np.float64)
    G[ERR_ATT, 0:3] = -C_n_b
    G[ERR_VEL, 3:6] = C_n_b
    G[ERR_BG, 6:9] = np.eye(3, dtype=np.float64)
    G[ERR_BA, 9:12] = np.eye(3, dtype=np.float64)
    return G


def discretize_error_state_system(
    F: ArrayLike,
    G: ArrayLike,
    Qc: ArrayLike,
    dt_s: float,
) -> tuple[FloatArray, FloatArray]:
    r"""
    Discretize the continuous error-state model using a first-order approximation.

    Approximation
    -------------
        Phi ≈ I + F dt
        Qd  ≈ G Qc G^T dt

    Notes
    -----
    This is intentionally simple and transparent for the current project stage.
    A later version can replace it with a Van Loan / matrix exponential method
    once the estimator stack is more mature.
    """
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")

    Fm = _covariance(_symmetrize(np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE))), name="dummy", n=ERROR_STATE_SIZE)  # overwritten
    del Fm  # silence static analyzers in minimal environments

    F_arr = _as_float_array(F)
    G_arr = _as_float_array(G)
    Qc_arr = _as_float_array(Qc)

    if F_arr.shape != (ERROR_STATE_SIZE, ERROR_STATE_SIZE):
        raise ValueError(
            f"F must have shape ({ERROR_STATE_SIZE}, {ERROR_STATE_SIZE}), got {F_arr.shape}."
        )
    if G_arr.shape != (ERROR_STATE_SIZE, PROCESS_NOISE_SIZE):
        raise ValueError(
            f"G must have shape ({ERROR_STATE_SIZE}, {PROCESS_NOISE_SIZE}), got {G_arr.shape}."
        )
    if Qc_arr.shape != (PROCESS_NOISE_SIZE, PROCESS_NOISE_SIZE):
        raise ValueError(
            f"Qc must have shape ({PROCESS_NOISE_SIZE}, {PROCESS_NOISE_SIZE}), got {Qc_arr.shape}."
        )

    Phi = np.eye(ERROR_STATE_SIZE, dtype=np.float64) + F_arr * dt
    Qd = G_arr @ Qc_arr @ G_arr.T * dt
    Qd = _symmetrize(Qd)
    return Phi, Qd


def build_error_state_propagation_matrices(
    state: ErrorStateINSState,
    omega_ib_b_meas_radps: ArrayLike,
    f_ib_b_meas_mps2: ArrayLike,
    dt_s: float,
) -> ErrorStatePropagationMatrices:
    """
    Build continuous and discrete propagation matrices for one time step.
    """
    F = continuous_error_state_jacobian(
        state.nominal,
        omega_ib_b_meas_radps=omega_ib_b_meas_radps,
        f_ib_b_meas_mps2=f_ib_b_meas_mps2,
    )
    G = continuous_noise_input_matrix(state.nominal)
    Qc = state.process_noise.Qc
    Phi, Qd = discretize_error_state_system(F, G, Qc, dt_s)
    return ErrorStatePropagationMatrices(
        F=F,
        G=G,
        Qc=Qc,
        Phi=Phi,
        Qd=Qd,
    )


# -----------------------------------------------------------------------------
# Measurement-model helpers
# -----------------------------------------------------------------------------


def velocity_measurement_model_ned(state: ErrorStateINSState) -> FloatArray:
    """
    Predicted NED velocity measurement.

    Returns
    -------
    np.ndarray, shape (3,)
        Predicted velocity observation [m/s].
    """
    return state.nominal.v_ned_mps.copy()


def velocity_measurement_jacobian_ned() -> FloatArray:
    """
    Jacobian for a direct NED velocity observation.

    Measurement
    -----------
        z_v = v^n + noise

    Error model
    -----------
        residual ≈ H delta_x + noise
    """
    H = np.zeros((3, ERROR_STATE_SIZE), dtype=np.float64)
    H[:, ERR_VEL] = np.eye(3, dtype=np.float64)
    return H


def position_measurement_model_geodetic(state: ErrorStateINSState) -> FloatArray:
    """
    Predicted geodetic position measurement.

    Returns
    -------
    np.ndarray, shape (3,)
        `[lat, lon, h]`.
    """
    return np.array(
        [
            state.nominal.lat_rad,
            state.nominal.lon_rad,
            state.nominal.height_m,
        ],
        dtype=np.float64,
    )


def position_measurement_jacobian_geodetic() -> FloatArray:
    """
    Jacobian for a direct geodetic position observation.
    """
    H = np.zeros((3, ERROR_STATE_SIZE), dtype=np.float64)
    H[:, ERR_POS] = np.eye(3, dtype=np.float64)
    return H


def depth_measurement_model(
    state: ErrorStateINSState,
    *,
    reference_surface_height_m: float = 0.0,
) -> float:
    r"""
    Predicted signed depth measurement.

    Convention
    ----------
        depth = h_ref - h

    so:
    - positive depth means below the reference surface
    - negative depth means above the reference surface
    """
    return float(reference_surface_height_m - state.nominal.height_m)


def depth_measurement_jacobian() -> FloatArray:
    r"""
    Jacobian for a signed-depth observation:

        d = h_ref - h

    hence:

        partial d / partial [d_lat, d_lon, d_h] = [0, 0, -1]
    """
    H = np.zeros((1, ERROR_STATE_SIZE), dtype=np.float64)
    H[0, 2] = -1.0
    return H


def velocity_measurement_residual_ned(
    measured_velocity_ned_mps: ArrayLike,
    state: ErrorStateINSState,
) -> FloatArray:
    """
    Velocity residual for a direct NED velocity measurement.
    """
    z = _vec3(measured_velocity_ned_mps, name="measured_velocity_ned_mps")
    return z - velocity_measurement_model_ned(state)


def position_measurement_residual_geodetic(
    measured_lat_rad: float,
    measured_lon_rad: float,
    measured_height_m: float,
    state: ErrorStateINSState,
) -> FloatArray:
    """
    Position residual for a direct geodetic measurement.

    Longitude residual is wrapped to [-pi, pi).
    """
    pred = position_measurement_model_geodetic(state)
    return np.array(
        [
            float(measured_lat_rad) - pred[0],
            float(wrap_angle_pi(measured_lon_rad - pred[1])),
            float(measured_height_m) - pred[2],
        ],
        dtype=np.float64,
    )


def depth_measurement_residual(
    measured_depth_m: float,
    state: ErrorStateINSState,
    *,
    reference_surface_height_m: float = 0.0,
) -> FloatArray:
    """
    Signed-depth residual for a scalar depth aid.
    """
    pred = depth_measurement_model(
        state,
        reference_surface_height_m=reference_surface_height_m,
    )
    return np.array([float(measured_depth_m) - pred], dtype=np.float64)


# -----------------------------------------------------------------------------
# Filter class
# -----------------------------------------------------------------------------


class ErrorStateINS:
    """
    Stateful nominal + covariance local-level error-state INS.

    Typical workflow
    ----------------
    1) Initialize from a truth/sample prior or other initial condition.
    2) Call `predict(...)` for each IMU sample.
    3) Call `linear_update(...)` whenever an aiding measurement arrives.

    Notes
    -----
    This class performs closed-loop correction:
    - the nominal state is propagated nonlinearly
    - the covariance is propagated linearly
    - measurement updates compute a small `delta_x`
    - that correction is injected back into the nominal state
    """

    def __init__(self, state: ErrorStateINSState) -> None:
        self.state = state

    @classmethod
    def from_truth_trajectory_start(
        cls,
        truth: TruthTrajectory,
        process_noise: ErrorStateINSProcessNoise,
        *,
        P0: Optional[ArrayLike] = None,
        gyro_bias_radps: ArrayLike | float = 0.0,
        accel_bias_mps2: ArrayLike | float = 0.0,
    ) -> "ErrorStateINS":
        """
        Initialize the filter from the first truth-trajectory sample.
        """
        nominal = ErrorStateINSNominalState.from_truth_trajectory_start(
            truth,
            gyro_bias_radps=gyro_bias_radps,
            accel_bias_mps2=accel_bias_mps2,
        )
        if P0 is None:
            P0_arr = np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE), dtype=np.float64)
        else:
            P0_arr = _covariance(P0, name="P0", n=ERROR_STATE_SIZE)
        return cls(
            ErrorStateINSState(
                nominal=nominal,
                P=P0_arr,
                process_noise=process_noise,
            )
        )

    @property
    def nominal(self) -> ErrorStateINSNominalState:
        """Current nominal state."""
        return self.state.nominal

    @property
    def covariance(self) -> FloatArray:
        """Current error covariance."""
        return self.state.P

    def copy(self) -> "ErrorStateINS":
        """Deep copy of the filter."""
        return ErrorStateINS(self.state.copy())

    def predict(
        self,
        omega_ib_b_meas_radps: ArrayLike,
        f_ib_b_meas_mps2: ArrayLike,
        dt_s: float,
    ) -> ErrorStatePropagationMatrices:
        """
        Perform one IMU-driven prediction step.

        Parameters
        ----------
        omega_ib_b_meas_radps : array-like, shape (3,)
            Gyroscope measurement [rad/s].
        f_ib_b_meas_mps2 : array-like, shape (3,)
            Accelerometer measurement [m/s^2].
        dt_s : float
            Time step [s].

        Returns
        -------
        ErrorStatePropagationMatrices
            Propagation matrices used for the covariance prediction.
        """
        mats = build_error_state_propagation_matrices(
            self.state,
            omega_ib_b_meas_radps=omega_ib_b_meas_radps,
            f_ib_b_meas_mps2=f_ib_b_meas_mps2,
            dt_s=dt_s,
        )

        self.state.nominal = propagate_nominal_state(
            self.state.nominal,
            omega_ib_b_meas_radps=omega_ib_b_meas_radps,
            f_ib_b_meas_mps2=f_ib_b_meas_mps2,
            dt_s=dt_s,
        )
        self.state.P = _symmetrize(
            mats.Phi @ self.state.P @ mats.Phi.T + mats.Qd
        )
        return mats

    def inject_error_state(self, delta_x: ArrayLike) -> None:
        r"""
        Inject a small closed-loop correction into the nominal state.

        State interpretation
        --------------------
            delta_x =
            [d_pos, d_vel, d_theta, d_bg, d_ba]

        The attitude correction uses a navigation-frame left-multiplicative update:

            C_n_b <- Exp(-[d_theta_n]_x) C_n_b
        """
        dx = _as_float_array(delta_x).reshape(-1)
        if dx.shape != (ERROR_STATE_SIZE,):
            raise ValueError(
                f"delta_x must have shape ({ERROR_STATE_SIZE},), got {dx.shape}."
            )

        d_pos = dx[ERR_POS]
        d_vel = dx[ERR_VEL]
        d_theta = dx[ERR_ATT]
        d_bg = dx[ERR_BG]
        d_ba = dx[ERR_BA]

        self.state.nominal.lat_rad = _check_latitude(
            self.state.nominal.lat_rad + float(d_pos[0]),
            name="corrected_lat_rad",
        )
        self.state.nominal.lon_rad = float(
            wrap_angle_pi(self.state.nominal.lon_rad + float(d_pos[1]))
        )
        self.state.nominal.height_m = float(
            self.state.nominal.height_m + float(d_pos[2])
        )
        self.state.nominal.v_ned_mps = (
            self.state.nominal.v_ned_mps + d_vel
        ).astype(np.float64)
        self.state.nominal.C_n_b = project_to_so3(
            dcm_from_rotvec(-d_theta) @ self.state.nominal.C_n_b
        )
        self.state.nominal.gyro_bias_radps = (
            self.state.nominal.gyro_bias_radps + d_bg
        ).astype(np.float64)
        self.state.nominal.accel_bias_mps2 = (
            self.state.nominal.accel_bias_mps2 + d_ba
        ).astype(np.float64)

    def linear_update(
        self,
        residual: ArrayLike,
        H: ArrayLike,
        R: ArrayLike,
    ) -> LinearizedMeasurementUpdate:
        """
        Perform a generic linearized Kalman measurement update.

        Parameters
        ----------
        residual : array-like, shape (m,)
            Residual:
                r = z - h(x_nom)
        H : array-like, shape (m, 15)
            Linearized measurement Jacobian.
        R : array-like, shape (m, m)
            Measurement covariance.

        Returns
        -------
        LinearizedMeasurementUpdate
            Update details including the applied correction `delta_x`.
        """
        r = _as_float_array(residual).reshape(-1)
        Hm = _as_float_array(H)
        Rm = _as_float_array(R)

        m = r.shape[0]
        if Hm.shape != (m, ERROR_STATE_SIZE):
            raise ValueError(
                f"H must have shape ({m}, {ERROR_STATE_SIZE}), got {Hm.shape}."
            )
        if Rm.shape != (m, m):
            raise ValueError(f"R must have shape ({m}, {m}), got {Rm.shape}.")

        P = self.state.P
        S = _symmetrize(Hm @ P @ Hm.T + Rm)
        K = P @ Hm.T @ np.linalg.inv(S)
        delta_x = K @ r

        I = np.eye(ERROR_STATE_SIZE, dtype=np.float64)
        P_new = (I - K @ Hm) @ P @ (I - K @ Hm).T + K @ Rm @ K.T
        self.state.P = _symmetrize(P_new)

        self.inject_error_state(delta_x)

        return LinearizedMeasurementUpdate(
            residual=r.astype(np.float64),
            innovation_covariance=S.astype(np.float64),
            kalman_gain=K.astype(np.float64),
            delta_x=delta_x.astype(np.float64),
        )

    def update_with_velocity_ned(
        self,
        measured_velocity_ned_mps: ArrayLike,
        R_mps2: ArrayLike,
    ) -> LinearizedMeasurementUpdate:
        """
        Convenience wrapper: direct NED velocity update.
        """
        residual = velocity_measurement_residual_ned(
            measured_velocity_ned_mps,
            self.state,
        )
        H = velocity_measurement_jacobian_ned()
        R = _as_float_array(R_mps2)
        if R.shape != (3, 3):
            raise ValueError(f"R_mps2 must have shape (3, 3), got {R.shape}.")
        return self.linear_update(residual, H, R)

    def update_with_position_geodetic(
        self,
        measured_lat_rad: float,
        measured_lon_rad: float,
        measured_height_m: float,
        R: ArrayLike,
    ) -> LinearizedMeasurementUpdate:
        """
        Convenience wrapper: direct geodetic position update.
        """
        residual = position_measurement_residual_geodetic(
            measured_lat_rad=measured_lat_rad,
            measured_lon_rad=measured_lon_rad,
            measured_height_m=measured_height_m,
            state=self.state,
        )
        H = position_measurement_jacobian_geodetic()
        Rm = _as_float_array(R)
        if Rm.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {Rm.shape}.")
        return self.linear_update(residual, H, Rm)

    def update_with_depth(
        self,
        measured_depth_m: float,
        depth_variance_m2: float,
        *,
        reference_surface_height_m: float = 0.0,
    ) -> LinearizedMeasurementUpdate:
        """
        Convenience wrapper: scalar signed-depth update.
        """
        residual = depth_measurement_residual(
            measured_depth_m=measured_depth_m,
            state=self.state,
            reference_surface_height_m=reference_surface_height_m,
        )
        H = depth_measurement_jacobian()
        R = np.array([[float(depth_variance_m2)]], dtype=np.float64)
        if R[0, 0] < 0.0:
            raise ValueError("depth_variance_m2 must be nonnegative.")
        return self.linear_update(residual, H, R)


__all__ = [
    "ERR_ATT",
    "ERR_BA",
    "ERR_BG",
    "ERR_POS",
    "ERR_VEL",
    "ERROR_STATE_SIZE",
    "PROCESS_NOISE_SIZE",
    "ErrorStateINS",
    "ErrorStateINSNominalState",
    "ErrorStateINSProcessNoise",
    "ErrorStateINSState",
    "ErrorStatePropagationMatrices",
    "LinearizedMeasurementUpdate",
    "NominalStateDerivative",
    "build_error_state_propagation_matrices",
    "continuous_error_state_jacobian",
    "continuous_noise_input_matrix",
    "corrected_imu_inputs",
    "depth_measurement_jacobian",
    "depth_measurement_model",
    "depth_measurement_residual",
    "discretize_error_state_system",
    "gravity_gradient_ned_height_only",
    "nominal_state_derivative",
    "position_measurement_jacobian_geodetic",
    "position_measurement_model_geodetic",
    "position_measurement_residual_geodetic",
    "propagate_nominal_state",
    "velocity_measurement_jacobian_ned",
    "velocity_measurement_model_ned",
    "velocity_measurement_residual_ned",
]