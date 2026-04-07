"""
trajectory.py

Truth-trajectory containers and builders for the gravity-aided navigation
simulator.

This module sits between the low-level physics helpers (`earth.py`, `frames.py`,
`kinematics.py`) and the higher-level scenario / vehicle-model layer.

Why this file exists
--------------------
The repository's sensor layer already expects, sample by sample:
- geodetic position `(lat, lon, h)`
- NED velocity `v^n`
- NED velocity derivative `vdot^n`
- body attitude `C_n_b`
- body angular rate relative to the local navigation frame `omega_nb^b`

That interface is exactly what `imu.py` and `gravimeter.py` consume when they
produce ideal measurements. Therefore, the job of this module is to be the
single source of truth for constructing and validating those histories.

Conventions
-----------
- Angles are in radians.
- Height is ellipsoidal height above the WGS 84 ellipsoid [m].
- NED means [North, East, Down].
- `C_n_b` denotes the passive body->NED DCM such that:

      v^n = C_n_b v^b

- Quaternions are scalar-first Hamilton quaternions and represent the same
  passive transform convention as the DCMs in `gravnav.physics.frames`.

Primary references used here
----------------------------
1) INSTINCT / University of Stuttgart,
   "INS/GNSS Loosely-coupled Kalman Filter (local-navigation frame)"
   URL:
   https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html

   Used for the standard curvilinear position-rate equations in a local
   navigation frame:

       dot(phi)    = v_N / (R_N + h)
       dot(lambda) = v_E / ((R_E + h) cos(phi))
       dot(h)      = -v_D

   where:
   - phi is geodetic latitude
   - lambda is longitude
   - h is ellipsoidal height
   - R_N is the meridian radius of curvature
   - R_E is the prime vertical radius of curvature

2) AHRS documentation, "Attitude from angular rate"
   URL:
   https://ahrs.readthedocs.io/en/latest/filters/angular.html

   Used for the interpretation that attitude histories are propagated by
   accumulating angular rate over time and for documenting the meaning of a
   samplewise body-rate history.

3) NumPy documentation for `numpy.gradient`
   URL:
   https://numpy.org/doc/stable/reference/generated/numpy.gradient.html

   Used for sampled first derivatives with second-order central differences in
   the interior and one-sided boundary handling.

Design choices
--------------
- This module deliberately avoids scenario-specific motion logic. It only knows
  how to represent a truth trajectory and derive missing pieces from sampled
  histories.
- Vehicle-specific motion primitives belong in `truth.vehicle_models`.
- Scenario assembly belongs in `truth.scenarios`.
- The builders here are intentionally transparent and numerically conservative,
  because this layer becomes the reference truth for the rest of the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.earth import meridian_radius, prime_vertical_radius
from ..physics.frames import dcm_to_quaternion, project_to_so3, wrap_angle_pi
from ..physics.kinematics import (
    body_rate_from_consecutive_dcms,
    course_from_velocity_ned,
    first_derivative,
    flight_path_angle_from_velocity_ned,
    horizontal_speed_from_velocity_ned,
    speed_from_velocity_ned,
)

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _time_vector(times_s: ArrayLike) -> FloatArray:
    """
    Validate and return a strictly increasing time vector.
    """
    t = _as_float_array(times_s).reshape(-1)
    if t.ndim != 1:
        raise ValueError("times_s must be one-dimensional.")
    if t.size < 2:
        raise ValueError("times_s must contain at least two samples.")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    return t


def _history_1d(x: ArrayLike, n: int, *, name: str) -> FloatArray:
    """
    Validate and return a length-N scalar history.
    """
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}.")
    return arr


def _history_vec3(x: ArrayLike, n: int, *, name: str) -> FloatArray:
    """
    Validate and return a shape-(N,3) vector history.
    """
    arr = _as_float_array(x)
    if arr.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3), got {arr.shape}.")
    return arr


def _history_dcm(x: ArrayLike, n: int, *, name: str) -> FloatArray:
    """
    Validate and return a shape-(N,3,3) DCM history.
    """
    arr = _as_float_array(x)
    if arr.shape != (n, 3, 3):
        raise ValueError(f"{name} must have shape ({n}, 3, 3), got {arr.shape}.")
    return arr


def _wrap_longitude_rad(lon_rad: ArrayLike):
    """
    Wrap longitude(s) to [-pi, pi).
    """
    return wrap_angle_pi(lon_rad)


def _check_latitude_array(lat_rad: FloatArray, *, name: str = "lat_rad") -> None:
    """
    Require all latitudes to lie within [-pi/2, pi/2].
    """
    if np.any(np.abs(lat_rad) > 0.5 * np.pi + 1e-12):
        raise ValueError(f"{name} contains values outside [-pi/2, pi/2].")


def _project_dcm_history(C_hist: FloatArray) -> FloatArray:
    """
    Project every matrix in a DCM history onto SO(3).
    """
    out = np.empty_like(C_hist, dtype=np.float64)
    for k in range(C_hist.shape[0]):
        out[k] = project_to_so3(C_hist[k])
    return out


def geodetic_rates_from_velocity_ned(
    lat_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> FloatArray:
    r"""
    Convert NED velocity into curvilinear geodetic rates.

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
        `[dot(lat), dot(lon), dot(h)]` in `[rad/s, rad/s, m/s]`.

    Formula
    -------
    The standard local-navigation-frame curvilinear position equations are:

        dot(phi)    = v_N / (R_N + h)
        dot(lambda) = v_E / ((R_E + h) cos(phi))
        dot(h)      = -v_D

    where:
    - `R_N` is the meridian radius of curvature
    - `R_E` is the prime vertical radius of curvature

    Reference
    ---------
    INSTINCT / Groves/Titterton-style local-navigation equations:
    https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html
    """
    v_n = _as_float_array(v_ned_mps).reshape(-1)
    if v_n.shape != (3,):
        raise ValueError(f"v_ned_mps must have shape (3,), got {v_n.shape}.")

    phi = float(lat_rad)
    h = float(height_m)
    if abs(phi) > 0.5 * np.pi + 1e-12:
        raise ValueError("lat_rad must lie within [-pi/2, pi/2].")

    cos_phi = float(np.cos(phi))
    if abs(cos_phi) < 1e-12:
        raise ValueError(
            "Longitude rate is singular at the poles; cannot evaluate near |lat| = pi/2."
        )

    R_N = float(meridian_radius(phi))
    R_E = float(prime_vertical_radius(phi))

    return np.array(
        [
            v_n[0] / (R_N + h),
            v_n[1] / ((R_E + h) * cos_phi),
            -v_n[2],
        ],
        dtype=np.float64,
    )


def velocity_ned_from_geodetic_rates(
    lat_rad: float,
    height_m: float,
    lat_dot_radps: float,
    lon_dot_radps: float,
    height_dot_mps: float,
) -> FloatArray:
    r"""
    Convert curvilinear geodetic rates into NED velocity.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude [rad].
    height_m : float
        Ellipsoidal height [m].
    lat_dot_radps : float
        Latitude rate [rad/s].
    lon_dot_radps : float
        Longitude rate [rad/s].
    height_dot_mps : float
        Height rate [m/s]. Positive means climbing.

    Returns
    -------
    np.ndarray, shape (3,)
        NED velocity `[v_N, v_E, v_D]` [m/s].

    Formula
    -------
    Inverting the standard curvilinear position equations gives:

        v_N = (R_N + h) dot(phi)
        v_E = (R_E + h) cos(phi) dot(lambda)
        v_D = -dot(h)

    Reference
    ---------
    Same local-navigation-frame position equations as:
    https://unistuttgart-ins.github.io/INSTINCT/LooselyCoupledKF_n.html
    """
    phi = float(lat_rad)
    h = float(height_m)
    if abs(phi) > 0.5 * np.pi + 1e-12:
        raise ValueError("lat_rad must lie within [-pi/2, pi/2].")

    cos_phi = float(np.cos(phi))
    if abs(cos_phi) < 1e-12:
        raise ValueError(
            "Longitude rate conversion is singular at the poles; cannot evaluate near |lat| = pi/2."
        )

    R_N = float(meridian_radius(phi))
    R_E = float(prime_vertical_radius(phi))

    return np.array(
        [
            (R_N + h) * float(lat_dot_radps),
            (R_E + h) * cos_phi * float(lon_dot_radps),
            -float(height_dot_mps),
        ],
        dtype=np.float64,
    )


def integrate_geodetic_from_velocity_ned(
    times_s: ArrayLike,
    v_ned_mps: ArrayLike,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Integrate a sampled NED velocity history into geodetic position history.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    v_ned_mps : array-like, shape (N, 3)
        NED velocity history [m/s].
    lat0_rad : float
        Initial geodetic latitude [rad].
    lon0_rad : float
        Initial longitude [rad].
    height0_m : float
        Initial ellipsoidal height [m].

    Returns
    -------
    tuple of np.ndarray
        `(lat_rad, lon_rad, height_m)` each of shape `(N,)`.

    Method
    ------
    This uses a simple Heun / predictor-corrector step on the curvilinear-rate
    equations:

        xdot = f(x, v)
        x_{k+1} ≈ x_k + 0.5 * dt * ( f(x_k, v_k) + f(x_pred, v_{k+1}) )

    where `x = [lat, lon, h]^T`.

    Notes
    -----
    This is intentionally simple, deterministic, and adequate for the modest
    trajectory lengths / step sizes typical of the simulator. If later you need
    extreme-duration propagation near the poles, you can replace this with a
    higher-order or ECEF-based propagator.
    """
    t = _time_vector(times_s)
    v_hist = _history_vec3(v_ned_mps, t.size, name="v_ned_mps")

    lat = np.empty(t.size, dtype=np.float64)
    lon = np.empty(t.size, dtype=np.float64)
    h = np.empty(t.size, dtype=np.float64)

    lat[0] = float(lat0_rad)
    lon[0] = float(_wrap_longitude_rad(lon0_rad))
    h[0] = float(height0_m)
    _check_latitude_array(lat[:1])

    for k in range(t.size - 1):
        dt = float(t[k + 1] - t[k])

        rate0 = geodetic_rates_from_velocity_ned(
            lat_rad=float(lat[k]),
            height_m=float(h[k]),
            v_ned_mps=v_hist[k],
        )

        lat_pred = float(lat[k] + dt * rate0[0])
        lon_pred = float(_wrap_longitude_rad(lon[k] + dt * rate0[1]))
        h_pred = float(h[k] + dt * rate0[2])

        rate1 = geodetic_rates_from_velocity_ned(
            lat_rad=lat_pred,
            height_m=h_pred,
            v_ned_mps=v_hist[k + 1],
        )

        lat[k + 1] = lat[k] + 0.5 * dt * (rate0[0] + rate1[0])
        lon[k + 1] = _wrap_longitude_rad(lon[k] + 0.5 * dt * (rate0[1] + rate1[1]))
        h[k + 1] = h[k] + 0.5 * dt * (rate0[2] + rate1[2])

    _check_latitude_array(lat)
    return lat, lon, h


def dcm_body_to_ned_from_ypr(
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
) -> FloatArray:
    r"""
    Build the passive body->NED DCM from yaw, pitch, roll using the standard
    aerospace 3-2-1 sequence.

    Parameters
    ----------
    yaw_rad : float
        Yaw / heading angle [rad]. Positive clockwise from North toward East in
        the NED convention.
    pitch_rad : float
        Pitch angle [rad]. Positive nose-up.
    roll_rad : float
        Roll angle [rad]. Positive right wing down.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Passive DCM `C_n_b` mapping body-resolved vectors into NED.

    Formula
    -------
    The standard body-to-navigation 3-2-1 transformation is:

        C_n_b = R_z(psi) R_y(theta) R_x(phi)

    which expands to:

        [ cθ cψ,  sφ sθ cψ - cφ sψ,  cφ sθ cψ + sφ sψ ]
        [ cθ sψ,  sφ sθ sψ + cφ cψ,  cφ sθ sψ - sφ cψ ]
        [ -sθ,              sφ cθ,              cφ cθ ]

    for yaw `psi`, pitch `theta`, roll `phi`.

    Notes
    -----
    This function is included here because later trajectory / vehicle-model code
    often starts from yaw-pitch-roll schedules even though the rest of the stack
    uses DCMs and quaternions.
    """
    psi = float(yaw_rad)
    theta = float(pitch_rad)
    phi = float(roll_rad)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    return np.array(
        [
            [ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
            [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
            [-stheta, sphi * ctheta, cphi * ctheta],
        ],
        dtype=np.float64,
    )


def dcm_history_body_to_ned_from_ypr(
    yaw_rad: ArrayLike,
    pitch_rad: ArrayLike,
    roll_rad: ArrayLike,
) -> FloatArray:
    """
    Vectorized history version of `dcm_body_to_ned_from_ypr(...)`.

    Parameters
    ----------
    yaw_rad, pitch_rad, roll_rad : array-like, shape (N,)
        Euler-angle histories [rad].

    Returns
    -------
    np.ndarray, shape (N, 3, 3)
        Body->NED DCM history.
    """
    yaw = _as_float_array(yaw_rad).reshape(-1)
    pitch = _as_float_array(pitch_rad).reshape(-1)
    roll = _as_float_array(roll_rad).reshape(-1)

    if not (yaw.shape == pitch.shape == roll.shape):
        raise ValueError(
            f"yaw_rad, pitch_rad, and roll_rad must have the same shape; got {yaw.shape}, {pitch.shape}, {roll.shape}."
        )

    out = np.empty((yaw.size, 3, 3), dtype=np.float64)
    for k in range(yaw.size):
        out[k] = dcm_body_to_ned_from_ypr(yaw[k], pitch[k], roll[k])
    return out


def ypr_from_dcm_body_to_ned(C_n_b: ArrayLike) -> FloatArray:
    r"""
    Extract yaw, pitch, roll from a passive body->NED DCM using the standard
    3-2-1 aerospace convention.

    Parameters
    ----------
    C_n_b : array-like, shape (3, 3)
        Passive body->NED DCM.

    Returns
    -------
    np.ndarray, shape (3,)
        `[yaw, pitch, roll]` in radians.

    Formula
    -------
    For the standard 3-2-1 body->navigation matrix,

        pitch = asin(-C[2,0])
        roll  = atan2(C[2,1], C[2,2])
        yaw   = atan2(C[1,0], C[0,0])

    Notes
    -----
    This uses the conventional principal-value extraction and therefore inherits
    the usual pitch singularity near ±pi/2.
    """
    C = project_to_so3(_as_float_array(C_n_b).reshape(3, 3))
    pitch = np.arcsin(np.clip(-C[2, 0], -1.0, 1.0))
    roll = np.arctan2(C[2, 1], C[2, 2])
    yaw = np.arctan2(C[1, 0], C[0, 0])
    return np.array([wrap_angle_pi(yaw), pitch, wrap_angle_pi(roll)], dtype=np.float64)


def body_rate_history_from_dcm_history(
    times_s: ArrayLike,
    C_n_b: ArrayLike,
) -> FloatArray:
    r"""
    Estimate a samplewise body-rate history from a sampled DCM history.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    C_n_b : array-like, shape (N, 3, 3)
        Body->NED DCM history.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Approximate samplewise `omega_nb^b` history [rad/s].

    Method
    ------
    Interval rates are first computed from consecutive attitudes via the local
    rotation-vector relation:

        delta_C_b ≈ C_k^T C_{k+1}
        rotvec(delta_C_b) ≈ omega_nb^b * dt

    The resulting interval-centered rates are then promoted to sample-centered
    values by endpoint repetition and interior averaging:

        omega[0]      = omega_interval[0]
        omega[k]      = 0.5 * (omega_interval[k-1] + omega_interval[k])
        omega[N - 1]  = omega_interval[N - 2]

    Notes
    -----
    This is a pragmatic truth-generation helper, not a smoothing spline or a
    continuous-time attitude fit.
    """
    t = _time_vector(times_s)
    C_hist = _project_dcm_history(_history_dcm(C_n_b, t.size, name="C_n_b"))

    interval_rates = np.empty((t.size - 1, 3), dtype=np.float64)
    for k in range(t.size - 1):
        interval_rates[k] = body_rate_from_consecutive_dcms(
            C_n_b_prev=C_hist[k],
            C_n_b_next=C_hist[k + 1],
            dt_s=float(t[k + 1] - t[k]),
        )

    out = np.empty((t.size, 3), dtype=np.float64)
    out[0] = interval_rates[0]
    out[-1] = interval_rates[-1]
    if t.size > 2:
        out[1:-1] = 0.5 * (interval_rates[:-1] + interval_rates[1:])
    return out


@dataclass
class TrajectorySample:
    """
    One truth-trajectory sample.

    Attributes
    ----------
    time_s : float
        Sample timestamp [s].
    lat_rad : float
        Geodetic latitude [rad].
    lon_rad : float
        Longitude [rad].
    height_m : float
        Ellipsoidal height [m].
    v_ned_mps : np.ndarray, shape (3,)
        NED velocity [m/s].
    v_dot_ned_mps2 : np.ndarray, shape (3,)
        NED velocity derivative [m/s^2].
    C_n_b : np.ndarray, shape (3, 3)
        Passive body->NED DCM.
    omega_nb_b_radps : np.ndarray, shape (3,)
        Body rate with respect to NED, resolved in body [rad/s].
    """

    time_s: float
    lat_rad: float
    lon_rad: float
    height_m: float
    v_ned_mps: FloatArray
    v_dot_ned_mps2: FloatArray
    C_n_b: FloatArray
    omega_nb_b_radps: FloatArray

    @property
    def q_n_b(self) -> FloatArray:
        """Scalar-first quaternion representing the same body->NED attitude."""
        return dcm_to_quaternion(self.C_n_b)

    @property
    def speed_mps(self) -> float:
        """Total speed magnitude [m/s]."""
        return float(speed_from_velocity_ned(self.v_ned_mps))

    @property
    def ground_speed_mps(self) -> float:
        """Horizontal speed magnitude [m/s]."""
        return float(horizontal_speed_from_velocity_ned(self.v_ned_mps))


@dataclass
class TruthTrajectory:
    """
    Time history of truth states required by the simulator's sensor and estimator
    stack.

    Attributes
    ----------
    time_s : np.ndarray, shape (N,)
        Sample times [s].
    lat_rad : np.ndarray, shape (N,)
        Geodetic latitude history [rad].
    lon_rad : np.ndarray, shape (N,)
        Longitude history [rad].
    height_m : np.ndarray, shape (N,)
        Ellipsoidal height history [m].
    v_ned_mps : np.ndarray, shape (N, 3)
        NED velocity history [m/s].
    v_dot_ned_mps2 : np.ndarray, shape (N, 3)
        NED velocity-derivative history [m/s^2].
    C_n_b : np.ndarray, shape (N, 3, 3)
        Passive body->NED DCM history.
    omega_nb_b_radps : np.ndarray, shape (N, 3)
        Body-rate history with respect to the navigation frame, resolved in body.
    """

    time_s: FloatArray
    lat_rad: FloatArray
    lon_rad: FloatArray
    height_m: FloatArray
    v_ned_mps: FloatArray
    v_dot_ned_mps2: FloatArray
    C_n_b: FloatArray
    omega_nb_b_radps: FloatArray

    def __post_init__(self) -> None:
        n = len(np.asarray(self.time_s).reshape(-1))
        self.time_s = _time_vector(self.time_s)
        self.lat_rad = _history_1d(self.lat_rad, n, name="lat_rad")
        self.lon_rad = _history_1d(self.lon_rad, n, name="lon_rad")
        self.height_m = _history_1d(self.height_m, n, name="height_m")
        self.v_ned_mps = _history_vec3(self.v_ned_mps, n, name="v_ned_mps")
        self.v_dot_ned_mps2 = _history_vec3(
            self.v_dot_ned_mps2,
            n,
            name="v_dot_ned_mps2",
        )
        self.C_n_b = _project_dcm_history(_history_dcm(self.C_n_b, n, name="C_n_b"))
        self.omega_nb_b_radps = _history_vec3(
            self.omega_nb_b_radps,
            n,
            name="omega_nb_b_radps",
        )

        _check_latitude_array(self.lat_rad)
        self.lon_rad = _wrap_longitude_rad(self.lon_rad)

    def __len__(self) -> int:
        """Number of trajectory samples."""
        return int(self.time_s.shape[0])

    def copy(self) -> "TruthTrajectory":
        """Deep copy of the trajectory."""
        return TruthTrajectory(
            time_s=self.time_s.copy(),
            lat_rad=self.lat_rad.copy(),
            lon_rad=self.lon_rad.copy(),
            height_m=self.height_m.copy(),
            v_ned_mps=self.v_ned_mps.copy(),
            v_dot_ned_mps2=self.v_dot_ned_mps2.copy(),
            C_n_b=self.C_n_b.copy(),
            omega_nb_b_radps=self.omega_nb_b_radps.copy(),
        )

    def sample(self, index: int) -> TrajectorySample:
        """
        Return one trajectory sample.
        """
        return TrajectorySample(
            time_s=float(self.time_s[index]),
            lat_rad=float(self.lat_rad[index]),
            lon_rad=float(self.lon_rad[index]),
            height_m=float(self.height_m[index]),
            v_ned_mps=self.v_ned_mps[index].copy(),
            v_dot_ned_mps2=self.v_dot_ned_mps2[index].copy(),
            C_n_b=self.C_n_b[index].copy(),
            omega_nb_b_radps=self.omega_nb_b_radps[index].copy(),
        )

    @property
    def q_n_b(self) -> FloatArray:
        """
        Quaternion history corresponding to `C_n_b`.
        """
        out = np.empty((len(self), 4), dtype=np.float64)
        for k in range(len(self)):
            out[k] = dcm_to_quaternion(self.C_n_b[k])
        return out

    @property
    def speed_mps(self) -> FloatArray:
        """Total speed history [m/s]."""
        return np.asarray(speed_from_velocity_ned(self.v_ned_mps), dtype=np.float64)

    @property
    def ground_speed_mps(self) -> FloatArray:
        """Horizontal-speed history [m/s]."""
        return np.asarray(
            horizontal_speed_from_velocity_ned(self.v_ned_mps),
            dtype=np.float64,
        )

    @property
    def course_rad(self) -> FloatArray:
        """Ground-track course history [rad]."""
        return np.asarray(course_from_velocity_ned(self.v_ned_mps), dtype=np.float64)

    @property
    def flight_path_angle_rad(self) -> FloatArray:
        """Flight-path angle history [rad]. Positive means climbing."""
        return np.asarray(
            flight_path_angle_from_velocity_ned(self.v_ned_mps),
            dtype=np.float64,
        )

    @property
    def yaw_pitch_roll_rad(self) -> FloatArray:
        """
        Yaw-pitch-roll history extracted from `C_n_b` using the 3-2-1 convention.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Columns are `[yaw, pitch, roll]` in radians.
        """
        out = np.empty((len(self), 3), dtype=np.float64)
        for k in range(len(self)):
            out[k] = ypr_from_dcm_body_to_ned(self.C_n_b[k])
        return out


def build_truth_trajectory(
    times_s: ArrayLike,
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
    v_ned_mps: ArrayLike,
    C_n_b: ArrayLike,
    *,
    v_dot_ned_mps2: ArrayLike | None = None,
    omega_nb_b_radps: ArrayLike | None = None,
) -> TruthTrajectory:
    """
    Build a `TruthTrajectory` from fully sampled position, velocity, and attitude
    histories, deriving missing velocity derivatives and/or body rates when they
    are not provided.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    lat_rad, lon_rad, height_m : array-like, shape (N,)
        Geodetic position history.
    v_ned_mps : array-like, shape (N, 3)
        NED velocity history [m/s].
    C_n_b : array-like, shape (N, 3, 3)
        Body->NED attitude history.
    v_dot_ned_mps2 : array-like, shape (N, 3), optional
        NED velocity derivative history [m/s^2]. If omitted, it is estimated via
        `numpy.gradient` through `physics.kinematics.first_derivative(...)`.
    omega_nb_b_radps : array-like, shape (N, 3), optional
        Body-rate history [rad/s]. If omitted, it is estimated from successive
        DCMs.

    Returns
    -------
    TruthTrajectory
        Validated trajectory container.
    """
    t = _time_vector(times_s)
    n = t.size
    lat = _history_1d(lat_rad, n, name="lat_rad")
    lon = _history_1d(lon_rad, n, name="lon_rad")
    h = _history_1d(height_m, n, name="height_m")
    v_hist = _history_vec3(v_ned_mps, n, name="v_ned_mps")
    C_hist = _project_dcm_history(_history_dcm(C_n_b, n, name="C_n_b"))

    if v_dot_ned_mps2 is None:
        v_dot = first_derivative(t, v_hist)
    else:
        v_dot = _history_vec3(v_dot_ned_mps2, n, name="v_dot_ned_mps2")

    if omega_nb_b_radps is None:
        omega_hist = body_rate_history_from_dcm_history(t, C_hist)
    else:
        omega_hist = _history_vec3(
            omega_nb_b_radps,
            n,
            name="omega_nb_b_radps",
        )

    return TruthTrajectory(
        time_s=t,
        lat_rad=lat,
        lon_rad=lon,
        height_m=h,
        v_ned_mps=v_hist,
        v_dot_ned_mps2=v_dot,
        C_n_b=C_hist,
        omega_nb_b_radps=omega_hist,
    )


def build_truth_trajectory_from_velocity_attitude_history(
    times_s: ArrayLike,
    v_ned_mps: ArrayLike,
    C_n_b: ArrayLike,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
    v_dot_ned_mps2: ArrayLike | None = None,
    omega_nb_b_radps: ArrayLike | None = None,
) -> TruthTrajectory:
    """
    Build a trajectory from sampled NED velocity and attitude histories by
    integrating geodetic position from the supplied initial condition.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    v_ned_mps : array-like, shape (N, 3)
        NED velocity history [m/s].
    C_n_b : array-like, shape (N, 3, 3)
        Body->NED DCM history.
    lat0_rad, lon0_rad, height0_m : float
        Initial geodetic position.
    v_dot_ned_mps2 : array-like, optional
        Optional precomputed NED velocity derivative history [m/s^2].
    omega_nb_b_radps : array-like, optional
        Optional precomputed body-rate history [rad/s].

    Returns
    -------
    TruthTrajectory
        Truth trajectory.
    """
    t = _time_vector(times_s)
    v_hist = _history_vec3(v_ned_mps, t.size, name="v_ned_mps")

    lat, lon, h = integrate_geodetic_from_velocity_ned(
        t,
        v_hist,
        lat0_rad=float(lat0_rad),
        lon0_rad=float(lon0_rad),
        height0_m=float(height0_m),
    )

    return build_truth_trajectory(
        times_s=t,
        lat_rad=lat,
        lon_rad=lon,
        height_m=h,
        v_ned_mps=v_hist,
        C_n_b=C_n_b,
        v_dot_ned_mps2=v_dot_ned_mps2,
        omega_nb_b_radps=omega_nb_b_radps,
    )


def build_truth_trajectory_from_position_attitude_history(
    times_s: ArrayLike,
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
    C_n_b: ArrayLike,
    *,
    omega_nb_b_radps: ArrayLike | None = None,
) -> TruthTrajectory:
    r"""
    Build a trajectory from sampled geodetic position and attitude histories by
    differentiating position into NED velocity and acceleration.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    lat_rad, lon_rad, height_m : array-like, shape (N,)
        Geodetic position history.
    C_n_b : array-like, shape (N, 3, 3)
        Body->NED attitude history.
    omega_nb_b_radps : array-like, shape (N, 3), optional
        Optional precomputed body-rate history.

    Returns
    -------
    TruthTrajectory
        Truth trajectory.

    Method
    ------
    The sampled geodetic rates are estimated by finite differencing and converted
    samplewise into NED velocity using:

        v_N = (R_N + h) dot(phi)
        v_E = (R_E + h) cos(phi) dot(lambda)
        v_D = -dot(h)

    followed by another derivative to obtain `vdot^n`.
    """
    t = _time_vector(times_s)
    n = t.size
    lat = _history_1d(lat_rad, n, name="lat_rad")
    lon = _history_1d(lon_rad, n, name="lon_rad")
    h = _history_1d(height_m, n, name="height_m")

    lat_dot = first_derivative(t, lat)
    lon_dot = first_derivative(t, lon)
    h_dot = first_derivative(t, h)

    v_hist = np.empty((n, 3), dtype=np.float64)
    for k in range(n):
        v_hist[k] = velocity_ned_from_geodetic_rates(
            lat_rad=float(lat[k]),
            height_m=float(h[k]),
            lat_dot_radps=float(lat_dot[k]),
            lon_dot_radps=float(lon_dot[k]),
            height_dot_mps=float(h_dot[k]),
        )

    v_dot = first_derivative(t, v_hist)

    return build_truth_trajectory(
        times_s=t,
        lat_rad=lat,
        lon_rad=lon,
        height_m=h,
        v_ned_mps=v_hist,
        C_n_b=C_n_b,
        v_dot_ned_mps2=v_dot,
        omega_nb_b_radps=omega_nb_b_radps,
    )


def build_truth_trajectory_from_position_ypr_history(
    times_s: ArrayLike,
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    height_m: ArrayLike,
    yaw_rad: ArrayLike,
    pitch_rad: ArrayLike,
    roll_rad: ArrayLike,
    *,
    omega_nb_b_radps: ArrayLike | None = None,
) -> TruthTrajectory:
    """
    Convenience wrapper that builds attitude from yaw-pitch-roll histories and
    then constructs a truth trajectory from sampled geodetic position.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    lat_rad, lon_rad, height_m : array-like, shape (N,)
        Geodetic position history.
    yaw_rad, pitch_rad, roll_rad : array-like, shape (N,)
        Yaw-pitch-roll histories [rad].
    omega_nb_b_radps : array-like, shape (N, 3), optional
        Optional precomputed body-rate history.

    Returns
    -------
    TruthTrajectory
        Truth trajectory.
    """
    C_hist = dcm_history_body_to_ned_from_ypr(yaw_rad, pitch_rad, roll_rad)
    return build_truth_trajectory_from_position_attitude_history(
        times_s=times_s,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        height_m=height_m,
        C_n_b=C_hist,
        omega_nb_b_radps=omega_nb_b_radps,
    )


def build_truth_trajectory_from_velocity_ypr_history(
    times_s: ArrayLike,
    v_ned_mps: ArrayLike,
    yaw_rad: ArrayLike,
    pitch_rad: ArrayLike,
    roll_rad: ArrayLike,
    *,
    lat0_rad: float,
    lon0_rad: float,
    height0_m: float,
    v_dot_ned_mps2: ArrayLike | None = None,
    omega_nb_b_radps: ArrayLike | None = None,
) -> TruthTrajectory:
    """
    Convenience wrapper that builds attitude from yaw-pitch-roll histories and
    then constructs a truth trajectory from sampled NED velocity.

    Parameters
    ----------
    times_s : array-like, shape (N,)
        Sample times [s].
    v_ned_mps : array-like, shape (N, 3)
        NED velocity history [m/s].
    yaw_rad, pitch_rad, roll_rad : array-like, shape (N,)
        Yaw-pitch-roll histories [rad].
    lat0_rad, lon0_rad, height0_m : float
        Initial geodetic position.
    v_dot_ned_mps2 : array-like, shape (N, 3), optional
        Optional velocity-derivative history.
    omega_nb_b_radps : array-like, shape (N, 3), optional
        Optional body-rate history.

    Returns
    -------
    TruthTrajectory
        Truth trajectory.
    """
    C_hist = dcm_history_body_to_ned_from_ypr(yaw_rad, pitch_rad, roll_rad)
    return build_truth_trajectory_from_velocity_attitude_history(
        times_s=times_s,
        v_ned_mps=v_ned_mps,
        C_n_b=C_hist,
        lat0_rad=lat0_rad,
        lon0_rad=lon0_rad,
        height0_m=height0_m,
        v_dot_ned_mps2=v_dot_ned_mps2,
        omega_nb_b_radps=omega_nb_b_radps,
    )


__all__ = [
    "FloatArray",
    "TrajectorySample",
    "TruthTrajectory",
    "body_rate_history_from_dcm_history",
    "build_truth_trajectory",
    "build_truth_trajectory_from_position_attitude_history",
    "build_truth_trajectory_from_position_ypr_history",
    "build_truth_trajectory_from_velocity_attitude_history",
    "build_truth_trajectory_from_velocity_ypr_history",
    "dcm_body_to_ned_from_ypr",
    "dcm_history_body_to_ned_from_ypr",
    "geodetic_rates_from_velocity_ned",
    "integrate_geodetic_from_velocity_ned",
    "velocity_ned_from_geodetic_rates",
    "ypr_from_dcm_body_to_ned",
]