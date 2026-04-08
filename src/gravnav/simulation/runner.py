"""
runner.py

End-to-end scenario orchestration for the gravity-aided navigation simulator.

Why this file exists
--------------------
The repository already has the hard parts of the stack implemented:

- truth trajectory generation
- IMU / gravimeter / depth / velocity-aid sensor models
- local-level error-state INS propagation
- linearized aiding fusion helpers
- particle-filter gravity map matching
- integrity monitoring
- typed result containers

The remaining missing piece is the *orchestration layer* that turns all of those
into one runnable scenario simulation.

This module therefore provides:

1) a lightweight runner configuration model
2) practical scheduling policies for sparse aiding channels
3) initial INS covariance construction from intuitive NED uncertainty inputs
4) truth -> sensor -> estimator execution for one full scenario
5) logging into `ScenarioSimulationResult`

Design philosophy
-----------------
This file intentionally does not add new navigation physics. It only wires
together the physics and estimation layers that already exist elsewhere in the
repository.

Conventions
-----------
- Navigation frame: local NED
- Position state representation in the INS: geodetic [lat, lon, h]
- Gravity observation for map matching: scalar disturbance [m/s^2]
- Depth aiding sign convention:
      d = h_ref - h
- All sensor and estimator logs are stored in the repository's existing
  `ScenarioSimulationResult` container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.error_state_ins import (
    ERR_ATT,
    ERR_BA,
    ERR_BG,
    ERR_POS,
    ERR_VEL,
    ErrorStateINS,
    ErrorStateINSProcessNoise,
)
from ..estimators.fusion import (
    apply_depth_sensor_measurement,
    apply_depth_sensor_measurement_height_only,
    apply_geodetic_position_measurement,
    apply_velocity_aid_measurement,
    apply_velocity_aid_measurement_velocity_only,
    diagonal_covariance_from_std,
    summarize_update_result,
)
from ..estimators.integrity import IntegrityMonitor
from ..estimators.map_match_pf import (
    GravityMapParticleFilter,
    MapMatchPFSpec,
    geodetic_covariance_from_ned_covariance,
)
from ..physics.gravity_map import GravityGridMap
from ..sensors.depth import DepthMeasurement, DepthSensor, DepthSensorSpec
from ..sensors.gravimeter import (
    GravimeterMeasurement,
    GravimeterSpec,
    ScalarGravimeterSensor,
)
from ..sensors.imu import (
    IMUSensor,
    IMUSpec,
    build_imu_truth_kinematics,
    build_interval_imu_truth_kinematics,
)
from ..sensors.velocity_aid import (
    VelocityAidMeasurement,
    VelocityAidSensor,
    VelocityAidSpec,
)
from .results import (
    ScenarioSimulationResult,
    SimulationEstimatorLog,
    SimulationMetadata,
    SimulationSensorLog,
)
from ..truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from ..truth.trajectory import TruthTrajectory
from ..utils.config import load_scenario_spec

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Convert a scalar or shape-(3,) input into a shape-(3,) vector.
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


def _positive_scalar(x: float, *, name: str) -> float:
    """
    Validate and return a positive scalar.
    """
    value = float(x)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _should_use_sensor_turn_on_bias(
    configured_std: ArrayLike | float,
) -> bool:
    """
    Return True when a bias-uncertainty config is effectively all zeros.
    """
    arr = _axis3(configured_std, name="configured_std")
    return bool(np.all(arr == 0.0))


# -----------------------------------------------------------------------------
# Schedules and runner configuration
# -----------------------------------------------------------------------------


@dataclass
class PeriodicUpdateSchedule:
    """
    Lightweight triggering policy for sparse aiding channels.

    Exactly one of the following common patterns can be used:
    - `every_steps = N`: trigger every N simulation samples
    - `period_s = T`: trigger when time crosses multiples of T
    - neither provided: trigger on every sample

    Parameters
    ----------
    enabled : bool, default=True
        Master enable flag.
    every_steps : int, optional
        Trigger every N samples.
    period_s : float, optional
        Trigger every T seconds using the scenario time base.
    start_time_s : float, default=0.0
        Earliest time at which triggering is allowed.

    Notes
    -----
    The schedule is evaluated on each interval `(t_prev, t_now]` so that
    period-based triggering remains robust when floating-point time grids are
    only approximately uniform.
    """

    enabled: bool = True
    every_steps: Optional[int] = None
    period_s: Optional[float] = None
    start_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.start_time_s = float(self.start_time_s)

        if self.every_steps is not None:
            self.every_steps = int(self.every_steps)
            if self.every_steps <= 0:
                raise ValueError("every_steps must be positive when provided.")

        if self.period_s is not None:
            self.period_s = float(self.period_s)
            if self.period_s <= 0.0:
                raise ValueError("period_s must be positive when provided.")

    def should_trigger(
        self,
        sample_index: int,
        time_s: float,
        previous_time_s: float,
    ) -> bool:
        """
        Return True if the schedule should fire for the current sample.
        """
        if not self.enabled:
            return False

        t_now = float(time_s)
        t_prev = float(previous_time_s)
        start = float(self.start_time_s)

        if t_now < start:
            return False

        if self.every_steps is not None:
            return bool(sample_index % self.every_steps == 0)

        if self.period_s is None:
            return True

        period = float(self.period_s)
        eps = 1.0e-12 * max(1.0, abs(period), abs(t_now))

        if t_prev < start:
            prev_bucket = -1
        else:
            prev_bucket = int(np.floor((t_prev - start + eps) / period))

        now_bucket = int(np.floor((t_now - start + eps) / period))
        return bool(now_bucket > prev_bucket)


@dataclass
class VelocityAidFusionConfig:
    """
    Configuration for external velocity aiding into the INS.

    Parameters
    ----------
    enabled : bool, default=True
        Whether velocity aiding is fused.
    schedule : PeriodicUpdateSchedule
        Triggering policy for velocity samples.
    measurement_std_mps : scalar or shape (3,), default=(0.05, 0.05, 0.05)
        Measurement standard deviation used in the fusion covariance.
    measurement_frame : {"ned", "body"}, default="ned"
        Which velocity-aid measurement interface to use.
    nis_threshold : float, optional
        Optional innovation gate threshold.
    velocity_only_update : bool, default=True
        When True, constrain the velocity-aid correction to the velocity state
        instead of allowing the update to move attitude and bias states through
        the full covariance structure. This is the safer default for the
        repository's current compact INS model.
    """

    enabled: bool = True
    schedule: PeriodicUpdateSchedule = field(default_factory=PeriodicUpdateSchedule)
    measurement_std_mps: ArrayLike | float = (0.05, 0.05, 0.05)
    measurement_frame: str = "ned"
    nis_threshold: Optional[float] = None
    velocity_only_update: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.measurement_std_mps = _nonnegative_axis3(
            self.measurement_std_mps,
            name="measurement_std_mps",
        )
        self.measurement_frame = str(self.measurement_frame).strip().lower()
        if self.measurement_frame not in {"ned", "body"}:
            raise ValueError("measurement_frame must be 'ned' or 'body'.")
        self.velocity_only_update = bool(self.velocity_only_update)
        if self.nis_threshold is not None and float(self.nis_threshold) < 0.0:
            raise ValueError("nis_threshold must be nonnegative when provided.")


@dataclass
class DepthFusionConfig:
    """
    Configuration for scalar depth aiding into the INS.

    Parameters
    ----------
    enabled : bool, default=True
        Whether depth aiding is fused.
    schedule : PeriodicUpdateSchedule
        Triggering policy for depth samples.
    measurement_std_m : float, default=0.5
        Depth-measurement standard deviation [m].
    nis_threshold : float, optional
        Optional innovation gate threshold.
    height_only_update : bool, default=True
        When True, constrain the depth correction to the height state instead of
        feeding a scalar depth residual through the full cross-covariance
        structure. This is the safer default for the repository's current
        compact INS model.
    """

    enabled: bool = True
    schedule: PeriodicUpdateSchedule = field(default_factory=PeriodicUpdateSchedule)
    measurement_std_m: float = 0.5
    nis_threshold: Optional[float] = None
    height_only_update: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.measurement_std_m = _positive_scalar(
            self.measurement_std_m,
            name="measurement_std_m",
        )
        self.height_only_update = bool(self.height_only_update)
        if self.nis_threshold is not None and float(self.nis_threshold) < 0.0:
            raise ValueError("nis_threshold must be nonnegative when provided.")


@dataclass
class MapMatchFeedbackConfig:
    """
    Configuration for gravity map matching and optional PF-to-INS feedback.

    Parameters
    ----------
    enabled : bool, default=True
        Enable the PF map-matching layer.
    schedule : PeriodicUpdateSchedule
        Triggering policy for PF gravity updates.
    pf_spec : MapMatchPFSpec
        PF algorithm specification.
    gravity_meas_std_mps2 : float, optional
        Optional measurement standard deviation override for the gravity update.
    use_depth_measurement : bool, default=True
        Whether to include depth likelihood when depth data is available.
    use_last_depth_measurement : bool, default=True
        Whether the PF may reuse the latest depth measurement if no new one was
        sampled on the current step.
    depth_meas_std_m : float, optional
        Optional depth measurement standard deviation override for the PF.
    inject_position_to_ins : bool, default=False
        Whether PF mean/covariance are converted into a geodetic pseudo-measurement
        and fused back into the INS.
    feedback_covariance_inflation : float, default=1.0
        Scalar inflation applied to PF covariance before INS feedback.
    feedback_min_std_geodetic : scalar or shape (3,), default=(0, 0, 0)
        Lower bound on PF-derived geodetic standard deviations before feedback.
    feedback_nis_threshold : float, optional
        Optional gate threshold for the PF pseudo-measurement update.
    """

    enabled: bool = True
    schedule: PeriodicUpdateSchedule = field(default_factory=PeriodicUpdateSchedule)
    pf_spec: MapMatchPFSpec = field(default_factory=MapMatchPFSpec)
    gravity_meas_std_mps2: Optional[float] = None
    use_depth_measurement: bool = True
    use_last_depth_measurement: bool = True
    depth_meas_std_m: Optional[float] = None
    inject_position_to_ins: bool = False
    feedback_covariance_inflation: float = 1.0
    feedback_min_std_geodetic: ArrayLike | float = (0.0, 0.0, 0.0)
    feedback_nis_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.gravity_meas_std_mps2 is not None:
            self.gravity_meas_std_mps2 = _positive_scalar(
                self.gravity_meas_std_mps2,
                name="gravity_meas_std_mps2",
            )
        if self.depth_meas_std_m is not None:
            self.depth_meas_std_m = _positive_scalar(
                self.depth_meas_std_m,
                name="depth_meas_std_m",
            )
        self.feedback_covariance_inflation = _positive_scalar(
            self.feedback_covariance_inflation,
            name="feedback_covariance_inflation",
        )
        self.feedback_min_std_geodetic = _nonnegative_axis3(
            self.feedback_min_std_geodetic,
            name="feedback_min_std_geodetic",
        )
        if self.feedback_nis_threshold is not None and float(self.feedback_nis_threshold) < 0.0:
            raise ValueError(
                "feedback_nis_threshold must be nonnegative when provided."
            )


@dataclass
class IntegrityMonitorConfig:
    """
    Configuration for covariance/error integrity monitoring.
    """

    enabled: bool = True
    horizontal_alert_limit_m: Optional[float] = None
    vertical_alert_limit_m: Optional[float] = None
    horizontal_k_sigma: float = 6.0
    vertical_k_sigma: float = 6.0
    radial_k_sigma: float = 6.0
    consistency_confidence: float = 0.95

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)

        if self.horizontal_alert_limit_m is not None:
            self.horizontal_alert_limit_m = _positive_scalar(
                self.horizontal_alert_limit_m,
                name="horizontal_alert_limit_m",
            )
        if self.vertical_alert_limit_m is not None:
            self.vertical_alert_limit_m = _positive_scalar(
                self.vertical_alert_limit_m,
                name="vertical_alert_limit_m",
            )

        self.horizontal_k_sigma = _positive_scalar(
            self.horizontal_k_sigma,
            name="horizontal_k_sigma",
        )
        self.vertical_k_sigma = _positive_scalar(
            self.vertical_k_sigma,
            name="vertical_k_sigma",
        )
        self.radial_k_sigma = _positive_scalar(
            self.radial_k_sigma,
            name="radial_k_sigma",
        )

        c = float(self.consistency_confidence)
        if not (0.0 < c < 1.0):
            raise ValueError("consistency_confidence must lie in (0, 1).")
        self.consistency_confidence = c


@dataclass
class SimulationRunnerConfig:
    """
    End-to-end orchestration settings for one scenario run.

    Parameters
    ----------
    process_noise : ErrorStateINSProcessNoise, optional
        INS process-noise model. If omitted, it is derived from the IMU spec.
    initial_position_std_m : scalar or shape (3,), default=(25, 25, 5)
        Initial position uncertainty in local NED metres.
    initial_velocity_std_mps : scalar or shape (3,), default=(0.5, 0.5, 0.2)
        Initial velocity uncertainty [m/s].
    initial_attitude_std_rad : scalar or shape (3,), default=(2deg, 2deg, 5deg)
        Initial small-angle attitude uncertainty [rad].
    initial_gyro_bias_std_radps : scalar or shape (3,), default=0
        Initial gyroscope bias uncertainty [rad/s].
    initial_accel_bias_std_mps2 : scalar or shape (3,), default=0
        Initial accelerometer bias uncertainty [m/s^2].
    reference_surface_height_m : float, default=0.0
        Reference surface used for signed depth.
    gravity_override_mps2 : float, optional
        Optional scalar gravity magnitude override forwarded to IMU truth helpers.
    velocity_aid : VelocityAidFusionConfig
        Velocity-aid orchestration policy.
    depth_aid : DepthFusionConfig
        Depth-aid orchestration policy.
    map_match : MapMatchFeedbackConfig
        Gravity PF and PF-feedback policy.
    integrity : IntegrityMonitorConfig
        Integrity-monitoring policy.
    metadata_extra : dict[str, Any], default={}
        Extra metadata copied into the run result.
    """

    process_noise: Optional[ErrorStateINSProcessNoise] = None
    initial_position_std_m: ArrayLike | float = (25.0, 25.0, 5.0)
    initial_velocity_std_mps: ArrayLike | float = (0.50, 0.50, 0.20)
    initial_attitude_std_rad: ArrayLike | float = (
        np.deg2rad(2.0),
        np.deg2rad(2.0),
        np.deg2rad(5.0),
    )
    initial_gyro_bias_std_radps: ArrayLike | float = 0.0
    initial_accel_bias_std_mps2: ArrayLike | float = 0.0
    reference_surface_height_m: float = 0.0
    gravity_override_mps2: Optional[float] = None

    velocity_aid: VelocityAidFusionConfig = field(default_factory=VelocityAidFusionConfig)
    depth_aid: DepthFusionConfig = field(default_factory=DepthFusionConfig)
    map_match: MapMatchFeedbackConfig = field(default_factory=MapMatchFeedbackConfig)
    integrity: IntegrityMonitorConfig = field(default_factory=IntegrityMonitorConfig)

    metadata_extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_position_std_m = _nonnegative_axis3(
            self.initial_position_std_m,
            name="initial_position_std_m",
        )
        self.initial_velocity_std_mps = _nonnegative_axis3(
            self.initial_velocity_std_mps,
            name="initial_velocity_std_mps",
        )
        self.initial_attitude_std_rad = _nonnegative_axis3(
            self.initial_attitude_std_rad,
            name="initial_attitude_std_rad",
        )
        self.initial_gyro_bias_std_radps = _nonnegative_axis3(
            self.initial_gyro_bias_std_radps,
            name="initial_gyro_bias_std_radps",
        )
        self.initial_accel_bias_std_mps2 = _nonnegative_axis3(
            self.initial_accel_bias_std_mps2,
            name="initial_accel_bias_std_mps2",
        )
        self.reference_surface_height_m = float(self.reference_surface_height_m)
        if self.gravity_override_mps2 is not None:
            self.gravity_override_mps2 = float(self.gravity_override_mps2)

        if not isinstance(self.metadata_extra, dict):
            self.metadata_extra = dict(self.metadata_extra)


# -----------------------------------------------------------------------------
# Construction helpers
# -----------------------------------------------------------------------------


def process_noise_from_imu_spec(
    imu_spec: IMUSpec,
    *,
    name: Optional[str] = None,
) -> ErrorStateINSProcessNoise:
    """
    Build an INS process-noise model directly from an IMU sensor spec.

    This keeps the INS covariance propagation aligned with the stochastic IMU
    driving the nominal mechanization unless the caller explicitly overrides it.
    """
    return ErrorStateINSProcessNoise(
        gyro_white_noise_radps_per_sqrt_hz=imu_spec.gyro_noise_density_radps_per_sqrt_hz,
        accel_white_noise_mps2_per_sqrt_hz=imu_spec.accel_noise_density_mps2_per_sqrt_hz,
        gyro_bias_random_walk_radps_per_sqrt_s=imu_spec.gyro_bias_random_walk_radps_per_sqrt_s,
        accel_bias_random_walk_mps2_per_sqrt_s=imu_spec.accel_bias_random_walk_mps2_per_sqrt_s,
        name=imu_spec.name if name is None else str(name),
    )


def build_initial_covariance_geodetic(
    *,
    lat_ref_rad: float,
    height_ref_m: float,
    position_std_m: ArrayLike | float = (25.0, 25.0, 5.0),
    velocity_std_mps: ArrayLike | float = (0.50, 0.50, 0.20),
    attitude_std_rad: ArrayLike | float = (
        np.deg2rad(2.0),
        np.deg2rad(2.0),
        np.deg2rad(5.0),
    ),
    gyro_bias_std_radps: ArrayLike | float = 0.0,
    accel_bias_std_mps2: ArrayLike | float = 0.0,
) -> FloatArray:
    """
    Build a practical 15x15 initial covariance for the repository's INS state.

    Parameters
    ----------
    lat_ref_rad : float
        Reference latitude used for converting local NED position covariance into
        the geodetic INS position-state covariance.
    height_ref_m : float
        Reference ellipsoidal height [m].
    position_std_m : scalar or shape (3,), default=(25, 25, 5)
        Initial NED position standard deviations [m].
    velocity_std_mps : scalar or shape (3,), default=(0.5, 0.5, 0.2)
        Initial NED velocity standard deviations [m/s].
    attitude_std_rad : scalar or shape (3,), default=(2deg, 2deg, 5deg)
        Initial small-angle attitude standard deviations [rad].
    gyro_bias_std_radps : scalar or shape (3,), default=0
        Initial gyro-bias standard deviations [rad/s].
    accel_bias_std_mps2 : scalar or shape (3,), default=0
        Initial accel-bias standard deviations [m/s^2].

    Returns
    -------
    np.ndarray, shape (15, 15)
        Initial error-state covariance matrix.

    Notes
    -----
    Users usually think about initial position uncertainty in metres, not in
    radians of latitude/longitude. This helper converts the local NED covariance
    into the geodetic sub-block expected by the INS.
    """
    pos_std = _nonnegative_axis3(position_std_m, name="position_std_m")
    vel_std = _nonnegative_axis3(velocity_std_mps, name="velocity_std_mps")
    att_std = _nonnegative_axis3(attitude_std_rad, name="attitude_std_rad")
    bg_std = _nonnegative_axis3(gyro_bias_std_radps, name="gyro_bias_std_radps")
    ba_std = _nonnegative_axis3(accel_bias_std_mps2, name="accel_bias_std_mps2")

    P = np.zeros((15, 15), dtype=np.float64)

    pos_geodetic_cov = geodetic_covariance_from_ned_covariance(
        lat_ref_rad=float(lat_ref_rad),
        height_ref_m=float(height_ref_m),
        ned_cov_m2=np.diag(pos_std ** 2),
    )
    P[ERR_POS, ERR_POS] = pos_geodetic_cov
    P[ERR_VEL, ERR_VEL] = np.diag(vel_std ** 2)
    P[ERR_ATT, ERR_ATT] = np.diag(att_std ** 2)
    P[ERR_BG, ERR_BG] = np.diag(bg_std ** 2)
    P[ERR_BA, ERR_BA] = np.diag(ba_std ** 2)
    return P


def resolve_truth_trajectory(
    scenario_or_truth: TruthTrajectory | ScenarioSpec | str | Path,
    *,
    dt_s: Optional[float] = None,
) -> TruthTrajectory:
    """
    Resolve a `TruthTrajectory` from either:
    - an already-built `TruthTrajectory`
    - a typed `ScenarioSpec`
    - a scenario config path
    - a built-in scenario name
    """
    if isinstance(scenario_or_truth, TruthTrajectory):
        return scenario_or_truth.copy()

    if isinstance(scenario_or_truth, ScenarioSpec):
        return build_truth_trajectory_from_scenario(scenario_or_truth, dt_s=dt_s)

    if isinstance(scenario_or_truth, (str, Path)):
        scenario = load_scenario_spec(scenario_or_truth)
        return build_truth_trajectory_from_scenario(scenario, dt_s=dt_s)

    raise TypeError(
        "scenario_or_truth must be TruthTrajectory, ScenarioSpec, str, or Path; "
        f"got {type(scenario_or_truth).__name__}."
    )


def resolve_map_model(map_model_or_path: Any | str | Path | None) -> Any | None:
    """
    Resolve a gravity-map model from either:
    - an already-instantiated object
    - a path to a `GravityGridMap` NPZ archive
    - None
    """
    if map_model_or_path is None:
        return None
    if isinstance(map_model_or_path, GravityGridMap):
        return map_model_or_path
    if isinstance(map_model_or_path, (str, Path)):
        return GravityGridMap.from_npz(map_model_or_path)
    return map_model_or_path


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


class ScenarioSimulationRunner:
    """
    End-to-end scenario runner for the current simulator stack.

    This class owns the orchestration policy between already-implemented modules.

    Responsibilities
    ----------------
    - turn truth into simulated sensor streams
    - propagate the local-level INS from IMU samples
    - optionally fuse velocity and depth aiding
    - optionally run gravity-map particle filtering
    - optionally feed PF position back into the INS
    - optionally log integrity snapshots
    - package the result into `ScenarioSimulationResult`

    Non-responsibilities
    --------------------
    - plotting
    - Monte Carlo orchestration
    - CLI parsing
    - notebook/report generation
    """

    def __init__(
        self,
        config: Optional[SimulationRunnerConfig] = None,
    ) -> None:
        self.config = SimulationRunnerConfig() if config is None else config

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _make_integrity_monitor(self) -> Optional[IntegrityMonitor]:
        """
        Build the configured integrity monitor, or return None if disabled.
        """
        cfg = self.config.integrity
        if not cfg.enabled:
            return None

        return IntegrityMonitor(
            horizontal_alert_limit_m=cfg.horizontal_alert_limit_m,
            vertical_alert_limit_m=cfg.vertical_alert_limit_m,
            horizontal_k_sigma=cfg.horizontal_k_sigma,
            vertical_k_sigma=cfg.vertical_k_sigma,
            radial_k_sigma=cfg.radial_k_sigma,
            consistency_confidence=cfg.consistency_confidence,
        )

    def _build_initial_ins(
        self,
        truth: TruthTrajectory,
        imu_sensor: IMUSensor,
    ) -> ErrorStateINS:
        """
        Build the initial INS aligned to the start of the truth trajectory.
        """
        cfg = self.config
        process_noise = (
            process_noise_from_imu_spec(imu_sensor.spec)
            if cfg.process_noise is None
            else cfg.process_noise
        )

        # If the caller leaves the initial bias uncertainty at zero, use the
        # sensor's own turn-on bias uncertainty as the prior instead of
        # artificially locking the bias states to exactly zero.
        gyro_bias_std = (
            imu_sensor.spec.gyro_turn_on_bias_std_radps
            if _should_use_sensor_turn_on_bias(cfg.initial_gyro_bias_std_radps)
            else cfg.initial_gyro_bias_std_radps
        )
        accel_bias_std = (
            imu_sensor.spec.accel_turn_on_bias_std_mps2
            if _should_use_sensor_turn_on_bias(cfg.initial_accel_bias_std_mps2)
            else cfg.initial_accel_bias_std_mps2
        )

        P0 = build_initial_covariance_geodetic(
            lat_ref_rad=float(truth.lat_rad[0]),
            height_ref_m=float(truth.height_m[0]),
            position_std_m=cfg.initial_position_std_m,
            velocity_std_mps=cfg.initial_velocity_std_mps,
            attitude_std_rad=cfg.initial_attitude_std_rad,
            gyro_bias_std_radps=gyro_bias_std,
            accel_bias_std_mps2=accel_bias_std,
        )

        return ErrorStateINS.from_truth_trajectory_start(
            truth,
            process_noise=process_noise,
            P0=P0,
        )

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    def run_truth(
        self,
        truth: TruthTrajectory,
        *,
        imu_sensor: IMUSensor,
        gravimeter_sensor: Optional[ScalarGravimeterSensor] = None,
        depth_sensor: Optional[DepthSensor] = None,
        velocity_aid_sensor: Optional[VelocityAidSensor] = None,
        map_model: Any | str | Path | None = None,
        metadata: Optional[SimulationMetadata] = None,
    ) -> ScenarioSimulationResult:
        """
        Run the full simulation stack over a supplied truth trajectory.

        Parameters
        ----------
        truth : TruthTrajectory
            Truth trajectory to simulate.
        imu_sensor : IMUSensor
            IMU simulator used for every propagation step.
        gravimeter_sensor : ScalarGravimeterSensor, optional
            Scalar gravimeter simulator used for map matching.
        depth_sensor : DepthSensor, optional
            Depth-aiding sensor simulator.
        velocity_aid_sensor : VelocityAidSensor, optional
            External velocity-aid simulator.
        map_model : object, path, or None, optional
            Gravity-map object or path to a `GravityGridMap` NPZ file.
        metadata : SimulationMetadata, optional
            Optional metadata template.

        Returns
        -------
        ScenarioSimulationResult
            Full typed run result.
        """
        if len(truth) < 1:
            raise ValueError("truth must contain at least one sample.")

        cfg = self.config
        sensors = SimulationSensorLog()
        estimators = SimulationEstimatorLog()

        meta = SimulationMetadata() if metadata is None else metadata.copy()
        meta.extra.update(dict(cfg.metadata_extra))

        ins = self._build_initial_ins(truth, imu_sensor)

        pf: Optional[GravityMapParticleFilter] = None
        resolved_map = resolve_map_model(map_model)
        if cfg.map_match.enabled and gravimeter_sensor is not None and resolved_map is not None:
            pf = GravityMapParticleFilter(cfg.map_match.pf_spec, resolved_map)
            pf.reset_from_ins(ins)

        integrity = self._make_integrity_monitor()

        estimators.append_ins_state(ins)

        if integrity is not None:
            snap0 = integrity.snapshot_from_ins(
                ins,
                true_lat_rad=float(truth.lat_rad[0]),
                true_lon_rad=float(truth.lon_rad[0]),
                true_height_m=float(truth.height_m[0]),
                time_s=float(truth.time_s[0]),
            )
            estimators.integrity_snapshots.append(snap0)

        velocity_R = diagonal_covariance_from_std(cfg.velocity_aid.measurement_std_mps)
        depth_variance_m2 = float(cfg.depth_aid.measurement_std_m ** 2)

        last_depth_measurement: Optional[DepthMeasurement] = None
        last_velocity_sample_time_s: Optional[float] = None
        last_depth_sample_time_s: Optional[float] = None

        for k in range(1, len(truth)):
            t_now = float(truth.time_s[k])
            t_prev = float(truth.time_s[k - 1])
            dt = t_now - t_prev
            if dt <= 0.0:
                raise ValueError(
                    f"truth.time_s must be strictly increasing; got dt={dt} at index {k}."
                )

            # ----------------------------------------------------------
            # Truth -> ideal inertial quantities
            # ----------------------------------------------------------
            kin_imu = build_interval_imu_truth_kinematics(
                v_ned_prev_mps=truth.v_ned_mps[k - 1],
                v_ned_next_mps=truth.v_ned_mps[k],
                C_n_b_prev=truth.C_n_b[k - 1],
                C_n_b_next=truth.C_n_b[k],
                lat_prev_rad=float(truth.lat_rad[k - 1]),
                height_prev_m=float(truth.height_m[k - 1]),
                dt_s=dt,
                gravity_override_mps2=cfg.gravity_override_mps2,
            )

            kin_point = build_imu_truth_kinematics(
                v_dot_ned_mps2=truth.v_dot_ned_mps2[k],
                v_ned_mps=truth.v_ned_mps[k],
                C_n_b=truth.C_n_b[k],
                omega_nb_b_radps=truth.omega_nb_b_radps[k],
                lat_rad=float(truth.lat_rad[k]),
                height_m=float(truth.height_m[k]),
                gravity_override_mps2=cfg.gravity_override_mps2,
            )

            # ----------------------------------------------------------
            # IMU propagation
            # ----------------------------------------------------------
            imu_meas = imu_sensor.measure_from_ideal(
                ideal_omega_ib_b_radps=kin_imu.omega_ib_b_radps,
                ideal_f_ib_b_mps2=kin_imu.f_b_mps2,
                dt_s=dt,
                time_s=t_now,
            )
            sensors.imu_samples.append(imu_meas)

            ins.predict(
                imu_meas.omega_ib_b_radps,
                imu_meas.f_ib_b_mps2,
                dt,
            )

            if pf is not None:
                pf.predict_from_ins(ins, dt)

            # ----------------------------------------------------------
            # Depth aiding
            # ----------------------------------------------------------
            current_depth_measurement: Optional[DepthMeasurement] = None
            if depth_sensor is not None and cfg.depth_aid.enabled:
                if cfg.depth_aid.schedule.should_trigger(k, t_now, t_prev):
                    dt_depth = (
                        dt
                        if last_depth_sample_time_s is None
                        else t_now - last_depth_sample_time_s
                    )

                    current_depth_measurement = depth_sensor.measure_depth_from_height(
                        height_m=float(truth.height_m[k]),
                        dt_s=dt_depth,
                        reference_surface_height_m=cfg.reference_surface_height_m,
                        time_s=t_now,
                    )
                    sensors.depth_samples.append(current_depth_measurement)

                    last_depth_sample_time_s = t_now
                    last_depth_measurement = current_depth_measurement

                    if cfg.depth_aid.height_only_update:
                        depth_update = apply_depth_sensor_measurement_height_only(
                            ins,
                            current_depth_measurement,
                            depth_variance_m2=depth_variance_m2,
                            reference_surface_height_m=cfg.reference_surface_height_m,
                            nis_threshold=cfg.depth_aid.nis_threshold,
                            label="depth_height_only",
                        )
                    else:
                        depth_update = apply_depth_sensor_measurement(
                            ins,
                            current_depth_measurement,
                            depth_variance_m2=depth_variance_m2,
                            reference_surface_height_m=cfg.reference_surface_height_m,
                            nis_threshold=cfg.depth_aid.nis_threshold,
                            label="depth",
                        )
                    estimators.add_custom_sample(
                        "depth_updates",
                        summarize_update_result(depth_update),
                    )

            # ----------------------------------------------------------
            # Velocity aiding
            # ----------------------------------------------------------
            if velocity_aid_sensor is not None and cfg.velocity_aid.enabled:
                if cfg.velocity_aid.schedule.should_trigger(k, t_now, t_prev):
                    dt_vel = (
                        dt
                        if last_velocity_sample_time_s is None
                        else t_now - last_velocity_sample_time_s
                    )

                    if cfg.velocity_aid.measurement_frame == "body":
                        vel_meas = velocity_aid_sensor.measure_velocity_body_from_truth(
                            velocity_ned_mps=truth.v_ned_mps[k],
                            C_n_b=truth.C_n_b[k],
                            dt_s=dt_vel,
                            time_s=t_now,
                        )
                    else:
                        vel_meas = velocity_aid_sensor.measure_velocity_ned_from_truth(
                            velocity_ned_mps=truth.v_ned_mps[k],
                            dt_s=dt_vel,
                            time_s=t_now,
                        )

                    sensors.velocity_aid_samples.append(vel_meas)
                    last_velocity_sample_time_s = t_now

                    if cfg.velocity_aid.velocity_only_update:
                        vel_update = apply_velocity_aid_measurement_velocity_only(
                            ins,
                            vel_meas,
                            velocity_R,
                            nis_threshold=cfg.velocity_aid.nis_threshold,
                            label=f"velocity_{vel_meas.frame}_velocity_only",
                        )
                    else:
                        vel_update = apply_velocity_aid_measurement(
                            ins,
                            vel_meas,
                            velocity_R,
                            nis_threshold=cfg.velocity_aid.nis_threshold,
                            label=f"velocity_{vel_meas.frame}",
                        )
                    estimators.add_custom_sample(
                        "velocity_updates",
                        summarize_update_result(vel_update),
                    )

            # ----------------------------------------------------------
            # Gravimeter sampling
            # ----------------------------------------------------------
            gravimeter_meas: Optional[GravimeterMeasurement] = None
            if gravimeter_sensor is not None:
                gravimeter_meas = gravimeter_sensor.measure_disturbance_from_specific_force_body(
                    specific_force_body_mps2=kin_point.f_b_mps2,
                    C_n_b=truth.C_n_b[k],
                    v_dot_ned_mps2=truth.v_dot_ned_mps2[k],
                    v_ned_mps=truth.v_ned_mps[k],
                    lat_rad=float(truth.lat_rad[k]),
                    height_m=float(truth.height_m[k]),
                    dt_s=dt,
                    body_angular_rate_b_radps=truth.omega_nb_b_radps[k],
                    time_s=t_now,
                )
                sensors.gravimeter_samples.append(gravimeter_meas)

            # ----------------------------------------------------------
            # Gravity map matching and optional PF feedback
            # ----------------------------------------------------------
            if pf is not None and gravimeter_meas is not None:
                if cfg.map_match.schedule.should_trigger(k, t_now, t_prev):
                    depth_for_pf: Optional[DepthMeasurement] = None
                    if cfg.map_match.use_depth_measurement:
                        if current_depth_measurement is not None:
                            depth_for_pf = current_depth_measurement
                        elif cfg.map_match.use_last_depth_measurement:
                            depth_for_pf = last_depth_measurement

                    pf_update = pf.update_from_gravimeter_measurement(
                        gravimeter_meas,
                        gravity_meas_std_mps2=cfg.map_match.gravity_meas_std_mps2,
                        ins_or_state=ins,
                        depth_measurement=depth_for_pf,
                        depth_meas_std_m=cfg.map_match.depth_meas_std_m,
                        reference_surface_height_m=cfg.reference_surface_height_m,
                    )
                    estimators.pf_updates.append(pf_update)

                    if cfg.map_match.inject_position_to_ins:
                        z_pf, R_pf = pf.as_geodetic_pseudo_measurement(
                            min_std_geodetic=cfg.map_match.feedback_min_std_geodetic,
                            covariance_inflation=cfg.map_match.feedback_covariance_inflation,
                        )
                        z_pf = np.asarray(z_pf, dtype=np.float64).reshape(3)
                        R_pf = np.asarray(R_pf, dtype=np.float64)

                        valid_feedback = (
                            np.all(np.isfinite(z_pf))
                            and np.all(np.isfinite(R_pf))
                            and R_pf.shape == (3, 3)
                            and abs(float(z_pf[0])) <= 0.5 * np.pi
                        )

                        if valid_feedback:
                            pf_feedback = apply_geodetic_position_measurement(
                                ins,
                                measured_lat_rad=float(z_pf[0]),
                                measured_lon_rad=float(z_pf[1]),
                                measured_height_m=float(z_pf[2]),
                                R=R_pf,
                                nis_threshold=cfg.map_match.feedback_nis_threshold,
                                label="pf_position",
                                time_s=t_now,
                            )
                            estimators.add_custom_sample(
                                "pf_feedback_updates",
                                summarize_update_result(pf_feedback),
                            )
                        else:
                            estimators.add_custom_sample(
                                "pf_feedback_updates",
                                {
                                    "label": "pf_position",
                                    "time_s": t_now,
                                    "accepted": False,
                                    "reason": "invalid_pseudo_measurement",
                                    "z": z_pf.copy(),
                                    "R": R_pf.copy(),
                                },
                            )

            # ----------------------------------------------------------
            # Log estimator state after all current-step updates
            # ----------------------------------------------------------
            estimators.append_ins_state(ins)

            if integrity is not None:
                snap = integrity.snapshot_from_ins(
                    ins,
                    true_lat_rad=float(truth.lat_rad[k]),
                    true_lon_rad=float(truth.lon_rad[k]),
                    true_height_m=float(truth.height_m[k]),
                    time_s=t_now,
                )
                estimators.integrity_snapshots.append(snap)

        return ScenarioSimulationResult(
            truth=truth.copy(),
            sensors=sensors,
            estimators=estimators,
            metadata=meta,
        )

    def run_scenario(
        self,
        scenario_or_truth: TruthTrajectory | ScenarioSpec | str | Path,
        *,
        imu_sensor: IMUSensor,
        gravimeter_sensor: Optional[ScalarGravimeterSensor] = None,
        depth_sensor: Optional[DepthSensor] = None,
        velocity_aid_sensor: Optional[VelocityAidSensor] = None,
        map_model: Any | str | Path | None = None,
        metadata: Optional[SimulationMetadata] = None,
        dt_s: Optional[float] = None,
    ) -> ScenarioSimulationResult:
        """
        Resolve a scenario/trajectory input and run the full simulation.

        Parameters
        ----------
        scenario_or_truth : TruthTrajectory, ScenarioSpec, str, or Path
            Scenario source.
        imu_sensor, gravimeter_sensor, depth_sensor, velocity_aid_sensor
            Concrete sensor objects.
        map_model : object, path, or None, optional
            Gravity-map backend or path to an NPZ gravity grid.
        metadata : SimulationMetadata, optional
            Optional metadata template.
        dt_s : float, optional
            Optional sample period override forwarded when a scenario must be
            converted into a truth trajectory.

        Returns
        -------
        ScenarioSimulationResult
            Full typed run result.
        """
        truth = resolve_truth_trajectory(scenario_or_truth, dt_s=dt_s)

        meta = SimulationMetadata() if metadata is None else metadata.copy()
        if meta.scenario_name is None:
            if isinstance(scenario_or_truth, ScenarioSpec):
                meta.scenario_name = scenario_or_truth.name
            elif isinstance(scenario_or_truth, (str, Path)):
                meta.scenario_name = Path(str(scenario_or_truth)).stem

        return self.run_truth(
            truth,
            imu_sensor=imu_sensor,
            gravimeter_sensor=gravimeter_sensor,
            depth_sensor=depth_sensor,
            velocity_aid_sensor=velocity_aid_sensor,
            map_model=map_model,
            metadata=meta,
        )

    def run_with_specs(
        self,
        scenario_or_truth: TruthTrajectory | ScenarioSpec | str | Path,
        *,
        imu_spec: IMUSpec,
        gravimeter_spec: Optional[GravimeterSpec] = None,
        depth_spec: Optional[DepthSensorSpec] = None,
        velocity_aid_spec: Optional[VelocityAidSpec] = None,
        map_model: Any | str | Path | None = None,
        metadata: Optional[SimulationMetadata] = None,
        dt_s: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> ScenarioSimulationResult:
        """
        Convenience wrapper that instantiates sensor objects from specs.

        Parameters
        ----------
        scenario_or_truth : TruthTrajectory, ScenarioSpec, str, or Path
            Scenario source.
        imu_spec : IMUSpec
            IMU spec used to instantiate the IMU simulator.
        gravimeter_spec : GravimeterSpec, optional
            Gravimeter spec.
        depth_spec : DepthSensorSpec, optional
            Depth-sensor spec.
        velocity_aid_spec : VelocityAidSpec, optional
            Velocity-aid spec.
        map_model : object, path, or None, optional
            Gravity-map backend or NPZ path.
        metadata : SimulationMetadata, optional
            Optional metadata template.
        dt_s : float, optional
            Optional scenario sample-period override.
        seed : int, optional
            Root seed for reproducible child sensor RNGs.

        Returns
        -------
        ScenarioSimulationResult
            Full typed run result.
        """
        root_rng = np.random.default_rng(seed)

        def child_rng() -> np.random.Generator:
            child_seed = int(
                root_rng.integers(
                    0,
                    np.iinfo(np.uint64).max,
                    dtype=np.uint64,
                )
            )
            return np.random.default_rng(child_seed)

        imu_sensor = IMUSensor(imu_spec, rng=child_rng())
        gravimeter_sensor = (
            None
            if gravimeter_spec is None
            else ScalarGravimeterSensor(gravimeter_spec, rng=child_rng())
        )
        depth_sensor = (
            None
            if depth_spec is None
            else DepthSensor(depth_spec, rng=child_rng())
        )
        velocity_aid_sensor = (
            None
            if velocity_aid_spec is None
            else VelocityAidSensor(velocity_aid_spec, rng=child_rng())
        )

        meta = SimulationMetadata() if metadata is None else metadata.copy()
        meta.rng_state = {"seed": None if seed is None else int(seed)}

        return self.run_scenario(
            scenario_or_truth,
            imu_sensor=imu_sensor,
            gravimeter_sensor=gravimeter_sensor,
            depth_sensor=depth_sensor,
            velocity_aid_sensor=velocity_aid_sensor,
            map_model=map_model,
            metadata=meta,
            dt_s=dt_s,
        )


__all__ = [
    "DepthFusionConfig",
    "IntegrityMonitorConfig",
    "MapMatchFeedbackConfig",
    "PeriodicUpdateSchedule",
    "ScenarioSimulationRunner",
    "SimulationRunnerConfig",
    "VelocityAidFusionConfig",
    "build_initial_covariance_geodetic",
    "process_noise_from_imu_spec",
    "resolve_map_model",
    "resolve_truth_trajectory",
]
