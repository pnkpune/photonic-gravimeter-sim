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

from ..analysis.observability import (
    ObservabilityAnalyzer,
    summarize_observability_snapshot,
)
from ..estimators.error_state_ins import (
    ERR_ATT,
    ERR_BA,
    ERR_BG,
    ERR_POS,
    ERR_VEL,
    ErrorStateINS,
    ErrorStateINSState,
    ErrorStatePropagationMatrices,
    ErrorStateINSProcessNoise,
)
from ..estimators.feedback_policy import (
    DirectionalFeedbackController,
    DirectionalFeedbackSpec,
    SequenceLagSmootherController,
    SequenceLagSmootherSpec,
    SequenceFeedbackController,
    SequenceFeedbackSpec,
    summarize_directional_feedback,
    summarize_sequence_lag_smoother,
    summarize_sequence_feedback,
)
from ..estimators.gravity_sequence_match import (
    GravitySequenceMatcher,
    GravitySequenceMatcherSpec,
    SequenceAnchorEstimate,
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
from ..estimators.integrity import integrity_snapshot_from_ins
from ..estimators.map_match_pf import (
    GravityMapParticleFilter,
    MapMatchPFSpec,
    evaluate_gravity_map_horizontal_gradient,
    geodetic_covariance_from_ned_covariance,
)
from ..physics.gravity_map import GravityGridMap
from ..sensors.depth import DepthMeasurement, DepthSensor, DepthSensorSpec
from ..sensors.gravimeter import (
    GravimeterMeasurement,
    GravimeterSpec,
    ScalarGravimeterSensor,
)
from ..sensors.gravity_gradiometer import (
    GravityGradiometerSensor,
    GravityGradiometerSpec,
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


def _resolve_sequence_lag_output_steps(
    smoother_spec: SequenceLagSmootherSpec,
    sequence_spec: GravitySequenceMatcherSpec,
    *,
    update_stride_steps: int = 1,
) -> int:
    """
    Resolve the configured lag-smoothed output delay in steps.
    """
    if smoother_spec.output_lag_steps is not None:
        return max(0, int(smoother_spec.output_lag_steps))
    return max(0, int(sequence_spec.window_size // 2) * max(1, int(update_stride_steps)))


def _sequence_anchor_max_horizontal_std_m(anchor: SequenceAnchorEstimate) -> float:
    """
    Conservative horizontal-std proxy for ranking overlapping anchors.
    """
    P_h = np.asarray(anchor.covariance_ned_m2[:2, :2], dtype=np.float64)
    diag = np.maximum(np.diag(0.5 * (P_h + P_h.T)), 0.0)
    return float(np.sqrt(np.max(diag)))


def _sequence_anchor_quality_key(anchor: SequenceAnchorEstimate) -> tuple[float, float, float]:
    """
    Sort key for choosing the strongest anchor per delayed step.

    Higher peak probability wins, then lower entropy, then lower horizontal
    standard deviation.
    """
    return (
        -float(anchor.marginal_peak_probability),
        float(anchor.posterior_entropy_nats),
        _sequence_anchor_max_horizontal_std_m(anchor),
    )


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
    matcher : {"pf", "sequence"}, default="pf"
        Which map-matching algorithm to run.
    schedule : PeriodicUpdateSchedule
        Triggering policy for map-matching gravity updates.
    pf_spec : MapMatchPFSpec
        PF algorithm specification, used only when ``matcher="pf"``.
    sequence_spec : GravitySequenceMatcherSpec
        Sequence-matching algorithm specification, used only when
        ``matcher="sequence"``.
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
        and fused back into the INS.  This is the **legacy** full-3D feedback mode.
    use_directional_feedback : bool, default=False
        Whether to use observability-aware directional feedback instead of
        full-3D position feedback.  When True, ``inject_position_to_ins`` is
        ignored and the ``directional_feedback_spec`` controls all feedback
        behavior.  This is the **recommended** feedback mode.
    directional_feedback_spec : DirectionalFeedbackSpec
        Configuration for directional feedback.  Only used when
        ``use_directional_feedback=True``.
    feedback_covariance_inflation : float, default=1.0
        Scalar inflation applied to PF covariance before INS feedback
        (legacy mode only).
    feedback_min_std_geodetic : scalar or shape (3,), default=(0, 0, 0)
        Lower bound on PF-derived geodetic standard deviations before feedback
        (legacy mode only).
    feedback_nis_threshold : float, optional
        Optional gate threshold for the PF pseudo-measurement update
        (legacy mode only).
    use_sequence_feedback : bool, default=False
        Whether to feed delayed sequence-matcher horizontal bias estimates into
        the live INS. Only valid when ``matcher="sequence"``.
    sequence_feedback_spec : SequenceFeedbackSpec
        Configuration for delayed sequence feedback.
    use_sequence_lag_smoother : bool, default=False
        Whether to publish a separate bounded-lag navigation track driven by
        sequence anchors. This path does not mutate the live INS.
    sequence_lag_smoother_spec : SequenceLagSmootherSpec
        Configuration for the bounded-lag sequence smoother.
    """

    enabled: bool = True
    matcher: str = "pf"
    schedule: PeriodicUpdateSchedule = field(default_factory=PeriodicUpdateSchedule)
    pf_spec: MapMatchPFSpec = field(default_factory=MapMatchPFSpec)
    sequence_spec: GravitySequenceMatcherSpec = field(
        default_factory=GravitySequenceMatcherSpec
    )
    gravity_meas_std_mps2: Optional[float] = None
    use_depth_measurement: bool = True
    use_last_depth_measurement: bool = True
    depth_meas_std_m: Optional[float] = None
    inject_position_to_ins: bool = False
    use_directional_feedback: bool = False
    use_sequence_feedback: bool = False
    use_sequence_lag_smoother: bool = False
    directional_feedback_spec: DirectionalFeedbackSpec = field(
        default_factory=DirectionalFeedbackSpec
    )
    sequence_feedback_spec: SequenceFeedbackSpec = field(
        default_factory=SequenceFeedbackSpec
    )
    sequence_lag_smoother_spec: SequenceLagSmootherSpec = field(
        default_factory=SequenceLagSmootherSpec
    )
    feedback_covariance_inflation: float = 1.0
    feedback_min_std_geodetic: ArrayLike | float = (0.0, 0.0, 0.0)
    feedback_nis_threshold: Optional[float] = None
    use_gradiometer: bool = False
    gradient_meas_std_per_s2: Optional[float] = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.matcher = str(self.matcher).strip().lower()
        if self.matcher not in {"pf", "sequence"}:
            raise ValueError("matcher must be 'pf' or 'sequence'.")
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
        self.use_gradiometer = bool(self.use_gradiometer)
        self.use_sequence_feedback = bool(self.use_sequence_feedback)
        self.use_sequence_lag_smoother = bool(self.use_sequence_lag_smoother)
        if self.gradient_meas_std_per_s2 is not None:
            self.gradient_meas_std_per_s2 = _positive_scalar(
                self.gradient_meas_std_per_s2,
                name="gradient_meas_std_per_s2",
            )
        if self.matcher != "pf":
            if self.inject_position_to_ins:
                raise ValueError(
                    "inject_position_to_ins is only supported with matcher='pf'."
                )
            if self.use_directional_feedback:
                raise ValueError(
                    "use_directional_feedback is only supported with matcher='pf'."
                )
        if self.matcher != "sequence" and self.use_sequence_feedback:
            raise ValueError(
                "use_sequence_feedback is only supported with matcher='sequence'."
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
class ObservabilityAnalysisConfig:
    """
    Configuration for online observability analysis during PF updates.

    Parameters
    ----------
    enabled : bool, default=True
        Enable observability analysis and logging.
    window_size : int, default=30
        Sliding-window length in PF updates.
    gravity_noise_std_mps2 : float, optional
        Gravity measurement standard deviation used in the Gramian. When
        omitted, the PF gravity measurement standard deviation is reused.
    eigenvalue_threshold : float, default=1e-20
        Small eigenvalues below this threshold are treated as zero.
    min_rank_for_feedback : int, default=2
        Minimum observable rank required for the snapshot to recommend
        closed-loop feedback.
    min_gradient_norm : float, default=1e-10
        Minimum horizontal gradient magnitude considered informative.
    delta_north_m : float, default=50
        Finite-difference step for the north gradient component [m].
    delta_east_m : float, default=50
        Finite-difference step for the east gradient component [m].
    delta_down_m : float, default=5
        Finite-difference step for the down gradient component [m].
    """

    enabled: bool = True
    window_size: int = 30
    gravity_noise_std_mps2: Optional[float] = None
    eigenvalue_threshold: float = 1.0e-20
    min_rank_for_feedback: int = 2
    min_gradient_norm: float = 1.0e-10
    delta_north_m: float = 50.0
    delta_east_m: float = 50.0
    delta_down_m: float = 5.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.window_size = int(self.window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.gravity_noise_std_mps2 is not None:
            self.gravity_noise_std_mps2 = _positive_scalar(
                self.gravity_noise_std_mps2,
                name="gravity_noise_std_mps2",
            )
        self.eigenvalue_threshold = _positive_scalar(
            self.eigenvalue_threshold,
            name="eigenvalue_threshold",
        )
        self.min_rank_for_feedback = int(self.min_rank_for_feedback)
        if self.min_rank_for_feedback < 1 or self.min_rank_for_feedback > 3:
            raise ValueError("min_rank_for_feedback must be in {1, 2, 3}.")
        self.min_gradient_norm = _positive_scalar(
            self.min_gradient_norm,
            name="min_gradient_norm",
        )
        self.delta_north_m = _positive_scalar(
            self.delta_north_m,
            name="delta_north_m",
        )
        self.delta_east_m = _positive_scalar(
            self.delta_east_m,
            name="delta_east_m",
        )
        self.delta_down_m = _positive_scalar(
            self.delta_down_m,
            name="delta_down_m",
        )


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
    observability : ObservabilityAnalysisConfig
        Online observability-analysis policy for PF update times.
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
    observability: ObservabilityAnalysisConfig = field(default_factory=ObservabilityAnalysisConfig)
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

    def _apply_depth_update(
        self,
        ins: ErrorStateINS,
        measurement: DepthMeasurement,
        *,
        depth_variance_m2: float,
        estimators: Optional[SimulationEstimatorLog] = None,
        stream_name: Optional[str] = None,
    ) -> Any:
        """
        Apply one configured depth-aiding update.
        """
        cfg = self.config
        if cfg.depth_aid.height_only_update:
            update = apply_depth_sensor_measurement_height_only(
                ins,
                measurement,
                depth_variance_m2=depth_variance_m2,
                reference_surface_height_m=cfg.reference_surface_height_m,
                nis_threshold=cfg.depth_aid.nis_threshold,
                label="depth_height_only",
            )
        else:
            update = apply_depth_sensor_measurement(
                ins,
                measurement,
                depth_variance_m2=depth_variance_m2,
                reference_surface_height_m=cfg.reference_surface_height_m,
                nis_threshold=cfg.depth_aid.nis_threshold,
                label="depth",
            )
        if estimators is not None and stream_name is not None:
            estimators.add_custom_sample(
                stream_name,
                summarize_update_result(update),
            )
        return update

    def _apply_velocity_update(
        self,
        ins: ErrorStateINS,
        measurement: VelocityAidMeasurement,
        *,
        velocity_R: FloatArray,
        estimators: Optional[SimulationEstimatorLog] = None,
        stream_name: Optional[str] = None,
    ) -> Any:
        """
        Apply one configured velocity-aiding update.
        """
        cfg = self.config
        if cfg.velocity_aid.velocity_only_update:
            update = apply_velocity_aid_measurement_velocity_only(
                ins,
                measurement,
                velocity_R,
                nis_threshold=cfg.velocity_aid.nis_threshold,
                label=f"velocity_{measurement.frame}_velocity_only",
            )
        else:
            update = apply_velocity_aid_measurement(
                ins,
                measurement,
                velocity_R,
                nis_threshold=cfg.velocity_aid.nis_threshold,
                label=f"velocity_{measurement.frame}",
            )
        if estimators is not None and stream_name is not None:
            estimators.add_custom_sample(
                stream_name,
                summarize_update_result(update),
            )
        return update

    def _replay_ins_segment(
        self,
        *,
        truth: TruthTrajectory,
        start_index: int,
        end_index: int,
        initial_state: Any,
        imu_samples: list[Any],
        depth_measurements_by_step: list[Optional[DepthMeasurement]],
        velocity_measurements_by_step: list[Optional[VelocityAidMeasurement]],
        depth_variance_m2: float,
        velocity_R: FloatArray,
        sequence_lag_smoother_ctrl: Optional[SequenceLagSmootherController] = None,
        anchors_by_step: Optional[dict[int, list[SequenceAnchorEstimate]]] = None,
        lag_smoother_log: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[ErrorStateINS, list[Any]]:
        """
        Replay one lag segment from a corrected historical INS state.

        The caller supplies the corrected INS state at ``start_index``. The
        helper then replays the stored IMU and aiding measurements through
        ``end_index`` inclusive and returns the replayed live filter plus the
        state history `[start_index, ..., end_index]`.
        """
        if start_index < 0 or end_index < start_index:
            raise ValueError(
                "Replay indices must satisfy 0 <= start_index <= end_index."
            )
        if end_index >= len(truth):
            raise IndexError(
                f"end_index {end_index} is out of bounds for truth length {len(truth)}."
            )
        if len(imu_samples) < end_index:
            raise ValueError(
                "imu_samples does not contain enough entries for the requested replay."
            )

        replay_ins = ErrorStateINS(initial_state.copy())

        if sequence_lag_smoother_ctrl is not None and anchors_by_step is not None:
            for anchor in anchors_by_step.get(start_index, []):
                lag_result = sequence_lag_smoother_ctrl.evaluate_anchor(anchor, replay_ins)
                if lag_smoother_log is not None:
                    row = summarize_sequence_lag_smoother(lag_result)
                    row["replay_step_index"] = int(start_index)
                    lag_smoother_log.append(row)

        replayed_states = [replay_ins.state.copy()]

        for step_idx in range(start_index + 1, end_index + 1):
            dt = float(truth.time_s[step_idx] - truth.time_s[step_idx - 1])
            imu_meas = imu_samples[step_idx - 1]

            replay_ins.predict(
                imu_meas.omega_ib_b_radps,
                imu_meas.f_ib_b_mps2,
                dt,
            )

            depth_meas = depth_measurements_by_step[step_idx]
            if depth_meas is not None and self.config.depth_aid.enabled:
                self._apply_depth_update(
                    replay_ins,
                    depth_meas,
                    depth_variance_m2=depth_variance_m2,
                )

            vel_meas = velocity_measurements_by_step[step_idx]
            if vel_meas is not None and self.config.velocity_aid.enabled:
                self._apply_velocity_update(
                    replay_ins,
                    vel_meas,
                    velocity_R=velocity_R,
                )

            if sequence_lag_smoother_ctrl is not None and anchors_by_step is not None:
                for anchor in anchors_by_step.get(step_idx, []):
                    lag_result = sequence_lag_smoother_ctrl.evaluate_anchor(anchor, replay_ins)
                    if lag_smoother_log is not None:
                        row = summarize_sequence_lag_smoother(lag_result)
                        row["replay_step_index"] = int(step_idx)
                        lag_smoother_log.append(row)

            replayed_states.append(replay_ins.state.copy())

        return replay_ins, replayed_states

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
        gradiometer_sensor: Optional[GravityGradiometerSensor] = None,
        pf_rng: Optional[np.random.Generator] = None,
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
        pf_rng : numpy.random.Generator, optional
            Optional RNG for the PF. Supply this when reproducible PF histories
            are required across repeated runs.
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
        sequence_matcher: Optional[GravitySequenceMatcher] = None
        sequence_feedback_ctrl: Optional[SequenceFeedbackController] = None
        sequence_lag_smoother_ctrl: Optional[SequenceLagSmootherController] = None
        directional_feedback_ctrl: Optional[DirectionalFeedbackController] = None
        observability: Optional[ObservabilityAnalyzer] = None
        resolved_map = resolve_map_model(map_model)
        if cfg.map_match.enabled and gravimeter_sensor is not None and resolved_map is not None:
            if cfg.map_match.matcher == "pf":
                pf = GravityMapParticleFilter(
                    cfg.map_match.pf_spec,
                    resolved_map,
                    rng=pf_rng,
                )
                pf.reset_from_ins(ins)
                if cfg.map_match.use_directional_feedback:
                    directional_feedback_ctrl = DirectionalFeedbackController(
                        cfg.map_match.directional_feedback_spec,
                    )
                if cfg.observability.enabled:
                    observability = ObservabilityAnalyzer(
                        resolved_map,
                        window_size=cfg.observability.window_size,
                        gravity_noise_std_mps2=(
                            (
                                pf.spec.gravity_meas_std_mps2
                                if cfg.map_match.gravity_meas_std_mps2 is None
                                else cfg.map_match.gravity_meas_std_mps2
                            )
                            if cfg.observability.gravity_noise_std_mps2 is None
                            else cfg.observability.gravity_noise_std_mps2
                        ),
                        eigenvalue_threshold=cfg.observability.eigenvalue_threshold,
                        min_rank_for_feedback=cfg.observability.min_rank_for_feedback,
                        min_gradient_norm=cfg.observability.min_gradient_norm,
                    )
            else:
                sequence_matcher = GravitySequenceMatcher(
                    cfg.map_match.sequence_spec,
                    resolved_map,
                )
                if cfg.map_match.use_sequence_feedback:
                    sequence_feedback_ctrl = SequenceFeedbackController(
                        cfg.map_match.sequence_feedback_spec,
                    )
                if cfg.map_match.use_sequence_lag_smoother:
                    cfg.map_match.sequence_lag_smoother_spec.enabled = True
                    sequence_lag_smoother_ctrl = SequenceLagSmootherController(
                        cfg.map_match.sequence_lag_smoother_spec,
                    )

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
        depth_measurements_by_step: list[Optional[DepthMeasurement]] = [None] * len(truth)
        velocity_measurements_by_step: list[Optional[VelocityAidMeasurement]] = [None] * len(truth)
        prediction_mats_by_step: list[Optional[ErrorStatePropagationMatrices]] = [None] * max(0, len(truth) - 1)
        sequence_anchor_candidates_by_step: dict[int, list[SequenceAnchorEstimate]] = {}
        sequence_updates_by_step: dict[int, Any] = {}
        truth_time_s = np.asarray(truth.time_s, dtype=np.float64)
        if truth_time_s.size >= 2:
            min_truth_dt_s = float(np.min(np.diff(truth_time_s)))
        else:
            min_truth_dt_s = 1.0
        time_alignment_tol_s = max(1.0e-9, 0.25 * min_truth_dt_s)
        if cfg.map_match.schedule.every_steps is not None:
            sequence_update_stride_steps = max(1, int(cfg.map_match.schedule.every_steps))
        elif cfg.map_match.schedule.period_s is not None:
            sequence_update_stride_steps = max(
                1,
                int(round(float(cfg.map_match.schedule.period_s) / min_truth_dt_s)),
            )
        else:
            sequence_update_stride_steps = 1
        sequence_window_span_steps = max(
            1,
            int(cfg.map_match.sequence_spec.window_size) * sequence_update_stride_steps,
        )
        lag_output_lag_steps = _resolve_sequence_lag_output_steps(
            cfg.map_match.sequence_lag_smoother_spec,
            cfg.map_match.sequence_spec,
            update_stride_steps=sequence_update_stride_steps,
        )
        lag_buffer_steps = max(
            lag_output_lag_steps,
            sequence_window_span_steps,
        ) + sequence_window_span_steps
        last_published_lag_smoothed_step = -1

        def _record_sequence_match_outputs(
            seq_updates: list[Any],
        ) -> None:
            if sequence_lag_smoother_ctrl is None:
                return

            def _truth_step_from_time(time_s: float) -> int:
                t = float(time_s)
                idx = int(np.searchsorted(truth_time_s, t, side="left"))
                candidate_indices: list[int] = []
                if idx < len(truth_time_s):
                    candidate_indices.append(idx)
                if idx > 0:
                    candidate_indices.append(idx - 1)
                if len(candidate_indices) == 0:
                    raise ValueError("Truth trajectory is empty.")
                best_idx = min(
                    candidate_indices,
                    key=lambda i: abs(float(truth_time_s[i]) - t),
                )
                if abs(float(truth_time_s[best_idx]) - t) > time_alignment_tol_s:
                    raise ValueError(
                        "Sequence output time cannot be aligned to the truth grid: "
                        f"time_s={t:.9f}, nearest_truth_time_s={float(truth_time_s[best_idx]):.9f}, "
                        f"tol_s={time_alignment_tol_s:.9f}."
                    )
                return int(best_idx)

            for update in seq_updates:
                update_step = _truth_step_from_time(float(update.time_s))
                sequence_updates_by_step[update_step] = update
                for anchor in update.anchor_estimates:
                    anchor_step = _truth_step_from_time(float(anchor.time_s))
                    sequence_anchor_candidates_by_step.setdefault(
                        anchor_step,
                        [],
                    ).append(anchor)

        def _select_lag_anchor_measurements(
            start_index: int,
            end_index: int,
        ) -> dict[int, list[SequenceAnchorEstimate]]:
            if sequence_lag_smoother_ctrl is None:
                return {}

            best_by_step: dict[int, SequenceAnchorEstimate] = {}
            for step_index, anchors in sequence_anchor_candidates_by_step.items():
                if step_index < start_index or step_index > end_index or len(anchors) == 0:
                    continue
                best_by_step[int(step_index)] = min(
                    anchors,
                    key=_sequence_anchor_quality_key,
                )

            if len(best_by_step) == 0:
                return {}

            priority_step = max(start_index, end_index - lag_output_lag_steps)
            selected_steps: list[int] = []
            if priority_step in best_by_step:
                selected_steps.append(int(priority_step))

            max_offset = max(priority_step - start_index, end_index - priority_step)
            for offset in range(1, max_offset + 1):
                if len(selected_steps) >= sequence_lag_smoother_ctrl.spec.max_anchor_count:
                    break
                left = priority_step - offset
                if left in best_by_step and left not in selected_steps:
                    selected_steps.append(int(left))
                    if len(selected_steps) >= sequence_lag_smoother_ctrl.spec.max_anchor_count:
                        break
                right = priority_step + offset
                if right in best_by_step and right not in selected_steps:
                    selected_steps.append(int(right))
                    if len(selected_steps) >= sequence_lag_smoother_ctrl.spec.max_anchor_count:
                        break

            if len(selected_steps) < sequence_lag_smoother_ctrl.spec.max_anchor_count:
                remaining = sorted(
                    (
                        step_index,
                        anchor,
                    )
                    for step_index, anchor in best_by_step.items()
                    if step_index not in selected_steps
                )
                remaining.sort(key=lambda item: _sequence_anchor_quality_key(item[1]))
                for step_index, _ in remaining:
                    selected_steps.append(int(step_index))
                    if len(selected_steps) >= sequence_lag_smoother_ctrl.spec.max_anchor_count:
                        break

            selected: dict[int, list[SequenceAnchorEstimate]] = {}
            for step_index in sorted(selected_steps):
                anchor = best_by_step[step_index]
                selected.setdefault(int(step_index), []).append(anchor)
            return selected

        def _publish_lag_smoothed_states(
            current_step: int,
            *,
            final_flush: bool = False,
        ) -> None:
            nonlocal last_published_lag_smoothed_step

            if (
                sequence_lag_smoother_ctrl is None
                or len(estimators.ins_states) == 0
                or len(sequence_anchor_candidates_by_step) == 0
            ):
                return

            end_index = int(current_step)
            start_index = max(0, end_index - lag_buffer_steps)
            anchors_by_step = _select_lag_anchor_measurements(start_index, end_index)
            lag_log_rows: list[dict[str, Any]] = []

            replay_ins, replayed_states = self._replay_ins_segment(
                truth=truth,
                start_index=start_index,
                end_index=end_index,
                initial_state=estimators.ins_states[start_index],
                imu_samples=sensors.imu_samples,
                depth_measurements_by_step=depth_measurements_by_step,
                velocity_measurements_by_step=velocity_measurements_by_step,
                depth_variance_m2=depth_variance_m2,
                velocity_R=velocity_R,
                sequence_lag_smoother_ctrl=sequence_lag_smoother_ctrl,
                anchors_by_step=anchors_by_step,
                lag_smoother_log=lag_log_rows,
            )

            for row in lag_log_rows:
                estimators.add_custom_sample(
                    "sequence_lag_smoother_anchors",
                    row,
                )

            if sequence_lag_smoother_ctrl.spec.publish_current_replayed_state:
                estimators.add_custom_sample(
                    "sequence_lag_smoother_preview",
                    {
                        "time_s": float(truth.time_s[end_index]),
                        "step_index": int(end_index),
                        "start_index": int(start_index),
                        "end_index": int(end_index),
                        "num_anchor_steps": int(len(anchors_by_step)),
                        "lat_rad": float(replay_ins.state.nominal.lat_rad),
                        "lon_rad": float(replay_ins.state.nominal.lon_rad),
                        "height_m": float(replay_ins.state.nominal.height_m),
                    },
                )

            publish_upto = end_index if final_flush else max(
                -1,
                end_index - lag_output_lag_steps,
            )
            publish_start = max(last_published_lag_smoothed_step + 1, start_index)

            for step_index in range(publish_start, publish_upto + 1):
                local_index = step_index - start_index
                if local_index < 0 or local_index >= len(replayed_states):
                    continue
                replayed_state = replayed_states[local_index].copy()
                published_state = replayed_state.copy()

                publish_anchor_applied = False
                publish_source = "replay"
                publish_anchor = None
                direct_update = sequence_updates_by_step.get(step_index)
                if direct_update is not None:
                    P_pos = np.asarray(
                        direct_update.estimate.covariance_geodetic,
                        dtype=np.float64,
                    )
                    P_pos = 0.5 * (P_pos + P_pos.T)
                    P_pos += np.diag(np.full(3, 1.0e-12, dtype=np.float64))
                    published_state.nominal.lat_rad = float(direct_update.estimate.lat_rad)
                    published_state.nominal.lon_rad = float(direct_update.estimate.lon_rad)
                    published_state.nominal.height_m = float(direct_update.estimate.height_m)
                    published_state.P[ERR_POS, :] = 0.0
                    published_state.P[:, ERR_POS] = 0.0
                    published_state.P[ERR_POS, ERR_POS] = P_pos
                    publish_anchor_applied = True
                    publish_source = "sequence_update"
                else:
                    anchor_candidates = sequence_anchor_candidates_by_step.get(step_index, [])
                    if len(anchor_candidates) > 0:
                        publish_anchor = min(
                            anchor_candidates,
                            key=_sequence_anchor_quality_key,
                        )
                        publish_anchor_std = _sequence_anchor_max_horizontal_std_m(
                            publish_anchor
                        )
                        if (
                            float(publish_anchor.marginal_peak_probability)
                            >= sequence_lag_smoother_ctrl.spec.min_peak_probability
                            and publish_anchor_std
                            <= sequence_lag_smoother_ctrl.spec.max_horizontal_std_m
                        ):
                            P_pos = np.asarray(
                                publish_anchor.covariance_geodetic,
                                dtype=np.float64,
                            )
                            P_pos = 0.5 * (P_pos + P_pos.T)
                            P_pos += np.diag(np.full(3, 1.0e-12, dtype=np.float64))
                            published_state.nominal.lat_rad = float(publish_anchor.lat_rad)
                            published_state.nominal.lon_rad = float(publish_anchor.lon_rad)
                            published_state.nominal.height_m = float(publish_anchor.height_m)
                            published_state.P[ERR_POS, :] = 0.0
                            published_state.P[:, ERR_POS] = 0.0
                            published_state.P[ERR_POS, ERR_POS] = P_pos
                            publish_anchor_applied = True
                            publish_source = "sequence_anchor"

                estimators.append_lag_smoothed_state(published_state)

                if integrity is not None:
                    snap = integrity_snapshot_from_ins(
                        published_state,
                        true_lat_rad=float(truth.lat_rad[step_index]),
                        true_lon_rad=float(truth.lon_rad[step_index]),
                        true_height_m=float(truth.height_m[step_index]),
                        horizontal_alert_limit_m=integrity.horizontal_alert_limit_m,
                        vertical_alert_limit_m=integrity.vertical_alert_limit_m,
                        horizontal_k_sigma=integrity.horizontal_k_sigma,
                        vertical_k_sigma=integrity.vertical_k_sigma,
                        radial_k_sigma=integrity.radial_k_sigma,
                        consistency_confidence=integrity.consistency_confidence,
                        time_s=float(truth.time_s[step_index]),
                    )
                    estimators.lag_smoothed_integrity_snapshots.append(snap)

                applied_here = [
                    row
                    for row in lag_log_rows
                    if int(row["replay_step_index"]) == int(step_index) and bool(row.get("applied"))
                ]
                estimators.add_custom_sample(
                    "sequence_lag_smoothed_publish",
                    {
                        "step_index": int(step_index),
                        "time_s": float(truth.time_s[step_index]),
                        "start_index": int(start_index),
                        "end_index": int(end_index),
                        "num_anchor_steps_considered": int(len(anchors_by_step)),
                        "num_anchor_updates_applied_here": int(len(applied_here)),
                        "publish_anchor_applied": bool(publish_anchor_applied),
                        "publish_source": publish_source,
                        "publish_anchor_peak_probability": (
                            None
                            if publish_anchor is None
                            else float(publish_anchor.marginal_peak_probability)
                        ),
                    },
                )
                last_published_lag_smoothed_step = int(step_index)

            stale_keys = [
                step_index
                for step_index in sequence_anchor_candidates_by_step
                if step_index <= last_published_lag_smoothed_step
            ]
            for step_index in stale_keys:
                sequence_anchor_candidates_by_step.pop(step_index, None)
                sequence_updates_by_step.pop(step_index, None)

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

            prediction_mats_by_step[k - 1] = ins.predict(
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
                    depth_measurements_by_step[k] = current_depth_measurement
                    sensors.depth_samples.append(current_depth_measurement)

                    last_depth_sample_time_s = t_now
                    last_depth_measurement = current_depth_measurement

                    self._apply_depth_update(
                        ins,
                        current_depth_measurement,
                        depth_variance_m2=depth_variance_m2,
                        estimators=estimators,
                        stream_name="depth_updates",
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

                    velocity_measurements_by_step[k] = vel_meas
                    sensors.velocity_aid_samples.append(vel_meas)
                    last_velocity_sample_time_s = t_now

                    self._apply_velocity_update(
                        ins,
                        vel_meas,
                        velocity_R=velocity_R,
                        estimators=estimators,
                        stream_name="velocity_updates",
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
            # Gravity gradiometer sampling
            # ----------------------------------------------------------
            gradiometer_meas = None
            if (
                gradiometer_sensor is not None
                and resolved_map is not None
                and cfg.map_match.use_gradiometer
            ):
                # Truth horizontal gradient at the true position (scalar inputs
                # broadcast to length-1 arrays in the helper).
                grad_truth = evaluate_gravity_map_horizontal_gradient(
                    resolved_map,
                    np.atleast_1d(np.float64(truth.lat_rad[k])),
                    np.atleast_1d(np.float64(truth.lon_rad[k])),
                    np.atleast_1d(np.float64(truth.height_m[k])),
                ).reshape(2)
                gradiometer_meas = gradiometer_sensor.measure(
                    ideal_gradient_per_s2=grad_truth,
                    dt_s=dt,
                    time_s=t_now,
                )
                sensors.add_custom_sample("gradiometer", gradiometer_meas)

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
                        measured_gradient_per_s2=(
                            None
                            if gradiometer_meas is None
                            else np.asarray(gradiometer_meas.value_per_s2, dtype=np.float64)
                        ),
                        gradient_meas_std_per_s2=cfg.map_match.gradient_meas_std_per_s2,
                        reference_surface_height_m=cfg.reference_surface_height_m,
                    )
                    estimators.pf_updates.append(pf_update)

                    obs_snapshot = None
                    if observability is not None:
                        obs_snapshot = observability.update(
                            lat_rad=float(ins.state.nominal.lat_rad),
                            lon_rad=float(ins.state.nominal.lon_rad),
                            height_m=float(ins.state.nominal.height_m),
                            time_s=t_now,
                            delta_north_m=cfg.observability.delta_north_m,
                            delta_east_m=cfg.observability.delta_east_m,
                            delta_down_m=cfg.observability.delta_down_m,
                        )
                        estimators.add_custom_sample(
                            "observability",
                            summarize_observability_snapshot(obs_snapshot),
                        )

                    # --- Directional feedback (new, recommended) ---
                    if directional_feedback_ctrl is not None:
                        df_result = directional_feedback_ctrl.evaluate(
                            pf_update,
                            ins,
                            num_particles=cfg.map_match.pf_spec.num_particles,
                            time_s=t_now,
                            observability_snapshot=obs_snapshot,
                        )
                        estimators.add_custom_sample(
                            "pf_directional_feedback",
                            summarize_directional_feedback(df_result),
                        )

                    # --- Legacy full-3D position feedback ---
                    elif cfg.map_match.inject_position_to_ins:
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

            elif sequence_matcher is not None and gravimeter_meas is not None:
                if cfg.map_match.schedule.should_trigger(k, t_now, t_prev):
                    depth_for_matcher: Optional[DepthMeasurement] = None
                    if cfg.map_match.use_depth_measurement:
                        if current_depth_measurement is not None:
                            depth_for_matcher = current_depth_measurement
                        elif cfg.map_match.use_last_depth_measurement:
                            depth_for_matcher = last_depth_measurement

                    seq_updates = sequence_matcher.update_from_gravimeter_measurement(
                        gravimeter_meas,
                        gravity_meas_std_mps2=cfg.map_match.gravity_meas_std_mps2,
                        ins_or_state=ins,
                        depth_measurement=depth_for_matcher,
                        measured_gradient_per_s2=(
                            None
                            if gradiometer_meas is None
                            else np.asarray(gradiometer_meas.value_per_s2, dtype=np.float64)
                        ),
                        gradient_meas_std_per_s2=cfg.map_match.gradient_meas_std_per_s2,
                        reference_surface_height_m=cfg.reference_surface_height_m,
                    )
                    if len(seq_updates) > 0:
                        estimators.sequence_updates.extend(seq_updates)
                        _record_sequence_match_outputs(seq_updates)
                        if sequence_feedback_ctrl is not None:
                            selected_update = seq_updates[-1]
                            seq_fb_summary: dict[str, Any]

                            if sequence_feedback_ctrl.spec.mode == "lag_replay":
                                target_step = max(
                                    0,
                                    k - int(selected_update.delayed_by_steps),
                                )
                                delayed_ins = ErrorStateINS(
                                    estimators.ins_states[target_step].copy()
                                )
                                seq_fb_result = sequence_feedback_ctrl.evaluate(
                                    selected_update,
                                    delayed_ins,
                                    current_time_s=t_now,
                                )
                                replayed_steps = 0
                                matcher_reset = False
                                if seq_fb_result.applied:
                                    replay_ins, replayed_states = self._replay_ins_segment(
                                        truth=truth,
                                        start_index=target_step,
                                        end_index=k,
                                        initial_state=delayed_ins.state,
                                        imu_samples=sensors.imu_samples,
                                        depth_measurements_by_step=depth_measurements_by_step,
                                        velocity_measurements_by_step=velocity_measurements_by_step,
                                        depth_variance_m2=depth_variance_m2,
                                        velocity_R=velocity_R,
                                    )

                                    historical_states = [s.copy() for s in replayed_states[:-1]]
                                    estimators.ins_states[target_step:] = historical_states
                                    ins.state = replay_ins.state.copy()
                                    replayed_steps = max(0, k - target_step)

                                    if integrity is not None:
                                        for state_index, replayed_state in enumerate(
                                            historical_states,
                                            start=target_step,
                                        ):
                                            replay_time_s = float(truth.time_s[state_index])
                                            snap = integrity_snapshot_from_ins(
                                                replayed_state,
                                                true_lat_rad=float(truth.lat_rad[state_index]),
                                                true_lon_rad=float(truth.lon_rad[state_index]),
                                                true_height_m=float(truth.height_m[state_index]),
                                                horizontal_alert_limit_m=integrity.horizontal_alert_limit_m,
                                                vertical_alert_limit_m=integrity.vertical_alert_limit_m,
                                                horizontal_k_sigma=integrity.horizontal_k_sigma,
                                                vertical_k_sigma=integrity.vertical_k_sigma,
                                                radial_k_sigma=integrity.radial_k_sigma,
                                                consistency_confidence=integrity.consistency_confidence,
                                                time_s=replay_time_s,
                                            )
                                            estimators.integrity_snapshots[state_index] = snap
                                            integrity.history.snapshots[state_index] = snap

                                    if sequence_feedback_ctrl.spec.reset_matcher_after_apply:
                                        sequence_matcher.reset()
                                        sequence_matcher.update_from_gravimeter_measurement(
                                            gravimeter_meas,
                                            gravity_meas_std_mps2=cfg.map_match.gravity_meas_std_mps2,
                                            ins_or_state=ins,
                                            depth_measurement=depth_for_matcher,
                                            measured_gradient_per_s2=(
                                                None
                                                if gradiometer_meas is None
                                                else np.asarray(
                                                    gradiometer_meas.value_per_s2,
                                                    dtype=np.float64,
                                                )
                                            ),
                                            gradient_meas_std_per_s2=cfg.map_match.gradient_meas_std_per_s2,
                                            reference_surface_height_m=cfg.reference_surface_height_m,
                                        )
                                        matcher_reset = True

                                seq_fb_summary = summarize_sequence_feedback(seq_fb_result)
                                seq_fb_summary["target_step_index"] = int(target_step)
                                seq_fb_summary["replayed_steps"] = int(replayed_steps)
                                seq_fb_summary["matcher_reset"] = bool(matcher_reset)
                            else:
                                seq_fb_result = sequence_feedback_ctrl.evaluate(
                                    selected_update,
                                    ins,
                                    current_time_s=t_now,
                                )
                                seq_fb_summary = summarize_sequence_feedback(seq_fb_result)
                                seq_fb_summary["target_step_index"] = None
                                seq_fb_summary["replayed_steps"] = 0
                                seq_fb_summary["matcher_reset"] = False

                            estimators.add_custom_sample(
                                "sequence_feedback",
                                seq_fb_summary,
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

            if sequence_lag_smoother_ctrl is not None and sequence_matcher is not None:
                _publish_lag_smoothed_states(k, final_flush=False)

        if sequence_matcher is not None:
            final_seq_updates = sequence_matcher.finalize()
            if len(final_seq_updates) > 0:
                estimators.sequence_updates.extend(final_seq_updates)
                _record_sequence_match_outputs(final_seq_updates)
            if sequence_lag_smoother_ctrl is not None:
                _publish_lag_smoothed_states(len(truth) - 1, final_flush=True)

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
        gradiometer_sensor: Optional[GravityGradiometerSensor] = None,
        pf_rng: Optional[np.random.Generator] = None,
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
            gradiometer_sensor=gradiometer_sensor,
            pf_rng=pf_rng,
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
        gradiometer_spec: Optional[GravityGradiometerSpec] = None,
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
        gradiometer_spec : GravityGradiometerSpec, optional
            Horizontal gravity-gradiometer spec.
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
        gradiometer_sensor = (
            None
            if gradiometer_spec is None
            else GravityGradiometerSensor(gradiometer_spec, rng=child_rng())
        )
        pf_rng = child_rng()

        meta = SimulationMetadata() if metadata is None else metadata.copy()
        meta.rng_state = {"seed": None if seed is None else int(seed)}

        return self.run_scenario(
            scenario_or_truth,
            imu_sensor=imu_sensor,
            gravimeter_sensor=gravimeter_sensor,
            depth_sensor=depth_sensor,
            velocity_aid_sensor=velocity_aid_sensor,
            gradiometer_sensor=gradiometer_sensor,
            pf_rng=pf_rng,
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
    "GravityGradiometerSpec",
    "GravitySequenceMatcherSpec",
    "build_initial_covariance_geodetic",
    "process_noise_from_imu_spec",
    "resolve_map_model",
    "resolve_truth_trajectory",
]
