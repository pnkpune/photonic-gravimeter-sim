"""
feedback_policy.py

Observability-aware directional PF-to-INS feedback policy for the
gravity-aided navigation simulator.

This module replaces the naive "inject full 3D pseudo-position" approach
with a **directional feedback** strategy that respects the observability
geometry of scalar gravity map matching.

Why this file exists
--------------------
Scalar gravity provides a 1D measurement in 3D position space.  The PF
posterior is therefore ridge-shaped: well-constrained perpendicular to
the local gravity gradient and poorly constrained along it.  Injecting
the full posterior mean as a 3D pseudo-position measurement treats a
ridge as a point estimate, feeding confident-but-wrong information along
the poorly-observed direction into the INS.  This destroys the filter.

The fix is to:
1) Eigendecompose the PF posterior NED covariance.
2) Identify the well-constrained direction(s) (smallest eigenvalues).
3) Inject a rank-1 (or rank-k) pseudo-measurement only along those
   directions, leaving the INS free along the ridge.
4) Gate the injection on quantitative observability indicators.

This is mathematically equivalent to treating the PF as a bearing-only
sensor that says "you are somewhere along this line" rather than "you
are at this point."

Conventions
-----------
- Navigation frame is NED: x = North, y = East, z = Down
- PF posterior covariance is in local NED [m^2]
- Error-state vector is 15-state as in error_state_ins.py
- Feedback measurement goes through the standard LinearMeasurement /
  apply_linear_measurement path in fusion.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..analysis.observability import ObservabilitySnapshot
from ..ml.feedback_trust import (
    SequenceFeedbackTrustModel,
    extract_sequence_feedback_features,
    load_sequence_feedback_trust_model,
)
from .error_state_ins import (
    ERROR_STATE_SIZE,
    ERR_POS,
    ErrorStateINS,
    ErrorStateINSState,
)
from .gravity_sequence_match import SequenceAnchorEstimate, SequenceMatchUpdateResult
from .fusion import (
    FusionUpdateResult,
    LinearMeasurement,
    apply_linear_measurement,
    make_directional_position_measurement,
    make_horizontal_ned_position_measurement,
)
from .map_match_pf import (
    MapMatchPFUpdateResult,
    ParticleCloudEstimate,
    geodetic_offsets_to_local_ned,
)

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DirectionalFeedbackSpec:
    """
    Configuration for observability-aware directional PF feedback.

    Parameters
    ----------
    enabled : bool, default=True
        Master switch for directional feedback.
    min_eigenvalue_ratio : float, default=4.0
        Minimum ratio of largest to smallest eigenvalue of the PF NED
        covariance.  If the ratio is below this, the posterior is too
        isotropic (not ridge-shaped enough) to extract a reliable
        well-constrained direction.
    max_constrained_eigenvalue_m2 : float, default=1e6
        Maximum variance [m^2] along the well-constrained direction.
        If the smallest eigenvalue exceeds this, even the best direction
        is too uncertain to be useful.
    min_pf_ins_covariance_ratio : float, default=0.0
        The PF variance along the constrained direction must be smaller
        than this fraction of the INS variance in the same direction for
        feedback to be allowed.  Set to 0 to disable this check.
    min_ess_fraction : float, default=0.0
        Minimum ESS as a fraction of particle count.  If ESS is above
        this, it means particles weren't discriminated.  Counter-intuitive:
        low ESS (before resampling) means the measurement was informative.
        Set to 0 to disable (useful during initial development).
    max_ess_fraction : float, default=1.0
        Maximum ESS fraction — feedback is suppressed when ESS is above
        this.  Values near 1.0 mean particles are nearly uniform (no info).
    persistence_count : int, default=1
        Number of consecutive informative updates required before the
        first injection.
    max_correction_norm_m : float, default=500.0
        Maximum magnitude of the projected scalar correction [m].
        Corrections larger than this are clipped.
    base_inflation : float, default=2.0
        Baseline covariance inflation applied to the directional
        measurement variance.
    adaptive_inflation : bool, default=True
        Whether to scale inflation adaptively based on observability
        quality.
    num_directions : int, default=1
        How many eigenvector directions to inject (1 = rank-1, up to 3).
        Start with 1; increase only when multi-modal sensing provides
        richer observability.
    nis_threshold : float or None, default=None
        Optional NIS gate for the directional measurement update.
    horizontal_only : bool, default=True
        When True, the eigendecomposition is performed on the 2x2 horizontal
        (N, E) block of the PF NED covariance, and the constrained direction
        is forced into the horizontal plane with zero vertical component.
        This prevents the controller from selecting the vertical axis as
        best-constrained just because depth aiding has already made it tight,
        which is a degenerate failure mode of the naive 3D eigendecomposition.
        Vertical aiding should be handled independently by the depth sensor.
    require_observability_recommended : bool, default=False
        When True, require the online observability analyzer to recommend that
        feedback is appropriate at the current step.
    min_observable_rank : int or None, default=None
        Optional lower bound on the online observability rank.
    min_information_density : float or None, default=None
        Optional lower bound on the online observability information-density
        metric.
    min_observability_gradient_norm : float or None, default=None
        Optional lower bound on the horizontal gravity-gradient norm from the
        observability analyzer.
    """

    # Defaults are tuned as a SAFE NO-OP against the current maritime_baseline
    # PF posterior: gates are strict enough that feedback never fires on that
    # scenario, which by construction ties the observe-only baseline with 0%
    # HMI.  Relax these once a better measurement channel (e.g. gravity
    # gradient, Priority 3) supplies a properly calibrated posterior.
    enabled: bool = True
    min_eigenvalue_ratio: float = 8.0
    max_constrained_eigenvalue_m2: float = 1.0e6
    min_pf_ins_covariance_ratio: float = 0.0
    min_ess_fraction: float = 0.0
    max_ess_fraction: float = 0.95
    persistence_count: int = 3
    max_correction_norm_m: float = 2.0
    base_inflation: float = 15.0
    adaptive_inflation: bool = True
    num_directions: int = 1
    nis_threshold: Optional[float] = None
    horizontal_only: bool = True
    require_observability_recommended: bool = False
    min_observable_rank: Optional[int] = None
    min_information_density: Optional[float] = None
    min_observability_gradient_norm: Optional[float] = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.min_eigenvalue_ratio = float(self.min_eigenvalue_ratio)
        self.max_constrained_eigenvalue_m2 = float(self.max_constrained_eigenvalue_m2)
        self.min_pf_ins_covariance_ratio = float(self.min_pf_ins_covariance_ratio)
        self.min_ess_fraction = float(self.min_ess_fraction)
        self.max_ess_fraction = float(self.max_ess_fraction)
        self.persistence_count = max(1, int(self.persistence_count))
        self.max_correction_norm_m = float(self.max_correction_norm_m)
        self.base_inflation = float(self.base_inflation)
        self.adaptive_inflation = bool(self.adaptive_inflation)
        self.num_directions = max(1, min(3, int(self.num_directions)))
        self.horizontal_only = bool(self.horizontal_only)
        self.require_observability_recommended = bool(
            self.require_observability_recommended
        )
        if self.min_observable_rank is not None:
            self.min_observable_rank = int(self.min_observable_rank)
            if self.min_observable_rank < 1 or self.min_observable_rank > 3:
                raise ValueError("min_observable_rank must be in {1, 2, 3}.")
        if self.min_information_density is not None:
            self.min_information_density = float(self.min_information_density)
        if self.min_observability_gradient_norm is not None:
            self.min_observability_gradient_norm = float(
                self.min_observability_gradient_norm
            )
            if self.min_observability_gradient_norm < 0.0:
                raise ValueError(
                    "min_observability_gradient_norm must be nonnegative when provided."
                )


@dataclass
class DirectionalFeedbackDiagnostics:
    """
    Diagnostics from one directional-feedback evaluation.

    Attributes
    ----------
    eigenvalues_ned_m2 : np.ndarray, shape (3,)
        Eigenvalues of PF NED covariance, sorted ascending.
    eigenvectors_ned : np.ndarray, shape (3, 3)
        Corresponding eigenvectors as columns, sorted by eigenvalue.
    eigenvalue_ratio : float
        Ratio of largest to smallest eigenvalue.
    ess_fraction : float
        ESS as a fraction of particle count.
    constrained_direction_ned : np.ndarray, shape (3,)
        Unit vector along the best-constrained direction in NED.
    projected_correction_m : float
        Scalar correction projected along the constrained direction.
    projected_variance_m2 : float
        PF posterior variance along the constrained direction.
    ins_projected_variance_m2 : float
        INS covariance projected along the constrained direction.
    feedback_allowed : bool
        Whether all acceptance gates passed.
    rejection_reason : str or None
        Human-readable reason for rejection, if rejected.
    consecutive_informative : int
        Number of consecutive informative updates seen so far.
    inflation_applied : float
        Actual covariance inflation applied to the measurement.
    observable_rank : int or None
        Online observability rank if available.
    information_density : float or None
        Online information-density metric if available.
    gradient_norm_horizontal : float or None
        Horizontal gravity-gradient norm if available.
    observability_feedback_recommended : bool or None
        Recommendation flag from the observability analyzer if available.
    """

    eigenvalues_ned_m2: FloatArray
    eigenvectors_ned: FloatArray
    eigenvalue_ratio: float
    ess_fraction: float
    constrained_direction_ned: FloatArray
    projected_correction_m: float
    projected_variance_m2: float
    ins_projected_variance_m2: float
    feedback_allowed: bool
    rejection_reason: Optional[str]
    consecutive_informative: int
    inflation_applied: float
    observable_rank: Optional[int] = None
    information_density: Optional[float] = None
    gradient_norm_horizontal: Optional[float] = None
    observability_feedback_recommended: Optional[bool] = None


@dataclass
class DirectionalFeedbackResult:
    """
    Complete result of one directional-feedback attempt.

    Attributes
    ----------
    diagnostics : DirectionalFeedbackDiagnostics
        Observability analysis and gate evaluation.
    fusion_result : FusionUpdateResult or None
        The actual INS update result, if feedback was applied.
    """

    diagnostics: DirectionalFeedbackDiagnostics
    fusion_result: Optional[FusionUpdateResult] = None

    @property
    def applied(self) -> bool:
        """True if a measurement was actually injected into the INS."""
        return self.fusion_result is not None and self.fusion_result.accepted


@dataclass
class SequenceFeedbackSpec:
    """
    Configuration for delayed sequence-to-INS feedback.

    Two modes are supported:

    - ``bias_transfer``:
      Treat the delayed sequence estimate as a current-state horizontal bias
      estimate and inject it directly into the live INS with explicit
      delay-driven covariance inflation. This is the original experimental path
      and is retained mainly for benchmarking.

    - ``lag_replay``:
      Apply the delayed sequence estimate to the INS state at the delayed time,
      then replay the stored IMU and aiding measurements forward. This is the
      safer fixed-lag architecture for sequence outputs because the estimate is
      fused at the time it actually describes.

    Measurement geometry is configured independently:

    - ``full_horizontal``:
      Inject the full 2D horizontal posterior mean offset.

    - ``directional_horizontal``:
      Eigendecompose the horizontal 2x2 posterior covariance and inject only
      the best-constrained horizontal component. This is the richer delayed
      measurement model for sequence feedback.
    """

    enabled: bool = True
    mode: str = "lag_replay"
    measurement_geometry: str = "directional_horizontal"
    min_window_size: int = 5
    min_peak_probability: float = 0.12
    min_horizontal_eigenvalue_ratio: float = 1.0
    min_projected_std_m: float = 0.0
    max_horizontal_std_m: float = 80.0
    max_correction_norm_m: float = 150.0
    covariance_inflation: float = 3.0
    transfer_rw_std_mps: float = 0.6
    reset_matcher_after_apply: bool = True
    nis_threshold: Optional[float] = 25.0
    trust_model_export_path: Optional[str] = None
    trust_gate_source: str = "heuristic"
    min_trust_probability: float = 0.5
    max_predicted_error_delta_m: Optional[float] = None
    apply_trust_gain_alpha: bool = False
    trust_gain_alpha_min: float = 0.0
    trust_gain_alpha_max: float = 1.0
    fixed_gain_alpha_override: Optional[float] = None
    learned_gain_cooldown_s: float = 0.0
    learned_gain_max_applied_updates: Optional[int] = None
    apply_trust_covariance_scale: bool = False
    trust_covariance_scale_min: float = 0.75
    trust_covariance_scale_max: float = 2.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"bias_transfer", "lag_replay"}:
            raise ValueError("mode must be 'bias_transfer' or 'lag_replay'.")
        self.measurement_geometry = str(self.measurement_geometry).strip().lower()
        if self.measurement_geometry not in {
            "full_horizontal",
            "directional_horizontal",
        }:
            raise ValueError(
                "measurement_geometry must be 'full_horizontal' or "
                "'directional_horizontal'."
            )
        self.min_window_size = max(1, int(self.min_window_size))
        self.min_peak_probability = float(self.min_peak_probability)
        self.min_horizontal_eigenvalue_ratio = float(
            self.min_horizontal_eigenvalue_ratio
        )
        self.min_projected_std_m = float(self.min_projected_std_m)
        self.max_horizontal_std_m = float(self.max_horizontal_std_m)
        self.max_correction_norm_m = float(self.max_correction_norm_m)
        self.covariance_inflation = float(self.covariance_inflation)
        self.transfer_rw_std_mps = float(self.transfer_rw_std_mps)
        self.reset_matcher_after_apply = bool(self.reset_matcher_after_apply)
        self.trust_model_export_path = (
            None
            if self.trust_model_export_path is None
            else str(self.trust_model_export_path).strip()
        )
        self.trust_gate_source = str(self.trust_gate_source).strip().lower()
        if self.trust_gate_source not in {"heuristic", "trust_model", "both"}:
            raise ValueError(
                "trust_gate_source must be 'heuristic', 'trust_model', or 'both'."
            )
        self.min_trust_probability = float(self.min_trust_probability)
        self.max_predicted_error_delta_m = (
            None
            if self.max_predicted_error_delta_m is None
            else float(self.max_predicted_error_delta_m)
        )
        self.apply_trust_gain_alpha = bool(self.apply_trust_gain_alpha)
        self.trust_gain_alpha_min = float(self.trust_gain_alpha_min)
        self.trust_gain_alpha_max = float(self.trust_gain_alpha_max)
        self.fixed_gain_alpha_override = (
            None
            if self.fixed_gain_alpha_override is None
            else float(self.fixed_gain_alpha_override)
        )
        self.learned_gain_cooldown_s = float(self.learned_gain_cooldown_s)
        self.learned_gain_max_applied_updates = (
            None
            if self.learned_gain_max_applied_updates is None
            else int(self.learned_gain_max_applied_updates)
        )
        self.apply_trust_covariance_scale = bool(self.apply_trust_covariance_scale)
        self.trust_covariance_scale_min = float(self.trust_covariance_scale_min)
        self.trust_covariance_scale_max = float(self.trust_covariance_scale_max)
        if not (0.0 <= self.min_peak_probability <= 1.0):
            raise ValueError("min_peak_probability must lie in [0, 1].")
        if self.min_horizontal_eigenvalue_ratio < 1.0:
            raise ValueError("min_horizontal_eigenvalue_ratio must be >= 1.0.")
        if self.min_projected_std_m < 0.0:
            raise ValueError("min_projected_std_m must be nonnegative.")
        if self.max_horizontal_std_m <= 0.0:
            raise ValueError("max_horizontal_std_m must be positive.")
        if self.max_correction_norm_m <= 0.0:
            raise ValueError("max_correction_norm_m must be positive.")
        if self.covariance_inflation <= 0.0:
            raise ValueError("covariance_inflation must be positive.")
        if self.transfer_rw_std_mps < 0.0:
            raise ValueError("transfer_rw_std_mps must be nonnegative.")
        if self.nis_threshold is not None and float(self.nis_threshold) < 0.0:
            raise ValueError("nis_threshold must be nonnegative when provided.")
        if not (0.0 <= self.min_trust_probability <= 1.0):
            raise ValueError("min_trust_probability must lie in [0, 1].")
        if (
            self.max_predicted_error_delta_m is not None
            and not np.isfinite(self.max_predicted_error_delta_m)
        ):
            raise ValueError(
                "max_predicted_error_delta_m must be finite when provided."
            )
        if self.trust_gain_alpha_min < 0.0:
            raise ValueError("trust_gain_alpha_min must be nonnegative.")
        if self.trust_gain_alpha_max <= 0.0:
            raise ValueError("trust_gain_alpha_max must be positive.")
        if self.trust_gain_alpha_min > self.trust_gain_alpha_max:
            raise ValueError(
                "trust_gain_alpha_min must be <= trust_gain_alpha_max."
            )
        if (
            self.fixed_gain_alpha_override is not None
            and not np.isfinite(self.fixed_gain_alpha_override)
        ):
            raise ValueError("fixed_gain_alpha_override must be finite when provided.")
        if self.learned_gain_cooldown_s < 0.0:
            raise ValueError("learned_gain_cooldown_s must be nonnegative.")
        if (
            self.learned_gain_max_applied_updates is not None
            and self.learned_gain_max_applied_updates < 1
        ):
            raise ValueError(
                "learned_gain_max_applied_updates must be at least 1 when provided."
            )
        if self.trust_covariance_scale_min <= 0.0:
            raise ValueError("trust_covariance_scale_min must be positive.")
        if self.trust_covariance_scale_max <= 0.0:
            raise ValueError("trust_covariance_scale_max must be positive.")
        if self.trust_covariance_scale_min > self.trust_covariance_scale_max:
            raise ValueError(
                "trust_covariance_scale_min must be <= trust_covariance_scale_max."
            )
        if (
            self.trust_gate_source in {"trust_model", "both"}
            and not self.trust_model_export_path
        ):
            raise ValueError(
                "trust_model_export_path is required when trust_gate_source uses a trust model."
            )


@dataclass
class SequenceFeedbackDiagnostics:
    """
    Diagnostics from one delayed sequence-feedback evaluation.
    """

    mode: str
    measurement_geometry: str
    age_s: float
    window_size_used: int
    delayed_by_steps: int
    horizontal_offset_ned_m: FloatArray
    horizontal_std_m: FloatArray
    correction_norm_m: float
    projected_correction_m: float
    projected_std_m: float
    horizontal_eigenvalue_ratio: float
    constrained_direction_ned: FloatArray
    marginal_peak_probability: float
    posterior_entropy_nats: float
    covariance_inflation_applied: float
    transfer_std_m: float
    heuristic_allowed: bool
    heuristic_rejection_reason: Optional[str]
    trust_gate_source: str
    trust_probability: Optional[float]
    trust_allowed: Optional[bool]
    trust_rejection_reason: Optional[str]
    gain_alpha_applied: float
    trust_covariance_scale: float
    predicted_error_delta_m: Optional[float]
    runtime_budget_active: bool
    runtime_budget_allowed: bool
    runtime_budget_rejection_reason: Optional[str]
    applied_update_count_before: int
    cooldown_remaining_s: float
    feedback_allowed: bool
    rejection_reason: Optional[str]


@dataclass
class SequenceFeedbackResult:
    """
    Complete result of one delayed sequence-feedback attempt.
    """

    diagnostics: SequenceFeedbackDiagnostics
    fusion_result: Optional[FusionUpdateResult] = None

    @property
    def applied(self) -> bool:
        """True if a sequence-derived measurement was injected into the INS."""
        return self.fusion_result is not None and self.fusion_result.accepted


@dataclass
class SequenceLagSmootherSpec:
    """
    Configuration for bounded-lag sequence-driven navigation smoothing.

    This path is deliberately output-only in v1: it produces a delayed,
    higher-quality navigation track without mutating the live INS path.
    """

    enabled: bool = False
    output_lag_steps: Optional[int] = None
    measurement_geometry: str = "directional_horizontal"
    min_peak_probability: float = 0.05
    min_horizontal_eigenvalue_ratio: float = 1.15
    max_horizontal_std_m: float = 120.0
    max_correction_norm_m: float = 30.0
    covariance_inflation: float = 10.0
    max_anchor_count: int = 3
    publish_current_replayed_state: bool = False

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.output_lag_steps is not None:
            self.output_lag_steps = int(self.output_lag_steps)
            if self.output_lag_steps < 0:
                raise ValueError("output_lag_steps must be nonnegative when provided.")
        self.measurement_geometry = str(self.measurement_geometry).strip().lower()
        if self.measurement_geometry not in {
            "full_horizontal",
            "directional_horizontal",
        }:
            raise ValueError(
                "measurement_geometry must be 'full_horizontal' or "
                "'directional_horizontal'."
            )
        self.min_peak_probability = float(self.min_peak_probability)
        if not (0.0 <= self.min_peak_probability <= 1.0):
            raise ValueError("min_peak_probability must lie in [0, 1].")
        self.min_horizontal_eigenvalue_ratio = float(
            self.min_horizontal_eigenvalue_ratio
        )
        if self.min_horizontal_eigenvalue_ratio < 1.0:
            raise ValueError("min_horizontal_eigenvalue_ratio must be >= 1.0.")
        self.max_horizontal_std_m = float(self.max_horizontal_std_m)
        if self.max_horizontal_std_m <= 0.0:
            raise ValueError("max_horizontal_std_m must be positive.")
        self.max_correction_norm_m = float(self.max_correction_norm_m)
        if self.max_correction_norm_m <= 0.0:
            raise ValueError("max_correction_norm_m must be positive.")
        self.covariance_inflation = float(self.covariance_inflation)
        if self.covariance_inflation <= 0.0:
            raise ValueError("covariance_inflation must be positive.")
        self.max_anchor_count = int(self.max_anchor_count)
        if self.max_anchor_count < 1:
            raise ValueError("max_anchor_count must be at least 1.")
        self.publish_current_replayed_state = bool(self.publish_current_replayed_state)


@dataclass
class SequenceLagSmootherDiagnostics:
    """
    Diagnostics for one lag-smoother anchor evaluation.
    """

    measurement_geometry: str
    time_s: float
    global_index: int
    delayed_by_steps: int
    horizontal_offset_ned_m: FloatArray
    horizontal_std_m: FloatArray
    correction_norm_m: float
    projected_correction_m: float
    projected_std_m: float
    horizontal_eigenvalue_ratio: float
    constrained_direction_ned: FloatArray
    marginal_peak_probability: float
    posterior_entropy_nats: float
    covariance_inflation_applied: float
    feedback_allowed: bool
    rejection_reason: Optional[str]


@dataclass
class SequenceLagSmootherResult:
    """
    Result of applying one anchor inside the bounded-lag smoother.
    """

    diagnostics: SequenceLagSmootherDiagnostics
    fusion_result: Optional[FusionUpdateResult] = None

    @property
    def applied(self) -> bool:
        return self.fusion_result is not None and self.fusion_result.accepted


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _symmetrize(M: FloatArray) -> FloatArray:
    """Return the symmetric part of a square matrix."""
    return 0.5 * (M + M.T)


def eigendecompose_ned_covariance(
    P_ned: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Eigendecompose a 3x3 NED covariance matrix.

    Returns
    -------
    eigenvalues : np.ndarray, shape (3,)
        Eigenvalues sorted ascending (smallest first).
    eigenvectors : np.ndarray, shape (3, 3)
        Corresponding eigenvectors as columns.
    """
    P = _symmetrize(np.asarray(P_ned, dtype=np.float64))
    vals, vecs = np.linalg.eigh(P)
    # eigh already returns ascending order
    idx = np.argsort(vals)
    return vals[idx].astype(np.float64), vecs[:, idx].astype(np.float64)


def compute_adaptive_inflation(
    eigenvalue_ratio: float,
    ess_fraction: float,
    pf_ins_variance_ratio: float,
    base_inflation: float,
) -> float:
    """
    Compute adaptive covariance inflation.

    Higher inflation when:
    - eigenvalue ratio is low (posterior is isotropic = ambiguous)
    - ESS fraction is high (particles are uniform = no discrimination)
    - PF variance is large relative to INS variance

    Lower inflation when observability is clearly good.
    """
    # Penalty for low eigenvalue ratio (poor ridge structure)
    ratio_factor = max(1.0, 10.0 / max(eigenvalue_ratio, 1.0))

    # Penalty for high ESS (no particle discrimination)
    ess_factor = max(1.0, ess_fraction * 3.0)

    # Penalty for PF variance being close to or larger than INS variance
    cov_factor = max(1.0, pf_ins_variance_ratio * 2.0)

    return base_inflation * ratio_factor * ess_factor * cov_factor


def _state_from_filter_or_state(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> ErrorStateINSState:
    """Return ErrorStateINSState from filter or state."""
    if isinstance(ins_or_state, ErrorStateINS):
        return ins_or_state.state
    if isinstance(ins_or_state, ErrorStateINSState):
        return ins_or_state
    raise TypeError(
        f"Expected ErrorStateINS or ErrorStateINSState, got {type(ins_or_state).__name__}."
    )


class DirectionalFeedbackController:
    """
    Stateful controller for observability-aware directional PF feedback.

    Tracks consecutive informative updates and manages the persistence
    gate.  Call `evaluate(...)` at each PF update to get diagnostics and
    optionally inject a directional measurement.
    """

    def __init__(self, spec: DirectionalFeedbackSpec) -> None:
        self.spec = spec
        self._consecutive_informative: int = 0

    @property
    def consecutive_informative(self) -> int:
        """Number of consecutive informative PF updates."""
        return self._consecutive_informative

    def reset(self) -> None:
        """Reset the persistence counter."""
        self._consecutive_informative = 0

    def evaluate(
        self,
        pf_update: MapMatchPFUpdateResult,
        ins: ErrorStateINS,
        *,
        num_particles: int,
        time_s: Optional[float] = None,
        observability_snapshot: Optional[ObservabilitySnapshot] = None,
    ) -> DirectionalFeedbackResult:
        """
        Evaluate whether directional feedback should be injected and, if so,
        construct and apply the measurement.

        Parameters
        ----------
        pf_update : MapMatchPFUpdateResult
            Result of the most recent PF update.
        ins : ErrorStateINS
            Current INS filter (will be modified in-place if feedback is applied).
        num_particles : int
            Total particle count (for ESS normalization).
        time_s : float, optional
            Timestamp.
        observability_snapshot : ObservabilitySnapshot, optional
            Online observability analysis aligned to the current PF update.
            When the spec enables observability-based gates, the absence of
            this snapshot suppresses feedback.

        Returns
        -------
        DirectionalFeedbackResult
            Diagnostics and optional fusion result.
        """
        est = pf_update.estimate
        P_ned = est.covariance_ned_m2

        # --- Eigendecomposition ---
        if self.spec.horizontal_only:
            # Only the 2x2 horizontal (N, E) block of P_ned is used.
            # This forces the constrained direction into the horizontal plane.
            # Depth aiding already handles vertical; gravity-map matching is
            # there to constrain horizontal drift.
            P_h = _symmetrize(np.asarray(P_ned, dtype=np.float64))[:2, :2]
            w_h, V_h = np.linalg.eigh(P_h)
            order = np.argsort(w_h)
            w_h = np.maximum(w_h[order], 1.0e-10)
            V_h = V_h[:, order]
            # Embed into 3D with zero down component.
            # Use a large finite sentinel for the unused down eigenvalue so
            # downstream JSON serialization stays valid.
            eigenvalues = np.array([w_h[0], w_h[1], 1.0e18], dtype=np.float64)
            eigenvectors = np.zeros((3, 3), dtype=np.float64)
            eigenvectors[:2, 0] = V_h[:, 0]
            eigenvectors[:2, 1] = V_h[:, 1]
            eigenvectors[2, 2] = 1.0
            # eigenvalue_ratio uses the horizontal pair only; the infinite
            # down eigenvalue is ignored so the ratio reflects true
            # horizontal anisotropy.
            eigenvalue_ratio = float(w_h[1] / w_h[0])
        else:
            eigenvalues, eigenvectors = eigendecompose_ned_covariance(P_ned)
            eigenvalues = np.maximum(eigenvalues, 1.0e-10)
            eigenvalue_ratio = float(eigenvalues[2] / eigenvalues[0])

        ess_fraction = float(pf_update.effective_sample_size_after / max(num_particles, 1))

        # Best-constrained direction = eigenvector of smallest eigenvalue
        e_best = eigenvectors[:, 0].copy()

        # --- Compute projected correction in NED ---
        # Get offset from INS position to PF mean in local NED
        ins_state = _state_from_filter_or_state(ins)
        pf_offset_ned = geodetic_offsets_to_local_ned(
            np.array([est.lat_rad]),
            np.array([est.lon_rad]),
            np.array([est.height_m]),
            lat_ref_rad=ins_state.nominal.lat_rad,
            lon_ref_rad=ins_state.nominal.lon_rad,
            height_ref_m=ins_state.nominal.height_m,
        )[0]  # shape (3,)

        projected_correction = float(e_best @ pf_offset_ned)
        projected_variance = float(eigenvalues[0])

        # --- INS covariance in the constrained direction ---
        # Extract position sub-block from INS covariance (in geodetic coords)
        # We need position covariance in NED.  The INS stores covariance in
        # the error-state which has position as [d_lat, d_lon, d_h].
        # For a local approximation, convert geodetic position covariance
        # to NED using the same scale factors.
        from ..physics.earth import meridian_radius, prime_vertical_radius
        phi = ins_state.nominal.lat_rad
        h = ins_state.nominal.height_m
        M = float(meridian_radius(phi))
        N = float(prime_vertical_radius(phi))
        cos_phi = max(abs(float(np.cos(phi))), 1.0e-8)

        # J_ned_geo maps geodetic position errors to NED:
        #   dN = d_lat * (M + h)
        #   dE = d_lon * (N + h) * cos(phi)
        #   dD = -d_h
        J_ned_geo = np.diag([
            M + h,
            (N + h) * cos_phi,
            -1.0,
        ])

        P_ins_pos_geo = ins.covariance[ERR_POS, :][:, ERR_POS]
        P_ins_pos_ned = J_ned_geo @ P_ins_pos_geo @ J_ned_geo.T
        ins_projected_variance = float(e_best @ P_ins_pos_ned @ e_best)

        pf_ins_ratio = projected_variance / max(ins_projected_variance, 1.0e-10)

        observable_rank = None
        information_density = None
        gradient_norm_horizontal = None
        observability_feedback_recommended = None
        if observability_snapshot is not None:
            observable_rank = int(observability_snapshot.observable_rank)
            information_density = float(observability_snapshot.information_density)
            gradient_norm_horizontal = float(
                observability_snapshot.gradient_norm_horizontal
            )
            observability_feedback_recommended = bool(
                observability_snapshot.feedback_recommended
            )

        # --- Gate evaluation ---
        feedback_allowed = True
        rejection_reason: Optional[str] = None

        if not self.spec.enabled:
            feedback_allowed = False
            rejection_reason = "disabled"

        elif eigenvalue_ratio < self.spec.min_eigenvalue_ratio:
            feedback_allowed = False
            rejection_reason = (
                f"eigenvalue_ratio={eigenvalue_ratio:.1f} < "
                f"min={self.spec.min_eigenvalue_ratio:.1f}"
            )

        elif projected_variance > self.spec.max_constrained_eigenvalue_m2:
            feedback_allowed = False
            rejection_reason = (
                f"constrained_var={projected_variance:.1f} > "
                f"max={self.spec.max_constrained_eigenvalue_m2:.1f}"
            )

        elif ess_fraction > self.spec.max_ess_fraction:
            feedback_allowed = False
            rejection_reason = (
                f"ess_fraction={ess_fraction:.3f} > "
                f"max={self.spec.max_ess_fraction:.3f} (particles not discriminated)"
            )

        elif ess_fraction < self.spec.min_ess_fraction:
            feedback_allowed = False
            rejection_reason = (
                f"ess_fraction={ess_fraction:.3f} < "
                f"min={self.spec.min_ess_fraction:.3f}"
            )

        elif (
            self.spec.min_pf_ins_covariance_ratio > 0.0
            and pf_ins_ratio > self.spec.min_pf_ins_covariance_ratio
        ):
            feedback_allowed = False
            rejection_reason = (
                f"pf_ins_ratio={pf_ins_ratio:.3f} > "
                f"max={self.spec.min_pf_ins_covariance_ratio:.3f}"
            )

        elif (
            self.spec.require_observability_recommended
            and observability_snapshot is None
        ):
            feedback_allowed = False
            rejection_reason = "observability_missing"

        elif (
            self.spec.require_observability_recommended
            and not bool(observability_feedback_recommended)
        ):
            feedback_allowed = False
            rejection_reason = "observability_feedback_not_recommended"

        elif (
            self.spec.min_observable_rank is not None
            and observability_snapshot is None
        ):
            feedback_allowed = False
            rejection_reason = "observability_missing"

        elif (
            self.spec.min_observable_rank is not None
            and int(observable_rank) < self.spec.min_observable_rank
        ):
            feedback_allowed = False
            rejection_reason = (
                f"observable_rank={int(observable_rank)} < "
                f"min={self.spec.min_observable_rank}"
            )

        elif (
            self.spec.min_information_density is not None
            and observability_snapshot is None
        ):
            feedback_allowed = False
            rejection_reason = "observability_missing"

        elif (
            self.spec.min_information_density is not None
            and (
                information_density is None
                or not np.isfinite(information_density)
                or information_density < self.spec.min_information_density
            )
        ):
            density_str = "nan" if information_density is None else f"{information_density:.2f}"
            feedback_allowed = False
            rejection_reason = (
                f"information_density={density_str} < "
                f"min={self.spec.min_information_density:.2f}"
            )

        elif (
            self.spec.min_observability_gradient_norm is not None
            and observability_snapshot is None
        ):
            feedback_allowed = False
            rejection_reason = "observability_missing"

        elif (
            self.spec.min_observability_gradient_norm is not None
            and (
                gradient_norm_horizontal is None
                or gradient_norm_horizontal < self.spec.min_observability_gradient_norm
            )
        ):
            grad_str = "nan" if gradient_norm_horizontal is None else f"{gradient_norm_horizontal:.3e}"
            feedback_allowed = False
            rejection_reason = (
                f"gradient_norm={grad_str} < "
                f"min={self.spec.min_observability_gradient_norm:.3e}"
            )

        # Update persistence counter
        if feedback_allowed:
            self._consecutive_informative += 1
        else:
            self._consecutive_informative = 0

        # Persistence gate
        if (
            feedback_allowed
            and self._consecutive_informative < self.spec.persistence_count
        ):
            feedback_allowed = False
            rejection_reason = (
                f"persistence={self._consecutive_informative} < "
                f"required={self.spec.persistence_count}"
            )

        # --- Compute inflation ---
        if self.spec.adaptive_inflation:
            inflation = compute_adaptive_inflation(
                eigenvalue_ratio=eigenvalue_ratio,
                ess_fraction=ess_fraction,
                pf_ins_variance_ratio=pf_ins_ratio,
                base_inflation=self.spec.base_inflation,
            )
        else:
            inflation = self.spec.base_inflation

        # --- Build diagnostics ---
        diagnostics = DirectionalFeedbackDiagnostics(
            eigenvalues_ned_m2=eigenvalues.copy(),
            eigenvectors_ned=eigenvectors.copy(),
            eigenvalue_ratio=eigenvalue_ratio,
            ess_fraction=ess_fraction,
            constrained_direction_ned=e_best.copy(),
            projected_correction_m=projected_correction,
            projected_variance_m2=projected_variance,
            ins_projected_variance_m2=ins_projected_variance,
            feedback_allowed=feedback_allowed,
            rejection_reason=rejection_reason,
            consecutive_informative=self._consecutive_informative,
            inflation_applied=inflation,
            observable_rank=observable_rank,
            information_density=information_density,
            gradient_norm_horizontal=gradient_norm_horizontal,
            observability_feedback_recommended=observability_feedback_recommended,
        )

        if not feedback_allowed:
            return DirectionalFeedbackResult(
                diagnostics=diagnostics,
                fusion_result=None,
            )

        # --- Build and apply directional measurement ---
        # Clip correction magnitude
        clipped_correction = np.clip(
            projected_correction,
            -self.spec.max_correction_norm_m,
            self.spec.max_correction_norm_m,
        )

        # Build rank-1 (or rank-k) measurement through the existing
        # LinearMeasurement interface.
        #
        # For each constrained direction i:
        #   z_i = e_i^T @ (PF_mean_ned - INS_pos_ned)  [scalar projected offset]
        #   h_i = 0  (predicted is zero because the INS is the reference)
        #   H_i = e_i^T mapped to the position block of the 15-state error
        #   R_i = lambda_i * inflation  [PF variance along that direction, inflated]

        n_dirs = min(self.spec.num_directions, 3)
        z_list = []
        H_list = []
        R_diag = []

        for i in range(n_dirs):
            e_i = eigenvectors[:, i]
            correction_i = float(e_i @ pf_offset_ned)

            # Clip
            correction_i = np.clip(
                correction_i,
                -self.spec.max_correction_norm_m,
                self.spec.max_correction_norm_m,
            )

            # Scalar measurement
            z_list.append(correction_i)

            # Map NED direction to geodetic error-state position block
            # The error state position is [d_lat, d_lon, d_h].
            # NED offset = J_ned_geo @ geodetic_error
            # So geodetic_error = J_ned_geo^{-1} @ NED_offset
            # And H_geo = e_i^T @ J_ned_geo projected onto position block
            #
            # But the measurement is z = e_i^T @ NED_offset = e_i^T @ J_ned_geo @ delta_pos_geo
            # So H_row[ERR_POS] = e_i^T @ J_ned_geo
            H_row = np.zeros(ERROR_STATE_SIZE, dtype=np.float64)
            H_row[ERR_POS] = e_i @ J_ned_geo

            H_list.append(H_row)
            R_diag.append(float(eigenvalues[i]) * inflation)

        z_arr = np.array(z_list, dtype=np.float64)
        H_arr = np.array(H_list, dtype=np.float64)
        R_arr = np.diag(np.array(R_diag, dtype=np.float64))

        meas = LinearMeasurement(
            label="pf_directional",
            z=z_arr,
            h=np.zeros(n_dirs, dtype=np.float64),
            H=H_arr,
            R=R_arr,
            time_s=time_s,
        )

        fusion_result = apply_linear_measurement(
            ins,
            meas,
            nis_threshold=self.spec.nis_threshold,
        )

        return DirectionalFeedbackResult(
            diagnostics=diagnostics,
            fusion_result=fusion_result,
        )


class SequenceFeedbackController:
    """
    Conservative controller for delayed sequence-estimate feedback.

    In ``bias_transfer`` mode, the delayed sequence estimate is treated as a
    current-state horizontal bias estimate.

    In ``lag_replay`` mode, the caller is expected to pass the INS state at the
    delayed estimate time. The controller fuses the delayed horizontal offset at
    that time, and the caller is then responsible for replaying the segment
    forward to the present.
    """

    def __init__(self, spec: SequenceFeedbackSpec) -> None:
        self.spec = spec
        self._trust_model: Optional[SequenceFeedbackTrustModel] = None
        self._applied_update_count: int = 0
        self._last_applied_time_s: Optional[float] = None
        if self.spec.trust_model_export_path:
            self._trust_model = load_sequence_feedback_trust_model(
                self.spec.trust_model_export_path
            )

    def reset(self) -> None:
        self._applied_update_count = 0
        self._last_applied_time_s = None

    def _runtime_budget_active(self) -> bool:
        return bool(
            self.spec.apply_trust_gain_alpha
            and self.spec.fixed_gain_alpha_override is None
            and (
                self.spec.learned_gain_cooldown_s > 0.0
                or self.spec.learned_gain_max_applied_updates is not None
            )
        )

    def evaluate(
        self,
        sequence_update: SequenceMatchUpdateResult,
        ins: ErrorStateINS,
        *,
        current_time_s: float,
        live_ins_state: Optional[ErrorStateINS | ErrorStateINSState] = None,
        previous_live_ins_state: Optional[ErrorStateINS | ErrorStateINSState] = None,
    ) -> SequenceFeedbackResult:
        """
        Evaluate whether delayed sequence feedback should be applied and, if so,
        inject a conservative horizontal pseudo-position measurement.

        Notes
        -----
        The meaning of ``ins`` depends on ``spec.mode``:

        - ``bias_transfer``: ``ins`` is the live current INS state.
        - ``lag_replay``: ``ins`` is the delayed INS state aligned to
          ``sequence_update.time_s``.
        """
        age_s = max(0.0, float(current_time_s) - float(sequence_update.time_s))
        trust_live_ins_state = ins if live_ins_state is None else live_ins_state
        offset_h = np.asarray(
            sequence_update.posterior_mean_offset_ned_m[:2],
            dtype=np.float64,
        )
        P_h = _symmetrize(
            np.asarray(
                sequence_update.estimate.covariance_ned_m2[:2, :2],
                dtype=np.float64,
            )
        )
        eigvals_h, eigvecs_h = np.linalg.eigh(P_h)
        order = np.argsort(eigvals_h)
        eigvals_h = np.maximum(eigvals_h[order], 1.0e-12)
        eigvecs_h = eigvecs_h[:, order]
        horizontal_eigenvalue_ratio = float(eigvals_h[1] / eigvals_h[0])
        best_dir_h = eigvecs_h[:, 0]
        constrained_direction_ned = np.array(
            [best_dir_h[0], best_dir_h[1], 0.0],
            dtype=np.float64,
        )
        std_h = np.sqrt(np.maximum(np.diag(P_h), 0.0))
        correction_norm = float(np.linalg.norm(offset_h))
        projected_correction = float(best_dir_h @ offset_h)
        projected_std = float(np.sqrt(eigvals_h[0]))
        peak_prob = float(sequence_update.marginal_peak_probability)
        entropy = float(sequence_update.posterior_entropy_nats)
        if self.spec.mode == "bias_transfer":
            transfer_std_m = float(self.spec.transfer_rw_std_mps * age_s)
            measurement_time_s = float(current_time_s)
            measurement_label = "sequence_horizontal_bias"
        else:
            transfer_std_m = 0.0
            measurement_time_s = float(sequence_update.time_s)
            measurement_label = "sequence_horizontal_delayed"

        preconditions_allowed = True
        precondition_rejection_reason: Optional[str] = None
        heuristic_allowed = True
        heuristic_rejection_reason: Optional[str] = None

        if not self.spec.enabled:
            preconditions_allowed = False
            precondition_rejection_reason = "disabled"
        elif int(sequence_update.window_size_used) < self.spec.min_window_size:
            heuristic_allowed = False
            heuristic_rejection_reason = (
                f"window_size={int(sequence_update.window_size_used)} < "
                f"min={self.spec.min_window_size}"
            )
        elif not np.isfinite(peak_prob) or peak_prob < self.spec.min_peak_probability:
            heuristic_allowed = False
            heuristic_rejection_reason = (
                f"peak_probability={peak_prob:.3f} < "
                f"min={self.spec.min_peak_probability:.3f}"
            )
        elif self.spec.measurement_geometry == "directional_horizontal":
            if horizontal_eigenvalue_ratio < self.spec.min_horizontal_eigenvalue_ratio:
                heuristic_allowed = False
                heuristic_rejection_reason = (
                    f"horizontal_eigenvalue_ratio={horizontal_eigenvalue_ratio:.3f} < "
                    f"min={self.spec.min_horizontal_eigenvalue_ratio:.3f}"
                )
            elif (
                not np.isfinite(projected_std)
                or projected_std < self.spec.min_projected_std_m
            ):
                heuristic_allowed = False
                heuristic_rejection_reason = (
                    f"projected_std={projected_std:.3f} < "
                    f"min={self.spec.min_projected_std_m:.3f}"
                )
            elif not np.isfinite(projected_std) or projected_std > self.spec.max_horizontal_std_m:
                heuristic_allowed = False
                heuristic_rejection_reason = (
                    f"projected_std={projected_std:.3f} > "
                    f"max={self.spec.max_horizontal_std_m:.3f}"
                )
        elif np.any(~np.isfinite(std_h)) or float(np.max(std_h)) > self.spec.max_horizontal_std_m:
            heuristic_allowed = False
            heuristic_rejection_reason = (
                f"horizontal_std_max={float(np.max(std_h)):.3f} > "
                f"max={self.spec.max_horizontal_std_m:.3f}"
            )
        elif not np.isfinite(correction_norm) or correction_norm > self.spec.max_correction_norm_m:
            heuristic_allowed = False
            heuristic_rejection_reason = (
                f"correction_norm={correction_norm:.3f} > "
                f"max={self.spec.max_correction_norm_m:.3f}"
            )

        trust_probability: Optional[float] = None
        trust_allowed: Optional[bool] = None
        trust_rejection_reason: Optional[str] = None
        predicted_error_delta_m: Optional[float] = None
        gain_alpha_applied = (
            1.0
            if self.spec.fixed_gain_alpha_override is None
            else float(self.spec.fixed_gain_alpha_override)
        )
        trust_covariance_scale = 1.0
        if self.spec.trust_gate_source in {"trust_model", "both"}:
            if self._trust_model is None:
                trust_allowed = False
                trust_rejection_reason = "trust_model_missing"
            else:
                feature_vector, _ = extract_sequence_feedback_features(
                    sequence_update,
                    live_ins_state=trust_live_ins_state,
                    current_time_s=current_time_s,
                    previous_live_ins_state=previous_live_ins_state,
                )
                trust_probability = float(
                    self._trust_model.predict_trust_probability(
                        feature_vector,
                        mode=self.spec.mode,
                    )[0]
                )
                predicted_error_delta_m = float(
                    self._trust_model.predict_error_delta_m(
                        feature_vector,
                        mode=self.spec.mode,
                    )[0]
                )
                if self.spec.apply_trust_gain_alpha and self.spec.fixed_gain_alpha_override is None:
                    gain_alpha_applied = float(
                        self._trust_model.predict_gain_alpha(
                            feature_vector,
                            mode=self.spec.mode,
                            alpha_min=self.spec.trust_gain_alpha_min,
                            alpha_max=self.spec.trust_gain_alpha_max,
                        )[0]
                    )
                trust_covariance_scale = float(
                    self._trust_model.predict_covariance_scale(
                        feature_vector,
                        mode=self.spec.mode,
                        scale_min=self.spec.trust_covariance_scale_min,
                        scale_max=self.spec.trust_covariance_scale_max,
                    )[0]
                )
                trust_allowed = bool(
                    np.isfinite(trust_probability)
                    and trust_probability >= self.spec.min_trust_probability
                )
                if not trust_allowed:
                    trust_rejection_reason = (
                        f"trust_probability={trust_probability:.3f} < "
                        f"min={self.spec.min_trust_probability:.3f}"
                    )
                elif (
                    self.spec.max_predicted_error_delta_m is not None
                    and predicted_error_delta_m is not None
                ):
                    trust_allowed = bool(
                        np.isfinite(predicted_error_delta_m)
                        and predicted_error_delta_m
                        <= float(self.spec.max_predicted_error_delta_m)
                    )
                    if not trust_allowed:
                        trust_rejection_reason = (
                            f"predicted_error_delta_m={predicted_error_delta_m:.3f} > "
                            f"max={float(self.spec.max_predicted_error_delta_m):.3f}"
                        )

        gain_alpha_applied = float(
            np.clip(
                gain_alpha_applied,
                self.spec.trust_gain_alpha_min,
                self.spec.trust_gain_alpha_max,
            )
        )

        if self.spec.apply_trust_covariance_scale and not np.isfinite(trust_covariance_scale):
            trust_covariance_scale = 1.0
        if not np.isfinite(gain_alpha_applied):
            gain_alpha_applied = 1.0
        if gain_alpha_applied <= 1.0e-6:
            if self.spec.trust_gate_source == "heuristic":
                heuristic_allowed = False
                if heuristic_rejection_reason is None:
                    heuristic_rejection_reason = "gain_alpha_applied<=1e-6"
            else:
                trust_allowed = False
                if trust_rejection_reason is None:
                    trust_rejection_reason = "gain_alpha_applied<=1e-6"

        if not preconditions_allowed:
            feedback_allowed = False
            rejection_reason = precondition_rejection_reason
        elif self.spec.trust_gate_source == "heuristic":
            feedback_allowed = heuristic_allowed
            rejection_reason = heuristic_rejection_reason
        elif self.spec.trust_gate_source == "trust_model":
            feedback_allowed = bool(trust_allowed)
            rejection_reason = trust_rejection_reason
        else:
            feedback_allowed = heuristic_allowed and bool(trust_allowed)
            if not heuristic_allowed:
                rejection_reason = heuristic_rejection_reason
            elif not bool(trust_allowed):
                rejection_reason = trust_rejection_reason
            else:
                rejection_reason = None

        runtime_budget_active = self._runtime_budget_active()
        runtime_budget_allowed = True
        runtime_budget_rejection_reason: Optional[str] = None
        cooldown_remaining_s = 0.0
        applied_update_count_before = int(self._applied_update_count)
        if feedback_allowed and runtime_budget_active:
            if (
                self.spec.learned_gain_max_applied_updates is not None
                and self._applied_update_count
                >= int(self.spec.learned_gain_max_applied_updates)
            ):
                runtime_budget_allowed = False
                runtime_budget_rejection_reason = (
                    "learned_gain_max_applied_updates_reached"
                )
            elif (
                self.spec.learned_gain_cooldown_s > 0.0
                and self._last_applied_time_s is not None
            ):
                elapsed_s = max(0.0, float(current_time_s) - float(self._last_applied_time_s))
                cooldown_remaining_s = max(
                    0.0,
                    float(self.spec.learned_gain_cooldown_s) - elapsed_s,
                )
                if cooldown_remaining_s > 1.0e-9:
                    runtime_budget_allowed = False
                    runtime_budget_rejection_reason = (
                        "learned_gain_cooldown_active"
                    )
            if not runtime_budget_allowed:
                feedback_allowed = False
                rejection_reason = runtime_budget_rejection_reason

        diagnostics = SequenceFeedbackDiagnostics(
            mode=self.spec.mode,
            measurement_geometry=self.spec.measurement_geometry,
            age_s=age_s,
            window_size_used=int(sequence_update.window_size_used),
            delayed_by_steps=int(sequence_update.delayed_by_steps),
            horizontal_offset_ned_m=offset_h.copy(),
            horizontal_std_m=std_h.copy(),
            correction_norm_m=correction_norm,
            projected_correction_m=projected_correction,
            projected_std_m=projected_std,
            horizontal_eigenvalue_ratio=horizontal_eigenvalue_ratio,
            constrained_direction_ned=constrained_direction_ned.copy(),
            marginal_peak_probability=peak_prob,
            posterior_entropy_nats=entropy,
            covariance_inflation_applied=float(
                self.spec.covariance_inflation
                * (
                    trust_covariance_scale
                    if self.spec.apply_trust_covariance_scale
                    else 1.0
                )
            ),
            transfer_std_m=transfer_std_m,
            heuristic_allowed=bool(preconditions_allowed and heuristic_allowed),
            heuristic_rejection_reason=(
                precondition_rejection_reason
                if not preconditions_allowed
                else heuristic_rejection_reason
            ),
            trust_gate_source=self.spec.trust_gate_source,
            trust_probability=trust_probability,
            trust_allowed=trust_allowed,
            trust_rejection_reason=trust_rejection_reason,
            gain_alpha_applied=float(gain_alpha_applied),
            trust_covariance_scale=float(trust_covariance_scale),
            predicted_error_delta_m=predicted_error_delta_m,
            runtime_budget_active=bool(runtime_budget_active),
            runtime_budget_allowed=bool(runtime_budget_allowed),
            runtime_budget_rejection_reason=runtime_budget_rejection_reason,
            applied_update_count_before=int(applied_update_count_before),
            cooldown_remaining_s=float(cooldown_remaining_s),
            feedback_allowed=feedback_allowed,
            rejection_reason=rejection_reason,
        )

        if not feedback_allowed:
            return SequenceFeedbackResult(
                diagnostics=diagnostics,
                fusion_result=None,
            )

        if self.spec.measurement_geometry == "directional_horizontal":
            projected_variance = float(
                eigvals_h[0] * diagnostics.covariance_inflation_applied
            )
            if transfer_std_m > 0.0:
                projected_variance += transfer_std_m**2
            measurement = make_directional_position_measurement(
                ins,
                constrained_direction_ned,
                float(projected_correction * diagnostics.gain_alpha_applied),
                projected_variance,
                label=f"{measurement_label}_directional",
                time_s=measurement_time_s,
            )
        else:
            R_h = _symmetrize(P_h * float(diagnostics.covariance_inflation_applied))
            if transfer_std_m > 0.0:
                R_h += np.diag(np.full(2, transfer_std_m**2, dtype=np.float64))

            measurement = make_horizontal_ned_position_measurement(
                ins,
                offset_h * float(diagnostics.gain_alpha_applied),
                R_h,
                label=measurement_label,
                time_s=measurement_time_s,
            )
        fusion_result = apply_linear_measurement(
            ins,
            measurement,
            nis_threshold=self.spec.nis_threshold,
        )
        if fusion_result.accepted:
            self._applied_update_count += 1
            self._last_applied_time_s = float(current_time_s)
        return SequenceFeedbackResult(
            diagnostics=diagnostics,
            fusion_result=fusion_result,
        )


class SequenceLagSmootherController:
    """
    Conservative bounded-lag sequence-anchor fusion policy.

    The controller evaluates one anchor estimate at a time against the INS
    state aligned to that anchor's timestamp. It is intentionally separate from
    the live closed-loop feedback path.
    """

    def __init__(self, spec: SequenceLagSmootherSpec) -> None:
        self.spec = spec

    def evaluate_anchor(
        self,
        anchor: SequenceAnchorEstimate,
        ins: ErrorStateINS,
    ) -> SequenceLagSmootherResult:
        """
        Evaluate and optionally inject one delayed sequence anchor.
        """
        offset_h = np.asarray(
            anchor.posterior_mean_offset_ned_m[:2],
            dtype=np.float64,
        )
        P_h = _symmetrize(
            np.asarray(
                anchor.covariance_ned_m2[:2, :2],
                dtype=np.float64,
            )
        )
        eigvals_h, eigvecs_h = np.linalg.eigh(P_h)
        order = np.argsort(eigvals_h)
        eigvals_h = np.maximum(eigvals_h[order], 1.0e-12)
        eigvecs_h = eigvecs_h[:, order]
        best_dir_h = eigvecs_h[:, 0]
        constrained_direction_ned = np.array(
            [best_dir_h[0], best_dir_h[1], 0.0],
            dtype=np.float64,
        )
        horizontal_std = np.sqrt(np.maximum(np.diag(P_h), 0.0))
        horizontal_eigenvalue_ratio = float(eigvals_h[1] / eigvals_h[0])
        projected_correction = float(best_dir_h @ offset_h)
        projected_std = float(np.sqrt(eigvals_h[0]))
        correction_norm = float(np.linalg.norm(offset_h))
        peak_prob = float(anchor.marginal_peak_probability)
        entropy = float(anchor.posterior_entropy_nats)

        feedback_allowed = True
        rejection_reason: Optional[str] = None

        if not self.spec.enabled:
            feedback_allowed = False
            rejection_reason = "disabled"
        elif not np.isfinite(peak_prob) or peak_prob < self.spec.min_peak_probability:
            feedback_allowed = False
            rejection_reason = (
                f"peak_probability={peak_prob:.3f} < "
                f"min={self.spec.min_peak_probability:.3f}"
            )
        elif not np.isfinite(correction_norm) or correction_norm > self.spec.max_correction_norm_m:
            feedback_allowed = False
            rejection_reason = (
                f"correction_norm={correction_norm:.3f} > "
                f"max={self.spec.max_correction_norm_m:.3f}"
            )
        elif self.spec.measurement_geometry == "directional_horizontal":
            if horizontal_eigenvalue_ratio < self.spec.min_horizontal_eigenvalue_ratio:
                feedback_allowed = False
                rejection_reason = (
                    f"horizontal_eigenvalue_ratio={horizontal_eigenvalue_ratio:.3f} < "
                    f"min={self.spec.min_horizontal_eigenvalue_ratio:.3f}"
                )
            elif not np.isfinite(projected_std) or projected_std > self.spec.max_horizontal_std_m:
                feedback_allowed = False
                rejection_reason = (
                    f"projected_std={projected_std:.3f} > "
                    f"max={self.spec.max_horizontal_std_m:.3f}"
                )
        elif np.any(~np.isfinite(horizontal_std)) or float(np.max(horizontal_std)) > self.spec.max_horizontal_std_m:
            feedback_allowed = False
            rejection_reason = (
                f"horizontal_std_max={float(np.max(horizontal_std)):.3f} > "
                f"max={self.spec.max_horizontal_std_m:.3f}"
            )

        diagnostics = SequenceLagSmootherDiagnostics(
            measurement_geometry=self.spec.measurement_geometry,
            time_s=float(anchor.time_s),
            global_index=int(anchor.global_index),
            delayed_by_steps=int(anchor.delayed_by_steps),
            horizontal_offset_ned_m=offset_h.copy(),
            horizontal_std_m=horizontal_std.copy(),
            correction_norm_m=correction_norm,
            projected_correction_m=projected_correction,
            projected_std_m=projected_std,
            horizontal_eigenvalue_ratio=horizontal_eigenvalue_ratio,
            constrained_direction_ned=constrained_direction_ned.copy(),
            marginal_peak_probability=peak_prob,
            posterior_entropy_nats=entropy,
            covariance_inflation_applied=float(self.spec.covariance_inflation),
            feedback_allowed=feedback_allowed,
            rejection_reason=rejection_reason,
        )
        if not feedback_allowed:
            return SequenceLagSmootherResult(diagnostics=diagnostics, fusion_result=None)

        if self.spec.measurement_geometry == "directional_horizontal":
            projected_variance = float(
                eigvals_h[0] * self.spec.covariance_inflation
            )
            measurement = make_directional_position_measurement(
                ins,
                constrained_direction_ned,
                projected_correction,
                projected_variance,
                label="sequence_lag_anchor_directional",
                time_s=float(anchor.time_s),
            )
        else:
            R_h = _symmetrize(P_h * float(self.spec.covariance_inflation))
            measurement = make_horizontal_ned_position_measurement(
                ins,
                offset_h,
                R_h,
                label="sequence_lag_anchor_full_horizontal",
                time_s=float(anchor.time_s),
            )

        fusion_result = apply_linear_measurement(
            ins,
            measurement,
            nis_threshold=None,
        )
        return SequenceLagSmootherResult(
            diagnostics=diagnostics,
            fusion_result=fusion_result,
        )


def summarize_directional_feedback(result: DirectionalFeedbackResult) -> dict:
    """
    Produce a JSON-serializable summary of a directional feedback result.
    """
    d = result.diagnostics
    summary = {
        "feedback_allowed": d.feedback_allowed,
        "applied": result.applied,
        "rejection_reason": d.rejection_reason,
        "eigenvalue_ratio": d.eigenvalue_ratio,
        "eigenvalues_m2": d.eigenvalues_ned_m2.tolist(),
        "ess_fraction": d.ess_fraction,
        "projected_correction_m": d.projected_correction_m,
        "projected_variance_m2": d.projected_variance_m2,
        "ins_projected_variance_m2": d.ins_projected_variance_m2,
        "consecutive_informative": d.consecutive_informative,
        "inflation_applied": d.inflation_applied,
        "constrained_direction_ned": d.constrained_direction_ned.tolist(),
        "observable_rank": d.observable_rank,
        "information_density": d.information_density,
        "gradient_norm_horizontal": d.gradient_norm_horizontal,
        "observability_feedback_recommended": d.observability_feedback_recommended,
    }
    if result.fusion_result is not None:
        summary["nis"] = result.fusion_result.nis
        summary["gate_accepted"] = result.fusion_result.accepted
    return summary


def summarize_sequence_feedback(result: SequenceFeedbackResult) -> dict:
    """
    Produce a JSON-serializable summary of a sequence feedback result.
    """
    d = result.diagnostics
    summary = {
        "mode": d.mode,
        "measurement_geometry": d.measurement_geometry,
        "feedback_allowed": d.feedback_allowed,
        "applied": result.applied,
        "rejection_reason": d.rejection_reason,
        "heuristic_allowed": d.heuristic_allowed,
        "heuristic_rejection_reason": d.heuristic_rejection_reason,
        "trust_gate_source": d.trust_gate_source,
        "trust_probability": d.trust_probability,
        "trust_allowed": d.trust_allowed,
        "trust_rejection_reason": d.trust_rejection_reason,
        "gain_alpha_applied": d.gain_alpha_applied,
        "trust_covariance_scale": d.trust_covariance_scale,
        "predicted_error_delta_m": d.predicted_error_delta_m,
        "runtime_budget_active": d.runtime_budget_active,
        "runtime_budget_allowed": d.runtime_budget_allowed,
        "runtime_budget_rejection_reason": d.runtime_budget_rejection_reason,
        "applied_update_count_before": d.applied_update_count_before,
        "cooldown_remaining_s": d.cooldown_remaining_s,
        "age_s": d.age_s,
        "window_size_used": d.window_size_used,
        "delayed_by_steps": d.delayed_by_steps,
        "horizontal_offset_ned_m": d.horizontal_offset_ned_m.tolist(),
        "horizontal_std_m": d.horizontal_std_m.tolist(),
        "correction_norm_m": d.correction_norm_m,
        "projected_correction_m": d.projected_correction_m,
        "projected_std_m": d.projected_std_m,
        "horizontal_eigenvalue_ratio": d.horizontal_eigenvalue_ratio,
        "constrained_direction_ned": d.constrained_direction_ned.tolist(),
        "marginal_peak_probability": d.marginal_peak_probability,
        "posterior_entropy_nats": d.posterior_entropy_nats,
        "covariance_inflation_applied": d.covariance_inflation_applied,
        "transfer_std_m": d.transfer_std_m,
    }
    if result.fusion_result is not None:
        summary["nis"] = result.fusion_result.nis
        summary["gate_accepted"] = result.fusion_result.accepted
    return summary


def summarize_sequence_lag_smoother(result: SequenceLagSmootherResult) -> dict:
    """
    Produce a JSON-serializable summary of one lag-smoother anchor evaluation.
    """
    d = result.diagnostics
    summary = {
        "measurement_geometry": d.measurement_geometry,
        "feedback_allowed": d.feedback_allowed,
        "applied": result.applied,
        "rejection_reason": d.rejection_reason,
        "time_s": d.time_s,
        "global_index": d.global_index,
        "delayed_by_steps": d.delayed_by_steps,
        "horizontal_offset_ned_m": d.horizontal_offset_ned_m.tolist(),
        "horizontal_std_m": d.horizontal_std_m.tolist(),
        "correction_norm_m": d.correction_norm_m,
        "projected_correction_m": d.projected_correction_m,
        "projected_std_m": d.projected_std_m,
        "horizontal_eigenvalue_ratio": d.horizontal_eigenvalue_ratio,
        "constrained_direction_ned": d.constrained_direction_ned.tolist(),
        "marginal_peak_probability": d.marginal_peak_probability,
        "posterior_entropy_nats": d.posterior_entropy_nats,
        "covariance_inflation_applied": d.covariance_inflation_applied,
    }
    if result.fusion_result is not None:
        summary["nis"] = result.fusion_result.nis
        summary["gate_accepted"] = result.fusion_result.accepted
    return summary


__all__ = [
    "DirectionalFeedbackController",
    "DirectionalFeedbackDiagnostics",
    "DirectionalFeedbackResult",
    "DirectionalFeedbackSpec",
    "SequenceLagSmootherController",
    "SequenceLagSmootherDiagnostics",
    "SequenceLagSmootherResult",
    "SequenceLagSmootherSpec",
    "SequenceFeedbackController",
    "SequenceFeedbackDiagnostics",
    "SequenceFeedbackResult",
    "SequenceFeedbackSpec",
    "compute_adaptive_inflation",
    "eigendecompose_ned_covariance",
    "summarize_directional_feedback",
    "summarize_sequence_lag_smoother",
    "summarize_sequence_feedback",
]
