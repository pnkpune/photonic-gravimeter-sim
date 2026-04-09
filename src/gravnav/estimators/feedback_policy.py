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

from .error_state_ins import (
    ERROR_STATE_SIZE,
    ERR_POS,
    ErrorStateINS,
    ErrorStateINSState,
)
from .fusion import (
    FusionUpdateResult,
    LinearMeasurement,
    apply_linear_measurement,
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
    """

    enabled: bool = True
    min_eigenvalue_ratio: float = 4.0
    max_constrained_eigenvalue_m2: float = 1.0e6
    min_pf_ins_covariance_ratio: float = 0.0
    min_ess_fraction: float = 0.0
    max_ess_fraction: float = 0.95
    persistence_count: int = 1
    max_correction_norm_m: float = 500.0
    base_inflation: float = 2.0
    adaptive_inflation: bool = True
    num_directions: int = 1
    nis_threshold: Optional[float] = None

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

        Returns
        -------
        DirectionalFeedbackResult
            Diagnostics and optional fusion result.
        """
        est = pf_update.estimate
        P_ned = est.covariance_ned_m2

        # --- Eigendecomposition ---
        eigenvalues, eigenvectors = eigendecompose_ned_covariance(P_ned)

        # Clamp eigenvalues to a small positive floor
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
    "compute_adaptive_inflation",
    "eigendecompose_ned_covariance",
    "summarize_directional_feedback",
]
