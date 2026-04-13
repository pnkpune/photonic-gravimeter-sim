"""
gravity_sequence_match.py

Sliding-window sequence-based gravity map matching using a discrete HMM/Viterbi
formulation.

Why this file exists
--------------------
The repository already contains a practical particle-filter gravity matcher.
That works for pointwise map matching, but Priority 4 in the roadmap is to
exploit trajectory history explicitly rather than treating each gravity sample
as an independent localization event.

This module adds a bounded first implementation of that idea:
- discretize a local candidate grid around the current INS position
- score each candidate with a gravity/gradient emission likelihood
- connect consecutive grids with a motion-continuity transition model
- run forward/backward and Viterbi over a sliding window
- emit delayed sequence estimates for the center of the window

Scope of this first implementation
----------------------------------
- 2D horizontal matching around the INS center in local NED
- fixed height per update, taken from INS or depth aiding
- delayed observe-only sequence estimates
- no direct closed-loop INS feedback yet

This is deliberate. The current repo still treats PF feedback as an open
research path, so the sequence matcher is introduced first as a benchmarkable
alternative map-matching estimator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.frames import wrap_angle_pi
from ..sensors.depth import DepthMeasurement
from ..sensors.gravimeter import GravimeterMeasurement
from .error_state_ins import ErrorStateINS, ErrorStateINSState
from .map_match_pf import (
    FloatArray,
    _as_float_array,
    _state_from_filter_or_state,
    apply_ned_offsets_to_geodetic,
    diagonal_gaussian_log_likelihood,
    evaluate_gravity_map_disturbance,
    evaluate_gravity_map_horizontal_gradient,
    geodetic_covariance_from_ned_covariance,
    geodetic_offsets_to_local_ned,
    gaussian_log_likelihood_scalar,
    weighted_covariance,
    weighted_geodetic_mean,
)


def _positive_scalar(x: float, *, name: str) -> float:
    """Validate and return a positive scalar."""
    value = float(x)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _axis2(x: ArrayLike | float, *, name: str) -> FloatArray:
    """Convert a scalar or length-2 vector into a float64 2-vector."""
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(2, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"{name} must be scalar or shape (2,), got {arr.shape}.")
    return arr.astype(np.float64)


def _logsumexp(x: FloatArray, axis: int) -> FloatArray:
    """
    Stable log-sum-exp reduction.

    The implementation is NumPy-only to preserve the repository's current
    dependency footprint.
    """
    xmax = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(
        xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True)),
        axis=axis,
    ).astype(np.float64)


@dataclass
class SequenceAmbiguityDiagnostics:
    """
    Compact posterior-ambiguity and grid-coverage diagnostics.
    """

    posterior_candidate_ess: float
    posterior_candidate_ess_fraction: float
    edge_mass_fraction: float
    support_radius_n_m: float
    support_radius_e_m: float
    horizontal_covariance_eigenvalue_ratio: float
    grid_saturated_north: bool
    grid_saturated_east: bool
    grid_saturated_any: bool
    gravity_predicted_spread_mps2: float
    gravity_information_ratio: float
    bathymetry_predicted_spread_m: Optional[float]
    bathymetry_information_ratio: Optional[float]
    dominant_failure_mode: str
    grid_mode: str
    grid_half_span_m: FloatArray
    grid_spacing_m: FloatArray


@dataclass
class GravitySequenceMatcherSpec:
    """
    Configuration for the sliding-window sequence matcher.

    Parameters
    ----------
    window_size : int, default=9
        Number of gravity updates in the HMM/Viterbi window.
    grid_half_span_m : scalar or shape (2,), default=(120, 120)
        Half-span of the candidate grid about the INS center in north/east [m].
    grid_spacing_m : scalar or shape (2,), default=(20, 20)
        Candidate-grid spacing in north/east [m].
    transition_std_m : scalar or shape (2,), default=(20, 20)
        Transition standard deviation for the candidate offset change between
        consecutive windows [m]. Small values enforce stronger trajectory
        continuity relative to the INS centerline.
    center_prior_std_m : scalar or shape (2,), default=(80, 80)
        Soft per-step prior standard deviation on the candidate offset from the
        INS center [m].
    gravity_meas_std_mps2 : float, default=1e-5
        Scalar gravity measurement standard deviation [m/s^2].
    gradient_meas_std_per_s2 : float or None, default=None
        Optional horizontal-gravity-gradient measurement standard deviation.
    bathymetry_meas_std_m : float or None, default=None
        Optional seabed-clearance / water-depth measurement standard deviation.
    bathymetry_weight : float, default=1.0
        Relative weight applied to the bathymetry log-likelihood term.
    height_std_m : float, default=2
        Vertical covariance floor used when packaging delayed estimates.
    adaptive_grid_enabled : bool, default=False
        Enable bounded two-level grid adaptation based on ambiguity/coverage.
    expanded_grid_half_span_m : scalar or shape (2,), optional
        Recovery-grid half-span. Defaults to 2x the nominal span.
    expanded_grid_spacing_m : scalar or shape (2,), optional
        Recovery-grid spacing. Defaults to a scale preserving nominal grid count.
    name : str, default="gravity_sequence_match"
        Human-readable identifier.
    """

    window_size: int = 9
    grid_half_span_m: ArrayLike | float = (120.0, 120.0)
    grid_spacing_m: ArrayLike | float = (20.0, 20.0)
    transition_std_m: ArrayLike | float = (20.0, 20.0)
    center_prior_std_m: ArrayLike | float = (80.0, 80.0)
    gravity_meas_std_mps2: float = 1.0e-5
    gradient_meas_std_per_s2: Optional[float] = None
    bathymetry_meas_std_m: Optional[float] = None
    bathymetry_weight: float = 1.0
    height_std_m: float = 2.0
    adaptive_grid_enabled: bool = False
    expanded_grid_half_span_m: Optional[ArrayLike | float] = None
    expanded_grid_spacing_m: Optional[ArrayLike | float] = None
    adaptive_expand_edge_mass_fraction: float = 0.20
    adaptive_expand_support_radius_fraction: float = 0.85
    adaptive_contract_edge_mass_fraction: float = 0.05
    adaptive_contract_support_radius_fraction: float = 0.55
    ambiguity_support_threshold_peak_fraction: float = 0.05
    ambiguity_edge_mass_fraction: float = 0.20
    ambiguity_support_radius_fraction: float = 0.85
    ambiguity_min_posterior_ess_fraction: float = 0.03
    ambiguity_max_peak_probability: float = 0.85
    ambiguity_min_gravity_information_ratio: float = 0.50
    ambiguity_min_bathymetry_information_ratio: float = 0.50
    name: str = "gravity_sequence_match"

    def __post_init__(self) -> None:
        self.window_size = int(self.window_size)
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")

        self.grid_half_span_m = _axis2(self.grid_half_span_m, name="grid_half_span_m")
        self.grid_spacing_m = _axis2(self.grid_spacing_m, name="grid_spacing_m")
        self.transition_std_m = _axis2(self.transition_std_m, name="transition_std_m")
        self.center_prior_std_m = _axis2(
            self.center_prior_std_m,
            name="center_prior_std_m",
        )

        if np.any(self.grid_half_span_m <= 0.0):
            raise ValueError("grid_half_span_m must be positive.")
        if np.any(self.grid_spacing_m <= 0.0):
            raise ValueError("grid_spacing_m must be positive.")
        if np.any(self.transition_std_m <= 0.0):
            raise ValueError("transition_std_m must be positive.")
        if np.any(self.center_prior_std_m <= 0.0):
            raise ValueError("center_prior_std_m must be positive.")

        self.gravity_meas_std_mps2 = _positive_scalar(
            self.gravity_meas_std_mps2,
            name="gravity_meas_std_mps2",
        )
        if self.gradient_meas_std_per_s2 is not None:
            self.gradient_meas_std_per_s2 = _positive_scalar(
                self.gradient_meas_std_per_s2,
                name="gradient_meas_std_per_s2",
            )
        if self.bathymetry_meas_std_m is not None:
            self.bathymetry_meas_std_m = _positive_scalar(
                self.bathymetry_meas_std_m,
                name="bathymetry_meas_std_m",
            )
        self.bathymetry_weight = _positive_scalar(
            self.bathymetry_weight,
            name="bathymetry_weight",
        )
        self.height_std_m = _positive_scalar(self.height_std_m, name="height_std_m")
        self.adaptive_grid_enabled = bool(self.adaptive_grid_enabled)
        if self.expanded_grid_half_span_m is None:
            self.expanded_grid_half_span_m = 2.0 * self.grid_half_span_m
        else:
            self.expanded_grid_half_span_m = _axis2(
                self.expanded_grid_half_span_m,
                name="expanded_grid_half_span_m",
            )
        if self.expanded_grid_spacing_m is None:
            scale = self.expanded_grid_half_span_m / self.grid_half_span_m
            self.expanded_grid_spacing_m = self.grid_spacing_m * scale
        else:
            self.expanded_grid_spacing_m = _axis2(
                self.expanded_grid_spacing_m,
                name="expanded_grid_spacing_m",
            )
        if np.any(self.expanded_grid_half_span_m <= 0.0):
            raise ValueError("expanded_grid_half_span_m must be positive.")
        if np.any(self.expanded_grid_spacing_m <= 0.0):
            raise ValueError("expanded_grid_spacing_m must be positive.")
        self.adaptive_expand_edge_mass_fraction = float(
            self.adaptive_expand_edge_mass_fraction
        )
        self.adaptive_expand_support_radius_fraction = float(
            self.adaptive_expand_support_radius_fraction
        )
        self.adaptive_contract_edge_mass_fraction = float(
            self.adaptive_contract_edge_mass_fraction
        )
        self.adaptive_contract_support_radius_fraction = float(
            self.adaptive_contract_support_radius_fraction
        )
        self.ambiguity_support_threshold_peak_fraction = float(
            self.ambiguity_support_threshold_peak_fraction
        )
        self.ambiguity_edge_mass_fraction = float(self.ambiguity_edge_mass_fraction)
        self.ambiguity_support_radius_fraction = float(
            self.ambiguity_support_radius_fraction
        )
        self.ambiguity_min_posterior_ess_fraction = float(
            self.ambiguity_min_posterior_ess_fraction
        )
        self.ambiguity_max_peak_probability = float(
            self.ambiguity_max_peak_probability
        )
        self.ambiguity_min_gravity_information_ratio = float(
            self.ambiguity_min_gravity_information_ratio
        )
        self.ambiguity_min_bathymetry_information_ratio = float(
            self.ambiguity_min_bathymetry_information_ratio
        )
        if not (0.0 <= self.adaptive_expand_edge_mass_fraction <= 1.0):
            raise ValueError("adaptive_expand_edge_mass_fraction must be in [0, 1].")
        if not (0.0 <= self.adaptive_contract_edge_mass_fraction <= 1.0):
            raise ValueError(
                "adaptive_contract_edge_mass_fraction must be in [0, 1]."
            )
        if not (
            0.0
            < self.adaptive_contract_support_radius_fraction
            < self.adaptive_expand_support_radius_fraction
            <= 1.0
        ):
            raise ValueError(
                "adaptive support-radius fractions must satisfy "
                "0 < contract < expand <= 1."
            )
        if not (0.0 < self.ambiguity_support_threshold_peak_fraction <= 1.0):
            raise ValueError(
                "ambiguity_support_threshold_peak_fraction must be in (0, 1]."
            )
        if not (0.0 <= self.ambiguity_edge_mass_fraction <= 1.0):
            raise ValueError("ambiguity_edge_mass_fraction must be in [0, 1].")
        if not (0.0 < self.ambiguity_support_radius_fraction <= 1.0):
            raise ValueError("ambiguity_support_radius_fraction must be in (0, 1].")
        if not (0.0 < self.ambiguity_min_posterior_ess_fraction <= 1.0):
            raise ValueError(
                "ambiguity_min_posterior_ess_fraction must be in (0, 1]."
            )
        if not (0.0 < self.ambiguity_max_peak_probability <= 1.0):
            raise ValueError("ambiguity_max_peak_probability must be in (0, 1].")
        if self.ambiguity_min_gravity_information_ratio < 0.0:
            raise ValueError(
                "ambiguity_min_gravity_information_ratio must be nonnegative."
            )
        if self.ambiguity_min_bathymetry_information_ratio < 0.0:
            raise ValueError(
                "ambiguity_min_bathymetry_information_ratio must be nonnegative."
            )


@dataclass
class SequenceMatchEstimate:
    """
    Delayed sequence-based position estimate.
    """

    lat_rad: float
    lon_rad: float
    height_m: float
    covariance_ned_m2: FloatArray
    covariance_geodetic: FloatArray
    predicted_disturbance_mps2: Optional[float] = None
    marginal_peak_probability: Optional[float] = None
    predicted_bathymetry_m: Optional[float] = None

    @property
    def geodetic_vector(self) -> FloatArray:
        """Return `[lat, lon, h]` as a float64 vector."""
        return np.array(
            [self.lat_rad, self.lon_rad, self.height_m],
            dtype=np.float64,
        )


@dataclass
class SequenceAnchorEstimate:
    """
    One delayed anchor estimate extracted from a sequence-inference window.

    These anchors expose a few interior states of the window so a bounded-lag
    smoother can use more of the sequence information than a single delayed
    center estimate.
    """

    time_s: float
    lat_rad: float
    lon_rad: float
    height_m: float
    covariance_ned_m2: FloatArray
    covariance_geodetic: FloatArray
    posterior_mean_offset_ned_m: FloatArray
    viterbi_offset_ned_m: FloatArray
    marginal_peak_probability: float
    posterior_entropy_nats: float
    global_index: int
    delayed_by_steps: int

    @property
    def geodetic_vector(self) -> FloatArray:
        """Return `[lat, lon, h]` as a float64 vector."""
        return np.array(
            [self.lat_rad, self.lon_rad, self.height_m],
            dtype=np.float64,
        )


@dataclass
class SequenceMatchUpdateResult:
    """
    Diagnostics/result of one emitted sequence estimate.
    """

    estimate: SequenceMatchEstimate
    global_index: int
    time_s: float
    window_size_used: int
    delayed_by_steps: int
    num_candidates: int
    posterior_entropy_nats: float
    marginal_peak_probability: float
    predicted_disturbance_mean_mps2: float
    predicted_disturbance_std_mps2: float
    used_gradient: bool
    used_bathymetry: bool
    viterbi_log_score: float
    viterbi_offset_ned_m: FloatArray
    posterior_mean_offset_ned_m: FloatArray
    ambiguity_diagnostics: SequenceAmbiguityDiagnostics
    anchor_estimates: tuple[SequenceAnchorEstimate, ...] = ()


@dataclass
class _SequenceObservation:
    """
    One window element for the sequence matcher.
    """

    global_index: int
    time_s: float
    center_lat_rad: float
    center_lon_rad: float
    center_height_m: float
    candidate_offsets_ned_m: FloatArray
    candidate_lat_rad: FloatArray
    candidate_lon_rad: FloatArray
    candidate_height_m: FloatArray
    log_emission: FloatArray
    predicted_disturbance_mps2: FloatArray
    predicted_bathymetry_m: Optional[FloatArray]
    gravity_meas_std_mps2: float
    bathymetry_meas_std_m: Optional[float]
    grid_mode: str
    grid_half_span_m: FloatArray
    grid_spacing_m: FloatArray
    used_gradient: bool
    used_bathymetry: bool


class GravitySequenceMatcher:
    """
    Sliding-window HMM/Viterbi gravity sequence matcher.

    This matcher is observe-only in the current implementation. It emits delayed
    position estimates for benchmark and analysis use, but does not yet feed
    those estimates back into the INS.
    """

    def __init__(
        self,
        spec: GravitySequenceMatcherSpec,
        map_model: Any,
        *,
        bathymetry_map: Any | None = None,
    ) -> None:
        self.spec = spec
        self.map_model = map_model
        self.bathymetry_map = bathymetry_map
        self._grid_catalog = {
            "nominal": {
                "half_span_m": np.asarray(self.spec.grid_half_span_m, dtype=np.float64),
                "spacing_m": np.asarray(self.spec.grid_spacing_m, dtype=np.float64),
            },
            "expanded": {
                "half_span_m": np.asarray(
                    self.spec.expanded_grid_half_span_m,
                    dtype=np.float64,
                ),
                "spacing_m": np.asarray(
                    self.spec.expanded_grid_spacing_m,
                    dtype=np.float64,
                ),
            },
        }
        for grid in self._grid_catalog.values():
            grid["offsets_ned_m"] = self._build_candidate_grid_offsets(
                grid["half_span_m"],
                grid["spacing_m"],
            )
        self._transition_log_cache: dict[tuple[str, str], FloatArray] = {}
        self._window: Deque[_SequenceObservation] = deque(maxlen=self.spec.window_size)
        self._next_global_index = 0
        self._last_emitted_global_index = -1
        self._active_grid_mode = "nominal"

    def reset(self) -> None:
        """Clear internal history and start a fresh sequence."""
        self._window.clear()
        self._next_global_index = 0
        self._last_emitted_global_index = -1
        self._active_grid_mode = "nominal"

    def _build_candidate_grid_offsets(
        self,
        half_span_m: FloatArray,
        spacing_m: FloatArray,
    ) -> FloatArray:
        """
        Build one local candidate grid in NED coordinates.
        """
        half_n, half_e = np.asarray(half_span_m, dtype=np.float64)
        step_n, step_e = np.asarray(spacing_m, dtype=np.float64)

        north_axis = np.arange(-half_n, half_n + 0.5 * step_n, step_n, dtype=np.float64)
        east_axis = np.arange(-half_e, half_e + 0.5 * step_e, step_e, dtype=np.float64)
        north_mesh, east_mesh = np.meshgrid(north_axis, east_axis, indexing="ij")

        offsets = np.column_stack(
            [
                north_mesh.reshape(-1),
                east_mesh.reshape(-1),
                np.zeros(north_mesh.size, dtype=np.float64),
            ]
        )
        return offsets.astype(np.float64)

    def _grid_definition(
        self,
        mode: str,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Return `(offsets, half_span, spacing)` for a named grid mode.
        """
        if mode not in self._grid_catalog:
            raise KeyError(f"Unknown grid mode {mode!r}.")
        grid = self._grid_catalog[mode]
        return (
            np.asarray(grid["offsets_ned_m"], dtype=np.float64),
            np.asarray(grid["half_span_m"], dtype=np.float64),
            np.asarray(grid["spacing_m"], dtype=np.float64),
        )

    def _transition_log_between(
        self,
        prev_obs: _SequenceObservation,
        curr_obs: _SequenceObservation,
    ) -> FloatArray:
        """
        Build or reuse the transition log-likelihood matrix between two grids.

        Consecutive windows are centered on consecutive INS states, so the
        expected candidate offset relative to the INS center is approximately
        constant when the INS prior is locally accurate. The transition model
        therefore penalizes changes in local grid offset from one step to the
        next.
        """
        cache_key = (str(prev_obs.grid_mode), str(curr_obs.grid_mode))
        cached = self._transition_log_cache.get(cache_key)
        if cached is not None:
            return cached

        prev_offsets = np.asarray(prev_obs.candidate_offsets_ned_m[:, :2], dtype=np.float64)
        curr_offsets = np.asarray(curr_obs.candidate_offsets_ned_m[:, :2], dtype=np.float64)
        delta = curr_offsets[None, :, :] - prev_offsets[:, None, :]
        sigma = self.spec.transition_std_m
        var = sigma**2
        log_t = (
            -0.5
            * np.sum(
                (delta**2) / var[None, None, :] + np.log(2.0 * np.pi * var[None, None, :]),
                axis=2,
            )
        ).astype(np.float64)
        self._transition_log_cache[cache_key] = log_t
        return log_t

    @staticmethod
    def _horizontal_axis_edge_mask(
        offsets_axis_m: FloatArray,
        *,
        half_span_m: float,
        spacing_m: float,
    ) -> NDArray[np.bool_]:
        tol = max(1.0e-9, 0.51 * float(spacing_m))
        return np.abs(np.abs(offsets_axis_m) - float(half_span_m)) <= tol

    def _anchor_local_indices(self, target_local_index: int, window_size: int) -> tuple[int, ...]:
        """
        Select a small, deterministic set of interior anchor indices.

        The default set is the emitted target plus two symmetric interior points
        around it. Near the window edges, indices are clipped and deduplicated.
        """
        mid = int(target_local_index)
        delta = max(1, int(window_size // 4))
        raw = (
            max(0, mid - delta),
            max(0, min(window_size - 1, mid)),
            min(window_size - 1, mid + delta),
        )
        deduped: list[int] = []
        for idx in raw:
            if idx not in deduped:
                deduped.append(int(idx))
        return tuple(deduped)

    def _center_height_from_measurements(
        self,
        *,
        ins_height_m: float,
        depth_measurement: Optional[DepthMeasurement],
        reference_surface_height_m: float,
    ) -> float:
        """
        Resolve the candidate-grid center height from INS or depth aiding.
        """
        if depth_measurement is None:
            return float(ins_height_m)
        return float(reference_surface_height_m) - float(depth_measurement.value_m)

    def _predict_bathymetry_clearance_m(
        self,
        lat_deg: FloatArray,
        lon_deg: FloatArray,
        *,
        depth_measurement: Optional[DepthMeasurement],
        reference_surface_height_m: float,
    ) -> Optional[FloatArray]:
        if self.bathymetry_map is None:
            return None
        if not hasattr(self.bathymetry_map, "evaluate_water_depth_m"):
            raise AttributeError(
                "bathymetry_map must define evaluate_water_depth_m(lat_deg, lon_deg, ...)."
            )

        water_depth = np.asarray(
            self.bathymetry_map.evaluate_water_depth_m(
                lat_deg,
                lon_deg,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        platform_depth = (
            0.0
            if depth_measurement is None
            else float(depth_measurement.value_m)
        )
        clearance = water_depth - platform_depth
        return np.maximum(clearance, 0.0).astype(np.float64)

    def _ambiguity_diagnostics(
        self,
        obs: _SequenceObservation,
        weights: FloatArray,
    ) -> SequenceAmbiguityDiagnostics:
        """
        Summarize posterior ambiguity and candidate-grid coverage for one update.
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != obs.candidate_offsets_ned_m.shape[0]:
            raise ValueError("weights shape does not match candidate grid.")
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            fallback = np.zeros_like(w)
            fallback[0] = 1.0
            w = fallback
        else:
            w = w / total

        ess = float(1.0 / np.sum(np.maximum(w, 0.0) ** 2))
        ess_fraction = ess / max(float(w.size), 1.0)
        peak_probability = float(np.max(w))

        offsets = np.asarray(obs.candidate_offsets_ned_m[:, :2], dtype=np.float64)
        half_span = np.asarray(obs.grid_half_span_m, dtype=np.float64)
        spacing = np.asarray(obs.grid_spacing_m, dtype=np.float64)

        north_edge = self._horizontal_axis_edge_mask(
            offsets[:, 0],
            half_span_m=float(half_span[0]),
            spacing_m=float(spacing[0]),
        )
        east_edge = self._horizontal_axis_edge_mask(
            offsets[:, 1],
            half_span_m=float(half_span[1]),
            spacing_m=float(spacing[1]),
        )
        edge_mask = north_edge | east_edge
        edge_mass_fraction = float(np.sum(w[edge_mask]))

        support_threshold = peak_probability * self.spec.ambiguity_support_threshold_peak_fraction
        support_mask = w >= support_threshold
        if not np.any(support_mask):
            support_mask[int(np.argmax(w))] = True
        support_offsets = offsets[support_mask]
        support_radius_n = float(np.max(np.abs(support_offsets[:, 0]))) if support_offsets.size else 0.0
        support_radius_e = float(np.max(np.abs(support_offsets[:, 1]))) if support_offsets.size else 0.0
        support_radius_fraction_n = support_radius_n / max(float(half_span[0]), 1.0e-9)
        support_radius_fraction_e = support_radius_e / max(float(half_span[1]), 1.0e-9)

        mean_offset = np.sum(w[:, None] * offsets, axis=0)
        centered = offsets - mean_offset[None, :]
        if offsets.shape[0] == 1:
            horiz_cov = np.diag(np.array([0.0, 0.0], dtype=np.float64))
        else:
            horiz_cov = np.sum(
                w[:, None, None] * centered[:, :, None] * centered[:, None, :],
                axis=0,
            ).astype(np.float64)
        eigvals = np.linalg.eigvalsh(horiz_cov)
        eigvals = np.maximum(eigvals, 0.0)
        principal_ratio = float((eigvals[-1] + 1.0e-12) / (eigvals[0] + 1.0e-12))

        gravity_values = np.asarray(obs.predicted_disturbance_mps2, dtype=np.float64)
        gravity_valid = gravity_values[np.isfinite(gravity_values)]
        gravity_spread = (
            float(np.std(gravity_valid))
            if gravity_valid.size > 1
            else 0.0
        )
        gravity_information_ratio = gravity_spread / max(obs.gravity_meas_std_mps2, 1.0e-12)

        bathy_spread: Optional[float] = None
        bathy_information_ratio: Optional[float] = None
        if obs.predicted_bathymetry_m is not None and obs.bathymetry_meas_std_m is not None:
            bathy_values = np.asarray(obs.predicted_bathymetry_m, dtype=np.float64)
            bathy_valid = bathy_values[np.isfinite(bathy_values)]
            bathy_spread = float(np.std(bathy_valid)) if bathy_valid.size > 1 else 0.0
            bathy_information_ratio = bathy_spread / max(obs.bathymetry_meas_std_m, 1.0e-12)

        grid_saturated_north = bool(
            float(np.sum(w[north_edge])) >= 0.5 * self.spec.ambiguity_edge_mass_fraction
            or support_radius_fraction_n >= self.spec.ambiguity_support_radius_fraction
        )
        grid_saturated_east = bool(
            float(np.sum(w[east_edge])) >= 0.5 * self.spec.ambiguity_edge_mass_fraction
            or support_radius_fraction_e >= self.spec.ambiguity_support_radius_fraction
        )
        grid_saturated_any = bool(grid_saturated_north or grid_saturated_east)

        gravity_informative = (
            gravity_information_ratio >= self.spec.ambiguity_min_gravity_information_ratio
        )
        bathymetry_informative = (
            obs.used_bathymetry
            and bathy_information_ratio is not None
            and bathy_information_ratio >= self.spec.ambiguity_min_bathymetry_information_ratio
        )

        if edge_mass_fraction >= self.spec.ambiguity_edge_mass_fraction or grid_saturated_any:
            failure_mode = "edge_clipped"
        elif (
            obs.used_bathymetry
            and bathy_information_ratio is not None
            and gravity_informative
            and bathy_information_ratio < self.spec.ambiguity_min_bathymetry_information_ratio
        ):
            failure_mode = "bathymetry_noninformative"
        elif not (gravity_informative or bathymetry_informative):
            failure_mode = "flat_signature"
        elif (
            ess_fraction < self.spec.ambiguity_min_posterior_ess_fraction
            and peak_probability >= self.spec.ambiguity_max_peak_probability
        ):
            failure_mode = "prior_dominated"
        elif (
            obs.used_bathymetry
            and bathy_information_ratio is not None
            and bathy_information_ratio < self.spec.ambiguity_min_bathymetry_information_ratio
        ):
            failure_mode = "bathymetry_noninformative"
        else:
            failure_mode = "informative"

        return SequenceAmbiguityDiagnostics(
            posterior_candidate_ess=ess,
            posterior_candidate_ess_fraction=ess_fraction,
            edge_mass_fraction=edge_mass_fraction,
            support_radius_n_m=support_radius_n,
            support_radius_e_m=support_radius_e,
            horizontal_covariance_eigenvalue_ratio=principal_ratio,
            grid_saturated_north=grid_saturated_north,
            grid_saturated_east=grid_saturated_east,
            grid_saturated_any=grid_saturated_any,
            gravity_predicted_spread_mps2=gravity_spread,
            gravity_information_ratio=gravity_information_ratio,
            bathymetry_predicted_spread_m=bathy_spread,
            bathymetry_information_ratio=bathy_information_ratio,
            dominant_failure_mode=failure_mode,
            grid_mode=str(obs.grid_mode),
            grid_half_span_m=np.asarray(obs.grid_half_span_m, dtype=np.float64),
            grid_spacing_m=np.asarray(obs.grid_spacing_m, dtype=np.float64),
        )

    def _maybe_update_active_grid_mode(
        self,
        diagnostics: SequenceAmbiguityDiagnostics,
    ) -> None:
        """
        Bounded two-level adaptive grid controller for subsequent observations.
        """
        if not self.spec.adaptive_grid_enabled:
            self._active_grid_mode = "nominal"
            return

        support_radius_fraction = max(
            diagnostics.support_radius_n_m
            / max(float(diagnostics.grid_half_span_m[0]), 1.0e-9),
            diagnostics.support_radius_e_m
            / max(float(diagnostics.grid_half_span_m[1]), 1.0e-9),
        )

        if self._active_grid_mode == "nominal":
            if (
                diagnostics.edge_mass_fraction >= self.spec.adaptive_expand_edge_mass_fraction
                or support_radius_fraction >= self.spec.adaptive_expand_support_radius_fraction
            ):
                self._active_grid_mode = "expanded"
        else:
            if (
                diagnostics.edge_mass_fraction <= self.spec.adaptive_contract_edge_mass_fraction
                and support_radius_fraction <= self.spec.adaptive_contract_support_radius_fraction
            ):
                self._active_grid_mode = "nominal"

    def _build_observation(
        self,
        *,
        measured_disturbance_mps2: float,
        gravity_meas_std_mps2: float,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        measured_gradient_per_s2: Optional[ArrayLike],
        gradient_meas_std_per_s2: Optional[float],
        measured_bathymetry_m: Optional[float],
        bathymetry_meas_std_m: Optional[float],
        depth_measurement: Optional[DepthMeasurement],
        reference_surface_height_m: float,
        time_s: float,
    ) -> _SequenceObservation:
        """
        Build one sequence-observation object around the current INS state.
        """
        state = _state_from_filter_or_state(ins_or_state)
        grid_mode = str(self._active_grid_mode)
        candidate_offsets, grid_half_span_m, grid_spacing_m = self._grid_definition(
            grid_mode
        )
        lat_c = float(state.nominal.lat_rad)
        lon_c = float(wrap_angle_pi(state.nominal.lon_rad))
        h_c = self._center_height_from_measurements(
            ins_height_m=float(state.nominal.height_m),
            depth_measurement=depth_measurement,
            reference_surface_height_m=reference_surface_height_m,
        )

        num_candidates = candidate_offsets.shape[0]
        lat = np.full(num_candidates, lat_c, dtype=np.float64)
        lon = np.full(num_candidates, lon_c, dtype=np.float64)
        h = np.full(num_candidates, h_c, dtype=np.float64)
        lat, lon, h = apply_ned_offsets_to_geodetic(
            lat,
            lon,
            h,
            candidate_offsets,
        )

        pred_g = evaluate_gravity_map_disturbance(
            self.map_model,
            lat,
            lon,
            h,
        )
        lat_deg = np.rad2deg(lat)
        lon_deg = np.rad2deg(lon)
        valid_g = np.isfinite(pred_g)
        log_emission = np.zeros(pred_g.shape, dtype=np.float64)
        if np.any(valid_g):
            gravity_log = np.full(pred_g.shape, -1.0e12, dtype=np.float64)
            gravity_log[valid_g] = gaussian_log_likelihood_scalar(
                float(measured_disturbance_mps2) - pred_g[valid_g],
                sigma=gravity_meas_std_mps2,
            )
            log_emission += gravity_log
        log_emission += diagonal_gaussian_log_likelihood(
            candidate_offsets[:, :2],
            self.spec.center_prior_std_m,
        )

        used_gradient = False
        used_bathymetry = False
        if measured_gradient_per_s2 is not None:
            sigma_grad = (
                self.spec.gradient_meas_std_per_s2
                if gradient_meas_std_per_s2 is None
                else float(gradient_meas_std_per_s2)
            )
            if sigma_grad is None or sigma_grad <= 0.0:
                raise ValueError(
                    "A positive gradient standard deviation is required when "
                    "measured_gradient_per_s2 is provided."
                )

            grad_obs = np.asarray(measured_gradient_per_s2, dtype=np.float64).reshape(-1)
            if grad_obs.shape != (2,):
                raise ValueError(
                    "measured_gradient_per_s2 must have shape (2,), got "
                    f"{grad_obs.shape}."
                )

            grad_pred = evaluate_gravity_map_horizontal_gradient(
                self.map_model,
                lat,
                lon,
                h,
            ).T
            valid_grad = np.all(np.isfinite(grad_pred), axis=1)
            if np.any(valid_grad):
                grad_log = np.full(log_emission.shape, -1.0e12, dtype=np.float64)
                grad_log[valid_grad] = diagonal_gaussian_log_likelihood(
                    grad_obs[None, :] - grad_pred[valid_grad],
                    np.array([sigma_grad, sigma_grad], dtype=np.float64),
                )
                log_emission += grad_log
                used_gradient = True

        pred_bathymetry = self._predict_bathymetry_clearance_m(
            lat_deg,
            lon_deg,
            depth_measurement=depth_measurement,
            reference_surface_height_m=reference_surface_height_m,
        )
        if measured_bathymetry_m is not None:
            sigma_bathy = (
                self.spec.bathymetry_meas_std_m
                if bathymetry_meas_std_m is None
                else float(bathymetry_meas_std_m)
            )
            if sigma_bathy is None or sigma_bathy <= 0.0:
                raise ValueError(
                    "A positive bathymetry standard deviation is required when "
                    "measured_bathymetry_m is provided."
                )
            if pred_bathymetry is None:
                raise ValueError(
                    "measured_bathymetry_m was provided but the matcher has no bathymetry_map."
                )
            valid_mask = np.isfinite(pred_bathymetry)
            if np.any(valid_mask):
                bathy_log = np.full(pred_bathymetry.shape, -1.0e12, dtype=np.float64)
                bathy_log[valid_mask] = gaussian_log_likelihood_scalar(
                    float(measured_bathymetry_m) - pred_bathymetry[valid_mask],
                    sigma=sigma_bathy,
                )
                log_emission += self.spec.bathymetry_weight * bathy_log
                used_bathymetry = True

        obs = _SequenceObservation(
            global_index=self._next_global_index,
            time_s=float(time_s),
            center_lat_rad=lat_c,
            center_lon_rad=lon_c,
            center_height_m=h_c,
            candidate_offsets_ned_m=np.asarray(candidate_offsets, dtype=np.float64),
            candidate_lat_rad=lat,
            candidate_lon_rad=lon,
            candidate_height_m=h,
            log_emission=np.asarray(log_emission, dtype=np.float64),
            predicted_disturbance_mps2=np.asarray(pred_g, dtype=np.float64),
            predicted_bathymetry_m=None
            if pred_bathymetry is None
            else np.asarray(pred_bathymetry, dtype=np.float64),
            gravity_meas_std_mps2=float(gravity_meas_std_mps2),
            bathymetry_meas_std_m=(
                None
                if measured_bathymetry_m is None
                else (
                    self.spec.bathymetry_meas_std_m
                    if bathymetry_meas_std_m is None
                    else float(bathymetry_meas_std_m)
                )
            ),
            grid_mode=grid_mode,
            grid_half_span_m=np.asarray(grid_half_span_m, dtype=np.float64),
            grid_spacing_m=np.asarray(grid_spacing_m, dtype=np.float64),
            used_gradient=used_gradient,
            used_bathymetry=used_bathymetry,
        )
        local_weights = self._stable_posterior_weights(obs.log_emission, obs)
        self._maybe_update_active_grid_mode(
            self._ambiguity_diagnostics(obs, local_weights)
        )
        self._next_global_index += 1
        return obs

    def _run_window_inference(
        self,
        window: list[_SequenceObservation],
    ) -> tuple[
        list[FloatArray],
        list[FloatArray],
        list[FloatArray],
        list[NDArray[np.int64]],
    ]:
        """
        Run forward/backward and Viterbi over the supplied window.
        """
        if len(window) == 0:
            raise ValueError("window must be non-empty.")

        log_e = [
            np.where(np.isfinite(obs.log_emission), obs.log_emission, -1.0e12).astype(
                np.float64
            )
            for obs in window
        ]
        num_steps = len(window)

        alpha = [np.empty_like(le) for le in log_e]
        beta = [np.empty_like(le) for le in log_e]
        delta = [np.empty_like(le) for le in log_e]
        psi = [np.zeros(le.shape[0], dtype=np.int64) for le in log_e]

        alpha[0] = log_e[0].copy()
        delta[0] = log_e[0].copy()

        for t in range(1, num_steps):
            trans_log = self._transition_log_between(window[t - 1], window[t])
            trans_terms = alpha[t - 1][:, None] + trans_log
            alpha[t] = log_e[t] + _logsumexp(trans_terms, axis=0)

            delta_terms = delta[t - 1][:, None] + trans_log
            psi[t] = np.argmax(delta_terms, axis=0).astype(np.int64)
            delta[t] = log_e[t] + np.max(delta_terms, axis=0)

        beta[-1] = np.zeros_like(log_e[-1], dtype=np.float64)
        for t in range(num_steps - 2, -1, -1):
            trans_log = self._transition_log_between(window[t], window[t + 1])
            beta_terms = trans_log + log_e[t + 1][None, :] + beta[t + 1][None, :]
            beta[t] = _logsumexp(beta_terms, axis=1)

        return alpha, beta, delta, psi

    @staticmethod
    def _fallback_candidate_index(
        obs: _SequenceObservation,
        log_posterior: FloatArray,
    ) -> int:
        """
        Select a deterministic fallback candidate when posterior weights collapse.

        Preference order:
        1. highest finite log-posterior value
        2. candidate closest to the INS-centered origin in horizontal N/E offset
        """
        lp = np.asarray(log_posterior, dtype=np.float64).reshape(-1)
        finite = np.isfinite(lp)
        if np.any(finite):
            masked = np.where(finite, lp, -np.inf)
            return int(np.argmax(masked))

        offsets = np.asarray(obs.candidate_offsets_ned_m, dtype=np.float64)
        horiz_norm = np.linalg.norm(offsets[:, :2], axis=1)
        return int(np.argmin(horiz_norm))

    def _stable_posterior_weights(
        self,
        log_posterior: FloatArray,
        obs: _SequenceObservation,
    ) -> FloatArray:
        """
        Convert log-posterior values into normalized weights with safe fallbacks.

        Multi-modal likelihoods can legitimately eliminate every candidate in a
        window if a supporting map is partly out of bounds or numerically rough.
        In that case, degrade to a deterministic one-hot fallback instead of
        crashing the sequence path.
        """
        lp = np.asarray(log_posterior, dtype=np.float64).reshape(-1)
        finite = np.isfinite(lp)
        if np.any(finite):
            stable = lp[finite] - float(np.max(lp[finite]))
            w_valid = np.exp(np.clip(stable, -700.0, 0.0))
            total = float(np.sum(w_valid))
            if total > 0.0 and np.isfinite(total):
                weights = np.zeros_like(lp, dtype=np.float64)
                weights[finite] = w_valid / total
                return weights

        fallback = self._fallback_candidate_index(obs, lp)
        weights = np.zeros_like(lp, dtype=np.float64)
        weights[fallback] = 1.0
        return weights

    def _estimate_for_window_index(
        self,
        window: list[_SequenceObservation],
        alpha: list[FloatArray],
        beta: list[FloatArray],
        delta: list[FloatArray],
        psi: list[NDArray[np.int64]],
        target_local_index: int,
    ) -> tuple[
        SequenceAnchorEstimate,
        bool,
        bool,
        float,
        float,
        Optional[float],
        SequenceAmbiguityDiagnostics,
    ]:
        """
        Build one posterior/viterbi estimate for a selected window index.

        Returns the anchor estimate plus:
        - whether the underlying observation used gradient information
        - the predicted disturbance mean
        - the predicted disturbance std
        """
        target = int(target_local_index)
        if target < 0 or target >= len(window):
            raise IndexError(
                f"target_local_index {target} is out of bounds for window length {len(window)}."
            )

        obs = window[target]
        log_gamma = np.asarray(alpha[target] + beta[target], dtype=np.float64)
        weights = self._stable_posterior_weights(log_gamma, obs)
        lat_hat, lon_hat, h_hat = weighted_geodetic_mean(
            obs.candidate_lat_rad,
            obs.candidate_lon_rad,
            obs.candidate_height_m,
            weights,
        )
        d_ned = geodetic_offsets_to_local_ned(
            obs.candidate_lat_rad,
            obs.candidate_lon_rad,
            obs.candidate_height_m,
            lat_ref_rad=lat_hat,
            lon_ref_rad=lon_hat,
            height_ref_m=h_hat,
        )
        P_ned = weighted_covariance(d_ned, weights)
        P_ned = np.asarray(P_ned, dtype=np.float64)
        P_ned[2, 2] = max(float(P_ned[2, 2]), self.spec.height_std_m**2)
        P_geo = geodetic_covariance_from_ned_covariance(
            lat_ref_rad=float(lat_hat),
            height_ref_m=float(h_hat),
            ned_cov_m2=P_ned,
        )

        pred_g_values = np.asarray(obs.predicted_disturbance_mps2, dtype=np.float64)
        pred_g_mean = float(np.sum(weights * pred_g_values))
        pred_g_var = float(
            np.sum(weights * (pred_g_values - pred_g_mean) ** 2)
        )
        pred_g_std = float(np.sqrt(max(pred_g_var, 0.0)))
        pred_bath_mean = None
        if obs.predicted_bathymetry_m is not None:
            pred_bath = np.asarray(obs.predicted_bathymetry_m, dtype=np.float64)
            valid_bath = np.isfinite(pred_bath)
            if np.any(valid_bath):
                w_bath = weights[valid_bath]
                w_bath_sum = float(np.sum(w_bath))
                if w_bath_sum > 0.0 and np.isfinite(w_bath_sum):
                    pred_bath_mean = float(
                        np.sum((w_bath / w_bath_sum) * pred_bath[valid_bath])
                    )
        mean_offset_ned = np.sum(
            weights[:, None] * obs.candidate_offsets_ned_m,
            axis=0,
        ).astype(np.float64)

        peak_prob = float(np.max(weights))
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))

        ambiguity = self._ambiguity_diagnostics(obs, weights)

        best_last = int(np.argmax(delta[-1]))
        path = np.empty(len(window), dtype=np.int64)
        path[-1] = best_last
        for t in range(len(window) - 1, 0, -1):
            path[t - 1] = psi[t][path[t]]

        best_idx = int(path[target])
        best_offset = obs.candidate_offsets_ned_m[best_idx].copy()

        anchor = SequenceAnchorEstimate(
            time_s=float(obs.time_s),
            lat_rad=float(lat_hat),
            lon_rad=float(lon_hat),
            height_m=float(h_hat),
            covariance_ned_m2=P_ned.astype(np.float64),
            covariance_geodetic=P_geo.astype(np.float64),
            posterior_mean_offset_ned_m=mean_offset_ned.astype(np.float64),
            viterbi_offset_ned_m=np.asarray(best_offset, dtype=np.float64),
            marginal_peak_probability=peak_prob,
            posterior_entropy_nats=entropy,
            global_index=int(obs.global_index),
            delayed_by_steps=len(window) - 1 - target,
        )

        return (
            anchor,
            bool(obs.used_gradient),
            bool(obs.used_bathymetry),
            pred_g_mean,
            pred_g_std,
            pred_bath_mean,
            ambiguity,
        )

    def _emit_result_for_index(
        self,
        window: list[_SequenceObservation],
        target_local_index: int,
    ) -> SequenceMatchUpdateResult:
        """
        Build one delayed sequence estimate for a target window index.
        """
        alpha, beta, delta, psi = self._run_window_inference(window)
        best_last = int(np.argmax(delta[-1]))
        (
            target_anchor,
            used_gradient,
            used_bathymetry,
            pred_g_mean,
            pred_g_std,
            pred_bath_mean,
            ambiguity,
        ) = self._estimate_for_window_index(
            window,
            alpha,
            beta,
            delta,
            psi,
            target_local_index,
        )
        anchor_estimates = tuple(
            self._estimate_for_window_index(
                window,
                alpha,
                beta,
                delta,
                psi,
                idx,
            )[0]
            for idx in self._anchor_local_indices(target_local_index, len(window))
        )
        estimate = SequenceMatchEstimate(
            lat_rad=float(target_anchor.lat_rad),
            lon_rad=float(target_anchor.lon_rad),
            height_m=float(target_anchor.height_m),
            covariance_ned_m2=np.asarray(target_anchor.covariance_ned_m2, dtype=np.float64),
            covariance_geodetic=np.asarray(target_anchor.covariance_geodetic, dtype=np.float64),
            predicted_disturbance_mps2=float(pred_g_mean),
            marginal_peak_probability=float(target_anchor.marginal_peak_probability),
            predicted_bathymetry_m=pred_bath_mean,
        )

        return SequenceMatchUpdateResult(
            estimate=estimate,
            global_index=int(target_anchor.global_index),
            time_s=float(target_anchor.time_s),
            window_size_used=len(window),
            delayed_by_steps=int(target_anchor.delayed_by_steps),
            num_candidates=int(window[target_local_index].candidate_offsets_ned_m.shape[0]),
            posterior_entropy_nats=float(target_anchor.posterior_entropy_nats),
            marginal_peak_probability=float(target_anchor.marginal_peak_probability),
            predicted_disturbance_mean_mps2=pred_g_mean,
            predicted_disturbance_std_mps2=pred_g_std,
            used_gradient=used_gradient,
            used_bathymetry=used_bathymetry,
            ambiguity_diagnostics=ambiguity,
            viterbi_log_score=float(delta[-1][best_last]),
            viterbi_offset_ned_m=np.asarray(target_anchor.viterbi_offset_ned_m, dtype=np.float64),
            posterior_mean_offset_ned_m=np.asarray(
                target_anchor.posterior_mean_offset_ned_m,
                dtype=np.float64,
            ),
            anchor_estimates=anchor_estimates,
        )

    def update(
        self,
        measured_disturbance_mps2: float,
        *,
        gravity_meas_std_mps2: Optional[float] = None,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        depth_measurement: Optional[DepthMeasurement] = None,
        reference_surface_height_m: float = 0.0,
        measured_gradient_per_s2: Optional[ArrayLike] = None,
        gradient_meas_std_per_s2: Optional[float] = None,
        measured_bathymetry_m: Optional[float] = None,
        bathymetry_meas_std_m: Optional[float] = None,
        time_s: Optional[float] = None,
    ) -> list[SequenceMatchUpdateResult]:
        """
        Ingest one gravity observation and emit a delayed sequence estimate when
        the window is sufficiently populated.
        """
        sigma_g = (
            self.spec.gravity_meas_std_mps2
            if gravity_meas_std_mps2 is None
            else float(gravity_meas_std_mps2)
        )
        if sigma_g <= 0.0:
            raise ValueError("gravity_meas_std_mps2 must be positive.")

        state = _state_from_filter_or_state(ins_or_state)
        obs_time = (
            float(time_s)
            if time_s is not None
            else float(getattr(state.nominal, "time_s", np.nan))
        )
        if not np.isfinite(obs_time):
            raise ValueError(
                "Sequence matcher requires a finite time_s, either explicitly or "
                "via ins_or_state.nominal.time_s."
            )

        obs = self._build_observation(
            measured_disturbance_mps2=float(measured_disturbance_mps2),
            gravity_meas_std_mps2=sigma_g,
            ins_or_state=state,
            measured_gradient_per_s2=measured_gradient_per_s2,
            gradient_meas_std_per_s2=gradient_meas_std_per_s2,
            measured_bathymetry_m=measured_bathymetry_m,
            bathymetry_meas_std_m=bathymetry_meas_std_m,
            depth_measurement=depth_measurement,
            reference_surface_height_m=reference_surface_height_m,
            time_s=obs_time,
        )
        self._window.append(obs)

        if len(self._window) < self.spec.window_size:
            return []

        target_global_index = obs.global_index - (self.spec.window_size // 2)
        window = list(self._window)
        results: list[SequenceMatchUpdateResult] = []
        for target_obs in window:
            if target_obs.global_index <= self._last_emitted_global_index:
                continue
            if target_obs.global_index > target_global_index:
                continue
            target_local_index = target_obs.global_index - window[0].global_index
            results.append(self._emit_result_for_index(window, target_local_index))
            self._last_emitted_global_index = target_obs.global_index
        return results

    def update_from_gravimeter_measurement(
        self,
        measurement: GravimeterMeasurement,
        *,
        gravity_meas_std_mps2: Optional[float] = None,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        depth_measurement: Optional[DepthMeasurement] = None,
        reference_surface_height_m: Optional[float] = None,
        measured_gradient_per_s2: Optional[ArrayLike] = None,
        gradient_meas_std_per_s2: Optional[float] = None,
        measured_bathymetry_m: Optional[float] = None,
        bathymetry_meas_std_m: Optional[float] = None,
    ) -> list[SequenceMatchUpdateResult]:
        """
        Convenience wrapper for disturbance-gravimeter measurements.
        """
        if measurement.kind != "disturbance":
            raise ValueError(
                "update_from_gravimeter_measurement(...) requires a disturbance "
                f"measurement, got kind={measurement.kind!r}."
            )

        href = 0.0 if reference_surface_height_m is None else float(reference_surface_height_m)
        if depth_measurement is not None and reference_surface_height_m is None:
            href = float(depth_measurement.reference_surface_height_m)

        return self.update(
            measured_disturbance_mps2=float(measurement.value_mps2),
            gravity_meas_std_mps2=gravity_meas_std_mps2,
            ins_or_state=ins_or_state,
            depth_measurement=depth_measurement,
            reference_surface_height_m=href,
            measured_gradient_per_s2=measured_gradient_per_s2,
            gradient_meas_std_per_s2=gradient_meas_std_per_s2,
            measured_bathymetry_m=measured_bathymetry_m,
            bathymetry_meas_std_m=bathymetry_meas_std_m,
            time_s=None if measurement.time_s is None else float(measurement.time_s),
        )

    def finalize(self) -> list[SequenceMatchUpdateResult]:
        """
        Flush any remaining delayed estimates at the end of a run.
        """
        if len(self._window) == 0:
            return []

        window = list(self._window)
        results: list[SequenceMatchUpdateResult] = []
        for target_local_index, obs in enumerate(window):
            if obs.global_index <= self._last_emitted_global_index:
                continue
            result = self._emit_result_for_index(window, target_local_index)
            self._last_emitted_global_index = obs.global_index
            results.append(result)
        return results


__all__ = [
    "GravitySequenceMatcher",
    "GravitySequenceMatcherSpec",
    "SequenceAmbiguityDiagnostics",
    "SequenceAnchorEstimate",
    "SequenceMatchEstimate",
    "SequenceMatchUpdateResult",
]
