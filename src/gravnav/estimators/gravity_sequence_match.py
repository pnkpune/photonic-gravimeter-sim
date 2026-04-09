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
    height_std_m : float, default=2
        Vertical covariance floor used when packaging delayed estimates.
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
    height_std_m: float = 2.0
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
        self.height_std_m = _positive_scalar(self.height_std_m, name="height_std_m")


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
    time_s: float
    window_size_used: int
    delayed_by_steps: int
    num_candidates: int
    posterior_entropy_nats: float
    marginal_peak_probability: float
    predicted_disturbance_mean_mps2: float
    predicted_disturbance_std_mps2: float
    used_gradient: bool
    viterbi_log_score: float
    viterbi_offset_ned_m: FloatArray
    posterior_mean_offset_ned_m: FloatArray


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
    used_gradient: bool


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
    ) -> None:
        self.spec = spec
        self.map_model = map_model
        self._grid_offsets_ned_m = self._build_candidate_grid_offsets()
        self._transition_log = self._build_transition_log_matrix()
        self._window: Deque[_SequenceObservation] = deque(maxlen=self.spec.window_size)
        self._next_global_index = 0
        self._last_emitted_global_index = -1

    def reset(self) -> None:
        """Clear internal history and start a fresh sequence."""
        self._window.clear()
        self._next_global_index = 0
        self._last_emitted_global_index = -1

    def _build_candidate_grid_offsets(self) -> FloatArray:
        """
        Build the fixed local candidate grid in NED coordinates.
        """
        half_n, half_e = self.spec.grid_half_span_m
        step_n, step_e = self.spec.grid_spacing_m

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

    def _build_transition_log_matrix(self) -> FloatArray:
        """
        Build the candidate-to-candidate transition log-likelihood matrix.

        Consecutive windows are centered on consecutive INS states, so the
        expected candidate offset relative to the INS center is approximately
        constant when the INS prior is locally accurate. The transition model
        therefore penalizes changes in local grid offset from one step to the
        next.
        """
        prev_offsets = self._grid_offsets_ned_m[:, :2]
        curr_offsets = self._grid_offsets_ned_m[:, :2]
        delta = curr_offsets[None, :, :] - prev_offsets[:, None, :]
        sigma = self.spec.transition_std_m
        var = sigma**2
        return (
            -0.5
            * np.sum(
                (delta**2) / var[None, None, :] + np.log(2.0 * np.pi * var[None, None, :]),
                axis=2,
            )
        ).astype(np.float64)

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

    def _build_observation(
        self,
        *,
        measured_disturbance_mps2: float,
        gravity_meas_std_mps2: float,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        measured_gradient_per_s2: Optional[ArrayLike],
        gradient_meas_std_per_s2: Optional[float],
        depth_measurement: Optional[DepthMeasurement],
        reference_surface_height_m: float,
        time_s: float,
    ) -> _SequenceObservation:
        """
        Build one sequence-observation object around the current INS state.
        """
        state = _state_from_filter_or_state(ins_or_state)
        lat_c = float(state.nominal.lat_rad)
        lon_c = float(wrap_angle_pi(state.nominal.lon_rad))
        h_c = self._center_height_from_measurements(
            ins_height_m=float(state.nominal.height_m),
            depth_measurement=depth_measurement,
            reference_surface_height_m=reference_surface_height_m,
        )

        num_candidates = self._grid_offsets_ned_m.shape[0]
        lat = np.full(num_candidates, lat_c, dtype=np.float64)
        lon = np.full(num_candidates, lon_c, dtype=np.float64)
        h = np.full(num_candidates, h_c, dtype=np.float64)
        lat, lon, h = apply_ned_offsets_to_geodetic(
            lat,
            lon,
            h,
            self._grid_offsets_ned_m,
        )

        pred_g = evaluate_gravity_map_disturbance(
            self.map_model,
            lat,
            lon,
            h,
        )
        log_emission = gaussian_log_likelihood_scalar(
            float(measured_disturbance_mps2) - pred_g,
            sigma=gravity_meas_std_mps2,
        )
        log_emission += diagonal_gaussian_log_likelihood(
            self._grid_offsets_ned_m[:, :2],
            self.spec.center_prior_std_m,
        )

        used_gradient = False
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
            log_emission += diagonal_gaussian_log_likelihood(
                grad_obs[None, :] - grad_pred,
                np.array([sigma_grad, sigma_grad], dtype=np.float64),
            )
            used_gradient = True

        obs = _SequenceObservation(
            global_index=self._next_global_index,
            time_s=float(time_s),
            center_lat_rad=lat_c,
            center_lon_rad=lon_c,
            center_height_m=h_c,
            candidate_offsets_ned_m=self._grid_offsets_ned_m.copy(),
            candidate_lat_rad=lat,
            candidate_lon_rad=lon,
            candidate_height_m=h,
            log_emission=np.asarray(log_emission, dtype=np.float64),
            predicted_disturbance_mps2=np.asarray(pred_g, dtype=np.float64),
            used_gradient=used_gradient,
        )
        self._next_global_index += 1
        return obs

    def _run_window_inference(
        self,
        window: list[_SequenceObservation],
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64]]:
        """
        Run forward/backward and Viterbi over the supplied window.
        """
        if len(window) == 0:
            raise ValueError("window must be non-empty.")

        log_e = np.stack([obs.log_emission for obs in window], axis=0)
        num_steps, num_candidates = log_e.shape

        alpha = np.empty((num_steps, num_candidates), dtype=np.float64)
        beta = np.empty((num_steps, num_candidates), dtype=np.float64)
        delta = np.empty((num_steps, num_candidates), dtype=np.float64)
        psi = np.zeros((num_steps, num_candidates), dtype=np.int64)

        alpha[0] = log_e[0]
        delta[0] = log_e[0]

        for t in range(1, num_steps):
            trans_terms = alpha[t - 1][:, None] + self._transition_log
            alpha[t] = log_e[t] + _logsumexp(trans_terms, axis=0)

            delta_terms = delta[t - 1][:, None] + self._transition_log
            psi[t] = np.argmax(delta_terms, axis=0).astype(np.int64)
            delta[t] = log_e[t] + np.max(delta_terms, axis=0)

        beta[-1] = 0.0
        for t in range(num_steps - 2, -1, -1):
            beta_terms = self._transition_log + log_e[t + 1][None, :] + beta[t + 1][None, :]
            beta[t] = _logsumexp(beta_terms, axis=1)

        return alpha, beta, delta, psi

    def _emit_result_for_index(
        self,
        window: list[_SequenceObservation],
        target_local_index: int,
    ) -> SequenceMatchUpdateResult:
        """
        Build one delayed sequence estimate for a target window index.
        """
        alpha, beta, delta, psi = self._run_window_inference(window)
        target = int(target_local_index)
        if target < 0 or target >= len(window):
            raise IndexError(
                f"target_local_index {target} is out of bounds for window length {len(window)}."
            )

        log_gamma = alpha[target] + beta[target]
        log_gamma -= _logsumexp(log_gamma, axis=0)
        weights = np.exp(log_gamma)
        weights /= np.sum(weights)

        obs = window[target]
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

        pred_g_mean = float(np.sum(weights * obs.predicted_disturbance_mps2))
        pred_g_var = float(
            np.sum(weights * (obs.predicted_disturbance_mps2 - pred_g_mean) ** 2)
        )
        pred_g_std = float(np.sqrt(max(pred_g_var, 0.0)))
        mean_offset_ned = np.sum(
            weights[:, None] * obs.candidate_offsets_ned_m,
            axis=0,
        ).astype(np.float64)

        peak_prob = float(np.max(weights))
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))

        best_last = int(np.argmax(delta[-1]))
        path = np.empty(len(window), dtype=np.int64)
        path[-1] = best_last
        for t in range(len(window) - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]

        best_idx = int(path[target])
        best_offset = obs.candidate_offsets_ned_m[best_idx].copy()

        estimate = SequenceMatchEstimate(
            lat_rad=float(lat_hat),
            lon_rad=float(lon_hat),
            height_m=float(h_hat),
            covariance_ned_m2=P_ned.astype(np.float64),
            covariance_geodetic=P_geo.astype(np.float64),
            predicted_disturbance_mps2=pred_g_mean,
            marginal_peak_probability=peak_prob,
        )

        return SequenceMatchUpdateResult(
            estimate=estimate,
            time_s=float(obs.time_s),
            window_size_used=len(window),
            delayed_by_steps=len(window) - 1 - target,
            num_candidates=int(obs.candidate_offsets_ned_m.shape[0]),
            posterior_entropy_nats=entropy,
            marginal_peak_probability=peak_prob,
            predicted_disturbance_mean_mps2=pred_g_mean,
            predicted_disturbance_std_mps2=pred_g_std,
            used_gradient=bool(obs.used_gradient),
            viterbi_log_score=float(delta[-1, best_last]),
            viterbi_offset_ned_m=np.asarray(best_offset, dtype=np.float64),
            posterior_mean_offset_ned_m=mean_offset_ned,
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
    "SequenceMatchEstimate",
    "SequenceMatchUpdateResult",
]
