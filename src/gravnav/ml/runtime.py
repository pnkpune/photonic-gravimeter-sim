"""
Runtime learned localizer that plugs into the existing sequence-output path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.error_state_ins import ErrorStateINS, ErrorStateINSState
from ..estimators.gravity_sequence_match import (
    GravitySequenceMatcher,
    GravitySequenceMatcherSpec,
    SequenceMatchUpdateResult,
)
from ..estimators.map_match_pf import (
    evaluate_gravity_map_horizontal_gradient,
    geodetic_covariance_from_ned_covariance,
)
from .models import RuntimeStudentModel, summarize_query_windows

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


@dataclass
class LearnedLocalizerSpec:
    model_export_path: str | Path
    sequence_spec: GravitySequenceMatcherSpec
    reliability_threshold: float = 0.65
    covariance_scale_min: float = 0.5
    covariance_scale_max: float = 6.0
    name: str = "learned_sequence_localizer"
    target_teacher_parameter_count: int = 320_000_000
    target_student_parameter_count: int = 48_000_000
    target_reliability_parameter_count: int = 3_000_000

    def __post_init__(self) -> None:
        self.model_export_path = str(Path(self.model_export_path).expanduser())
        self.reliability_threshold = float(self.reliability_threshold)
        self.covariance_scale_min = float(self.covariance_scale_min)
        self.covariance_scale_max = float(self.covariance_scale_max)
        if self.covariance_scale_min <= 0.0:
            raise ValueError("covariance_scale_min must be positive.")
        if self.covariance_scale_max < self.covariance_scale_min:
            raise ValueError(
                "covariance_scale_max must be >= covariance_scale_min."
            )


class NeuralEarthSignatureLocalizer:
    """
    Learned delayed localizer with the same emitted update contract as the
    classical sequence matcher.
    """

    def __init__(
        self,
        spec: LearnedLocalizerSpec,
        map_model: Any,
        *,
        bathymetry_map: Any | None = None,
        magnetic_map: Any | None = None,
    ) -> None:
        self.spec = spec
        self.model = RuntimeStudentModel.from_npz(spec.model_export_path)
        self.model.reliability_threshold = float(spec.reliability_threshold)
        self._matcher = GravitySequenceMatcher(
            spec.sequence_spec,
            map_model,
            bathymetry_map=bathymetry_map,
            magnetic_map=magnetic_map,
        )
        self._query_history: deque[FloatArray] = deque(
            maxlen=int(spec.sequence_spec.window_size)
        )
        self._last_query_summary = np.zeros(
            len(self.model.spec.query_feature_names) * 4,
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._matcher.reset()
        self._query_history.clear()
        self._last_query_summary = np.zeros_like(self._last_query_summary)

    @staticmethod
    def _measurement_feature_vector(
        *,
        measured_disturbance_mps2: float,
        measured_gradient_per_s2: Optional[ArrayLike],
        measured_bathymetry_m: Optional[float],
        measured_bathymetry_gradient_m_per_m: Optional[float],
        measured_bathymetry_rugosity_m: Optional[float],
        measured_magnetic_total_nt: Optional[float],
        measured_magnetic_gradient_nt_per_m: Optional[float],
        current_track_unit_ned: Optional[ArrayLike] = None,
        current_context_ned_mps: Optional[ArrayLike] = None,
        tide_surface_height_m: float = 0.0,
        tide_gravity_correction_mps2: float = 0.0,
    ) -> FloatArray:
        grad = (
            np.full(2, np.nan, dtype=np.float64)
            if measured_gradient_per_s2 is None
            else _as_float_array(measured_gradient_per_s2).reshape(2)
        )
        current = (
            np.zeros(3, dtype=np.float64)
            if current_context_ned_mps is None
            else _axis3(current_context_ned_mps, name="current_context_ned_mps")
        )
        track_unit = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if current_track_unit_ned is None
            else _axis3(current_track_unit_ned, name="current_track_unit_ned")
        )
        track_norm = float(np.linalg.norm(track_unit[:2]))
        if track_norm > 1.0e-9:
            track_unit[:2] = track_unit[:2] / track_norm
        bathy_grad_ne = (
            np.full(2, np.nan, dtype=np.float64)
            if measured_bathymetry_gradient_m_per_m is None
            else float(measured_bathymetry_gradient_m_per_m) * track_unit[:2]
        )
        magnetic_grad_ne = (
            np.full(2, np.nan, dtype=np.float64)
            if measured_magnetic_gradient_nt_per_m is None
            else float(measured_magnetic_gradient_nt_per_m) * track_unit[:2]
        )
        return np.array(
            [
                float(measured_disturbance_mps2),
                float(grad[0]),
                float(grad[1]),
                np.nan if measured_bathymetry_m is None else float(measured_bathymetry_m),
                float(bathy_grad_ne[0]),
                float(bathy_grad_ne[1]),
                np.nan
                if measured_bathymetry_rugosity_m is None
                else float(measured_bathymetry_rugosity_m),
                np.nan
                if measured_magnetic_total_nt is None
                else float(measured_magnetic_total_nt),
                float(magnetic_grad_ne[0]),
                float(magnetic_grad_ne[1]),
                float(current[0]),
                float(current[1]),
                float(tide_surface_height_m),
                float(tide_gravity_correction_mps2),
            ],
            dtype=np.float64,
        )

    def _candidate_feature_matrix(
        self,
        obs: Any,
    ) -> FloatArray:
        g_grad = evaluate_gravity_map_horizontal_gradient(
            self._matcher.map_model,
            obs.candidate_lat_rad,
            obs.candidate_lon_rad,
            obs.candidate_height_m,
        ).T
        if obs.predicted_bathymetry_m is None:
            bathy = np.full(obs.candidate_lat_rad.shape, np.nan, dtype=np.float64)
            bathy_grad = np.full((obs.candidate_lat_rad.size, 2), np.nan, dtype=np.float64)
            bathy_rug = np.full(obs.candidate_lat_rad.shape, np.nan, dtype=np.float64)
        else:
            bathy = np.asarray(obs.predicted_bathymetry_m, dtype=np.float64)
            lat_deg = np.rad2deg(obs.candidate_lat_rad)
            lon_deg = np.rad2deg(obs.candidate_lon_rad)
            if self._matcher.bathymetry_map is not None and hasattr(
                self._matcher.bathymetry_map,
                "evaluate_water_depth_gradient_m_per_m",
            ):
                bathy_grad = np.asarray(
                    self._matcher.bathymetry_map.evaluate_water_depth_gradient_m_per_m(
                        lat_deg,
                        lon_deg,
                        reference_surface_height_m=float(obs.center_height_m),
                    ),
                    dtype=np.float64,
                )
            else:
                bathy_grad = np.full((obs.candidate_lat_rad.size, 2), np.nan, dtype=np.float64)
            bathy_rug = (
                np.asarray(obs.predicted_bathymetry_rugosity_m, dtype=np.float64)
                if obs.predicted_bathymetry_rugosity_m is not None
                else np.full(obs.candidate_lat_rad.shape, np.nan, dtype=np.float64)
            )

        if obs.predicted_magnetic_total_nt is None:
            mag = np.full(obs.candidate_lat_rad.shape, np.nan, dtype=np.float64)
            mag_grad = np.full((obs.candidate_lat_rad.size, 2), np.nan, dtype=np.float64)
        else:
            mag = np.asarray(obs.predicted_magnetic_total_nt, dtype=np.float64)
            lat_deg = np.rad2deg(obs.candidate_lat_rad)
            lon_deg = np.rad2deg(obs.candidate_lon_rad)
            if self._matcher.magnetic_map is not None and hasattr(
                self._matcher.magnetic_map,
                "evaluate_horizontal_gradient_nt_per_m",
            ):
                mag_grad = np.asarray(
                    self._matcher.magnetic_map.evaluate_horizontal_gradient_nt_per_m(
                        lat_deg,
                        lon_deg,
                    ),
                    dtype=np.float64,
                )
            else:
                mag_grad = np.full((obs.candidate_lat_rad.size, 2), np.nan, dtype=np.float64)

        return np.column_stack(
            [
                np.asarray(obs.predicted_disturbance_mps2, dtype=np.float64),
                g_grad[:, 0],
                g_grad[:, 1],
                bathy.reshape(-1),
                bathy_grad[:, 0],
                bathy_grad[:, 1],
                bathy_rug.reshape(-1),
                mag.reshape(-1),
                mag_grad[:, 0],
                mag_grad[:, 1],
                np.zeros(obs.candidate_lat_rad.shape, dtype=np.float64),
                np.zeros(obs.candidate_lat_rad.shape, dtype=np.float64),
                np.zeros(obs.candidate_lat_rad.shape, dtype=np.float64),
                np.zeros(obs.candidate_lat_rad.shape, dtype=np.float64),
            ]
        ).astype(np.float64)

    def _reliability_features_from_update(
        self,
        update: SequenceMatchUpdateResult,
    ) -> FloatArray:
        diag = update.ambiguity_diagnostics
        return np.array(
            [
                float(update.marginal_peak_probability),
                float(update.posterior_entropy_nats),
                float(update.predicted_disturbance_std_mps2),
                float(diag.edge_mass_fraction),
                max(
                    float(diag.support_radius_n_m)
                    / max(float(diag.grid_half_span_m[0]), 1.0e-9),
                    float(diag.support_radius_e_m)
                    / max(float(diag.grid_half_span_m[1]), 1.0e-9),
                ),
                float(diag.posterior_candidate_ess_fraction),
                float(diag.gravity_information_ratio),
                float(np.nanmean(np.abs(self._last_query_summary))),
            ],
            dtype=np.float64,
        )

    def _covariance_features_from_update(
        self,
        update: SequenceMatchUpdateResult,
    ) -> FloatArray:
        diag = update.ambiguity_diagnostics
        return np.array(
            [
                float(update.marginal_peak_probability),
                float(update.posterior_entropy_nats),
                float(diag.edge_mass_fraction),
                float(diag.posterior_candidate_ess_fraction),
                float(diag.gravity_information_ratio),
                float(update.predicted_disturbance_std_mps2),
            ],
            dtype=np.float64,
        )

    def _decorate_update(
        self,
        update: SequenceMatchUpdateResult,
    ) -> SequenceMatchUpdateResult:
        reliability_features = self._reliability_features_from_update(update)
        publish_prob = float(
            self.model.publishability_probability_from_features(reliability_features)
        )
        covariance_features = self._covariance_features_from_update(update)
        cov_scale = float(self.model.covariance_scale_from_features(covariance_features))
        cov_scale = float(
            np.clip(
                cov_scale,
                self.spec.covariance_scale_min,
                self.spec.covariance_scale_max,
            )
        )
        update.publishability_probability = publish_prob
        update.localizer_name = str(self.spec.name)
        update.learned_covariance_scale = cov_scale
        update.estimate.covariance_ned_m2 = (
            np.asarray(update.estimate.covariance_ned_m2, dtype=np.float64) * cov_scale
        )
        update.estimate.covariance_geodetic = geodetic_covariance_from_ned_covariance(
            lat_ref_rad=float(update.estimate.lat_rad),
            height_ref_m=float(update.estimate.height_m),
            ned_cov_m2=np.asarray(update.estimate.covariance_ned_m2, dtype=np.float64),
        )
        return update

    def update(
        self,
        measured_disturbance_mps2: float,
        *,
        gravity_meas_std_mps2: Optional[float] = None,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
        search_center_offset_ned_m: Optional[ArrayLike] = None,
        current_track_unit_ned: Optional[ArrayLike] = None,
        depth_measurement: Optional[Any] = None,
        reference_surface_height_m: float = 0.0,
        measured_gradient_per_s2: Optional[ArrayLike] = None,
        gradient_meas_std_per_s2: Optional[float] = None,
        measured_bathymetry_m: Optional[float] = None,
        bathymetry_meas_std_m: Optional[float] = None,
        measured_bathymetry_gradient_m_per_m: Optional[float] = None,
        bathymetry_gradient_meas_std_m_per_m: Optional[float] = None,
        measured_bathymetry_rugosity_m: Optional[float] = None,
        bathymetry_rugosity_meas_std_m: Optional[float] = None,
        measured_magnetic_total_nt: Optional[float] = None,
        magnetic_meas_std_nt: Optional[float] = None,
        measured_magnetic_gradient_nt_per_m: Optional[float] = None,
        magnetic_gradient_meas_std_nt_per_m: Optional[float] = None,
        time_s: Optional[float] = None,
    ) -> list[SequenceMatchUpdateResult]:
        state = (
            ins_or_state.state
            if isinstance(ins_or_state, ErrorStateINS)
            else ins_or_state
        )
        obs_time = (
            float(time_s)
            if time_s is not None
            else float(getattr(state.nominal, "time_s", np.nan))
        )
        obs = self._matcher._build_observation(
            measured_disturbance_mps2=float(measured_disturbance_mps2),
            gravity_meas_std_mps2=(
                self._matcher.spec.gravity_meas_std_mps2
                if gravity_meas_std_mps2 is None
                else float(gravity_meas_std_mps2)
            ),
            ins_or_state=state,
            search_center_offset_ned_m=search_center_offset_ned_m,
            current_track_unit_ned=current_track_unit_ned,
            measured_gradient_per_s2=measured_gradient_per_s2,
            gradient_meas_std_per_s2=gradient_meas_std_per_s2,
            measured_bathymetry_m=measured_bathymetry_m,
            bathymetry_meas_std_m=bathymetry_meas_std_m,
            measured_bathymetry_gradient_m_per_m=measured_bathymetry_gradient_m_per_m,
            bathymetry_gradient_meas_std_m_per_m=bathymetry_gradient_meas_std_m_per_m,
            measured_bathymetry_rugosity_m=measured_bathymetry_rugosity_m,
            bathymetry_rugosity_meas_std_m=bathymetry_rugosity_meas_std_m,
            measured_magnetic_total_nt=measured_magnetic_total_nt,
            magnetic_meas_std_nt=magnetic_meas_std_nt,
            measured_magnetic_gradient_nt_per_m=measured_magnetic_gradient_nt_per_m,
            magnetic_gradient_meas_std_nt_per_m=magnetic_gradient_meas_std_nt_per_m,
            depth_measurement=depth_measurement,
            reference_surface_height_m=reference_surface_height_m,
            time_s=obs_time,
        )
        measurement_feature = self._measurement_feature_vector(
            measured_disturbance_mps2=float(measured_disturbance_mps2),
            measured_gradient_per_s2=measured_gradient_per_s2,
            measured_bathymetry_m=measured_bathymetry_m,
            measured_bathymetry_gradient_m_per_m=measured_bathymetry_gradient_m_per_m,
            measured_bathymetry_rugosity_m=measured_bathymetry_rugosity_m,
            measured_magnetic_total_nt=measured_magnetic_total_nt,
            measured_magnetic_gradient_nt_per_m=measured_magnetic_gradient_nt_per_m,
            current_track_unit_ned=current_track_unit_ned,
        )
        self._query_history.append(np.nan_to_num(measurement_feature, nan=0.0))
        query_window = np.stack(list(self._query_history), axis=0)
        if query_window.shape[0] < self.spec.sequence_spec.window_size:
            pad = np.repeat(
                query_window[:1, :],
                self.spec.sequence_spec.window_size - query_window.shape[0],
                axis=0,
            )
            query_window = np.concatenate([pad, query_window], axis=0)
        query_window = query_window[-self.spec.sequence_spec.window_size :]
        self._last_query_summary = summarize_query_windows(query_window[None, :, :])[0]

        candidate_features = np.nan_to_num(self._candidate_feature_matrix(obs), nan=0.0)
        candidate_offsets = np.asarray(obs.candidate_offsets_ned_m, dtype=np.float64)
        scores = self.model.predict_candidate_scores(
            query_windows=query_window[None, :, :],
            candidate_features=candidate_features[None, :, :],
            candidate_offsets_ned_m=candidate_offsets[None, :, :],
            analytic_log_emission=np.asarray(obs.log_emission, dtype=np.float64)[
                None, :
            ],
        )[0]
        obs.log_emission = np.asarray(scores, dtype=np.float64)
        learned_weights = self._matcher._stable_posterior_weights(obs.log_emission, obs)
        self._matcher._maybe_update_active_grid_mode(
            self._matcher._ambiguity_diagnostics(obs, learned_weights)
        )
        self._matcher._window.append(obs)

        if len(self._matcher._window) < self._matcher.spec.window_size:
            return []

        target_global_index = obs.global_index - (self._matcher.spec.window_size // 2)
        window = list(self._matcher._window)
        results: list[SequenceMatchUpdateResult] = []
        for target_obs in window:
            if target_obs.global_index <= self._matcher._last_emitted_global_index:
                continue
            if target_obs.global_index > target_global_index:
                continue
            target_local_index = target_obs.global_index - window[0].global_index
            result = self._matcher._emit_result_for_index(window, target_local_index)
            self._matcher._last_emitted_global_index = target_obs.global_index
            results.append(self._decorate_update(result))
        return results

    def update_from_gravimeter_measurement(
        self,
        measurement: Any,
        **kwargs: Any,
    ) -> list[SequenceMatchUpdateResult]:
        if measurement.kind != "disturbance":
            raise ValueError(
                "learned localizer requires disturbance gravimeter measurements."
            )
        return self.update(
            float(measurement.value_mps2),
            time_s=None if measurement.time_s is None else float(measurement.time_s),
            **kwargs,
        )

    def finalize(self) -> list[SequenceMatchUpdateResult]:
        results = self._matcher.finalize()
        return [self._decorate_update(result) for result in results]
