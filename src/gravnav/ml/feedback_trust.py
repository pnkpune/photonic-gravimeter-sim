"""
Compact ML trust model for hybrid sequence-feedback gating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.error_state_ins import ERR_POS, ErrorStateINS, ErrorStateINSState
from ..estimators.gravity_sequence_match import SequenceMatchUpdateResult
from ..estimators.map_match_pf import geodetic_offsets_to_local_ned
from ..physics.earth import meridian_radius, prime_vertical_radius

FloatArray = NDArray[np.float64]

SEQUENCE_FEEDBACK_MODES = ("lag_replay", "bias_transfer")
SEQUENCE_FEEDBACK_FEATURE_NAMES = (
    "age_s",
    "window_size_used",
    "delayed_by_steps",
    "marginal_peak_probability",
    "posterior_entropy_nats",
    "edge_mass_fraction",
    "posterior_candidate_ess_fraction",
    "gravity_information_ratio",
    "bathymetry_information_ratio",
    "magnetic_information_ratio",
    "support_radius_fraction_n",
    "support_radius_fraction_e",
    "horizontal_eigenvalue_ratio",
    "projected_std_m",
    "horizontal_std_max_m",
    "correction_norm_m",
    "projected_correction_m",
    "live_projected_std_m",
    "live_horizontal_std_max_m",
    "live_horizontal_speed_mps",
    "live_horizontal_turn_rate_rps",
    "grid_is_expanded",
    "used_gradient",
    "used_bathymetry",
    "used_magnetics",
    "has_publishability_probability",
    "publishability_probability",
    "has_support_expansion_probability",
    "support_expansion_probability",
    "has_learned_covariance_scale",
    "learned_covariance_scale",
)


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _state_from_filter_or_state(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> ErrorStateINSState:
    if isinstance(ins_or_state, ErrorStateINS):
        return ins_or_state.state
    if isinstance(ins_or_state, ErrorStateINSState):
        return ins_or_state
    raise TypeError(
        "Expected ErrorStateINS or ErrorStateINSState, got "
        f"{type(ins_or_state).__name__}."
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sigmoid(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    positive = arr >= 0.0
    out = np.empty_like(arr, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _stable_std(x: FloatArray) -> FloatArray:
    std = np.std(np.asarray(x, dtype=np.float64), axis=0)
    return np.maximum(std, 1.0e-9)


def _horizontal_heading_rate_rps(
    state: ErrorStateINSState,
    previous_state: Optional[ErrorStateINSState],
) -> float:
    if previous_state is None:
        return 0.0
    dt = float(state.nominal.time_s) - float(previous_state.nominal.time_s)
    if not np.isfinite(dt) or dt <= 1.0e-9:
        return 0.0
    v_now = np.asarray(state.nominal.v_ned_mps, dtype=np.float64)[:2]
    v_prev = np.asarray(previous_state.nominal.v_ned_mps, dtype=np.float64)[:2]
    if float(np.linalg.norm(v_now)) <= 1.0e-9 or float(np.linalg.norm(v_prev)) <= 1.0e-9:
        return 0.0
    heading_now = float(np.arctan2(v_now[1], v_now[0]))
    heading_prev = float(np.arctan2(v_prev[1], v_prev[0]))
    delta = float(np.arctan2(np.sin(heading_now - heading_prev), np.cos(heading_now - heading_prev)))
    return abs(delta) / dt


def _ins_horizontal_covariance_ned(
    state: ErrorStateINSState,
) -> FloatArray:
    phi = float(state.nominal.lat_rad)
    h = float(state.nominal.height_m)
    M = float(meridian_radius(phi))
    N = float(prime_vertical_radius(phi))
    cos_phi = max(abs(float(np.cos(phi))), 1.0e-8)
    J_ned_geo = np.diag([M + h, (N + h) * cos_phi, -1.0]).astype(np.float64)
    P_geo = np.asarray(state.P[ERR_POS, ERR_POS], dtype=np.float64)
    return 0.5 * (J_ned_geo @ P_geo @ J_ned_geo.T + (J_ned_geo @ P_geo @ J_ned_geo.T).T)


def extract_sequence_feedback_features(
    sequence_update: SequenceMatchUpdateResult,
    *,
    live_ins_state: ErrorStateINS | ErrorStateINSState,
    current_time_s: float,
    previous_live_ins_state: Optional[ErrorStateINS | ErrorStateINSState] = None,
) -> tuple[FloatArray, dict[str, float]]:
    """
    Extract one fixed-length trust-model feature vector from a delayed sequence
    update and the live INS context at the decision time.
    """
    live_state = _state_from_filter_or_state(live_ins_state)
    prev_state = (
        None
        if previous_live_ins_state is None
        else _state_from_filter_or_state(previous_live_ins_state)
    )

    age_s = max(0.0, float(current_time_s) - float(sequence_update.time_s))
    diag = sequence_update.ambiguity_diagnostics
    offset_h = np.asarray(sequence_update.posterior_mean_offset_ned_m[:2], dtype=np.float64)
    P_h = 0.5 * (
        np.asarray(sequence_update.estimate.covariance_ned_m2[:2, :2], dtype=np.float64)
        + np.asarray(sequence_update.estimate.covariance_ned_m2[:2, :2], dtype=np.float64).T
    )
    eigvals_h, eigvecs_h = np.linalg.eigh(P_h)
    order = np.argsort(eigvals_h)
    eigvals_h = np.maximum(eigvals_h[order], 1.0e-12)
    eigvecs_h = eigvecs_h[:, order]
    best_dir_h = np.asarray(eigvecs_h[:, 0], dtype=np.float64)
    projected_correction_m = float(best_dir_h @ offset_h)
    projected_std_m = float(np.sqrt(eigvals_h[0]))
    horizontal_std = np.sqrt(np.maximum(np.diag(P_h), 0.0))
    correction_norm_m = float(np.linalg.norm(offset_h))
    horizontal_eigenvalue_ratio = float(eigvals_h[1] / eigvals_h[0])

    P_live_ned = _ins_horizontal_covariance_ned(live_state)[:2, :2]
    live_projected_std_m = float(
        np.sqrt(max(float(best_dir_h @ P_live_ned @ best_dir_h), 0.0))
    )
    live_horizontal_std = np.sqrt(np.maximum(np.diag(P_live_ned), 0.0))
    live_speed_mps = float(np.linalg.norm(np.asarray(live_state.nominal.v_ned_mps, dtype=np.float64)[:2]))
    turn_rate_rps = float(_horizontal_heading_rate_rps(live_state, prev_state))

    half_span_n = max(float(diag.grid_half_span_m[0]), 1.0e-9)
    half_span_e = max(float(diag.grid_half_span_m[1]), 1.0e-9)
    support_radius_fraction_n = float(diag.support_radius_n_m) / half_span_n
    support_radius_fraction_e = float(diag.support_radius_e_m) / half_span_e

    publishability = (
        0.5
        if sequence_update.publishability_probability is None
        else float(sequence_update.publishability_probability)
    )
    support_expansion = (
        0.0
        if sequence_update.support_expansion_probability is None
        else float(sequence_update.support_expansion_probability)
    )
    learned_cov_scale = (
        1.0
        if sequence_update.learned_covariance_scale is None
        else float(sequence_update.learned_covariance_scale)
    )

    values = {
        "age_s": age_s,
        "window_size_used": float(sequence_update.window_size_used),
        "delayed_by_steps": float(sequence_update.delayed_by_steps),
        "marginal_peak_probability": float(sequence_update.marginal_peak_probability),
        "posterior_entropy_nats": float(sequence_update.posterior_entropy_nats),
        "edge_mass_fraction": float(diag.edge_mass_fraction),
        "posterior_candidate_ess_fraction": float(diag.posterior_candidate_ess_fraction),
        "gravity_information_ratio": float(diag.gravity_information_ratio),
        "bathymetry_information_ratio": (
            0.0
            if diag.bathymetry_information_ratio is None
            else float(diag.bathymetry_information_ratio)
        ),
        "magnetic_information_ratio": (
            0.0
            if diag.magnetic_information_ratio is None
            else float(diag.magnetic_information_ratio)
        ),
        "support_radius_fraction_n": support_radius_fraction_n,
        "support_radius_fraction_e": support_radius_fraction_e,
        "horizontal_eigenvalue_ratio": horizontal_eigenvalue_ratio,
        "projected_std_m": projected_std_m,
        "horizontal_std_max_m": float(np.max(horizontal_std)),
        "correction_norm_m": correction_norm_m,
        "projected_correction_m": projected_correction_m,
        "live_projected_std_m": live_projected_std_m,
        "live_horizontal_std_max_m": float(np.max(live_horizontal_std)),
        "live_horizontal_speed_mps": live_speed_mps,
        "live_horizontal_turn_rate_rps": turn_rate_rps,
        "grid_is_expanded": 1.0 if str(diag.grid_mode) == "expanded" else 0.0,
        "used_gradient": 1.0 if bool(sequence_update.used_gradient) else 0.0,
        "used_bathymetry": 1.0 if bool(sequence_update.used_bathymetry) else 0.0,
        "used_magnetics": 1.0 if bool(sequence_update.used_magnetics) else 0.0,
        "has_publishability_probability": (
            1.0 if sequence_update.publishability_probability is not None else 0.0
        ),
        "publishability_probability": publishability,
        "has_support_expansion_probability": (
            1.0 if sequence_update.support_expansion_probability is not None else 0.0
        ),
        "support_expansion_probability": support_expansion,
        "has_learned_covariance_scale": (
            1.0 if sequence_update.learned_covariance_scale is not None else 0.0
        ),
        "learned_covariance_scale": learned_cov_scale,
    }
    vector = np.asarray(
        [float(values[name]) for name in SEQUENCE_FEEDBACK_FEATURE_NAMES],
        dtype=np.float64,
    )
    return vector, values


@dataclass
class SequenceFeedbackEventCorpus:
    feature_names: tuple[str, ...]
    features: FloatArray
    region_names: tuple[str, ...]
    region_index: NDArray[np.int64]
    event_region_names: tuple[str, ...]
    event_seed: NDArray[np.int64]
    current_time_s: FloatArray
    update_time_s: FloatArray
    current_step_index: NDArray[np.int64]
    target_step_index: NDArray[np.int64]
    lag_replay_applied: NDArray[np.bool_]
    lag_replay_improves_error: NDArray[np.bool_]
    lag_replay_hmi_safe: NDArray[np.bool_]
    lag_replay_useful_and_safe: NDArray[np.bool_]
    lag_replay_error_delta_m: FloatArray
    lag_replay_best_gain_alpha: FloatArray
    bias_transfer_applied: NDArray[np.bool_]
    bias_transfer_improves_error: NDArray[np.bool_]
    bias_transfer_hmi_safe: NDArray[np.bool_]
    bias_transfer_useful_and_safe: NDArray[np.bool_]
    bias_transfer_error_delta_m: FloatArray
    bias_transfer_best_gain_alpha: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_examples(self) -> int:
        return int(self.features.shape[0])

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            feature_names=np.asarray(self.feature_names, dtype=object),
            features=np.asarray(self.features, dtype=np.float64),
            region_names=np.asarray(self.region_names, dtype=object),
            region_index=np.asarray(self.region_index, dtype=np.int64),
            event_region_names=np.asarray(self.event_region_names, dtype=object),
            event_seed=np.asarray(self.event_seed, dtype=np.int64),
            current_time_s=np.asarray(self.current_time_s, dtype=np.float64),
            update_time_s=np.asarray(self.update_time_s, dtype=np.float64),
            current_step_index=np.asarray(self.current_step_index, dtype=np.int64),
            target_step_index=np.asarray(self.target_step_index, dtype=np.int64),
            lag_replay_applied=np.asarray(self.lag_replay_applied, dtype=bool),
            lag_replay_improves_error=np.asarray(self.lag_replay_improves_error, dtype=bool),
            lag_replay_hmi_safe=np.asarray(self.lag_replay_hmi_safe, dtype=bool),
            lag_replay_useful_and_safe=np.asarray(self.lag_replay_useful_and_safe, dtype=bool),
            lag_replay_error_delta_m=np.asarray(self.lag_replay_error_delta_m, dtype=np.float64),
            lag_replay_best_gain_alpha=np.asarray(
                self.lag_replay_best_gain_alpha, dtype=np.float64
            ),
            bias_transfer_applied=np.asarray(self.bias_transfer_applied, dtype=bool),
            bias_transfer_improves_error=np.asarray(self.bias_transfer_improves_error, dtype=bool),
            bias_transfer_hmi_safe=np.asarray(self.bias_transfer_hmi_safe, dtype=bool),
            bias_transfer_useful_and_safe=np.asarray(self.bias_transfer_useful_and_safe, dtype=bool),
            bias_transfer_error_delta_m=np.asarray(self.bias_transfer_error_delta_m, dtype=np.float64),
            bias_transfer_best_gain_alpha=np.asarray(
                self.bias_transfer_best_gain_alpha, dtype=np.float64
            ),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "SequenceFeedbackEventCorpus":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=True) as data:
            features = np.asarray(data["features"], dtype=np.float64)
            return cls(
                feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
                features=features,
                region_names=tuple(str(x) for x in data["region_names"].tolist()),
                region_index=np.asarray(data["region_index"], dtype=np.int64),
                event_region_names=tuple(str(x) for x in data["event_region_names"].tolist()),
                event_seed=(
                    np.asarray(data["event_seed"], dtype=np.int64)
                    if "event_seed" in data
                    else np.full(features.shape[0], -1, dtype=np.int64)
                ),
                current_time_s=np.asarray(data["current_time_s"], dtype=np.float64),
                update_time_s=np.asarray(data["update_time_s"], dtype=np.float64),
                current_step_index=np.asarray(data["current_step_index"], dtype=np.int64),
                target_step_index=np.asarray(data["target_step_index"], dtype=np.int64),
                lag_replay_applied=np.asarray(data["lag_replay_applied"], dtype=bool),
                lag_replay_improves_error=np.asarray(data["lag_replay_improves_error"], dtype=bool),
                lag_replay_hmi_safe=np.asarray(data["lag_replay_hmi_safe"], dtype=bool),
                lag_replay_useful_and_safe=np.asarray(data["lag_replay_useful_and_safe"], dtype=bool),
                lag_replay_error_delta_m=np.asarray(data["lag_replay_error_delta_m"], dtype=np.float64),
                lag_replay_best_gain_alpha=(
                    np.asarray(data["lag_replay_best_gain_alpha"], dtype=np.float64)
                    if "lag_replay_best_gain_alpha" in data
                    else np.asarray(
                        data["lag_replay_useful_and_safe"], dtype=np.float64
                    )
                ),
                bias_transfer_applied=np.asarray(data["bias_transfer_applied"], dtype=bool),
                bias_transfer_improves_error=np.asarray(data["bias_transfer_improves_error"], dtype=bool),
                bias_transfer_hmi_safe=np.asarray(data["bias_transfer_hmi_safe"], dtype=bool),
                bias_transfer_useful_and_safe=np.asarray(data["bias_transfer_useful_and_safe"], dtype=bool),
                bias_transfer_error_delta_m=np.asarray(data["bias_transfer_error_delta_m"], dtype=np.float64),
                bias_transfer_best_gain_alpha=(
                    np.asarray(data["bias_transfer_best_gain_alpha"], dtype=np.float64)
                    if "bias_transfer_best_gain_alpha" in data
                    else np.asarray(
                        data["bias_transfer_useful_and_safe"], dtype=np.float64
                    )
                ),
                metadata=json.loads(str(data["metadata_json"].item())),
            )

    def mode_useful_and_safe(self, mode: str) -> NDArray[np.bool_]:
        mode_key = str(mode).strip().lower()
        if mode_key == "lag_replay":
            return np.asarray(self.lag_replay_useful_and_safe, dtype=bool)
        if mode_key == "bias_transfer":
            return np.asarray(self.bias_transfer_useful_and_safe, dtype=bool)
        raise KeyError(f"Unsupported feedback mode {mode!r}.")

    def mode_error_delta_m(self, mode: str) -> FloatArray:
        mode_key = str(mode).strip().lower()
        if mode_key == "lag_replay":
            return np.asarray(self.lag_replay_error_delta_m, dtype=np.float64)
        if mode_key == "bias_transfer":
            return np.asarray(self.bias_transfer_error_delta_m, dtype=np.float64)
        raise KeyError(f"Unsupported feedback mode {mode!r}.")

    def mode_hmi_safe(self, mode: str) -> NDArray[np.bool_]:
        mode_key = str(mode).strip().lower()
        if mode_key == "lag_replay":
            return np.asarray(self.lag_replay_hmi_safe, dtype=bool)
        if mode_key == "bias_transfer":
            return np.asarray(self.bias_transfer_hmi_safe, dtype=bool)
        raise KeyError(f"Unsupported feedback mode {mode!r}.")

    def mode_best_gain_alpha(self, mode: str) -> FloatArray:
        mode_key = str(mode).strip().lower()
        if mode_key == "lag_replay":
            return np.asarray(self.lag_replay_best_gain_alpha, dtype=np.float64)
        if mode_key == "bias_transfer":
            return np.asarray(self.bias_transfer_best_gain_alpha, dtype=np.float64)
        raise KeyError(f"Unsupported feedback mode {mode!r}.")

    def region_example_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for idx, name in enumerate(self.region_names):
            out[str(name)] = int(np.sum(self.region_index == idx))
        return out

    def subset(self, mask: ArrayLike) -> "SequenceFeedbackEventCorpus":
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.shape[0] != self.num_examples:
            raise ValueError(
                "subset mask must match the corpus length, "
                f"got {keep.shape[0]} for {self.num_examples}."
            )
        subset_region_names = tuple(
            str(self.event_region_names[idx])
            for idx, selected in enumerate(keep.tolist())
            if selected
        )
        unique_regions = tuple(dict.fromkeys(subset_region_names).keys())
        region_lookup = {name: idx for idx, name in enumerate(unique_regions)}
        return SequenceFeedbackEventCorpus(
            feature_names=self.feature_names,
            features=self.features[keep],
            region_names=unique_regions,
            region_index=np.asarray(
                [region_lookup[name] for name in subset_region_names],
                dtype=np.int64,
            ),
            event_region_names=subset_region_names,
            event_seed=np.asarray(self.event_seed[keep], dtype=np.int64),
            current_time_s=np.asarray(self.current_time_s[keep], dtype=np.float64),
            update_time_s=np.asarray(self.update_time_s[keep], dtype=np.float64),
            current_step_index=np.asarray(self.current_step_index[keep], dtype=np.int64),
            target_step_index=np.asarray(self.target_step_index[keep], dtype=np.int64),
            lag_replay_applied=np.asarray(self.lag_replay_applied[keep], dtype=bool),
            lag_replay_improves_error=np.asarray(
                self.lag_replay_improves_error[keep],
                dtype=bool,
            ),
            lag_replay_hmi_safe=np.asarray(self.lag_replay_hmi_safe[keep], dtype=bool),
            lag_replay_useful_and_safe=np.asarray(
                self.lag_replay_useful_and_safe[keep],
                dtype=bool,
            ),
            lag_replay_error_delta_m=np.asarray(
                self.lag_replay_error_delta_m[keep],
                dtype=np.float64,
            ),
            lag_replay_best_gain_alpha=np.asarray(
                self.lag_replay_best_gain_alpha[keep],
                dtype=np.float64,
            ),
            bias_transfer_applied=np.asarray(self.bias_transfer_applied[keep], dtype=bool),
            bias_transfer_improves_error=np.asarray(
                self.bias_transfer_improves_error[keep],
                dtype=bool,
            ),
            bias_transfer_hmi_safe=np.asarray(
                self.bias_transfer_hmi_safe[keep],
                dtype=bool,
            ),
            bias_transfer_useful_and_safe=np.asarray(
                self.bias_transfer_useful_and_safe[keep],
                dtype=bool,
            ),
            bias_transfer_error_delta_m=np.asarray(
                self.bias_transfer_error_delta_m[keep],
                dtype=np.float64,
            ),
            bias_transfer_best_gain_alpha=np.asarray(
                self.bias_transfer_best_gain_alpha[keep],
                dtype=np.float64,
            ),
            metadata=dict(self.metadata),
        )


@dataclass
class SequenceFeedbackTrustModelSpec:
    feature_names: tuple[str, ...] = SEQUENCE_FEEDBACK_FEATURE_NAMES
    mode_names: tuple[str, ...] = SEQUENCE_FEEDBACK_MODES
    l2: float = 1.0e-2
    learning_rate: float = 0.15
    max_iter: int = 300
    trust_threshold: float = 0.5
    covariance_scale_reference_error_m: float = 50.0
    unsafe_negative_weight: float = 4.0
    positive_error_weight_scale: float = 1.5
    positive_error_weight_reference_m: float = 50.0
    gain_alpha_l2: float = 1.0e-2
    gain_focus_alpha_min: float = 0.25
    gain_focus_alpha_max: float = 0.65
    gain_focus_weight: float = 2.0
    gain_nonzero_safe_weight: float = 1.5
    gain_zero_alpha_weight: float = 0.75
    gain_alpha_reference: float = 0.25
    gain_alpha_prediction_scale: float = 0.35
    gain_alpha_safe_max: float = 0.50
    primary_failure_negative_weight: float = 4.0
    name: str = "sequence_feedback_trust_reference"


@dataclass
class SequenceFeedbackTrustModel:
    spec: SequenceFeedbackTrustModelSpec
    feature_mean: FloatArray
    feature_std: FloatArray
    useful_weights: FloatArray
    useful_bias: FloatArray
    error_delta_weights: FloatArray
    error_delta_bias: FloatArray
    gain_alpha_weights: FloatArray
    gain_alpha_bias: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def _prepare(self, features: ArrayLike) -> FloatArray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(self.spec.feature_names):
            raise ValueError(
                f"Expected feature dimension {len(self.spec.feature_names)}, got {x.shape[1]}."
            )
        return (x - self.feature_mean) / self.feature_std

    def _mode_index(self, mode: str) -> int:
        mode_key = str(mode).strip().lower()
        try:
            return tuple(str(x).strip().lower() for x in self.spec.mode_names).index(mode_key)
        except ValueError as exc:
            raise KeyError(f"Unsupported feedback mode {mode!r}.") from exc

    def predict_trust_probability(
        self,
        features: ArrayLike,
        *,
        mode: str,
    ) -> FloatArray:
        x = self._prepare(features)
        idx = self._mode_index(mode)
        logits = x @ self.useful_weights[idx] + float(self.useful_bias[idx])
        return _sigmoid(logits)

    def predict_error_delta_m(
        self,
        features: ArrayLike,
        *,
        mode: str,
    ) -> FloatArray:
        x = self._prepare(features)
        idx = self._mode_index(mode)
        return (x @ self.error_delta_weights[idx] + float(self.error_delta_bias[idx])).astype(np.float64)

    def predict_covariance_scale(
        self,
        features: ArrayLike,
        *,
        mode: str,
        scale_min: float,
        scale_max: float,
    ) -> FloatArray:
        trust = self.predict_trust_probability(features, mode=mode)
        error_delta = self.predict_error_delta_m(features, mode=mode)
        ref = max(float(self.spec.covariance_scale_reference_error_m), 1.0)
        trust_term = 1.0 + (0.5 - trust) * 1.25
        delta_term = 1.0 + np.clip(error_delta / ref, -0.4, 1.0)
        scale = trust_term * delta_term
        return np.clip(scale, float(scale_min), float(scale_max)).astype(np.float64)

    def _calibrate_gain_alpha(
        self,
        raw_pred: FloatArray,
        *,
        trust_probability: FloatArray,
        alpha_min: float,
        alpha_max: float,
    ) -> FloatArray:
        threshold = float(self.spec.trust_threshold)
        denom = max(1.0 - threshold, 1.0e-6)
        trust_confidence = np.clip((trust_probability - threshold) / denom, 0.0, 1.0)
        ref = float(np.clip(self.spec.gain_alpha_reference, 0.0, 1.0))
        scale = float(max(self.spec.gain_alpha_prediction_scale, 0.0))
        pred = ref + scale * trust_confidence * (raw_pred - ref)
        upper = min(float(alpha_max), float(self.spec.gain_alpha_safe_max))
        return np.clip(pred, float(alpha_min), upper).astype(np.float64)

    def predict_gain_alpha(
        self,
        features: ArrayLike,
        *,
        mode: str,
        alpha_min: float,
        alpha_max: float,
    ) -> FloatArray:
        x = self._prepare(features)
        idx = self._mode_index(mode)
        raw_pred = x @ self.gain_alpha_weights[idx] + float(self.gain_alpha_bias[idx])
        trust = self.predict_trust_probability(features, mode=mode)
        return self._calibrate_gain_alpha(
            np.asarray(raw_pred, dtype=np.float64),
            trust_probability=np.asarray(trust, dtype=np.float64),
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
        )

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            spec_json=np.array(json.dumps(_jsonable(asdict(self.spec)))),
            feature_mean=np.asarray(self.feature_mean, dtype=np.float64),
            feature_std=np.asarray(self.feature_std, dtype=np.float64),
            useful_weights=np.asarray(self.useful_weights, dtype=np.float64),
            useful_bias=np.asarray(self.useful_bias, dtype=np.float64),
            error_delta_weights=np.asarray(self.error_delta_weights, dtype=np.float64),
            error_delta_bias=np.asarray(self.error_delta_bias, dtype=np.float64),
            gain_alpha_weights=np.asarray(self.gain_alpha_weights, dtype=np.float64),
            gain_alpha_bias=np.asarray(self.gain_alpha_bias, dtype=np.float64),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "SequenceFeedbackTrustModel":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            return cls(
                spec=SequenceFeedbackTrustModelSpec(**json.loads(str(data["spec_json"].item()))),
                feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
                feature_std=np.asarray(data["feature_std"], dtype=np.float64),
                useful_weights=np.asarray(data["useful_weights"], dtype=np.float64),
                useful_bias=np.asarray(data["useful_bias"], dtype=np.float64),
                error_delta_weights=np.asarray(data["error_delta_weights"], dtype=np.float64),
                error_delta_bias=np.asarray(data["error_delta_bias"], dtype=np.float64),
                gain_alpha_weights=(
                    np.asarray(data["gain_alpha_weights"], dtype=np.float64)
                    if "gain_alpha_weights" in data
                    else np.zeros_like(
                        np.asarray(data["error_delta_weights"], dtype=np.float64)
                    )
                ),
                gain_alpha_bias=(
                    np.asarray(data["gain_alpha_bias"], dtype=np.float64)
                    if "gain_alpha_bias" in data
                    else np.ones_like(
                        np.asarray(data["error_delta_bias"], dtype=np.float64)
                    )
                ),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


@dataclass(frozen=True)
class SequenceFeedbackTrustCommitteeMember:
    name: str
    drop_seed: int
    model_path: str
    model: SequenceFeedbackTrustModel


@dataclass(frozen=True)
class SequenceFeedbackTrustCommitteePrediction:
    trust_probability: float
    predicted_error_delta_m: float
    gain_alpha: float
    covariance_scale: float
    trust_allowed: bool
    member_trust_probabilities: tuple[float, ...]
    member_predicted_error_delta_m: tuple[float, ...]
    member_gain_alpha: tuple[float, ...]
    member_covariance_scale: tuple[float, ...]
    members_passing: int
    rejection_reason: Optional[str]


@dataclass
class SequenceFeedbackTrustCommittee:
    aggregator: str
    members: tuple[SequenceFeedbackTrustCommitteeMember, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest_json(
        cls,
        path: str | Path,
    ) -> "SequenceFeedbackTrustCommittee":
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        members: list[SequenceFeedbackTrustCommitteeMember] = []
        for raw_member in payload.get("members", []):
            model_path = Path(str(raw_member["model_path"])).expanduser()
            if not model_path.is_absolute():
                model_path = (manifest_path.parent / model_path).resolve()
            else:
                model_path = model_path.resolve()
            members.append(
                SequenceFeedbackTrustCommitteeMember(
                    name=str(raw_member["name"]),
                    drop_seed=int(raw_member["drop_seed"]),
                    model_path=str(model_path),
                    model=load_sequence_feedback_trust_model(model_path),
                )
            )
        return cls(
            aggregator=str(payload.get("aggregator", "conservative_unanimity_v1")),
            members=tuple(members),
            metadata={
                "manifest_path": str(manifest_path),
                **dict(payload.get("metadata", {})),
            },
        )

    def predict(
        self,
        features: ArrayLike,
        *,
        mode: str,
        min_trust_probability: float,
        max_predicted_error_delta_m: Optional[float],
        alpha_min: float,
        alpha_max: float,
        covariance_scale_min: float,
        covariance_scale_max: float,
    ) -> SequenceFeedbackTrustCommitteePrediction:
        if self.aggregator != "conservative_unanimity_v1":
            raise ValueError(f"Unsupported committee aggregator {self.aggregator!r}.")
        if not self.members:
            raise ValueError("Trust committee must include at least one member.")

        member_trust: list[float] = []
        member_delta: list[float] = []
        member_gain: list[float] = []
        member_cov_scale: list[float] = []
        member_pass: list[bool] = []
        rejection_reason: Optional[str] = None
        for member in self.members:
            trust_probability = float(
                member.model.predict_trust_probability(features, mode=mode)[0]
            )
            predicted_error_delta_m = float(
                member.model.predict_error_delta_m(features, mode=mode)[0]
            )
            gain_alpha = float(
                member.model.predict_gain_alpha(
                    features,
                    mode=mode,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                )[0]
            )
            covariance_scale = float(
                member.model.predict_covariance_scale(
                    features,
                    mode=mode,
                    scale_min=covariance_scale_min,
                    scale_max=covariance_scale_max,
                )[0]
            )
            trust_ok = bool(
                np.isfinite(trust_probability)
                and trust_probability >= float(min_trust_probability)
            )
            delta_ok = True
            if max_predicted_error_delta_m is not None:
                delta_ok = bool(
                    np.isfinite(predicted_error_delta_m)
                    and predicted_error_delta_m
                    <= float(max_predicted_error_delta_m)
                )
            passes = bool(trust_ok and delta_ok)
            if not passes and rejection_reason is None:
                if not trust_ok:
                    rejection_reason = (
                        f"committee_member_{member.name}_trust_probability_"
                        f"{trust_probability:.3f}_below_{float(min_trust_probability):.3f}"
                    )
                else:
                    rejection_reason = (
                        f"committee_member_{member.name}_predicted_error_delta_m_"
                        f"{predicted_error_delta_m:.3f}_above_"
                        f"{float(max_predicted_error_delta_m):.3f}"
                    )
            member_trust.append(trust_probability)
            member_delta.append(predicted_error_delta_m)
            member_gain.append(gain_alpha)
            member_cov_scale.append(covariance_scale)
            member_pass.append(passes)

        return SequenceFeedbackTrustCommitteePrediction(
            trust_probability=float(np.min(np.asarray(member_trust, dtype=np.float64))),
            predicted_error_delta_m=float(
                np.max(np.asarray(member_delta, dtype=np.float64))
            ),
            gain_alpha=float(np.min(np.asarray(member_gain, dtype=np.float64))),
            covariance_scale=float(
                np.max(np.asarray(member_cov_scale, dtype=np.float64))
            ),
            trust_allowed=bool(all(member_pass)),
            member_trust_probabilities=tuple(float(x) for x in member_trust),
            member_predicted_error_delta_m=tuple(float(x) for x in member_delta),
            member_gain_alpha=tuple(float(x) for x in member_gain),
            member_covariance_scale=tuple(float(x) for x in member_cov_scale),
            members_passing=int(sum(1 for passed in member_pass if passed)),
            rejection_reason=rejection_reason,
        )


def load_sequence_feedback_trust_model(
    path: str | Path,
) -> SequenceFeedbackTrustModel:
    return SequenceFeedbackTrustModel.from_npz(path)


def load_sequence_feedback_trust_committee(
    path: str | Path,
) -> SequenceFeedbackTrustCommittee:
    return SequenceFeedbackTrustCommittee.from_manifest_json(path)


def _normalize_failure_seed_pairs(
    manifest: Optional[
        Mapping[str, Sequence[int]]
        | Sequence[Mapping[str, Any]]
        | Sequence[tuple[str, int]]
        | set[tuple[str, int]]
    ],
) -> set[tuple[str, int]]:
    if manifest is None:
        return set()
    if isinstance(manifest, set):
        return {(str(region), int(seed)) for region, seed in manifest}
    if isinstance(manifest, Sequence) and manifest and isinstance(manifest[0], tuple):
        return {(str(region), int(seed)) for region, seed in manifest}
    if isinstance(manifest, Mapping):
        payload = manifest
        if "failure_pairs" in payload:
            return _normalize_failure_seed_pairs(payload["failure_pairs"])
        if "regions" in payload and isinstance(payload["regions"], Mapping):
            return _normalize_failure_seed_pairs(payload["regions"])
        pairs: set[tuple[str, int]] = set()
        for region_name, seeds in payload.items():
            if isinstance(seeds, (list, tuple, set)):
                for seed in seeds:
                    pairs.add((str(region_name), int(seed)))
        return pairs
    pairs = set()
    for row in manifest:
        region_name = str(row["region_name"])
        for seed in row.get("seeds", []):
            pairs.add((region_name, int(seed)))
    return pairs


def load_sequence_feedback_failure_manifest(
    path: str | Path,
) -> set[tuple[str, int]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return _normalize_failure_seed_pairs(payload)


def _primary_failure_negative_mask(
    corpus: SequenceFeedbackEventCorpus,
    manifest: Optional[
        Mapping[str, Sequence[int]]
        | Sequence[Mapping[str, Any]]
        | Sequence[tuple[str, int]]
        | set[tuple[str, int]]
    ],
) -> NDArray[np.bool_]:
    pairs = _normalize_failure_seed_pairs(manifest)
    if not pairs:
        return np.zeros(corpus.num_examples, dtype=bool)
    return np.asarray(
        [
            (str(region_name), int(seed)) in pairs
            for region_name, seed in zip(
                corpus.event_region_names,
                corpus.event_seed.tolist(),
            )
        ],
        dtype=bool,
    )


def _fit_balanced_logistic(
    x: FloatArray,
    y: NDArray[np.bool_],
    *,
    l2: float,
    learning_rate: float,
    max_iter: int,
    sample_weights: Optional[FloatArray] = None,
) -> tuple[FloatArray, float]:
    X = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    pos = float(np.sum(target > 0.5))
    neg = float(target.size - pos)
    if pos <= 0.0:
        return w, float(np.log(1.0e-6 / (1.0 - 1.0e-6)))
    if neg <= 0.0:
        return w, float(np.log((1.0 - 1.0e-6) / 1.0e-6))
    balance_weights = np.where(target > 0.5, 0.5 / pos, 0.5 / neg).astype(np.float64)
    if sample_weights is None:
        external_weights = np.ones(target.size, dtype=np.float64)
    else:
        external_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if external_weights.shape != target.shape:
            raise ValueError(
                "sample_weights must match the target shape, "
                f"got {external_weights.shape} for {target.shape}."
            )
        external_weights = np.maximum(external_weights, 1.0e-9)
        external_weights /= float(np.mean(external_weights))
    balance_weights *= external_weights * float(target.size)
    lr = float(learning_rate)
    ridge = float(l2)
    for _ in range(int(max_iter)):
        logits = X @ w + b
        probs = _sigmoid(logits)
        residual = (probs - target) * balance_weights
        grad_w = (X.T @ residual) / float(target.size) + ridge * w
        grad_b = float(np.sum(residual) / float(target.size))
        w -= lr * grad_w
        b -= lr * grad_b
    return w.astype(np.float64), float(b)


def _fit_ridge_regression(
    x: FloatArray,
    y: FloatArray,
    *,
    l2: float,
    sample_weights: Optional[FloatArray] = None,
) -> tuple[FloatArray, float]:
    X = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    X_aug = np.column_stack([X, np.ones(X.shape[0], dtype=np.float64)])
    if sample_weights is None:
        weights = np.ones(target.size, dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape != target.shape:
            raise ValueError(
                "sample_weights must match the target shape, "
                f"got {weights.shape} for {target.shape}."
            )
        weights = np.maximum(weights, 1.0e-9)
        weights /= float(np.mean(weights))
    weighted_X = X_aug * weights[:, None]
    weighted_y = target * weights
    gram = weighted_X.T @ X_aug + float(l2) * np.eye(X_aug.shape[1], dtype=np.float64)
    sol = np.linalg.solve(gram, weighted_X.T @ weighted_y)
    return np.asarray(sol[:-1], dtype=np.float64), float(sol[-1])


def _median_or_zero(values: FloatArray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _gain_histogram_summary(
    corpus: SequenceFeedbackEventCorpus,
    *,
    mode: str,
) -> dict[str, dict[str, int]]:
    target = np.asarray(corpus.mode_best_gain_alpha(mode), dtype=np.float64)
    out: dict[str, dict[str, int]] = {}
    for region_idx, region_name in enumerate(corpus.region_names):
        region_mask = np.asarray(corpus.region_index == region_idx, dtype=bool)
        if not np.any(region_mask):
            out[str(region_name)] = {}
            continue
        rounded = np.round(target[region_mask], 2)
        unique, counts = np.unique(rounded, return_counts=True)
        out[str(region_name)] = {
            f"{float(alpha):.2f}": int(count)
            for alpha, count in zip(unique.tolist(), counts.tolist())
        }
    return out


def _binary_metrics(
    y_true: NDArray[np.bool_],
    prob: FloatArray,
    *,
    threshold: float,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=bool).reshape(-1)
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    pred = p >= float(threshold)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & (~truth)))
    fn = int(np.sum((~pred) & truth))
    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    f1 = 0.0 if (precision + recall) == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {
        "positive_fraction": float(np.mean(pred.astype(np.float64))),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": float(np.mean((p - truth.astype(np.float64)) ** 2)),
    }


def fit_sequence_feedback_trust_model(
    corpus: SequenceFeedbackEventCorpus,
    *,
    spec: Optional[SequenceFeedbackTrustModelSpec] = None,
    hard_negative_manifest: Optional[
        Mapping[str, Sequence[int]]
        | Sequence[Mapping[str, Any]]
        | Sequence[tuple[str, int]]
        | set[tuple[str, int]]
    ] = None,
) -> SequenceFeedbackTrustModel:
    model_spec = SequenceFeedbackTrustModelSpec() if spec is None else spec
    X = np.asarray(corpus.features, dtype=np.float64)
    mean = np.mean(X, axis=0)
    std = _stable_std(X)
    Xn = (X - mean) / std
    primary_failure_mask = _primary_failure_negative_mask(
        corpus,
        hard_negative_manifest,
    )

    useful_weights = np.zeros((len(model_spec.mode_names), X.shape[1]), dtype=np.float64)
    useful_bias = np.zeros(len(model_spec.mode_names), dtype=np.float64)
    error_weights = np.zeros_like(useful_weights)
    error_bias = np.zeros_like(useful_bias)
    gain_alpha_weights = np.zeros_like(useful_weights)
    gain_alpha_bias = np.ones_like(useful_bias)

    training_metrics: dict[str, Any] = {}
    gain_histogram_summary: dict[str, Any] = {}
    for mode_idx, mode_name in enumerate(model_spec.mode_names):
        useful_target = corpus.mode_useful_and_safe(mode_name)
        error_target = corpus.mode_error_delta_m(mode_name)
        gain_alpha_target = np.clip(
            corpus.mode_best_gain_alpha(mode_name),
            0.0,
            1.0,
        )
        hmi_safe_target = corpus.mode_hmi_safe(mode_name)
        severity_weights = np.ones_like(error_target, dtype=np.float64)
        negative_mask = ~useful_target
        severity_ref = max(float(model_spec.positive_error_weight_reference_m), 1.0)
        severity = np.clip(np.maximum(error_target, 0.0) / severity_ref, 0.0, 4.0)
        severity_weights[negative_mask] *= (
            1.0 + float(model_spec.positive_error_weight_scale) * severity[negative_mask]
        )
        severity_weights[negative_mask & (~hmi_safe_target)] *= float(
            model_spec.unsafe_negative_weight
        )
        primary_failure_negative_mask = negative_mask & primary_failure_mask
        if str(mode_name) == "lag_replay":
            severity_weights[primary_failure_negative_mask] *= float(
                model_spec.primary_failure_negative_weight
            )
        gain_alpha_weights_target = np.ones_like(gain_alpha_target, dtype=np.float64)
        zero_gain_mask = gain_alpha_target <= 1.0e-6
        mid_gain_mask = (
            gain_alpha_target >= float(model_spec.gain_focus_alpha_min)
        ) & (
            gain_alpha_target <= float(model_spec.gain_focus_alpha_max)
        )
        nonzero_safe_mask = useful_target & hmi_safe_target & (~zero_gain_mask)
        gain_alpha_weights_target[zero_gain_mask] *= float(
            model_spec.gain_zero_alpha_weight
        )
        gain_alpha_weights_target[mid_gain_mask] *= float(
            model_spec.gain_focus_weight
        )
        gain_alpha_weights_target[nonzero_safe_mask] *= float(
            model_spec.gain_nonzero_safe_weight
        )
        if str(mode_name) == "lag_replay":
            gain_alpha_weights_target[primary_failure_negative_mask] *= float(
                model_spec.primary_failure_negative_weight
            )
        useful_weights[mode_idx], useful_bias[mode_idx] = _fit_balanced_logistic(
            Xn,
            useful_target,
            l2=model_spec.l2,
            learning_rate=model_spec.learning_rate,
            max_iter=model_spec.max_iter,
            sample_weights=severity_weights,
        )
        error_weights[mode_idx], error_bias[mode_idx] = _fit_ridge_regression(
            Xn,
            error_target,
            l2=model_spec.l2,
            sample_weights=severity_weights,
        )
        gain_alpha_weights[mode_idx], gain_alpha_bias[mode_idx] = _fit_ridge_regression(
            Xn,
            gain_alpha_target,
            l2=model_spec.gain_alpha_l2,
            sample_weights=gain_alpha_weights_target,
        )
        prob = _sigmoid(Xn @ useful_weights[mode_idx] + useful_bias[mode_idx])
        raw_pred_gain_alpha = np.asarray(
            Xn @ gain_alpha_weights[mode_idx] + gain_alpha_bias[mode_idx],
            dtype=np.float64,
        )
        pred_gain_alpha = SequenceFeedbackTrustModel(
            spec=model_spec,
            feature_mean=mean.astype(np.float64),
            feature_std=std.astype(np.float64),
            useful_weights=useful_weights,
            useful_bias=useful_bias,
            error_delta_weights=error_weights,
            error_delta_bias=error_bias,
            gain_alpha_weights=gain_alpha_weights,
            gain_alpha_bias=gain_alpha_bias,
            metadata={},
        )._calibrate_gain_alpha(
            raw_pred_gain_alpha,
            trust_probability=np.asarray(prob, dtype=np.float64),
            alpha_min=0.0,
            alpha_max=1.0,
        )
        applied_mask = (prob >= float(model_spec.trust_threshold)) & (
            pred_gain_alpha > 1.0e-6
        )
        metrics = _binary_metrics(
            useful_target,
            prob,
            threshold=model_spec.trust_threshold,
        )
        metrics["error_delta_rmse_m"] = float(
            np.sqrt(
                np.mean(
                    (
                        (Xn @ error_weights[mode_idx] + error_bias[mode_idx])
                        - error_target
                    )
                    ** 2
                )
            )
        )
        metrics["gain_alpha_rmse"] = float(
            np.sqrt(np.mean((pred_gain_alpha - gain_alpha_target) ** 2))
        )
        metrics["median_applied_alpha"] = _median_or_zero(pred_gain_alpha[applied_mask])
        metrics["median_target_gain_alpha"] = _median_or_zero(
            gain_alpha_target[useful_target]
        )
        metrics["primary_failure_negative_count"] = int(
            np.sum(primary_failure_negative_mask)
        )
        training_metrics[str(mode_name)] = metrics
        gain_histogram_summary[str(mode_name)] = _gain_histogram_summary(
            corpus,
            mode=str(mode_name),
        )

    return SequenceFeedbackTrustModel(
        spec=model_spec,
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        useful_weights=useful_weights,
        useful_bias=useful_bias,
        error_delta_weights=error_weights,
        error_delta_bias=error_bias,
        gain_alpha_weights=gain_alpha_weights,
        gain_alpha_bias=gain_alpha_bias,
        metadata={
            "feature_names": list(model_spec.feature_names),
            "training_metrics": training_metrics,
            "gain_histogram_summary": gain_histogram_summary,
            "num_examples": int(corpus.num_examples),
            "region_example_counts": corpus.region_example_counts(),
            "primary_failure_negative_example_count": int(np.sum(primary_failure_mask)),
            "primary_failure_seed_pairs": [
                {"region_name": str(region_name), "seed": int(seed)}
                for region_name, seed in sorted(
                    _normalize_failure_seed_pairs(hard_negative_manifest)
                )
            ],
        },
    )


def cross_validate_sequence_feedback_trust_model(
    corpus: SequenceFeedbackEventCorpus,
    *,
    spec: Optional[SequenceFeedbackTrustModelSpec] = None,
    hard_negative_manifest: Optional[
        Mapping[str, Sequence[int]]
        | Sequence[Mapping[str, Any]]
        | Sequence[tuple[str, int]]
        | set[tuple[str, int]]
    ] = None,
) -> dict[str, Any]:
    model_spec = SequenceFeedbackTrustModelSpec() if spec is None else spec
    folds: list[dict[str, Any]] = []
    aggregate_by_mode: dict[str, dict[str, list[float]]] = {
        str(mode): {
            "positive_fraction": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "brier_score": [],
            "error_delta_rmse_m": [],
            "gain_alpha_rmse": [],
            "median_applied_alpha": [],
        }
        for mode in model_spec.mode_names
    }
    for region_idx, region_name in enumerate(corpus.region_names):
        test_mask = np.asarray(corpus.region_index == region_idx, dtype=bool)
        train_mask = ~test_mask
        if not np.any(train_mask) or not np.any(test_mask):
            continue
        train_corpus = corpus.subset(train_mask)
        model = fit_sequence_feedback_trust_model(
            train_corpus,
            spec=model_spec,
            hard_negative_manifest=hard_negative_manifest,
        )
        fold_summary: dict[str, Any] = {"held_out_region": str(region_name), "num_examples": int(np.sum(test_mask))}
        for mode in model_spec.mode_names:
            prob = model.predict_trust_probability(corpus.features[test_mask], mode=mode)
            pred_delta = model.predict_error_delta_m(corpus.features[test_mask], mode=mode)
            pred_gain_alpha = model.predict_gain_alpha(
                corpus.features[test_mask],
                mode=mode,
                alpha_min=0.0,
                alpha_max=1.0,
            )
            metrics = _binary_metrics(
                corpus.mode_useful_and_safe(mode)[test_mask],
                prob,
                threshold=model_spec.trust_threshold,
            )
            metrics["error_delta_rmse_m"] = float(
                np.sqrt(
                    np.mean(
                        (
                            pred_delta
                            - corpus.mode_error_delta_m(mode)[test_mask]
                        )
                        ** 2
                    )
                )
            )
            metrics["gain_alpha_rmse"] = float(
                np.sqrt(
                    np.mean(
                        (
                            pred_gain_alpha
                            - corpus.mode_best_gain_alpha(mode)[test_mask]
                        )
                        ** 2
                    )
                )
            )
            applied_mask = (prob >= float(model_spec.trust_threshold)) & (
                pred_gain_alpha > 1.0e-6
            )
            metrics["median_applied_alpha"] = _median_or_zero(
                pred_gain_alpha[applied_mask]
            )
            fold_summary[str(mode)] = metrics
            for key, value in metrics.items():
                aggregate_by_mode[str(mode)][str(key)].append(float(value))
        folds.append(fold_summary)

    aggregate = {
        mode: {
            f"median_{metric}": float(np.median(values)) if values else float("nan")
            for metric, values in metrics.items()
        }
        for mode, metrics in aggregate_by_mode.items()
    }
    return {
        "num_folds": len(folds),
        "feature_names": list(model_spec.feature_names),
        "modes": list(model_spec.mode_names),
        "trust_threshold": float(model_spec.trust_threshold),
        "primary_failure_seed_pairs": [
            {"region_name": str(region_name), "seed": int(seed)}
            for region_name, seed in sorted(
                _normalize_failure_seed_pairs(hard_negative_manifest)
            )
        ],
        "folds": folds,
        "aggregate": aggregate,
    }


__all__ = [
    "SEQUENCE_FEEDBACK_FEATURE_NAMES",
    "SEQUENCE_FEEDBACK_MODES",
    "SequenceFeedbackEventCorpus",
    "SequenceFeedbackTrustCommittee",
    "SequenceFeedbackTrustCommitteeMember",
    "SequenceFeedbackTrustCommitteePrediction",
    "SequenceFeedbackTrustModel",
    "SequenceFeedbackTrustModelSpec",
    "cross_validate_sequence_feedback_trust_model",
    "extract_sequence_feedback_features",
    "fit_sequence_feedback_trust_model",
    "load_sequence_feedback_failure_manifest",
    "load_sequence_feedback_trust_committee",
    "load_sequence_feedback_trust_model",
]
