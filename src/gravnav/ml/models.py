"""
Compact reference models for learned Earth-signature localization.

These implementations are intentionally NumPy-only so the repository can ship
an end-to-end learned-localizer path without introducing a hard deep-learning
dependency. The large-model parameter targets remain product goals; the classes
here are lightweight reference implementations that preserve the same runtime
contract and export surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


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


def _softmax_rows(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    xmax = np.max(arr, axis=-1, keepdims=True)
    ex = np.exp(np.clip(arr - xmax, -700.0, 0.0))
    denom = np.sum(ex, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1.0e-300)
    return ex / denom


def _stable_std(x: FloatArray) -> FloatArray:
    std = np.std(np.asarray(x, dtype=np.float64), axis=0)
    return np.maximum(std, 1.0e-9)


def _ridge_regression(
    X: FloatArray,
    Y: FloatArray,
    *,
    l2: float,
) -> FloatArray:
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    gram = x.T @ x + float(l2) * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, x.T @ y).astype(np.float64)


def _conv_patch_stem(
    patches: FloatArray,
) -> FloatArray:
    """
    Small deterministic convolution-like stem.

    The stem extracts local means and first differences from each channel. It is
    cheap to evaluate and preserves enough local structure for PCA-based latent
    fitting on the real public patch tensors used by the corpus builder.
    """
    p = np.asarray(patches, dtype=np.float64)
    if p.ndim != 4:
        raise ValueError("patches must have shape (N, H, W, C).")
    north = 0.5 * (
        np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)
    )
    east = 0.5 * (
        np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)
    )
    smooth = (
        p
        + np.roll(p, 1, axis=1)
        + np.roll(p, -1, axis=1)
        + np.roll(p, 1, axis=2)
        + np.roll(p, -1, axis=2)
    ) / 5.0
    return np.concatenate([p, smooth, north, east], axis=-1).astype(np.float64)


def summarize_query_windows(query_windows: FloatArray) -> FloatArray:
    """
    Convert delayed measurement windows into compact query vectors.
    """
    q = np.asarray(query_windows, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("query_windows must have shape (N, T, F).")
    last = q[:, -1, :]
    mean = np.mean(q, axis=1)
    std = np.std(q, axis=1)
    if q.shape[1] > 1:
        slope = (q[:, -1, :] - q[:, 0, :]) / float(q.shape[1] - 1)
    else:
        slope = np.zeros_like(last)
    return np.concatenate([last, mean, std, slope], axis=1).astype(np.float64)


@dataclass
class OceanTeacherModelSpec:
    patch_channels: tuple[str, ...]
    latent_dim: int = 48
    num_latents: int = 16
    target_parameter_count: int = 320_000_000
    target_parameter_count_range: tuple[int, int] = (250_000_000, 400_000_000)
    name: str = "ocean_teacher_reference"


@dataclass
class OceanTeacherModel:
    spec: OceanTeacherModelSpec
    input_mean: FloatArray
    input_std: FloatArray
    latent_projection: FloatArray
    latent_tokens: FloatArray
    reconstruction_projection: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_parameter_count(self) -> int:
        count = 0
        for arr in (
            self.input_mean,
            self.input_std,
            self.latent_projection,
            self.latent_tokens,
            self.reconstruction_projection,
        ):
            count += int(np.asarray(arr).size)
        return int(count)

    def _prepare(self, patch_tensors: FloatArray) -> FloatArray:
        stem = _conv_patch_stem(patch_tensors)
        flat = stem.reshape(stem.shape[0], -1)
        return ((flat - self.input_mean) / self.input_std).astype(np.float64)

    def encode_patches(self, patch_tensors: FloatArray) -> FloatArray:
        x = self._prepare(patch_tensors)
        latents = x @ self.latent_projection
        return np.tanh(latents + np.mean(self.latent_tokens, axis=0, keepdims=True))

    def reconstruct_stem(self, patch_tensors: FloatArray) -> FloatArray:
        z = self.encode_patches(patch_tensors)
        return (z @ self.reconstruction_projection).astype(np.float64)

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            spec_json=np.array(json.dumps(_jsonable(asdict(self.spec)))),
            input_mean=np.asarray(self.input_mean, dtype=np.float64),
            input_std=np.asarray(self.input_std, dtype=np.float64),
            latent_projection=np.asarray(self.latent_projection, dtype=np.float64),
            latent_tokens=np.asarray(self.latent_tokens, dtype=np.float64),
            reconstruction_projection=np.asarray(
                self.reconstruction_projection,
                dtype=np.float64,
            ),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "OceanTeacherModel":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            spec = OceanTeacherModelSpec(**json.loads(str(data["spec_json"].item())))
            return cls(
                spec=spec,
                input_mean=np.asarray(data["input_mean"], dtype=np.float64),
                input_std=np.asarray(data["input_std"], dtype=np.float64),
                latent_projection=np.asarray(
                    data["latent_projection"],
                    dtype=np.float64,
                ),
                latent_tokens=np.asarray(data["latent_tokens"], dtype=np.float64),
                reconstruction_projection=np.asarray(
                    data["reconstruction_projection"],
                    dtype=np.float64,
                ),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


@dataclass
class RuntimeStudentModelSpec:
    candidate_feature_names: tuple[str, ...]
    query_feature_names: tuple[str, ...]
    embedding_dim: int = 32
    target_parameter_count: int = 48_000_000
    target_parameter_count_range: tuple[int, int] = (35_000_000, 60_000_000)
    target_reliability_head_parameter_count: int = 3_000_000
    name: str = "runtime_student_reference"


@dataclass
class RuntimeStudentModel:
    spec: RuntimeStudentModelSpec
    candidate_mean: FloatArray
    candidate_std: FloatArray
    candidate_projection: FloatArray
    query_mean: FloatArray
    query_std: FloatArray
    query_projection: FloatArray
    offset_linear: FloatArray
    analytic_log_emission_gain: float
    covariance_features_mean: FloatArray
    covariance_features_std: FloatArray
    covariance_weights: FloatArray
    covariance_bias: float
    reliability_features_mean: FloatArray
    reliability_features_std: FloatArray
    reliability_weights: FloatArray
    reliability_bias: float
    reliability_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_parameter_count(self) -> int:
        count = 0
        for arr in (
            self.candidate_mean,
            self.candidate_std,
            self.candidate_projection,
            self.query_mean,
            self.query_std,
            self.query_projection,
            self.offset_linear,
            self.covariance_features_mean,
            self.covariance_features_std,
            self.covariance_weights,
            self.reliability_features_mean,
            self.reliability_features_std,
            self.reliability_weights,
        ):
            count += int(np.asarray(arr).size)
        return int(count + 4)

    def encode_candidate_features(self, candidate_features: FloatArray) -> FloatArray:
        c = np.asarray(candidate_features, dtype=np.float64)
        c_norm = (c - self.candidate_mean) / self.candidate_std
        return np.tanh(c_norm @ self.candidate_projection).astype(np.float64)

    def encode_query_windows(self, query_windows: FloatArray) -> FloatArray:
        summary = summarize_query_windows(query_windows)
        q_norm = (summary - self.query_mean) / self.query_std
        return np.tanh(q_norm @ self.query_projection).astype(np.float64)

    @staticmethod
    def _sigmoid(x: ArrayLike) -> FloatArray:
        arr = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))

    def predict_candidate_scores(
        self,
        *,
        query_windows: FloatArray,
        candidate_features: FloatArray,
        candidate_offsets_ned_m: FloatArray,
        analytic_log_emission: Optional[FloatArray] = None,
    ) -> FloatArray:
        q_emb = self.encode_query_windows(query_windows)
        c_emb = self.encode_candidate_features(candidate_features)
        if q_emb.ndim == 1:
            q_emb = q_emb[None, :]
        if c_emb.ndim == 2:
            c_emb = c_emb[None, :, :]
        if c_emb.shape[0] != q_emb.shape[0]:
            raise ValueError("query_windows and candidate_features batch sizes differ.")
        scores = np.einsum("bd,bcd->bc", q_emb, c_emb) / np.sqrt(
            max(c_emb.shape[-1], 1)
        )
        offsets = np.asarray(candidate_offsets_ned_m, dtype=np.float64)[..., :2]
        scores += np.tensordot(offsets, self.offset_linear, axes=([-1], [0]))
        if analytic_log_emission is not None:
            scores += float(self.analytic_log_emission_gain) * np.asarray(
                analytic_log_emission,
                dtype=np.float64,
            )
        return scores.astype(np.float64)

    def posterior_from_scores(self, scores: FloatArray) -> FloatArray:
        return _softmax_rows(np.asarray(scores, dtype=np.float64))

    def covariance_scale_from_features(
        self,
        covariance_features: FloatArray,
    ) -> FloatArray:
        feats = np.asarray(covariance_features, dtype=np.float64)
        norm = (feats - self.covariance_features_mean) / self.covariance_features_std
        raw = norm @ self.covariance_weights + float(self.covariance_bias)
        return np.exp(np.clip(raw, -2.0, 2.0)).astype(np.float64)

    def publishability_probability_from_features(
        self,
        reliability_features: FloatArray,
    ) -> FloatArray:
        feats = np.asarray(reliability_features, dtype=np.float64)
        norm = (feats - self.reliability_features_mean) / self.reliability_features_std
        raw = norm @ self.reliability_weights + float(self.reliability_bias)
        return self._sigmoid(raw).astype(np.float64)

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            spec_json=np.array(json.dumps(_jsonable(asdict(self.spec)))),
            candidate_mean=np.asarray(self.candidate_mean, dtype=np.float64),
            candidate_std=np.asarray(self.candidate_std, dtype=np.float64),
            candidate_projection=np.asarray(
                self.candidate_projection,
                dtype=np.float64,
            ),
            query_mean=np.asarray(self.query_mean, dtype=np.float64),
            query_std=np.asarray(self.query_std, dtype=np.float64),
            query_projection=np.asarray(self.query_projection, dtype=np.float64),
            offset_linear=np.asarray(self.offset_linear, dtype=np.float64),
            analytic_log_emission_gain=np.array(
                float(self.analytic_log_emission_gain),
                dtype=np.float64,
            ),
            covariance_features_mean=np.asarray(
                self.covariance_features_mean,
                dtype=np.float64,
            ),
            covariance_features_std=np.asarray(
                self.covariance_features_std,
                dtype=np.float64,
            ),
            covariance_weights=np.asarray(self.covariance_weights, dtype=np.float64),
            covariance_bias=np.array(float(self.covariance_bias), dtype=np.float64),
            reliability_features_mean=np.asarray(
                self.reliability_features_mean,
                dtype=np.float64,
            ),
            reliability_features_std=np.asarray(
                self.reliability_features_std,
                dtype=np.float64,
            ),
            reliability_weights=np.asarray(
                self.reliability_weights,
                dtype=np.float64,
            ),
            reliability_bias=np.array(float(self.reliability_bias), dtype=np.float64),
            reliability_threshold=np.array(
                float(self.reliability_threshold),
                dtype=np.float64,
            ),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "RuntimeStudentModel":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            spec = RuntimeStudentModelSpec(
                **json.loads(str(data["spec_json"].item()))
            )
            return cls(
                spec=spec,
                candidate_mean=np.asarray(data["candidate_mean"], dtype=np.float64),
                candidate_std=np.asarray(data["candidate_std"], dtype=np.float64),
                candidate_projection=np.asarray(
                    data["candidate_projection"],
                    dtype=np.float64,
                ),
                query_mean=np.asarray(data["query_mean"], dtype=np.float64),
                query_std=np.asarray(data["query_std"], dtype=np.float64),
                query_projection=np.asarray(
                    data["query_projection"],
                    dtype=np.float64,
                ),
                offset_linear=np.asarray(data["offset_linear"], dtype=np.float64),
                analytic_log_emission_gain=float(
                    data["analytic_log_emission_gain"]
                ),
                covariance_features_mean=np.asarray(
                    data["covariance_features_mean"],
                    dtype=np.float64,
                ),
                covariance_features_std=np.asarray(
                    data["covariance_features_std"],
                    dtype=np.float64,
                ),
                covariance_weights=np.asarray(
                    data["covariance_weights"],
                    dtype=np.float64,
                ),
                covariance_bias=float(data["covariance_bias"]),
                reliability_features_mean=np.asarray(
                    data["reliability_features_mean"],
                    dtype=np.float64,
                ),
                reliability_features_std=np.asarray(
                    data["reliability_features_std"],
                    dtype=np.float64,
                ),
                reliability_weights=np.asarray(
                    data["reliability_weights"],
                    dtype=np.float64,
                ),
                reliability_bias=float(data["reliability_bias"]),
                reliability_threshold=float(data["reliability_threshold"]),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def fit_teacher_from_patches(
    patch_tensors: FloatArray,
    *,
    spec: OceanTeacherModelSpec,
) -> OceanTeacherModel:
    stem = _conv_patch_stem(np.asarray(patch_tensors, dtype=np.float64))
    flat = stem.reshape(stem.shape[0], -1)
    mean = np.mean(flat, axis=0)
    std = _stable_std(flat)
    x = (flat - mean) / std
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    latent_dim = min(int(spec.latent_dim), int(vt.shape[0]))
    projection = vt[:latent_dim].T.astype(np.float64)
    latents = x @ projection
    recon = _ridge_regression(latents, x, l2=1.0e-3)
    latent_tokens = np.repeat(
        np.mean(latents, axis=0, keepdims=True),
        max(int(spec.num_latents), 1),
        axis=0,
    )
    return OceanTeacherModel(
        spec=spec,
        input_mean=mean.astype(np.float64),
        input_std=std.astype(np.float64),
        latent_projection=projection,
        latent_tokens=latent_tokens.astype(np.float64),
        reconstruction_projection=recon.astype(np.float64),
        metadata={
            "fit_method": "svd_latent_projection",
            "reference_parameter_count": int(
                mean.size + std.size + projection.size + latent_tokens.size + recon.size
            ),
        },
    )


def fit_runtime_student_initialization(
    *,
    patch_summary_features: FloatArray,
    teacher_embeddings: FloatArray,
    query_summaries: FloatArray,
    candidate_feature_names: tuple[str, ...],
    query_feature_names: tuple[str, ...],
    embedding_dim: int,
    analytic_log_emission_gain: float = 0.35,
) -> RuntimeStudentModel:
    patch_summary_full = np.asarray(patch_summary_features, dtype=np.float64)
    candidate_dim = int(len(candidate_feature_names))
    if patch_summary_full.shape[1] < candidate_dim:
        raise ValueError(
            "patch_summary_features do not cover the runtime candidate feature set."
        )
    patch_summary = patch_summary_full[:, :candidate_dim]
    teacher_emb = np.asarray(teacher_embeddings, dtype=np.float64)
    teacher_dim = min(int(embedding_dim), int(teacher_emb.shape[1]))
    target = teacher_emb[:, :teacher_dim]

    cand_mean = np.mean(patch_summary, axis=0)
    cand_std = _stable_std(patch_summary)
    cand_proj = _ridge_regression(
        (patch_summary - cand_mean) / cand_std,
        target,
        l2=1.0e-3,
    )

    query = np.asarray(query_summaries, dtype=np.float64)
    query_mean = np.mean(query, axis=0)
    query_std = _stable_std(query)
    q_centered = (query - query_mean) / query_std
    _, _, vt = np.linalg.svd(q_centered, full_matrices=False)
    query_projection = vt[:teacher_dim].T.astype(np.float64)

    return RuntimeStudentModel(
        spec=RuntimeStudentModelSpec(
            candidate_feature_names=tuple(candidate_feature_names),
            query_feature_names=tuple(query_feature_names),
            embedding_dim=teacher_dim,
        ),
        candidate_mean=cand_mean.astype(np.float64),
        candidate_std=cand_std.astype(np.float64),
        candidate_projection=cand_proj.astype(np.float64),
        query_mean=query_mean.astype(np.float64),
        query_std=query_std.astype(np.float64),
        query_projection=query_projection,
        offset_linear=np.zeros(2, dtype=np.float64),
        analytic_log_emission_gain=float(analytic_log_emission_gain),
        covariance_features_mean=np.zeros(6, dtype=np.float64),
        covariance_features_std=np.ones(6, dtype=np.float64),
        covariance_weights=np.zeros(6, dtype=np.float64),
        covariance_bias=0.0,
        reliability_features_mean=np.zeros(8, dtype=np.float64),
        reliability_features_std=np.ones(8, dtype=np.float64),
        reliability_weights=np.zeros(8, dtype=np.float64),
        reliability_bias=0.0,
        reliability_threshold=0.65,
        metadata={"fit_method": "ridge_plus_svd_initialization"},
    )
