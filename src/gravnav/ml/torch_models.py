"""
Optional torch-based learned localizer models.

This module is intentionally import-safe when torch is unavailable. The rest of
the repository can continue using the NumPy reference path without a hard
dependency on torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import RealOceanCorpus
from .models import RuntimeStudentModel, RuntimeStudentModelSpec, summarize_query_windows

FloatArray = NDArray[np.float64]

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised when torch is absent
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_TorchModuleBase = nn.Module if nn is not None else object


def torch_is_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised when torch is absent
        raise ImportError(
            "torch is required for the torch learned-localizer backend. "
            "Install it in the ML virtual environment and rerun."
        )


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _stable_std(x: FloatArray) -> FloatArray:
    std = np.std(np.asarray(x, dtype=np.float64), axis=0)
    return np.maximum(std, 1.0e-9)


def _softmax_rows(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    xmax = np.max(arr, axis=-1, keepdims=True)
    ex = np.exp(np.clip(arr - xmax, -700.0, 0.0))
    denom = np.maximum(np.sum(ex, axis=-1, keepdims=True), 1.0e-300)
    return ex / denom


def _to_torch(x: ArrayLike, *, device: str) -> Tensor:
    _require_torch()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


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


@dataclass
class TorchRuntimeStudentModelSpec:
    candidate_feature_names: tuple[str, ...]
    query_feature_names: tuple[str, ...]
    embedding_dim: int = 64
    query_hidden_dim: int = 128
    candidate_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    head_hidden_dim: int = 32
    dropout_prob: float = 0.0
    target_parameter_count: int = 48_000_000
    target_parameter_count_range: tuple[int, int] = (10_000_000, 60_000_000)
    target_reliability_head_parameter_count: int = 3_000_000
    name: str = "runtime_student_torch"


@dataclass
class TorchDelayedLocalizerTrainingSpec:
    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    offset_loss_weight: float = 0.25
    label_smoothing: float = 0.0
    validation_fraction: float = 0.2
    min_validation_examples_per_region: int = 1
    min_train_examples_per_region: int = 1
    min_epochs: int = 10
    early_stopping_patience: int = 12
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    head_epochs: int = 40
    head_learning_rate: float = 5.0e-4
    reliability_threshold: float = 0.65
    analytic_log_emission_gain: float = 0.35
    device: str = "cpu"
    random_seed: int = 42
    name: str = "torch_delayed_localizer_training"


class _TorchDelayedLocalizerNet(_TorchModuleBase):  # type: ignore[misc]
    def __init__(
        self,
        *,
        spec: TorchRuntimeStudentModelSpec,
        query_summary_dim: int,
        candidate_dim: int,
        query_mean: FloatArray,
        query_std: FloatArray,
        candidate_mean: FloatArray,
        candidate_std: FloatArray,
        analytic_log_emission_gain: float,
    ) -> None:
        _require_torch()
        super().__init__()
        self.spec = spec
        self.query_summary_dim = int(query_summary_dim)
        self.candidate_dim = int(candidate_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.query_summary_dim, int(spec.query_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(spec.dropout_prob)),
            nn.Linear(int(spec.query_hidden_dim), int(spec.embedding_dim)),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim, int(spec.candidate_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(spec.dropout_prob)),
            nn.Linear(int(spec.candidate_hidden_dim), int(spec.embedding_dim)),
        )
        fusion_dim = int(spec.embedding_dim) * 3 + 3
        self.score_head = nn.Sequential(
            nn.Linear(fusion_dim, int(spec.fusion_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(spec.dropout_prob)),
            nn.Linear(int(spec.fusion_hidden_dim), 1),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(8, int(spec.head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(spec.dropout_prob)),
            nn.Linear(int(spec.head_hidden_dim), 1),
        )
        self.covariance_head = nn.Sequential(
            nn.Linear(6, int(spec.head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(spec.dropout_prob)),
            nn.Linear(int(spec.head_hidden_dim), 1),
        )
        self.analytic_gain = nn.Parameter(
            torch.tensor(float(analytic_log_emission_gain), dtype=torch.float32)
        )
        self.register_buffer("query_mean", _to_torch(query_mean, device="cpu"))
        self.register_buffer("query_std", _to_torch(query_std, device="cpu"))
        self.register_buffer("candidate_mean", _to_torch(candidate_mean, device="cpu"))
        self.register_buffer("candidate_std", _to_torch(candidate_std, device="cpu"))
        self.register_buffer("reliability_mean", torch.zeros(8, dtype=torch.float32))
        self.register_buffer("reliability_std", torch.ones(8, dtype=torch.float32))
        self.register_buffer("covariance_mean", torch.zeros(6, dtype=torch.float32))
        self.register_buffer("covariance_std", torch.ones(6, dtype=torch.float32))

    @staticmethod
    def summarize_query_windows_torch(query_windows: Tensor) -> Tensor:
        last = query_windows[:, -1, :]
        mean = torch.mean(query_windows, dim=1)
        std = torch.std(query_windows, dim=1, correction=0)
        if query_windows.shape[1] > 1:
            slope = (query_windows[:, -1, :] - query_windows[:, 0, :]) / float(
                query_windows.shape[1] - 1
            )
        else:
            slope = torch.zeros_like(last)
        return torch.cat([last, mean, std, slope], dim=1)

    def score_candidates(
        self,
        *,
        query_windows: Tensor,
        candidate_features: Tensor,
        candidate_offsets_ned_m: Tensor,
        analytic_log_emission: Optional[Tensor] = None,
    ) -> Tensor:
        q_summary = self.summarize_query_windows_torch(query_windows)
        q_norm = (q_summary - self.query_mean) / self.query_std
        c_norm = (candidate_features - self.candidate_mean) / self.candidate_std
        q_emb = self.query_encoder(q_norm)
        c_emb = self.candidate_encoder(c_norm)
        q_expand = q_emb.unsqueeze(1).expand(-1, c_emb.shape[1], -1)
        analytic_feature = (
            torch.zeros_like(candidate_offsets_ned_m[..., :1])
            if analytic_log_emission is None
            else analytic_log_emission.unsqueeze(-1)
        )
        fusion = torch.cat(
            [
                q_expand,
                c_emb,
                q_expand * c_emb,
                candidate_offsets_ned_m[..., :2],
                analytic_feature,
            ],
            dim=-1,
        )
        scores = self.score_head(fusion).squeeze(-1)
        if analytic_log_emission is not None:
            scores = scores + self.analytic_gain * analytic_log_emission
        return scores

    def set_head_feature_stats(
        self,
        *,
        reliability_features: FloatArray,
        covariance_features: FloatArray,
    ) -> None:
        rel_mean = np.mean(reliability_features, axis=0)
        rel_std = _stable_std(reliability_features)
        cov_mean = np.mean(covariance_features, axis=0)
        cov_std = _stable_std(covariance_features)
        with torch.no_grad():
            self.reliability_mean.copy_(_to_torch(rel_mean, device="cpu"))
            self.reliability_std.copy_(_to_torch(rel_std, device="cpu"))
            self.covariance_mean.copy_(_to_torch(cov_mean, device="cpu"))
            self.covariance_std.copy_(_to_torch(cov_std, device="cpu"))

    def reliability_logits(self, features: Tensor) -> Tensor:
        norm = (features - self.reliability_mean) / self.reliability_std
        return self.reliability_head(norm).squeeze(-1)

    def covariance_log_scale(self, features: Tensor) -> Tensor:
        norm = (features - self.covariance_mean) / self.covariance_std
        return self.covariance_head(norm).squeeze(-1)


@dataclass
class TorchRuntimeStudentModel:
    spec: TorchRuntimeStudentModelSpec
    net: _TorchDelayedLocalizerNet
    reliability_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    @property
    def reference_parameter_count(self) -> int:
        _require_torch()
        return int(sum(int(p.numel()) for p in self.net.parameters()))

    def predict_candidate_scores(
        self,
        *,
        query_windows: FloatArray,
        candidate_features: FloatArray,
        candidate_offsets_ned_m: FloatArray,
        analytic_log_emission: Optional[FloatArray] = None,
    ) -> FloatArray:
        _require_torch()
        with torch.no_grad():
            scores = self.net.score_candidates(
                query_windows=_to_torch(query_windows, device=self.device),
                candidate_features=_to_torch(candidate_features, device=self.device),
                candidate_offsets_ned_m=_to_torch(
                    candidate_offsets_ned_m,
                    device=self.device,
                ),
                analytic_log_emission=(
                    None
                    if analytic_log_emission is None
                    else _to_torch(analytic_log_emission, device=self.device)
                ),
            )
        return np.asarray(scores.detach().cpu().numpy(), dtype=np.float64)

    @staticmethod
    def posterior_from_scores(scores: FloatArray) -> FloatArray:
        return _softmax_rows(scores)

    def covariance_scale_from_features(
        self,
        covariance_features: FloatArray,
    ) -> FloatArray:
        _require_torch()
        with torch.no_grad():
            log_scale = self.net.covariance_log_scale(
                _to_torch(covariance_features, device=self.device)
            )
            scale = torch.exp(torch.clamp(log_scale, -2.0, 2.0))
        return np.asarray(scale.detach().cpu().numpy(), dtype=np.float64)

    def publishability_probability_from_features(
        self,
        reliability_features: FloatArray,
    ) -> FloatArray:
        _require_torch()
        with torch.no_grad():
            logits = self.net.reliability_logits(
                _to_torch(reliability_features, device=self.device)
            )
            probs = torch.sigmoid(logits)
        return np.asarray(probs.detach().cpu().numpy(), dtype=np.float64)

    def save_pt(self, path: str | Path) -> Path:
        _require_torch()
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spec": _jsonable(asdict(self.spec)),
            "state_dict": self.net.state_dict(),
            "reliability_threshold": float(self.reliability_threshold),
            "metadata": _jsonable(self.metadata),
        }
        torch.save(payload, p)
        return p

    @classmethod
    def from_pt(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "TorchRuntimeStudentModel":
        _require_torch()
        p = Path(path).expanduser().resolve()
        payload = torch.load(p, map_location=device)
        spec = TorchRuntimeStudentModelSpec(**payload["spec"])
        state_dict = payload["state_dict"]
        query_mean = np.asarray(state_dict["query_mean"].cpu().numpy(), dtype=np.float64)
        query_std = np.asarray(state_dict["query_std"].cpu().numpy(), dtype=np.float64)
        candidate_mean = np.asarray(
            state_dict["candidate_mean"].cpu().numpy(),
            dtype=np.float64,
        )
        candidate_std = np.asarray(
            state_dict["candidate_std"].cpu().numpy(),
            dtype=np.float64,
        )
        net = _TorchDelayedLocalizerNet(
            spec=spec,
            query_summary_dim=int(query_mean.size),
            candidate_dim=int(candidate_mean.size),
            query_mean=query_mean,
            query_std=query_std,
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            analytic_log_emission_gain=float(
                state_dict["analytic_gain"].detach().cpu().numpy().item()
            ),
        )
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        return cls(
            spec=spec,
            net=net,
            reliability_threshold=float(payload["reliability_threshold"]),
            metadata=dict(payload.get("metadata", {})),
            device=str(device),
        )


def _posterior_feature_arrays(
    corpus: RealOceanCorpus,
    posterior: FloatArray,
    scores: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    peak = np.max(posterior, axis=1)
    entropy = -np.sum(posterior * np.log(np.maximum(posterior, 1.0e-300)), axis=1)
    expected_offsets = np.sum(
        posterior[:, :, None] * corpus.candidate_offsets_ned_m[..., :2],
        axis=1,
    )
    horizontal_error = np.linalg.norm(
        expected_offsets - corpus.truth_offsets_ned_m[:, :2],
        axis=1,
    )
    reliability_features = np.column_stack(
        [
            peak,
            entropy,
            horizontal_error,
            np.mean(np.abs(corpus.query_windows[:, -1, :]), axis=1),
            np.std(corpus.query_windows[:, -1, :], axis=1),
            np.max(np.abs(corpus.truth_offsets_ned_m[:, :2]), axis=1),
            corpus.covariance_targets,
            np.mean(np.abs(corpus.candidate_offsets_ned_m[..., :2]), axis=(1, 2)),
        ]
    ).astype(np.float64)
    covariance_features = np.column_stack(
        [
            peak,
            entropy,
            horizontal_error,
            corpus.covariance_targets,
            np.mean(np.abs(corpus.truth_offsets_ned_m[:, :2]), axis=1),
            np.std(scores, axis=1),
        ]
    ).astype(np.float64)
    return reliability_features, covariance_features


def _stratified_validation_indices(
    corpus: RealOceanCorpus,
    *,
    validation_fraction: float,
    min_validation_examples_per_region: int,
    min_train_examples_per_region: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if corpus.num_examples < 2 or validation_fraction <= 0.0:
        full = np.arange(corpus.num_examples, dtype=np.int64)
        return full, np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    train_parts: list[NDArray[np.int64]] = []
    val_parts: list[NDArray[np.int64]] = []
    for region_idx, _ in enumerate(corpus.region_names):
        region_indices = np.flatnonzero(corpus.region_index == region_idx).astype(np.int64)
        if region_indices.size == 0:
            continue
        shuffled = region_indices.copy()
        rng.shuffle(shuffled)
        desired_val = int(round(float(validation_fraction) * float(shuffled.size)))
        desired_val = max(int(min_validation_examples_per_region), desired_val)
        max_val = max(0, int(shuffled.size) - int(min_train_examples_per_region))
        val_count = min(max_val, desired_val)
        if val_count <= 0:
            train_parts.append(np.sort(shuffled))
            continue
        val_parts.append(np.sort(shuffled[:val_count]))
        train_parts.append(np.sort(shuffled[val_count:]))

    train_idx = (
        np.sort(np.concatenate(train_parts)).astype(np.int64)
        if len(train_parts) > 0
        else np.empty(0, dtype=np.int64)
    )
    val_idx = (
        np.sort(np.concatenate(val_parts)).astype(np.int64)
        if len(val_parts) > 0
        else np.empty(0, dtype=np.int64)
    )
    if train_idx.size == 0:
        full = np.arange(corpus.num_examples, dtype=np.int64)
        return full, np.empty(0, dtype=np.int64)
    return train_idx, val_idx


def _candidate_loss_and_offsets(
    net: _TorchDelayedLocalizerNet,
    *,
    query_windows: Tensor,
    candidate_features: Tensor,
    candidate_offsets: Tensor,
    analytic: Tensor,
    labels: Tensor,
    truth_offsets: Tensor,
    offset_loss_weight: float,
    label_smoothing: float,
) -> tuple[Tensor, Tensor, Tensor]:
    logits = net.score_candidates(
        query_windows=query_windows,
        candidate_features=candidate_features,
        candidate_offsets_ned_m=candidate_offsets,
        analytic_log_emission=analytic,
    )
    posterior = torch.softmax(logits, dim=1)
    expected_offsets = torch.sum(
        posterior.unsqueeze(-1) * candidate_offsets[:, :, :2],
        dim=1,
    )
    ce_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=float(label_smoothing),
    )
    offset_loss = F.smooth_l1_loss(expected_offsets, truth_offsets[:, :2])
    loss = ce_loss + float(offset_loss_weight) * offset_loss
    return loss, logits, expected_offsets


def train_torch_delayed_localizer(
    corpus: RealOceanCorpus,
    *,
    spec: TorchDelayedLocalizerTrainingSpec | None = None,
    model_spec: TorchRuntimeStudentModelSpec | None = None,
) -> TorchRuntimeStudentModel:
    _require_torch()
    train_spec = (
        TorchDelayedLocalizerTrainingSpec()
        if spec is None
        else spec
    )
    np.random.seed(int(train_spec.random_seed))
    torch.manual_seed(int(train_spec.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(train_spec.random_seed))
    runtime_spec = (
        TorchRuntimeStudentModelSpec(
            candidate_feature_names=tuple(corpus.metadata["candidate_feature_names"]),
            query_feature_names=tuple(corpus.metadata["query_feature_names"]),
        )
        if model_spec is None
        else model_spec
    )
    query_summaries = summarize_query_windows(corpus.query_windows)
    net = _TorchDelayedLocalizerNet(
        spec=runtime_spec,
        query_summary_dim=int(query_summaries.shape[1]),
        candidate_dim=int(corpus.candidate_features.shape[2]),
        query_mean=np.mean(query_summaries, axis=0),
        query_std=_stable_std(query_summaries),
        candidate_mean=np.mean(corpus.candidate_features.reshape(-1, corpus.candidate_features.shape[-1]), axis=0),
        candidate_std=_stable_std(
            corpus.candidate_features.reshape(-1, corpus.candidate_features.shape[-1])
        ),
        analytic_log_emission_gain=float(train_spec.analytic_log_emission_gain),
    )
    device = str(train_spec.device)
    net.to(device)

    query_windows = _to_torch(corpus.query_windows, device=device)
    candidate_features = _to_torch(corpus.candidate_features, device=device)
    candidate_offsets = _to_torch(corpus.candidate_offsets_ned_m, device=device)
    analytic = _to_torch(corpus.analytic_log_emission, device=device)
    labels = torch.as_tensor(corpus.labels, dtype=torch.long, device=device)
    truth_offsets = _to_torch(corpus.truth_offsets_ned_m, device=device)
    train_indices_np, val_indices_np = _stratified_validation_indices(
        corpus,
        validation_fraction=float(train_spec.validation_fraction),
        min_validation_examples_per_region=int(
            train_spec.min_validation_examples_per_region
        ),
        min_train_examples_per_region=int(train_spec.min_train_examples_per_region),
        seed=int(train_spec.random_seed),
    )
    train_indices = torch.as_tensor(train_indices_np, dtype=torch.long, device=device)
    val_indices = torch.as_tensor(val_indices_np, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(train_spec.learning_rate),
        weight_decay=float(train_spec.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_spec.lr_scheduler_factor),
        patience=int(train_spec.lr_scheduler_patience),
        min_lr=float(train_spec.min_learning_rate),
    )
    num_examples = int(corpus.num_examples)
    batch_size = max(1, int(train_spec.batch_size))
    best_metric = float("inf")
    best_epoch = -1
    best_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in net.state_dict().items()
    }
    patience_counter = 0
    train_count = int(train_indices.numel())
    for epoch in range(int(train_spec.epochs)):
        net.train()
        perm = train_indices[torch.randperm(train_count, device=device)]
        for start in range(0, train_count, batch_size):
            batch = perm[start : start + batch_size]
            loss, _, _ = _candidate_loss_and_offsets(
                net,
                query_windows=query_windows[batch],
                candidate_features=candidate_features[batch],
                candidate_offsets=candidate_offsets[batch],
                analytic=analytic[batch],
                labels=labels[batch],
                truth_offsets=truth_offsets[batch],
                offset_loss_weight=float(train_spec.offset_loss_weight),
                label_smoothing=float(train_spec.label_smoothing),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(train_spec.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=float(train_spec.gradient_clip_norm),
                )
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_loss, _, _ = _candidate_loss_and_offsets(
                net,
                query_windows=query_windows[train_indices],
                candidate_features=candidate_features[train_indices],
                candidate_offsets=candidate_offsets[train_indices],
                analytic=analytic[train_indices],
                labels=labels[train_indices],
                truth_offsets=truth_offsets[train_indices],
                offset_loss_weight=float(train_spec.offset_loss_weight),
                label_smoothing=float(train_spec.label_smoothing),
            )
            if val_indices.numel() > 0:
                val_loss, _, _ = _candidate_loss_and_offsets(
                    net,
                    query_windows=query_windows[val_indices],
                    candidate_features=candidate_features[val_indices],
                    candidate_offsets=candidate_offsets[val_indices],
                    analytic=analytic[val_indices],
                    labels=labels[val_indices],
                    truth_offsets=truth_offsets[val_indices],
                    offset_loss_weight=float(train_spec.offset_loss_weight),
                    label_smoothing=float(train_spec.label_smoothing),
                )
                monitor_metric = float(val_loss.detach().cpu().item())
            else:
                monitor_metric = float(train_loss.detach().cpu().item())
        scheduler.step(monitor_metric)
        if monitor_metric + 1.0e-8 < best_metric:
            best_metric = monitor_metric
            best_epoch = int(epoch)
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        if (
            epoch + 1 >= int(train_spec.min_epochs)
            and patience_counter >= int(train_spec.early_stopping_patience)
        ):
            break

    net.load_state_dict(best_state_dict)
    net.to(device)

    net.eval()
    with torch.no_grad():
        train_scores = net.score_candidates(
            query_windows=query_windows,
            candidate_features=candidate_features,
            candidate_offsets_ned_m=candidate_offsets,
            analytic_log_emission=analytic,
        )
    scores_np = np.asarray(train_scores.detach().cpu().numpy(), dtype=np.float64)
    posterior_np = _softmax_rows(scores_np)
    reliability_features, covariance_features = _posterior_feature_arrays(
        corpus,
        posterior_np,
        scores_np,
    )
    net.set_head_feature_stats(
        reliability_features=reliability_features,
        covariance_features=covariance_features,
    )

    rel_x = _to_torch(reliability_features, device=device)
    rel_y = _to_torch(corpus.publishability_labels.astype(np.float64), device=device)
    cov_x = _to_torch(covariance_features, device=device)
    cov_y = _to_torch(np.log(np.maximum(corpus.covariance_targets, 1.0e-6)), device=device)

    rel_optimizer = torch.optim.AdamW(
        net.reliability_head.parameters(),
        lr=float(train_spec.head_learning_rate),
        weight_decay=float(train_spec.weight_decay),
    )
    cov_optimizer = torch.optim.AdamW(
        net.covariance_head.parameters(),
        lr=float(train_spec.head_learning_rate),
        weight_decay=float(train_spec.weight_decay),
    )
    for _ in range(int(train_spec.head_epochs)):
        rel_logits = net.reliability_logits(rel_x)
        rel_loss = F.binary_cross_entropy_with_logits(rel_logits, rel_y)
        rel_optimizer.zero_grad(set_to_none=True)
        rel_loss.backward()
        rel_optimizer.step()

        cov_pred = net.covariance_log_scale(cov_x)
        cov_loss = F.mse_loss(cov_pred, cov_y)
        cov_optimizer.zero_grad(set_to_none=True)
        cov_loss.backward()
        cov_optimizer.step()

    net.eval()
    model = TorchRuntimeStudentModel(
        spec=runtime_spec,
        net=net,
        reliability_threshold=float(train_spec.reliability_threshold),
        metadata={
            "training_name": train_spec.name,
            "device": device,
            "num_examples": num_examples,
            "random_seed": int(train_spec.random_seed),
            "train_examples": int(train_indices.numel()),
            "validation_examples": int(val_indices.numel()),
            "best_epoch": int(best_epoch),
            "best_monitor_metric": float(best_metric),
        },
        device=device,
    )
    return model


def load_runtime_localizer_model(path: str | Path) -> RuntimeStudentModel | TorchRuntimeStudentModel:
    p = Path(path).expanduser().resolve()
    if p.suffix == ".npz":
        return RuntimeStudentModel.from_npz(p)
    if p.suffix in {".pt", ".pth"}:
        return TorchRuntimeStudentModel.from_pt(p)
    raise ValueError(
        f"Unsupported learned-localizer model format {p.suffix!r}; expected .npz or .pt/.pth."
    )
