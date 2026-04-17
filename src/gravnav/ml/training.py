"""
Training and evaluation helpers for the learned Earth-signature localizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import RealOceanCorpus
from .models import (
    OceanTeacherModel,
    OceanTeacherModelSpec,
    RuntimeStudentModel,
    fit_runtime_student_initialization,
    fit_teacher_from_patches,
    summarize_query_windows,
)

FloatArray = NDArray[np.float64]


@dataclass
class TeacherTrainingSpec:
    latent_dim: int = 48
    num_latents: int = 16
    name: str = "teacher_training_reference"


@dataclass
class DelayedLocalizerTrainingSpec:
    epochs: int = 250
    learning_rate: float = 0.05
    l2: float = 1.0e-4
    analytic_log_emission_gain: float = 0.35
    reliability_threshold: float = 0.65
    name: str = "delayed_localizer_training_reference"


def train_ocean_teacher(
    corpus: RealOceanCorpus,
    *,
    spec: TeacherTrainingSpec | None = None,
) -> OceanTeacherModel:
    train_spec = TeacherTrainingSpec() if spec is None else spec
    model_spec = OceanTeacherModelSpec(
        patch_channels=tuple(corpus.metadata["patch_channel_names"]),
        latent_dim=int(train_spec.latent_dim),
        num_latents=int(train_spec.num_latents),
    )
    return fit_teacher_from_patches(
        corpus.patch_tensors,
        spec=model_spec,
    )


def fit_delayed_localizer(
    corpus: RealOceanCorpus,
    *,
    teacher_spec: TeacherTrainingSpec | None = None,
    train_spec: DelayedLocalizerTrainingSpec | None = None,
) -> RuntimeStudentModel:
    teacher = train_ocean_teacher(corpus, spec=teacher_spec)
    student = distill_runtime_student(
        corpus,
        teacher,
        analytic_log_emission_gain=(
            DelayedLocalizerTrainingSpec().analytic_log_emission_gain
            if train_spec is None
            else float(train_spec.analytic_log_emission_gain)
        ),
    )
    return train_delayed_localizer(
        corpus,
        student,
        spec=train_spec,
    )


def distill_runtime_student(
    corpus: RealOceanCorpus,
    teacher: OceanTeacherModel,
    *,
    analytic_log_emission_gain: float = 0.35,
) -> RuntimeStudentModel:
    teacher_embeddings = teacher.encode_patches(corpus.patch_tensors)
    query_summaries = summarize_query_windows(corpus.query_windows)
    return fit_runtime_student_initialization(
        patch_summary_features=corpus.patch_summary_features,
        teacher_embeddings=teacher_embeddings,
        query_summaries=query_summaries,
        candidate_feature_names=tuple(corpus.metadata["candidate_feature_names"]),
        query_feature_names=tuple(corpus.metadata["query_feature_names"]),
        embedding_dim=min(32, teacher_embeddings.shape[1]),
        analytic_log_emission_gain=analytic_log_emission_gain,
    )


def _query_gradient(
    student: RuntimeStudentModel,
    corpus: RealOceanCorpus,
    query_projection: FloatArray,
    offset_linear: FloatArray,
    analytic_gain: float,
) -> tuple[FloatArray, FloatArray, float, FloatArray]:
    query_summary = summarize_query_windows(corpus.query_windows)
    q_norm = (query_summary - student.query_mean) / student.query_std
    q_emb = np.tanh(q_norm @ query_projection)
    cand_emb = student.encode_candidate_features(corpus.candidate_features)
    logits = np.einsum("bd,bcd->bc", q_emb, cand_emb) / np.sqrt(
        max(q_emb.shape[1], 1)
    )
    logits += np.tensordot(
        corpus.candidate_offsets_ned_m[..., :2],
        offset_linear,
        axes=([-1], [0]),
    )
    logits += float(analytic_gain) * corpus.analytic_log_emission
    xmax = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits - xmax, -700.0, 0.0))
    probs = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1.0e-300)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), corpus.labels] = 1.0
    diff = (probs - one_hot) / max(probs.shape[0], 1)
    grad_q = np.einsum("bc,bcd->bd", diff, cand_emb) / np.sqrt(max(q_emb.shape[1], 1))
    tanh_grad = (1.0 - q_emb**2) * grad_q
    grad_query_projection = q_norm.T @ tanh_grad
    grad_offset_linear = np.einsum(
        "bc,bck->k",
        diff,
        corpus.candidate_offsets_ned_m[..., :2],
    )
    loss = float(
        -np.mean(
            np.log(
                np.maximum(
                    probs[np.arange(probs.shape[0]), corpus.labels],
                    1.0e-300,
                )
            )
        )
    )
    return (
        grad_query_projection.astype(np.float64),
        grad_offset_linear.astype(np.float64),
        loss,
        probs.astype(np.float64),
    )


def _fit_linear_head(
    X: FloatArray,
    y: FloatArray,
    *,
    positive_output: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    x = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = np.mean(x, axis=0)
    std = np.maximum(np.std(x, axis=0), 1.0e-9)
    x_norm = (x - mean) / std
    x_aug = np.column_stack([x_norm, np.ones(x_norm.shape[0], dtype=np.float64)])
    if positive_output:
        target = np.log(np.maximum(y_arr, 1.0e-6))
    else:
        target = y_arr
    weights = np.linalg.lstsq(x_aug, target, rcond=None)[0]
    return (
        mean.astype(np.float64),
        std.astype(np.float64),
        np.asarray(weights[:-1], dtype=np.float64),
        float(weights[-1]),
    )


def train_delayed_localizer(
    corpus: RealOceanCorpus,
    student: RuntimeStudentModel,
    *,
    spec: DelayedLocalizerTrainingSpec | None = None,
) -> RuntimeStudentModel:
    train_spec = DelayedLocalizerTrainingSpec() if spec is None else spec
    query_projection = np.asarray(student.query_projection, dtype=np.float64).copy()
    offset_linear = np.asarray(student.offset_linear, dtype=np.float64).copy()
    analytic_gain = float(train_spec.analytic_log_emission_gain)

    for _ in range(int(train_spec.epochs)):
        grad_q, grad_offset, _, _ = _query_gradient(
            student,
            corpus,
            query_projection,
            offset_linear,
            analytic_gain,
        )
        query_projection -= float(train_spec.learning_rate) * (
            grad_q + float(train_spec.l2) * query_projection
        )
        offset_linear -= float(train_spec.learning_rate) * (
            grad_offset + float(train_spec.l2) * offset_linear
        )

    student.query_projection = query_projection.astype(np.float64)
    student.offset_linear = offset_linear.astype(np.float64)
    student.analytic_log_emission_gain = analytic_gain

    scores = student.predict_candidate_scores(
        query_windows=corpus.query_windows,
        candidate_features=corpus.candidate_features,
        candidate_offsets_ned_m=corpus.candidate_offsets_ned_m,
        analytic_log_emission=corpus.analytic_log_emission,
    )
    posterior = student.posterior_from_scores(scores)
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

    (
        student.covariance_features_mean,
        student.covariance_features_std,
        student.covariance_weights,
        student.covariance_bias,
    ) = _fit_linear_head(
        covariance_features,
        np.maximum(corpus.covariance_targets, 1.0e-3),
        positive_output=True,
    )
    (
        student.reliability_features_mean,
        student.reliability_features_std,
        student.reliability_weights,
        student.reliability_bias,
    ) = _fit_linear_head(
        reliability_features,
        corpus.publishability_labels.astype(np.float64),
        positive_output=False,
    )
    student.reliability_threshold = float(train_spec.reliability_threshold)
    student.metadata.update(
        {
            "training_name": train_spec.name,
            "reference_loss": float(
                -np.mean(
                    np.log(
                        np.maximum(
                            posterior[np.arange(posterior.shape[0]), corpus.labels],
                            1.0e-300,
                        )
                    )
                )
            ),
            "reliability_feature_dim": int(reliability_features.shape[1]),
        }
    )
    return student


def _edge_mask_from_offsets(
    offsets_ned_m: FloatArray,
) -> NDArray[np.bool_]:
    offsets = np.asarray(offsets_ned_m, dtype=np.float64)
    north = offsets[:, 0]
    east = offsets[:, 1]
    return np.asarray(
        np.isclose(north, np.min(north))
        | np.isclose(north, np.max(north))
        | np.isclose(east, np.min(east))
        | np.isclose(east, np.max(east)),
        dtype=bool,
    )


def _runtime_like_head_features(
    corpus: RealOceanCorpus,
    posterior: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    post = np.asarray(posterior, dtype=np.float64)
    query_summary = summarize_query_windows(corpus.query_windows)
    peak = np.max(post, axis=1)
    entropy = -np.sum(post * np.log(np.maximum(post, 1.0e-300)), axis=1)
    ess_fraction = 1.0 / np.maximum(np.sum(post**2, axis=1), 1.0e-300)
    ess_fraction = ess_fraction / np.maximum(float(post.shape[1]), 1.0)
    gravity_meas_std = float(corpus.metadata.get("gravity_meas_std_mps2", 1.0e-5))
    support_threshold_peak_fraction = float(
        corpus.metadata.get("ambiguity_support_threshold_peak_fraction", 0.25)
    )

    predicted_std = np.empty(corpus.num_examples, dtype=np.float64)
    edge_mass = np.empty(corpus.num_examples, dtype=np.float64)
    support_radius_fraction = np.empty(corpus.num_examples, dtype=np.float64)
    gravity_info_ratio = np.empty(corpus.num_examples, dtype=np.float64)

    for idx in range(corpus.num_examples):
        w = np.asarray(post[idx], dtype=np.float64)
        offsets = np.asarray(corpus.candidate_offsets_ned_m[idx, :, :2], dtype=np.float64)
        gravity_values = np.asarray(corpus.candidate_features[idx, :, 0], dtype=np.float64)
        edge_mask = _edge_mask_from_offsets(offsets)
        edge_mass[idx] = float(np.sum(w[edge_mask]))

        support_threshold = float(np.max(w)) * support_threshold_peak_fraction
        support_mask = np.asarray(w >= support_threshold, dtype=bool)
        if not np.any(support_mask):
            support_mask[int(np.argmax(w))] = True
        support_offsets = offsets[support_mask]
        half_span_n = max(float(np.max(np.abs(offsets[:, 0]))), 1.0e-9)
        half_span_e = max(float(np.max(np.abs(offsets[:, 1]))), 1.0e-9)
        support_radius_fraction[idx] = float(
            max(
                float(np.max(np.abs(support_offsets[:, 0]))) / half_span_n,
                float(np.max(np.abs(support_offsets[:, 1]))) / half_span_e,
            )
        )

        valid_g = np.isfinite(gravity_values)
        if np.any(valid_g):
            valid_w = w[valid_g]
            valid_w = valid_w / max(float(np.sum(valid_w)), 1.0e-12)
            valid_g_values = gravity_values[valid_g]
            g_mean = float(np.sum(valid_w * valid_g_values))
            predicted_std[idx] = float(
                np.sqrt(
                    max(
                        float(np.sum(valid_w * (valid_g_values - g_mean) ** 2)),
                        0.0,
                    )
                )
            )
            gravity_spread = (
                float(np.std(valid_g_values))
                if valid_g_values.size > 1
                else 0.0
            )
            gravity_info_ratio[idx] = gravity_spread / max(gravity_meas_std, 1.0e-12)
        else:
            predicted_std[idx] = 0.0
            gravity_info_ratio[idx] = 0.0

    reliability_features = np.column_stack(
        [
            peak,
            entropy,
            predicted_std,
            edge_mass,
            support_radius_fraction,
            ess_fraction,
            gravity_info_ratio,
            np.nanmean(np.abs(query_summary), axis=1),
        ]
    ).astype(np.float64)
    covariance_features = np.column_stack(
        [
            peak,
            entropy,
            edge_mass,
            ess_fraction,
            gravity_info_ratio,
            predicted_std,
        ]
    ).astype(np.float64)
    support_features = reliability_features.copy()
    return reliability_features, covariance_features, support_features


def evaluate_runtime_student(
    corpus: RealOceanCorpus,
    student: Any,
) -> dict[str, Any]:
    scores = student.predict_candidate_scores(
        query_windows=corpus.query_windows,
        candidate_features=corpus.candidate_features,
        candidate_offsets_ned_m=corpus.candidate_offsets_ned_m,
        analytic_log_emission=corpus.analytic_log_emission,
    )
    posterior = student.posterior_from_scores(scores)
    pred_idx = np.argmax(posterior, axis=1)
    top1 = float(np.mean(pred_idx == corpus.labels))
    expected_offsets = np.sum(
        posterior[:, :, None] * corpus.candidate_offsets_ned_m[..., :2],
        axis=1,
    )
    horizontal_error = np.linalg.norm(
        expected_offsets - corpus.truth_offsets_ned_m[:, :2],
        axis=1,
    )
    reliability_features, _, support_features = _runtime_like_head_features(
        corpus,
        posterior,
    )
    publish_prob = student.publishability_probability_from_features(
        reliability_features
    )
    support_prob = (
        student.support_expansion_probability_from_features(support_features)
        if hasattr(student, "support_expansion_probability_from_features")
        else np.zeros(corpus.num_examples, dtype=np.float64)
    )
    return {
        "top1_accuracy": top1,
        "median_horizontal_error_m": float(np.median(horizontal_error)),
        "p90_horizontal_error_m": float(np.quantile(horizontal_error, 0.90)),
        "mean_horizontal_error_m": float(np.mean(horizontal_error)),
        "mean_publishability_probability": float(np.mean(publish_prob)),
        "publishability_positive_fraction": float(
            np.mean(publish_prob >= student.reliability_threshold)
        ),
        "mean_support_expansion_probability": float(np.mean(support_prob)),
        "support_expansion_positive_fraction": float(
            np.mean(np.asarray(support_prob, dtype=np.float64) >= 0.5)
        ),
    }


def cross_validate_delayed_localizer(
    corpus: RealOceanCorpus,
    *,
    held_out_regions: tuple[str, ...] | None = None,
    teacher_spec: TeacherTrainingSpec | None = None,
    train_spec: DelayedLocalizerTrainingSpec | None = None,
) -> dict[str, Any]:
    fold_regions = corpus.region_names if held_out_regions is None else tuple(
        str(region) for region in held_out_regions
    )
    unknown = [region for region in fold_regions if region not in corpus.region_names]
    if len(unknown) > 0:
        raise ValueError(f"Unknown held-out regions: {unknown}.")
    if len(fold_regions) == 0:
        raise ValueError("At least one held-out region is required.")

    folds: list[dict[str, Any]] = []
    for held_out_region in fold_regions:
        train_corpus = corpus.select_regions(
            exclude=(held_out_region,),
            name=f"train_excluding_{held_out_region}",
        )
        eval_corpus = corpus.select_regions(
            include=(held_out_region,),
            name=f"held_out_{held_out_region}",
        )
        model = fit_delayed_localizer(
            train_corpus,
            teacher_spec=teacher_spec,
            train_spec=train_spec,
        )
        metrics = evaluate_runtime_student(eval_corpus, model)
        folds.append(
            {
                "held_out_region": held_out_region,
                "train_examples": int(train_corpus.num_examples),
                "eval_examples": int(eval_corpus.num_examples),
                "train_region_example_counts": train_corpus.region_example_counts(),
                "eval_region_example_counts": eval_corpus.region_example_counts(),
                "reference_parameter_count": int(model.reference_parameter_count),
                **metrics,
            }
        )

    medians = {
        "median_top1_accuracy": float(
            np.median([float(fold["top1_accuracy"]) for fold in folds])
        ),
        "median_horizontal_error_m": float(
            np.median([float(fold["median_horizontal_error_m"]) for fold in folds])
        ),
        "median_p90_horizontal_error_m": float(
            np.median([float(fold["p90_horizontal_error_m"]) for fold in folds])
        ),
        "median_mean_publishability_probability": float(
            np.median([float(fold["mean_publishability_probability"]) for fold in folds])
        ),
        "median_publishability_positive_fraction": float(
            np.median(
                [float(fold["publishability_positive_fraction"]) for fold in folds]
            )
        ),
    }
    return {
        "num_folds": int(len(folds)),
        "folds": folds,
        "aggregate": medians,
        "teacher_spec": asdict(TeacherTrainingSpec() if teacher_spec is None else teacher_spec),
        "train_spec": asdict(
            DelayedLocalizerTrainingSpec() if train_spec is None else train_spec
        ),
    }
