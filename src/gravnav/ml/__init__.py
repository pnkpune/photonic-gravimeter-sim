"""
NumPy-based ML tooling for learned Earth-signature localization.
"""

from .data import (
    RealOceanCorpus,
    RealOceanCorpusSpec,
    build_real_ocean_corpus,
    load_real_ocean_corpus,
)
from .models import (
    OceanTeacherModel,
    OceanTeacherModelSpec,
    RuntimeStudentModel,
    RuntimeStudentModelSpec,
)
from .runtime import (
    LearnedLocalizerSpec,
    NeuralEarthSignatureLocalizer,
)
from .feedback_trust import (
    SEQUENCE_FEEDBACK_FEATURE_NAMES,
    SEQUENCE_FEEDBACK_MODES,
    SequenceFeedbackEventCorpus,
    SequenceFeedbackTrustCommittee,
    SequenceFeedbackTrustCommitteeMember,
    SequenceFeedbackTrustCommitteePrediction,
    SequenceFeedbackTrustModel,
    SequenceFeedbackTrustModelSpec,
    cross_validate_sequence_feedback_trust_model,
    extract_sequence_feedback_features,
    fit_sequence_feedback_trust_model,
    load_sequence_feedback_failure_manifest,
    load_sequence_feedback_trust_committee,
    load_sequence_feedback_trust_model,
)
from .training import (
    aggregate_runtime_student_folds,
    cross_validate_delayed_localizer,
    DEFAULT_FIXED_PUBLISHABILITY_THRESHOLD,
    DelayedLocalizerTrainingSpec,
    TeacherTrainingSpec,
    fit_delayed_localizer,
    distill_runtime_student,
    evaluate_runtime_student,
    train_delayed_localizer,
    train_ocean_teacher,
)
from .experiment_registry import (
    DEFAULT_EXPERIMENT_CONFIG,
    ResolvedRegionSet,
    list_corpus_presets,
    list_region_sets,
    load_real_ocean_experiment_config,
    resolve_corpus_preset,
    resolve_region_set,
)

try:  # Optional torch backend
    from .torch_models import (
        TorchDelayedLocalizerTrainingSpec,
        TorchRuntimeStudentModel,
        TorchRuntimeStudentModelSpec,
        load_runtime_localizer_model,
        torch_is_available,
        train_torch_delayed_localizer,
    )
except Exception:  # pragma: no cover - optional dependency path
    pass

__all__ = [
    "DelayedLocalizerTrainingSpec",
    "DEFAULT_EXPERIMENT_CONFIG",
    "LearnedLocalizerSpec",
    "NeuralEarthSignatureLocalizer",
    "OceanTeacherModel",
    "OceanTeacherModelSpec",
    "RealOceanCorpus",
    "RealOceanCorpusSpec",
    "ResolvedRegionSet",
    "RuntimeStudentModel",
    "RuntimeStudentModelSpec",
    "TeacherTrainingSpec",
    "TorchDelayedLocalizerTrainingSpec",
    "TorchRuntimeStudentModel",
    "TorchRuntimeStudentModelSpec",
    "build_real_ocean_corpus",
    "aggregate_runtime_student_folds",
    "cross_validate_delayed_localizer",
    "cross_validate_sequence_feedback_trust_model",
    "DEFAULT_FIXED_PUBLISHABILITY_THRESHOLD",
    "distill_runtime_student",
    "evaluate_runtime_student",
    "extract_sequence_feedback_features",
    "fit_delayed_localizer",
    "fit_sequence_feedback_trust_model",
    "list_corpus_presets",
    "list_region_sets",
    "load_sequence_feedback_trust_model",
    "load_runtime_localizer_model",
    "load_real_ocean_experiment_config",
    "load_real_ocean_corpus",
    "resolve_corpus_preset",
    "resolve_region_set",
    "SEQUENCE_FEEDBACK_FEATURE_NAMES",
    "SEQUENCE_FEEDBACK_MODES",
    "SequenceFeedbackEventCorpus",
    "SequenceFeedbackTrustCommittee",
    "SequenceFeedbackTrustCommitteeMember",
    "SequenceFeedbackTrustCommitteePrediction",
    "SequenceFeedbackTrustModel",
    "SequenceFeedbackTrustModelSpec",
    "load_sequence_feedback_failure_manifest",
    "load_sequence_feedback_trust_committee",
    "torch_is_available",
    "train_delayed_localizer",
    "train_torch_delayed_localizer",
    "train_ocean_teacher",
]
