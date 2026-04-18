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
from .training import (
    cross_validate_delayed_localizer,
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
    "cross_validate_delayed_localizer",
    "distill_runtime_student",
    "evaluate_runtime_student",
    "fit_delayed_localizer",
    "list_corpus_presets",
    "list_region_sets",
    "load_runtime_localizer_model",
    "load_real_ocean_experiment_config",
    "load_real_ocean_corpus",
    "resolve_corpus_preset",
    "resolve_region_set",
    "torch_is_available",
    "train_delayed_localizer",
    "train_torch_delayed_localizer",
    "train_ocean_teacher",
]
