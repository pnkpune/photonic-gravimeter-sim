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
    DelayedLocalizerTrainingSpec,
    TeacherTrainingSpec,
    distill_runtime_student,
    evaluate_runtime_student,
    train_delayed_localizer,
    train_ocean_teacher,
)

__all__ = [
    "DelayedLocalizerTrainingSpec",
    "LearnedLocalizerSpec",
    "NeuralEarthSignatureLocalizer",
    "OceanTeacherModel",
    "OceanTeacherModelSpec",
    "RealOceanCorpus",
    "RealOceanCorpusSpec",
    "RuntimeStudentModel",
    "RuntimeStudentModelSpec",
    "TeacherTrainingSpec",
    "build_real_ocean_corpus",
    "distill_runtime_student",
    "evaluate_runtime_student",
    "load_real_ocean_corpus",
    "train_delayed_localizer",
    "train_ocean_teacher",
]
