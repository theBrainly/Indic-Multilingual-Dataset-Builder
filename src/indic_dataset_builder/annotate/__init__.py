"""Annotation & human-in-the-loop quality assurance.

Covers the JD's "Annotation & Human Feedback Systems" area:
- inter-annotator agreement (Cohen's / Fleiss' kappa, Krippendorff's alpha)
- annotator consistency vs. gold labels + majority-vote aggregation
- preference-dataset validation (chosen/rejected, transitivity checks)
- reasoning-trace / code-correctness review hooks
"""
from .schema import Annotation, AnnotationItem, PreferencePair  # noqa: F401
from .agreement import (  # noqa: F401
    cohens_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    percent_agreement,
)
from .qa import AnnotationQA  # noqa: F401
from .preference import validate_preferences  # noqa: F401
