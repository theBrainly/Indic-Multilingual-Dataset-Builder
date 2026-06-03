"""Indic Multilingual Dataset Builder.

An end-to-end curation stack for multilingual (Indic-focused) LLM training data:
collect -> clean -> align -> validate -> enrich -> govern -> export.
"""

__version__ = "0.1.0"

from .schema import Document, AlignedPair, ProvenanceRecord  # noqa: F401
