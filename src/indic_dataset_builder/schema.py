"""Core data models for the curation pipeline.

Everything that flows through the pipeline is a :class:`Document`. Documents
carry their own provenance and the ordered list of stages that touched them,
which is what makes a curation run auditable and reproducible.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    """Deterministic content hash used for exact dedup and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Stage(str, Enum):
    """Pipeline stages, recorded on each document for traceability."""

    COLLECTED = "collected"
    NORMALIZED = "normalized"
    LANGUAGE_TAGGED = "language_tagged"
    FILTERED = "filtered"
    ALIGNED = "aligned"
    EXACT_DEDUP = "exact_dedup"
    NEAR_DEDUP = "near_dedup"
    SEMANTIC_DEDUP = "semantic_dedup"
    CONTAMINATION_CHECKED = "contamination_checked"
    ENRICHED = "enriched"
    EXPORTED = "exported"


class ProvenanceRecord(BaseModel):
    """Where a document came from and how it was acquired."""

    source_type: str                      # file | web | api | warc | hf | synthetic
    source_id: str                        # url, file path, endpoint, dataset name
    collected_at: str = Field(default_factory=_utcnow)
    collector: str = "indic-dataset-builder"
    license: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A single text record moving through the pipeline."""

    id: str
    text: str
    language: Optional[str] = None        # ISO 639-1 (e.g. "hi"), None if unknown
    language_confidence: float = 0.0
    source: ProvenanceRecord
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stages: List[Stage] = Field(default_factory=list)

    # Curation bookkeeping
    hash: Optional[str] = None
    dropped: bool = False
    drop_reason: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:  # pydantic v2 hook
        if self.hash is None:
            self.hash = content_hash(self.text)

    def touch(self, stage: Stage) -> "Document":
        """Record that a pipeline stage processed this document."""
        if stage not in self.stages:
            self.stages.append(stage)
        return self

    def drop(self, reason: str) -> "Document":
        """Flag a document as dropped while keeping the audit reason."""
        self.dropped = True
        self.drop_reason = reason
        return self

    def refresh_hash(self) -> "Document":
        self.hash = content_hash(self.text)
        return self

    def to_export_dict(self) -> Dict[str, Any]:
        """Flat dict suitable for JSONL / Parquet export."""
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "language_confidence": round(self.language_confidence, 4),
            "source_type": self.source.source_type,
            "source_id": self.source.source_id,
            "collected_at": self.source.collected_at,
            "license": self.source.license,
            "hash": self.hash,
            "stages": ",".join(s.value for s in self.stages),
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }


class AlignedPair(BaseModel):
    """A parallel (translation) pair linking two documents across languages."""

    id: str
    source_doc_id: str
    target_doc_id: str
    source_language: str
    target_language: str
    source_text: str
    target_text: str
    alignment_score: float = 0.0
    method: str = "length+lexical"

    def to_export_dict(self) -> Dict[str, Any]:
        return self.model_dump()
