"""Data models for annotation workflows."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Annotation(BaseModel):
    """A single label applied by one annotator to one item."""

    item_id: str
    annotator_id: str
    label: Any                       # category, score, or boolean
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnnotationItem(BaseModel):
    """An item to be annotated, with all collected annotations."""

    id: str
    text: str
    task: str = "classification"     # classification | preference | trace_review
    annotations: List[Annotation] = Field(default_factory=list)
    gold_label: Optional[Any] = None  # ground truth, if known

    def labels(self) -> List[Any]:
        return [a.label for a in self.annotations]

    def annotators(self) -> List[str]:
        return [a.annotator_id for a in self.annotations]


class PreferencePair(BaseModel):
    """A preference judgement: `chosen` is preferred over `rejected`."""

    id: str
    prompt: str
    chosen: str
    rejected: str
    annotator_id: str
    margin: float = 1.0              # strength of preference, 0-1
    metadata: Dict[str, Any] = Field(default_factory=dict)
