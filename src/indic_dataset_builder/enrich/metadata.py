"""Per-document metadata enrichment.

Adds cheap, useful descriptors (counts, ratios, script) that downstream
sampling, filtering, and dataset-card generation rely on. Computed once here
so the rest of the pipeline and any consumer can read them off the record.
"""
from __future__ import annotations

from typing import List

from ..schema import Document, Stage
from ..clean.text_utils import words as _words


def enrich_metadata(docs: List[Document]) -> List[Document]:
    for doc in docs:
        text = doc.text
        words = _words(text)
        doc.metadata.update({
            "char_count": len(text),
            "word_count": len(words),
            "line_count": text.count("\n") + 1,
            "unique_word_ratio": round(
                len(set(w.lower() for w in words)) / len(words), 4
            ) if words else 0.0,
            "avg_word_length": round(
                sum(len(w) for w in words) / len(words), 2
            ) if words else 0.0,
        })
        doc.touch(Stage.ENRICHED)
    return docs
