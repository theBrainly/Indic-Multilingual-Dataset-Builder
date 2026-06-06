"""Per-document quality scoring.

Beyond pass/fail filtering, a continuous 0-1 quality score lets a model team
sample or weight training data. The score blends length adequacy, lexical
diversity (type-token ratio), symbol/digit cleanliness, and language-ID
confidence. It is intentionally interpretable rather than a learned classifier.
"""
from __future__ import annotations

from typing import List

from ..schema import Document
from ..clean.text_utils import symbol_count, words as _words


class QualityScorer:
    def score(self, doc: Document) -> float:
        text = doc.text
        words = _words(text)
        if not words:
            return 0.0

        # length adequacy: saturates around 50+ words
        length_score = min(len(words) / 50.0, 1.0)

        # lexical diversity (type-token ratio), penalize heavy repetition
        ttr = len(set(w.lower() for w in words)) / len(words)

        # cleanliness: fewer stray symbols is better
        symbol_ratio = symbol_count(text) / max(len(text), 1)
        clean_score = max(0.0, 1.0 - symbol_ratio * 2)

        # language identification confidence
        lang_score = doc.language_confidence if doc.language else 0.0

        score = (
            0.30 * length_score
            + 0.30 * ttr
            + 0.20 * clean_score
            + 0.20 * lang_score
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def annotate(self, docs: List[Document]) -> List[Document]:
        for doc in docs:
            doc.metadata["quality_score"] = self.score(doc)
        return docs
