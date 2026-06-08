"""Corpus-level diversity analysis.

A curated corpus should be reported on, not just produced. This computes the
distribution across languages and sources, vocabulary size, token counts, and
a normalized Shannon-entropy "balance" score for language diversity — exactly
the kind of summary a model team needs before deciding to train on a corpus.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

from ..schema import Document
from ..clean.text_utils import words as _words


def _entropy_balance(counts: Counter) -> float:
    """Normalized Shannon entropy in [0,1]; 1 == perfectly balanced."""
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0 if len(counts) <= 1 else 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(h / math.log2(len(counts)), 4)


def corpus_diversity(docs: List[Document]) -> Dict[str, Any]:
    lang_counts: Counter = Counter()
    source_counts: Counter = Counter()
    vocab: set[str] = set()
    total_tokens = 0
    total_chars = 0

    for doc in docs:
        lang_counts[doc.language or "und"] += 1
        source_counts[doc.source.source_type] += 1
        words = _words(doc.text.lower())
        total_tokens += len(words)
        total_chars += len(doc.text)
        vocab.update(words)

    return {
        "num_documents": len(docs),
        "num_languages": len(lang_counts),
        "language_distribution": dict(lang_counts.most_common()),
        "language_balance": _entropy_balance(lang_counts),
        "source_distribution": dict(source_counts.most_common()),
        "total_tokens": total_tokens,
        "total_characters": total_chars,
        "vocabulary_size": len(vocab),
        "type_token_ratio": round(len(vocab) / total_tokens, 4) if total_tokens else 0.0,
        "avg_doc_tokens": round(total_tokens / len(docs), 2) if docs else 0.0,
    }
