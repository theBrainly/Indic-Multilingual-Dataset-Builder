"""Exact deduplication via content hashing.

The cheapest and highest-precision dedup pass: identical documents (after
normalization) share a SHA-256 content hash, so we keep the first occurrence
and drop the rest. Runs before the more expensive near/semantic passes.
"""
from __future__ import annotations

from typing import List, Tuple

from ..schema import Document, Stage


def exact_dedup(docs: List[Document]) -> Tuple[List[Document], int]:
    """Keep first occurrence of each content hash.

    Returns (kept_documents, num_dropped). Dropped docs are flagged in place
    with the id of the document they duplicated, preserving the audit trail.
    """
    seen: dict[str, str] = {}
    kept: List[Document] = []
    dropped = 0
    for doc in docs:
        doc.touch(Stage.EXACT_DEDUP)
        h = doc.hash or doc.refresh_hash().hash
        if h in seen:
            doc.drop(f"exact_duplicate_of:{seen[h]}")
            dropped += 1
        else:
            seen[h] = doc.id
            kept.append(doc)
    return kept, dropped
