"""Near-duplicate detection with MinHash + LSH.

Exact hashing misses documents that differ by a boilerplate header, a date,
or minor edits. MinHash estimates Jaccard similarity over character shingles,
and an LSH index makes finding candidate pairs near-linear instead of O(n^2) —
the standard approach for deduping web-scale corpora (CCNet, The Pile, etc.).

Uses `datasketch` when available (pure-python, in the core deps) and falls
back to an internal MinHash implementation otherwise.
"""
from __future__ import annotations

import hashlib
from typing import List, Set, Tuple

from ..schema import Document, Stage


def _shingles(text: str, k: int) -> Set[str]:
    """Character k-gram shingles — script-agnostic, good for Indic text."""
    text = " ".join(text.split())
    if len(text) <= k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


class NearDuplicateFinder:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128,
                 shingle_size: int = 4):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

    def find_and_drop(self, docs: List[Document]) -> Tuple[List[Document], List[Tuple[str, str, float]]]:
        """Drop near-duplicates, keeping the first member of each cluster.

        Returns (kept_docs, dropped_edges) where each edge is
        (dropped_id, kept_id, estimated_jaccard).
        """
        try:
            return self._with_datasketch(docs)
        except ImportError:
            return self._fallback(docs)

    # ---- datasketch path -------------------------------------------------
    def _with_datasketch(self, docs: List[Document]):
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        minhashes: dict[str, "MinHash"] = {}
        kept: List[Document] = []
        edges: List[Tuple[str, str, float]] = []

        for doc in docs:
            doc.touch(Stage.NEAR_DEDUP)
            mh = MinHash(num_perm=self.num_perm)
            for sh in _shingles(doc.text, self.shingle_size):
                mh.update(sh.encode("utf-8"))
            candidates = lsh.query(mh)
            if candidates:
                kept_id = candidates[0]
                est = mh.jaccard(minhashes[kept_id])
                doc.drop(f"near_duplicate_of:{kept_id}")
                edges.append((doc.id, kept_id, round(float(est), 4)))
                continue
            lsh.insert(doc.id, mh)
            minhashes[doc.id] = mh
            kept.append(doc)
        return kept, edges

    # ---- dependency-free fallback ---------------------------------------
    def _fallback(self, docs: List[Document]):
        """Banding LSH over a hand-rolled MinHash. Slower but no deps."""
        kept: List[Document] = []
        signatures: List[Tuple[str, List[int], Set[str]]] = []
        edges: List[Tuple[str, str, float]] = []

        for doc in docs:
            doc.touch(Stage.NEAR_DEDUP)
            shset = _shingles(doc.text, self.shingle_size)
            sig = self._minhash_signature(shset)
            match = None
            for kid, ksig, kshset in signatures:
                if self._signature_similarity(sig, ksig) >= self.threshold:
                    match = (kid, self._jaccard(shset, kshset))
                    break
            if match:
                doc.drop(f"near_duplicate_of:{match[0]}")
                edges.append((doc.id, match[0], round(match[1], 4)))
            else:
                signatures.append((doc.id, sig, shset))
                kept.append(doc)
        return kept, edges

    def _minhash_signature(self, shingles: Set[str]) -> List[int]:
        if not shingles:
            return [0] * self.num_perm
        sig = []
        for seed in range(self.num_perm):
            mn = min(
                int(hashlib.md5(f"{seed}:{sh}".encode()).hexdigest(), 16)
                for sh in shingles
            )
            sig.append(mn)
        return sig

    @staticmethod
    def _signature_similarity(a: List[int], b: List[int]) -> float:
        if not a or not b:
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)
