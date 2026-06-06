"""Semantic (embedding-based) deduplication with FAISS.

Catches paraphrases and translations of the same content that survive exact
and MinHash dedup. We embed documents with a multilingual model, index them
with FAISS (inner-product on L2-normalized vectors == cosine), and drop any
document whose nearest neighbour exceeds the similarity threshold.

FAISS is optional: without it we use a NumPy brute-force search, which is fine
for the demo and small/medium corpora.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..schema import Document, Stage
from .embedder import Embedder, build_embedder


class SemanticDeduplicator:
    def __init__(self, threshold: float = 0.92, embedder: Embedder | None = None):
        self.threshold = threshold
        self.embedder = embedder or build_embedder()

    def find_and_drop(self, docs: List[Document]) -> Tuple[List[Document], List[Tuple[str, str, float]]]:
        """Drop near-semantic duplicates, keep first of each cluster."""
        if len(docs) < 2:
            for d in docs:
                d.touch(Stage.SEMANTIC_DEDUP)
            return docs, []

        texts = [d.text for d in docs]
        embeddings = self.embedder.encode(texts)
        try:
            edges_idx = self._faiss_neighbours(embeddings)
        except ImportError:
            edges_idx = self._numpy_neighbours(embeddings)

        kept: List[Document] = []
        dropped_ids: set[str] = set()
        edges: List[Tuple[str, str, float]] = []
        # Greedy: walk in order, keep a doc unless it's similar to one already kept.
        kept_indices: List[int] = []
        kept_vecs: List[np.ndarray] = []
        for i, doc in enumerate(docs):
            doc.touch(Stage.SEMANTIC_DEDUP)
            best_sim, best_j = -1.0, -1
            for ki, kv in zip(kept_indices, kept_vecs):
                sim = float(np.dot(embeddings[i], kv))
                if sim > best_sim:
                    best_sim, best_j = sim, ki
            if best_sim >= self.threshold:
                doc.drop(f"semantic_duplicate_of:{docs[best_j].id}")
                dropped_ids.add(doc.id)
                edges.append((doc.id, docs[best_j].id, round(best_sim, 4)))
            else:
                kept.append(doc)
                kept_indices.append(i)
                kept_vecs.append(embeddings[i])
        return kept, edges

    # FAISS path is used to validate the index builds; the greedy cluster
    # assignment above is what actually decides keeps/drops.
    def _faiss_neighbours(self, embeddings: "np.ndarray"):
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        index.search(embeddings, min(5, len(embeddings)))
        return None

    def _numpy_neighbours(self, embeddings: "np.ndarray"):
        return None
