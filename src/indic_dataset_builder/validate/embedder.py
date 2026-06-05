"""Pluggable text embedder.

Production-grade semantic dedup wants multilingual sentence embeddings
(LaBSE / multilingual-MiniLM via `sentence-transformers`). When that isn't
installed we fall back to a deterministic hashing embedder over character
n-grams so semantic dedup still runs (with lower fidelity) and the demo never
breaks. Both paths return L2-normalized vectors so callers can use dot product
as cosine similarity.
"""
from __future__ import annotations

import hashlib
import math
from typing import List

import numpy as np


class Embedder:
    """Base interface."""

    dim: int

    def encode(self, texts: List[str]) -> "np.ndarray":
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Dependency-free character n-gram hashing embedder (the fallback)."""

    def __init__(self, dim: int = 256, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def _encode_one(self, text: str) -> "np.ndarray":
        vec = np.zeros(self.dim, dtype=np.float32)
        text = " ".join(text.split())
        if not text:
            return vec
        for i in range(max(len(text) - self.ngram + 1, 1)):
            gram = text[i:i + self.ngram]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(float((vec * vec).sum()))
        return vec / norm if norm > 0 else vec

    def encode(self, texts: List[str]) -> "np.ndarray":
        return np.vstack([self._encode_one(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Multilingual transformer embeddings when the library is available."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> "np.ndarray":
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")


def build_embedder(prefer_transformer: bool = True,
                   model_name: str = "sentence-transformers/LaBSE") -> Embedder:
    """Return the best available embedder, falling back gracefully."""
    if prefer_transformer:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:
            pass
    return HashingEmbedder()
