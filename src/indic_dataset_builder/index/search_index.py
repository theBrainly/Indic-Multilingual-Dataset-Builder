"""Keyword index with an Elasticsearch backend and an in-memory fallback.

Used for: corpus search during manual review, exact-text lookups for dedup/
contamination spot-checks, and retrieving examples by metadata. Both backends
share the same interface so calling code is backend-agnostic.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..clean.text_utils import words as _words


class SearchIndex:
    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        raise NotImplementedError

    def exact_match(self, text: str) -> List[str]:
        raise NotImplementedError


class InMemoryIndex(SearchIndex):
    """A small TF-IDF inverted index — no external services required."""

    def __init__(self) -> None:
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc: tf}
        self._docs: Dict[str, str] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._exact: Dict[str, List[str]] = defaultdict(list)

    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._docs[doc_id] = text
        self._meta[doc_id] = metadata or {}
        self._exact[text.strip().lower()].append(doc_id)
        for term, tf in Counter(t.lower() for t in _words(text)).items():
            self._postings[term][doc_id] = tf

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = len(self._postings.get(term, {}))
        if df == 0:
            return 0.0
        return math.log((n + 1) / (df + 1)) + 1.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        for term in (t.lower() for t in _words(query)):
            idf = self._idf(term)
            for doc_id, tf in self._postings.get(term, {}).items():
                scores[doc_id] += tf * idf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(d, round(s, 4)) for d, s in ranked[:top_k]]

    def exact_match(self, text: str) -> List[str]:
        return list(self._exact.get(text.strip().lower(), []))

    def __len__(self) -> int:
        return len(self._docs)


class ElasticsearchIndex(SearchIndex):
    """Thin Elasticsearch wrapper (used when a server is reachable)."""

    def __init__(self, hosts: List[str] | None = None, index: str = "indic-corpus"):
        from elasticsearch import Elasticsearch  # type: ignore

        self.es = Elasticsearch(hosts or ["http://localhost:9200"])
        self.index = index
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index)

    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.es.index(index=self.index, id=doc_id,
                      document={"text": text, **(metadata or {})})

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        res = self.es.search(index=self.index, query={"match": {"text": query}},
                             size=top_k)
        return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

    def exact_match(self, text: str) -> List[str]:
        res = self.es.search(index=self.index,
                             query={"match_phrase": {"text": text}}, size=50)
        return [h["_id"] for h in res["hits"]["hits"]]


def build_index(prefer_elasticsearch: bool = False, **kwargs) -> SearchIndex:
    """Return an Elasticsearch index if requested and reachable, else in-memory."""
    if prefer_elasticsearch:
        try:
            return ElasticsearchIndex(**kwargs)
        except Exception:
            pass
    return InMemoryIndex()
