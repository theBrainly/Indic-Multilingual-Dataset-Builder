"""Benchmark contamination / training-test leakage detection.

If evaluation benchmark text leaks into training data, reported model scores
are inflated and untrustworthy. Following the GPT-3 / Llama methodology, we
flag any training document that shares a sufficiently long n-gram (default
13-gram) with any benchmark example. Matches are reported, and contaminated
documents are dropped so the curated corpus is safe to train on.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..schema import Document, Stage


def _word_ngrams(text: str, n: int) -> Set[str]:
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class ContaminationDetector:
    def __init__(self, benchmark_docs: List[Document], ngram: int = 13):
        self.ngram = ngram
        self.benchmark_index: Dict[str, str] = {}
        for bd in benchmark_docs:
            for g in _word_ngrams(bd.text.lower(), ngram):
                self.benchmark_index.setdefault(g, bd.id)

    def check_and_drop(self, docs: List[Document]) -> Tuple[List[Document], List[Tuple[str, str, str]]]:
        """Drop contaminated docs. Returns (kept, hits).

        hits = list of (train_doc_id, benchmark_id, matching_ngram).
        """
        kept: List[Document] = []
        hits: List[Tuple[str, str, str]] = []
        for doc in docs:
            doc.touch(Stage.CONTAMINATION_CHECKED)
            hit = self._first_overlap(doc.text.lower())
            if hit:
                bench_id, gram = hit
                doc.drop(f"contaminated_by:{bench_id}")
                doc.metadata["contamination_ngram"] = gram
                hits.append((doc.id, bench_id, gram))
            else:
                kept.append(doc)
        return kept, hits

    def _first_overlap(self, text: str):
        for g in _word_ngrams(text, self.ngram):
            if g in self.benchmark_index:
                return self.benchmark_index[g], g
        return None
