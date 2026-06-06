"""Build parallel (translation) pairs across languages.

Multilingual training and evaluation needs aligned parallel data. Documents
that share an explicit `parallel_id` in their metadata are linked directly;
otherwise we fall back to a lightweight heuristic combining character-length
ratio and shared non-alphabetic anchors (numbers, URLs, named entities) — a
cheap stand-in for an embedding-based aligner like LASER/LaBSE, which can be
slotted in via the same interface.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, List

from ..schema import AlignedPair, Document, Stage

_ANCHOR_RE = re.compile(r"\d+|https?://\S+|[A-Z][a-zA-Z]{2,}")


class ParallelAligner:
    def __init__(self, length_ratio_max: float = 2.5):
        self.length_ratio_max = length_ratio_max

    def align(self, docs: List[Document]) -> List[AlignedPair]:
        pairs: List[AlignedPair] = []
        # 1) explicit links via metadata `parallel_id`
        groups: Dict[str, List[Document]] = {}
        for doc in docs:
            pid = doc.metadata.get("parallel_id")
            if pid is not None:
                groups.setdefault(str(pid), []).append(doc)

        for pid, members in groups.items():
            for a, b in combinations(members, 2):
                if a.language == b.language:
                    continue
                score = self._score(a, b)
                pairs.append(self._make_pair(f"align-{pid}-{a.id}-{b.id}",
                                             a, b, score, "explicit"))
                a.touch(Stage.ALIGNED)
                b.touch(Stage.ALIGNED)
        return pairs

    def _score(self, a: Document, b: Document) -> float:
        la, lb = len(a.text), len(b.text)
        if min(la, lb) == 0:
            return 0.0
        ratio = max(la, lb) / min(la, lb)
        length_score = max(0.0, 1.0 - (ratio - 1.0) / self.length_ratio_max)
        anchors_a = set(_ANCHOR_RE.findall(a.text))
        anchors_b = set(_ANCHOR_RE.findall(b.text))
        if anchors_a or anchors_b:
            inter = len(anchors_a & anchors_b)
            union = len(anchors_a | anchors_b) or 1
            anchor_score = inter / union
        else:
            anchor_score = length_score
        return round(0.6 * length_score + 0.4 * anchor_score, 4)

    @staticmethod
    def _make_pair(pair_id: str, a: Document, b: Document,
                   score: float, method: str) -> AlignedPair:
        return AlignedPair(
            id=pair_id,
            source_doc_id=a.id, target_doc_id=b.id,
            source_language=a.language or "und",
            target_language=b.language or "und",
            source_text=a.text, target_text=b.text,
            alignment_score=score, method=method,
        )
