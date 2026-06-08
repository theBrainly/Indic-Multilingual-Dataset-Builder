"""Annotation quality assurance.

Turns raw annotations into actionable QA signals:
- majority-vote aggregation (with ties flagged)
- per-annotator accuracy vs. gold labels
- per-annotator agreement with the consensus (spot unreliable annotators)
- items needing adjudication (low agreement / no majority)
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from .agreement import cohens_kappa
from .schema import AnnotationItem


class AnnotationQA:
    def majority_vote(self, item: AnnotationItem) -> tuple[Optional[Any], float]:
        """Return (consensus_label, agreement_fraction). None label if tie."""
        labels = item.labels()
        if not labels:
            return None, 0.0
        counts = Counter(labels)
        top, n = counts.most_common(1)[0]
        # detect a tie for the top spot
        if list(counts.values()).count(n) > 1:
            return None, n / len(labels)
        return top, n / len(labels)

    def aggregate(self, items: List[AnnotationItem]) -> List[Dict[str, Any]]:
        out = []
        for it in items:
            label, agreement = self.majority_vote(it)
            out.append({
                "item_id": it.id,
                "consensus": label,
                "agreement": round(agreement, 4),
                "n_annotations": len(it.annotations),
                "needs_adjudication": label is None or agreement < 0.66,
            })
        return out

    def annotator_accuracy(self, items: List[AnnotationItem]) -> Dict[str, float]:
        """Accuracy of each annotator against gold labels (items that have them)."""
        correct: Counter = Counter()
        total: Counter = Counter()
        for it in items:
            if it.gold_label is None:
                continue
            for a in it.annotations:
                total[a.annotator_id] += 1
                if a.label == it.gold_label:
                    correct[a.annotator_id] += 1
        return {ann: round(correct[ann] / total[ann], 4)
                for ann in total if total[ann]}

    def annotator_vs_consensus(self, items: List[AnnotationItem]) -> Dict[str, float]:
        """Cohen's kappa of each annotator against the majority consensus."""
        # Collect per-annotator (label, consensus) aligned pairs.
        pairs: Dict[str, List[tuple]] = {}
        for it in items:
            consensus, _ = self.majority_vote(it)
            if consensus is None:
                continue
            for a in it.annotations:
                pairs.setdefault(a.annotator_id, []).append((a.label, consensus))
        result = {}
        for ann, pl in pairs.items():
            a_labels = [p[0] for p in pl]
            c_labels = [p[1] for p in pl]
            result[ann] = cohens_kappa(a_labels, c_labels)
        return result

    def report(self, items: List[AnnotationItem]) -> Dict[str, Any]:
        agg = self.aggregate(items)
        return {
            "num_items": len(items),
            "needs_adjudication": sum(1 for a in agg if a["needs_adjudication"]),
            "mean_agreement": round(
                sum(a["agreement"] for a in agg) / len(agg), 4) if agg else 0.0,
            "annotator_accuracy": self.annotator_accuracy(items),
            "annotator_vs_consensus_kappa": self.annotator_vs_consensus(items),
        }
