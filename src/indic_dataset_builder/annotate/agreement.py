"""Inter-annotator agreement metrics.

Agreement metrics tell you whether your labels are trustworthy. We implement
the standard ones from scratch (no sklearn dependency) so they're transparent
and auditable:

- percent agreement      — simplest, ignores chance
- Cohen's kappa          — two annotators, chance-corrected
- Fleiss' kappa          — many annotators, fixed items
- Krippendorff's alpha   — any number of annotators, handles missing data
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Sequence


def percent_agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Fraction of items where two annotators agree."""
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa for two annotators over the same items."""
    n = len(a)
    if n == 0:
        return 0.0
    po = percent_agreement(a, b)
    count_a = Counter(a)
    count_b = Counter(b)
    labels = set(count_a) | set(count_b)
    pe = sum((count_a.get(l, 0) / n) * (count_b.get(l, 0) / n) for l in labels)
    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def fleiss_kappa(item_label_lists: List[Sequence[Any]]) -> float:
    """Fleiss' kappa: a fixed number of raters per item, many items.

    `item_label_lists` is a list (one entry per item) of the labels assigned to
    that item. Each item must have the same number of ratings.
    """
    items = [list(x) for x in item_label_lists if x]
    if not items:
        return 0.0
    n_raters = len(items[0])
    if any(len(it) != n_raters for it in items) or n_raters < 2:
        raise ValueError("Fleiss' kappa needs an equal (>=2) number of raters per item")

    categories = sorted({lbl for it in items for lbl in it}, key=str)
    N = len(items)

    # P_i: agreement for each item
    p_items = []
    cat_totals = Counter()
    for it in items:
        counts = Counter(it)
        for c in categories:
            cat_totals[c] += counts.get(c, 0)
        p_i = (sum(v * v for v in counts.values()) - n_raters) / (n_raters * (n_raters - 1))
        p_items.append(p_i)

    p_bar = sum(p_items) / N
    pj = {c: cat_totals[c] / (N * n_raters) for c in categories}
    pe_bar = sum(v * v for v in pj.values())
    if pe_bar == 1.0:
        return 1.0
    return round((p_bar - pe_bar) / (1 - pe_bar), 4)


def krippendorff_alpha(annotations: Dict[str, Dict[str, Any]]) -> float:
    """Krippendorff's alpha (nominal), tolerant of missing annotations.

    `annotations` maps annotator_id -> {item_id: label}. Items may be missing
    for some annotators. Uses the standard nominal-difference formulation.
    """
    # Build per-item list of labels across annotators.
    items: Dict[str, List[Any]] = {}
    for coder, labelmap in annotations.items():
        for item_id, label in labelmap.items():
            items.setdefault(item_id, []).append(label)

    # Only items with >= 2 ratings contribute to reliability.
    units = [labels for labels in items.values() if len(labels) >= 2]
    if not units:
        return 0.0

    # Observed disagreement Do
    total_pairs = 0
    observed_disagree = 0
    value_counts: Counter = Counter()
    for labels in units:
        m = len(labels)
        for x, y in combinations(labels, 2):
            observed_disagree += 0 if x == y else 1
        total_pairs += m * (m - 1) // 2
        value_counts.update(labels)

    if total_pairs == 0:
        return 0.0
    Do = observed_disagree / total_pairs

    # Expected disagreement De from the global value distribution.
    total_vals = sum(value_counts.values())
    if total_vals < 2:
        return 0.0
    same = sum(c * (c - 1) for c in value_counts.values())
    Pe_same = same / (total_vals * (total_vals - 1))
    De = 1 - Pe_same
    if De == 0:
        return 1.0
    return round(1 - (Do / De), 4)
