"""Preference-dataset validation (for RLHF / DPO style data).

Preference data quality issues are subtle: identical chosen/rejected text,
duplicated pairs, and *cyclic* preferences (A>B, B>C, C>A) that make the data
logically inconsistent. This validator flags all three and reports a clean set.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .schema import PreferencePair


def _has_cycle(edges: Dict[str, set]) -> bool:
    """Detect a cycle in the preference graph (chosen -> rejected edges)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = defaultdict(int)

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in edges.get(node, ()):  # node preferred over nxt
            if color[nxt] == GRAY:
                return True
            if color[nxt] == WHITE and visit(nxt):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in list(edges))


def validate_preferences(pairs: List[PreferencePair]) -> Dict[str, Any]:
    """Validate a set of preference pairs and return a report + clean subset."""
    clean: List[PreferencePair] = []
    issues: List[Dict[str, str]] = []
    seen: set = set()

    # Build a per-prompt preference graph to check transitivity/cycles.
    graphs: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for p in pairs:
        if p.chosen.strip() == p.rejected.strip():
            issues.append({"id": p.id, "issue": "chosen_equals_rejected"})
            continue
        key = (p.prompt, p.chosen, p.rejected)
        if key in seen:
            issues.append({"id": p.id, "issue": "duplicate_pair"})
            continue
        seen.add(key)
        graphs[p.prompt][p.chosen].add(p.rejected)
        clean.append(p)

    cyclic_prompts = [prompt for prompt, g in graphs.items() if _has_cycle(g)]
    for prompt in cyclic_prompts:
        issues.append({"id": prompt[:40], "issue": "cyclic_preferences"})

    return {
        "total": len(pairs),
        "clean": len(clean),
        "issues": issues,
        "cyclic_prompts": len(cyclic_prompts),
        "clean_pairs": clean,
    }
