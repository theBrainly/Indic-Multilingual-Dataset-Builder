"""Heuristic quality filters (Gopher / C4 / RefinedWeb style).

These cheap rules remove the bulk of low-signal junk before the expensive
dedup and semantic stages. Every rejection is recorded on the document
(`drop_reason`) rather than silently discarded, so the curation process is
auditable and filter thresholds can be tuned from the quality report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..schema import Document, Stage
from .text_utils import digit_count, symbol_count, words


@dataclass
class FilterConfig:
    min_chars: int = 20
    max_chars: int = 100_000
    min_words: int = 4
    max_symbol_ratio: float = 0.30
    max_digit_ratio: float = 0.40
    max_repeated_line_ratio: float = 0.30
    min_mean_word_length: float = 1.5
    require_known_language: bool = True

    @classmethod
    def from_dict(cls, d: Dict) -> "FilterConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class HeuristicFilter:
    def __init__(self, config: Optional[FilterConfig] = None):
        self.cfg = config or FilterConfig()

    def evaluate(self, doc: Document) -> Optional[str]:
        """Return a drop reason if the doc fails a rule, else None."""
        text = doc.text
        n_chars = len(text)
        if n_chars < self.cfg.min_chars:
            return f"too_short(<{self.cfg.min_chars}chars)"
        if n_chars > self.cfg.max_chars:
            return f"too_long(>{self.cfg.max_chars}chars)"

        words_list = words(text)
        if len(words_list) < self.cfg.min_words:
            return f"too_few_words(<{self.cfg.min_words})"

        mean_wlen = sum(len(w) for w in words_list) / max(len(words_list), 1)
        if mean_wlen < self.cfg.min_mean_word_length:
            return "mean_word_length_too_low"

        symbol_ratio = symbol_count(text) / max(n_chars, 1)
        if symbol_ratio > self.cfg.max_symbol_ratio:
            return f"symbol_ratio_high({symbol_ratio:.2f})"

        digit_ratio = digit_count(text) / max(n_chars, 1)
        if digit_ratio > self.cfg.max_digit_ratio:
            return f"digit_ratio_high({digit_ratio:.2f})"

        repeat_ratio = self._repeated_line_ratio(text)
        if repeat_ratio > self.cfg.max_repeated_line_ratio:
            return f"repeated_lines({repeat_ratio:.2f})"

        if self.cfg.require_known_language and not doc.language:
            return "unknown_language"

        return None

    @staticmethod
    def _repeated_line_ratio(text: str) -> float:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) <= 1:
            return 0.0
        unique = len(set(lines))
        return 1.0 - (unique / len(lines))

    def apply(self, docs: List[Document]) -> List[Document]:
        """Mark failing docs as dropped; return the surviving documents."""
        kept: List[Document] = []
        for doc in docs:
            reason = self.evaluate(doc)
            doc.touch(Stage.FILTERED)
            if reason:
                doc.drop(reason)
            else:
                kept.append(doc)
        return kept
