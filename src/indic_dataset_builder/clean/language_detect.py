"""Script-based language identification for Indic + reference languages.

Most Indic languages live in their own dedicated Unicode block, so counting
characters per script gives fast, dependency-free, and surprisingly accurate
language ID. Where scripts are shared (Devanagari -> hi/mr/ne; Bengali ->
bn/as), we report the script's primary language and expose the ambiguity via
confidence + metadata so a heavier model (fastText/CLD3) can refine later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from ..schema import Document, Stage

_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "languages.yaml"


class LanguageDetector:
    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        # Build an ordered list of (lang_code, ranges) for primary languages only.
        # The first language per script is treated as that script's default.
        self._script_primary: Dict[str, str] = {}
        self._lang_ranges: List[Tuple[str, List[Tuple[int, int]]]] = []
        for code, meta in cfg["languages"].items():
            ranges = [(lo, hi) for lo, hi in meta["unicode_ranges"]]
            self._lang_ranges.append((code, ranges))
            self._script_primary.setdefault(meta["script"], code)
        self._code_to_script = {
            code: meta["script"] for code, meta in cfg["languages"].items()
        }

    def detect(self, text: str) -> Tuple[Optional[str], float]:
        """Return (language_code, confidence) based on script distribution."""
        counts: Dict[str, int] = {}
        total = 0
        for ch in text:
            cp = ord(ch)
            if ch.isspace() or not ch.isprintable():
                continue
            total += 1
            for code, ranges in self._lang_ranges:
                if any(lo <= cp <= hi for lo, hi in ranges):
                    script = self._code_to_script[code]
                    primary = self._script_primary[script]
                    counts[primary] = counts.get(primary, 0) + 1
                    break
        if total == 0 or not counts:
            return None, 0.0
        best_lang = max(counts, key=counts.get)
        confidence = counts[best_lang] / total
        return best_lang, round(confidence, 4)

    def tag(self, doc: Document) -> Document:
        """Attach language + confidence to a document if not already set."""
        if doc.language and doc.language_confidence >= 1.0:
            return doc.touch(Stage.LANGUAGE_TAGGED)  # source pre-tagged
        lang, conf = self.detect(doc.text)
        doc.language = lang
        doc.language_confidence = conf
        if lang and self._code_to_script.get(lang) in {"Devanagari", "Bengali"}:
            # Flag scripts shared by multiple languages for optional refinement.
            doc.metadata["script_ambiguous"] = True
        return doc.touch(Stage.LANGUAGE_TAGGED)
