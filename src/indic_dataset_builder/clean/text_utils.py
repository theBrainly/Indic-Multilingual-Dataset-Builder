"""Indic-aware text tokenization helpers.

Python's stdlib ``\\w`` treats Indic combining vowel signs (matras, Unicode
category Mn) as non-word characters, which makes word counts and symbol ratios
badly wrong for Devanagari, Tamil, Bengali, etc. We use the third-party
``regex`` module with Unicode property classes so a "word" correctly includes
its combining marks. All quality/metric code shares these helpers so behaviour
is consistent across the pipeline.
"""
from __future__ import annotations

from typing import List

import regex  # third-party, supports \p{...}

# A word = run of letters/numbers plus any attached combining marks.
_WORD_RE = regex.compile(r"[\p{L}\p{N}][\p{L}\p{N}\p{M}]*", regex.UNICODE)
# A symbol = a visible character that is not a letter, number, mark or space.
_SYMBOL_RE = regex.compile(r"[^\p{L}\p{N}\p{M}\s]", regex.UNICODE)
_DIGIT_RE = regex.compile(r"\p{Nd}", regex.UNICODE)


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def symbol_count(text: str) -> int:
    return len(_SYMBOL_RE.findall(text))


def digit_count(text: str) -> int:
    return len(_DIGIT_RE.findall(text))
