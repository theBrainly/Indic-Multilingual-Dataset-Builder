"""Text normalization, with special care for Indic scripts.

Cleaning Indic text well matters because the same logical character can be
encoded multiple ways (e.g. Devanagari nukta forms), and crawled text is full
of zero-width joiners, NBSPs, and inconsistent whitespace. Normalizing here
makes downstream dedup and language ID far more reliable.
"""
from __future__ import annotations

import re
import unicodedata

from ..schema import Document, Stage

# Zero-width and invisible characters frequently found in crawled Indic text.
_INVISIBLES = dict.fromkeys(map(ord, [
    "\u200b",  # zero width space
    "\u200c",  # ZWNJ — NOTE: meaningful in some Indic scripts; kept by default
    "\u200d",  # ZWJ  — meaningful in some Indic scripts; kept by default
    "\ufeff",  # BOM
    "\u00ad",  # soft hyphen
]), None)

# By default we only strip truly invisible junk that never carries meaning.
_ALWAYS_STRIP = {ord("\u200b"): None, ord("\ufeff"): None, ord("\u00ad"): None}

_MULTISPACE = re.compile(r"[ \t\u00a0\u2000-\u200a]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_DANDA_FIX = re.compile(r"\s+([।॥])")  # tidy spacing before Devanagari danda


def normalize_text(text: str, strip_joiners: bool = False) -> str:
    """Normalize a single string.

    - Unicode NFC (canonical composition) so visually identical strings hash
      identically — critical for exact dedup across Indic encodings.
    - Strip invisible/BOM characters.
    - Collapse whitespace, trim trailing spaces per line.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    table = dict(_INVISIBLES) if strip_joiners else dict(_ALWAYS_STRIP)
    text = text.translate(table)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTISPACE.sub(" ", text)
    text = _DANDA_FIX.sub(r"\1", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def normalize_document(doc: Document, strip_joiners: bool = False) -> Document:
    """Normalize a document's text in place and refresh its content hash."""
    doc.text = normalize_text(doc.text, strip_joiners=strip_joiners)
    doc.refresh_hash()
    return doc.touch(Stage.NORMALIZED)
