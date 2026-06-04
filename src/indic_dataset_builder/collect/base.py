"""Collector base class + factory.

Collectors are the acquisition layer. Each yields :class:`Document`s with
provenance attached, so downstream stages never have to guess where data
came from.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Iterator

from ..schema import Document, ProvenanceRecord, Stage


class Collector(abc.ABC):
    """Abstract acquisition source."""

    source_type: str = "base"

    @abc.abstractmethod
    def collect(self) -> Iterator[Document]:
        """Yield documents with provenance attached."""
        raise NotImplementedError

    def _make_doc(self, doc_id: str, text: str, source_id: str,
                  license: str | None = None, **meta: Any) -> Document:
        doc = Document(
            id=doc_id,
            text=text,
            source=ProvenanceRecord(
                source_type=self.source_type,
                source_id=source_id,
                license=license,
            ),
            metadata=meta,
        )
        return doc.touch(Stage.COLLECTED)


def build_collector(source_cfg: Dict[str, Any]) -> Collector:
    """Factory: turn a config block into a concrete collector."""
    from .file_loader import FileCollector
    from .web_scraper import WebCollector
    from .api_ingest import APICollector
    from .warc_reader import WARCCollector

    stype = source_cfg["type"]
    registry = {
        "file": FileCollector,
        "web": WebCollector,
        "api": APICollector,
        "warc": WARCCollector,
    }
    if stype not in registry:
        raise ValueError(f"Unknown collector type: {stype!r}")
    return registry[stype](source_cfg)
