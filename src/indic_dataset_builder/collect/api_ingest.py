"""API-based ingestion collector.

Handles paginated JSON APIs (the common case for ingesting documents,
comments, or transcripts). Network is optional: a `fixture` block lets the
demo and tests run a deterministic ingestion offline.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List

from ..schema import Document
from .base import Collector


class APICollector(Collector):
    source_type = "api"

    def __init__(self, cfg: Dict[str, Any]):
        self.endpoint = cfg.get("endpoint", "")
        self.text_field = cfg.get("text_field", "text")
        self.id_field = cfg.get("id_field", "id")
        self.records_path = cfg.get("records_path", "data")  # JSON key holding list
        self.max_pages = int(cfg.get("max_pages", 1))
        self.license = cfg.get("license")
        self.headers = cfg.get("headers", {})
        self.fixture: List[Dict[str, Any]] | None = cfg.get("fixture")

    def collect(self) -> Iterator[Document]:
        if self.fixture is not None:
            yield from self._from_records(self.fixture)
            return
        try:
            import requests  # type: ignore
        except ImportError:
            raise RuntimeError(
                "API ingestion requires `requests`. Install `.[full]` or "
                "provide a `fixture` list in the source config."
            )
        for page in range(self.max_pages):
            resp = requests.get(
                self.endpoint, headers=self.headers,
                params={"page": page}, timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            records = payload.get(self.records_path, payload) \
                if isinstance(payload, dict) else payload
            if not records:
                break
            yield from self._from_records(records)

    def _from_records(self, records: List[Dict[str, Any]]) -> Iterator[Document]:
        for i, rec in enumerate(records):
            text = rec.get(self.text_field, "")
            if not text:
                continue
            doc_id = str(rec.get(self.id_field, f"api-{i}"))
            meta = {k: v for k, v in rec.items()
                    if k not in (self.text_field, self.id_field)}
            yield self._make_doc(doc_id, text, self.endpoint or "api-fixture",
                                 license=self.license, **meta)
