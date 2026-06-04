"""Common-Crawl / WARC-style collector.

Reads a minimal WARC-like record stream. A full implementation would use
`warcio` to stream Common Crawl segments from S3; here we parse a compact
line-delimited WARC subset so the concept is demonstrable offline and the
extraction code path (HTML -> text) is shared with the web collector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

from ..schema import Document
from .base import Collector
from .web_scraper import _extract_text_from_html


class WARCCollector(Collector):
    source_type = "warc"

    def __init__(self, cfg: Dict[str, Any]):
        self.path = Path(cfg["path"])
        self.license = cfg.get("license", "Common Crawl ToU")

    def collect(self) -> Iterator[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"WARC file not found: {self.path}")
        # Prefer warcio for real .warc/.warc.gz files when available.
        try:
            from warcio.archiveiterator import ArchiveIterator  # type: ignore

            with self.path.open("rb") as stream:
                for i, record in enumerate(ArchiveIterator(stream)):
                    if record.rec_type != "response":
                        continue
                    url = record.rec_headers.get_header("WARC-Target-URI")
                    html = record.content_stream().read().decode("utf-8", "ignore")
                    text = _extract_text_from_html(html)
                    if text:
                        yield self._make_doc(f"cc-{i}", text, url or str(self.path),
                                             license=self.license)
            return
        except ImportError:
            pass

        # Fallback: simple "URI<TAB>HTML" line format for offline demos.
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            if "\t" not in line:
                continue
            url, html = line.split("\t", 1)
            text = _extract_text_from_html(html)
            if text:
                yield self._make_doc(f"cc-{i}", text, url, license=self.license)
