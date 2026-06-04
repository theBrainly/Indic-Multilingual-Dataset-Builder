"""Web scraping collector (requests + BeautifulSoup), with offline fixtures.

In production this is where a Scrapy spider or a polite async crawler would
live. To keep the demo runnable without network access, the collector falls
back to bundled HTML fixtures and always honours robots-style politeness
settings (delay, user-agent) when it does hit the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List

from ..schema import Document
from .base import Collector


def _extract_text_from_html(html: str) -> str:
    """Extract main text from HTML, preferring BeautifulSoup if available."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except Exception:
        # Dependency-free fallback: crude tag stripping.
        import re

        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return " ".join(text.split())


class WebCollector(Collector):
    source_type = "web"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.user_agent = cfg.get(
            "user_agent", "indic-dataset-builder/0.1 (+research; respect-robots)"
        )
        self.delay = float(cfg.get("delay", 1.0))
        self.license = cfg.get("license")
        self.urls: List[str] = self._load_urls(cfg)
        self.fixtures_dir = cfg.get("fixtures_dir")

    def _load_urls(self, cfg: Dict[str, Any]) -> List[str]:
        if "urls" in cfg:
            return list(cfg["urls"])
        if "urls_file" in cfg and Path(cfg["urls_file"]).exists():
            return [u.strip() for u in Path(cfg["urls_file"]).read_text().splitlines()
                    if u.strip() and not u.startswith("#")]
        return []

    def collect(self) -> Iterator[Document]:
        # Offline mode: read bundled HTML fixtures.
        if self.fixtures_dir and Path(self.fixtures_dir).exists():
            for i, fp in enumerate(sorted(Path(self.fixtures_dir).glob("*.html"))):
                text = _extract_text_from_html(fp.read_text(encoding="utf-8"))
                if text:
                    yield self._make_doc(f"web-fixture-{i}", text, fp.name,
                                         license=self.license, retrieved="fixture")
            return

        # Live mode (only when requests is installed and URLs are configured).
        try:
            import time
            import requests  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Live web scraping requires `requests`. Install with "
                "`pip install -e '.[full]'` or supply `fixtures_dir`."
            )
        headers = {"User-Agent": self.user_agent}
        for i, url in enumerate(self.urls):
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            text = _extract_text_from_html(resp.text)
            if text:
                yield self._make_doc(f"web-{i}", text, url,
                                     license=self.license,
                                     status=resp.status_code)
            time.sleep(self.delay)  # politeness
