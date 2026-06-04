"""A real Scrapy spider for polite, large-scale Indic web crawling.

Run with Scrapy installed:

    scrapy runspider src/indic_dataset_builder/collect/scrapy_spider.py \
        -a start_urls="https://example.org" -o crawled.jsonl

The spider respects robots.txt and an autothrottle/download delay, extracts
main text, tags the language via the project's script-based detector, and emits
records in the same shape the file collector reads back in — so the crawl feeds
straight into the curation pipeline.
"""
from __future__ import annotations

try:
    import scrapy  # type: ignore
    _BASE = scrapy.Spider
except Exception:  # Scrapy not installed; keep module importable.
    scrapy = None
    _BASE = object


class IndicWebSpider(_BASE):  # type: ignore[misc]
    name = "indic_web"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.0,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
        "USER_AGENT": "indic-dataset-builder/0.1 (+research; respect-robots)",
        "DEPTH_LIMIT": 2,
    }

    def __init__(self, start_urls: str = "", allowed_domains: str = "",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [u for u in start_urls.split(",") if u]
        self.allowed_domains = [d for d in allowed_domains.split(",") if d]

    def parse(self, response):  # pragma: no cover - needs live Scrapy engine
        from .web_scraper import _extract_text_from_html
        from ..clean.language_detect import LanguageDetector

        text = _extract_text_from_html(response.text)
        if len(text) >= 50:
            lang, conf = LanguageDetector().detect(text)
            yield {
                "id": response.url,
                "text": text,
                "language": lang,
                "language_confidence": conf,
                "source_url": response.url,
            }
        # follow in-domain links
        for href in response.css("a::attr(href)").getall():
            yield response.follow(href, callback=self.parse)
