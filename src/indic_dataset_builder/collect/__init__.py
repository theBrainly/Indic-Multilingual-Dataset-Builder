"""Data acquisition: files, web scraping, API ingestion, WARC/Common-Crawl."""
from .base import Collector, build_collector  # noqa: F401
from .file_loader import FileCollector  # noqa: F401
from .web_scraper import WebCollector  # noqa: F401
from .api_ingest import APICollector  # noqa: F401
from .warc_reader import WARCCollector  # noqa: F401
