"""Search/indexing backends for retrieval, dedup lookups and QA spot-checks.

Provides a uniform interface with two backends:
- ElasticsearchIndex — when an Elasticsearch server + client are available.
- InMemoryIndex — a dependency-free inverted index fallback that supports the
  same keyword search and exact-match lookups, so the demo/tests run offline.
"""
from .search_index import InMemoryIndex, build_index  # noqa: F401
