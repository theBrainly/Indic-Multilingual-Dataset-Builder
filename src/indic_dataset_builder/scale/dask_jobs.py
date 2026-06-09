"""Distributed exact-dedup + heuristic filtering over sharded Parquet.

The two heaviest per-document stages, expressed as data-parallel operations:

- **Heuristic filter** -> `map_partitions`: each partition filtered independently.
- **Exact dedup** -> shuffle-by-hash + drop_duplicates: the canonical web-scale
  dedup pattern (group identical content hashes, keep one).

With Dask installed this runs across partitions/workers; otherwise it falls
back to pandas with the same semantics. Either way the result is deterministic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

from ..clean.normalizer import normalize_text
from ..clean.text_utils import digit_count, symbol_count, word_count

try:  # pragma: no cover - depends on environment
    import dask.dataframe as dd  # type: ignore
    DEFAULT_BACKEND = "dask"
except Exception:  # pragma: no cover
    dd = None
    DEFAULT_BACKEND = "pandas"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _passes_filter(text: str, cfg: Dict[str, Any]) -> bool:
    n = len(text)
    if n < cfg.get("min_chars", 20) or n > cfg.get("max_chars", 100_000):
        return False
    if word_count(text) < cfg.get("min_words", 4):
        return False
    if symbol_count(text) / max(n, 1) > cfg.get("max_symbol_ratio", 0.30):
        return False
    if digit_count(text) / max(n, 1) > cfg.get("max_digit_ratio", 0.40):
        return False
    return True


def _prepare_partition(pdf, text_col: str, cfg: Dict[str, Any]):
    """Pandas-level transform applied per partition (Dask) or once (pandas)."""
    pdf = pdf.copy()
    pdf[text_col] = pdf[text_col].fillna("").map(normalize_text)
    pdf = pdf[pdf[text_col].map(lambda t: _passes_filter(t, cfg))]
    pdf["content_hash"] = pdf[text_col].map(_content_hash)
    return pdf


def distributed_dedup_filter(
    input_path: str,
    output_path: str,
    text_col: str = "text",
    filter_cfg: Dict[str, Any] | None = None,
    backend: str | None = None,
    npartitions: int = 4,
) -> Dict[str, Any]:
    """Filter + exact-dedup a Parquet dataset at scale.

    Returns a stats dict and writes the curated Parquet to `output_path`.
    """
    import pandas as pd

    cfg = filter_cfg or {}
    backend = backend or DEFAULT_BACKEND
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if backend == "dask" and dd is not None:
        ddf = dd.read_parquet(input_path)
        n_input = int(ddf.shape[0].compute())
        meta = ddf._meta.assign(content_hash="x")
        ddf = ddf.map_partitions(_prepare_partition, text_col, cfg, meta=meta)
        # shuffle-by-hash then keep first per hash
        ddf = ddf.drop_duplicates(subset=["content_hash"])
        result = ddf.compute()
    else:
        df = pd.read_parquet(input_path)
        n_input = len(df)
        df = _prepare_partition(df, text_col, cfg)
        result = df.drop_duplicates(subset=["content_hash"])

    result.to_parquet(output_path, engine="pyarrow", index=False)
    return {
        "backend": backend if (backend == "dask" and dd is not None) else "pandas",
        "input_rows": n_input,
        "output_rows": len(result),
        "removed": n_input - len(result),
        "output_path": output_path,
    }
