"""Dataset exporters.

JSONL is the lingua franca of LLM training data; Parquet (columnar, via Arrow)
is the scalable choice for large corpora and integrates with Spark/Dask and
Hugging Face Datasets. All exporters consume the same flat record dict from
`Document.to_export_dict()`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..schema import Document, Stage


def export_jsonl(docs: List[Document], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            doc.touch(Stage.EXPORTED)
            fh.write(json.dumps(doc.to_export_dict(), ensure_ascii=False) + "\n")
    return path


def export_parquet(docs: List[Document], path: str | Path) -> Path:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in docs:
        doc.touch(Stage.EXPORTED)
        rows.append(doc.to_export_dict())
    df = pd.DataFrame(rows)
    # pyarrow engine writes a portable, columnar file with snappy compression.
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path


def export_hf_dataset(docs: List[Document], path: str | Path) -> Path:
    """Export a Hugging Face `datasets` dataset to disk if the lib is present."""
    path = Path(path)
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        # Graceful fallback: still produce a JSONL the HF loader can read.
        return export_jsonl(docs, path.with_suffix(".jsonl"))

    rows = [doc.to_export_dict() for doc in docs]
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(path))
    return path
