"""Load documents from local JSONL / Parquet / text files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

from ..schema import Document
from .base import Collector


class FileCollector(Collector):
    source_type = "file"

    def __init__(self, cfg: Dict[str, Any]):
        self.path = Path(cfg["path"])
        self.license = cfg.get("license")
        self.text_field = cfg.get("text_field", "text")
        self.id_field = cfg.get("id_field", "id")

    def collect(self) -> Iterator[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Source file not found: {self.path}")
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            yield from self._read_jsonl()
        elif suffix == ".parquet":
            yield from self._read_parquet()
        else:
            yield from self._read_text()

    def _read_jsonl(self) -> Iterator[Document]:
        with self.path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(self.text_field, "")
                if not text:
                    continue
                doc_id = str(obj.get(self.id_field, f"{self.path.stem}-{i}"))
                meta = {k: v for k, v in obj.items()
                        if k not in (self.text_field, self.id_field)}
                # carry an explicit pre-tagged language if the source provides one
                lang = meta.pop("language", None)
                doc = self._make_doc(doc_id, text, str(self.path),
                                     license=self.license, **meta)
                if lang:
                    doc.language = lang
                    doc.language_confidence = 1.0
                yield doc

    def _read_parquet(self) -> Iterator[Document]:
        import pandas as pd

        df = pd.read_parquet(self.path)
        for i, row in df.iterrows():
            text = str(row.get(self.text_field, "") or "")
            if not text:
                continue
            doc_id = str(row.get(self.id_field, f"{self.path.stem}-{i}"))
            yield self._make_doc(doc_id, text, str(self.path), license=self.license)

    def _read_text(self) -> Iterator[Document]:
        text = self.path.read_text(encoding="utf-8")
        # one document per non-empty paragraph
        for i, para in enumerate(p for p in text.split("\n\n") if p.strip()):
            yield self._make_doc(f"{self.path.stem}-{i}", para.strip(),
                                 str(self.path), license=self.license)
