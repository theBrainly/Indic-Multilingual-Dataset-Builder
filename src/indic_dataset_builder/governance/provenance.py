"""Provenance, versioning and reproducible manifests.

Reproducible curation means a dataset version is a deterministic function of
its inputs and config. We compute the version as a hash over the sorted
content hashes of the kept documents plus the config, and emit a manifest that
records exactly how the corpus was produced (counts dropped per stage, source
breakdown, config snapshot). This is the audit trail a model team needs to
trust and reproduce a corpus.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..schema import Document


def dataset_version(docs: List[Document], config: Dict[str, Any]) -> str:
    """Deterministic content-addressed version id for the curated corpus."""
    hasher = hashlib.sha256()
    for h in sorted(d.hash or "" for d in docs):
        hasher.update(h.encode("utf-8"))
    hasher.update(json.dumps(config, sort_keys=True, default=str).encode("utf-8"))
    return hasher.hexdigest()[:16]


def build_manifest(
    kept: List[Document],
    config: Dict[str, Any],
    stage_stats: Dict[str, Any],
    diversity: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the full run manifest."""
    ds_cfg = config.get("dataset", {})
    sources: Dict[str, int] = {}
    licenses: Dict[str, int] = {}
    for d in kept:
        sources[d.source.source_type] = sources.get(d.source.source_type, 0) + 1
        lic = d.source.license or "unknown"
        licenses[lic] = licenses.get(lic, 0) + 1

    return {
        "dataset_name": ds_cfg.get("name", "unnamed"),
        "declared_version": ds_cfg.get("version"),
        "content_version": dataset_version(kept, config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "indic-dataset-builder",
        "num_records": len(kept),
        "sources": sources,
        "licenses": licenses,
        "stage_stats": stage_stats,
        "diversity": diversity,
        "config_snapshot": config,
    }


def write_manifest(manifest: Dict[str, Any], path: str) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
