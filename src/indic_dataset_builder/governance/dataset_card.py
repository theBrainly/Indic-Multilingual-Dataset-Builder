"""Render a Hugging Face style dataset card (README.md) from a run manifest.

A dataset card documents provenance, composition, curation steps, and known
limitations — table stakes for responsible, reusable training data.
"""
from __future__ import annotations

from typing import Any, Dict


def render_dataset_card(manifest: Dict[str, Any]) -> str:
    d = manifest
    div = d.get("diversity", {})
    stats = d.get("stage_stats", {})

    def _table(mapping: Dict[str, Any]) -> str:
        if not mapping:
            return "_none_"
        return "\n".join(f"| {k} | {v} |" for k, v in mapping.items())

    lang_rows = _table(div.get("language_distribution", {}))
    source_rows = _table(d.get("sources", {}))
    license_rows = _table(d.get("licenses", {}))
    stage_rows = _table(stats)

    return f"""---
dataset_name: {d.get('dataset_name')}
version: {d.get('content_version')}
language: {list(div.get('language_distribution', {}).keys())}
license: {list(d.get('licenses', {}).keys())}
---

# {d.get('dataset_name')}

Curated with **indic-dataset-builder**. This card is generated automatically
from the run manifest for full reproducibility.

- **Content version:** `{d.get('content_version')}`
- **Generated at:** {d.get('generated_at')}
- **Records:** {d.get('num_records'):,}
- **Languages:** {div.get('num_languages')}
- **Total tokens:** {div.get('total_tokens'):,}
- **Vocabulary size:** {div.get('vocabulary_size'):,}
- **Language balance (norm. entropy):** {div.get('language_balance')}

## Language distribution
| language | documents |
|---|---|
{lang_rows}

## Sources
| source type | documents |
|---|---|
{source_rows}

## Licenses
| license | documents |
|---|---|
{license_rows}

## Curation summary (records removed per stage)
| stage | removed |
|---|---|
{stage_rows}

## Curation pipeline
1. **Collect** — multi-source acquisition with provenance.
2. **Clean** — Unicode/Indic normalization, script-based language ID,
   heuristic quality filtering.
3. **Align** — parallel pairs built across languages.
4. **Validate** — exact, near-duplicate (MinHash/LSH) and semantic
   (embeddings + FAISS) deduplication; benchmark contamination removal.
5. **Enrich** — per-document metadata + corpus diversity analysis.
6. **Export & Govern** — JSONL/Parquet output with manifest + this card.

## Known limitations
- Script-based language ID reports a primary language for scripts shared by
  multiple languages (e.g. Devanagari → hi); refine with fastText/CLD3 if exact
  language separation is required.
- Contamination detection covers the supplied benchmark set only.
"""
