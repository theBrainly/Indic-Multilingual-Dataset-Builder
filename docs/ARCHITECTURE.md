# Architecture

The system is a linear, streaming-friendly curation pipeline. Each stage is an
independent module operating on `Document` objects, so stages can be reordered,
swapped, or run standalone.

## Data model

`Document` (in `schema.py`) is the single unit of data. It carries:

- `text`, `language`, `language_confidence`
- `source` — a `ProvenanceRecord` (type, id, collected_at, license)
- `metadata` — free-form enrichment (counts, quality score, flags)
- `stages` — ordered list of pipeline stages that touched it (audit trail)
- `hash` — SHA-256 of normalized text (exact dedup + versioning)
- `dropped` / `drop_reason` — filtering decisions are recorded, not silent

`AlignedPair` links two `Document`s as a parallel/translation pair.

## Stages

| Stage | Module | Key technique |
|---|---|---|
| Collect | `collect/` | pluggable collectors (file, web, API, WARC) via a factory |
| Normalize | `clean/normalizer.py` | Unicode NFC + Indic-aware whitespace/joiner handling |
| Language ID | `clean/language_detect.py` | Unicode-script counting (dependency-free) |
| Filter | `clean/heuristic_filters.py` | Gopher/C4-style heuristic rules |
| Align | `align/aligner.py` | length-ratio + lexical-anchor scoring |
| Exact dedup | `validate/exact_dedup.py` | content-hash equality |
| Near dedup | `validate/minhash_lsh.py` | MinHash + LSH over char shingles |
| Semantic dedup | `validate/semantic_dedup.py` | embeddings + FAISS / cosine |
| Contamination | `validate/contamination.py` | n-gram overlap vs. benchmarks |
| Quality score | `validate/quality_scorer.py` | interpretable 0-1 blend |
| Enrich | `enrich/` | per-doc metadata + corpus diversity |
| Export | `export/` | JSONL, Parquet (Arrow), HF datasets |
| Govern | `governance/` | manifest, content version, dataset card |

## Graceful degradation

Heavy/optional dependencies are isolated behind interfaces:

- **Embeddings**: `validate/embedder.py` returns a `SentenceTransformerEmbedder`
  (LaBSE, multilingual) when available, else a dependency-free
  `HashingEmbedder`. Both emit L2-normalized vectors.
- **FAISS**: used for ANN search when installed; NumPy brute force otherwise.
- **requests / BeautifulSoup**: live scraping when present; offline HTML
  fixtures and crude tag-stripping otherwise.
- **datasketch**: used for MinHash/LSH; a hand-rolled banded LSH is the fallback.
- **HF datasets**: `save_to_disk` when present; JSONL fallback otherwise.

This guarantees the full pipeline runs anywhere, which matters for CI and for
reproducible demos, while still using best-in-class tools in production.

## Reproducibility & governance

A run is defined by `config + input manifest`. The dataset `content_version`
is a hash over the sorted content hashes of kept documents plus the config, so
identical inputs yield an identical version. The `manifest.json` records
per-stage removal counts, source/license breakdown, diversity stats, and the
full config snapshot. `DATASET_CARD.md` is rendered from that manifest.

## Scaling path

The streaming `Iterator[Document]` interface is deliberately Spark/Dask
friendly: collectors map to partitioned readers, the per-document stages map to
`map`/`filter`, and dedup maps to a shuffle-by-hash (exact), LSH-band shuffle
(near), or an ANN index served from FAISS/ScaNN (semantic). Parquet output
integrates directly with Spark and Hugging Face Datasets.
