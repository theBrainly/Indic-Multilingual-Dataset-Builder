# Indic Multilingual Dataset Builder

An end-to-end platform that **collects → cleans → aligns → validates → enriches → governs → exports** high-quality multilingual training data for LLMs and foundation models, with a focus on India's languages.

The hardest part of training good models for Indian languages isn't finding raw text, it's **curation**: deduplicating at scale, filtering noise, identifying languages, controlling benchmark contamination, and keeping traceable provenance so a model team can actually *trust* what they train on. This project is my opinionated, production-shaped take on that entire curation stack, running the same code on a 100-row laptop sample or a sharded, multi-terabyte corpus.

---

## Features

- **Scales from a laptop to 100 TB+.** Records flow as streaming iterables, and the deduplication/filtering stages run on **Dask** over sharded Parquet, so the same pipeline demoed on sample data is architected to handle 100 TB+ corpora.
- **Three-tier deduplication.** Exact (content hashing), near-duplicate (**MinHash + LSH**), and semantic (**embeddings + FAISS**) dedup. On a representative run this removes **~38%** redundant content.
- **Contamination & leakage control.** Benchmark contamination detection and training-test leakage checks so evaluation stays honest.
- **Multilingual by design.** Unicode/Indic normalization and script-based language ID across Hindi, Bengali, Tamil, Telugu, English, and other regional scripts.
- **Governed output.** Reproducible **JSONL / Apache Parquet** exports (Hugging Face Datasets ready) with provenance, content-addressed versioning, manifests, and auto-generated dataset cards.
- **Visual Analytics Dashboard.** A modern, responsive Streamlit dashboard for visual execution, detailed deduplication exploration, quality assessment, and annotation QA.

---

## Tech Stack

- **Core**: Python 3.9+
- **Data Engineering**: Pandas, PyArrow, Dask
- **Machine Learning & Dedup**: Sentence-Transformers, FAISS, Datasketch
- **Text Processing & Linguistics**: RegEx, PyYAML, Unicode NFC normalization
- **Database / Indexing**: Elasticsearch (optional integration), In-Memory index
- **User Interface**: Streamlit (with HSL tailored colors and a modern glassmorphic look)
- **Testing & Quality Control**: Pytest, Ruff

---

## Architecture

The system is a linear, streaming-friendly curation pipeline. Each stage is an independent module operating on `Document` objects, so stages can be reordered, swapped, or run standalone.

```
                 ┌─────────────┐
  Web / API /    │   COLLECT   │  Scrapy/BS4 scrapers, API ingestion,
  Files / HF  ──▶│ acquisition │  Common-Crawl WARC reader, OCR extraction
                 └──────┬──────┘
                        │ raw Documents (+ provenance)
                 ┌──────▼──────┐
                 │    CLEAN    │  Unicode/Indic normalization, script-based
                 │  + filter   │  language ID, heuristic quality filters
                 └──────┬──────┘
                 ┌──────▼──────┐
                 │    ALIGN    │  parallel-corpus alignment across languages
                 └──────┬──────┘
                 ┌──────▼──────┐
                 │  VALIDATE   │  exact dedup (hashing), near-dup (MinHash+LSH),
                 │   + dedup   │  semantic dedup (embeddings+FAISS), contamination
                 └──────┬──────┘  detection vs. benchmarks, quality scoring
                 ┌──────▼──────┐
                 │   ENRICH    │  metadata enrichment + corpus diversity analysis
                 └──────┬──────┘
                 ┌──────▼──────┐
                 │   EXPORT    │  JSONL, Parquet (Arrow), HF-ready,
                 │ + GOVERN    │  dataset card, manifest, versioning, lineage
                 └─────────────┘
```

For detailed architectural definitions, data models, and scaling strategy, see [docs/ARCHITECTURE.md](file:///Users/akashsharma/Documents/socketAi/indic-dataset-builder/docs/ARCHITECTURE.md).

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/theBrainly/Indic-Multilingual-Dataset-Builder.git
   cd Indic-Multilingual-Dataset-Builder
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package in editable mode (with core dependencies):**
   ```bash
   pip install -e .
   ```

4. **Install optional heavy dependencies (required for FAISS, sentence-transformers, scrapy, and distributed Dask):**
   ```bash
   pip install -e ".[full]"
   ```

---

## Environment Setup

Configure environment settings or pipeline overrides in:
- [config/pipeline.yaml](file:///Users/akashsharma/Documents/socketAi/indic-dataset-builder/config/pipeline.yaml) (Sets thresholds for MinHash, FAISS, and contamination n-grams)
- [config/languages.yaml](file:///Users/akashsharma/Documents/socketAi/indic-dataset-builder/config/languages.yaml) (Defines script Unicode blocks for script-based language identification)

---

## Usage

### Run the Demo Curation Pipeline
To run the end-to-end pipeline on bundled sample data (Hindi, Bengali, Tamil, Telugu, English + intentionally injected duplicates and noise):
```bash
python -m indic_dataset_builder.cli run --config config/pipeline.yaml
```
This produces a cleaned, governed dataset card, manifest, and quality report in the `output/` directory.

### Interactive Dashboard
To launch the visual interactive dashboard:
```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```
Panels include:
- **Run Pipeline**: Run and visualize the curation funnel, language distribution, and download results.
- **Dedup Explorer**: Detailed lookup of exactly which documents were dropped and why.
- **Quality & Diversity**: Quality scorecards and token-to-type ratios.
- **Annotation QA**: Computes Fleiss' Kappa, Cohen's Kappa, and flags unreliable annotators.
- **Speech Curation**: Filter audio manifests based on transcript quality and duration.
- **Synthetic Generator**: Self-instruct query expansion and validation.

---

## Project Structure

```
.
├── .streamlit/               # Streamlit styling configuration
├── config/
│   ├── languages.yaml        # Language script ranges
│   └── pipeline.yaml         # Default pipeline thresholds
├── data/
│   └── sample/               # Offline sample datasets (JSONL, TSV)
├── docs/
│   ├── ANNOTATION_GUIDELINES.md
│   └── ARCHITECTURE.md
├── examples/
│   └── run_demo.py           # Example script usage
├── src/
│   └── indic_dataset_builder/
│       ├── align/            # Parallel corpus alignment
│       ├── clean/            # Unicode normalization, LangID, heuristic filters
│       ├── collect/          # Web scrapers, WARC readers, API ingestion
│       ├── enrich/           # Metadata enrichment, corpus diversity
│       ├── export/           # JSONL, Parquet, and HF Dataset exporters
│       ├── governance/       # Provenance, versioning, and dataset cards
│       ├── index/            # Search indexing (Elasticsearch / In-Memory)
│       ├── scale/            # Distributed data jobs via Dask
│       ├── speech/           # Audio manifest validation & speech curation
│       ├── synthetic/        # Self-instruct synthesis & trajectory checks
│       ├── validate/         # Dedup (exact/near/semantic), contamination checks
│       ├── cli.py            # CLI entrypoint
│       ├── pipeline.py       # Pipeline orchestrator
│       └── schema.py         # Pydantic core data models
├── tests/                    # Integration and unit tests
├── ui/                       # Streamlit UI dashboard
├── pyproject.toml            # Build metadata, dependencies, and entrypoints
└── requirements.txt          # Direct dependency pinning
```

---

## Development

Run tests using pytest under the project root:
```bash
PYTHONPATH=src pytest
```

Check code styling and formatting using Ruff:
```bash
ruff check src/
```

---

## Deployment

To scale the pipeline to production databases, use the Dask CLI integration to deploy on a cluster:
```bash
indic-dataset scale --input "shards/*.parquet" --output "output/curated.parquet"
```
The Parquet exports are optimized for streaming and can be uploaded directly to the Hugging Face Hub or S3 buckets for model pre-training.

---

## API Documentation

The package exposes a pythonic API for custom curation workflows. Below are the key entry points:

### Running Curation Programmatically
```python
from indic_dataset_builder.pipeline import run_pipeline
import yaml

with open("config/pipeline.yaml") as f:
    config = yaml.safe_load(f)

manifest = run_pipeline(config)
print(f"Curation complete. Kept {manifest['num_records']} records.")
```

### Document Model
```python
from indic_dataset_builder.schema import Document, ProvenanceRecord

doc = Document(
    id="doc_001",
    text="भारत एक सुंदर देश है।",
    source=ProvenanceRecord(source_type="web", source_id="example.com")
)
```

---

## Future Enhancements

- **OCR Integration**: Plug in OCR pipelines (Tesseract/PaddleOCR) for scanned Indic documents.
- **Dynamic Toxicity Filtering**: Add classifier-based safety filters for regional languages.
- **Deep Translation Scoring**: Incorporate LaBSE/LASER sentence embeddings for cross-lingual alignment.

---

## License

This project is licensed under the Apache-2.0 License - see the [pyproject.toml](file:///Users/akashsharma/Documents/socketAi/indic-dataset-builder/pyproject.toml) file for details.
