"""Run the full curation pipeline on the bundled sample data.

    python examples/run_demo.py

Ingests intentionally messy multilingual data (duplicates, noise, a paraphrase,
and a benchmark-contaminated document), runs the full pipeline, and prints a
summary plus the location of the curated outputs.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from indic_dataset_builder.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "pipeline.yaml").read_text())
    manifest = run_pipeline(config)

    print("\n" + "=" * 60)
    print("CURATION SUMMARY")
    print("=" * 60)
    print(f"Dataset:          {manifest['dataset_name']}")
    print(f"Content version:  {manifest['content_version']}")
    print(f"Kept records:     {manifest['num_records']}")
    print("\nRemoved per stage:")
    for stage, n in manifest["stage_stats"].items():
        print(f"  {stage:32s} {n}")
    print("\nDiversity:")
    div = manifest["diversity"]
    print(f"  languages:        {div['num_languages']}")
    print(f"  language balance: {div['language_balance']}")
    print(f"  distribution:     {div['language_distribution']}")
    if "contamination_hits" in manifest:
        print("\nContamination hits:")
        for hit in manifest["contamination_hits"]:
            print(f"  {hit['train_doc']} <- {hit['benchmark']}")
    print("\nOutputs written to ./output/ "
          "(dataset.jsonl, dataset.parquet, manifest.json, DATASET_CARD.md)")


if __name__ == "__main__":
    main()
