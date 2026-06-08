"""Command-line interface for the Indic Multilingual Dataset Builder.

Subcommands:
  run       — execute the full pipeline from a config file
  dedup     — run exact + near + semantic dedup on a JSONL file
  validate  — produce a quality/diversity report for a JSONL file
  stats     — print corpus diversity for a JSONL/Parquet dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import yaml

from .clean import LanguageDetector, normalize_document
from .collect.file_loader import FileCollector
from .enrich import corpus_diversity
from .pipeline import run_pipeline
from .schema import Document
from .validate import (
    NearDuplicateFinder,
    QualityScorer,
    SemanticDeduplicator,
    exact_dedup,
)


def _load_jsonl(path: str) -> List[Document]:
    return list(FileCollector({"path": path}).collect())


def cmd_run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = run_pipeline(cfg)
    print("\n=== Curation complete ===")
    print(f"records:          {manifest['num_records']}")
    print(f"content version:  {manifest['content_version']}")
    print(f"languages:        {manifest['diversity'].get('num_languages')}")
    print(f"output dir:       {cfg.get('export', {}).get('output_dir', 'output')}")
    return 0


def cmd_dedup(args: argparse.Namespace) -> int:
    docs = _load_jsonl(args.input)
    docs = [normalize_document(d) for d in docs]
    n0 = len(docs)
    docs, n_exact = exact_dedup(docs)
    docs, near_edges = NearDuplicateFinder(threshold=args.threshold).find_and_drop(docs)
    docs, sem_edges = SemanticDeduplicator(threshold=args.semantic_threshold).find_and_drop(docs)
    print(f"input:               {n0}")
    print(f"exact removed:       {n_exact}")
    print(f"near removed:        {len(near_edges)}")
    print(f"semantic removed:    {len(sem_edges)}")
    print(f"kept:                {len(docs)}")
    if args.output:
        from .export import export_jsonl
        export_jsonl(docs, args.output)
        print(f"wrote:               {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    docs = _load_jsonl(args.input)
    detector = LanguageDetector()
    scorer = QualityScorer()
    rows = []
    for d in docs:
        normalize_document(d)
        detector.tag(d)
        rows.append({"id": d.id, "language": d.language,
                     "lang_conf": d.language_confidence,
                     "quality": scorer.score(d)})
    avg_q = sum(r["quality"] for r in rows) / len(rows) if rows else 0.0
    low = [r for r in rows if r["quality"] < 0.4]
    print(f"documents:        {len(rows)}")
    print(f"avg quality:      {avg_q:.3f}")
    print(f"low-quality(<0.4):{len(low)}")
    print(json.dumps(corpus_diversity(docs), ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        docs = list(FileCollector({"path": str(path)}).collect()) if False else None
        print(json.dumps({
            "num_documents": len(df),
            "languages": df["language"].value_counts().to_dict()
            if "language" in df else {},
        }, ensure_ascii=False, indent=2))
        return 0
    docs = _load_jsonl(args.input)
    LanguageDetector_ = LanguageDetector()
    for d in docs:
        LanguageDetector_.tag(d)
    print(json.dumps(corpus_diversity(docs), ensure_ascii=False, indent=2))
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    from .annotate import AnnotationQA, fleiss_kappa
    from .annotate.schema import Annotation, AnnotationItem

    items: List[AnnotationItem] = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj["annotations"] = [Annotation(**a) for a in obj.get("annotations", [])]
        items.append(AnnotationItem(**obj))
    report = AnnotationQA().report(items)
    # Fleiss' kappa when every item has the same number of raters.
    label_lists = [it.labels() for it in items]
    if label_lists and len({len(x) for x in label_lists}) == 1:
        report["fleiss_kappa"] = fleiss_kappa(label_lists)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_speech(args: argparse.Namespace) -> int:
    from .speech import SpeechCurator, load_common_voice

    samples = list(load_common_voice(args.input))
    curator = SpeechCurator()
    kept, stats = curator.curate(samples)
    print(json.dumps({"curation": stats, "stats": curator.stats(kept)},
                     ensure_ascii=False, indent=2))
    if args.output:
        curator.write_manifest(kept, args.output)
        print(f"wrote manifest: {args.output}")
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    from .synthetic import SelfInstructBuilder, validate_instructions
    from .synthetic.schema import InstructionExample

    seeds = [InstructionExample(**json.loads(ln))
             for ln in Path(args.input).read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    generated = SelfInstructBuilder().build(seeds, per_seed=args.per_seed)
    report = validate_instructions(generated)
    print(json.dumps({"seeds": len(seeds), "generated": len(generated),
                      "clean": report["clean"], "issues": len(report["issues"])},
                     ensure_ascii=False, indent=2))
    if args.output:
        from .export import export_jsonl  # reuse? different schema; write directly
        with Path(args.output).open("w", encoding="utf-8") as fh:
            for ex in report["clean_examples"]:
                fh.write(json.dumps(ex.model_dump(), ensure_ascii=False) + "\n")
        print(f"wrote: {args.output}")
    return 0


def cmd_scale(args: argparse.Namespace) -> int:
    from .scale import distributed_dedup_filter

    stats = distributed_dedup_filter(args.input, args.output, text_col=args.text_col)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="indic-dataset",
        description="Collect, clean, align, validate & export multilingual LLM datasets.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run the full pipeline from a config file")
    pr.add_argument("--config", required=True)
    pr.set_defaults(func=cmd_run)

    pd_ = sub.add_parser("dedup", help="dedup a JSONL file (exact+near+semantic)")
    pd_.add_argument("--input", required=True)
    pd_.add_argument("--output", default=None)
    pd_.add_argument("--threshold", type=float, default=0.8)
    pd_.add_argument("--semantic-threshold", type=float, default=0.92)
    pd_.set_defaults(func=cmd_dedup)

    pv = sub.add_parser("validate", help="quality + diversity report for a JSONL file")
    pv.add_argument("--input", required=True)
    pv.set_defaults(func=cmd_validate)

    ps = sub.add_parser("stats", help="corpus diversity for a JSONL/Parquet dataset")
    ps.add_argument("--input", required=True)
    ps.set_defaults(func=cmd_stats)

    pa = sub.add_parser("annotate", help="inter-annotator agreement + QA report")
    pa.add_argument("--input", required=True, help="annotations JSONL")
    pa.set_defaults(func=cmd_annotate)

    psp = sub.add_parser("speech", help="curate a Common Voice TSV speech manifest")
    psp.add_argument("--input", required=True)
    psp.add_argument("--output", default=None, help="output manifest JSONL")
    psp.set_defaults(func=cmd_speech)

    psy = sub.add_parser("synthetic", help="generate + validate self-instruct data")
    psy.add_argument("--input", required=True, help="seed instructions JSONL")
    psy.add_argument("--output", default=None)
    psy.add_argument("--per-seed", type=int, default=4)
    psy.set_defaults(func=cmd_synthetic)

    psc = sub.add_parser("scale", help="distributed dedup+filter over Parquet (Dask)")
    psc.add_argument("--input", required=True, help="input parquet path/glob")
    psc.add_argument("--output", required=True)
    psc.add_argument("--text-col", default="text")
    psc.set_defaults(func=cmd_scale)
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
