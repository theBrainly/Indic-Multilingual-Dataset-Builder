"""End-to-end curation pipeline orchestrator.

Wires the stages together: collect -> clean -> align -> validate -> enrich ->
export + govern. Tracks how many documents each stage removed so the run
manifest and dataset card tell the full curation story.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .align import ParallelAligner
from .clean import HeuristicFilter, LanguageDetector, normalize_document
from .clean.heuristic_filters import FilterConfig
from .collect import build_collector
from .enrich import corpus_diversity, enrich_metadata
from .export import export_jsonl, export_parquet
from .governance.dataset_card import render_dataset_card
from .governance.provenance import build_manifest, write_manifest
from .schema import Document
from .validate import (
    ContaminationDetector,
    NearDuplicateFinder,
    QualityScorer,
    SemanticDeduplicator,
    exact_dedup,
)


class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.stage_stats: Dict[str, int] = {}
        self.aligned_pairs = []
        self.contamination_hits = []

    def _log(self, msg: str) -> None:
        print(f"[pipeline] {msg}")

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        docs = self.collect()
        docs = self.clean(docs)
        self.align(docs)
        docs = self.validate(docs)
        docs = self.enrich(docs)
        manifest = self.export(docs)
        return manifest

    # ---- COLLECT -----------------------------------------------------
    def collect(self) -> List[Document]:
        docs: List[Document] = []
        for src in self.cfg.get("collect", {}).get("sources", []):
            collector = build_collector(src)
            n_before = len(docs)
            docs.extend(collector.collect())
            self._log(f"collected {len(docs) - n_before} docs from "
                      f"{src['type']}:{src.get('path', src.get('endpoint', '...'))}")
        self.stage_stats["collected"] = len(docs)
        return docs

    # ---- CLEAN -------------------------------------------------------
    def clean(self, docs: List[Document]) -> List[Document]:
        cfg = self.cfg.get("clean", {})
        if cfg.get("normalize", True):
            docs = [normalize_document(d) for d in docs]
        if cfg.get("detect_language", True):
            detector = LanguageDetector()
            docs = [detector.tag(d) for d in docs]
        before = len(docs)
        hf = HeuristicFilter(FilterConfig.from_dict(cfg.get("filters", {})))
        docs = hf.apply(docs)
        self.stage_stats["filtered_out"] = before - len(docs)
        self._log(f"heuristic filter kept {len(docs)}/{before}")
        return docs

    # ---- ALIGN -------------------------------------------------------
    def align(self, docs: List[Document]) -> None:
        cfg = self.cfg.get("align", {})
        if not cfg.get("enabled", False):
            return
        aligner = ParallelAligner(length_ratio_max=cfg.get("length_ratio_max", 2.5))
        self.aligned_pairs = aligner.align(docs)
        self.stage_stats["aligned_pairs"] = len(self.aligned_pairs)
        self._log(f"built {len(self.aligned_pairs)} parallel pairs")

    # ---- VALIDATE ----------------------------------------------------
    def validate(self, docs: List[Document]) -> List[Document]:
        cfg = self.cfg.get("validate", {})

        if cfg.get("exact_dedup", True):
            docs, n = exact_dedup(docs)
            self.stage_stats["exact_duplicates_removed"] = n
            self._log(f"exact dedup removed {n}")

        nd = cfg.get("near_dedup", {})
        if nd.get("enabled", True):
            finder = NearDuplicateFinder(
                threshold=nd.get("threshold", 0.8),
                num_perm=nd.get("num_perm", 128),
                shingle_size=nd.get("shingle_size", 4),
            )
            before = len(docs)
            docs, edges = finder.find_and_drop(docs)
            self.stage_stats["near_duplicates_removed"] = before - len(docs)
            self._log(f"near dedup removed {before - len(docs)}")

        sd = cfg.get("semantic_dedup", {})
        if sd.get("enabled", True):
            dedup = SemanticDeduplicator(threshold=sd.get("threshold", 0.92))
            before = len(docs)
            docs, edges = dedup.find_and_drop(docs)
            self.stage_stats["semantic_duplicates_removed"] = before - len(docs)
            self._log(f"semantic dedup removed {before - len(docs)}")

        cont = cfg.get("contamination", {})
        if cont.get("enabled", False):
            bench_path = cont.get("benchmarks_file")
            if bench_path and Path(bench_path).exists():
                bench_docs = self._load_benchmarks(bench_path)
                detector = ContaminationDetector(bench_docs, ngram=cont.get("ngram", 13))
                before = len(docs)
                docs, hits = detector.check_and_drop(docs)
                self.contamination_hits = hits
                self.stage_stats["contaminated_removed"] = before - len(docs)
                self._log(f"contamination removed {before - len(docs)}")

        QualityScorer().annotate(docs)
        return docs

    def _load_benchmarks(self, path: str) -> List[Document]:
        from .collect.file_loader import FileCollector

        return list(FileCollector({"path": path}).collect())

    # ---- ENRICH ------------------------------------------------------
    def enrich(self, docs: List[Document]) -> List[Document]:
        cfg = self.cfg.get("enrich", {})
        if cfg.get("metadata", True):
            docs = enrich_metadata(docs)
        self.diversity = corpus_diversity(docs) if cfg.get("diversity", True) else {}
        return docs

    # ---- EXPORT + GOVERN --------------------------------------------
    def export(self, docs: List[Document]) -> Dict[str, Any]:
        cfg = self.cfg.get("export", {})
        out_dir = Path(cfg.get("output_dir", "output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        formats = cfg.get("formats", ["jsonl"])

        if "jsonl" in formats:
            p = export_jsonl(docs, out_dir / "dataset.jsonl")
            self._log(f"wrote {p}")
        if "parquet" in formats:
            p = export_parquet(docs, out_dir / "dataset.parquet")
            self._log(f"wrote {p}")
        if self.aligned_pairs:
            with (out_dir / "aligned_pairs.jsonl").open("w", encoding="utf-8") as fh:
                for pair in self.aligned_pairs:
                    fh.write(json.dumps(pair.to_export_dict(), ensure_ascii=False) + "\n")

        manifest = build_manifest(
            docs, self.cfg, self.stage_stats, getattr(self, "diversity", {})
        )
        if self.contamination_hits:
            manifest["contamination_hits"] = [
                {"train_doc": t, "benchmark": b, "ngram": g}
                for t, b, g in self.contamination_hits
            ]
        if cfg.get("write_manifest", True):
            write_manifest(manifest, str(out_dir / "manifest.json"))
            self._log(f"wrote {out_dir / 'manifest.json'}")
        if cfg.get("write_dataset_card", True):
            (out_dir / "DATASET_CARD.md").write_text(
                render_dataset_card(manifest), encoding="utf-8"
            )
            self._log(f"wrote {out_dir / 'DATASET_CARD.md'}")
        return manifest


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    return Pipeline(config).run()
