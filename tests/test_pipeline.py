"""Tests for the core curation stages."""
from __future__ import annotations

from pathlib import Path

import yaml

from indic_dataset_builder.clean import LanguageDetector, normalize_text
from indic_dataset_builder.clean.heuristic_filters import FilterConfig, HeuristicFilter
from indic_dataset_builder.pipeline import run_pipeline
from indic_dataset_builder.schema import Document, ProvenanceRecord
from indic_dataset_builder.validate import NearDuplicateFinder, exact_dedup
from indic_dataset_builder.validate.contamination import ContaminationDetector

ROOT = Path(__file__).resolve().parents[1]


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text,
                    source=ProvenanceRecord(source_type="test", source_id="t"))


def test_normalize_collapses_whitespace_and_nfc():
    assert normalize_text("hello   world\r\n\r\n\r\nfoo") == "hello world\n\nfoo"
    # NFC: composed and decomposed forms become identical
    assert normalize_text("a\u0301") == normalize_text("\u00e1")


def test_language_detection_indic_scripts():
    det = LanguageDetector()
    assert det.detect("भारत एक देश है")[0] == "hi"
    assert det.detect("বাংলা ভাষা")[0] == "bn"
    assert det.detect("தமிழ் மொழி")[0] == "ta"
    assert det.detect("hello world")[0] == "en"


def test_exact_dedup_keeps_first():
    docs = [_doc("a", "same text here friend"), _doc("b", "same text here friend"),
            _doc("c", "different text entirely now")]
    kept, dropped = exact_dedup(docs)
    assert dropped == 1
    assert {d.id for d in kept} == {"a", "c"}


def test_near_dedup_catches_minor_edits():
    docs = [
        _doc("a", "the quick brown fox jumps over the lazy sleeping dog today"),
        _doc("b", "the quick brown fox jumps over the lazy sleeping dog today!"),
        _doc("c", "completely unrelated sentence about multilingual datasets here"),
    ]
    kept, edges = NearDuplicateFinder(threshold=0.7).find_and_drop(docs)
    kept_ids = {d.id for d in kept}
    assert "a" in kept_ids and "c" in kept_ids
    assert "b" not in kept_ids


def test_heuristic_filter_drops_noise():
    cfg = FilterConfig(require_known_language=False)
    hf = HeuristicFilter(cfg)
    noise = _doc("n", "!!! ### $$$ %%% &&& *** @@@")
    good = _doc("g", "This is a perfectly reasonable English sentence with content.")
    kept = hf.apply([noise, good])
    assert {d.id for d in kept} == {"g"}
    assert noise.dropped and noise.drop_reason


def test_contamination_detection():
    bench = [_doc("b1", "the secret benchmark phrase that must not leak into training data ever")]
    det = ContaminationDetector(bench, ngram=5)
    train = [
        _doc("t1", "the secret benchmark phrase that must not leak into training data ever again"),
        _doc("t2", "a totally clean and original document about Indic language curation work"),
    ]
    kept, hits = det.check_and_drop(train)
    assert {d.id for d in kept} == {"t2"}
    assert hits and hits[0][0] == "t1"


def test_end_to_end_pipeline(tmp_path):
    config = yaml.safe_load((ROOT / "config" / "pipeline.yaml").read_text())
    config["export"]["output_dir"] = str(tmp_path / "out")
    manifest = run_pipeline(config)

    # messy input is curated down to clean, deduped, uncontaminated docs
    assert manifest["num_records"] > 0
    assert manifest["num_records"] < manifest["stage_stats"]["collected"]
    assert manifest["stage_stats"]["exact_duplicates_removed"] >= 1
    assert manifest["stage_stats"].get("contaminated_removed", 0) >= 1
    assert manifest["diversity"]["num_languages"] >= 4
    assert (tmp_path / "out" / "dataset.jsonl").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert (tmp_path / "out" / "DATASET_CARD.md").exists()
