"""Tests for annotation, speech, synthetic, indexing and scale modules."""
from __future__ import annotations

from pathlib import Path

from indic_dataset_builder.annotate import (
    AnnotationQA,
    cohens_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    validate_preferences,
)
from indic_dataset_builder.annotate.schema import (
    Annotation,
    AnnotationItem,
    PreferencePair,
)
from indic_dataset_builder.index import build_index
from indic_dataset_builder.speech import SpeechCurator
from indic_dataset_builder.speech.schema import AudioSample
from indic_dataset_builder.synthetic import (
    SelfInstructBuilder,
    validate_instructions,
    validate_trajectory,
)
from indic_dataset_builder.synthetic.schema import (
    InstructionExample,
    ToolCall,
    ToolUseTrajectory,
)

ROOT = Path(__file__).resolve().parents[1]


# ---- annotation -----------------------------------------------------------
def test_cohens_kappa_perfect_and_chance():
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # total disagreement on a balanced set -> negative kappa
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) < 0


def test_fleiss_kappa_full_agreement():
    items = [["keep", "keep", "keep"], ["reject", "reject", "reject"]]
    assert fleiss_kappa(items) == 1.0


def test_krippendorff_alpha_runs_with_missing():
    ann = {
        "a": {"i1": "x", "i2": "y", "i3": "x"},
        "b": {"i1": "x", "i2": "y"},          # missing i3
        "c": {"i1": "x", "i3": "x"},
    }
    alpha = krippendorff_alpha(ann)
    assert -1.0 <= alpha <= 1.0
    assert alpha > 0.5  # mostly-agreeing data


def test_annotation_qa_report():
    items = [
        AnnotationItem(id="1", text="t", gold_label="keep", annotations=[
            Annotation(item_id="1", annotator_id="a", label="keep"),
            Annotation(item_id="1", annotator_id="b", label="keep"),
            Annotation(item_id="1", annotator_id="c", label="reject"),
        ]),
    ]
    report = AnnotationQA().report(items)
    assert report["num_items"] == 1
    assert report["annotator_accuracy"]["a"] == 1.0
    assert report["annotator_accuracy"]["c"] == 0.0


def test_preference_validation_flags_cycle_and_equal():
    pairs = [
        PreferencePair(id="p1", prompt="q", chosen="A", rejected="B", annotator_id="a"),
        PreferencePair(id="p2", prompt="q", chosen="B", rejected="C", annotator_id="a"),
        PreferencePair(id="p3", prompt="q", chosen="C", rejected="A", annotator_id="a"),
        PreferencePair(id="p4", prompt="q", chosen="X", rejected="X", annotator_id="a"),
    ]
    report = validate_preferences(pairs)
    assert report["cyclic_prompts"] == 1
    assert any(i["issue"] == "chosen_equals_rejected" for i in report["issues"])


# ---- speech ---------------------------------------------------------------
def test_speech_curation_filters_and_dedups():
    samples = [
        AudioSample(id="1", audio_path="a.wav", transcript="भारत मेरा देश है",
                    language="hi", duration_sec=3.0, speaker_id="s1"),
        AudioSample(id="2", audio_path="b.wav", transcript="भारत मेरा देश है",
                    language="hi", duration_sec=3.1, speaker_id="s1"),  # dup text
        AudioSample(id="3", audio_path="c.wav", transcript="",
                    language="hi", duration_sec=2.0),                   # empty
        AudioSample(id="4", audio_path="d.wav", transcript="ok fine words here",
                    language="en", duration_sec=0.1),                   # too short
    ]
    kept, stats = SpeechCurator().curate(samples)
    ids = {s.id for s in kept}
    assert ids == {"1"}
    assert stats["dropped_reasons"]
    report = SpeechCurator().stats(kept)
    assert report["num_utterances"] == 1


def test_load_common_voice_sample():
    from indic_dataset_builder.speech import load_common_voice

    samples = list(load_common_voice(str(ROOT / "data" / "sample" / "common_voice.tsv")))
    kept, _ = SpeechCurator().curate(samples)
    # 7 rows -> drop empty, drop 1-char, drop duplicate Tamil line
    assert 0 < len(kept) < len(samples)
    langs = {s.language for s in kept}
    assert "hi" in langs and "ta" in langs and "bn" in langs


# ---- synthetic ------------------------------------------------------------
def test_self_instruct_generation_and_validation():
    seeds = [InstructionExample(id="s1", instruction="Translate to Hindi.",
                                output="अनुवाद", language="hi")]
    gen = SelfInstructBuilder().build(seeds, per_seed=4)
    assert len(gen) >= 1
    report = validate_instructions(gen)
    assert report["clean"] == len(report["clean_examples"])


def test_trajectory_validation():
    traj = ToolUseTrajectory(
        id="t1", goal="find weather", final_answer="22C",
        available_tools=["search"],
        steps=[ToolCall(tool="search", arguments={"q": "weather"})],
    )
    assert validate_trajectory(traj)["valid"]
    bad = ToolUseTrajectory(id="t2", goal="", available_tools=["search"],
                            steps=[ToolCall(tool="unknown")])
    res = validate_trajectory(bad)
    assert not res["valid"] and "empty_goal" in res["problems"]


# ---- index ----------------------------------------------------------------
def test_in_memory_index_search_and_exact():
    idx = build_index(prefer_elasticsearch=False)
    idx.add("d1", "multilingual dataset curation for indic languages")
    idx.add("d2", "speech recognition and text to speech systems")
    idx.add("d3", "multilingual dataset curation for indic languages")
    results = idx.search("indic dataset", top_k=2)
    assert results and results[0][0] in {"d1", "d3"}
    assert set(idx.exact_match("multilingual dataset curation for indic languages")) == {"d1", "d3"}


# ---- scale ----------------------------------------------------------------
def test_distributed_dedup_filter(tmp_path):
    import pandas as pd
    from indic_dataset_builder.scale import distributed_dedup_filter

    df = pd.DataFrame({"text": [
        "This is a clean and reasonably long English sentence about data.",
        "This is a clean and reasonably long English sentence about data.",  # dup
        "!!! ### $$$ %%%",   # noise -> filtered
        "तमिल भाषा एक प्राचीन और समृद्ध भाषा है जो सदियों से बोली जाती है।",
    ]})
    src = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    df.to_parquet(src, index=False)
    stats = distributed_dedup_filter(str(src), str(out), text_col="text")
    assert stats["input_rows"] == 4
    assert stats["output_rows"] == 2  # one dup + one noise removed
    assert out.exists()
