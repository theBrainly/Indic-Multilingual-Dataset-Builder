"""Indic Multilingual Dataset Builder — Streamlit dashboard.

A visual front-end over the existing curation pipeline. It does NOT reimplement
any logic: every panel calls the same modules used by the CLI/pipeline, so the
dashboard and the library can never drift apart.

Launch:
    pip install -e ".[ui]"
    streamlit run ui/app.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Make the package importable when run via `streamlit run ui/app.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indic_dataset_builder.align import ParallelAligner  # noqa: E402
from indic_dataset_builder.clean import (  # noqa: E402
    HeuristicFilter,
    LanguageDetector,
    normalize_document,
)
from indic_dataset_builder.clean.heuristic_filters import FilterConfig  # noqa: E402
from indic_dataset_builder.collect.file_loader import FileCollector  # noqa: E402
from indic_dataset_builder.enrich import corpus_diversity, enrich_metadata  # noqa: E402
from indic_dataset_builder.pipeline import run_pipeline  # noqa: E402
from indic_dataset_builder.validate import (  # noqa: E402
    ContaminationDetector,
    NearDuplicateFinder,
    QualityScorer,
    SemanticDeduplicator,
    exact_dedup,
)

st.set_page_config(
    page_title="Indic Dataset Builder",
    page_icon="🪔",
    layout="wide",
)

SAMPLE_DIR = ROOT / "data" / "sample"


# --------------------------------------------------------------------------
# Theme / styling
# --------------------------------------------------------------------------
_CSS = """
<style>
/* ---- base ---- */
.stApp {
    background:
        radial-gradient(1200px 600px at 100% -10%, #eef0ff 0%, rgba(238,240,255,0) 55%),
        radial-gradient(1000px 500px at -10% 0%, #f3ecff 0%, rgba(243,236,255,0) 50%),
        #fbfbfd;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

/* ---- typography ---- */
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; color: #14162b; }
h2 { margin-top: 0.4rem; }
.idb-sub { color: #6b7090; font-size: 0.96rem; margin-top: -0.3rem; }

/* ---- hero ---- */
.idb-hero {
    border-radius: 22px; padding: 30px 34px; margin-bottom: 22px; color: #fff;
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    box-shadow: 0 18px 40px -18px rgba(99,102,241,0.65);
    position: relative; overflow: hidden;
}
.idb-hero:after {
    content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;
    background: radial-gradient(circle, rgba(255,255,255,0.25), transparent 70%);
}
.idb-hero h1 { color:#fff; margin:0 0 6px 0; font-size: 2.0rem; }
.idb-hero p { color: rgba(255,255,255,0.92); margin:0; font-size: 1.02rem; }
.idb-pills { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.idb-pill {
    background: rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.28);
    padding:5px 12px; border-radius: 999px; font-size:0.8rem; color:#fff;
    backdrop-filter: blur(4px);
}

/* ---- metric cards ---- */
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #ececf5; border-radius: 16px;
    padding: 16px 18px; box-shadow: 0 6px 20px -14px rgba(30,32,70,0.35);
    transition: transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px -16px rgba(99,102,241,0.55);
}
div[data-testid="stMetricLabel"] p {
    color:#8a8fb0; font-size:0.78rem; font-weight:600;
    text-transform: uppercase; letter-spacing: .05em;
}
div[data-testid="stMetricValue"] { color:#14162b; font-weight:700; }

/* ---- buttons ---- */
.stButton > button {
    border-radius: 12px; font-weight: 600; border: 1px solid #e2e2ee;
    padding: 0.5rem 1.1rem; transition: all .15s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg,#4f46e5,#7c3aed); border:none; color:#fff;
    box-shadow: 0 10px 24px -12px rgba(99,102,241,0.8);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.07); transform: translateY(-1px); }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: #14162b; border-right: 1px solid #20233f;
}
section[data-testid="stSidebar"] * { color: #d7d9ee; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] .idb-brand { color:#fff; }
.idb-brand { font-size:1.15rem; font-weight:700; letter-spacing:-0.01em; }
.idb-brand-sub { color:#8c90c0 !important; font-size:0.78rem; margin-top:-4px; }
section[data-testid="stSidebar"] label { font-size:0.92rem; }

/* ---- expander / tables / inputs ---- */
div[data-testid="stExpander"] {
    border:1px solid #ececf5; border-radius:14px; background:#fff;
}
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #ececf5; }
[data-testid="stJson"] { background:#fff; border:1px solid #ececf5; border-radius:14px; padding:8px 12px; }

/* ---- section card wrapper ---- */
.idb-card {
    background:#fff; border:1px solid #ececf5; border-radius:18px;
    padding:18px 22px; box-shadow:0 8px 24px -18px rgba(30,32,70,0.4); margin-bottom:8px;
}
hr { border-color:#ececf5; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pill_html = ""
    if pills:
        pill_html = '<div class="idb-pills">' + "".join(
            f'<span class="idb-pill">{p}</span>' for p in pills) + "</div>"
    st.markdown(
        f'<div class="idb-hero"><h1>{title}</h1><p>{subtitle}</p>{pill_html}</div>',
        unsafe_allow_html=True,
    )


def _section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="idb-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"### {title}{sub}", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _save_upload_to_temp(uploaded, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    return tmp.name


def _load_docs(path: str):
    return list(FileCollector({"path": path}).collect())


def _base_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "pipeline.yaml").read_text())


def _lang_chart(distribution: dict):
    if not distribution:
        st.info("No language data.")
        return
    df = pd.DataFrame(
        {"language": list(distribution.keys()),
         "documents": list(distribution.values())}
    ).set_index("language")
    st.bar_chart(df, color="#6366f1")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home():
    _hero(
        "🪔 Indic Multilingual Dataset Builder",
        "Collect → Clean → Align → Validate → Enrich → Govern → Export "
        "high-quality multilingual datasets for LLM training.",
        pills=["15 languages", "3 dedup methods", "Contamination control",
               "Provenance & versioning", "Annotation QA"],
    )
    cols = st.columns(4)
    cols[0].metric("Pipeline stages", "7")
    cols[1].metric("Supported languages", "15")
    cols[2].metric("Dedup methods", "3")
    cols[3].metric("Curation modules", "13")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="idb-card"><h3>What this does</h3>'
            '<p class="idb-sub">A thin visual layer over the curation library — '
            'every panel calls the same modules used by the CLI, so the UI and '
            'engine never drift.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="idb-card"><h3>Where to start</h3>'
            '<p class="idb-sub">Open <b>Run Pipeline</b> to curate a dataset '
            'end-to-end, then explore <b>Dedup</b>, <b>Quality</b> and '
            '<b>Annotation QA</b>.</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    _section("Modules", "Each maps to a stage of the curation pipeline")
    grid = st.columns(3)
    cards = [
        ("🚀 Run Pipeline", "Curate end-to-end and download results."),
        ("🔁 Dedup Explorer", "See which docs were removed and why."),
        ("📊 Quality & Diversity", "Language mix, quality, diversity."),
        ("🧑‍🤝‍🧑 Annotation QA", "Kappa/alpha + annotator scorecards."),
        ("🎙️ Speech", "Curate an ASR/TTS Common Voice manifest."),
        ("🤖 Synthetic", "Generate & validate self-instruct data."),
    ]
    for i, (title, desc) in enumerate(cards):
        with grid[i % 3]:
            st.markdown(
                f'<div class="idb-card"><h3 style="font-size:1.05rem;margin:0 0 4px 0">'
                f'{title}</h3><p class="idb-sub">{desc}</p></div>',
                unsafe_allow_html=True,
            )



def page_pipeline():
    _hero("🚀 Run the curation pipeline",
          "Ingest, clean, align, deduplicate, decontaminate and export — in one run.",
          pills=["exact + near + semantic dedup", "contamination check", "exports"])
    src = st.radio("Input", ["Use bundled sample", "Upload a JSONL file"],
                   horizontal=True)
    if src == "Upload a JSONL file":
        up = st.file_uploader("JSONL with a `text` field per line", type=["jsonl"])
        input_path = _save_upload_to_temp(up, ".jsonl") if up else None
    else:
        input_path = str(SAMPLE_DIR / "raw.jsonl")

    with st.expander("Curation settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        near_t = c1.slider("Near-dup Jaccard threshold", 0.5, 1.0, 0.8, 0.01)
        sem_t = c2.slider("Semantic cosine threshold", 0.5, 1.0, 0.92, 0.01)
        min_words = c3.number_input("Min words per doc", 1, 100, 4)
        contam = st.checkbox("Run benchmark contamination check", value=True)

    if st.button("🚀 Run pipeline", type="primary", disabled=input_path is None):
        cfg = _base_config()
        out_dir = tempfile.mkdtemp(prefix="idb_out_")
        cfg["collect"]["sources"] = [{"type": "file", "path": input_path}]
        cfg["clean"]["filters"]["min_words"] = int(min_words)
        cfg["validate"]["near_dedup"]["threshold"] = float(near_t)
        cfg["validate"]["semantic_dedup"]["threshold"] = float(sem_t)
        cfg["validate"]["contamination"]["enabled"] = bool(contam)
        cfg["validate"]["contamination"]["benchmarks_file"] = str(
            SAMPLE_DIR / "benchmark.jsonl")
        cfg["export"]["output_dir"] = out_dir

        with st.spinner("Curating..."):
            manifest = run_pipeline(cfg)
        st.session_state["manifest"] = manifest
        st.session_state["out_dir"] = out_dir

    manifest = st.session_state.get("manifest")
    if not manifest:
        return

    st.success(f"Done — content version `{manifest['content_version']}`")
    stats = manifest["stage_stats"]
    div = manifest["diversity"]

    m = st.columns(5)
    m[0].metric("Collected", stats.get("collected", 0))
    m[1].metric("Kept", manifest["num_records"])
    m[2].metric("Filtered out", stats.get("filtered_out", 0))
    m[3].metric("Duplicates removed",
                stats.get("exact_duplicates_removed", 0)
                + stats.get("near_duplicates_removed", 0)
                + stats.get("semantic_duplicates_removed", 0))
    m[4].metric("Contaminated", stats.get("contaminated_removed", 0))

    st.subheader("Curation funnel")
    funnel = pd.DataFrame({
        "stage": ["collected", "after filter", "kept (final)"],
        "documents": [
            stats.get("collected", 0),
            stats.get("collected", 0) - stats.get("filtered_out", 0),
            manifest["num_records"],
        ],
    }).set_index("stage")
    st.bar_chart(funnel, color="#7c3aed")

    left, right = st.columns(2)
    with left:
        st.subheader("Language distribution")
        _lang_chart(div.get("language_distribution", {}))
    with right:
        st.subheader("Corpus stats")
        st.json({
            "languages": div.get("num_languages"),
            "language_balance": div.get("language_balance"),
            "total_tokens": div.get("total_tokens"),
            "vocabulary_size": div.get("vocabulary_size"),
            "type_token_ratio": div.get("type_token_ratio"),
        })

    if manifest.get("contamination_hits"):
        st.subheader("⚠️ Contamination hits (removed)")
        st.dataframe(pd.DataFrame(manifest["contamination_hits"]),
                     width="stretch")

    # downloads
    out_dir = st.session_state.get("out_dir")
    if out_dir:
        st.subheader("Download curated outputs")
        d1, d2, d3 = st.columns(3)
        ds = Path(out_dir) / "dataset.jsonl"
        card = Path(out_dir) / "DATASET_CARD.md"
        man = Path(out_dir) / "manifest.json"
        if ds.exists():
            d1.download_button("dataset.jsonl", ds.read_bytes(),
                               file_name="dataset.jsonl")
        if card.exists():
            d2.download_button("DATASET_CARD.md", card.read_bytes(),
                               file_name="DATASET_CARD.md")
        if man.exists():
            d3.download_button("manifest.json", man.read_bytes(),
                               file_name="manifest.json")
        with st.expander("Preview dataset card"):
            if card.exists():
                st.markdown(card.read_text())


def page_dedup():
    _hero("🔁 Dedup explorer",
          "See which documents each dedup pass removes — and exactly why.",
          pills=["exact (hash)", "near (MinHash/LSH)", "semantic (embeddings)"])
    src = st.radio("Input", ["Bundled sample", "Upload JSONL"], horizontal=True,
                   key="dedup_src")
    if src == "Upload JSONL":
        up = st.file_uploader("JSONL", type=["jsonl"], key="dedup_up")
        path = _save_upload_to_temp(up, ".jsonl") if up else None
    else:
        path = str(SAMPLE_DIR / "raw.jsonl")

    c1, c2 = st.columns(2)
    near_t = c1.slider("Near-dup threshold", 0.5, 1.0, 0.8, 0.01, key="d_near")
    sem_t = c2.slider("Semantic threshold", 0.5, 1.0, 0.92, 0.01, key="d_sem")

    if st.button("Run dedup", type="primary", disabled=path is None):
        docs = [normalize_document(d) for d in _load_docs(path)]
        LanguageDetector_ = LanguageDetector()
        for d in docs:
            LanguageDetector_.tag(d)
        n0 = len(docs)
        removed_rows = []

        kept, n_exact = exact_dedup(docs)
        for d in docs:
            if d.dropped and d.drop_reason and d.drop_reason.startswith("exact"):
                removed_rows.append({"id": d.id, "stage": "exact",
                                     "reason": d.drop_reason,
                                     "text": d.text[:80]})

        kept, near_edges = NearDuplicateFinder(threshold=near_t).find_and_drop(kept)
        for dropped_id, kept_id, sim in near_edges:
            removed_rows.append({"id": dropped_id, "stage": "near",
                                 "reason": f"~{kept_id} (J={sim})", "text": ""})

        kept, sem_edges = SemanticDeduplicator(threshold=sem_t).find_and_drop(kept)
        for dropped_id, kept_id, sim in sem_edges:
            removed_rows.append({"id": dropped_id, "stage": "semantic",
                                 "reason": f"~{kept_id} (cos={sim})", "text": ""})

        m = st.columns(4)
        m[0].metric("Input", n0)
        m[1].metric("Exact removed", n_exact)
        m[2].metric("Near removed", len(near_edges))
        m[3].metric("Semantic removed", len(sem_edges))
        st.metric("Kept", len(kept))

        if removed_rows:
            st.subheader("Removed documents")
            st.dataframe(pd.DataFrame(removed_rows), width="stretch")
        else:
            st.info("No duplicates found at these thresholds.")


def page_quality():
    _hero("📊 Quality & diversity",
          "Language mix, per-document quality scores, and corpus diversity.",
          pills=["language ID", "quality scoring", "diversity entropy"])
    src = st.radio("Input", ["Bundled sample", "Upload JSONL"], horizontal=True,
                   key="q_src")
    if src == "Upload JSONL":
        up = st.file_uploader("JSONL", type=["jsonl"], key="q_up")
        path = _save_upload_to_temp(up, ".jsonl") if up else None
    else:
        path = str(SAMPLE_DIR / "raw.jsonl")

    if st.button("Analyze", type="primary", disabled=path is None):
        docs = [normalize_document(d) for d in _load_docs(path)]
        det = LanguageDetector()
        scorer = QualityScorer()
        for d in docs:
            det.tag(d)
        enrich_metadata(docs)
        rows = [{
            "id": d.id, "language": d.language,
            "lang_conf": d.language_confidence,
            "quality": scorer.score(d),
            "words": d.metadata.get("word_count"),
            "text": d.text[:60],
        } for d in docs]
        div = corpus_diversity(docs)

        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", div["num_documents"])
        c2.metric("Languages", div["num_languages"])
        c3.metric("Language balance", div["language_balance"])

        left, right = st.columns(2)
        with left:
            st.subheader("Language distribution")
            _lang_chart(div["language_distribution"])
        with right:
            st.subheader("Quality score distribution")
            qdf = pd.DataFrame(rows)
            st.bar_chart(qdf.set_index("id")["quality"], color="#6366f1")

        st.subheader("Per-document detail")
        st.dataframe(pd.DataFrame(rows), width="stretch")


def page_annotation():
    _hero("🧑‍🤝‍🧑 Annotation QA",
          "Inter-annotator agreement and annotator scorecards for trustworthy labels.",
          pills=["Cohen's κ", "Fleiss' κ", "consensus & adjudication"])
    from indic_dataset_builder.annotate import AnnotationQA, fleiss_kappa
    from indic_dataset_builder.annotate.schema import Annotation, AnnotationItem

    src = st.radio("Input", ["Bundled sample", "Upload JSONL"], horizontal=True,
                   key="a_src")
    if src == "Upload JSONL":
        up = st.file_uploader("Annotations JSONL", type=["jsonl"], key="a_up")
        path = _save_upload_to_temp(up, ".jsonl") if up else None
    else:
        path = str(SAMPLE_DIR / "annotations.jsonl")

    if st.button("Compute agreement", type="primary", disabled=path is None):
        items = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["annotations"] = [Annotation(**a) for a in obj.get("annotations", [])]
            items.append(AnnotationItem(**obj))

        report = AnnotationQA().report(items)
        label_lists = [it.labels() for it in items]
        fk = (fleiss_kappa(label_lists)
              if label_lists and len({len(x) for x in label_lists}) == 1 else None)

        c1, c2, c3 = st.columns(3)
        c1.metric("Items", report["num_items"])
        c2.metric("Mean agreement", report["mean_agreement"])
        c3.metric("Fleiss' κ", fk if fk is not None else "n/a")

        st.subheader("Annotator scorecards")
        acc = report["annotator_accuracy"]
        kap = report["annotator_vs_consensus_kappa"]
        sc = pd.DataFrame({
            "annotator": list(acc.keys()),
            "gold_accuracy": [acc[a] for a in acc],
            "consensus_kappa": [kap.get(a, float("nan")) for a in acc],
        }).set_index("annotator")
        st.dataframe(sc, width="stretch")
        st.bar_chart(sc)

        flagged = [a for a in acc if acc[a] < 0.8 or kap.get(a, 1) < 0.4]
        if flagged:
            st.warning(f"Annotators to review: {', '.join(flagged)}")
        st.caption(f"{report['needs_adjudication']} item(s) need adjudication.")


def page_speech():
    _hero("🎙️ Speech dataset curation",
          "Validate, filter and dedup an ASR/TTS manifest (Common Voice format).",
          pills=["transcript normalize", "duration filter", "hours & speakers"])
    from indic_dataset_builder.speech import SpeechCurator, load_common_voice

    src = st.radio("Input", ["Bundled sample", "Upload TSV"], horizontal=True,
                   key="s_src")
    if src == "Upload TSV":
        up = st.file_uploader("Common Voice TSV", type=["tsv"], key="s_up")
        path = _save_upload_to_temp(up, ".tsv") if up else None
    else:
        path = str(SAMPLE_DIR / "common_voice.tsv")

    if st.button("Curate manifest", type="primary", disabled=path is None):
        samples = list(load_common_voice(path))
        curator = SpeechCurator()
        kept, cur = curator.curate(samples)
        stats = curator.stats(kept)

        c1, c2, c3 = st.columns(3)
        c1.metric("Input utterances", cur["input"])
        c2.metric("Kept", cur["kept"])
        c3.metric("Speakers", stats["num_speakers"])

        st.subheader("Hours per language")
        _lang_chart({k: round(v, 4)
                     for k, v in stats["hours_per_language"].items()})
        st.caption("Hours are 0 here because sample clips aren't bundled; "
                   "install `soundfile` and point to real audio for true durations.")

        if cur["dropped_reasons"]:
            st.subheader("Dropped reasons")
            st.json(cur["dropped_reasons"])
        st.subheader("Kept transcripts")
        st.dataframe(pd.DataFrame([{
            "id": s.id, "language": s.language, "speaker": s.speaker_id,
            "transcript": s.transcript,
        } for s in kept]), width="stretch")


def page_synthetic():
    _hero("🤖 Synthetic data",
          "Generate self-instruct examples from seeds and validate them.",
          pills=["self-instruct", "dedup vs seeds", "schema validation"])
    from indic_dataset_builder.synthetic import (
        SelfInstructBuilder,
        validate_instructions,
    )
    from indic_dataset_builder.synthetic.schema import InstructionExample

    src = st.radio("Input", ["Bundled sample", "Upload JSONL"], horizontal=True,
                   key="syn_src")
    if src == "Upload JSONL":
        up = st.file_uploader("Seed instructions JSONL", type=["jsonl"], key="syn_up")
        path = _save_upload_to_temp(up, ".jsonl") if up else None
    else:
        path = str(SAMPLE_DIR / "seeds.jsonl")

    per_seed = st.slider("Examples per seed", 1, 10, 4)

    if st.button("Generate", type="primary", disabled=path is None):
        seeds = [InstructionExample(**json.loads(ln))
                 for ln in Path(path).read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        gen = SelfInstructBuilder().build(seeds, per_seed=per_seed)
        report = validate_instructions(gen)

        c1, c2, c3 = st.columns(3)
        c1.metric("Seeds", len(seeds))
        c2.metric("Generated", len(gen))
        c3.metric("Clean (valid)", report["clean"])

        st.subheader("Generated examples")
        st.dataframe(pd.DataFrame([{
            "instruction": e.instruction, "output": e.output,
            "language": e.language, "task_type": e.task_type,
        } for e in report["clean_examples"]]), width="stretch")


PAGES = {
    "🏠 Home": page_home,
    "🚀 Run Pipeline": page_pipeline,
    "🔁 Dedup Explorer": page_dedup,
    "📊 Quality & Diversity": page_quality,
    "🧑‍🤝‍🧑 Annotation QA": page_annotation,
    "🎙️ Speech": page_speech,
    "🤖 Synthetic": page_synthetic,
}


def main():
    _inject_css()
    with st.sidebar:
        st.markdown(
            '<div class="idb-brand">🪔 Dataset Builder</div>'
            '<div class="idb-brand-sub">Indic Multilingual Curation Platform</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption(
            "A visual layer over the curation library. Every panel calls the "
            "same modules used by the CLI."
        )
    PAGES[choice]()


if __name__ == "__main__":
    main()
