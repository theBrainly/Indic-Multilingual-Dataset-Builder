"""Speech dataset curation: validate, filter, dedup, manifest, report.

Reuses the text normalizer so transcripts get the same Indic-aware cleaning as
the text corpus. Produces a NeMo/ESPnet-style JSONL manifest plus a stats
report (total hours, per-language hours, speaker diversity).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..clean.normalizer import normalize_text
from ..clean.text_utils import word_count
from .schema import AudioSample


@dataclass
class SpeechFilterConfig:
    min_duration: float = 0.5
    max_duration: float = 30.0
    min_words: int = 1
    max_chars_per_sec: float = 25.0      # flag impossibly fast "speech"
    require_transcript: bool = True
    drop_empty_after_norm: bool = True


class SpeechCurator:
    def __init__(self, config: SpeechFilterConfig | None = None):
        self.cfg = config or SpeechFilterConfig()

    def _evaluate(self, s: AudioSample) -> str | None:
        if self.cfg.require_transcript and not s.transcript.strip():
            return "empty_transcript"
        if word_count(s.transcript) < self.cfg.min_words:
            return "too_few_words"
        if s.duration_sec is not None:
            if s.duration_sec < self.cfg.min_duration:
                return f"too_short({s.duration_sec}s)"
            if s.duration_sec > self.cfg.max_duration:
                return f"too_long({s.duration_sec}s)"
            cps = len(s.transcript) / max(s.duration_sec, 0.01)
            if cps > self.cfg.max_chars_per_sec:
                return f"chars_per_sec_high({cps:.1f})"
        return None

    def curate(self, samples: List[AudioSample]) -> tuple[List[AudioSample], Dict[str, Any]]:
        kept: List[AudioSample] = []
        seen_transcripts: set = set()
        dropped_reasons: Dict[str, int] = {}

        for s in samples:
            # normalize transcript (Indic-aware), in place
            s.transcript = normalize_text(s.transcript)
            reason = self._evaluate(s)
            if reason is None and s.transcript.lower() in seen_transcripts:
                reason = "duplicate_transcript"
            if reason:
                s.drop(reason)
                dropped_reasons[reason.split("(")[0]] = (
                    dropped_reasons.get(reason.split("(")[0], 0) + 1)
                continue
            seen_transcripts.add(s.transcript.lower())
            kept.append(s)

        return kept, {"dropped_reasons": dropped_reasons,
                      "kept": len(kept), "input": len(samples)}

    def stats(self, samples: List[AudioSample]) -> Dict[str, Any]:
        total_sec = sum(s.duration_sec or 0.0 for s in samples)
        by_lang_sec: Dict[str, float] = {}
        speakers: set = set()
        for s in samples:
            by_lang_sec[s.language or "und"] = (
                by_lang_sec.get(s.language or "und", 0.0) + (s.duration_sec or 0.0))
            if s.speaker_id:
                speakers.add(s.speaker_id)
        return {
            "num_utterances": len(samples),
            "total_hours": round(total_sec / 3600.0, 4),
            "hours_per_language": {k: round(v / 3600.0, 4)
                                   for k, v in by_lang_sec.items()},
            "num_speakers": len(speakers),
            "num_languages": len(by_lang_sec),
        }

    def write_manifest(self, samples: List[AudioSample], path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s.to_manifest_row(), ensure_ascii=False) + "\n")
        return p
