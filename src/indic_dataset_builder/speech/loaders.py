"""Loaders for common speech-dataset layouts.

- Mozilla Common Voice: a TSV with columns like client_id, path, sentence,
  locale, up_votes/down_votes.
- OpenSLR: a transcript file mapping utterance-id -> text, with audio under a
  parallel directory tree.

Loaders return :class:`AudioSample`s. Durations are read from the manifest when
present, or computed via `soundfile` if it's installed and the audio exists.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

from .schema import AudioSample


def _maybe_duration(audio_path: str) -> Optional[float]:
    """Read duration via soundfile if available and file exists, else None."""
    p = Path(audio_path)
    if not p.exists():
        return None
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(p))
        return round(info.frames / float(info.samplerate), 3)
    except Exception:
        return None


def load_common_voice(tsv_path: str, language: Optional[str] = None,
                      clips_dir: Optional[str] = None) -> Iterator[AudioSample]:
    tsv = Path(tsv_path)
    base = Path(clips_dir) if clips_dir else tsv.parent / "clips"
    with tsv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            sentence = row.get("sentence", "").strip()
            if not sentence:
                continue
            rel = row.get("path", f"cv-{i}.mp3")
            audio_path = str(base / rel)
            up = int(row.get("up_votes", 0) or 0)
            down = int(row.get("down_votes", 0) or 0)
            yield AudioSample(
                id=row.get("client_id", f"cv-{i}")[:16] + f"-{i}",
                audio_path=audio_path,
                transcript=sentence,
                language=row.get("locale", language),
                duration_sec=_maybe_duration(audio_path),
                speaker_id=row.get("client_id"),
                source="common_voice",
                metadata={"up_votes": up, "down_votes": down},
            )


def load_openslr(transcript_file: str, audio_dir: str,
                 language: Optional[str] = None,
                 audio_ext: str = ".wav") -> Iterator[AudioSample]:
    """OpenSLR style: lines of 'utt_id transcript...'."""
    base = Path(audio_dir)
    for line in Path(transcript_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        utt_id, transcript = parts
        audio_path = str(base / f"{utt_id}{audio_ext}")
        yield AudioSample(
            id=utt_id,
            audio_path=audio_path,
            transcript=transcript.strip(),
            language=language,
            duration_sec=_maybe_duration(audio_path),
            speaker_id=utt_id.rsplit("_", 1)[0] if "_" in utt_id else None,
            source="openslr",
        )
