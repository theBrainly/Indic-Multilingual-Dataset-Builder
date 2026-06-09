"""Audio sample model for speech datasets."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AudioSample(BaseModel):
    """One utterance: audio clip + transcript + metadata."""

    id: str
    audio_path: str
    transcript: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    speaker_id: Optional[str] = None
    split: Optional[str] = None          # train | dev | test
    source: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    dropped: bool = False
    drop_reason: Optional[str] = None

    def drop(self, reason: str) -> "AudioSample":
        self.dropped = True
        self.drop_reason = reason
        return self

    def to_manifest_row(self) -> Dict[str, Any]:
        """NeMo/ESPnet-style manifest row."""
        return {
            "audio_filepath": self.audio_path,
            "text": self.transcript,
            "duration": self.duration_sec,
            "language": self.language,
            "speaker_id": self.speaker_id,
        }
