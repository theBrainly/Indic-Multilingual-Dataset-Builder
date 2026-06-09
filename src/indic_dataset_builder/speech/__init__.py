"""Speech/Audio dataset curation (ASR / TTS / speech translation).

Covers the JD's "Speech & Audio Dataset Curation" area. Speech-dataset curation
is largely manifest curation: validating audio/transcript manifests, normalizing
transcripts, filtering by duration/quality, deduping transcripts, and reporting
hours and speaker diversity. Loaders for Common Voice and OpenSLR-style layouts
are provided. Actual audio decoding (soundfile/librosa) is optional.
"""
from .schema import AudioSample  # noqa: F401
from .loaders import load_common_voice, load_openslr  # noqa: F401
from .curate import SpeechCurator  # noqa: F401
