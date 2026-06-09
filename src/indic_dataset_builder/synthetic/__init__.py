"""Synthetic data development & validation.

Covers the JD's "Synthetic Data Development" area: self-instruct datasets,
reasoning traces, tool-use trajectories, and preference data. Generation uses a
pluggable `Generator` interface — a template generator runs offline for the
demo, and an LLM-backed generator can be dropped in. Validation (dedup vs.
seeds, quality, schema checks) is the curation-critical part and is fully real.
"""
from .schema import InstructionExample, ToolUseTrajectory  # noqa: F401
from .self_instruct import TemplateGenerator, SelfInstructBuilder  # noqa: F401
from .validate import validate_instructions, validate_trajectory  # noqa: F401
