"""Self-instruct style data generation.

A `Generator` produces new instruction examples from seed tasks. The bundled
`TemplateGenerator` is deterministic and offline (good for demos/CI); an
LLM-backed generator implementing the same `.generate()` interface can be
substituted in production. `SelfInstructBuilder` orchestrates generation +
on-the-fly dedup against the seed pool, mirroring the real self-instruct loop.
"""
from __future__ import annotations

import abc
import hashlib
from typing import List

from .schema import InstructionExample


class Generator(abc.ABC):
    @abc.abstractmethod
    def generate(self, seed: InstructionExample, n: int) -> List[InstructionExample]:
        ...


class TemplateGenerator(Generator):
    """Offline, deterministic paraphrase-style expansion of a seed task."""

    _TEMPLATES = [
        "{instruction}",
        "Please {lower_instruction}",
        "Can you {lower_instruction}",
        "{instruction} Explain briefly.",
        "As an expert, {lower_instruction}",
    ]

    def generate(self, seed: InstructionExample, n: int) -> List[InstructionExample]:
        out: List[InstructionExample] = []
        for i in range(n):
            tpl = self._TEMPLATES[i % len(self._TEMPLATES)]
            instr = tpl.format(
                instruction=seed.instruction,
                lower_instruction=seed.instruction[0].lower() + seed.instruction[1:]
                if seed.instruction else seed.instruction,
            )
            new_id = hashlib.md5(f"{seed.id}-{i}-{instr}".encode()).hexdigest()[:12]
            out.append(InstructionExample(
                id=f"syn-{new_id}",
                instruction=instr,
                input=seed.input,
                output=seed.output,
                language=seed.language,
                task_type=seed.task_type,
                origin="synthetic",
                metadata={"seed_id": seed.id},
            ))
        return out


class SelfInstructBuilder:
    def __init__(self, generator: Generator | None = None):
        self.generator = generator or TemplateGenerator()

    def build(self, seeds: List[InstructionExample],
              per_seed: int = 4) -> List[InstructionExample]:
        """Generate new examples, deduping against seeds and each other."""
        seen = {self._norm(s.instruction) for s in seeds}
        results: List[InstructionExample] = []
        for seed in seeds:
            for ex in self.generator.generate(seed, per_seed):
                key = self._norm(ex.instruction)
                if key in seen:
                    continue
                seen.add(key)
                results.append(ex)
        return results

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.lower().split())
