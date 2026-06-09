"""Validation for synthetic data.

Generation is cheap; validation is what makes synthetic data safe to train on.
We check instruction examples for empty/degenerate fields, output==instruction
echoes, and length sanity; and we validate tool-use trajectories for
well-formedness (declared tools, non-empty steps, final answer present).
"""
from __future__ import annotations

from typing import Any, Dict, List

from .schema import InstructionExample, ToolUseTrajectory


def validate_instructions(examples: List[InstructionExample]) -> Dict[str, Any]:
    clean: List[InstructionExample] = []
    issues: List[Dict[str, str]] = []
    for ex in examples:
        if not ex.instruction.strip():
            issues.append({"id": ex.id, "issue": "empty_instruction"})
            continue
        if not ex.output.strip():
            issues.append({"id": ex.id, "issue": "empty_output"})
            continue
        if ex.output.strip().lower() == ex.instruction.strip().lower():
            issues.append({"id": ex.id, "issue": "output_echoes_instruction"})
            continue
        if len(ex.output) < 2:
            issues.append({"id": ex.id, "issue": "output_too_short"})
            continue
        clean.append(ex)
    return {"total": len(examples), "clean": len(clean),
            "issues": issues, "clean_examples": clean}


def validate_trajectory(traj: ToolUseTrajectory) -> Dict[str, Any]:
    problems: List[str] = []
    if not traj.goal.strip():
        problems.append("empty_goal")
    if not traj.steps:
        problems.append("no_steps")
    if not traj.final_answer.strip():
        problems.append("no_final_answer")
    allowed = set(traj.available_tools)
    for i, step in enumerate(traj.steps):
        if allowed and step.tool not in allowed:
            problems.append(f"step{i}_undeclared_tool:{step.tool}")
        if not step.tool.strip():
            problems.append(f"step{i}_empty_tool")
    return {"id": traj.id, "valid": not problems, "problems": problems}
