"""Schemas for synthetic training data."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InstructionExample(BaseModel):
    """A self-instruct style (instruction, input, output) triple."""

    id: str
    instruction: str
    input: str = ""
    output: str
    language: Optional[str] = None
    task_type: str = "open_qa"
    origin: str = "synthetic"            # synthetic | seed | human
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def as_text(self) -> str:
        parts = [self.instruction]
        if self.input:
            parts.append(self.input)
        parts.append(self.output)
        return "\n".join(parts)


class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None


class ToolUseTrajectory(BaseModel):
    """A tool-use trajectory: a goal and the sequence of tool calls to reach it."""

    id: str
    goal: str
    steps: List[ToolCall] = Field(default_factory=list)
    final_answer: str = ""
    available_tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
