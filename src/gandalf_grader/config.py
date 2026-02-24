"""Configuration models for the verifier."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter


class MCPServer(BaseModel):
    """Configuration for a stdio MCP server.

    Only stdio transport is supported (OpenHands SDK limitation).
    """

    name: str
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)


class VerifierConfig(BaseModel):
    """Top-level verifier configuration loaded from a TOML file.

    mode controls how rubric criteria are evaluated:
      - "sequential" (default): each criterion is evaluated in its own agent
        session (one invocation of gandalf-grader-judge per criterion).
      - "batch": all criteria are sent to a single agent session, which writes
        a JSON array of verdicts in one go.

    judge_timeout is the per-criterion budget in seconds, regardless of mode.
    In batch mode the effective timeout is ``judge_timeout * N_criteria``,
    optionally capped by batch_timeout.
    """

    model: str = "google/gemini-2.5-flash"
    instructions: str
    rubric_path: str
    workdir: str
    trajectory_path: str
    sandbox_user: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    output_dir: str = "/logs/verifier"
    judge_timeout: int = 300
    judge_guidance_path: str | None = None
    batch_timeout: int | None = None
    mode: Literal["sequential", "batch"] = "sequential"


class RubricItem(BaseModel):
    """A single rubric item with evaluation criteria and weight."""

    criteria: str
    weight: float


class JudgeInput(BaseModel):
    """Input passed to the inner judge for a single criteria evaluation."""

    model: str
    instructions: str
    final_output: str
    criteria: str
    workdir: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    judge_guidance: str = ""


class BatchCriterion(BaseModel):
    """A single criterion entry within a batch judge input."""

    index: int
    criteria: str
    weight: float


class BatchJudgeInput(BaseModel):
    """Input passed to the inner judge for batch (all-criteria) evaluation."""

    model: str
    instructions: str
    final_output: str
    criteria: list[BatchCriterion]
    workdir: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    judge_guidance: str = ""


class Verdict(BaseModel):
    """Verdict returned by the inner judge."""

    passed: bool
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


class CriteriaResult(BaseModel):
    """Result for a single criteria evaluation."""

    criteria: str
    weight: float
    passed: bool
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


class EvaluationInfo(BaseModel):
    """Full evaluation output with score, per-criteria results, and LLM usage."""

    score: float
    criteria_results: list[CriteriaResult]
    llm_usage: dict[str, float | int | str] = Field(default_factory=dict)


def load_config(path: str) -> VerifierConfig:
    """Load verifier configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return VerifierConfig.model_validate(data)


def load_rubric(path: str) -> list[RubricItem]:
    """Load rubric items from a JSON file."""
    raw = Path(path).read_bytes()
    return TypeAdapter(list[RubricItem]).validate_json(raw)
