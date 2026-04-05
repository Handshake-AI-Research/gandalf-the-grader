"""Configuration models for the verifier."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator


class MCPServer(BaseModel):
    """Configuration for a stdio MCP server.

    Only stdio transport is supported (OpenHands SDK limitation).
    """

    name: str
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)


class GraderConfig(BaseModel):
    """Top-level verifier configuration loaded from a TOML file.

    mode controls the *granularity* of rubric evaluation — how many criteria
    are sent to each judge session:
      - "individual" (default): one agent session per rubric criterion.
      - "batch": all criteria evaluated in a single agent session.

    max_concurrency controls the *parallelism* — how many judge sessions run
    at the same time:
      - None (default): no parallelism (1 session at a time).
      - N (>= 1): up to N sessions in parallel.

    These are orthogonal axes.  For batch mode, max_concurrency > 1 splits
    criteria into N positional chunks, each evaluated as a separate batch
    session.  For individual mode, max_concurrency > 1 runs N individual
    criterion evaluations in parallel.

    judge_timeout is the per-criterion budget in seconds, regardless of mode.
    In batch mode the effective timeout per session is
    ``judge_timeout * N_criteria_in_session``, optionally capped by
    batch_timeout.  When max_concurrency > 1, N_criteria_in_session is the
    chunk size (not the full rubric), and batch_timeout applies to each
    chunk independently.
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
    mode: Literal["individual", "batch"] = "individual"
    max_concurrency: int | None = Field(default=None, ge=1)
    judge_retries: int = 1

    @field_validator("mode", mode="before")
    @classmethod
    def _migrate_sequential(cls, v: str) -> str:
        if v == "sequential":
            import warnings
            warnings.warn(
                'mode="sequential" is deprecated, use mode="individual" instead',
                DeprecationWarning,
                stacklevel=2,
            )
            return "individual"
        return v


class RubricItem(BaseModel):
    """A single rubric item with evaluation criterion and weight.

    Weight can be negative to penalise undesired outcomes.  The sign of the
    weight carries the semantics: positive means "reward when met", negative
    means "penalise when met".
    """

    criterion: str
    weight: float


class JudgeInput(BaseModel):
    """Input passed to the inner judge for a single criterion evaluation."""

    model: str
    instructions: str
    final_output: str
    criterion: str
    workdir: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    judge_guidance: str = ""


class BatchCriterion(BaseModel):
    """A single criterion entry within a batch judge input.

    The judge sees only the index and criterion text — weights are intentionally
    omitted so the judge evaluates each criterion on its own merits.
    """

    index: int
    criterion: str


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

    met: bool | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


class CriterionResult(BaseModel):
    """Result for a single criterion evaluation."""

    criterion: str
    weight: float
    met: bool | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


class EvaluationInfo(BaseModel):
    """Full evaluation output with reward/raw score, per-criteria results, and LLM usage."""

    reward: float
    raw_score: float
    minimum_score: float = 0.0
    maximum_score: float = 0.0
    criteria_results: list[CriterionResult]
    llm_usage: dict[str, float | int | str] = Field(default_factory=dict)
    errored_criteria_count: int = 0
    evaluated_criteria_pct: float = 100.0


def load_config(path: str) -> GraderConfig:
    """Load verifier configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return GraderConfig.model_validate(data)


def load_rubric(path: str) -> list[RubricItem]:
    """Load rubric items from a JSON file."""
    raw = Path(path).read_bytes()
    return TypeAdapter(list[RubricItem]).validate_json(raw)
