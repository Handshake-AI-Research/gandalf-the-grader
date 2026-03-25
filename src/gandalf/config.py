"""Configuration models for the grader."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class MCPServer(BaseModel):
    """Configuration for a stdio MCP server.

    Only stdio transport is supported (OpenHands SDK limitation).
    """

    name: str
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)


class RubricItem(BaseModel):
    """A single rubric item with evaluation criteria and weight.

    Weight can be negative to penalise undesired outcomes.  The sign of the
    weight carries the semantics: positive means "reward when met", negative
    means "penalise when met".
    """

    criteria: str
    weight: float


class GraderConfig(BaseModel):
    """Top-level grader configuration loaded from a TOML file.

    mode controls how rubric criteria are evaluated:
      - "sequential" (default): each criterion is evaluated in its own agent
        session (one invocation of gandalf-the-grader-judge per criterion).
      - "batch": all criteria are sent to a single agent session, which writes
        a JSON array of verdicts in one go.

    judge_timeout is the per-criterion budget in seconds, regardless of mode.
    In batch mode the effective timeout is ``judge_timeout * N_criteria``,
    optionally capped by batch_timeout.
    """

    model: str = "gemini/gemini-2.5-flash"
    instructions: str
    rubric: list[RubricItem] | None = None
    rubric_path: str | None = None
    workdir: str
    trajectory_path: str
    sandbox_user: str | None = None
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    output_dir: str
    judge_timeout: int = 300
    judge_guidance: str | None = None
    judge_guidance_path: str | None = None
    judge_prompt: str | None = None
    judge_prompt_path: str | None = None
    batch_timeout: int | None = None
    mode: Literal["sequential", "batch"] = "batch"
    judge_retries: int = 1

    @model_validator(mode="after")
    def _check_no_inline_and_path(self) -> "GraderConfig":
        if self.rubric is not None and self.rubric_path is not None:
            msg = "Cannot set both 'rubric' and 'rubric_path'"
            raise ValueError(msg)
        if self.rubric is None and self.rubric_path is None:
            msg = "Must set either 'rubric' or 'rubric_path'"
            raise ValueError(msg)
        if self.judge_guidance is not None and self.judge_guidance_path is not None:
            msg = "Cannot set both 'judge_guidance' and 'judge_guidance_path'"
            raise ValueError(msg)
        if self.judge_prompt is not None and self.judge_prompt_path is not None:
            msg = "Cannot set both 'judge_prompt' and 'judge_prompt_path'"
            raise ValueError(msg)
        return self


class JudgeInput(BaseModel):
    """Input passed to the inner judge for a single criteria evaluation."""

    model: str
    instructions: str
    final_output: str
    criteria: str
    workdir: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    judge_guidance: str = ""
    judge_prompt: str | None = None


class BatchJudgeInput(BaseModel):
    """Input passed to the inner judge for batch (all-criteria) evaluation.

    Weights are intentionally omitted from criteria so the judge evaluates
    each criterion on its own merits.  Indices are derived from position.
    """

    model: str
    instructions: str
    final_output: str
    criteria: list[str]
    workdir: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    judge_guidance: str = ""
    judge_prompt: str | None = None


class LLMUsage(BaseModel):
    """Aggregate LLM token and cost metrics from judge sessions."""

    cost_usd: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0


class Verdict(BaseModel):
    """Verdict returned by the inner judge."""

    met: bool | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


class CriteriaResult(BaseModel):
    """Result for a single criteria evaluation."""

    criteria: str
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
    criteria_results: list[CriteriaResult]
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    errored_criteria_count: int = 0
    evaluated_criteria_pct: float = 100.0


def load_config(path: str) -> GraderConfig:
    """Load grader configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return GraderConfig.model_validate(data)


def load_rubric(path: str) -> list[RubricItem]:
    """Load rubric items from a JSON file."""
    raw = Path(path).read_bytes()
    return TypeAdapter(list[RubricItem]).validate_json(raw)
