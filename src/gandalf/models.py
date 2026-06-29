"""Data models and configuration loaders for the grader."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class MCPServer(BaseModel):
    """Configuration for an MCP server.

    Supports stdio (subprocess) and remote network transports
    (streamable-http, http, sse).  Stdio servers require ``command``;
    remote servers require ``url``.
    """

    name: str
    transport: Literal["stdio", "streamable-http", "http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "MCPServer":
        if self.transport == "stdio":
            if not self.command:
                msg = f"MCP server {self.name!r}: 'command' is required for stdio transport"
                raise ValueError(msg)
        elif not self.url:
            msg = f"MCP server {self.name!r}: 'url' is required for transport {self.transport!r}"
            raise ValueError(msg)
        return self


class RubricItem(BaseModel):
    """A single rubric item with an evaluation criterion and weight.

    Weight can be negative to penalise undesired outcomes.  The sign of the
    weight carries the semantics: positive means "reward when met", negative
    means "penalise when met".
    """

    criterion: str
    weight: float


class GraderConfig(BaseModel):
    """Top-level grader configuration loaded from a TOML file.

    mode controls how rubric criteria are evaluated:
      - "individual": each criterion is evaluated in its own agent
        session (one invocation of gandalf-the-grader-judge per criterion).
      - "batch" (default): all criteria are sent to a single agent session,
        which writes a JSON array of verdicts in one go.

    batch_splits controls how many chunks criteria are split into for batch
    mode.  Ignored in individual mode.
      - None (default): all criteria in a single batch session.
      - N (>= 2): criteria are split into N positional chunks, each sent as
        a separate batch session.

    max_concurrency controls how many judge sessions run in parallel:
      - None (default): interpreted as 1 for individual mode, and
        batch_splits for batch mode (i.e. all splits run in parallel).
      - N (>= 1): up to N sessions in parallel.

    judge_timeout is the per-criterion budget in seconds, regardless of mode.
    In batch mode the effective timeout per session is
    ``judge_timeout * n_criteria_in_session``, optionally capped by
    batch_timeout.
    """

    model: str = "gemini/gemini-2.5-flash"
    grading_mode: Literal["rubric", "guidance"] = "rubric"
    instructions: str | None = None
    instructions_path: str | None = None
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
    mode: Literal["individual", "batch"] = "batch"
    batch_splits: int | None = Field(default=None, ge=2)
    max_concurrency: int | None = Field(default=None, ge=1)
    judge_retries: int = 1

    @model_validator(mode="after")
    def _check_no_inline_and_path(self) -> "GraderConfig":
        if self.instructions is not None and self.instructions_path is not None:
            msg = "Cannot set both 'instructions' and 'instructions_path'"
            raise ValueError(msg)
        if self.rubric is not None and self.rubric_path is not None:
            msg = "Cannot set both 'rubric' and 'rubric_path'"
            raise ValueError(msg)
        if self.judge_guidance is not None and self.judge_guidance_path is not None:
            msg = "Cannot set both 'judge_guidance' and 'judge_guidance_path'"
            raise ValueError(msg)
        if self.judge_prompt is not None and self.judge_prompt_path is not None:
            msg = "Cannot set both 'judge_prompt' and 'judge_prompt_path'"
            raise ValueError(msg)

        if self.grading_mode == "rubric":
            if self.rubric is None and self.rubric_path is None:
                msg = "Must set either 'rubric' or 'rubric_path'"
                raise ValueError(msg)
            if self.batch_splits is not None and self.mode != "batch":
                msg = "'batch_splits' can only be used with mode='batch'"
                raise ValueError(msg)
        else:
            if self.rubric is not None or self.rubric_path is not None:
                msg = "Do not set 'rubric' or 'rubric_path' when grading_mode='guidance'"
                raise ValueError(msg)
            if self.judge_guidance is None and self.judge_guidance_path is None:
                msg = "Must set either 'judge_guidance' or 'judge_guidance_path' when grading_mode='guidance'"
                raise ValueError(msg)
            if self.judge_guidance is not None and not self.judge_guidance.strip():
                msg = "'judge_guidance' must not be empty when grading_mode='guidance'"
                raise ValueError(msg)
            if self.batch_splits is not None:
                msg = "'batch_splits' cannot be used when grading_mode='guidance'"
                raise ValueError(msg)
            if self.batch_timeout is not None:
                msg = "'batch_timeout' cannot be used when grading_mode='guidance'"
                raise ValueError(msg)
            if self.max_concurrency is not None:
                msg = "'max_concurrency' cannot be used when grading_mode='guidance'"
                raise ValueError(msg)
        return self


class _BaseJudgeInput(BaseModel):
    """Shared fields for all judge input types."""

    model: str
    instructions: str
    final_output: str
    workdir: str
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    judge_guidance: str = ""
    judge_prompt: str | None = None


class JudgeInput(_BaseJudgeInput):
    """Input passed to the inner judge for a single criterion evaluation."""

    criterion: str


class BatchJudgeInput(_BaseJudgeInput):
    """Input passed to the inner judge for batch (all-criteria) evaluation.

    Weights are intentionally omitted from criteria so the judge evaluates
    each criterion on its own merits.  Indices are derived from position.
    """

    criteria: list[str]


class GuidanceJudgeInput(_BaseJudgeInput):
    """Input passed to the inner judge for one holistic guidance evaluation."""

    trajectory_path: str


class LLMUsage(BaseModel):
    """Aggregate LLM token and cost metrics from judge sessions."""

    cost_usd: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Sum two LLMUsage instances field-by-field."""
        return LLMUsage(
            cost_usd=self.cost_usd + other.cost_usd,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


class Verdict(BaseModel):
    """Verdict returned by the inner judge."""

    met: bool | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "Verdict":
        """Create a Verdict from a raw JSON-parsed dict, normalizing types."""
        raw_met = data.get("met")
        return cls(
            met=bool(raw_met) if raw_met is not None else None,
            reasoning=str(data.get("reasoning", "No reasoning provided.")),
            evidence=list(data.get("evidence", [])),
        )

    @classmethod
    def errors(cls, n: int, reason: str) -> list["Verdict"]:
        """Return *n* error verdicts (met=None) that all share the same reason."""
        return [cls(met=None, reasoning=reason) for _ in range(n)]


class GuidanceScore(BaseModel):
    """Holistic score returned by guidance-mode judging."""

    score: float | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "GuidanceScore":
        """Create a GuidanceScore from a raw JSON-parsed dict, validating score bounds."""
        if "score" not in data:
            return cls(score=None, reasoning="Guidance score missing 'score' field.")

        raw_score = data.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            return cls(score=None, reasoning="Guidance score must be numeric in the range [0, 1].")

        score = float(raw_score)
        if score < 0.0 or score > 1.0:
            return cls(score=None, reasoning=f"Guidance score {score} is outside the valid range [0, 1].")

        reasoning = str(data.get("reasoning", "")).strip()
        if not reasoning:
            return cls(score=None, reasoning="Guidance score missing non-empty 'reasoning' field.")

        evidence = data.get("evidence")
        if not isinstance(evidence, list):
            return cls(score=None, reasoning="Guidance score must include an 'evidence' array.")

        evidence_items = [str(item).strip() for item in evidence if str(item).strip()]
        if not evidence_items:
            return cls(score=None, reasoning="Guidance score must include at least one evidence item.")

        return cls(
            score=score,
            reasoning=reasoning,
            evidence=evidence_items,
        )

    @classmethod
    def error(cls, reason: str) -> "GuidanceScore":
        """Return an invalid score with an error reason."""
        return cls(score=None, reasoning=reason, evidence=[])


class CriterionResult(BaseModel):
    """Result for a single criterion evaluation."""

    criterion: str
    weight: float
    met: bool | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)


class EvaluationInfo(BaseModel):
    """Full evaluation output with reward/raw score, per-criterion results, and LLM usage."""

    reward: float
    raw_score: float
    minimum_score: float = 0.0
    maximum_score: float = 0.0
    criterion_results: list[CriterionResult]
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    errored_criterion_count: int = 0
    evaluated_criteria_pct: float = 100.0


class GuidanceEvaluationInfo(BaseModel):
    """Detailed guidance-mode evaluation output."""

    grading_mode: Literal["guidance"] = "guidance"
    reward: float | None
    score: float | None
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    errored: bool = False
    error: str | None = None


def load_config(path: str) -> GraderConfig:
    """Load grader configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return GraderConfig.model_validate(data)


def load_rubric(path: str) -> list[RubricItem]:
    """Load rubric items from a JSON file."""
    raw = Path(path).read_bytes()
    return TypeAdapter(list[RubricItem]).validate_json(raw)
