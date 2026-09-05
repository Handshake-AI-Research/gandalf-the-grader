"""Tests for gandalf.models."""

import os
import pathlib
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from infinity_grader.models import (
    BatchJudgeInput,
    CriterionResult,
    EvaluationInfo,
    GraderConfig,
    JudgeInput,
    MCPServer,
    RubricItem,
    Verdict,
    load_config,
    load_rubric,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadConfig:
    def test_parses_all_fields(self) -> None:
        cfg = load_config(os.path.join(FIXTURES, "sample_grader.toml"))
        assert cfg.model == "gemini/gemini-2.5-flash"
        assert cfg.sandbox_user == "sandbox"
        assert cfg.instructions == "Build a web app that displays hello world."
        assert cfg.rubric_path == "/tests/rubric.json"
        assert cfg.workdir == "/home/agent/workspace"
        assert cfg.trajectory_path == "/logs/agent/trajectory.json"
        assert cfg.output_dir == "/logs/grader"
        assert cfg.judge_timeout == 120

    def test_parses_mcp_servers(self) -> None:
        cfg = load_config(os.path.join(FIXTURES, "sample_grader.toml"))
        assert len(cfg.mcp_servers) == 1
        mcp = cfg.mcp_servers[0]
        assert mcp.name == "magic-server"
        assert mcp.transport == "stdio"
        assert mcp.command == "/usr/bin/mcp-server"
        assert mcp.args == ["--verbose"]

    def test_defaults_model(self, tmp_path: pathlib.Path) -> None:
        toml_content = """\
sandbox_user = "sandbox"
instructions = "Do something."
rubric_path = "/tests/rubric.json"
workdir = "/workspace"
trajectory_path = "/logs/trajectory.json"
output_dir = "/logs/grader"
"""
        p = tmp_path / "grader.toml"
        p.write_text(toml_content)
        cfg = load_config(str(p))
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_defaults_timeout(self, tmp_path: pathlib.Path) -> None:
        toml_content = """\
model = "openai/gpt-4o"
sandbox_user = "sandbox"
instructions = "Do something."
rubric_path = "/tests/rubric.json"
workdir = "/workspace"
trajectory_path = "/logs/trajectory.json"
output_dir = "/logs/grader"
"""
        p = tmp_path / "grader.toml"
        p.write_text(toml_content)
        cfg = load_config(str(p))
        assert cfg.judge_timeout == 300

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/grader.toml")

    def test_missing_required_field_raises(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "bad.toml"
        p.write_text('model = "x"\n')
        with pytest.raises(ValidationError):
            load_config(str(p))


class TestLoadRubric:
    def test_parses_items(self) -> None:
        rubric = load_rubric(os.path.join(FIXTURES, "sample_rubric.json"))
        assert len(rubric) == 3
        assert rubric[0].criterion == "The file index.html exists in the workspace"
        assert rubric[0].weight == 1.0
        assert rubric[1].weight == 2.0

    def test_empty_rubric(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("[]")
        rubric = load_rubric(str(p))
        assert rubric == []

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_rubric("/nonexistent/rubric.json")

    def test_parses_negative_weight_items(self) -> None:
        rubric = load_rubric(os.path.join(FIXTURES, "sample_rubric_with_negatives.json"))
        assert len(rubric) == 3
        assert rubric[0].weight == 2.0
        assert rubric[1].weight == 3.0
        assert rubric[2].weight == -1.0


class TestPydanticModels:
    def test_mcp_server_defaults(self) -> None:
        srv = MCPServer(name="test", command="/bin/test")
        assert srv.transport == "stdio"
        assert srv.args == []
        assert srv.url is None
        assert srv.headers == {}

    def test_mcp_server_stdio_requires_command(self) -> None:
        with pytest.raises(ValidationError, match="'command' is required"):
            MCPServer(name="test")

    def test_mcp_server_remote_streamable_http(self) -> None:
        srv = MCPServer(
            name="remote",
            transport="streamable-http",
            url="http://localhost:8000/mcp",
        )
        assert srv.transport == "streamable-http"
        assert srv.url == "http://localhost:8000/mcp"
        assert srv.command is None

    def test_mcp_server_remote_http_with_headers(self) -> None:
        srv = MCPServer(
            name="remote",
            transport="http",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert srv.transport == "http"
        assert srv.headers == {"Authorization": "Bearer token"}

    def test_mcp_server_remote_sse(self) -> None:
        srv = MCPServer(
            name="remote",
            transport="sse",
            url="http://localhost:8000/sse",
        )
        assert srv.transport == "sse"

    def test_mcp_server_remote_requires_url(self) -> None:
        with pytest.raises(ValidationError, match="'url' is required"):
            MCPServer(name="remote", transport="streamable-http")

    def test_mcp_server_rejects_unknown_transport(self) -> None:
        with pytest.raises(ValidationError):
            MCPServer(name="test", command="/bin/test", transport="grpc")  # type: ignore[arg-type]

    def test_grader_config_has_trajectory_path(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/grader",
        )
        assert cfg.trajectory_path == "/logs/trajectory.json"
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_grader_config_judge_guidance_path_defaults_none(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/grader",
        )
        assert cfg.judge_guidance_path is None

    def test_grader_config_judge_guidance_path_set(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/grader",
            judge_guidance_path="/opt/grader/judge-guidance.md",
        )
        assert cfg.judge_guidance_path == "/opt/grader/judge-guidance.md"

    def test_judge_input_includes_final_output(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="agent said done",
            criterion="check something",
            workdir="/workspace",
        )
        assert ji.final_output == "agent said done"

    def test_judge_input_guidance_defaults_empty(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="done",
            criterion="check",
            workdir="/workspace",
        )
        assert ji.judge_guidance == ""

    def test_judge_input_guidance_roundtrip(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="done",
            criterion="check",
            workdir="/workspace",
            judge_guidance="Use openpyxl for .xlsx files.",
        )
        raw = ji.model_dump_json()
        restored = JudgeInput.model_validate_json(raw)
        assert restored.judge_guidance == "Use openpyxl for .xlsx files."

    def test_verdict_defaults(self) -> None:
        v = Verdict(met=True, reasoning="ok")
        assert v.evidence == []

    def test_verdict_with_evidence(self) -> None:
        v = Verdict(met=False, reasoning="fail", evidence=["check1", "check2"])
        assert len(v.evidence) == 2

    def test_verdict_met_none(self) -> None:
        v = Verdict(met=None, reasoning="error")
        assert v.met is None
        data = v.model_dump()
        assert data["met"] is None

    def test_verdict_none_serialization_roundtrip(self) -> None:
        v = Verdict(met=None, reasoning="error")
        raw = v.model_dump_json()
        restored = Verdict.model_validate_json(raw)
        assert restored.met is None

    def test_criterion_result(self) -> None:
        r = CriterionResult(
            criterion="test",
            weight=1.0,
            met=True,
            reasoning="ok",
        )
        assert r.evidence == []

    def test_criterion_result_negative_weight(self) -> None:
        r = CriterionResult(
            criterion="used hardcoded values",
            weight=-1.0,
            met=True,
            reasoning="found hardcoded values",
        )
        assert r.weight == -1.0

    def test_criterion_result_met_none(self) -> None:
        r = CriterionResult(criterion="test", weight=1.0, met=None, reasoning="error")
        assert r.met is None
        data = r.model_dump()
        assert data["met"] is None

    def test_evaluation_info(self) -> None:
        info = EvaluationInfo(
            reward=0.5,
            raw_score=3.0,
            minimum_score=-1.0,
            maximum_score=6.0,
            criterion_results=[
                CriterionResult(criterion="c1", weight=3.0, met=True, reasoning="ok"),
                CriterionResult(criterion="c2", weight=3.0, met=False, reasoning="fail"),
                CriterionResult(criterion="c3", weight=-1.0, met=False, reasoning="avoided"),
            ],
        )
        assert info.reward == 0.5
        assert info.raw_score == 3.0
        assert info.minimum_score == -1.0
        assert info.maximum_score == 6.0
        assert len(info.criterion_results) == 3

    def test_grader_config_sandbox_user_defaults_none(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            output_dir="/logs/grader",
        )
        assert cfg.sandbox_user is None

    def test_grader_config_sandbox_user_explicit(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            output_dir="/logs/grader",
            sandbox_user="sandbox",
        )
        assert cfg.sandbox_user == "sandbox"

    def test_grader_config_sandbox_user_omitted_from_toml(self, tmp_path: pathlib.Path) -> None:
        toml_content = """\
instructions = "Do something."
rubric_path = "/tests/rubric.json"
workdir = "/workspace"
trajectory_path = "/logs/trajectory.json"
output_dir = "/logs/grader"
"""
        p = tmp_path / "grader.toml"
        p.write_text(toml_content)
        cfg = load_config(str(p))
        assert cfg.sandbox_user is None

    def test_grader_config_judge_retries_default(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/grader",
        )
        assert cfg.judge_retries == 1

    def test_grader_config_judge_retries_explicit(self) -> None:
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/grader",
            judge_retries=3,
        )
        assert cfg.judge_retries == 3

    def test_evaluation_info_errored_fields(self) -> None:
        info = EvaluationInfo(
            reward=0.5,
            raw_score=1.0,
            criterion_results=[
                CriterionResult(criterion="c1", weight=1.0, met=True, reasoning="ok"),
                CriterionResult(criterion="c2", weight=1.0, met=None, reasoning="error"),
            ],
            errored_criterion_count=1,
            evaluated_criteria_pct=50.0,
        )
        assert info.errored_criterion_count == 1
        assert info.evaluated_criteria_pct == 50.0

    def test_evaluation_info_errored_fields_default(self) -> None:
        info = EvaluationInfo(
            reward=1.0,
            raw_score=1.0,
            criterion_results=[
                CriterionResult(criterion="c1", weight=1.0, met=True, reasoning="ok"),
            ],
        )
        assert info.errored_criterion_count == 0
        assert info.evaluated_criteria_pct == 100.0

    def test_judge_input_model_copy(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="agent said done",
            criterion="check something",
            workdir="/workspace",
        )
        cloned = ji.model_copy(update={"workdir": "/new-workspace"})
        assert cloned.workdir == "/new-workspace"
        assert ji.workdir == "/workspace"

    def test_judge_input_serialization(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="agent said done",
            criterion="check something",
            workdir="/workspace",
            mcp_servers=[MCPServer(name="srv", command="/bin/srv")],
        )
        raw = ji.model_dump_json()
        restored = JudgeInput.model_validate_json(raw)
        assert restored.model == ji.model
        assert restored.final_output == ji.final_output
        assert len(restored.mcp_servers) == 1


class TestMutualExclusivity:
    """Verify that inline and path variants cannot both be set."""

    def base_kwargs(self) -> dict[str, Any]:
        return {
            "instructions": "test",
            "rubric_path": "/rubric.json",
            "workdir": "/workspace",
            "trajectory_path": "/logs/trajectory.json",
            "sandbox_user": "sandbox",
            "output_dir": "/logs/grader",
        }

    def test_instructions_inline_only(self) -> None:
        cfg = GraderConfig(**self.base_kwargs())
        assert cfg.instructions == "test"
        assert cfg.instructions_path is None

    def test_instructions_path_only(self) -> None:
        kw = self.base_kwargs()
        del kw["instructions"]
        cfg = GraderConfig(**kw, instructions_path="/some/instructions.md")
        assert cfg.instructions_path == "/some/instructions.md"
        assert cfg.instructions is None

    def test_instructions_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="instructions"):
            GraderConfig(
                **self.base_kwargs(),
                instructions_path="/some/instructions.md",
            )

    def test_instructions_neither_is_valid(self) -> None:
        kw = self.base_kwargs()
        del kw["instructions"]
        cfg = GraderConfig(**kw)
        assert cfg.instructions is None
        assert cfg.instructions_path is None

    def test_rubric_inline_only(self) -> None:
        kw = self.base_kwargs()
        del kw["rubric_path"]
        cfg = GraderConfig(**kw, rubric=[RubricItem(criterion="c", weight=1.0)])
        assert cfg.rubric is not None
        assert cfg.rubric_path is None

    def test_rubric_path_only(self) -> None:
        cfg = GraderConfig(**self.base_kwargs())
        assert cfg.rubric_path == "/rubric.json"
        assert cfg.rubric is None

    def test_rubric_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="rubric"):
            GraderConfig(
                **self.base_kwargs(),
                rubric=[RubricItem(criterion="c", weight=1.0)],
            )

    def test_rubric_neither_raises(self) -> None:
        kw = self.base_kwargs()
        del kw["rubric_path"]
        with pytest.raises(ValidationError, match="rubric"):
            GraderConfig(**kw)

    def test_judge_guidance_inline_only(self) -> None:
        cfg = GraderConfig(**self.base_kwargs(), judge_guidance="inline text")
        assert cfg.judge_guidance == "inline text"
        assert cfg.judge_guidance_path is None

    def test_judge_guidance_path_only(self) -> None:
        cfg = GraderConfig(**self.base_kwargs(), judge_guidance_path="/some/file.md")
        assert cfg.judge_guidance_path == "/some/file.md"
        assert cfg.judge_guidance is None

    def test_judge_guidance_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="judge_guidance"):
            GraderConfig(
                **self.base_kwargs(),
                judge_guidance="inline",
                judge_guidance_path="/some/file.md",
            )

    def test_judge_prompt_inline_only(self) -> None:
        cfg = GraderConfig(**self.base_kwargs(), judge_prompt="template text")
        assert cfg.judge_prompt == "template text"
        assert cfg.judge_prompt_path is None

    def test_judge_prompt_path_only(self) -> None:
        cfg = GraderConfig(**self.base_kwargs(), judge_prompt_path="/some/template.j2")
        assert cfg.judge_prompt_path == "/some/template.j2"
        assert cfg.judge_prompt is None

    def test_judge_prompt_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="judge_prompt"):
            GraderConfig(
                **self.base_kwargs(),
                judge_prompt="inline",
                judge_prompt_path="/some/template.j2",
            )

    def test_neither_set_is_valid(self) -> None:
        cfg = GraderConfig(**self.base_kwargs())
        assert cfg.judge_guidance is None
        assert cfg.judge_guidance_path is None
        assert cfg.judge_prompt is None
        assert cfg.judge_prompt_path is None


class TestJudgePrompt:
    """Verify judge_prompt field on JudgeInput / BatchJudgeInput."""

    def test_judge_input_defaults_none(self) -> None:
        ji = JudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criterion="c",
            workdir="/w",
        )
        assert ji.judge_prompt is None

    def test_judge_input_roundtrip(self) -> None:
        ji = JudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criterion="c",
            workdir="/w",
            judge_prompt="Hello {{ instructions }}",
        )
        raw = ji.model_dump_json()
        restored = JudgeInput.model_validate_json(raw)
        assert restored.judge_prompt == "Hello {{ instructions }}"

    def test_batch_judge_input_defaults_none(self) -> None:
        bji = BatchJudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criteria=[],
            workdir="/w",
        )
        assert bji.judge_prompt is None

    def test_batch_judge_input_roundtrip(self) -> None:
        bji = BatchJudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criteria=[],
            workdir="/w",
            judge_prompt="Batch {{ n_max }}",
        )
        raw = bji.model_dump_json()
        restored = BatchJudgeInput.model_validate_json(raw)
        assert restored.judge_prompt == "Batch {{ n_max }}"


class TestGraderConfigMode:
    """Validate mode, batch_splits, and max_concurrency field constraints."""

    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "instructions": "test",
        "rubric_path": "/rubric.json",
        "workdir": "/workspace",
        "trajectory_path": "/logs/trajectory.json",
        "output_dir": "/logs/grader",
    }

    def _cfg(self, **overrides: Any) -> GraderConfig:
        return GraderConfig(**{**self._DEFAULTS, **overrides})

    def test_mode_defaults_to_batch(self) -> None:
        assert self._cfg().mode == "batch"

    def test_mode_individual_accepted(self) -> None:
        assert self._cfg(mode="individual").mode == "individual"

    def test_mode_sequential_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(mode="sequential")

    def test_mode_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(mode="parallel")

    # -- batch_splits --

    def test_batch_splits_defaults_none(self) -> None:
        assert self._cfg().batch_splits is None

    def test_batch_splits_2_accepted(self) -> None:
        assert self._cfg(mode="batch", batch_splits=2).batch_splits == 2

    def test_batch_splits_10_accepted(self) -> None:
        assert self._cfg(mode="batch", batch_splits=10).batch_splits == 10

    def test_batch_splits_1_rejected(self) -> None:
        """batch_splits must be >= 2 (splitting into 1 chunk is meaningless)."""
        with pytest.raises(ValidationError):
            self._cfg(mode="batch", batch_splits=1)

    def test_batch_splits_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(mode="batch", batch_splits=0)

    def test_batch_splits_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(mode="batch", batch_splits=-1)

    def test_batch_splits_with_individual_mode_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"batch_splits.*batch"):
            self._cfg(mode="individual", batch_splits=3)

    # -- max_concurrency --

    def test_max_concurrency_defaults_none(self) -> None:
        assert self._cfg().max_concurrency is None

    def test_max_concurrency_1_accepted(self) -> None:
        assert self._cfg(max_concurrency=1).max_concurrency == 1

    def test_max_concurrency_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(max_concurrency=0)

    def test_max_concurrency_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(max_concurrency=-1)

    # -- combined --

    def test_batch_splits_and_max_concurrency_independent(self) -> None:
        """batch_splits=10 with max_concurrency=2 is valid — 10 chunks, 2 parallel."""
        cfg = self._cfg(mode="batch", batch_splits=10, max_concurrency=2)
        assert cfg.batch_splits == 10
        assert cfg.max_concurrency == 2

    def test_individual_with_max_concurrency(self) -> None:
        cfg = self._cfg(mode="individual", max_concurrency=4)
        assert cfg.max_concurrency == 4
        assert cfg.batch_splits is None
