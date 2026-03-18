"""Tests for gandalf.config."""

import os
import pathlib
from typing import Any

import pytest
from pydantic import ValidationError

from gandalf.config import (
    BatchJudgeInput,
    CriteriaResult,
    EvaluationInfo,
    GraderConfig,
    JudgeInput,
    MCPServer,
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
        assert rubric[0].criteria == "The file index.html exists in the workspace"
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

    def test_mcp_server_rejects_non_stdio_transport(self) -> None:
        with pytest.raises(ValidationError):
            MCPServer(name="test", command="/bin/test", transport="sse")  # type: ignore[arg-type]

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
            criteria="check something",
            workdir="/workspace",
        )
        assert ji.final_output == "agent said done"

    def test_judge_input_guidance_defaults_empty(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="done",
            criteria="check",
            workdir="/workspace",
        )
        assert ji.judge_guidance == ""

    def test_judge_input_guidance_roundtrip(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="done",
            criteria="check",
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

    def test_criteria_result(self) -> None:
        r = CriteriaResult(
            criteria="test",
            weight=1.0,
            met=True,
            reasoning="ok",
        )
        assert r.evidence == []

    def test_criteria_result_negative_weight(self) -> None:
        r = CriteriaResult(
            criteria="used hardcoded values",
            weight=-1.0,
            met=True,
            reasoning="found hardcoded values",
        )
        assert r.weight == -1.0

    def test_criteria_result_met_none(self) -> None:
        r = CriteriaResult(criteria="test", weight=1.0, met=None, reasoning="error")
        assert r.met is None
        data = r.model_dump()
        assert data["met"] is None

    def test_evaluation_info(self) -> None:
        info = EvaluationInfo(
            reward=0.5,
            raw_score=3.0,
            minimum_score=-1.0,
            maximum_score=6.0,
            criteria_results=[
                CriteriaResult(criteria="c1", weight=3.0, met=True, reasoning="ok"),
                CriteriaResult(criteria="c2", weight=3.0, met=False, reasoning="fail"),
                CriteriaResult(criteria="c3", weight=-1.0, met=False, reasoning="avoided"),
            ],
        )
        assert info.reward == 0.5
        assert info.raw_score == 3.0
        assert info.minimum_score == -1.0
        assert info.maximum_score == 6.0
        assert len(info.criteria_results) == 3

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
            criteria_results=[
                CriteriaResult(criteria="c1", weight=1.0, met=True, reasoning="ok"),
                CriteriaResult(criteria="c2", weight=1.0, met=None, reasoning="error"),
            ],
            errored_criteria_count=1,
            evaluated_criteria_pct=50.0,
        )
        assert info.errored_criteria_count == 1
        assert info.evaluated_criteria_pct == 50.0

    def test_evaluation_info_errored_fields_default(self) -> None:
        info = EvaluationInfo(
            reward=1.0,
            raw_score=1.0,
            criteria_results=[
                CriteriaResult(criteria="c1", weight=1.0, met=True, reasoning="ok"),
            ],
        )
        assert info.errored_criteria_count == 0
        assert info.evaluated_criteria_pct == 100.0

    def test_judge_input_model_copy(self) -> None:
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="agent said done",
            criteria="check something",
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
            criteria="check something",
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

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "instructions": "test",
            "rubric_path": "/rubric.json",
            "workdir": "/workspace",
            "trajectory_path": "/logs/trajectory.json",
            "sandbox_user": "sandbox",
            "output_dir": "/logs/grader",
        }

    def test_judge_guidance_inline_only(self) -> None:
        cfg = GraderConfig(**self._base_kwargs(), judge_guidance="inline text")
        assert cfg.judge_guidance == "inline text"
        assert cfg.judge_guidance_path is None

    def test_judge_guidance_path_only(self) -> None:
        cfg = GraderConfig(**self._base_kwargs(), judge_guidance_path="/some/file.md")
        assert cfg.judge_guidance_path == "/some/file.md"
        assert cfg.judge_guidance is None

    def test_judge_guidance_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="judge_guidance"):
            GraderConfig(
                **self._base_kwargs(),
                judge_guidance="inline",
                judge_guidance_path="/some/file.md",
            )

    def test_system_prompt_inline_only(self) -> None:
        cfg = GraderConfig(**self._base_kwargs(), system_prompt="template text")
        assert cfg.system_prompt == "template text"
        assert cfg.system_prompt_path is None

    def test_system_prompt_path_only(self) -> None:
        cfg = GraderConfig(**self._base_kwargs(), system_prompt_path="/some/template.j2")
        assert cfg.system_prompt_path == "/some/template.j2"
        assert cfg.system_prompt is None

    def test_system_prompt_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="system_prompt"):
            GraderConfig(
                **self._base_kwargs(),
                system_prompt="inline",
                system_prompt_path="/some/template.j2",
            )

    def test_neither_set_is_valid(self) -> None:
        cfg = GraderConfig(**self._base_kwargs())
        assert cfg.judge_guidance is None
        assert cfg.judge_guidance_path is None
        assert cfg.system_prompt is None
        assert cfg.system_prompt_path is None


class TestSystemPromptTemplate:
    """Verify system_prompt_template field on JudgeInput / BatchJudgeInput."""

    def test_judge_input_defaults_none(self) -> None:
        ji = JudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criteria="c",
            workdir="/w",
        )
        assert ji.system_prompt_template is None

    def test_judge_input_roundtrip(self) -> None:
        ji = JudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criteria="c",
            workdir="/w",
            system_prompt_template="Hello {{ instructions }}",
        )
        raw = ji.model_dump_json()
        restored = JudgeInput.model_validate_json(raw)
        assert restored.system_prompt_template == "Hello {{ instructions }}"

    def test_batch_judge_input_defaults_none(self) -> None:
        bji = BatchJudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criteria=[],
            workdir="/w",
        )
        assert bji.system_prompt_template is None

    def test_batch_judge_input_roundtrip(self) -> None:
        bji = BatchJudgeInput(
            model="m",
            instructions="i",
            final_output="o",
            criteria=[],
            workdir="/w",
            system_prompt_template="Batch {{ n_max }}",
        )
        raw = bji.model_dump_json()
        restored = BatchJudgeInput.model_validate_json(raw)
        assert restored.system_prompt_template == "Batch {{ n_max }}"
