"""Tests for gandalf.models."""

import os

import pytest
from pydantic import ValidationError

from gandalf.models import (
    CriterionResult,
    EvaluationInfo,
    JudgeInput,
    MCPServer,
    Verdict,
    GraderConfig,
    load_config,
    load_rubric,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadConfig:
    def test_parses_all_fields(self):
        cfg = load_config(os.path.join(FIXTURES, "sample_verifier.toml"))
        assert cfg.model == "google/gemini-2.5-flash"
        assert cfg.sandbox_user == "sandbox"
        assert cfg.instructions == "Build a web app that displays hello world."
        assert cfg.rubric_path == "/tests/rubric.json"
        assert cfg.workdir == "/home/agent/workspace"
        assert cfg.trajectory_path == "/logs/agent/trajectory.json"
        assert cfg.output_dir == "/logs/verifier"
        assert cfg.judge_timeout == 120

    def test_parses_mcp_servers(self):
        cfg = load_config(os.path.join(FIXTURES, "sample_verifier.toml"))
        assert len(cfg.mcp_servers) == 1
        mcp = cfg.mcp_servers[0]
        assert mcp.name == "magic-server"
        assert mcp.transport == "stdio"
        assert mcp.command == "/usr/bin/mcp-server"
        assert mcp.args == ["--verbose"]

    def test_defaults_model(self, tmp_path):
        toml_content = """\
sandbox_user = "sandbox"
instructions = "Do something."
rubric_path = "/tests/rubric.json"
workdir = "/workspace"
trajectory_path = "/logs/trajectory.json"
output_dir = "/logs/verifier"
"""
        p = tmp_path / "grader.toml"
        p.write_text(toml_content)
        cfg = load_config(str(p))
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_defaults_timeout(self, tmp_path):
        toml_content = """\
model = "openai/gpt-4o"
sandbox_user = "sandbox"
instructions = "Do something."
rubric_path = "/tests/rubric.json"
workdir = "/workspace"
trajectory_path = "/logs/trajectory.json"
output_dir = "/logs/verifier"
"""
        p = tmp_path / "grader.toml"
        p.write_text(toml_content)
        cfg = load_config(str(p))
        assert cfg.judge_timeout == 300

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/verifier.toml")

    def test_missing_required_field_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text('model = "x"\n')
        with pytest.raises(ValidationError):
            load_config(str(p))


class TestLoadRubric:
    def test_parses_items(self):
        rubric = load_rubric(os.path.join(FIXTURES, "sample_rubric.json"))
        assert len(rubric) == 3
        assert rubric[0].criterion == "The file index.html exists in the workspace"
        assert rubric[0].weight == 1.0
        assert rubric[1].weight == 2.0

    def test_empty_rubric(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        rubric = load_rubric(str(p))
        assert rubric == []

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_rubric("/nonexistent/rubric.json")

    def test_parses_negative_weight_items(self):
        rubric = load_rubric(os.path.join(FIXTURES, "sample_rubric_with_negatives.json"))
        assert len(rubric) == 3
        assert rubric[0].weight == 2.0
        assert rubric[1].weight == 3.0
        assert rubric[2].weight == -1.0


class TestPydanticModels:
    def test_mcp_server_defaults(self):
        srv = MCPServer(name="test", command="/bin/test")
        assert srv.transport == "stdio"
        assert srv.args == []

    def test_mcp_server_rejects_non_stdio_transport(self):
        with pytest.raises(ValidationError):
            MCPServer(name="test", command="/bin/test", transport="sse")

    def test_grader_config_has_trajectory_path(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
        )
        assert cfg.trajectory_path == "/logs/trajectory.json"
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_grader_config_judge_guidance_path_defaults_none(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
        )
        assert cfg.judge_guidance_path is None

    def test_grader_config_judge_guidance_path_set(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
            judge_guidance_path="/opt/verifier/judge-guidance.md",
        )
        assert cfg.judge_guidance_path == "/opt/verifier/judge-guidance.md"

    def test_judge_input_includes_final_output(self):
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="agent said done",
            criterion="check something",
            workdir="/workspace",
        )
        assert ji.final_output == "agent said done"

    def test_judge_input_guidance_defaults_empty(self):
        ji = JudgeInput(
            model="test-model",
            instructions="test",
            final_output="done",
            criterion="check",
            workdir="/workspace",
        )
        assert ji.judge_guidance == ""

    def test_judge_input_guidance_roundtrip(self):
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

    def test_verdict_defaults(self):
        v = Verdict(met=True, reasoning="ok")
        assert v.evidence == []

    def test_verdict_with_evidence(self):
        v = Verdict(met=False, reasoning="fail", evidence=["check1", "check2"])
        assert len(v.evidence) == 2

    def test_verdict_met_none(self):
        v = Verdict(met=None, reasoning="error")
        assert v.met is None
        data = v.model_dump()
        assert data["met"] is None

    def test_verdict_none_serialization_roundtrip(self):
        v = Verdict(met=None, reasoning="error")
        raw = v.model_dump_json()
        restored = Verdict.model_validate_json(raw)
        assert restored.met is None

    def test_criteria_result(self):
        r = CriterionResult(
            criterion="test",
            weight=1.0,
            met=True,
            reasoning="ok",
        )
        assert r.evidence == []

    def test_criteria_result_negative_weight(self):
        r = CriterionResult(
            criterion="used hardcoded values",
            weight=-1.0,
            met=True,
            reasoning="found hardcoded values",
        )
        assert r.weight == -1.0

    def test_criteria_result_met_none(self):
        r = CriterionResult(criterion="test", weight=1.0, met=None, reasoning="error")
        assert r.met is None
        data = r.model_dump()
        assert data["met"] is None

    def test_evaluation_info(self):
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

    def test_verifier_config_judge_retries_default(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
        )
        assert cfg.judge_retries == 1

    def test_verifier_config_judge_retries_explicit(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
            judge_retries=3,
        )
        assert cfg.judge_retries == 3

    def test_evaluation_info_errored_fields(self):
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

    def test_evaluation_info_errored_fields_default(self):
        info = EvaluationInfo(
            reward=1.0,
            raw_score=1.0,
            criterion_results=[
                CriterionResult(criterion="c1", weight=1.0, met=True, reasoning="ok"),
            ],
        )
        assert info.errored_criterion_count == 0
        assert info.evaluated_criteria_pct == 100.0

    def test_judge_input_model_copy(self):
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

    def test_judge_input_serialization(self):
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

    def test_verifier_config_max_concurrency_default(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
        )
        assert cfg.max_concurrency is None

    def test_verifier_config_max_concurrency_explicit(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
            max_concurrency=2,
        )
        assert cfg.max_concurrency == 2

    def test_verifier_config_max_concurrency_rejects_zero(self):
        with pytest.raises(ValidationError):
            GraderConfig(
                instructions="test",
                rubric_path="/rubric.json",
                workdir="/workspace",
                trajectory_path="/logs/trajectory.json",
                sandbox_user="sandbox",
                output_dir="/logs/verifier",
                max_concurrency=0,
            )

    def test_verifier_config_max_concurrency_rejects_negative(self):
        with pytest.raises(ValidationError):
            GraderConfig(
                instructions="test",
                rubric_path="/rubric.json",
                workdir="/workspace",
                trajectory_path="/logs/trajectory.json",
                sandbox_user="sandbox",
                output_dir="/logs/verifier",
                max_concurrency=-1,
            )

    def test_verifier_config_mode_sequential_migrates_to_individual(self):
        """mode='sequential' is accepted but migrated to 'individual' with a deprecation warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = GraderConfig(
                instructions="test",
                rubric_path="/rubric.json",
                workdir="/workspace",
                trajectory_path="/logs/trajectory.json",
                sandbox_user="sandbox",
                output_dir="/logs/verifier",
                mode="sequential",
            )
        assert cfg.mode == "individual"
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()

    def test_verifier_config_mode_individual(self):
        cfg = GraderConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir="/workspace",
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir="/logs/verifier",
            mode="individual",
        )
        assert cfg.mode == "individual"
