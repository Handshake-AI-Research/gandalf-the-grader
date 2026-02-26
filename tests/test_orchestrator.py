"""Tests for orchestrator-level functions (resolve_judge_guidance, evaluate_all_criteria)."""

import json
import subprocess
from unittest.mock import patch

import pytest

from gandalf_grader.__main__ import (
    _JUDGE_ENV_ALLOWLIST,
    _compute_and_write_results,
    _is_infra_failure,
    _judge_env_vars,
    evaluate_all_criteria,
    resolve_judge_guidance,
)
from gandalf_grader.config import BatchJudgeInput, CriteriaResult, RubricItem, VerifierConfig


def _make_config(**overrides) -> VerifierConfig:
    """Create a VerifierConfig with sensible defaults for testing."""
    defaults = {
        "instructions": "test",
        "rubric_path": "/rubric.json",
        "workdir": "/workspace",
        "trajectory_path": "/logs/trajectory.json",
        "sandbox_user": "sandbox",
    }
    defaults.update(overrides)
    return VerifierConfig(**defaults)


class TestResolveJudgeGuidance:
    def test_no_path_returns_empty(self, monkeypatch):
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        config = _make_config()
        assert resolve_judge_guidance(config) == ""

    def test_reads_file_from_toml_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        guidance_file = tmp_path / "guidance.md"
        guidance_file.write_text("Use openpyxl for .xlsx files.")
        config = _make_config(judge_guidance_path=str(guidance_file))
        assert resolve_judge_guidance(config) == "Use openpyxl for .xlsx files."

    def test_reads_file_from_env_var(self, tmp_path, monkeypatch):
        guidance_file = tmp_path / "guidance.md"
        guidance_file.write_text("From env var.")
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", str(guidance_file))
        config = _make_config()  # no judge_guidance_path in TOML
        assert resolve_judge_guidance(config) == "From env var."

    def test_toml_takes_precedence_over_env(self, tmp_path, monkeypatch):
        toml_file = tmp_path / "toml_guidance.md"
        toml_file.write_text("From TOML.")
        env_file = tmp_path / "env_guidance.md"
        env_file.write_text("From env.")
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", str(env_file))
        config = _make_config(judge_guidance_path=str(toml_file))
        assert resolve_judge_guidance(config) == "From TOML."

    def test_missing_configured_toml_path_exits(self, monkeypatch):
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        config = _make_config(judge_guidance_path="/nonexistent/guidance.md")
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)

    def test_missing_configured_env_path_exits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", "/nonexistent/guidance.md")
        config = _make_config()
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)

    def test_error_message_mentions_file_path(self, capsys, monkeypatch):
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        config = _make_config(judge_guidance_path="/missing/guidance.md")
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)
        stderr = capsys.readouterr().err
        assert "/missing/guidance.md" in stderr
        assert "judge_guidance_path" in stderr

    def test_error_message_mentions_env_var_source(self, capsys, monkeypatch):
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", "/missing/env_guidance.md")
        config = _make_config()
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)
        stderr = capsys.readouterr().err
        assert "/missing/env_guidance.md" in stderr
        assert "VERIFIER_JUDGE_GUIDANCE_PATH" in stderr


class TestJudgeEnvVars:
    """Tests for the env-var allowlist forwarded to the judge subprocess."""

    def test_only_allowlisted_vars_are_forwarded(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("SECRET_TOKEN", "should-not-leak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")
        result = _judge_env_vars()
        keys = {item.split("=", 1)[0] for item in result}
        assert "LLM_API_KEY" in keys
        assert "PATH" in keys
        assert "SECRET_TOKEN" not in keys
        assert "AWS_SECRET_ACCESS_KEY" not in keys

    def test_empty_values_are_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")
        result = _judge_env_vars()
        keys = {item.split("=", 1)[0] for item in result}
        assert "LLM_API_KEY" not in keys
        assert "LLM_BASE_URL" in keys

    def test_missing_vars_are_silently_skipped(self, monkeypatch):
        for key in _JUDGE_ENV_ALLOWLIST:
            monkeypatch.delenv(key, raising=False)
        assert _judge_env_vars() == []

    def test_all_allowlisted_vars_forwarded_when_present(self, monkeypatch):
        for key in _JUDGE_ENV_ALLOWLIST:
            monkeypatch.setenv(key, f"val-{key}")
        result = _judge_env_vars()
        keys = {item.split("=", 1)[0] for item in result}
        assert keys == set(_JUDGE_ENV_ALLOWLIST)


def _make_batch_input(tmp_path, n=2) -> BatchJudgeInput:
    """Create a BatchJudgeInput with *n* criteria rooted in tmp_path."""
    return BatchJudgeInput(
        model="test-model",
        instructions="do a thing",
        final_output="done",
        criteria=[
            {"index": i, "criteria": f"criterion {i}", "weight": 1.0}
            for i in range(n)
        ],
        workdir=str(tmp_path),
    )


def _run_ok(output_path, content):
    """Return a subprocess.CompletedProcess that succeeds and writes *content* to output_path."""
    import pathlib

    pathlib.Path(output_path).write_text(json.dumps(content))
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


class TestEvaluateAllCriteria:
    """Tests for evaluate_all_criteria IPC contract: dict, list, invalid shapes, failures."""

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_new_dict_shape(self, mock_run, mock_clone, tmp_path):
        """New object format: {verdicts: [...], llm_usage: {...}}."""
        mock_clone.return_value = str(tmp_path)
        output_content = {
            "verdicts": [
                {"index": 0, "passed": True, "reasoning": "ok", "evidence": []},
                {"index": 1, "passed": False, "reasoning": "no", "evidence": []},
            ],
            "llm_usage": {"cost_usd": 0.1, "prompt_tokens": 500},
        }

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        # We need to intercept the output file write. Patch tempfile.mktemp to return a known path.
        output_file = tmp_path / "batch_output.json"
        output_file.write_text(json.dumps(output_content))

        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 2
        assert verdicts[0]["passed"] is True
        assert verdicts[1]["passed"] is False
        assert usage["cost_usd"] == 0.1

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_legacy_array_shape(self, mock_run, mock_clone, tmp_path):
        """Legacy format: bare JSON array of verdicts, no usage info."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        legacy_verdicts = [
            {"index": 0, "passed": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "passed": False, "reasoning": "no", "evidence": []},
        ]
        output_file = tmp_path / "batch_output.json"
        output_file.write_text(json.dumps(legacy_verdicts))

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 2
        assert verdicts[0]["passed"] is True
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_unexpected_json_type_string(self, mock_run, mock_clone, tmp_path):
        """If the output file contains a JSON string, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        output_file = tmp_path / "batch_output.json"
        output_file.write_text(json.dumps("just a string"))

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 2
        assert all(v["passed"] is False for v in verdicts)
        assert "Unexpected JSON type" in verdicts[0]["reasoning"]
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_unexpected_json_type_number(self, mock_run, mock_clone, tmp_path):
        """If the output file contains a JSON number, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        output_file = tmp_path / "batch_output.json"
        output_file.write_text(json.dumps(42))

        judge_input = _make_batch_input(tmp_path, n=1)
        trace_path = str(tmp_path / "trace.txt")

        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 1
        assert verdicts[0]["passed"] is False
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_dict_without_expected_keys(self, mock_run, mock_clone, tmp_path):
        """Dict output missing 'verdicts' key: defaults to empty verdicts list."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        output_file = tmp_path / "batch_output.json"
        output_file.write_text(json.dumps({"unexpected": "shape"}))

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert verdicts == []
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_nonzero_exit_returns_fail_all(self, mock_run, mock_clone, tmp_path):
        """Non-zero exit code from subprocess returns fail-all with empty usage."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="segfault"
        )

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        output_file = tmp_path / "batch_output.json"
        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 2
        assert all(v["passed"] is False for v in verdicts)
        assert "exit 1" in verdicts[0]["reasoning"]
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_timeout_returns_fail_all(self, mock_run, mock_clone, tmp_path):
        """Subprocess timeout returns fail-all with empty usage."""
        mock_clone.return_value = str(tmp_path)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="judge", timeout=300)

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        output_file = tmp_path / "batch_output.json"
        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 2
        assert all(v["passed"] is False for v in verdicts)
        assert "timed out" in verdicts[0]["reasoning"].lower()
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_invalid_json_in_output_file(self, mock_run, mock_clone, tmp_path):
        """Non-JSON content in output file returns fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        output_file = tmp_path / "batch_output.json"
        output_file.write_text("not valid json {{{")

        judge_input = _make_batch_input(tmp_path, n=1)
        trace_path = str(tmp_path / "trace.txt")

        with patch("gandalf_grader.__main__.tempfile.mktemp", return_value=str(output_file)):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 1
        assert verdicts[0]["passed"] is False
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_missing_output_file(self, mock_run, mock_clone, tmp_path):
        """If the judge never wrote the output file, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        # Point to a path that does not exist
        with patch(
            "gandalf_grader.__main__.tempfile.mktemp",
            return_value=str(tmp_path / "nonexistent.json"),
        ):
            verdicts, usage = evaluate_all_criteria(
                judge_input, sandbox_user="sandbox", trace_path=trace_path
            )

        assert len(verdicts) == 2
        assert all(v["passed"] is False for v in verdicts)
        assert usage == {}


class TestIsInfraFailure:
    """Tests for _is_infra_failure detection."""

    def test_all_failed_no_evidence_is_infra(self):
        results = [
            CriteriaResult(criteria="c1", weight=1, passed=False, reasoning="Judge error", evidence=[]),
            CriteriaResult(criteria="c2", weight=1, passed=False, reasoning="Judge error", evidence=[]),
        ]
        assert _is_infra_failure(results) is True

    def test_one_passed_is_not_infra(self):
        results = [
            CriteriaResult(criteria="c1", weight=1, passed=True, reasoning="ok", evidence=[]),
            CriteriaResult(criteria="c2", weight=1, passed=False, reasoning="no", evidence=[]),
        ]
        assert _is_infra_failure(results) is False

    def test_failed_with_evidence_is_not_infra(self):
        results = [
            CriteriaResult(
                criteria="c1", weight=1, passed=False, reasoning="no",
                evidence=["Read file: contents were wrong"],
            ),
        ]
        assert _is_infra_failure(results) is False

    def test_empty_results_is_not_infra(self):
        assert _is_infra_failure([]) is False

    def test_mixed_evidence_is_not_infra(self):
        results = [
            CriteriaResult(criteria="c1", weight=1, passed=False, reasoning="error", evidence=[]),
            CriteriaResult(
                criteria="c2", weight=1, passed=False, reasoning="checked",
                evidence=["Ran command: no output"],
            ),
        ]
        assert _is_infra_failure(results) is False


class TestComputeAndWriteResults:
    """Tests for _compute_and_write_results reward.json / info.json output."""

    def _make_results(self, passed_flags):
        return [
            CriteriaResult(
                criteria=f"criterion {i}",
                weight=1.0,
                passed=p,
                reasoning="ok" if p else "fail",
                evidence=["checked"] if p else [],
            )
            for i, p in enumerate(passed_flags)
        ]

    def _make_rubric(self, n):
        return [RubricItem(criteria=f"criterion {i}", weight=1.0) for i in range(n)]

    def test_writes_both_files_by_default(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        results = self._make_results([True, False])
        rubric = self._make_rubric(2)

        _compute_and_write_results(config, rubric, results, {})

        assert (tmp_path / "reward.json").exists()
        assert (tmp_path / "info.json").exists()
        reward = json.loads((tmp_path / "reward.json").read_text())
        assert reward["score"] == 0.5

    def test_skips_reward_when_write_reward_false(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        results = self._make_results([False, False])
        rubric = self._make_rubric(2)

        _compute_and_write_results(config, rubric, results, {}, write_reward=False)

        assert not (tmp_path / "reward.json").exists()
        assert (tmp_path / "info.json").exists()

    def test_info_json_written_even_on_infra_failure(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        results = [
            CriteriaResult(
                criteria="c1", weight=1.0, passed=False,
                reasoning="Judge execution error: API down", evidence=[],
            ),
        ]
        rubric = self._make_rubric(1)

        _compute_and_write_results(config, rubric, results, {}, write_reward=False)

        assert not (tmp_path / "reward.json").exists()
        info = json.loads((tmp_path / "info.json").read_text())
        assert info["score"] == 0.0
        assert "API down" in info["criteria_results"][0]["reasoning"]
