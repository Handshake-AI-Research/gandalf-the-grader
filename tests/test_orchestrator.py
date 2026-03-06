"""Tests for orchestrator-level functions (resolve_judge_guidance, evaluate_all_criteria)."""

import json
import os
import pathlib
import shutil
import subprocess
from unittest.mock import patch

import pytest

from gandalf_grader.__main__ import (
    _JUDGE_ENV_ALLOWLIST,
    _clone_workspace,
    _judge_env_vars,
    evaluate_all_criteria,
    resolve_judge_guidance,
)
from gandalf_grader.config import BatchJudgeInput, VerifierConfig


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
        assert all(v["passed"] is None for v in verdicts)
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
        assert verdicts[0]["passed"] is None
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
        assert all(v["passed"] is None for v in verdicts)
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
        assert all(v["passed"] is None for v in verdicts)
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
        assert verdicts[0]["passed"] is None
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
        assert all(v["passed"] is None for v in verdicts)
        assert usage == {}


class TestRetryLogic:
    """Tests for retry and hard-fail logic in main()."""

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_sequential_retry_resolves_errored_criterion(
        self, mock_eval, mock_config, mock_rubric, mock_trajectory, mock_guidance, tmp_path
    ):
        """Sequential retry resolves an errored criterion on the second attempt."""
        from gandalf_grader.config import RubricItem

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        mock_config.return_value = VerifierConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir=str(tmp_path),
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir=output_dir,
            judge_retries=1,
            mode="sequential",
        )
        mock_rubric.return_value = [
            RubricItem(criteria="c1", weight=1.0),
            RubricItem(criteria="c2", weight=1.0),
        ]

        # First call: c1 passes, c2 errors. Retry: c2 passes.
        mock_eval.side_effect = [
            {"passed": True, "reasoning": "ok", "evidence": ["e1"]},
            {"passed": None, "reasoning": "timeout"},
            # retry for c2
            {"passed": True, "reasoning": "ok on retry", "evidence": ["e2"]},
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert info["criteria_results"][0]["passed"] is True
        assert info["criteria_results"][1]["passed"] is True
        assert info["errored_criteria_count"] == 0

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        assert reward["score"] == 1.0

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_all_criteria")
    def test_batch_retry_resolves_errored_criteria(
        self, mock_eval_all, mock_config, mock_rubric, mock_trajectory, mock_guidance, tmp_path
    ):
        """Batch retry resolves errored criteria with correct re-indexing."""
        from gandalf_grader.config import RubricItem

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        mock_config.return_value = VerifierConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir=str(tmp_path),
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir=output_dir,
            judge_retries=1,
            mode="batch",
        )
        mock_rubric.return_value = [
            RubricItem(criteria="c1", weight=1.0),
            RubricItem(criteria="c2", weight=1.0),
            RubricItem(criteria="c3", weight=1.0),
        ]

        # Initial batch: c1 passes, c2 errors, c3 errors
        initial_verdicts = [
            {"index": 0, "passed": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "passed": None, "reasoning": "timeout", "evidence": []},
            {"index": 2, "passed": None, "reasoning": "crash", "evidence": []},
        ]
        # Retry batch (re-indexed 0,1 -> original 1,2): both pass
        retry_verdicts = [
            {"index": 0, "passed": True, "reasoning": "ok retry", "evidence": []},
            {"index": 1, "passed": True, "reasoning": "ok retry 2", "evidence": []},
        ]
        mock_eval_all.side_effect = [
            (initial_verdicts, {"cost_usd": 0.1}),
            (retry_verdicts, {"cost_usd": 0.05}),
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert all(r["passed"] is True for r in info["criteria_results"])
        assert info["errored_criteria_count"] == 0

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        assert reward["score"] == 1.0

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_judge_retries_zero_disables_retry(
        self, mock_eval, mock_config, mock_rubric, mock_trajectory, mock_guidance, tmp_path
    ):
        """judge_retries=0 skips retry loop entirely — errors cause hard fail."""
        from gandalf_grader.config import RubricItem

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        mock_config.return_value = VerifierConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir=str(tmp_path),
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir=output_dir,
            judge_retries=0,
            mode="sequential",
        )
        mock_rubric.return_value = [RubricItem(criteria="c1", weight=1.0)]
        mock_eval.return_value = {"passed": None, "reasoning": "timeout"}

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        # info.json MUST be written even on hard fail
        assert (tmp_path / "output" / "info.json").exists()
        # reward.json must NOT be written
        assert not (tmp_path / "output" / "reward.json").exists()

        # Only 1 call — no retry
        assert mock_eval.call_count == 1

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_hard_fail_writes_info_not_reward(
        self, mock_eval, mock_config, mock_rubric, mock_trajectory, mock_guidance, tmp_path
    ):
        """Persistent errors: info.json written, reward.json NOT written, exit 1."""
        from gandalf_grader.config import RubricItem

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        mock_config.return_value = VerifierConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir=str(tmp_path),
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir=output_dir,
            judge_retries=1,
            mode="sequential",
        )
        mock_rubric.return_value = [RubricItem(criteria="c1", weight=1.0)]
        # Both initial and retry fail
        mock_eval.return_value = {"passed": None, "reasoning": "always fails"}

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert info["criteria_results"][0]["passed"] is None
        assert info["errored_criteria_count"] == 1
        assert not (tmp_path / "output" / "reward.json").exists()

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_all_resolved_after_retry_includes_errored_count_in_reward(
        self, mock_eval, mock_config, mock_rubric, mock_trajectory, mock_guidance, tmp_path
    ):
        """After retry resolves all errors: reward.json includes errored_criteria_count."""
        from gandalf_grader.config import RubricItem

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        mock_config.return_value = VerifierConfig(
            instructions="test",
            rubric_path="/rubric.json",
            workdir=str(tmp_path),
            trajectory_path="/logs/trajectory.json",
            sandbox_user="sandbox",
            output_dir=output_dir,
            judge_retries=1,
            mode="sequential",
        )
        mock_rubric.return_value = [
            RubricItem(criteria="c1", weight=1.0),
            RubricItem(criteria="c2", weight=1.0),
        ]

        # c1 passes, c2 errors initially, succeeds on retry (passes as False = legit fail)
        mock_eval.side_effect = [
            {"passed": True, "reasoning": "ok", "evidence": []},
            {"passed": None, "reasoning": "timeout"},
            {"passed": False, "reasoning": "genuinely failed", "evidence": []},
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        assert reward["score"] == 0.5

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert info["errored_criteria_count"] == 0


class TestCloneWorkspace:
    """Tests for _clone_workspace resilience to unreadable files."""

    def test_readable_files_are_cloned(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("hello")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.txt").write_text("world")

        clone_dir = _clone_workspace(str(workspace))
        try:
            assert (pathlib.Path(clone_dir) / "file.txt").read_text() == "hello"
            assert (pathlib.Path(clone_dir) / "subdir" / "nested.txt").read_text() == "world"
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_unreadable_files_are_skipped_not_fatal(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "readable.txt").write_text("ok")

        restricted = workspace / "restricted.txt"
        restricted.write_text("secret")
        restricted.chmod(0o000)

        try:
            clone_dir = _clone_workspace(str(workspace))
            cloned = pathlib.Path(clone_dir)
            assert (cloned / "readable.txt").read_text() == "ok"
            assert not (cloned / "restricted.txt").exists()
        finally:
            restricted.chmod(0o644)
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_skipped_files_are_logged(self, tmp_path, capsys):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        restricted = workspace / "noperm.txt"
        restricted.write_text("x")
        restricted.chmod(0o000)

        try:
            clone_dir = _clone_workspace(str(workspace))
            stderr = capsys.readouterr().err
            assert "skipped 1 unreadable path(s)" in stderr
            assert "noperm.txt" in stderr
        finally:
            restricted.chmod(0o644)
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_clone_is_group_writable(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("data")

        clone_dir = _clone_workspace(str(workspace))
        try:
            cloned = pathlib.Path(clone_dir)
            assert os.stat(clone_dir).st_mode & 0o070 == 0o070
            fstat = os.stat(cloned / "file.txt")
            assert fstat.st_mode & 0o060 == 0o060
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)
