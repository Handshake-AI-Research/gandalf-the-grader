"""Tests for orchestrator-level functions (resolve_judge_guidance, evaluate_all_criteria)."""

import json
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

from gandalf_grader.__main__ import (
    _JUDGE_ENV_ALLOWLIST,
    _judge_env_vars,
    _run_with_live_trace,
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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_new_dict_shape(self, mock_live_trace, mock_clone, tmp_path):
        """New object format: {verdicts: [...], llm_usage: {...}}."""
        mock_clone.return_value = str(tmp_path)
        output_content = {
            "verdicts": [
                {"index": 0, "passed": True, "reasoning": "ok", "evidence": []},
                {"index": 1, "passed": False, "reasoning": "no", "evidence": []},
            ],
            "llm_usage": {"cost_usd": 0.1, "prompt_tokens": 500},
        }

        mock_live_trace.return_value = (0, "", "", False)
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
        called_cmd = mock_live_trace.call_args.kwargs["cmd"]
        assert "PYTHONUNBUFFERED=1" in called_cmd

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_legacy_array_shape(self, mock_live_trace, mock_clone, tmp_path):
        """Legacy format: bare JSON array of verdicts, no usage info."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (0, "", "", False)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_unexpected_json_type_string(self, mock_live_trace, mock_clone, tmp_path):
        """If the output file contains a JSON string, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (0, "", "", False)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_unexpected_json_type_number(self, mock_live_trace, mock_clone, tmp_path):
        """If the output file contains a JSON number, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (0, "", "", False)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_dict_without_expected_keys(self, mock_live_trace, mock_clone, tmp_path):
        """Dict output missing 'verdicts' key: defaults to empty verdicts list."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (0, "", "", False)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_nonzero_exit_returns_fail_all(self, mock_live_trace, mock_clone, tmp_path):
        """Non-zero exit code from subprocess returns fail-all with empty usage."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (1, "", "segfault", False)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_timeout_returns_fail_all(self, mock_live_trace, mock_clone, tmp_path):
        """Subprocess timeout returns fail-all with empty usage."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (-1, "", "", True)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_invalid_json_in_output_file(self, mock_live_trace, mock_clone, tmp_path):
        """Non-JSON content in output file returns fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (0, "", "", False)

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
    @patch("gandalf_grader.__main__._run_with_live_trace")
    def test_missing_output_file(self, mock_live_trace, mock_clone, tmp_path):
        """If the judge never wrote the output file, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_live_trace.return_value = (0, "", "", False)

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


class TestLiveTraceRunner:
    def test_run_with_live_trace_captures_stdout_stderr(self, tmp_path):
        trace_path = tmp_path / "live_trace.txt"
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys,time; "
                "print('out-1', flush=True); "
                "print('err-1', file=sys.stderr, flush=True); "
                "time.sleep(0.05); "
                "print('out-2', flush=True)"
            ),
        ]

        returncode, stdout, stderr, timed_out = _run_with_live_trace(
            cmd=cmd,
            cwd=str(tmp_path),
            trace_path=str(trace_path),
            timeout=5,
        )

        assert returncode == 0
        assert timed_out is False
        assert "out-1" in stdout
        assert "out-2" in stdout
        assert "err-1" in stderr

        trace = trace_path.read_text()
        assert "exit_code: running" in trace
        assert "[stdout] out-1" in trace
        assert "[stderr] err-1" in trace
        assert "exit_code: 0" in trace

    def test_run_with_live_trace_writes_before_process_exit(self, tmp_path):
        trace_path = tmp_path / "live_trace_streaming.txt"
        cmd = [
            sys.executable,
            "-c",
            (
                "import time; "
                "print('first-line', flush=True); "
                "time.sleep(0.5); "
                "print('second-line', flush=True)"
            ),
        ]

        result_holder: dict[str, tuple[int, str, str, bool]] = {}

        def _runner() -> None:
            result_holder["result"] = _run_with_live_trace(
                cmd=cmd,
                cwd=str(tmp_path),
                trace_path=str(trace_path),
                timeout=5,
            )

        t = threading.Thread(target=_runner)
        t.start()

        saw_first_line = False
        deadline = time.time() + 2
        while time.time() < deadline:
            if trace_path.exists():
                trace = trace_path.read_text()
                if "[stdout] first-line" in trace:
                    saw_first_line = True
                    break
            time.sleep(0.02)

        assert saw_first_line is True
        t.join(timeout=5)
        assert "result" in result_holder

    def test_run_with_live_trace_writes_partial_chunk_before_newline(self, tmp_path):
        trace_path = tmp_path / "live_trace_partial.txt"
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys,time; "
                "sys.stdout.write('partial'); sys.stdout.flush(); "
                "time.sleep(0.5); "
                "sys.stdout.write('-done\\n'); sys.stdout.flush()"
            ),
        ]

        result_holder: dict[str, tuple[int, str, str, bool]] = {}

        def _runner() -> None:
            result_holder["result"] = _run_with_live_trace(
                cmd=cmd,
                cwd=str(tmp_path),
                trace_path=str(trace_path),
                timeout=5,
            )

        t = threading.Thread(target=_runner)
        t.start()

        saw_partial = False
        deadline = time.time() + 2
        while time.time() < deadline:
            if trace_path.exists():
                trace = trace_path.read_text()
                if "partial" in trace:
                    saw_partial = True
                    break
            time.sleep(0.02)

        assert saw_partial is True
        t.join(timeout=5)
        assert "result" in result_holder
