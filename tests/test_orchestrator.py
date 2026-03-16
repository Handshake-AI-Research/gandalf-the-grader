"""Tests for orchestrator-level functions (resolve_judge_guidance, evaluate_all_criteria)."""

import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from gandalf_grader.__main__ import (
    _JUDGE_ENV_ALLOWLIST,
    _clone_workspace,
    _judge_env_vars,
    _write_info,
    evaluate_all_criteria,
    resolve_judge_guidance,
)
from gandalf_grader.config import (
    BatchCriterion,
    BatchJudgeInput,
    CriteriaResult,
    RubricItem,
    VerifierConfig,
)


def _make_config(**overrides: Any) -> VerifierConfig:
    """Create a VerifierConfig with sensible defaults for testing."""
    defaults: dict[str, Any] = {
        "instructions": "test",
        "rubric_path": "/rubric.json",
        "workdir": "/workspace",
        "trajectory_path": "/logs/trajectory.json",
        "sandbox_user": "sandbox",
    }
    defaults.update(overrides)
    return VerifierConfig(**defaults)


class TestResolveJudgeGuidance:
    def test_no_path_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        config = _make_config()
        assert resolve_judge_guidance(config) == ""

    def test_reads_file_from_toml_path(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        guidance_file = tmp_path / "guidance.md"
        guidance_file.write_text("Use openpyxl for .xlsx files.")
        config = _make_config(judge_guidance_path=str(guidance_file))
        assert resolve_judge_guidance(config) == "Use openpyxl for .xlsx files."

    def test_reads_file_from_env_var(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        guidance_file = tmp_path / "guidance.md"
        guidance_file.write_text("From env var.")
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", str(guidance_file))
        config = _make_config()  # no judge_guidance_path in TOML
        assert resolve_judge_guidance(config) == "From env var."

    def test_toml_takes_precedence_over_env(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        toml_file = tmp_path / "toml_guidance.md"
        toml_file.write_text("From TOML.")
        env_file = tmp_path / "env_guidance.md"
        env_file.write_text("From env.")
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", str(env_file))
        config = _make_config(judge_guidance_path=str(toml_file))
        assert resolve_judge_guidance(config) == "From TOML."

    def test_missing_configured_toml_path_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        config = _make_config(judge_guidance_path="/nonexistent/guidance.md")
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)

    def test_missing_configured_env_path_exits(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", "/nonexistent/guidance.md")
        config = _make_config()
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)

    def test_error_message_mentions_file_path(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VERIFIER_JUDGE_GUIDANCE_PATH", raising=False)
        config = _make_config(judge_guidance_path="/missing/guidance.md")
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)
        stderr = capsys.readouterr().err
        assert "/missing/guidance.md" in stderr
        assert "judge_guidance_path" in stderr

    def test_error_message_mentions_env_var_source(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VERIFIER_JUDGE_GUIDANCE_PATH", "/missing/env_guidance.md")
        config = _make_config()
        with pytest.raises(SystemExit):
            resolve_judge_guidance(config)
        stderr = capsys.readouterr().err
        assert "/missing/env_guidance.md" in stderr
        assert "VERIFIER_JUDGE_GUIDANCE_PATH" in stderr


class TestJudgeEnvVars:
    """Tests for the env-var allowlist forwarded to the judge subprocess."""

    def test_only_allowlisted_vars_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_empty_values_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")
        result = _judge_env_vars()
        keys = {item.split("=", 1)[0] for item in result}
        assert "LLM_API_KEY" not in keys
        assert "LLM_BASE_URL" in keys

    def test_missing_vars_are_silently_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in _JUDGE_ENV_ALLOWLIST:
            monkeypatch.delenv(key, raising=False)
        assert _judge_env_vars() == []

    def test_all_allowlisted_vars_forwarded_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in _JUDGE_ENV_ALLOWLIST:
            monkeypatch.setenv(key, f"val-{key}")
        result = _judge_env_vars()
        keys = {item.split("=", 1)[0] for item in result}
        assert keys == set(_JUDGE_ENV_ALLOWLIST)


def _make_batch_input(tmp_path: pathlib.Path, n: int = 2) -> BatchJudgeInput:
    """Create a BatchJudgeInput with *n* criteria rooted in tmp_path."""
    return BatchJudgeInput(
        model="test-model",
        instructions="do a thing",
        final_output="done",
        criteria=[BatchCriterion(index=i, criteria=f"criterion {i}") for i in range(n)],
        workdir=str(tmp_path),
    )


def _run_ok(output_path: str, content: Any) -> subprocess.CompletedProcess[str]:
    """Return a subprocess.CompletedProcess that succeeds and writes *content* to output_path."""
    pathlib.Path(output_path).write_text(json.dumps(content))
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _make_run_writing(content: Any) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a mock_run side_effect that writes *content* to the --output path in the cmd."""

    def _side_effect(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        for i, arg in enumerate(cmd):
            if arg == "--output" and i + 1 < len(cmd):
                pathlib.Path(cmd[i + 1]).write_text(json.dumps(content))
                break
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    return _side_effect


class TestEvaluateAllCriteria:
    """Tests for evaluate_all_criteria IPC contract: dict, list, invalid shapes, failures."""

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_new_dict_shape(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """New object format: {verdicts: [...], llm_usage: {...}}."""
        mock_clone.return_value = str(tmp_path)
        output_content = {
            "verdicts": [
                {"index": 0, "met": True, "reasoning": "ok", "evidence": []},
                {"index": 1, "met": False, "reasoning": "no", "evidence": []},
            ],
            "llm_usage": {"cost_usd": 0.1, "prompt_tokens": 500},
        }

        mock_run.side_effect = _make_run_writing(output_content)
        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 2
        assert verdicts[0]["met"] is True
        assert verdicts[1]["met"] is False
        assert usage["cost_usd"] == 0.1

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_legacy_array_shape(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Legacy format: bare JSON array of verdicts, no usage info."""
        mock_clone.return_value = str(tmp_path)

        legacy_verdicts = [
            {"index": 0, "met": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "met": False, "reasoning": "no", "evidence": []},
        ]
        mock_run.side_effect = _make_run_writing(legacy_verdicts)

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 2
        assert verdicts[0]["met"] is True
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_unexpected_json_type_string(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """If the output file contains a JSON string, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_run.side_effect = _make_run_writing("just a string")

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 2
        assert all(v["met"] is None for v in verdicts)
        assert "Unexpected JSON type" in verdicts[0]["reasoning"]
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_unexpected_json_type_number(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """If the output file contains a JSON number, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        mock_run.side_effect = _make_run_writing(42)

        judge_input = _make_batch_input(tmp_path, n=1)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 1
        assert verdicts[0]["met"] is None
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_dict_without_expected_keys(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Dict output missing 'verdicts' key: defaults to empty verdicts list."""
        mock_clone.return_value = str(tmp_path)
        mock_run.side_effect = _make_run_writing({"unexpected": "shape"})

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert verdicts == []
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_nonzero_exit_returns_fail_all(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Non-zero exit code from subprocess returns fail-all with empty usage."""
        mock_clone.return_value = str(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="segfault")

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 2
        assert all(v["met"] is None for v in verdicts)
        assert "exit 1" in verdicts[0]["reasoning"]
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_timeout_returns_fail_all(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Subprocess timeout returns fail-all with empty usage."""
        mock_clone.return_value = str(tmp_path)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="judge", timeout=300)

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 2
        assert all(v["met"] is None for v in verdicts)
        assert "timed out" in verdicts[0]["reasoning"].lower()
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_invalid_json_in_output_file(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Non-JSON content in output file returns fail-all."""
        mock_clone.return_value = str(tmp_path)

        def _write_invalid(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            for i, arg in enumerate(cmd):
                if arg == "--output" and i + 1 < len(cmd):
                    pathlib.Path(cmd[i + 1]).write_text("not valid json {{{")
                    break
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        mock_run.side_effect = _write_invalid

        judge_input = _make_batch_input(tmp_path, n=1)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 1
        assert verdicts[0]["met"] is None
        assert usage == {}

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_empty_output_file(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """If the judge wrote nothing to the output file, return fail-all."""
        mock_clone.return_value = str(tmp_path)
        # mock_run does not write to the output file — it stays empty (pre-created by verifier)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        judge_input = _make_batch_input(tmp_path, n=2)
        trace_path = str(tmp_path / "trace.txt")

        verdicts, usage = evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=trace_path)

        assert len(verdicts) == 2
        assert all(v["met"] is None for v in verdicts)
        assert usage == {}


def _cr(weight: float, met: bool | None) -> CriteriaResult:
    """Helper to build a CriteriaResult for scoring tests."""
    return CriteriaResult(
        criteria="test",
        weight=weight,
        met=met,
        reasoning="test",
    )


class TestScoring:
    """Tests for _write_info scoring: raw_score, reward, and bounds.

    Each test asserts both raw_score and reward together so the full
    scoring pipeline is verified in one place per scenario.
    """

    def _info(self, results: list[CriteriaResult], tmp_path: pathlib.Path) -> dict[str, Any]:
        """Run _write_info and return parsed info.json."""
        config = _make_config(output_dir=str(tmp_path))
        errored = sum(1 for r in results if r.met is None)
        _write_info(config, results, {}, errored)
        with open(tmp_path / "info.json") as f:
            result: dict[str, Any] = json.load(f)
            return result

    # -- core scenarios (raw_score + reward together) --

    def test_all_positive_all_met(self, tmp_path: pathlib.Path) -> None:
        """weights=[2,3], met=[T,T] → raw=5, reward=1.0, min=0, max=5."""
        info = self._info([_cr(2.0, True), _cr(3.0, True)], tmp_path)
        assert info["raw_score"] == 5.0
        assert info["reward"] == 1.0
        assert info["minimum_score"] == 0.0
        assert info["maximum_score"] == 5.0

    def test_all_positive_partial_met(self, tmp_path: pathlib.Path) -> None:
        """weights=[2,3], met=[T,F] → raw=2, reward=0.4."""
        info = self._info([_cr(2.0, True), _cr(3.0, False)], tmp_path)
        assert info["raw_score"] == 2.0
        assert info["reward"] == 0.4

    def test_all_positive_none_met(self, tmp_path: pathlib.Path) -> None:
        """weights=[2,3], met=[F,F] → raw=0, reward=0.0."""
        info = self._info([_cr(2.0, False), _cr(3.0, False)], tmp_path)
        assert info["raw_score"] == 0.0
        assert info["reward"] == 0.0

    def test_mixed_negative_penalty_applied(self, tmp_path: pathlib.Path) -> None:
        """weights=[3,-1], met=[T,T] → raw=2, reward=2/3."""
        info = self._info([_cr(3.0, True), _cr(-1.0, True)], tmp_path)
        assert info["raw_score"] == 2.0
        assert info["reward"] == 0.6667

    def test_mixed_negative_drives_below_zero_clipped(self, tmp_path: pathlib.Path) -> None:
        """weights=[1,-3], met=[F,T] → raw=-3, reward=0.0 (clip lower bound)."""
        info = self._info([_cr(1.0, False), _cr(-3.0, True)], tmp_path)
        assert info["raw_score"] == -3.0
        assert info["reward"] == 0.0

    def test_negative_not_met_no_penalty(self, tmp_path: pathlib.Path) -> None:
        """weights=[3,-1], met=[T,F] → raw=3, reward=1.0."""
        info = self._info([_cr(3.0, True), _cr(-1.0, False)], tmp_path)
        assert info["raw_score"] == 3.0
        assert info["reward"] == 1.0

    def test_all_negative_denominator_zero(self, tmp_path: pathlib.Path) -> None:
        """weights=[-2,-3], met=[T,T] → raw=-5, reward=0.0 (no divide-by-zero)."""
        info = self._info([_cr(-2.0, True), _cr(-3.0, True)], tmp_path)
        assert info["raw_score"] == -5.0
        assert info["reward"] == 0.0

    def test_empty_rubric(self, tmp_path: pathlib.Path) -> None:
        """No criteria → raw=0, reward=0."""
        info = self._info([], tmp_path)
        assert info["raw_score"] == 0.0
        assert info["reward"] == 0.0

    def test_errored_positive_criterion(self, tmp_path: pathlib.Path) -> None:
        """weights=[3,2], met=[T,None] → raw=3, reward=3/5=0.6."""
        info = self._info([_cr(3.0, True), _cr(2.0, None)], tmp_path)
        assert info["raw_score"] == 3.0
        assert info["reward"] == 0.6

    def test_errored_negative_criterion(self, tmp_path: pathlib.Path) -> None:
        """weights=[3,-2], met=[T,None] → raw=3, reward=3/3=1.0."""
        info = self._info([_cr(3.0, True), _cr(-2.0, None)], tmp_path)
        assert info["raw_score"] == 3.0
        assert info["reward"] == 1.0

    # -- info.json shape --

    def test_info_json_contains_reward_and_raw_score(self, tmp_path: pathlib.Path) -> None:
        """info.json must contain both reward and raw_score fields."""
        info = self._info([_cr(2.0, True), _cr(3.0, False)], tmp_path)
        assert "reward" in info
        assert "raw_score" in info
        assert isinstance(info["reward"], float)
        assert isinstance(info["raw_score"], (int, float))

    def test_info_json_no_legacy_score_field(self, tmp_path: pathlib.Path) -> None:
        """The old 'score' key must not appear in info.json."""
        info = self._info([_cr(1.0, True)], tmp_path)
        assert "score" not in info

    def test_info_json_contains_minimum_and_maximum_score(self, tmp_path: pathlib.Path) -> None:
        info = self._info([_cr(10.0, True), _cr(5.0, False), _cr(-3.0, True)], tmp_path)
        assert info["minimum_score"] == -3.0
        assert info["maximum_score"] == 15.0


class TestOutputFilePermissions:
    """Ensure the judge output file is pre-created with world-writable permissions.

    Regression: the old code used tempfile.mktemp() which does NOT create the
    file, requiring sandbox_user to create it in /tmp.  On systems where /tmp
    is not world-writable, this caused a PermissionError.  The fix pre-creates
    the file and chmods it 0o666 so sandbox_user only needs to *write* to an
    existing file, not *create* one in a restricted directory.
    """

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_output_file_exists_before_subprocess(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Output file must be pre-created so sandbox_user can write it without /tmp access."""
        mock_clone.return_value = str(tmp_path)
        captured_cmd: dict[str, Any] = {}

        def _capture(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            output_path = cmd[cmd.index("--output") + 1]
            captured_cmd["output_path"] = output_path
            captured_cmd["existed_before_run"] = pathlib.Path(output_path).exists()
            # Simulate sandbox_user writing to the pre-created file
            pathlib.Path(output_path).write_text(json.dumps({"verdicts": [], "llm_usage": {}}))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        mock_run.side_effect = _capture
        judge_input = _make_batch_input(tmp_path, n=1)

        evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=str(tmp_path / "trace.txt"))

        assert "output_path" in captured_cmd, "subprocess was not called"
        assert captured_cmd.get("existed_before_run"), (
            "Output file was NOT pre-created before subprocess.run — "
            "sandbox_user would need to create it in /tmp (may not be world-writable)"
        )

    @patch("gandalf_grader.__main__._clone_workspace")
    @patch("gandalf_grader.__main__.subprocess.run")
    def test_output_file_is_world_writable(self, mock_run: Any, mock_clone: Any, tmp_path: pathlib.Path) -> None:
        """Pre-created output file must have world-write so sandbox_user can overwrite it.

        This test fails on the pre-fix code (tempfile.mktemp → file never created)
        and passes with the fix (NamedTemporaryFile + chmod 0o666).
        """
        mock_clone.return_value = str(tmp_path)
        captured: dict[str, Any] = {}

        def _capture_and_check_permissions(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            output_path = cmd[cmd.index("--output") + 1]
            captured["path"] = output_path
            captured["exists"] = pathlib.Path(output_path).exists()
            if captured["exists"]:
                captured["mode"] = os.stat(output_path).st_mode
            pathlib.Path(output_path).write_text(json.dumps({"verdicts": [], "llm_usage": {}}))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        mock_run.side_effect = _capture_and_check_permissions
        judge_input = _make_batch_input(tmp_path, n=1)
        evaluate_all_criteria(judge_input, sandbox_user="sandbox", trace_path=str(tmp_path / "trace.txt"))

        assert captured.get("exists"), (
            "Output file was NOT pre-created before subprocess.run — "
            "sandbox_user would need to create it in /tmp (may not be world-writable)"
        )
        mode = captured.get("mode", 0)
        assert mode & 0o002, (
            f"Output file missing world-write bit (mode={oct(mode)}) — "
            "sandbox_user cannot write to it without /tmp create access"
        )


class TestRetryLogic:
    """Tests for retry and hard-fail logic in main()."""

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_sequential_retry_resolves_errored_criterion(
        self,
        mock_eval: Any,
        mock_config: Any,
        mock_rubric: Any,
        mock_trajectory: Any,
        mock_guidance: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        """Sequential retry resolves an errored criterion on the second attempt."""

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
            {"met": True, "reasoning": "ok", "evidence": ["e1"]},
            {"met": None, "reasoning": "timeout"},
            # retry for c2
            {"met": True, "reasoning": "ok on retry", "evidence": ["e2"]},
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert info["criteria_results"][0]["met"] is True
        assert info["criteria_results"][1]["met"] is True
        assert info["errored_criteria_count"] == 0

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        assert reward["reward"] == 1.0  # all met: 2.0 / 2.0 = 1.0

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_all_criteria")
    def test_batch_retry_resolves_errored_criteria(
        self,
        mock_eval_all: Any,
        mock_config: Any,
        mock_rubric: Any,
        mock_trajectory: Any,
        mock_guidance: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        """Batch retry resolves errored criteria with correct re-indexing."""

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

        initial_verdicts = [
            {"index": 0, "met": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "met": None, "reasoning": "timeout", "evidence": []},
            {"index": 2, "met": None, "reasoning": "crash", "evidence": []},
        ]
        retry_verdicts = [
            {"index": 0, "met": True, "reasoning": "ok retry", "evidence": []},
            {"index": 1, "met": True, "reasoning": "ok retry 2", "evidence": []},
        ]
        mock_eval_all.side_effect = [
            (initial_verdicts, {"cost_usd": 0.1}),
            (retry_verdicts, {"cost_usd": 0.05}),
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert all(r["met"] is True for r in info["criteria_results"])
        assert info["errored_criteria_count"] == 0

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        assert reward["reward"] == 1.0  # all met: 3.0 / 3.0 = 1.0

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_judge_retries_zero_disables_retry(
        self,
        mock_eval: Any,
        mock_config: Any,
        mock_rubric: Any,
        mock_trajectory: Any,
        mock_guidance: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        """judge_retries=0 skips retry loop entirely — errors cause hard fail."""

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
        mock_eval.return_value = {"met": None, "reasoning": "timeout"}

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        assert (tmp_path / "output" / "info.json").exists()
        assert not (tmp_path / "output" / "reward.json").exists()
        assert mock_eval.call_count == 1

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_hard_fail_writes_info_not_reward(
        self,
        mock_eval: Any,
        mock_config: Any,
        mock_rubric: Any,
        mock_trajectory: Any,
        mock_guidance: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        """Persistent errors: info.json written, reward.json NOT written, exit 1."""

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
        mock_eval.return_value = {"met": None, "reasoning": "always fails"}

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert info["criteria_results"][0]["met"] is None
        assert info["errored_criteria_count"] == 1
        assert not (tmp_path / "output" / "reward.json").exists()

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_all_resolved_after_retry(
        self,
        mock_eval: Any,
        mock_config: Any,
        mock_rubric: Any,
        mock_trajectory: Any,
        mock_guidance: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        """After retry resolves all errors: reward.json written with correct reward."""

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

        mock_eval.side_effect = [
            {"met": True, "reasoning": "ok", "evidence": []},
            {"met": None, "reasoning": "timeout"},
            {"met": False, "reasoning": "genuinely failed", "evidence": []},
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        assert reward["reward"] == 0.5  # c1 met, c2 not: 1.0 / 2.0 = 0.5

        info = json.loads((tmp_path / "output" / "info.json").read_text())
        assert info["errored_criteria_count"] == 0

    @patch("gandalf_grader.__main__.resolve_judge_guidance", return_value="")
    @patch("gandalf_grader.__main__.load_trajectory_final_output", return_value="done")
    @patch("gandalf_grader.__main__.load_rubric")
    @patch("gandalf_grader.__main__.load_config")
    @patch("gandalf_grader.__main__.evaluate_criteria")
    def test_reward_json_with_negative_weights(
        self,
        mock_eval: Any,
        mock_config: Any,
        mock_rubric: Any,
        mock_trajectory: Any,
        mock_guidance: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        """reward.json must contain the [0,1] reward, not the raw score,
        when negative-weight criteria are present."""

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
        mock_rubric.return_value = [
            RubricItem(criteria="correct output", weight=3.0),
            RubricItem(criteria="used hardcoded values", weight=-1.0),
        ]

        # Both criteria met: raw = 3 + (-1) = 2, reward = 2/3 ≈ 0.6667
        mock_eval.side_effect = [
            {"met": True, "reasoning": "ok", "evidence": []},
            {"met": True, "reasoning": "hardcoded detected", "evidence": []},
        ]

        from gandalf_grader.__main__ import main

        with patch("sys.argv", ["prog", "--config", "dummy.toml"]):
            main()

        reward = json.loads((tmp_path / "output" / "reward.json").read_text())
        info = json.loads((tmp_path / "output" / "info.json").read_text())

        assert reward["reward"] == 0.6667
        assert info["raw_score"] == 2.0
        assert info["reward"] == 0.6667
        assert reward["reward"] == info["reward"]


class TestCloneWorkspace:
    """Tests for _clone_workspace resilience to unreadable files."""

    def test_readable_files_are_cloned(self, tmp_path: pathlib.Path) -> None:
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

    def test_unreadable_files_are_skipped_not_fatal(self, tmp_path: pathlib.Path) -> None:
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

    def test_skipped_files_are_logged(self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
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

    def test_unreadable_directory_is_skipped_not_fatal(self, tmp_path: pathlib.Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "readable.txt").write_text("ok")

        # Create a directory tree and make the parent unreadable
        restricted_dir = workspace / ".tool_cache"
        restricted_dir.mkdir()
        (restricted_dir / "data.bin").write_text("cached")
        restricted_dir.chmod(0o000)

        try:
            clone_dir = _clone_workspace(str(workspace))
            cloned = pathlib.Path(clone_dir)
            assert (cloned / "readable.txt").read_text() == "ok"
            # The restricted directory's contents should not appear
            assert not (cloned / ".tool_cache" / "data.bin").exists()
        finally:
            restricted_dir.chmod(0o755)
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_unreadable_directory_is_logged(self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        restricted_dir = workspace / ".cache"
        restricted_dir.mkdir()
        restricted_dir.chmod(0o000)

        try:
            clone_dir = _clone_workspace(str(workspace))
            stderr = capsys.readouterr().err
            assert "skipped 1 unreadable path(s)" in stderr
            assert ".cache" in stderr
        finally:
            restricted_dir.chmod(0o755)
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_clone_is_group_writable(self, tmp_path: pathlib.Path) -> None:
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

    def test_clone_is_world_accessible(self, tmp_path: pathlib.Path) -> None:
        """Clone dir must have world execute+write so sandbox_user can use it.

        Regression: shutil.copytree preserved the source workspace permissions
        (typically world-executable) on the root clone dir.  The new os.walk
        implementation creates clone_dir via mkdtemp (mode 0o700) and must
        explicitly grant world bits — otherwise sandbox_user (not in the
        verifier's group) cannot traverse or write to the workspace.

        This test fails on the pre-fix code (|0o070 → 0o770, no world bits)
        and passes with the fix (|0o077 → 0o777).
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("hello")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.txt").write_text("world")

        clone_dir = _clone_workspace(str(workspace))
        try:
            clone = pathlib.Path(clone_dir)

            # Root clone dir: world execute (traverse) + write (create files inside it)
            root_mode = clone.stat().st_mode
            assert root_mode & 0o001, (
                "clone root missing world execute — sandbox_user cannot traverse it "
                "(regression: os.walk+mkdtemp loses the world-execute bit that "
                "shutil.copytree preserved from the source workspace)"
            )
            assert root_mode & 0o002, "clone root missing world write — sandbox_user cannot create files in it"

            # Subdirectories must also have world execute+write
            sub_mode = (clone / "subdir").stat().st_mode
            assert sub_mode & 0o001, "subdir missing world execute"
            assert sub_mode & 0o002, "subdir missing world write"

            # Files must have world read so sandbox_user can inspect them
            file_mode = (clone / "file.txt").stat().st_mode
            assert file_mode & 0o004, "file missing world read"
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_executable_bits_are_preserved(self, tmp_path: pathlib.Path) -> None:
        """Cloned files must retain execute bits so scripts/binaries remain runnable."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        script = workspace / "run.sh"
        script.write_text("#!/bin/sh\necho hi")
        script.chmod(0o755)
        data = workspace / "data.txt"
        data.write_text("plain")

        clone_dir = _clone_workspace(str(workspace))
        try:
            cloned = pathlib.Path(clone_dir)
            cloned_script_mode = (cloned / "run.sh").stat().st_mode
            assert cloned_script_mode & 0o111, (
                f"Executable bits lost on cloned script (mode={oct(cloned_script_mode)}) — "
                "judge runs that execute workspace scripts will break"
            )
            # Non-executable file should NOT gain execute bits
            cloned_data_mode = (cloned / "data.txt").stat().st_mode
            assert not (cloned_data_mode & 0o111), (
                f"Non-executable file gained execute bits (mode={oct(cloned_data_mode)})"
            )
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_broken_symlink_is_skipped_not_fatal(self, tmp_path: pathlib.Path) -> None:
        """A broken symlink in the workspace must be skipped, not crash the clone.

        The old code caught only PermissionError; shutil.copy2 on a broken
        symlink raises FileNotFoundError (an OSError subclass), which would
        have propagated and aborted the entire clone.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "good.txt").write_text("ok")
        (workspace / "broken_link").symlink_to("/nonexistent/target")

        clone_dir = _clone_workspace(str(workspace))
        try:
            cloned = pathlib.Path(clone_dir)
            assert (cloned / "good.txt").read_text() == "ok"
            assert not (cloned / "broken_link").exists()
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def test_symlink_to_directory_is_skipped_not_fatal(self, tmp_path: pathlib.Path) -> None:
        """A symlink-to-directory in filenames must be skipped, not crash the clone.

        os.walk (followlinks=False) places dir-symlinks in filenames.
        shutil.copy2 on them raises IsADirectoryError (OSError subclass).
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "data.txt").write_text("data")
        (workspace / "good.txt").write_text("ok")
        (workspace / "dir_link").symlink_to(real_dir)

        clone_dir = _clone_workspace(str(workspace))
        try:
            cloned = pathlib.Path(clone_dir)
            assert (cloned / "good.txt").read_text() == "ok"
            # The symlink itself should not have been copied as a file
            assert not (cloned / "dir_link").is_file()
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)
