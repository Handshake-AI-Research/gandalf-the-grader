"""Tests for gandalf.judge."""

import json
import os
import pathlib
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from gandalf.judge import (
    _make_verdict_path,
    _read_batch_verdict,
    _read_verdict,
    build_judge_prompt,
    run_judge,
    run_judge_batch,
)


class TestBuildJudgePrompt:
    def test_contains_all_sections(self) -> None:
        prompt = build_judge_prompt(
            instructions="Build a web app",
            final_output="Done!",
            criteria="The file index.html exists",
            verdict_path="/tmp/verdict.json",
        )
        assert "Build a web app" in prompt
        assert "Done!" in prompt
        assert "The file index.html exists" in prompt
        assert "/tmp/verdict.json" in prompt

    def test_no_user_prompt_section(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert "Agent's Prompt" not in prompt

    def test_requests_evidence_field(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert '"evidence"' in prompt

    def test_includes_json_example(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert '"met"' in prompt
        assert '"reasoning"' in prompt

    def test_guidance_included_when_provided(self) -> None:
        guidance = "Use openpyxl to inspect .xlsx files. Do not cat binary files."
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
            judge_guidance=guidance,
        )
        assert guidance in prompt

    def test_no_guidance_block_when_empty(self) -> None:
        prompt_empty = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
            judge_guidance="",
        )
        prompt_default = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert prompt_empty == prompt_default

    def test_guidance_appears_before_task_instructions(self) -> None:
        guidance = "GUIDANCE_MARKER"
        prompt = build_judge_prompt(
            instructions="INSTRUCTIONS_MARKER",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
            judge_guidance=guidance,
        )
        assert prompt.index("GUIDANCE_MARKER") < prompt.index("INSTRUCTIONS_MARKER")

    def test_section_order_with_guidance(self) -> None:
        prompt = build_judge_prompt(
            instructions="INSTR",
            final_output="OUTPUT",
            criteria="CRIT",
            verdict_path="/tmp/v.json",
            judge_guidance="GUIDANCE",
        )
        preamble_idx = prompt.index("expert judge")
        guidance_idx = prompt.index("GUIDANCE")
        instr_idx = prompt.index("INSTR")
        output_idx = prompt.index("OUTPUT")
        crit_idx = prompt.index("CRIT")
        assert preamble_idx < guidance_idx < instr_idx < output_idx < crit_idx


class TestMakeVerdictPath:
    """Ensure verdict files go to workdir, not /tmp.

    Regression: the old code always used tempfile.gettempdir() (/tmp), requiring
    sandbox_user to have write access to /tmp.  The fix accepts a *dir* parameter
    and run_judge/run_judge_batch pass judge_input.workdir (which the grader
    has already made world-writable), so sandbox_user never needs /tmp write access.
    """

    def test_default_uses_system_tmpdir(self) -> None:
        path = _make_verdict_path()
        assert path.startswith(tempfile.gettempdir())
        assert "verdict_" in path
        assert path.endswith(".json")

    def test_dir_overrides_tmpdir(self, tmp_path: pathlib.Path) -> None:
        """When dir is provided the verdict path must be inside it, not in /tmp.

        This test fails on the pre-fix code (_make_verdict_path had no dir param)
        and passes with the fix.
        """
        path = _make_verdict_path(dir=str(tmp_path))
        assert path.startswith(str(tmp_path)), (
            f"Verdict path {path!r} should be inside workdir {tmp_path}, "
            "not in /tmp — sandbox_user may lack /tmp write access"
        )

    def test_run_judge_verdict_goes_to_workdir(self, tmp_path: pathlib.Path) -> None:
        """run_judge must pass workdir to _make_verdict_path, not rely on /tmp.

        This test fails on the pre-fix code (verdict_path always used /tmp)
        and passes with the fix (verdict_path uses judge_input.workdir).
        """
        input_data = {
            "model": "test-model",
            "instructions": "do a thing",
            "final_output": "done",
            "criteria": "check something",
            "workdir": str(tmp_path),
        }
        input_path = str(tmp_path / "input.json")
        (tmp_path / "input.json").write_text(json.dumps(input_data))
        output_path = str(tmp_path / "output.json")

        captured_verdict_dir = {}

        def _fake_make_verdict_path(prefix: str = "verdict_", dir: str | None = None) -> str:
            captured_verdict_dir["dir"] = dir
            # Return a path inside tmp_path so the test can write the verdict
            p = str(tmp_path / f"{prefix}test.json")
            (tmp_path / f"{prefix}test.json").write_text(json.dumps({"met": True, "reasoning": "ok", "evidence": []}))
            return p

        with (
            patch("gandalf.judge._make_verdict_path", side_effect=_fake_make_verdict_path),
            patch("gandalf.judge._run_agent_session", return_value={}),
        ):
            run_judge(input_path, output_path)

        assert captured_verdict_dir.get("dir") == str(tmp_path), (
            f"run_judge passed dir={captured_verdict_dir.get('dir')!r} to _make_verdict_path "
            f"but expected workdir={str(tmp_path)!r} — "
            "sandbox_user would need to create the verdict file in /tmp instead"
        )


class TestReadVerdict:
    def test_valid_verdict(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps(
                {
                    "met": True,
                    "reasoning": "Looks good.",
                    "evidence": ["checked file"],
                }
            )
        )
        result = _read_verdict(str(p))
        assert result.met is True
        assert result.reasoning == "Looks good."
        assert result.evidence == ["checked file"]

    def test_missing_evidence_defaults_to_empty(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"met": True, "reasoning": "ok"}))
        result = _read_verdict(str(p))
        assert result.met is True
        assert result.evidence == []

    def test_judge_writes_null_met_preserved(self, tmp_path: pathlib.Path) -> None:
        """If the judge writes {"met": null}, it must stay None, not become False."""
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"met": None, "reasoning": "judge errored internally"}))
        result = _read_verdict(str(p))
        assert result.met is None

    def test_empty_file(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("")
        result = _read_verdict(str(p))
        assert result.met is None
        assert "empty" in result.reasoning.lower()

    def test_missing_file(self) -> None:
        result = _read_verdict("/nonexistent/verdict.json")
        assert result.met is None
        assert "did not write" in result.reasoning.lower()

    def test_invalid_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("not json at all")
        result = _read_verdict(str(p))
        assert result.met is None
        assert "invalid JSON" in result.reasoning

    def test_missing_met_field(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"reasoning": "no met field"}))
        result = _read_verdict(str(p))
        assert result.met is None
        assert "missing" in result.reasoning.lower()


class TestReadBatchVerdict:
    def test_valid_batch(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps(
                [
                    {"index": 0, "met": True, "reasoning": "ok", "evidence": ["a"]},
                    {"index": 1, "met": False, "reasoning": "bad", "evidence": []},
                ]
            )
        )
        results = _read_batch_verdict(str(p), 2)
        assert len(results) == 2
        assert results[0]["met"] is True
        assert results[1]["met"] is False

    def test_missing_index_gets_default_fail(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": 0, "met": True, "reasoning": "ok"}]))
        results = _read_batch_verdict(str(p), 2)
        assert results[0]["met"] is True
        assert results[1]["met"] is None
        assert "did not return" in results[1]["reasoning"].lower()

    def test_non_integer_index_skipped(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": "zero", "met": True, "reasoning": "ok"}]))
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["met"] is None

    def test_out_of_range_index_skipped(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": 5, "met": True, "reasoning": "ok"}]))
        results = _read_batch_verdict(str(p), 2)
        assert all(r["met"] is None for r in results)

    def test_duplicate_index_last_wins(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps(
                [
                    {"index": 0, "met": False, "reasoning": "first"},
                    {"index": 0, "met": True, "reasoning": "second"},
                ]
            )
        )
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["met"] is True
        assert results[0]["reasoning"] == "second"

    def test_empty_file(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("")
        results = _read_batch_verdict(str(p), 2)
        assert len(results) == 2
        assert all(r["met"] is None for r in results)

    def test_missing_file(self) -> None:
        results = _read_batch_verdict("/nonexistent/verdict.json", 2)
        assert len(results) == 2
        assert all(r["met"] is None for r in results)

    def test_invalid_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("not json")
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["met"] is None

    def test_non_array_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"not": "an array"}))
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["met"] is None


MOCK_USAGE = {
    "cost_usd": 0.05,
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "cache_read_tokens": 200,
}


def _make_judge_input_json(tmp_path: pathlib.Path, criteria: str = "check something") -> str:
    """Write a minimal JudgeInput JSON file and return its path."""
    data = {
        "model": "test-model",
        "instructions": "do a thing",
        "final_output": "done",
        "criteria": criteria,
        "workdir": str(tmp_path),
    }
    p = tmp_path / "input.json"
    p.write_text(json.dumps(data))
    return str(p)


def _make_batch_judge_input_json(tmp_path: pathlib.Path, n: int = 2) -> str:
    """Write a minimal BatchJudgeInput JSON file and return its path."""
    data = {
        "model": "test-model",
        "instructions": "do a thing",
        "final_output": "done",
        "criteria": [{"index": i, "criteria": f"criterion {i}"} for i in range(n)],
        "workdir": str(tmp_path),
    }
    p = tmp_path / "batch_input.json"
    p.write_text(json.dumps(data))
    return str(p)


class TestRunJudge:
    """Tests for run_judge — mocks _run_agent_session to avoid OpenHands."""

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_success_includes_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        # Pre-create the verdict file that the agent would write.
        # _make_verdict_path uses tempfile.gettempdir(), so we patch it.
        verdict_data = {"met": True, "reasoning": "ok", "evidence": ["e1"]}
        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["met"] is True
        assert result["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_verdict_missing(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        """If _run_agent_session succeeds but verdict file is missing, cost is kept."""
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "no_such_verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["met"] is None
        assert result["llm_usage"]["cost_usd"] == 0.05
        assert result["llm_usage"]["prompt_tokens"] == 1000

    @patch(
        "gandalf.judge._run_agent_session",
        side_effect=RuntimeError("LLM exploded"),
    )
    def test_session_failure_has_empty_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        """If _run_agent_session itself raises, usage stays empty."""
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["met"] is None
        assert result["llm_usage"] == {}
        assert "LLM exploded" in result["reasoning"]

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    @patch(
        "gandalf.judge._read_verdict",
        side_effect=RuntimeError("Unexpected parsing error"),
    )
    def test_preserves_usage_when_read_verdict_raises(
        self, mock_read: Any, mock_session: Any, tmp_path: pathlib.Path
    ) -> None:
        """If _read_verdict raises after the session ran, usage is still preserved."""
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["met"] is None
        assert result["llm_usage"]["cost_usd"] == 0.05
        assert result["llm_usage"]["prompt_tokens"] == 1000
        assert "Unexpected parsing error" in result["reasoning"]


class TestRunJudgeBatch:
    """Tests for run_judge_batch — mocks _run_agent_session to avoid OpenHands."""

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_output_wraps_verdicts_and_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        verdict_data = [
            {"index": 0, "met": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "met": False, "reasoning": "bad", "evidence": []},
        ]
        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert "verdicts" in data
        assert "llm_usage" in data
        assert len(data["verdicts"]) == 2
        assert data["verdicts"][0]["met"] is True
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_no_per_verdict_usage_keys(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        """Verdicts should NOT contain llm_usage — it's a sibling field."""
        input_path = _make_batch_judge_input_json(tmp_path, n=1)
        output_path = str(tmp_path / "output.json")

        verdict_data = [{"index": 0, "met": True, "reasoning": "ok"}]
        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        for v in data["verdicts"]:
            assert "llm_usage" not in v

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_verdict_missing(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "no_such_verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert all(v["met"] is None for v in data["verdicts"])

    @patch(
        "gandalf.judge._run_agent_session",
        side_effect=RuntimeError("LLM exploded"),
    )
    def test_session_failure_has_empty_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"] == {}
        assert all(v["met"] is None for v in data["verdicts"])

    @patch("gandalf.judge._run_agent_session", return_value=MOCK_USAGE)
    @patch(
        "gandalf.judge._read_batch_verdict",
        side_effect=RuntimeError("Batch parsing blew up"),
    )
    def test_preserves_usage_when_read_batch_verdict_raises(
        self, mock_read: Any, mock_session: Any, tmp_path: pathlib.Path
    ) -> None:
        """If _read_batch_verdict raises after the session ran, usage is preserved."""
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert data["llm_usage"]["prompt_tokens"] == 1000
        assert all(v["met"] is None for v in data["verdicts"])
        assert "Batch parsing blew up" in data["verdicts"][0]["reasoning"]


class TestRunJudgeLLM:
    """End-to-end tests that call a real LLM via the OpenHands SDK.

    Each test is parameterized across providers. Tests for providers whose
    API key is not set in the environment are skipped automatically.
    """

    @pytest.mark.llm
    @pytest.mark.parametrize(
        ("model", "api_key_env"),
        [
            ("gemini/gemini-2.5-flash", "GOOGLE_API_KEY"),
            ("anthropic/claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
            ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ],
        ids=["gemini", "anthropic", "openai"],
    )
    def test_single_criterion_met(
        self,
        model: str,
        api_key_env: str,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Judge should detect that a file exists in the workspace."""
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            pytest.skip(f"{api_key_env} not set")
        monkeypatch.setenv("LLM_API_KEY", api_key)

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "hello.txt").write_text("hello world")

        judge_input = {
            "model": model,
            "instructions": "Create a file called hello.txt containing 'hello world'.",
            "final_output": "Done, I created hello.txt.",
            "criteria": "The file hello.txt exists in the workspace and contains 'hello world'.",
            "workdir": str(workspace),
        }
        input_path = str(tmp_path / "input.json")
        output_path = str(tmp_path / "output.json")
        pathlib.Path(input_path).write_text(json.dumps(judge_input))

        run_judge(input_path, output_path)

        result = json.loads(pathlib.Path(output_path).read_text())
        assert result["met"] is True
        assert result["reasoning"]
        assert isinstance(result["evidence"], list)
