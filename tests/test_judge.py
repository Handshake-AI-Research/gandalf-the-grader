"""Tests for gandalf_grader.judge."""

import json
from unittest.mock import patch

from gandalf_grader.judge import (
    _read_batch_verdict,
    _read_verdict,
    build_judge_prompt,
    run_judge,
    run_judge_batch,
)


class TestBuildJudgePrompt:
    def test_contains_all_sections(self):
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

    def test_no_user_prompt_section(self):
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert "Agent's Prompt" not in prompt

    def test_requests_evidence_field(self):
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert '"evidence"' in prompt

    def test_includes_json_example(self):
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
        )
        assert '"passed"' in prompt
        assert '"reasoning"' in prompt

    def test_guidance_included_when_provided(self):
        guidance = "Use openpyxl to inspect .xlsx files. Do not cat binary files."
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
            judge_guidance=guidance,
        )
        assert guidance in prompt

    def test_no_guidance_block_when_empty(self):
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

    def test_guidance_appears_before_task_instructions(self):
        guidance = "GUIDANCE_MARKER"
        prompt = build_judge_prompt(
            instructions="INSTRUCTIONS_MARKER",
            final_output="z",
            criteria="c",
            verdict_path="/tmp/v.json",
            judge_guidance=guidance,
        )
        assert prompt.index("GUIDANCE_MARKER") < prompt.index("INSTRUCTIONS_MARKER")

    def test_section_order_with_guidance(self):
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


class TestReadVerdict:
    def test_valid_verdict(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps(
                {
                    "passed": True,
                    "reasoning": "Looks good.",
                    "evidence": ["checked file"],
                }
            )
        )
        result = _read_verdict(str(p))
        assert result.passed is True
        assert result.reasoning == "Looks good."
        assert result.evidence == ["checked file"]

    def test_missing_evidence_defaults_to_empty(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"passed": True, "reasoning": "ok"}))
        result = _read_verdict(str(p))
        assert result.passed is True
        assert result.evidence == []

    def test_empty_file(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text("")
        result = _read_verdict(str(p))
        assert result.passed is False
        assert "empty" in result.reasoning.lower()

    def test_missing_file(self):
        result = _read_verdict("/nonexistent/verdict.json")
        assert result.passed is False
        assert "did not write" in result.reasoning.lower()

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text("not json at all")
        result = _read_verdict(str(p))
        assert result.passed is False
        assert "invalid JSON" in result.reasoning

    def test_missing_passed_field(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"reasoning": "no passed field"}))
        result = _read_verdict(str(p))
        assert result.passed is False
        assert "missing" in result.reasoning.lower()


class TestReadBatchVerdict:
    def test_valid_batch(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps(
                [
                    {"index": 0, "passed": True, "reasoning": "ok", "evidence": ["a"]},
                    {"index": 1, "passed": False, "reasoning": "bad", "evidence": []},
                ]
            )
        )
        results = _read_batch_verdict(str(p), 2)
        assert len(results) == 2
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False

    def test_missing_index_gets_default_fail(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": 0, "passed": True, "reasoning": "ok"}]))
        results = _read_batch_verdict(str(p), 2)
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False
        assert "did not return" in results[1]["reasoning"].lower()

    def test_non_integer_index_skipped(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps([{"index": "zero", "passed": True, "reasoning": "ok"}])
        )
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["passed"] is False

    def test_out_of_range_index_skipped(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": 5, "passed": True, "reasoning": "ok"}]))
        results = _read_batch_verdict(str(p), 2)
        assert all(r["passed"] is False for r in results)

    def test_duplicate_index_last_wins(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(
            json.dumps(
                [
                    {"index": 0, "passed": False, "reasoning": "first"},
                    {"index": 0, "passed": True, "reasoning": "second"},
                ]
            )
        )
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["passed"] is True
        assert results[0]["reasoning"] == "second"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text("")
        results = _read_batch_verdict(str(p), 2)
        assert len(results) == 2
        assert all(r["passed"] is False for r in results)

    def test_missing_file(self):
        results = _read_batch_verdict("/nonexistent/verdict.json", 2)
        assert len(results) == 2
        assert all(r["passed"] is False for r in results)

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text("not json")
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["passed"] is False

    def test_non_array_json(self, tmp_path):
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"not": "an array"}))
        results = _read_batch_verdict(str(p), 1)
        assert results[0]["passed"] is False


MOCK_USAGE = {
    "cost_usd": 0.05,
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "cache_read_tokens": 200,
}


def _make_judge_input_json(tmp_path, criteria="check something"):
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


def _make_batch_judge_input_json(tmp_path, n=2):
    """Write a minimal BatchJudgeInput JSON file and return its path."""
    data = {
        "model": "test-model",
        "instructions": "do a thing",
        "final_output": "done",
        "criteria": [
            {"index": i, "criteria": f"criterion {i}", "weight": 1.0}
            for i in range(n)
        ],
        "workdir": str(tmp_path),
    }
    p = tmp_path / "batch_input.json"
    p.write_text(json.dumps(data))
    return str(p)


class TestRunJudge:
    """Tests for run_judge — mocks _run_agent_session to avoid OpenHands."""

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_success_includes_usage(self, mock_session, tmp_path):
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        # Pre-create the verdict file that the agent would write.
        # _make_verdict_path uses tempfile.gettempdir(), so we patch it.
        verdict_data = {"passed": True, "reasoning": "ok", "evidence": ["e1"]}
        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["passed"] is True
        assert result["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_verdict_missing(self, mock_session, tmp_path):
        """If _run_agent_session succeeds but verdict file is missing, cost is kept."""
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "no_such_verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["passed"] is False
        assert result["llm_usage"]["cost_usd"] == 0.05
        assert result["llm_usage"]["prompt_tokens"] == 1000

    @patch(
        "gandalf_grader.judge._run_agent_session",
        side_effect=RuntimeError("LLM exploded"),
    )
    def test_session_failure_has_empty_usage(self, mock_session, tmp_path):
        """If _run_agent_session itself raises, usage stays empty."""
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["passed"] is False
        assert result["llm_usage"] == {}
        assert "LLM exploded" in result["reasoning"]

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    @patch(
        "gandalf_grader.judge._read_verdict",
        side_effect=RuntimeError("Unexpected parsing error"),
    )
    def test_preserves_usage_when_read_verdict_raises(
        self, mock_read, mock_session, tmp_path
    ):
        """If _read_verdict raises after the session ran, usage is still preserved."""
        input_path = _make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["passed"] is False
        assert result["llm_usage"]["cost_usd"] == 0.05
        assert result["llm_usage"]["prompt_tokens"] == 1000
        assert "Unexpected parsing error" in result["reasoning"]


class TestRunJudgeBatch:
    """Tests for run_judge_batch — mocks _run_agent_session to avoid OpenHands."""

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_output_wraps_verdicts_and_usage(self, mock_session, tmp_path):
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        verdict_data = [
            {"index": 0, "passed": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "passed": False, "reasoning": "bad", "evidence": []},
        ]
        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert "verdicts" in data
        assert "llm_usage" in data
        assert len(data["verdicts"]) == 2
        assert data["verdicts"][0]["passed"] is True
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_no_per_verdict_usage_keys(self, mock_session, tmp_path):
        """Verdicts should NOT contain llm_usage — it's a sibling field."""
        input_path = _make_batch_judge_input_json(tmp_path, n=1)
        output_path = str(tmp_path / "output.json")

        verdict_data = [{"index": 0, "passed": True, "reasoning": "ok"}]
        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        for v in data["verdicts"]:
            assert "llm_usage" not in v

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_verdict_missing(self, mock_session, tmp_path):
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "no_such_verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert all(v["passed"] is False for v in data["verdicts"])

    @patch(
        "gandalf_grader.judge._run_agent_session",
        side_effect=RuntimeError("LLM exploded"),
    )
    def test_session_failure_has_empty_usage(self, mock_session, tmp_path):
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"] == {}
        assert all(v["passed"] is False for v in data["verdicts"])

    @patch("gandalf_grader.judge._run_agent_session", return_value=MOCK_USAGE)
    @patch(
        "gandalf_grader.judge._read_batch_verdict",
        side_effect=RuntimeError("Batch parsing blew up"),
    )
    def test_preserves_usage_when_read_batch_verdict_raises(
        self, mock_read, mock_session, tmp_path
    ):
        """If _read_batch_verdict raises after the session ran, usage is preserved."""
        input_path = _make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf_grader.judge._make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert data["llm_usage"]["prompt_tokens"] == 1000
        assert all(v["passed"] is False for v in data["verdicts"])
        assert "Batch parsing blew up" in data["verdicts"][0]["reasoning"]
