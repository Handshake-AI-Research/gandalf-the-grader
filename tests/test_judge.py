"""Tests for gandalf.judge."""

import json
import os
import pathlib
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from litellm.exceptions import BadRequestError, RateLimitError
from openhands.sdk.llm import Message
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMNoResponseError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from openhands.sdk.llm.message import TextContent

from gandalf.judge import (
    GatewayConfigurationError,
    build_batch_judge_prompt,
    build_judge_prompt,
    classify_gateway_error,
    create_llm,
    make_verdict_path,
    mcp_server_to_config,
    parse_extra_headers,
    read_batch_verdict,
    read_verdict,
    run_judge,
    run_judge_batch,
)
from gandalf.models import LLMUsage, MCPServer
from tests.conftest import MOCK_USAGE


class TestBuildJudgePrompt:
    def test_contains_all_sections(self) -> None:
        prompt = build_judge_prompt(
            instructions="Build a web app",
            final_output="Done!",
            criterion="The file index.html exists",
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
            criterion="c",
            verdict_path="/tmp/v.json",
        )
        assert "Agent's Prompt" not in prompt

    def test_requests_evidence_field(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criterion="c",
            verdict_path="/tmp/v.json",
        )
        assert '"evidence"' in prompt

    def test_includes_json_example(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criterion="c",
            verdict_path="/tmp/v.json",
        )
        assert '"met"' in prompt
        assert '"reasoning"' in prompt

    def test_guidance_included_when_provided(self) -> None:
        guidance = "Use openpyxl to inspect .xlsx files. Do not cat binary files."
        prompt = build_judge_prompt(
            instructions="x",
            final_output="z",
            criterion="c",
            verdict_path="/tmp/v.json",
            judge_guidance=guidance,
        )
        assert guidance in prompt

    def test_no_guidance_block_when_empty(self) -> None:
        prompt_empty = build_judge_prompt(
            instructions="x",
            final_output="z",
            criterion="c",
            verdict_path="/tmp/v.json",
            judge_guidance="",
        )
        prompt_default = build_judge_prompt(
            instructions="x",
            final_output="z",
            criterion="c",
            verdict_path="/tmp/v.json",
        )
        assert prompt_empty == prompt_default

    def test_guidance_appears_before_task_instructions(self) -> None:
        guidance = "GUIDANCE_MARKER"
        prompt = build_judge_prompt(
            instructions="INSTRUCTIONS_MARKER",
            final_output="z",
            criterion="c",
            verdict_path="/tmp/v.json",
            judge_guidance=guidance,
        )
        assert prompt.index("GUIDANCE_MARKER") < prompt.index("INSTRUCTIONS_MARKER")

    def test_section_order_with_guidance(self) -> None:
        prompt = build_judge_prompt(
            instructions="INSTR",
            final_output="OUTPUT",
            criterion="CRIT",
            verdict_path="/tmp/v.json",
            judge_guidance="GUIDANCE",
        )
        preamble_idx = prompt.index("expert judge")
        guidance_idx = prompt.index("GUIDANCE")
        instr_idx = prompt.index("INSTR")
        output_idx = prompt.index("OUTPUT")
        crit_idx = prompt.index("CRIT")
        assert preamble_idx < guidance_idx < instr_idx < output_idx < crit_idx


class TestMCPServerToConfig:
    """Verify MCPServer is rendered to FastMCP MCPConfig server-entry shape."""

    def test_stdio_minimal(self) -> None:
        srv = MCPServer(name="x", command="/bin/x")
        assert mcp_server_to_config(srv) == {"command": "/bin/x"}

    def test_stdio_with_args(self) -> None:
        srv = MCPServer(name="x", command="/bin/x", args=["--verbose", "--port", "8000"])
        assert mcp_server_to_config(srv) == {
            "command": "/bin/x",
            "args": ["--verbose", "--port", "8000"],
        }

    def test_stdio_omits_empty_args(self) -> None:
        srv = MCPServer(name="x", command="/bin/x", args=[])
        assert "args" not in mcp_server_to_config(srv)

    def test_remote_streamable_http(self) -> None:
        srv = MCPServer(name="x", transport="streamable-http", url="http://localhost:8000/mcp")
        assert mcp_server_to_config(srv) == {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable-http",
        }

    def test_remote_with_headers(self) -> None:
        srv = MCPServer(
            name="x",
            transport="http",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert mcp_server_to_config(srv) == {
            "url": "https://api.example.com/mcp",
            "transport": "http",
            "headers": {"Authorization": "Bearer token"},
        }

    def test_remote_omits_empty_headers(self) -> None:
        srv = MCPServer(name="x", transport="sse", url="http://localhost:8000/sse")
        assert "headers" not in mcp_server_to_config(srv)


class TestMakeVerdictPath:
    """Ensure verdict files go to workdir, not /tmp.

    Regression: the old code always used tempfile.gettempdir() (/tmp), requiring
    sandbox_user to have write access to /tmp.  The fix accepts a *dir* parameter
    and run_judge/run_judge_batch pass judge_input.workdir (which the grader
    has already made world-writable), so sandbox_user never needs /tmp write access.
    """

    def test_default_uses_system_tmpdir(self) -> None:
        path = make_verdict_path()
        assert path.startswith(tempfile.gettempdir())
        assert "verdict_" in path
        assert path.endswith(".json")

    def test_dir_overrides_tmpdir(self, tmp_path: pathlib.Path) -> None:
        """When dir is provided the verdict path must be inside it, not in /tmp.

        This test fails on the pre-fix code (make_verdict_path had no dir param)
        and passes with the fix.
        """
        path = make_verdict_path(directory=str(tmp_path))
        assert path.startswith(str(tmp_path)), (
            f"Verdict path {path!r} should be inside workdir {tmp_path}, "
            "not in /tmp — sandbox_user may lack /tmp write access"
        )

    def test_run_judge_verdict_goes_to_workdir(self, tmp_path: pathlib.Path) -> None:
        """run_judge must pass workdir to make_verdict_path, not rely on /tmp.

        This test fails on the pre-fix code (verdict_path always used /tmp)
        and passes with the fix (verdict_path uses judge_input.workdir).
        """
        input_data = {
            "model": "test-model",
            "instructions": "do a thing",
            "final_output": "done",
            "criterion": "check something",
            "workdir": str(tmp_path),
        }
        input_path = str(tmp_path / "input.json")
        (tmp_path / "input.json").write_text(json.dumps(input_data))
        output_path = str(tmp_path / "output.json")

        captured_verdict_dir = {}

        def fake_make_verdict_path(prefix: str = "verdict_", directory: str | None = None) -> str:
            captured_verdict_dir["dir"] = directory
            # Return a path inside tmp_path so the test can write the verdict
            p = str(tmp_path / f"{prefix}test.json")
            (tmp_path / f"{prefix}test.json").write_text(json.dumps({"met": True, "reasoning": "ok", "evidence": []}))
            return p

        with (
            patch("gandalf.judge.make_verdict_path", side_effect=fake_make_verdict_path),
            patch("gandalf.judge.run_agent_session", return_value=LLMUsage()),
        ):
            run_judge(input_path, output_path)

        assert captured_verdict_dir.get("dir") == str(tmp_path), (
            f"run_judge passed dir={captured_verdict_dir.get('dir')!r} to make_verdict_path "
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
        result = read_verdict(str(p))
        assert result.met is True
        assert result.reasoning == "Looks good."
        assert result.evidence == ["checked file"]

    def test_missing_evidence_defaults_to_empty(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"met": True, "reasoning": "ok"}))
        result = read_verdict(str(p))
        assert result.met is True
        assert result.evidence == []

    def test_judge_writes_null_met_preserved(self, tmp_path: pathlib.Path) -> None:
        """If the judge writes {"met": null}, it must stay None, not become False."""
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"met": None, "reasoning": "judge errored internally"}))
        result = read_verdict(str(p))
        assert result.met is None

    def test_empty_file(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("")
        result = read_verdict(str(p))
        assert result.met is None
        assert "empty" in result.reasoning.lower()

    def test_missing_file(self) -> None:
        result = read_verdict("/nonexistent/verdict.json")
        assert result.met is None
        assert "did not write" in result.reasoning.lower()

    def test_invalid_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("not json at all")
        result = read_verdict(str(p))
        assert result.met is None
        assert "invalid JSON" in result.reasoning

    def test_missing_met_field(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"reasoning": "no met field"}))
        result = read_verdict(str(p))
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
        results = read_batch_verdict(str(p), 2)
        assert len(results) == 2
        assert results[0].met is True
        assert results[1].met is False

    def test_missing_index_gets_default_fail(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": 0, "met": True, "reasoning": "ok"}]))
        results = read_batch_verdict(str(p), 2)
        assert results[0].met is True
        assert results[1].met is None
        assert "did not return" in results[1].reasoning.lower()

    def test_non_integer_index_skipped(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": "zero", "met": True, "reasoning": "ok"}]))
        results = read_batch_verdict(str(p), 1)
        assert results[0].met is None

    def test_out_of_range_index_skipped(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps([{"index": 5, "met": True, "reasoning": "ok"}]))
        results = read_batch_verdict(str(p), 2)
        assert all(r.met is None for r in results)

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
        results = read_batch_verdict(str(p), 1)
        assert results[0].met is True
        assert results[0].reasoning == "second"

    def test_empty_file(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("")
        results = read_batch_verdict(str(p), 2)
        assert len(results) == 2
        assert all(r.met is None for r in results)

    def test_missing_file(self) -> None:
        results = read_batch_verdict("/nonexistent/verdict.json", 2)
        assert len(results) == 2
        assert all(r.met is None for r in results)

    def test_invalid_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text("not json")
        results = read_batch_verdict(str(p), 1)
        assert results[0].met is None

    def test_non_array_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "verdict.json"
        p.write_text(json.dumps({"not": "an array"}))
        results = read_batch_verdict(str(p), 1)
        assert results[0].met is None


def make_judge_input_json(tmp_path: pathlib.Path, criterion: str = "check something") -> str:
    """Write a minimal JudgeInput JSON file and return its path."""
    data = {
        "model": "test-model",
        "instructions": "do a thing",
        "final_output": "done",
        "criterion": criterion,
        "workdir": str(tmp_path),
    }
    p = tmp_path / "input.json"
    p.write_text(json.dumps(data))
    return str(p)


def make_batch_judge_input_json(tmp_path: pathlib.Path, n: int = 2) -> str:
    """Write a minimal BatchJudgeInput JSON file and return its path."""
    data = {
        "model": "test-model",
        "instructions": "do a thing",
        "final_output": "done",
        "criteria": [f"criterion {i}" for i in range(n)],
        "workdir": str(tmp_path),
    }
    p = tmp_path / "batch_input.json"
    p.write_text(json.dumps(data))
    return str(p)


class TestRunJudge:
    """Tests for run_judge — mocks run_agent_session to avoid OpenHands."""

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_success_includes_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        input_path = make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        # Pre-create the verdict file that the agent would write.
        # make_verdict_path uses tempfile.gettempdir(), so we patch it.
        verdict_data = {"met": True, "reasoning": "ok", "evidence": ["e1"]}
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["verdict"]["met"] is True
        assert result["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_verdict_missing(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        """If run_agent_session succeeds but verdict file is missing, cost is kept."""
        input_path = make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "no_such_verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        assert result["verdict"]["met"] is None
        assert result["llm_usage"]["cost_usd"] == 0.05
        assert result["llm_usage"]["prompt_tokens"] == 1000

    @patch(
        "gandalf.judge.run_agent_session",
        side_effect=RuntimeError("LLM exploded"),
    )
    def test_session_failure_has_empty_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        """If run_agent_session itself raises, usage stays empty."""
        input_path = make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        verdict = result["verdict"]
        assert verdict["met"] is None
        assert result["llm_usage"] == LLMUsage().model_dump()
        assert "LLM exploded" in verdict["reasoning"]

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    @patch(
        "gandalf.judge.read_verdict",
        side_effect=RuntimeError("Unexpected parsing error"),
    )
    def test_preserves_usage_when_read_verdict_raises(
        self,
        mock_read: Any,  # noqa: ARG002
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
    ) -> None:
        """If read_verdict raises after the session ran, usage is still preserved."""
        input_path = make_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge(input_path, output_path)

        result = json.loads((tmp_path / "output.json").read_text())
        verdict = result["verdict"]
        assert verdict["met"] is None
        assert result["llm_usage"]["cost_usd"] == 0.05
        assert result["llm_usage"]["prompt_tokens"] == 1000
        assert "Unexpected parsing error" in verdict["reasoning"]


class TestRunJudgeBatch:
    """Tests for run_judge_batch — mocks run_agent_session to avoid OpenHands."""

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_output_wraps_verdicts_and_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        input_path = make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        verdict_data = [
            {"index": 0, "met": True, "reasoning": "ok", "evidence": []},
            {"index": 1, "met": False, "reasoning": "bad", "evidence": []},
        ]
        with patch(
            "gandalf.judge.make_verdict_path",
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

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_session_usage_is_top_level(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        """Session-level llm_usage should be a sibling of verdicts, not duplicated per-verdict."""
        input_path = make_batch_judge_input_json(tmp_path, n=1)
        output_path = str(tmp_path / "output.json")

        verdict_data = [{"index": 0, "met": True, "reasoning": "ok"}]
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            (tmp_path / "verdict.json").write_text(json.dumps(verdict_data))
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert data["verdicts"][0]["met"] is True

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_verdict_missing(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        input_path = make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "no_such_verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert all(v["met"] is None for v in data["verdicts"])

    @patch(
        "gandalf.judge.run_agent_session",
        side_effect=RuntimeError("LLM exploded"),
    )
    def test_session_failure_has_empty_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        input_path = make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"] == LLMUsage().model_dump()
        assert all(v["met"] is None for v in data["verdicts"])

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    @patch(
        "gandalf.judge.read_batch_verdict",
        side_effect=RuntimeError("Batch parsing blew up"),
    )
    def test_preserves_usage_when_read_batch_verdict_raises(
        self,
        mock_read: Any,  # noqa: ARG002
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
    ) -> None:
        """If read_batch_verdict raises after the session ran, usage is preserved."""
        input_path = make_batch_judge_input_json(tmp_path, n=2)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "verdict.json"),
        ):
            run_judge_batch(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["llm_usage"]["cost_usd"] == 0.05
        assert data["llm_usage"]["prompt_tokens"] == 1000
        assert all(v["met"] is None for v in data["verdicts"])
        assert "Batch parsing blew up" in data["verdicts"][0]["reasoning"]


class TestBuildJudgePromptXMLTags:
    """Verify prompts use XML tags instead of Markdown headings."""

    def test_single_uses_xml_tags(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="y",
            criterion="c",
            verdict_path="/tmp/v.json",
        )
        assert "<task_instructions>" in prompt
        assert "</task_instructions>" in prompt
        assert "<agent_final_output>" in prompt
        assert "<evaluation_criterion>" in prompt
        assert "<judge_instructions>" in prompt
        assert "## " not in prompt

    def test_batch_uses_xml_tags(self) -> None:
        criteria = ["c0", "c1"]
        prompt = build_batch_judge_prompt(
            instructions="x",
            final_output="y",
            criteria=criteria,
            verdict_path="/tmp/v.json",
        )
        assert "<task_instructions>" in prompt
        assert "<evaluation_criteria>\n  [0] c0\n  [1] c1\n</evaluation_criteria>" in prompt
        assert "<judge_instructions>" in prompt
        assert "0 through 1" in prompt
        assert "## " not in prompt

    def test_single_guidance_uses_xml_tag(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="y",
            criterion="c",
            verdict_path="/tmp/v.json",
            judge_guidance="GUIDANCE_TEXT",
        )
        assert "<judge_guidance>" in prompt
        assert "GUIDANCE_TEXT" in prompt
        assert "</judge_guidance>" in prompt

    def test_single_no_guidance_tag_when_empty(self) -> None:
        prompt = build_judge_prompt(
            instructions="x",
            final_output="y",
            criterion="c",
            verdict_path="/tmp/v.json",
        )
        assert "<judge_guidance>" not in prompt


class TestBuildJudgePromptCustomTemplate:
    """Verify judge_prompt overrides the built-in prompt."""

    def test_single_custom_template(self) -> None:
        template = "CUSTOM: {{ instructions }} | {{ criterion }} | {{ verdict_path }}"
        prompt = build_judge_prompt(
            instructions="do stuff",
            final_output="done",
            criterion="check it",
            verdict_path="/tmp/v.json",
            judge_prompt=template,
        )
        assert prompt == "CUSTOM: do stuff | check it | /tmp/v.json"

    def test_batch_custom_template(self) -> None:
        template = "BATCH: {% for c in criteria %}[{{ loop.index0 }}] {{ c }} {% endfor %}"
        criteria = ["c0", "c1"]
        prompt = build_batch_judge_prompt(
            instructions="x",
            final_output="y",
            criteria=criteria,
            verdict_path="/tmp/v.json",
            judge_prompt=template,
        )
        assert "BATCH:" in prompt
        assert "[0] c0" in prompt
        assert "[1] c1" in prompt

    def test_batch_custom_template_loop_index(self) -> None:
        template = "{% for c in criteria %}{{ loop.index0 }}:{{ c }} {% endfor %}"
        criteria = ["c0", "c1"]
        prompt = build_batch_judge_prompt(
            instructions="x",
            final_output="y",
            criteria=criteria,
            verdict_path="/tmp/v.json",
            judge_prompt=template,
        )
        assert "0:c0" in prompt
        assert "1:c1" in prompt

    def test_custom_template_receives_judge_guidance(self) -> None:
        template = "{% if judge_guidance %}G:{{ judge_guidance }}{% endif %}"
        prompt = build_judge_prompt(
            instructions="x",
            final_output="y",
            criterion="c",
            verdict_path="/tmp/v.json",
            judge_guidance="be careful",
            judge_prompt=template,
        )
        assert prompt == "G:be careful"

    def test_custom_template_no_builtin_content(self) -> None:
        template = "ONLY THIS"
        prompt = build_judge_prompt(
            instructions="x",
            final_output="y",
            criterion="c",
            verdict_path="/tmp/v.json",
            judge_prompt=template,
        )
        assert prompt == "ONLY THIS"
        assert "expert judge" not in prompt


def set_gateway_env(monkeypatch: pytest.MonkeyPatch, base_url: str = "https://gateway.example/v1") -> None:
    monkeypatch.setenv("USE_LITELLM_PROXY", "true")
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", base_url)
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "gateway-key")
    monkeypatch.setenv("LLM_API_KEY", "gateway-key")
    monkeypatch.setenv("LLM_EXTRA_HEADERS_JSON", json.dumps({"CF-Access-Client-Id": "client-id"}))


class TestGatewayConfiguration:
    def test_gateway_llm_uses_model_key_and_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_gateway_env(monkeypatch)
        monkeypatch.setenv("LLM_BASE_URL", "https://stale-direct.example/v1")

        with patch("gandalf.judge.LLM") as mock_llm:
            create_llm("gemini/gemini-2.5-pro")

        mock_llm.assert_called_once_with(
            model="gemini/gemini-2.5-pro",
            api_key="gateway-key",
            extra_headers={"CF-Access-Client-Id": "client-id"},
        )

    def test_direct_mode_ignores_stale_gateway_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_gateway_env(monkeypatch)
        monkeypatch.setenv("USE_LITELLM_PROXY", "false")
        monkeypatch.setenv("LLM_API_KEY", "direct-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://direct.example/v1")
        monkeypatch.setenv("LLM_EXTRA_HEADERS_JSON", "not-json")

        with patch("gandalf.judge.LLM") as mock_llm:
            create_llm("anthropic/test-model")

        mock_llm.assert_called_once_with(
            model="anthropic/test-model",
            api_key="direct-key",
            base_url="https://direct.example/v1",
        )

    @pytest.mark.parametrize(
        "missing_name",
        ["LITELLM_PROXY_API_BASE", "LITELLM_PROXY_API_KEY", "LLM_API_KEY", "LLM_EXTRA_HEADERS_JSON"],
    )
    def test_gateway_requires_each_value(self, missing_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        set_gateway_env(monkeypatch)
        monkeypatch.delenv(missing_name)

        with pytest.raises(GatewayConfigurationError, match="Gateway configuration is incomplete"):
            create_llm("gemini/gemini-2.5-pro")

    @pytest.mark.parametrize(
        "headers",
        ["not-json", "[]", '{"header": 1}', '{"header": true}'],
    )
    def test_gateway_rejects_invalid_header_json(self, headers: str, monkeypatch: pytest.MonkeyPatch) -> None:
        set_gateway_env(monkeypatch)
        monkeypatch.setenv("LLM_EXTRA_HEADERS_JSON", headers)

        with pytest.raises(GatewayConfigurationError, match="Gateway configuration is invalid") as exc_info:
            create_llm("gemini/gemini-2.5-pro")

        assert headers not in str(exc_info.value)

    def test_header_parser_does_not_expose_secret_input(self) -> None:
        secret_input = '{"Authorization":"secret-value"'
        with pytest.raises(GatewayConfigurationError, match="Gateway configuration is invalid") as exc_info:
            parse_extra_headers(secret_input)
        assert "secret-value" not in str(exc_info.value)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("USE_LITELLM_PROXY", "TRUE"),
            ("LITELLM_PROXY_API_BASE", "https://gateway.example"),
            ("LITELLM_PROXY_API_BASE", "https://user:secret@gateway.example/v1"),
            ("LITELLM_PROXY_API_KEY", "different-key"),
        ],
    )
    def test_gateway_rejects_invalid_transport_values(
        self, name: str, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_gateway_env(monkeypatch)
        monkeypatch.setenv(name, value)
        with pytest.raises(GatewayConfigurationError, match="Gateway configuration is invalid"):
            create_llm("gemini/gemini-2.5-pro")

    @pytest.mark.parametrize(
        ("exception", "reason"),
        [
            (LLMAuthenticationError(), "auth-rejected"),
            (LLMBadRequestError(), "request-rejected"),
            (LLMRateLimitError(), "rate-limited"),
            (LLMServiceUnavailableError(), "gateway-server-error"),
            (LLMTimeoutError(), "gateway-unreachable"),
            (LLMNoResponseError(), "response-interrupted"),
        ],
    )
    def test_classifies_typed_gateway_failures(self, exception: BaseException, reason: str) -> None:
        marker = classify_gateway_error(exception)
        assert marker is not None
        assert marker.reason == reason

    def test_does_not_classify_plain_error_message(self) -> None:
        assert classify_gateway_error(RuntimeError("401 rate limit timeout")) is None

    def test_classification_preserves_typed_http_status(self) -> None:
        response = httpx.Response(429, request=httpx.Request("POST", "https://gateway.example/v1/chat/completions"))
        exception = RateLimitError(
            "secret response",
            llm_provider="openai",
            model="gemini/gemini-2.5-pro",
            response=response,
        )
        marker = classify_gateway_error(exception)
        assert marker is not None
        assert marker.model_dump() == {"version": 1, "reason": "rate-limited", "http_status": 429}

    def test_classifies_openai_bio_policy_from_structured_body(self) -> None:
        response = httpx.Response(400, request=httpx.Request("POST", "https://gateway.example/v1/chat/completions"))
        exception = BadRequestError(
            "This content was flagged for possible biological risk.",
            llm_provider="openai",
            model="openai/gpt-5.6-sol",
            response=response,
            body={
                "error": {
                    "message": "This content was flagged for possible biological risk.",
                    "type": "invalid_request_error",
                    "code": "bio_policy",
                }
            },
        )

        marker = classify_gateway_error(exception)

        assert marker is not None
        assert marker.model_dump() == {
            "version": 1,
            "reason": "provider-policy-rejected",
            "http_status": 400,
        }

    def test_does_not_infer_provider_policy_from_exception_text(self) -> None:
        response = httpx.Response(400, request=httpx.Request("POST", "https://gateway.example/v1/chat/completions"))
        exception = BadRequestError(
            "code: bio_policy",
            llm_provider="openai",
            model="openai/gpt-5.6-sol",
            response=response,
        )

        marker = classify_gateway_error(exception)

        assert marker is not None
        assert marker.reason == "request-rejected"

    @patch("gandalf.judge.run_agent_session", side_effect=LLMRateLimitError("secret response"))
    def test_single_output_carries_sanitized_marker(
        self,
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("USE_LITELLM_PROXY", "true")
        output_path = str(tmp_path / "output.json")
        with patch("gandalf.judge.make_verdict_path", return_value=str(tmp_path / "verdict.json")):
            run_judge(make_judge_input_json(tmp_path), output_path)

        raw_output = pathlib.Path(output_path).read_text()
        result = json.loads(raw_output)
        assert result["gateway_error"] == {"version": 1, "reason": "rate-limited"}
        assert result["verdict"]["met"] is None
        assert "gateway_error" not in result["verdict"]
        assert "secret response" not in raw_output

    @patch("gandalf.judge.run_agent_session", side_effect=LLMTimeoutError("secret response"))
    def test_batch_output_carries_sanitized_marker(
        self,
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("USE_LITELLM_PROXY", "true")
        output_path = str(tmp_path / "output.json")
        with patch("gandalf.judge.make_verdict_path", return_value=str(tmp_path / "verdict.json")):
            run_judge_batch(make_batch_judge_input_json(tmp_path), output_path)

        result = json.loads(pathlib.Path(output_path).read_text())
        assert result["gateway_error"] == {"version": 1, "reason": "gateway-unreachable"}
        assert all("gateway_error" not in verdict for verdict in result["verdicts"])


class TestGatewayTransport:
    def test_real_client_retries_same_gateway_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests: list[dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                requests.append(
                    {
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": body,
                    }
                )
                response: dict[str, Any]
                if len(requests) == 1:
                    response = {"error": {"message": "retry", "type": "rate_limit_error", "code": "rate_limit"}}
                    status = 429
                else:
                    response = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                    status = 200
                encoded = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            set_gateway_env(monkeypatch, f"http://127.0.0.1:{server.server_port}/v1")
            monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
            headers = {
                "CF-Access-Client-Id": "client-id",
                "CF-Access-Client-Secret": "client-secret",
                "x-litellm-spend-logs-metadata": '{"task":"task-id"}',
                "x-litellm-tags": "rle,verifier",
            }
            monkeypatch.setenv("LLM_EXTRA_HEADERS_JSON", json.dumps(headers))

            llm = create_llm("gemini/gemini-2.5-pro")
            llm.num_retries = 1
            llm.retry_min_wait = 0
            llm.retry_max_wait = 0
            llm.retry_multiplier = 0
            llm.reasoning_effort = "none"
            llm.completion(messages=[Message(role="user", content=[TextContent(text="hello")])])
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()

        assert len(requests) == 2
        for request in requests:
            assert request["path"] == "/v1/chat/completions"
            assert request["body"]["model"] == "gemini/gemini-2.5-pro"
            assert request["headers"]["cf-access-client-id"] == "client-id"
            assert request["headers"]["cf-access-client-secret"] == "client-secret"
            assert request["headers"]["x-litellm-spend-logs-metadata"] == '{"task":"task-id"}'
            assert request["headers"]["x-litellm-tags"] == "rle,verifier"

    def test_real_direct_client_ignores_stale_gateway_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests: list[dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                response = {
                    "id": "chatcmpl-direct",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            monkeypatch.setenv("USE_LITELLM_PROXY", "false")
            monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://127.0.0.1:1/v1")
            monkeypatch.setenv("LITELLM_PROXY_API_KEY", "stale-proxy-key")
            monkeypatch.setenv("LLM_EXTRA_HEADERS_JSON", "not-json")
            monkeypatch.setenv("LLM_API_KEY", "direct-key")
            monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")

            llm = create_llm("openai/gpt-4o-mini")
            llm.num_retries = 0
            llm.reasoning_effort = "none"
            llm.completion(messages=[Message(role="user", content=[TextContent(text="hello")])])
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()

        assert len(requests) == 1
        assert requests[0]["path"] == "/v1/chat/completions"
        assert requests[0]["body"]["model"] == "gpt-4o-mini"
        assert "CF-Access-Client-Id" not in requests[0]["headers"]


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
            "criterion": "The file hello.txt exists in the workspace and contains 'hello world'.",
            "workdir": str(workspace),
        }
        input_path = str(tmp_path / "input.json")
        output_path = str(tmp_path / "output.json")
        pathlib.Path(input_path).write_text(json.dumps(judge_input))

        run_judge(input_path, output_path)

        result = json.loads(pathlib.Path(output_path).read_text())
        verdict = result["verdict"]
        assert verdict["met"] is True
        assert verdict["reasoning"]
        assert isinstance(verdict["evidence"], list)
