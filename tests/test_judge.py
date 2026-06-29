"""Tests for gandalf.judge."""

import json
import os
import pathlib
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from gandalf.guidance_evidence import (
    extract_score_calibration_ceiling,
    has_above_midpoint_justification,
    has_action_side_effect_audit,
    has_foundational_failure_language,
    has_output_location_conflict_audit,
    has_score_calibration_audit,
    has_source_availability_audit,
    has_source_guidance_conflict_audit,
    has_source_guidance_conflict_language,
    has_source_verification_audit,
    requires_action_side_effect_audit,
    requires_output_location_conflict_audit,
    requires_source_availability_audit,
    requires_source_verification_audit,
)
from gandalf.judge import (
    build_batch_judge_prompt,
    build_guidance_judge_prompt,
    build_judge_prompt,
    make_verdict_path,
    mcp_server_to_config,
    read_batch_verdict,
    read_guidance_score,
    read_verdict,
    run_judge,
    run_judge_batch,
    run_judge_guidance,
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


class TestBuildGuidanceJudgePrompt:
    def test_contains_guidance_context_paths_and_score_path(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Build a workbook",
            final_output="Done!",
            judge_guidance="Score formulas and final state.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        assert "Build a workbook" in prompt
        assert "Done!" in prompt
        assert "Score formulas and final state." in prompt
        assert "/workspace/gandalf_trajectory.json" in prompt
        assert "/workspace/guidance_score.json" in prompt
        assert '"score"' in prompt
        assert '"reasoning"' in prompt
        assert '"evidence"' in prompt

    def test_instructs_judge_to_balance_artifact_and_trajectory_evidence(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Build a workbook",
            final_output="Done!",
            judge_guidance="Grade formulas, source data, and email actions.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "derive a short checklist" in lowered
        assert "workspace/artifact" in lowered
        assert "trajectory" in lowered
        assert "source-grounded" in lowered
        assert "positive and negative" in lowered

    def test_instructs_judge_to_apply_guidance_score_caps(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Build a workbook",
            final_output="Done!",
            judge_guidance="Hard penalty: fabricated data caps the score at 0.5.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "score bands" in lowered
        assert "hard penalties" in lowered
        assert "strictest applicable cap" in lowered
        assert "do not exceed that cap" in lowered

    def test_instructs_judge_to_place_scores_lower_in_band_for_foundational_failures(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Compute the core forecast and confirm or correct a hypothesis.",
            final_output="Done!",
            judge_guidance="0.4-0.6 gets the structure right but misses key analytical requirements.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "do not default to the top of an applicable band" in lowered
        assert "foundational calculation" in lowered
        assert "downstream dependent outputs" in lowered
        assert "explicit confirm/correct" in lowered
        assert "reserve scores near the top of a band" in lowered

    def test_instructs_judge_to_use_guidance_not_rubric_terminology(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Grade the final workbook.",
            final_output="Done!",
            judge_guidance="Known failure mode: wrong source caps score at 0.7.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "this is guidance-mode grading" in lowered
        assert "refer to these instructions as grading guidance" in lowered
        assert "do not call them a rubric" in lowered

    def test_instructs_judge_to_surface_source_guidance_conflicts(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Grade the category summary against source data.",
            final_output="Done!",
            judge_guidance="Expected category totals are listed below.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "source/guidance conflict audit" in lowered
        assert "accessible source data conflicts" in lowered
        assert "do not silently penalize source-grounded work" in lowered
        assert "explain which source you treated as authoritative" in lowered

    def test_instructs_judge_to_distinguish_uncached_source_formulas_from_posted_values(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Use the finance workbook as the book of record.",
            final_output="Done!",
            judge_guidance="Blank finance-book actuals should remain blank.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "formula cache/source-value audit" in lowered
        assert "formulas without cached values" in lowered
        assert "do not recompute source-workbook formulas" in lowered
        assert "posted or displayed book-of-record actuals" in lowered

    def test_instructs_judge_to_write_explicit_cap_audit(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Build a workbook",
            final_output="Done!",
            judge_guidance="Fabricated expiration dates cap the score at 0.50.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "score calibration/cap audit" in lowered
        assert "maximum score allowed" in lowered
        assert "concrete numeric maximum" in lowered
        assert "avoid vague maximums" in lowered
        assert "if a plausible cap or hard penalty is not applied" in lowered
        assert "explain why it does not apply" in lowered

    def test_instructs_judge_to_place_scores_below_caps_for_foundational_failures(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create an email with a demand forecast and verify the staffing hypothesis.",
            final_output="Done!",
            judge_guidance="0.4-0.6 has structure with notable gaps. 0.6-0.8 requires accurate numbers.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "near a declared ceiling" in lowered
        assert "foundational requirement failures" in lowered
        assert "central quantitative output" in lowered
        assert "required external verification" in lowered
        assert "minor, non-core issues remain" in lowered
        assert "top quarter of a band" in lowered
        assert "requirements for the next higher band" in lowered
        assert "move the score to the middle or lower part" in lowered
        assert "numeric score must match that placement word" in lowered
        assert "do not call a score middle-of-band" in lowered
        assert "compute the midpoint" in lowered
        assert "at or below that midpoint" in lowered
        assert "missing required deliverables or artifacts" in lowered

    def test_instructs_judge_to_put_above_midpoint_justification_in_cap_audit(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create an email with a demand forecast and verify the staffing hypothesis.",
            final_output="Done!",
            judge_guidance="0.4-0.6 has structure with notable gaps. 0.6-0.8 requires accurate numbers.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "if you choose a score above the midpoint" in lowered
        assert "score calibration/cap audit" in lowered
        assert "above the midpoint" in lowered
        assert "because" in lowered
        assert "central requirements" in lowered

    def test_instructs_judge_to_reconcile_instruction_guidance_conflicts(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Save the workbook to /home/agent/workspace/deliverables.",
            final_output="Done!",
            judge_guidance="Required artifact: workbook in /tmp/output.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "task instructions are the primary user-facing contract" in lowered
        assert "if the task instructions and grading guidance conflict" in lowered
        assert "output location" in lowered
        assert "do not penalize" in lowered
        assert "task-instructed" in lowered
        assert "before applying any output-location penalty" in lowered
        assert "output-location conflict audit" in lowered

    def test_instructs_judge_to_audit_actions_and_side_effects(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create a report. Do not send email or create calendar events.",
            final_output="Done!",
            judge_guidance="No emails or calendar events should be created.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "action/side-effect audit" in lowered
        assert "required or forbidden external actions" in lowered
        assert "email drafts/sends" in lowered
        assert "calendar events" in lowered
        assert "trajectory" in lowered
        assert "your evidence must include" in lowered
        assert 'one "action/side-effect audit" evidence item' in lowered

    def test_instructs_judge_not_to_count_file_imitation_as_external_action(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Draft an Outlook email to billing@example.com. Save it as a draft; do not send.",
            final_output="I saved draft_email.eml in deliverables.",
            judge_guidance="Required output: an Outlook draft addressed to billing@example.com.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "file imitation" in lowered
        assert "does not by itself prove" in lowered
        assert "live external artifact" in lowered
        assert "outlook draft" in lowered

    def test_instructs_judge_to_audit_source_availability_before_penalizing_missing_context(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Use Shopify and QuickBooks only where the files clearly support the interpretation.",
            final_output="The Shopify and QuickBooks folders were empty, so I used the financial summary.",
            judge_guidance="Expected sources include Shopify/orders.xlsx and QuickBooks/bills.csv.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "source availability audit" in lowered
        assert "accessible to you" in lowered
        assert "not accessible" in lowered
        assert "ignored an accessible source" in lowered
        assert "always include" in lowered
        assert "whether expected sources are available or missing" in lowered

    def test_instructs_judge_not_to_hard_cap_for_inaccessible_supporting_sources(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions=(
                "Build the forecast from the annual plan. Use Shopify and QuickBooks only as supporting context "
                "where files clearly support the interpretation."
            ),
            final_output="The Shopify and QuickBooks folders were empty, so I used available source files.",
            judge_guidance="Expected sources include Shopify/orders.xlsx and QuickBooks/bills.csv.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "do not apply a missing-source cap" in lowered
        assert "inaccessible supporting source" in lowered
        assert "reasonable source search" in lowered
        assert "ignored an accessible source" in lowered

    def test_instructs_judge_to_audit_independent_source_verification(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Prepare exact revenue calculations.",
            final_output="Done.",
            judge_guidance="Verification requirement: independently verify numerical claims against source files.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert "source verification audit" in lowered
        assert "independent verification" in lowered
        assert "recomputed" in lowered

    def test_instructs_judge_that_required_source_verification_must_be_evidence_item(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Prepare exact revenue calculations.",
            final_output="Done.",
            judge_guidance="Use the accompanying bronze deliverable as the source of truth for exact figures.",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )
        lowered = prompt.lower()
        assert 'your evidence must include one "source verification audit" evidence item' in lowered
        assert "source-of-truth" in lowered

    def test_guidance_prompt_surfaces_mandatory_evidence_checklist(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Prepare exact revenue calculations and create an Outlook draft; do not send it.",
            final_output="Done.",
            judge_guidance=(
                "Expected sources include Shopify/orders.json and QuickBooks/bills.csv. "
                "Independently verify numerical claims against source files."
            ),
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/guidance_score.json",
        )

        lowered = prompt.lower()

        assert "mandatory evidence checklist" in lowered
        assert "use these exact evidence labels" in lowered
        assert '"source availability audit"' in lowered
        assert '"source verification audit"' in lowered
        assert '"action/side-effect audit"' in lowered
        assert '"score calibration/cap audit"' in lowered
        assert "do not rename these audit labels" in lowered

    def test_instructs_judge_to_use_trajectory_file_not_inlined_json(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="x",
            final_output="z",
            judge_guidance="g",
            trajectory_path="/workspace/gandalf_trajectory.json",
            score_path="/workspace/score.json",
        )
        assert "<trajectory_file>" in prompt
        assert "inspect" in prompt.lower()
        assert "/workspace/gandalf_trajectory.json" in prompt
        assert '"steps"' not in prompt

    def test_instructs_judge_to_use_cloned_workspace_path_for_artifacts(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Save the workbook to /home/agent/workspace/deliverables/report.xlsx.",
            final_output="Done.",
            judge_guidance="Grade the saved workbook.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "<workspace_path>" in prompt
        assert "/tmp/judge_workspace_abc" in prompt
        assert "cloned judge workspace" in lowered
        assert "relative to this workspace path" in lowered
        assert "original container paths" in lowered
        assert "/home/agent/workspace/deliverables" in prompt
        assert "workspace_path/deliverables" in lowered

    def test_instructs_judge_to_inventory_cloned_workspace_before_original_paths(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Save the workbook to /home/agent/workspace/deliverables/report.xlsx.",
            final_output="Done.",
            judge_guidance="Required artifact: workbook in /tmp/output.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "first inventory the cloned workspace path" in lowered
        assert "before checking original absolute paths" in lowered
        assert 'find "$workspace_path"' in lowered
        assert "do not declare an artifact missing or inaccessible" in lowered

    def test_instructs_judge_to_classify_output_types_before_format_specific_checks(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create a workbook and memo.",
            final_output="Done.",
            judge_guidance="Grade the workbook formulas and memo content.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "classify the deliverable types" in lowered
        assert "xlsx" in lowered
        assert "docx" in lowered
        assert "do not apply office-document-specific checks" in lowered

    def test_instructs_judge_to_use_batched_bounded_artifact_inspection(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create an Excel workbook.",
            final_output="Done.",
            judge_guidance="Check all variance formulas and source rows.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "batched" in lowered
        assert "print findings or verdicts" in lowered
        assert "do not dump whole workbooks" in lowered
        assert "bounded" in lowered

    def test_instructs_judge_to_use_rendered_and_formula_evidence_for_office_documents(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create a spreadsheet and slide deck.",
            final_output="Done.",
            judge_guidance="Check formulas, charts, and layout.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "rendered screenshot" in lowered
        assert "visual or layout" in lowered
        assert "formula text" in lowered
        assert "cached values" in lowered

    def test_instructs_judge_to_apply_penalties_only_when_conditions_match(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create a markdown summary and a CSV export.",
            final_output="Done.",
            judge_guidance="Penalty rules: apply X1 for spreadsheet formula errors and P1 for slide-master changes.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "only apply penalty rules when their stated conditions match" in lowered
        assert "do not penalize choices the task explicitly requested" in lowered
        assert "not a hypothetical ideal" in lowered

    def test_instructs_judge_to_distinguish_reference_files_from_outputs(self) -> None:
        prompt = build_guidance_judge_prompt(
            instructions="Create a workbook using the provided input template.",
            final_output="Done.",
            judge_guidance="Compare against the golden workbook and source files.",
            trajectory_path="/tmp/judge_workspace_abc/gandalf_trajectory.json",
            score_path="/tmp/judge_workspace_abc/guidance_score.json",
            workspace_path="/tmp/judge_workspace_abc",
        )
        lowered = prompt.lower()
        assert "distinguish task inputs, source files, reference files, and golden files from agent outputs" in lowered
        assert "reference or golden files are not agent-produced outputs" in lowered
        assert "compare the output against them" in lowered

    def test_custom_template_receives_guidance_variables(self) -> None:
        template = "{{ instructions }}|{{ final_output }}|{{ judge_guidance }}|{{ trajectory_path }}|{{ score_path }}"
        prompt = build_guidance_judge_prompt(
            instructions="i",
            final_output="o",
            judge_guidance="g",
            trajectory_path="/t.json",
            score_path="/s.json",
            judge_prompt=template,
        )
        assert prompt == "i|o|g|/t.json|/s.json"


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


class TestReadGuidanceScore:
    def test_valid_score(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )
        result = read_guidance_score(str(p))
        assert result.score == 0.75
        assert result.reasoning == "Mostly correct."
        assert result.evidence == [
            "Workspace/artifact check: read /workspace/report.md.",
            "Inspected trajectory file: final command succeeded.",
            "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
        ]

    def test_missing_workspace_artifact_evidence_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "workspace/artifact" in result.reasoning.lower()

    def test_source_data_read_does_not_count_as_workspace_artifact_evidence(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Read source data at /workspace/data/orders.csv.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "workspace/artifact" in result.reasoning.lower()

    def test_inaccessible_original_path_does_not_count_as_workspace_artifact_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: in the current judge filesystem, /home/agent/workspace/deliverables and /workdir were not present, so I could not directly open the final XLSX from disk in this environment.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "workspace/artifact" in result.reasoning.lower()

    def test_output_location_audit_does_not_replace_workspace_artifact_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: in the current judge filesystem, /home/agent/workspace/deliverables and /workdir were not present, so I could not directly open the final XLSX from disk in this environment.",
                        "Output-location conflict audit: trajectory step 42 created /home/agent/workspace/deliverables/report.xlsx, so I treated the task-instructed location as acceptable.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_output_location_conflict_audit=True)

        assert result.score is None
        assert "workspace/artifact" in result.reasoning.lower()

    def test_concrete_missing_artifact_search_counts_as_workspace_artifact_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        evidence = [
            "Workspace/artifact check: searched /tmp/judge_workspace_abc/deliverables and found no generated cost-estimate workbook or written summary file; only source files and logs were present.",
            "Inspected trajectory file: final command succeeded.",
            "Score calibration/cap audit: missing artifacts cap the score at 0.6.",
        ]
        p.write_text(
            json.dumps(
                {
                    "score": 0.54,
                    "reasoning": "Analysis was partly correct but required artifacts are missing.",
                    "evidence": evidence,
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score == 0.54
        assert result.evidence == evidence

    def test_missing_trajectory_evidence_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "trajectory" in result.reasoning.lower()

    def test_command_output_does_not_count_as_trajectory_evidence(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Ran a command that printed workbook sheet names.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "trajectory" in result.reasoning.lower()

    def test_inaccessible_trajectory_file_does_not_count_as_trajectory_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Trajectory check: gandalf_trajectory.json was not accessible in the judge workspace.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "trajectory" in result.reasoning.lower()

    def test_missing_score_calibration_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": ["Checked report.md"],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "score calibration/cap audit" in result.reasoning.lower()

    def test_reports_multiple_validation_failures_at_once(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        evidence = ["Checked a generated report without concrete file, tool-use, source, or cap details."]
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": evidence,
                }
            )
        )

        result = read_guidance_score(
            str(p),
            require_source_availability_audit=True,
            require_source_verification_audit=True,
        )

        lowered = result.reasoning.lower()
        assert result.score is None
        assert "score calibration/cap audit" in lowered
        assert "workspace/artifact" in lowered
        assert "trajectory" in lowered
        assert "source availability audit" in lowered
        assert "source verification audit" in lowered
        assert result.evidence == evidence

    def test_accepts_raw_connector_json_source_verification_when_required(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        evidence = [
            "Workspace/artifact check: found deliverables/golden_spread_pricing_markdown_plan.xlsx plus pricing_markdown_brief.txt under the captured workspace deliverables directory.",
            "Trajectory check: steps 12-51 show Shopify and Square connector exports and local workbook generation.",
            "Source availability audit: accessible sources included raw API JSON dumps in deliverables/data for Shopify orders/inventory and Square orders/payments.",
            "Source verification audit: from raw Shopify and Square JSON I recomputed latest paid non-cancelled Shopify order date as 2026-04-18 and latest completed Square sale date as 2026-04-19.",
            "Score calibration/cap audit: required artifacts exist and maximum score allowed is 1.0.",
        ]
        p.write_text(
            json.dumps(
                {
                    "score": 0.66,
                    "reasoning": "Substantial but incomplete task success.",
                    "evidence": evidence,
                }
            )
        )

        result = read_guidance_score(
            str(p),
            require_source_availability_audit=True,
            require_source_verification_audit=True,
        )

        assert result.score == 0.66
        assert result.evidence == evidence

    def test_weak_score_calibration_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: reviewed the score.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "score calibration/cap audit" in result.reasoning.lower()

    def test_vague_score_calibration_audit_rejection_asks_for_parseable_maximum(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.62,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: the strictest practical ceiling is about the mid-0.6s, so 0.62 is appropriate.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        lowered = result.reasoning.lower()
        assert result.score is None
        assert "score calibration/cap audit" in lowered
        assert "parseable" in lowered
        assert "concrete numeric maximum" in lowered

    def test_bare_no_cap_score_calibration_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "score calibration/cap audit" in result.reasoning.lower()

    def test_score_above_declared_calibration_maximum_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.75,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: the strictest applicable cap is 0.60.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score is None
        assert "exceeds" in result.reasoning.lower()
        assert "0.6" in result.reasoning.lower()

    def test_score_at_declared_calibration_maximum_is_valid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.6,
                    "reasoning": "Mostly correct.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: the strictest applicable cap is 0.60.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score == 0.6

    def test_near_ceiling_score_with_foundational_failures_needs_above_midpoint_justification(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.58,
                    "reasoning": (
                        "The central quantitative output is materially wrong and the required external "
                        "verification was missed, but the artifact exists."
                    ),
                    "evidence": [
                        "Workspace/artifact check: read /workspace/email_draft.txt.",
                        "Inspected trajectory file: draft creation was present and no send action occurred.",
                        "Score calibration/cap audit: foundational requirement failures remain, so the maximum score allowed is 0.60.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        lowered = result.reasoning.lower()
        assert result.score is None
        assert "near declared ceiling" in lowered
        assert "foundational" in lowered
        assert "above-midpoint" in lowered

    def test_near_ceiling_score_with_foundational_failures_accepts_explicit_above_midpoint_justification(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.58,
                    "reasoning": (
                        "The central quantitative output is materially wrong, but all other central "
                        "requirements were satisfied."
                    ),
                    "evidence": [
                        "Workspace/artifact check: read /workspace/email_draft.txt.",
                        "Inspected trajectory file: draft creation was present and no send action occurred.",
                        (
                            "Score calibration/cap audit: foundational requirement failures remain, so the "
                            "maximum score allowed is 0.60. Above-midpoint justification: the required draft "
                            "action, recipient envelope, all four content sections, supplier normalization, "
                            "and actionable data-hygiene requirements were fully satisfied despite the "
                            "central forecast miss."
                        ),
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        assert result.score == 0.58

    def test_near_ceiling_score_with_missing_required_artifacts_needs_above_midpoint_justification(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.58,
                    "reasoning": (
                        "The response did not produce the required Excel workbook or separate written-summary "
                        "artifact, but it used the source data well."
                    ),
                    "evidence": [
                        (
                            "Workspace/artifact check: searched /workspace/deliverables and found no generated "
                            "deliverable workbook, markdown/text/doc/pdf summary, or output directory."
                        ),
                        "Inspected trajectory file: the final step contains only a textual answer.",
                        "Score calibration/cap audit: no Excel workbook/separate summary artifacts cap the maximum score allowed at 0.60.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        lowered = result.reasoning.lower()
        assert result.score is None
        assert "near declared ceiling" in lowered
        assert "above-midpoint" in lowered

    def test_high_score_with_foundational_failures_needs_above_midpoint_justification(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.58,
                    "reasoning": (
                        "The central Shopify sales methodology is materially wrong, and multiple central "
                        "quantitative requirements fail despite a useful workbook."
                    ),
                    "evidence": [
                        "Workspace/artifact check: read /workspace/deliverables/review.xlsx.",
                        "Inspected trajectory file: workbook creation command was present.",
                        (
                            "Score calibration/cap audit: the wrong revenue basis and missing buy-plan "
                            "comparisons make the maximum score allowed 0.65."
                        ),
                    ],
                }
            )
        )

        result = read_guidance_score(str(p))

        lowered = result.reasoning.lower()
        assert result.score is None
        assert "foundational" in lowered
        assert "above-midpoint" in lowered

    @pytest.mark.parametrize(
        ("payload", "expected_reason"),
        [
            ({"score": 1, "reasoning": "complete"}, "evidence"),
            ({"score": 1, "reasoning": "complete", "evidence": []}, "evidence"),
            ({"score": 1, "reasoning": "complete", "evidence": "Checked report.md"}, "evidence"),
            ({"score": 1, "reasoning": "", "evidence": ["Checked report.md"]}, "reasoning"),
            ({"score": 1, "evidence": ["Checked report.md"]}, "reasoning"),
        ],
    )
    def test_missing_audit_fields_are_invalid(
        self, payload: dict[str, Any], expected_reason: str, tmp_path: pathlib.Path
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(json.dumps(payload))
        result = read_guidance_score(str(p))
        assert result.score is None
        assert expected_reason in result.reasoning.lower()

    @pytest.mark.parametrize(
        ("payload", "expected_reason"),
        [
            ({"reasoning": "missing"}, "missing"),
            ({"score": "high", "reasoning": "bad"}, "numeric"),
            ({"score": True, "reasoning": "bad"}, "numeric"),
            ({"score": -0.01, "reasoning": "bad"}, "range"),
            ({"score": 1.01, "reasoning": "bad"}, "range"),
        ],
    )
    def test_invalid_score_shapes(self, payload: dict[str, Any], expected_reason: str, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(json.dumps(payload))
        result = read_guidance_score(str(p))
        assert result.score is None
        assert expected_reason in result.reasoning.lower()

    def test_invalid_json(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text("not json")
        result = read_guidance_score(str(p))
        assert result.score is None
        assert "invalid JSON" in result.reasoning

    def test_missing_required_action_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file for command failures",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_action_side_effect_audit=True)

        assert result.score is None
        assert "action/side-effect audit" in result.reasoning.lower()

    def test_weak_required_action_audit_label_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Action/side-effect audit: checked expected external actions before grading.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_action_side_effect_audit=True)

        assert result.score is None
        assert "action/side-effect audit" in result.reasoning.lower()

    def test_required_action_audit_accepts_named_evidence_item(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Action/side-effect audit: inspected trajectory; no send-email or calendar-event calls occurred.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_action_side_effect_audit=True)

        assert result.score == 0.8

    def test_required_action_audit_accepts_trajectory_forbidden_action_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Trajectory check: inspected gandalf_trajectory.json; no Shopify/QuickBooks updates, emails, calendar events, or other forbidden side effects were observed.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_action_side_effect_audit=True)

        assert result.score == 0.8

    def test_missing_required_output_location_conflict_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_output_location_conflict_audit=True)

        assert result.score is None
        assert "output-location conflict audit" in result.reasoning.lower()

    def test_required_output_location_conflict_audit_accepts_named_evidence_item(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Output-location conflict audit: task requested /home/agent/workspace/deliverables; guidance named /tmp/output; actual artifact was task-instructed and accessible.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_output_location_conflict_audit=True)

        assert result.score == 0.8

    def test_weak_required_output_location_conflict_audit_label_is_invalid(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Output-location conflict audit: checked the output path before grading.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_output_location_conflict_audit=True)

        assert result.score is None
        assert "output-location conflict audit" in result.reasoning.lower()

    def test_missing_required_source_availability_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        evidence = [
            "Workspace/artifact check: read /workspace/report.md.",
            "Inspected trajectory file: final command succeeded.",
            "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
        ]
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": evidence,
                }
            )
        )

        result = read_guidance_score(str(p), require_source_availability_audit=True)

        assert result.score is None
        assert "source availability audit" in result.reasoning.lower()
        assert result.evidence == evidence

    def test_weak_required_source_availability_audit_label_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source availability audit: checked expected sources before grading.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_availability_audit=True)

        assert result.score is None
        assert "source availability audit" in result.reasoning.lower()

    def test_required_source_availability_audit_accepts_named_evidence_item(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source availability audit: Shopify exports were not accessible; /workspace/plan.csv was accessible.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_availability_audit=True)

        assert result.score == 0.8

    def test_required_source_availability_audit_accepts_accessibility_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source availability audit: confirmed expected source workbooks exist under Operations; Shopify exports were missing.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_availability_audit=True)

        assert result.score == 0.8

    def test_required_source_availability_audit_accepts_trajectory_source_directory_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source availability audit: trajectory step 7 listed /home/agent/workspace/tool-outputs/shopify and quickbooks_sql source directories as empty.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_availability_audit=True)

        assert result.score == 0.8

    def test_missing_required_source_verification_audit_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_verification_audit=True)

        assert result.score is None
        assert "source verification audit" in result.reasoning.lower()

    def test_weak_required_source_verification_audit_label_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source verification audit: checked the source data and compared it to the artifact.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_verification_audit=True)

        assert result.score is None
        assert "source verification audit" in result.reasoning.lower()

    def test_required_source_verification_audit_accepts_named_evidence_item(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source verification audit: recomputed source totals from /workspace/orders.csv and compared them to the workbook.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_verification_audit=True)

        assert result.score == 0.8

    def test_required_source_verification_audit_accepts_independent_verification_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Independent source verification with Python on /workspace/orders.csv recalculated revenue and compared it to the workbook.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_verification_audit=True)

        assert result.score == 0.8

    def test_required_source_verification_audit_accepts_script_output_evidence(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Solid result.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/report.md.",
                        "Source verification script output from orders.csv: 20 Q1 rows, gross $5,782.75, and net payout $5,141.46.",
                        "Inspected trajectory file: final command succeeded.",
                        "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    ],
                }
            )
        )

        result = read_guidance_score(str(p), require_source_verification_audit=True)

        assert result.score == 0.8

    def test_unlabeled_source_guidance_conflict_is_invalid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        p.write_text(
            json.dumps(
                {
                    "score": 0.5,
                    "reasoning": "Mixed result with source and guidance conflicts.",
                    "evidence": [
                        "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
                        "Trajectory check: gandalf_trajectory.json shows the agent created the workbook.",
                        (
                            "Source availability audit: the source file 2026_COGS_and_Margin.xlsx and "
                            "Financial/Budget_FY2026.xlsx were accessible and inspected."
                        ),
                        (
                            "Source verification audit: independently recomputed category totals from "
                            "/workspace/2026_COGS_and_Margin.xlsx."
                        ),
                        (
                            "Category source check: the captured COGS tracker values match the workbook but do not "
                            "meet the guidance's expected category figures."
                        ),
                        "Score calibration/cap audit: mixed source-grounding defects cap the score at 0.65.",
                    ],
                }
            )
        )

        result = read_guidance_score(
            str(p),
            require_source_availability_audit=True,
            require_source_verification_audit=True,
        )

        assert result.score is None
        assert "source/guidance conflict audit" in result.reasoning.lower()

    def test_named_source_guidance_conflict_audit_is_valid(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "score.json"
        evidence = [
            "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
            "Trajectory check: gandalf_trajectory.json shows the agent created the workbook.",
            (
                "Source availability audit: the source file 2026_COGS_and_Margin.xlsx and "
                "Financial/Budget_FY2026.xlsx were accessible and inspected."
            ),
            (
                "Source verification audit: independently recomputed category totals from "
                "/workspace/2026_COGS_and_Margin.xlsx."
            ),
            (
                "Source/guidance conflict audit: the captured tracker source values match the workbook "
                "but do not meet the guidance's expected category figures; I treated the guidance's "
                "golden checkpoints as authoritative for scoring because it names them as expected values."
            ),
            "Score calibration/cap audit: mixed source-grounding defects cap the score at 0.65.",
        ]
        p.write_text(
            json.dumps(
                {
                    "score": 0.5,
                    "reasoning": "Mixed result with a reconciled source/guidance conflict.",
                    "evidence": evidence,
                }
            )
        )

        result = read_guidance_score(
            str(p),
            require_source_availability_audit=True,
            require_source_verification_audit=True,
        )

        assert result.score == 0.5
        assert result.evidence == evidence


class TestRequiresSourceAvailabilityAudit:
    def test_requires_audit_when_guidance_names_expected_source_files(self) -> None:
        assert requires_source_availability_audit(
            "Use Shopify and QuickBooks only where the files clearly support the interpretation.",
            "Expected sources include Shopify/orders.xlsx, QuickBooks/bills.csv, and Files/Home/plan.csv.",
        )

    def test_requires_audit_when_task_uses_connector_data_as_source(self) -> None:
        assert requires_source_availability_audit(
            "Use Shopify, Faire, and QuickBooks as supporting context where files clearly support it.",
            "Grade whether the analysis is source-grounded.",
        )

    def test_requires_audit_when_guidance_names_source_colon_files(self) -> None:
        assert requires_source_availability_audit(
            "Prepare the inventory plan.",
            "Trend factors must be checked. Source: inventory_ledger.csv aggregated for both years.",
        )

    def test_does_not_require_audit_for_output_file_paths_only(self) -> None:
        assert not requires_source_availability_audit(
            "Save the finished workbook to /home/agent/workspace/deliverables/report.xlsx.",
            "Grade the final artifact.",
        )

    def test_does_not_require_audit_for_external_action_only(self) -> None:
        assert not requires_source_availability_audit(
            "Create a report. Do not send email or create calendar events.",
            "Verify no live action happened.",
        )

    def test_foundational_failure_language_ignores_missing_source_data_caveat(self) -> None:
        assert not has_foundational_failure_language(
            "The workbook correctly flags missing recipe COGS rather than treating it as true healthy margin."
        )

    def test_foundational_failure_language_detects_missing_required_outputs(self) -> None:
        assert has_foundational_failure_language(
            "The response did not produce the required Excel workbook or separate written-summary artifact."
        )

    def test_has_above_midpoint_justification_accepts_midpoint_because_evidence(self) -> None:
        assert has_above_midpoint_justification(
            [
                (
                    "Score calibration/cap audit: I score 0.58, above the midpoint of the 0.41-0.60 band "
                    "only because the artifact is complete, source-grounded in many areas, and includes "
                    "budget, QuickBooks, HR, concentration, limitations, and recommendations."
                ),
            ]
        )

    def test_has_source_availability_audit_detects_named_evidence(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: Shopify was unavailable, while /workspace/source.csv was accessible.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_weak_named_evidence(self) -> None:
        assert not has_source_availability_audit(
            [
                "Source availability audit: checked expected sources before grading.",
            ]
        )

    def test_has_source_availability_audit_requires_named_audit_item(self) -> None:
        assert not has_source_availability_audit(
            [
                "Raw finance-source XML check: Copy of Jade Leaf LLC. Finances.xlsx August cells I3 and I6 "
                "contain formulas but no cached values.",
                "CSV verification: Labor_Sales/2025 labor v sales.csv has August Labor Cost $8,227.13.",
            ]
        )

    def test_has_source_availability_audit_detects_concrete_accessibility_evidence(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: confirmed expected source workbooks exist under Operations; the Shopify export was missing.",
            ]
        )

    def test_has_source_availability_audit_detects_workspace_containing_source_workbooks(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: /tmp/judge_workspace contains accessible source workbooks Operations/Delivery_Log_DAL_2024-2026.xlsx and Operations/Supplier_Log_DAL_2024-2026.xlsx.",
            ]
        )

    def test_has_source_availability_audit_detects_empty_tool_output_directories(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: trajectory step 7 listed /home/agent/workspace/tool-outputs/shopify and quickbooks_sql source directories as empty.",
            ]
        )

    def test_has_source_availability_audit_detects_no_named_source_export_found(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: initial rg --files listed annual plan files but no Shopify order-line export was found.",
            ]
        )

    def test_has_source_availability_audit_detects_no_expected_source_path_found(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: initial rg --files in /home/agent/workspace listed annual plan, fixed_asset_register.csv, and Faire files but no Shopify/orders.xlsx or orders.csv source export.",
            ]
        )

    def test_has_source_availability_audit_detects_not_present_expected_sources(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: expected CSVs such as shopify_products.csv, square_items.csv, qb_bills_all_20260429.csv, and qb_accounts_20260429.csv were not present as local files. Accessible sources included deliverables/data/shopify_orders.json and COGS_Recipe_Sheet.xlsx.",
            ]
        )

    def test_has_source_availability_audit_detects_sources_are_accessible(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: the plan CSV, Faire orders and payouts CSVs, captured Shopify export, generated workbook/email, and full trajectory are accessible in the cloned workspace.",
            ]
        )

    def test_has_source_availability_audit_detects_inverted_accessible_source_list(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: accessible in the cloned workspace were "
                "Financial/Astor_Budget_FY2025.xlsx, Merchandising/Buy_Plan_SS25.xlsx, "
                "Shopify orders/customers/products JSON exports, and TimeStation CSV files. "
                "QuickBooks bill data was not a standalone local file but was visible in the "
                "trajectory through quickbooks_sql search calls."
            ]
        )

    def test_has_source_availability_audit_detects_source_files_were_present(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: inventory_ledger.csv, inventory_master.csv, cycle_counts.csv, Water_Test_Appointments.csv, SKU_Cost_Master.csv, Campaigns.csv, Demand_Forecast_2026.xlsx, Budgets.xlsx, and related source files were present under the cloned workspace; source absence did not limit grading.",
            ]
        )

    def test_has_source_availability_audit_detects_independent_source_recomputation(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: retail_cafe_invoices_2025.csv was accessible and opened with Python; retail cafe invoices by Invoice Date were Jan $3,889.45, Feb $5,432.25, and Mar $7,448.47.",
            ]
        )

    def test_has_source_availability_audit_detects_connector_named_independent_verification(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: accessible Shopify source export tool-outputs/shopify/workspace/data/shopify_orders_2025.json was opened; using paid/partially_refunded, non-test, non-cancelled orders gave 2,961 orders and $462,401.04 subtotal revenue.",
            ]
        )

    def test_has_source_availability_audit_detects_retrieved_connector_source_outputs(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: trajectory/source check found accessible QuickBooks account, invoice, bill, and payment tool outputs in steps 13-16 and 20-25; step 28 exported Shopify orders; steps 10-11 viewed Faire orders and payouts exports.",
            ]
        )

    def test_has_source_availability_audit_detects_trajectory_check_with_source_exports(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: trajectory check confirmed QuickBooks account, invoice, bill, and payment source searches in steps 14-17 were accessible, step 29 exported Shopify February orders, and steps 11-12 viewed Faire orders and payouts files.",
            ]
        )

    def test_has_source_availability_audit_detects_trajectory_inspected_source_files(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: trajectory steps 4-38 show the agent inspected accessible source files, PDFs, finance/budget/goal workbooks, labor/sales reports, item sales, discounts, pricing, and inventory files.",
            ]
        )

    def test_has_source_availability_audit_detects_trajectory_explored_source_files(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: gandalf_trajectory.json has 56 steps; the agent inspected accessible source files and generated deliverables.",
            ]
        )

    def test_has_source_availability_audit_detects_trajectory_analytical_command_with_named_source(
        self,
    ) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: trajectory analytical command step 25 opened accessible Supplier_Pricing_History source category costs such as $12.70/$26.33 and claimed high margins.",
            ]
        )

    def test_has_source_availability_audit_detects_extracted_source_workbook(self) -> None:
        assert has_source_availability_audit(
            [
                "Source availability audit: trajectory step 17 opened accessible source workbook Copy of Jade Leaf LLC. Finances.xlsx with openpyxl data_only=True and showed August values blank for In-Store Cafe Sales.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_generic_source_reads(self) -> None:
        assert not has_source_availability_audit(
            [
                "Read source data at /workspace/data/orders.csv.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_vague_used_source_claim(self) -> None:
        assert not has_source_availability_audit(
            [
                "The response used source data for the analysis.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_no_source_requirement(self) -> None:
        assert not has_source_availability_audit(
            [
                "No source files should be required for this task.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_external_action_absence(self) -> None:
        assert not has_source_availability_audit(
            [
                "Action/side-effect audit: inspected trajectory and found no Shopify updates.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_source_connector_action_absence(self) -> None:
        assert not has_source_availability_audit(
            [
                "Trajectory check: tool calls were Shopify export_orders/customers/products; no Shopify updates were observed.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_missing_artifact_references(self) -> None:
        assert not has_source_availability_audit(
            [
                "Negative supporting-source check: grepping the workbook found no references to Shopify/orders.xlsx or QuickBooks/bills.csv.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_score_calibration_source_mentions(self) -> None:
        assert not has_source_availability_audit(
            [
                "Score calibration/cap audit: no hard caps for wrong revenue source, missing ranking, or missing plan comparison apply because the workbook includes directional context.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_connector_tool_call_list_without_status(self) -> None:
        assert not has_source_availability_audit(
            [
                "Trajectory check: gandalf_trajectory.json shows Shopify/Square export/list calls and QuickBooks search calls; no evidence of send/post/publish actions was found.",
            ]
        )

    def test_has_source_availability_audit_does_not_accept_unchecked_expectations(self) -> None:
        assert not has_source_availability_audit(
            [
                "Expected source files should be available for grading.",
            ]
        )


class TestRequiresSourceVerificationAudit:
    def test_requires_audit_when_guidance_requires_independent_verification(self) -> None:
        assert requires_source_verification_audit(
            "Build a workbook with exact revenue calculations.",
            "Verification requirement: independently verify numerical claims against source files.",
        )

    def test_requires_audit_when_guidance_says_recompute_from_source_of_truth(self) -> None:
        assert requires_source_verification_audit(
            "Prepare a management summary.",
            "Use source files as the source of truth for all exact figures and recompute key totals.",
        )

    def test_requires_audit_when_guidance_says_artifacts_checked_against_source_records(self) -> None:
        assert requires_source_verification_audit(
            "Create a pricing review workbook.",
            (
                "Use the source records as the source of truth. The model's workbook, final response, "
                "or any other artifact should be checked against the source records, not treated as "
                "the source of truth."
            ),
        )

    def test_requires_audit_when_guidance_uses_golden_deliverable_as_source_of_truth(self) -> None:
        assert requires_source_verification_audit(
            "Prepare a planning diagnostic workbook.",
            (
                "Use the accompanying golden deliverable file as the source of truth for all exact figures, "
                "detailed tables, and workbook structure. The golden deliverable tells you what the correct "
                "answers are."
            ),
        )

    def test_requires_audit_when_guidance_uses_deliverable_as_source_of_truth_for_exact_figures(self) -> None:
        assert requires_source_verification_audit(
            "Prepare a monthly variance review workbook.",
            "Use the accompanying bronze deliverable as the source of truth for exact figures.",
        )

    def test_requires_audit_when_guidance_requires_tie_out_or_tolerance_checks(self) -> None:
        assert requires_source_verification_audit(
            "Prepare a Q1 performance review.",
            "Dollar amounts accept within +/-2%; Q1 gross sales must tie out across Square and the COGS file.",
        )

    def test_does_not_require_audit_for_generic_source_availability(self) -> None:
        assert not requires_source_verification_audit(
            "Use Shopify and QuickBooks only where files clearly support the interpretation.",
            "Expected sources include Shopify/orders.xlsx and QuickBooks/bills.csv.",
        )

    def test_does_not_require_audit_for_generic_source_grounding_language(self) -> None:
        assert not requires_source_verification_audit(
            "Prepare a management summary.",
            "Grade whether the response is source-grounded and avoids unsupported claims.",
        )

    def test_has_source_verification_audit_detects_named_evidence(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification audit: independently recomputed totals from /workspace/orders.csv and matched workbook revenue $1,234.56.",
            ]
        )

    def test_source_guidance_conflict_language_detects_guidance_mismatch(self) -> None:
        assert has_source_guidance_conflict_language(
            "The captured COGS tracker values match the workbook but do not meet the guidance's expected category figures."
        )

    def test_source_guidance_conflict_audit_requires_named_authority_decision(self) -> None:
        assert has_source_guidance_conflict_audit(
            [
                (
                    "Source/guidance conflict audit: the captured tracker source values match the workbook "
                    "but do not meet the guidance's expected category figures; I treated the guidance's golden "
                    "checkpoints as authoritative for scoring."
                ),
            ]
        )
        assert not has_source_guidance_conflict_audit(
            [
                "Source/guidance conflict audit: the captured tracker source values do not meet the guidance figures.",
            ]
        )

    def test_source_guidance_conflict_audit_accepts_no_conflict_found(self) -> None:
        assert has_source_guidance_conflict_audit(
            [
                (
                    "Source/guidance conflict audit: no substantive conflict was found between the accessible "
                    "CSV-converted source workbooks and the guidance golden values for the scored checkpoints; "
                    "the raw Prairie Gas Co count appears as 47 in the CSV, within the guidance's stated tolerance "
                    "for 46 vs 47."
                ),
            ]
        )

    def test_source_guidance_conflict_audit_accepts_guidance_acceptance_decision(self) -> None:
        assert has_source_guidance_conflict_audit(
            [
                (
                    "Source/guidance conflict audit: my numeric-prefix extraction produced Q1 2026 = "
                    "15,516.35 rather than the golden exclusion anchor 15,459.95, but the guidance "
                    "explicitly accepts Q1 2026 values in the 15,460-15,585 range depending on "
                    "non-numeric handling. This does not excuse the agent's Q1 2025 and Q2 2025 "
                    "errors, because those guidance figures are unaffected and matched direct source parsing."
                ),
            ]
        )

    def test_has_source_verification_audit_detects_recomputed_results(self) -> None:
        assert has_source_verification_audit(
            [
                (
                    "Source verification audit: independently recomputed Shopify FY2025 included orders from "
                    "source JSON using financial_status paid or partially_refunded and not cancelled. "
                    "Results: 2,961 orders, subtotal_price net merchandise revenue $462,401.04."
                ),
            ]
        )

    def test_has_source_verification_audit_detects_independent_source_verification_evidence(self) -> None:
        assert has_source_verification_audit(
            [
                "Independent source verification with Python on /workspace/orders.csv calculated 2,961 orders and compared them to the workbook.",
            ]
        )

    def test_has_source_verification_audit_detects_connector_named_independent_verification(self) -> None:
        assert has_source_verification_audit(
            [
                "Independent Shopify verification from tool-outputs/shopify/workspace/data/shopify_orders_2025.json using paid/partially_refunded, non-test, non-cancelled orders gave 2,961 orders and $462,401.04 subtotal revenue.",
            ]
        )

    def test_has_source_verification_audit_detects_independent_source_calculation(self) -> None:
        assert has_source_verification_audit(
            [
                "Independent source calculation from 2025_year_end_summary.csv: TOTAL REVENUE row is Q1 22,648.48, Q2 36,661.08, Q3 23,036.36, Q4 67,422.88, annual 149,768.80.",
            ]
        )

    def test_has_source_verification_audit_detects_script_output_evidence(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification script output from faire-payouts-summary.csv: 20 Q1 payout rows, gross $5,782.75, net payout $5,141.46.",
            ]
        )

    def test_has_source_verification_audit_detects_verifier_script_totals_without_source_filename(self) -> None:
        assert has_source_verification_audit(
            [
                "Independent source verification with verify_sources.py: margin tracker Q1 totals are 315 rows, gross $193,175.25, cost $153,756.96, and budget source Total Revenue Q1 is $196,964.00.",
            ]
        )

    def test_has_source_verification_audit_detects_bronze_deliverable_source_truth(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification audit: compared workbook values against the bronze deliverable source of truth and confirmed POS actual $343.00, Online actual $1,645.25, Faire net $222.98, and Total Revenue $6,888.38.",
            ]
        )

    def test_has_source_verification_audit_detects_source_truth_checkpoint_comparison(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification audit: I compared the delivered calculations and shortlist against the grading source-of-truth checkpoints and inspected the agent's source choices in trajectory. The expected push-now shortlist is SPK-HUDS-PETN-2022, WOW-WEIN-GRNE-2019, WOW-PAGO-VERD-2021, ROS-MANO-TAVE-2020, RNW-TEMA-PINO-2023, and RNW-CATS-PINO-2020 with a 5-6% cap from Verdejo; the delivered shortlist/margin math uses supplier cost and a 15% tier, so the central source-backed margin and SKU findings are not verified or correct.",
            ]
        )

    def test_has_source_verification_audit_detects_cogs_tracker_and_budget_recomputation(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification audit: my verify_metrics.py recomputation from the accessible COGS tracker returned Q1 315 rows, gross $193,175.25, bought price $153,756.96, commission $39,050.14; the FY2026 budget recomputation returned Q1 Total Revenue $196,964.00, Total COGS $156,920.00, Gross Profit $40,044.00, and April Total Revenue plan $69,177.60.",
            ]
        )

    def test_has_source_verification_audit_detects_independent_source_check(self) -> None:
        assert has_source_verification_audit(
            [
                "Independent source check with Python on the invoice CSVs: retail cafe invoices by Invoice Date were Jan $3,889.45, Feb $5,432.25, and Mar $7,448.47.",
            ]
        )

    def test_has_source_verification_audit_detects_financial_source_check_with_file(self) -> None:
        assert has_source_verification_audit(
            [
                "Financial source check: historical_pnl_2021_2025.csv contains FY 2025 discounts of -$13,641.35 and returns of $978.08.",
            ]
        )

    def test_has_source_verification_audit_detects_source_check_with_python(self) -> None:
        assert has_source_verification_audit(
            [
                "Source check with Python: historical_pnl_2021_2025.csv contains FY 2025 Total discounts of -13641.35 and returns of 978.08.",
            ]
        )

    def test_has_source_verification_audit_detects_independently_checked_source_csvs(self) -> None:
        assert has_source_verification_audit(
            [
                "Independently checked available source CSVs with Python: 2026_annual_plan.csv gives Q1 total target $34,555.00 and Faire target $5,335.00.",
            ]
        )

    def test_has_source_verification_audit_detects_raw_connector_json_description(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification audit: from raw Shopify and Square JSON I recomputed latest paid non-cancelled Shopify order date as 2026-04-18 and latest completed Square sale date as 2026-04-19.",
            ]
        )

    def test_has_source_verification_audit_detects_raw_order_json_description(self) -> None:
        assert has_source_verification_audit(
            [
                "Source verification audit: recomputing from raw order JSON gave almond-butter-lover 15 Shopify units, 0 Square units, 15 total.",
            ]
        )

    def test_has_source_verification_audit_detects_csv_verification_with_source_path(self) -> None:
        assert has_source_verification_audit(
            [
                "CSV verification: Labor_Sales/2025 labor v sales.csv has August Labor Cost $8,227.13, Net Sales $20,179.93, and Labor Percentage 40.77%.",
            ]
        )

    def test_has_source_verification_audit_detects_raw_finance_source_xml_check(self) -> None:
        assert has_source_verification_audit(
            [
                "Raw finance-source XML check: in Copy of Jade Leaf LLC. Finances.xlsx, August cells I3, I6, I7, I10, and I11 contain formulas but no cached values.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_generic_source_grounding(self) -> None:
        assert not has_source_verification_audit(
            [
                "The final workbook is source-grounded and cites Shopify data.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_generic_source_check(self) -> None:
        assert not has_source_verification_audit(
            [
                "Source check: reviewed the workbook and it seemed plausible.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_source_choice_only(self) -> None:
        assert not has_source_verification_audit(
            [
                "Trajectory/source check: steps 13-16 and 20-25 used QuickBooks account, invoice, bill, and payment tools; step 28 exported Shopify orders.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_workspace_artifact_source_labels(self) -> None:
        assert not has_source_verification_audit(
            [
                "Workspace/artifact check: found /workspace/deliverables/report.xlsx (50,327 bytes) and parsed sheets including Channel_Source and Data_Sources_Limits.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_source_availability_searches(self) -> None:
        assert not has_source_verification_audit(
            [
                "Trajectory check: step 12's find command returned no files under /home/agent/workspace/tool-outputs and found only empty Shopify source directories.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_score_cap_source_mentions(self) -> None:
        assert not has_source_verification_audit(
            [
                "Score calibration/cap audit: wrong revenue source caps the score at 0.65 because source-grounding was incomplete.",
            ]
        )

    def test_has_source_verification_audit_does_not_accept_unchecked_expectations(self) -> None:
        assert not has_source_verification_audit(
            [
                "The workbook should be compared against source files.",
            ]
        )


class TestRequiresOutputLocationConflictAudit:
    def test_requires_audit_when_instruction_and_guidance_paths_differ(self) -> None:
        assert requires_output_location_conflict_audit(
            "Save the workbook to /home/agent/workspace/deliverables.",
            "Required artifact: workbook in /tmp/output.",
        )

    def test_does_not_require_audit_when_paths_match(self) -> None:
        assert not requires_output_location_conflict_audit(
            "Save the workbook to /tmp/output.",
            "Required artifact: workbook in /tmp/output.",
        )

    def test_does_not_require_audit_for_specific_file_inside_requested_directory(self) -> None:
        assert not requires_output_location_conflict_audit(
            "Save the workbook to /home/agent/workspace/deliverables.",
            "Required artifact: /home/agent/workspace/deliverables/report.xlsx.",
        )

    def test_ignores_schema_and_source_data_paths_for_output_conflicts(self) -> None:
        assert not requires_output_location_conflict_audit(
            "Save final files to /home/agent/workspace/deliverables.",
            (
                "Inspect source data at /Home/2025_year_end_summary.csv and /Fair/products.xlsx. "
                "Return JSON fields /score and /reasoning. "
                "The artifact path is /home/agent/workspace/deliverables/report.xlsx."
            ),
        )

    def test_does_not_require_audit_with_only_one_path_source(self) -> None:
        assert not requires_output_location_conflict_audit(
            "Save the workbook to /home/agent/workspace/deliverables.",
            "Grade the workbook holistically.",
        )

    def test_has_output_location_conflict_audit_detects_concrete_path_comparison(self) -> None:
        assert has_output_location_conflict_audit(
            [
                "Output-location conflict audit: task requested /home/agent/workspace/deliverables; guidance named /tmp/output; actual artifact was task-instructed and accessible.",
            ]
        )

    def test_has_output_location_conflict_audit_detects_external_artifact_location(self) -> None:
        assert has_output_location_conflict_audit(
            [
                "Output-location conflict audit: task and guidance required an Outlook Drafts email; trajectory verified the actual artifact location is Drafts and no workspace deliverable was required.",
            ]
        )

    def test_has_output_location_conflict_audit_does_not_accept_weak_named_evidence(self) -> None:
        assert not has_output_location_conflict_audit(
            [
                "Output-location conflict audit: checked the output path before grading.",
            ]
        )


class TestRequiresActionSideEffectAudit:
    def test_requires_audit_for_external_email_or_calendar_actions(self) -> None:
        assert requires_action_side_effect_audit(
            "Create a report. Do not send email or create calendar events.",
            "Verify no live action happened.",
        )

    def test_requires_audit_for_external_system_updates(self) -> None:
        assert requires_action_side_effect_audit(
            "Prepare a workbook only.",
            "Verify no Shopify update, Square update, or QuickBooks update occurred.",
        )

    def test_does_not_require_audit_for_supporting_schedules(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create an Excel workbook with supporting schedules and charts.",
            "The workbook should include a monthly demand schedule.",
        )

    def test_does_not_require_audit_for_submitted_as_quality_language(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create a manager-ready memo.",
            "Do not over-credit a model that submitted wrong figures or submits a technical summary.",
        )

    def test_does_not_require_audit_for_not_required_action_list(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create a workbook and save it to deliverables.",
            (
                "Not required: word document, PDF, email draft or sent email, calendar event, "
                "Shopify updates, QuickBooks updates, or budget edits."
            ),
        )

    def test_does_not_require_audit_for_prompt_does_not_ask_for_external_actions(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create an Excel workbook, Word summary, and management email draft in txt.",
            (
                "The prompt does not ask for an Outlook draft, sent email, PowerPoint deck, "
                "calendar event, Instagram payload, local pickup campaign, or wholesale partner outreach action."
            ),
        )

    def test_does_not_require_audit_for_no_external_artifact_required_sentence(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create a workbook and save it to deliverables.",
            "No word memo, email draft, sent email, PDF, calendar item, or system update is required.",
        )

    def test_does_not_require_audit_for_ground_truth_email_draft_reference(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create an Excel workbook, Word summary, and management email draft in txt.",
            "Do not use the submitted workbook, Word document, email draft, or any generated artifact as ground truth.",
        )

    def test_does_not_require_audit_for_text_draft_not_outlook_draft_disclaimer(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create an Excel workbook, Word summary, and management email draft in txt.",
            "It should not create or require an Outlook draft, because this prompt asks for a text email draft only.",
        )

    def test_does_not_require_audit_for_local_txt_email_draft_deliverable(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create an Excel workbook, a Word summary, and a management email draft in txt.",
            "A management email draft is created as a `.txt` file in `/home/agent/workspace/deliverables`.",
        )

    def test_does_not_require_audit_for_local_email_draft_rubric_section(self) -> None:
        assert not requires_action_side_effect_audit(
            "Create an Excel workbook, a Word summary, and a management email draft in txt.",
            (
                "The word summary and email draft are consistent with the workbook.\n\n"
                "Section 11: Word Summary and Management Email Draft\n"
                "Management email draft checks:\n"
                "- A `.txt` email draft exists in `/home/agent/workspace/deliverables`.\n"
                "- Creating an Outlook draft instead of the required `.txt` management email draft is wrong.\n"
                "If no `.txt` management email draft is produced, cap the score at 0.75.\n"
                "The word document and email draft should be consistent with the workbook.\n"
                "Strong work includes a concise management word summary, and an email draft that accurately "
                "summarizes the implications for management."
            ),
        )

    def test_has_action_side_effect_audit_detects_named_evidence(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Action/side-effect audit: inspected trajectory; no send-email or calendar-event calls occurred.",
            ]
        )

    def test_has_action_side_effect_audit_does_not_accept_weak_named_evidence(self) -> None:
        assert not has_action_side_effect_audit(
            [
                "Action/side-effect audit: checked expected external actions before grading.",
            ]
        )

    def test_has_action_side_effect_audit_detects_forbidden_action_absence(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Trajectory check: inspected gandalf_trajectory.json; no Shopify/QuickBooks updates, emails, calendar events, or other forbidden side effects were observed.",
            ]
        )

    def test_has_action_side_effect_audit_detects_no_live_mutations(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Trajectory check: no evidence of send/post/publish/schedule or live Shopify/Square/QuickBooks mutations was found.",
            ]
        )

    def test_has_action_side_effect_audit_detects_draft_state_evidence(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Action/side-effect audit: trajectory step 27 showed isDraft=true for the Outlook draft and no send-message call appeared.",
            ]
        )

    def test_has_action_side_effect_audit_detects_unlabeled_draft_rather_than_send_evidence(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Trajectory check: steps 14-17 used QuickBooks account searches, step 29 exported Shopify February orders, and step 54 called outlook__create_draft_message rather than a send function.",
            ]
        )

    def test_has_action_side_effect_audit_detects_unlabeled_no_email_send_evidence(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Trajectory check: no email send action or external email tooling was used; the agent created docx/txt draft files only, satisfying the draft-not-sent constraint.",
            ]
        )

    def test_has_action_side_effect_audit_detects_no_send_action_appears(self) -> None:
        assert has_action_side_effect_audit(
            [
                "Trajectory check: final creation step wrote /home/agent/workspace/deliverables/report.xlsx and draft files, then verified sheet names and email text; no Outlook send action appears.",
            ]
        )

    def test_has_action_side_effect_audit_does_not_accept_plain_email_content(self) -> None:
        assert not has_action_side_effect_audit(
            [
                "Email artifact check: parsed draft_email.txt and found a summary of Q1 actuals.",
            ]
        )

    def test_has_action_side_effect_audit_does_not_accept_source_export_calls_only(self) -> None:
        assert not has_action_side_effect_audit(
            [
                "Trajectory check: Shopify export_orders and QuickBooks search_bills calls were used as sources.",
            ]
        )

    def test_has_action_side_effect_audit_does_not_accept_unchecked_expectations(self) -> None:
        assert not has_action_side_effect_audit(
            [
                "The agent should not send email or update Shopify.",
            ]
        )


class TestScoreCalibrationAudit:
    def test_detects_no_cap_decision(self) -> None:
        assert has_score_calibration_audit(
            [
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ]
        )

    def test_detects_band_and_ceiling_decision(self) -> None:
        assert has_score_calibration_audit(
            [
                "Score calibration/cap audit: the verified gaps fit the 0.6-0.79 band; the strictest practical ceiling is about 0.65.",
            ]
        )

    def test_detects_applied_known_failure_cap(self) -> None:
        assert has_score_calibration_audit(
            [
                "Score calibration/cap audit: wrong-source known failure mode caps the result around 0.45-0.65 depending severity.",
            ]
        )

    def test_does_not_accept_label_without_calibration_decision(self) -> None:
        assert not has_score_calibration_audit(
            [
                "Score calibration/cap audit: reviewed the score.",
            ]
        )

    def test_does_not_accept_bare_no_hard_cap_phrase(self) -> None:
        assert not has_score_calibration_audit(
            [
                "Score calibration/cap audit: no hard cap applies.",
            ]
        )

    def test_does_not_accept_vague_non_numeric_ceiling(self) -> None:
        assert not has_score_calibration_audit(
            [
                "Score calibration/cap audit: the strictest practical ceiling is about the mid-0.6s due major source-grounding and numerical omissions.",
            ]
        )

    def test_does_not_accept_vague_ceiling_with_only_selected_score_numeric(self) -> None:
        assert not has_score_calibration_audit(
            [
                "Score calibration/cap audit: the strictest practical ceiling from the verified evidence is about the mid-0.6s due major source-grounding and numerical omissions, so 0.62 is appropriate.",
            ]
        )

    def test_does_not_accept_plain_band_without_maximum(self) -> None:
        assert not has_score_calibration_audit(
            [
                "Score calibration/cap audit: the verified gaps fit the 0.6-0.79 band, so 0.62 is appropriate.",
            ]
        )

    def test_extracts_explicit_cap_ceiling(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: the strictest applicable cap is 0.60.",
            ]
        ) == pytest.approx(0.6)

    def test_extracts_maximum_score_allowed(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ]
        ) == pytest.approx(1.0)

    def test_extracts_range_cap_upper_bound(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: wrong-source known failure mode caps the result around 0.45-0.65 depending severity.",
            ]
        ) == pytest.approx(0.65)

    def test_extracts_maximum_of_phrase(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: the hard cap applies, and the unsupported analysis supports a maximum of 0.5.",
            ]
        ) == pytest.approx(0.5)

    def test_extracts_maximum_score_i_allow_phrase(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: because both required artifacts are absent, the maximum score I allow is 0.6.",
            ]
        ) == pytest.approx(0.6)

    def test_extracts_score_as_maximum_phrase(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: central forecast numbers are outside tolerance, so I treated 0.59 as the maximum.",
            ]
        ) == pytest.approx(0.59)

    def test_ignores_considered_but_not_applied_cap_when_extracting_ceiling(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: applicable caps are wrong revenue basis, strictest here about 0.60. I considered but did not apply the fabricated-data cap at 0.50 because most bad figures are source-derived; maximum supported score is about 0.60.",
            ]
        ) == pytest.approx(0.6)

    def test_ignores_considered_cap_list_when_none_apply(self) -> None:
        evidence = [
            "Score calibration/cap audit: applicable score band is 0.61-0.80 because most required artifacts "
            "and major sections exist but several key source reconciliation requirements are incomplete. "
            "Considered caps for no spreadsheet (0.45), no txt drafts (0.70), today's-date anchors (0.65), "
            "recipe COGS not used (0.60), and forbidden live actions (0.60); none directly apply. "
            "Strictest applicable cap is therefore 1.00, and the chosen 0.72 reflects substantive omissions."
        ]

        assert has_score_calibration_audit(evidence)
        assert extract_score_calibration_ceiling(evidence) == pytest.approx(1.0)

    def test_extracts_maximum_justified_band_upper_bound(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: Maximum justified band is 0.4-0.6, and the severity of core-method failures supports 0.42.",
            ]
        ) == pytest.approx(0.6)

    def test_extracts_cap_not_final_score_from_no_lower_cap_clause(self) -> None:
        assert extract_score_calibration_ceiling(
            [
                "Score calibration/cap audit: the verified work falls in the 0.4-0.6 band. "
                "I set the strictest applicable cap at 0.55; no lower hard cap was explicit, "
                "and the final score of 0.45 reflects useful partial deliverables but major baseline misses.",
            ]
        ) == pytest.approx(0.55)

    def test_extracts_cap_when_no_lower_cap_note_uses_commas(self) -> None:
        evidence = [
            "Score calibration/cap audit: I set the strictest applicable cap at 0.55, "
            "no lower hard cap was explicit, and the final score of 0.45 reflects useful partial deliverables.",
        ]

        assert has_score_calibration_audit(evidence)
        assert extract_score_calibration_ceiling(evidence) == pytest.approx(0.55)

    def test_extracts_strictest_cap_not_selected_score_below_that_cap(self) -> None:
        evidence = [
            "Score calibration/cap audit: guidance says 0.4-0.6 is appropriate when core criteria are addressed "
            "but notable shortcomings remain. The strictest applicable cap I applied is 0.60 for no Excel workbook "
            "or separate summary artifacts; the unsupported merchant-fee adjustment justifies scoring below that "
            "cap at 0.58.",
        ]

        assert has_score_calibration_audit(evidence)
        assert extract_score_calibration_ceiling(evidence) == pytest.approx(0.60)

    def test_does_not_extract_plain_score_band_as_ceiling(self) -> None:
        assert (
            extract_score_calibration_ceiling(
                [
                    "Score calibration/cap audit: the verified gaps fit the 0.6-0.79 band, so 0.62 is appropriate.",
                ]
            )
            is None
        )

    def test_does_not_extract_mid_decimal_phrase_as_zero_ceiling(self) -> None:
        assert (
            extract_score_calibration_ceiling(
                [
                    "Score calibration/cap audit: the strictest practical ceiling is about the mid-0.6s, so 0.62 is appropriate.",
                ]
            )
            is None
        )


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


def make_guidance_judge_input_json(
    tmp_path: pathlib.Path,
    *,
    instructions: str = "do a thing",
    judge_guidance: str = "Grade holistically.",
) -> str:
    """Write a minimal GuidanceJudgeInput JSON file and return its path."""
    trajectory = tmp_path / "gandalf_trajectory.json"
    trajectory.write_text(json.dumps({"steps": []}))
    data = {
        "model": "test-model",
        "instructions": instructions,
        "final_output": "done",
        "workdir": str(tmp_path),
        "trajectory_path": str(trajectory),
        "judge_guidance": judge_guidance,
    }
    p = tmp_path / "guidance_input.json"
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


class TestRunJudgeGuidance:
    """Tests for run_judge_guidance — mocks run_agent_session to avoid OpenHands."""

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_output_wraps_score_and_usage(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        input_path = make_guidance_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        score_data = {
            "score": 0.8,
            "reasoning": "Solid result.",
            "evidence": [
                "Workspace/artifact check: read /workspace/output.txt.",
                "Inspected trajectory file: final command succeeded.",
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ],
        }
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps(score_data))
            run_judge_guidance(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["guidance_score"]["score"] == 0.8
        assert data["guidance_score"]["reasoning"] == "Solid result."
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_prompt_names_judge_workspace_path(self, mock_session: Any, tmp_path: pathlib.Path) -> None:
        input_path = make_guidance_judge_input_json(
            tmp_path,
            instructions="Save the workbook to /home/agent/workspace/deliverables/report.xlsx.",
        )
        output_path = str(tmp_path / "output.json")

        score_data = {
            "score": 0.8,
            "reasoning": "Solid result.",
            "evidence": [
                "Workspace/artifact check: read /workspace/output.txt.",
                "Inspected trajectory file: final command succeeded.",
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ],
        }
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps(score_data))
            run_judge_guidance(input_path, output_path)

        prompt = mock_session.call_args.args[3]
        assert "<workspace_path>" in prompt
        assert str(tmp_path) in prompt
        assert "original container paths" in prompt.lower()

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_preserves_usage_when_score_file_invalid(self, mock_session: Any, tmp_path: pathlib.Path) -> None:  # noqa: ARG002
        input_path = make_guidance_judge_input_json(tmp_path)
        output_path = str(tmp_path / "output.json")

        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps({"score": 2, "reasoning": "bad"}))
            run_judge_guidance(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["guidance_score"]["score"] is None
        assert "range" in data["guidance_score"]["reasoning"].lower()
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_requires_action_audit_when_task_mentions_external_actions(
        self,
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
    ) -> None:
        input_path = make_guidance_judge_input_json(
            tmp_path,
            instructions="Create the workbook, but do not send email or create calendar events.",
            judge_guidance="Grade the final artifact and verify no live action happened.",
        )
        output_path = str(tmp_path / "output.json")

        score_data = {
            "score": 0.8,
            "reasoning": "Solid result.",
            "evidence": [
                "Workspace/artifact check: read /workspace/report.md.",
                "Inspected trajectory file for command failures",
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ],
        }
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps(score_data))
            run_judge_guidance(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["guidance_score"]["score"] is None
        assert "action/side-effect audit" in data["guidance_score"]["reasoning"].lower()
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_requires_output_location_conflict_audit_when_task_and_guidance_paths_differ(
        self,
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
    ) -> None:
        input_path = make_guidance_judge_input_json(
            tmp_path,
            instructions="Save the workbook to /home/agent/workspace/deliverables.",
            judge_guidance="Required artifact: workbook in /tmp/output.",
        )
        output_path = str(tmp_path / "output.json")

        score_data = {
            "score": 0.8,
            "reasoning": "Solid result.",
            "evidence": [
                "Workspace/artifact check: read /workspace/report.md.",
                "Inspected trajectory file: final command succeeded.",
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ],
        }
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps(score_data))
            run_judge_guidance(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["guidance_score"]["score"] is None
        assert "output-location conflict audit" in data["guidance_score"]["reasoning"].lower()
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_requires_source_availability_audit_when_guidance_names_expected_sources(
        self,
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
    ) -> None:
        input_path = make_guidance_judge_input_json(
            tmp_path,
            instructions="Use Shopify and QuickBooks only where files clearly support the interpretation.",
            judge_guidance="Expected sources include Shopify/orders.xlsx and QuickBooks/bills.csv.",
        )
        output_path = str(tmp_path / "output.json")

        score_data = {
            "score": 0.8,
            "reasoning": "Solid result.",
            "evidence": [
                "Workspace/artifact check: read /workspace/report.md.",
                "Inspected trajectory file: final command succeeded.",
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ],
        }
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps(score_data))
            run_judge_guidance(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["guidance_score"]["score"] is None
        assert "source availability audit" in data["guidance_score"]["reasoning"].lower()
        assert data["llm_usage"]["cost_usd"] == 0.05

    @patch("gandalf.judge.run_agent_session", return_value=MOCK_USAGE)
    def test_requires_source_verification_audit_when_guidance_requires_independent_verification(
        self,
        mock_session: Any,  # noqa: ARG002
        tmp_path: pathlib.Path,
    ) -> None:
        input_path = make_guidance_judge_input_json(
            tmp_path,
            instructions="Build a workbook with exact revenue calculations.",
            judge_guidance="Verification requirement: independently verify numerical claims against source files.",
        )
        output_path = str(tmp_path / "output.json")

        score_data = {
            "score": 0.8,
            "reasoning": "Solid result.",
            "evidence": [
                "Workspace/artifact check: read /workspace/report.md.",
                "Source availability audit: source files were accessible in /workspace/data.",
                "Inspected trajectory file: final command succeeded.",
                "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
            ],
        }
        with patch(
            "gandalf.judge.make_verdict_path",
            return_value=str(tmp_path / "guidance_score.json"),
        ):
            (tmp_path / "guidance_score.json").write_text(json.dumps(score_data))
            run_judge_guidance(input_path, output_path)

        data = json.loads((tmp_path / "output.json").read_text())
        assert data["guidance_score"]["score"] is None
        assert "source verification audit" in data["guidance_score"]["reasoning"].lower()
        assert data["llm_usage"]["cost_usd"] == 0.05


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
