"""Tests for Harbor rollout helper scripts."""

import importlib.util
import json
import pathlib
from types import ModuleType

import pytest


def load_index_module() -> ModuleType:
    """Load scripts/index_harbor_rollouts.py without requiring scripts to be a package."""
    module_path = pathlib.Path(__file__).parents[1] / "scripts" / "index_harbor_rollouts.py"
    spec = importlib.util.spec_from_file_location("index_harbor_rollouts", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_script_module(name: str) -> ModuleType:
    """Load a script module without requiring scripts to be a package."""
    module_path = pathlib.Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discover_rollout_records_maps_trial_to_harbor_task(tmp_path: pathlib.Path) -> None:
    module = load_index_module()
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "sheet_env" / "rebalance_budget"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text("name = 'rebalance_budget'\n")
    (task_dir / "instruction.md").write_text("Do the task.")
    (task_dir / "tests" / "judge_guidance.md").write_text("Grade holistically.")
    (task_dir / "tests" / "rubric.json").write_text(json.dumps([]))

    rollouts_root = tmp_path / "rollouts"
    trial_dir = rollouts_root / "sheet_env__rebalance_budget" / "trial_0"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "artifacts" / "workspace").mkdir(parents=True)
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps({"steps": []}))

    records = module.discover_rollout_records(rollouts_root, tasks_root)

    assert len(records) == 1
    record = records[0]
    assert record["slug"] == "sheet_env__rebalance_budget"
    assert record["env"] == "sheet_env"
    assert record["task"] == "rebalance_budget"
    assert record["task_dir"] == str(task_dir)
    assert record["instruction_path"] == str(task_dir / "instruction.md")
    assert record["judge_guidance_path"] == str(task_dir / "tests" / "judge_guidance.md")
    assert record["rubric_path"] == str(task_dir / "tests" / "rubric.json")
    assert record["trajectory_path"] == str(trial_dir / "agent" / "trajectory.json")
    assert record["workspace_path"] == str(trial_dir / "artifacts" / "workspace")


def test_discover_rollout_records_ignores_artifact_trajectory_copy(tmp_path: pathlib.Path) -> None:
    module = load_index_module()
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "sheet_env" / "rebalance_budget"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the task.")
    (task_dir / "judge_guidance.md").write_text("Grade holistically.")
    (task_dir / "rubric.json").write_text(json.dumps([]))

    rollouts_root = tmp_path / "rollouts"
    trial_dir = rollouts_root / "sheet_env__rebalance_budget" / "trial_0"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "artifacts" / "agent").mkdir(parents=True)
    (trial_dir / "artifacts" / "workspace").mkdir(parents=True)
    canonical_trajectory = trial_dir / "agent" / "trajectory.json"
    artifact_copy = trial_dir / "artifacts" / "agent" / "trajectory.json"
    canonical_trajectory.write_text(json.dumps({"source": "canonical"}))
    artifact_copy.write_text(json.dumps({"source": "artifact"}))

    records = module.discover_rollout_records(rollouts_root, tasks_root)

    assert len(records) == 1
    assert records[0]["trajectory_path"] == str(canonical_trajectory)


def test_assign_splits_groups_same_slug_and_uses_all_splits() -> None:
    module = load_script_module("split_harbor_manifest")
    records = [{"env": "env-a", "slug": f"env-a__task-{i}", "trial_dir": f"/trial/{i}"} for i in range(9)]
    records.append({"env": "env-a", "slug": "env-a__task-1", "trial_dir": "/trial/duplicate"})

    split_records = module.assign_splits(records, seed="test-seed")
    splits = {record["split"] for record in split_records}
    assert splits == {"train", "eval", "test"}

    by_slug: dict[str, set[str]] = {}
    for record in split_records:
        by_slug.setdefault(record["slug"], set()).add(record["split"])
    assert all(len(split_values) == 1 for split_values in by_slug.values())


def test_analyze_guidance_eval_uses_score_and_evidence_signals(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_text(
        json.dumps(
            [
                {"criterion": "Workbook includes monthly revenue trend analysis", "weight": 1},
                {"criterion": "Evidence cites trajectory command failures", "weight": 1},
            ]
        )
    )
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text(
        "Grade monthly revenue trend analysis. Require trajectory evidence, "
        "score calibration/cap audit, and output-location conflict audit."
    )
    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text("Prepare the workbook, but do not send email or create calendar events.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The workbook includes monthly revenue trend analysis but misses one detail.",
                "evidence": [
                    "Read /workspace/deliverables/report.xlsx",
                    "Inspected trajectory command output for failures",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    "Output-location conflict audit: artifact path matches task instructions.",
                    "Action/side-effect audit: no email send or calendar event tool calls were present.",
                ],
            }
        )
    )
    info_path_without_action = tmp_path / "guidance_info_without_action.json"
    info_path_without_action.write_text(
        json.dumps(
            {
                "reasoning": "The workbook includes monthly revenue trend analysis.",
                "evidence": [
                    "Read /workspace/deliverables/report.xlsx",
                    "Inspected trajectory command output for failures",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                    "Output-location conflict audit: artifact path matches task instructions.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__task",
            "trial_dir": "/trial",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "rubric_path": str(rubric_path),
            "judge_guidance_path": str(guidance_path),
        },
        {
            "slug": "env__task_without_action_requirement",
            "trial_dir": "/trial-2",
            "split": "eval",
            "rubric_path": str(rubric_path),
            "judge_guidance_path": str(guidance_path),
        },
    ]
    rubric_results = [
        {"slug": "env__task", "trial_dir": "/trial", "status": "ok", "reward": 0.75},
        {"slug": "env__task_without_action_requirement", "trial_dir": "/trial-2", "status": "ok", "reward": 0.75},
    ]
    guidance_results = [
        {
            "slug": "env__task",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.8,
            "info_path": str(info_path),
        },
        {
            "slug": "env__task_without_action_requirement",
            "trial_dir": "/trial-2",
            "status": "ok",
            "reward": 0.8,
            "info_path": str(info_path_without_action),
        },
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["abs_diff"] == 0.05
    assert rows[0]["evidence_count"] == 5
    assert rows[0]["has_file_or_path_evidence"] is True
    assert rows[0]["mentions_trajectory_or_tools"] is True
    assert rows[0]["guidance_vocab_coverage"] > 0
    assert rows[0]["mentions_score_calibration_cap_audit"] is True
    assert rows[0]["mentions_output_location_conflict_audit"] is True
    assert rows[0]["mentions_action_side_effect_audit"] is True
    assert rows[0]["requires_action_or_side_effect_check"] is True
    assert rows[1]["mentions_action_side_effect_audit"] is False
    assert rows[1]["requires_action_or_side_effect_check"] is False
    assert "criterion_vocab_coverage" not in rows[0]
    assert summary["by_split"]["eval"]["threshold_agreement_0_5"] == 1.0
    assert summary["by_split"]["eval"]["mean_guidance_vocab_coverage"] > 0
    assert summary["by_split"]["eval"]["pct_mentions_action_side_effect_audit"] == 0.5
    assert summary["by_split"]["eval"]["pct_mentions_action_side_effect_audit_when_required"] == 1.0


def test_analyze_guidance_eval_flags_likely_rubric_task_mismatch(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    aligned_rubric_path = tmp_path / "aligned_rubric.json"
    aligned_rubric_path.write_text(
        json.dumps(
            [
                {"criterion": "Workbook includes FY2025 Shopify revenue trend by month.", "weight": 1},
                {"criterion": "Workbook compares Astor Apparel actuals to the P&L budget and buy plans.", "weight": 1},
                {
                    "criterion": "Workbook flags QuickBooks expense limitations and TimeStation missing wage rates.",
                    "weight": 1,
                },
            ]
        )
    )
    mismatched_rubric_path = tmp_path / "mismatched_rubric.json"
    mismatched_rubric_path.write_text(
        json.dumps(
            [
                {
                    "criterion": "The deliverable is a Word docx file for a Curated Prestige May 2026 campaign brief.",
                    "weight": 1,
                },
                {
                    "criterion": "The brief recommends Instagram stories, Square checkout links, and a handbag shipping incentive.",
                    "weight": 1,
                },
                {
                    "criterion": "The brief reports open accounts payable and overdue vendor balances for the campaign.",
                    "weight": 1,
                },
            ]
        )
    )
    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text(
        "Prepare an Astor Apparel FY2025 Excel workbook covering Shopify revenue trend, budget comparison, "
        "QuickBooks expenses, TimeStation labor limits, concentration, and recommended next steps."
    )
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text(
        "Grade the Astor Apparel FY2025 performance review workbook using Shopify orders, Astor budget files, "
        "buy plans, QuickBooks bills, and TimeStation source records."
    )
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "Checked the workbook.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
                    "Trajectory check: inspected tool calls.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__aligned",
            "trial_dir": "/trial-1",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "judge_guidance_path": str(guidance_path),
            "rubric_path": str(aligned_rubric_path),
        },
        {
            "slug": "env__mismatched",
            "trial_dir": "/trial-2",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "judge_guidance_path": str(guidance_path),
            "rubric_path": str(mismatched_rubric_path),
        },
    ]
    rubric_results = [
        {"slug": "env__aligned", "trial_dir": "/trial-1", "status": "ok", "reward": 0.8},
        {"slug": "env__mismatched", "trial_dir": "/trial-2", "status": "ok", "reward": 0.0},
    ]
    guidance_results = [
        {"slug": "env__aligned", "trial_dir": "/trial-1", "status": "ok", "reward": 0.8, "info_path": str(info_path)},
        {
            "slug": "env__mismatched",
            "trial_dir": "/trial-2",
            "status": "ok",
            "reward": 0.8,
            "info_path": str(info_path),
        },
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    rows_by_slug = {row["slug"]: row for row in rows}
    summary = module.summarize(rows)

    assert rows_by_slug["env__aligned"]["potential_rubric_task_mismatch"] is False
    assert rows_by_slug["env__aligned"]["rubric_task_vocab_coverage"] >= 0.5
    assert rows_by_slug["env__mismatched"]["potential_rubric_task_mismatch"] is True
    assert rows_by_slug["env__mismatched"]["rubric_task_vocab_coverage"] < 0.5
    assert summary["by_split"]["eval"]["pct_potential_rubric_task_mismatch"] == 0.5


def test_analyze_guidance_eval_flags_rubric_language_in_guidance_output(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade the workbook using score bands.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The workbook is mixed.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/report.xlsx.",
                    "Trajectory check: inspected tool calls.",
                    "Score calibration/cap audit: the rubric requires a maximum score allowed of 0.70.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__rubric_language",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [{"slug": "env__rubric_language", "trial_dir": "/trial", "status": "ok", "reward": 0.5}]
    guidance_results = [
        {
            "slug": "env__rubric_language",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["mentions_rubric_language"] is True
    assert summary["by_split"]["eval"]["pct_mentions_rubric_language"] == 1.0


def test_analyze_guidance_eval_flags_source_guidance_conflicts(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade category totals against expected golden figures.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The workbook follows the source but misses the stated checkpoint.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/report.xlsx.",
                    "Source verification audit: the captured COGS tracker recomputed to Handbags 104, matching the workbook but not the guidance's stated expected category distribution.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__source_guidance_conflict",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [{"slug": "env__source_guidance_conflict", "trial_dir": "/trial", "status": "ok", "reward": 0.5}]
    guidance_results = [
        {
            "slug": "env__source_guidance_conflict",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["mentions_source_guidance_conflict"] is True
    assert summary["by_split"]["eval"]["pct_mentions_source_guidance_conflict"] == 1.0


def test_analyze_guidance_eval_flags_formula_cache_source_audits(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade finance workbook actuals using displayed book-of-record values.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The workbook recomputed unposted finance lines.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/report.xlsx.",
                    "Formula cache/source-value audit: source finance workbook cells I3:I12 contain formulas without cached values, so they should remain blank posted actuals.",
                    "Score calibration/cap audit: strictest applicable cap is 0.50.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__formula_cache",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [{"slug": "env__formula_cache", "trial_dir": "/trial", "status": "ok", "reward": 0.5}]
    guidance_results = [
        {
            "slug": "env__formula_cache",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["mentions_formula_cache_source_audit"] is True
    assert summary["by_split"]["eval"]["pct_mentions_formula_cache_source_audit"] == 1.0


def test_analyze_guidance_eval_does_not_overcredit_source_and_command_evidence(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade the final deliverable and trajectory evidence.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The result was partially checked.",
                "evidence": [
                    "Read source data at /workspace/data/orders.csv.",
                    "Ran a command that printed workbook sheet names.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__source_only",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [{"slug": "env__source_only", "trial_dir": "/trial", "status": "ok", "reward": 0.5}]
    guidance_results = [
        {
            "slug": "env__source_only",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)

    assert rows[0]["has_file_or_path_evidence"] is False
    assert rows[0]["mentions_trajectory_or_tools"] is False


def test_analyze_guidance_eval_tracks_source_availability_audits(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text("Use Shopify and QuickBooks where files clearly support the interpretation.")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Expected sources include Shopify/orders.xlsx and QuickBooks/bills.csv.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The result checked available sources.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
                    "Source availability audit: Shopify exports were missing; QuickBooks bills.csv was accessible.",
                    "Inspected trajectory file: final command succeeded.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    info_path_without_source_audit = tmp_path / "guidance_info_without_source_audit.json"
    info_path_without_source_audit.write_text(
        json.dumps(
            {
                "reasoning": "The result checked sources but did not audit availability.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
                    "Inspected trajectory file: final command succeeded.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__with_source_audit",
            "trial_dir": "/trial-1",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "judge_guidance_path": str(guidance_path),
        },
        {
            "slug": "env__without_source_audit",
            "trial_dir": "/trial-2",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "judge_guidance_path": str(guidance_path),
        },
    ]
    rubric_results = [
        {"slug": "env__with_source_audit", "trial_dir": "/trial-1", "status": "ok", "reward": 0.5},
        {"slug": "env__without_source_audit", "trial_dir": "/trial-2", "status": "ok", "reward": 0.5},
    ]
    guidance_results = [
        {
            "slug": "env__with_source_audit",
            "trial_dir": "/trial-1",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path),
        },
        {
            "slug": "env__without_source_audit",
            "trial_dir": "/trial-2",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path_without_source_audit),
        },
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["requires_source_availability_check"] is True
    assert rows[0]["mentions_source_availability_audit"] is True
    assert rows[1]["requires_source_availability_check"] is True
    assert rows[1]["mentions_source_availability_audit"] is False
    assert rows[0]["required_audit_coverage"] == 1.0
    assert rows[1]["required_audit_coverage"] < rows[0]["required_audit_coverage"]
    assert rows[1]["evidence_quality"] < rows[0]["evidence_quality"]
    assert summary["by_split"]["eval"]["pct_requires_source_availability_check"] == 1.0
    assert summary["by_split"]["eval"]["pct_mentions_source_availability_audit"] == 0.5
    assert summary["by_split"]["eval"]["pct_mentions_source_availability_audit_when_required"] == 0.5
    assert summary["by_split"]["eval"]["mean_required_audit_coverage"] < 1.0


def test_analyze_guidance_eval_tracks_source_verification_audits(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text("Prepare exact revenue calculations in a workbook.")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Verification requirement: independently verify numerical claims against source files.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The result checked source-backed numbers.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
                    "Source availability audit: source.csv was available in the workspace.",
                    "Source verification audit: recomputed revenue totals from source.csv and compared them to report.xlsx.",
                    "Inspected trajectory file: final command succeeded.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    info_path_without_source_verification = tmp_path / "guidance_info_without_source_verification.json"
    info_path_without_source_verification.write_text(
        json.dumps(
            {
                "reasoning": "The result checked the workbook but did not independently verify the source numbers.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/deliverables/report.xlsx.",
                    "Source availability audit: source.csv was available in the workspace.",
                    "Inspected trajectory file: final command succeeded.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__with_source_verification",
            "trial_dir": "/trial-1",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "judge_guidance_path": str(guidance_path),
        },
        {
            "slug": "env__without_source_verification",
            "trial_dir": "/trial-2",
            "split": "eval",
            "instruction_path": str(instruction_path),
            "judge_guidance_path": str(guidance_path),
        },
    ]
    rubric_results = [
        {"slug": "env__with_source_verification", "trial_dir": "/trial-1", "status": "ok", "reward": 0.5},
        {"slug": "env__without_source_verification", "trial_dir": "/trial-2", "status": "ok", "reward": 0.5},
    ]
    guidance_results = [
        {
            "slug": "env__with_source_verification",
            "trial_dir": "/trial-1",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path),
        },
        {
            "slug": "env__without_source_verification",
            "trial_dir": "/trial-2",
            "status": "ok",
            "reward": 0.5,
            "info_path": str(info_path_without_source_verification),
        },
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["requires_source_verification_check"] is True
    assert rows[0]["mentions_source_verification_audit"] is True
    assert rows[1]["requires_source_verification_check"] is True
    assert rows[1]["mentions_source_verification_audit"] is False
    assert rows[0]["required_audit_coverage"] == 1.0
    assert rows[1]["required_audit_coverage"] < rows[0]["required_audit_coverage"]
    assert rows[1]["evidence_quality"] < rows[0]["evidence_quality"]
    assert summary["by_split"]["eval"]["pct_requires_source_verification_check"] == 1.0
    assert summary["by_split"]["eval"]["pct_mentions_source_verification_audit"] == 0.5
    assert summary["by_split"]["eval"]["pct_mentions_source_verification_audit_when_required"] == 0.5
    assert summary["by_split"]["eval"]["mean_required_audit_coverage"] < 1.0


def test_analyze_guidance_eval_tracks_declared_score_ceiling_violations(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade using score caps and verified artifacts.")
    over_cap_info_path = tmp_path / "over_cap_info.json"
    over_cap_info_path.write_text(
        json.dumps(
            {
                "reasoning": "The result exceeds its own cap.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/report.md.",
                    "Inspected trajectory file: final command succeeded.",
                    "Score calibration/cap audit: the strictest applicable cap is 0.60.",
                ],
            }
        )
    )
    within_cap_info_path = tmp_path / "within_cap_info.json"
    within_cap_info_path.write_text(
        json.dumps(
            {
                "reasoning": "The result stays inside its cap.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/report.md.",
                    "Inspected trajectory file: final command succeeded.",
                    "Score calibration/cap audit: the strictest applicable cap is 0.60.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__over_cap",
            "trial_dir": "/trial-1",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        },
        {
            "slug": "env__within_cap",
            "trial_dir": "/trial-2",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        },
    ]
    rubric_results = [
        {"slug": "env__over_cap", "trial_dir": "/trial-1", "status": "ok", "reward": 0.5},
        {"slug": "env__within_cap", "trial_dir": "/trial-2", "status": "ok", "reward": 0.5},
    ]
    guidance_results = [
        {
            "slug": "env__over_cap",
            "trial_dir": "/trial-1",
            "status": "ok",
            "reward": 0.75,
            "info_path": str(over_cap_info_path),
        },
        {
            "slug": "env__within_cap",
            "trial_dir": "/trial-2",
            "status": "ok",
            "reward": 0.55,
            "info_path": str(within_cap_info_path),
        },
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["declared_score_ceiling"] == 0.6
    assert rows[0]["guidance_exceeds_declared_score_ceiling"] is True
    assert rows[1]["declared_score_ceiling"] == 0.6
    assert rows[1]["guidance_exceeds_declared_score_ceiling"] is False
    assert summary["by_split"]["eval"]["pct_has_declared_score_ceiling"] == 1.0
    assert summary["by_split"]["eval"]["pct_guidance_exceeds_declared_score_ceiling"] == 0.5


def test_analyze_guidance_eval_flags_near_ceiling_foundational_failures(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade central forecast math and required external verification.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": (
                    "The central quantitative output is materially wrong, and the agent missed a required "
                    "external verification, but the artifact exists."
                ),
                "evidence": [
                    "Workspace/artifact check: read /workspace/email_draft.txt.",
                    "Inspected trajectory file: found a draft creation call.",
                    "Score calibration/cap audit: because foundational requirement failures remain, the maximum score allowed is 0.60.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__near_ceiling_foundational_failure",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [
        {
            "slug": "env__near_ceiling_foundational_failure",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.3,
        }
    ]
    guidance_results = [
        {
            "slug": "env__near_ceiling_foundational_failure",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.56,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["near_declared_ceiling_with_foundational_failure"] is True
    assert summary["by_split"]["eval"]["pct_near_declared_ceiling_with_foundational_failure"] == 1.0


def test_analyze_guidance_eval_ignores_low_ceiling_foundational_failures(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade central forecast math and required external verification.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The central quantitative output is materially wrong.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/email_draft.txt.",
                    "Inspected trajectory file: found a draft creation call.",
                    "Score calibration/cap audit: the maximum score allowed is 0.45.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__low_ceiling_foundational_failure",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [
        {
            "slug": "env__low_ceiling_foundational_failure",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.3,
        }
    ]
    guidance_results = [
        {
            "slug": "env__low_ceiling_foundational_failure",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.42,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["near_declared_ceiling_with_foundational_failure"] is False
    assert summary["by_split"]["eval"]["pct_near_declared_ceiling_with_foundational_failure"] == 0.0


def test_analyze_guidance_eval_flags_near_ceiling_missing_required_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade the required workbook and written-summary deliverables.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": (
                    "The response did not produce the required Excel workbook or separate written-summary "
                    "artifact, but it used the source data well."
                ),
                "evidence": [
                    (
                        "Workspace/artifact check: found source files but no generated deliverable workbook, "
                        "markdown/text/doc/pdf summary, or output directory."
                    ),
                    "Inspected trajectory file: the final step contains only a textual answer.",
                    (
                        "Score calibration/cap audit: no Excel workbook/separate summary artifacts cap the "
                        "maximum score allowed at 0.60."
                    ),
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__near_ceiling_missing_artifacts",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [
        {
            "slug": "env__near_ceiling_missing_artifacts",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.3,
        }
    ]
    guidance_results = [
        {
            "slug": "env__near_ceiling_missing_artifacts",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.58,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["near_declared_ceiling_with_foundational_failure"] is True
    assert summary["by_split"]["eval"]["pct_near_declared_ceiling_with_foundational_failure"] == 1.0


def test_analyze_guidance_eval_flags_high_foundational_failure_scores_without_justification(
    tmp_path: pathlib.Path,
) -> None:
    module = load_script_module("analyze_guidance_eval")
    guidance_path = tmp_path / "judge_guidance.md"
    guidance_path.write_text("Grade source-grounded workbook analysis.")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": (
                    "The central sales methodology is materially wrong, and multiple central quantitative "
                    "requirements fail despite a useful workbook."
                ),
                "evidence": [
                    "Workspace/artifact check: read /workspace/deliverables/review.xlsx.",
                    "Inspected trajectory file: workbook creation command was present.",
                    "Score calibration/cap audit: the wrong revenue basis makes the maximum score allowed 0.65.",
                ],
            }
        )
    )
    manifest = [
        {
            "slug": "env__high_foundational_failure_score",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(guidance_path),
        }
    ]
    rubric_results = [
        {
            "slug": "env__high_foundational_failure_score",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.3,
        }
    ]
    guidance_results = [
        {
            "slug": "env__high_foundational_failure_score",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.58,
            "info_path": str(info_path),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["near_declared_ceiling_with_foundational_failure"] is True
    assert summary["by_split"]["eval"]["pct_near_declared_ceiling_with_foundational_failure"] == 1.0


def test_analyze_guidance_eval_tracks_guidance_retry_and_usage_metrics(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    info_path = tmp_path / "guidance_info.json"
    info_path.write_text(
        json.dumps(
            {
                "reasoning": "The deliverable mostly matches the task.",
                "evidence": [
                    "Workspace/artifact check: read /workspace/report.xlsx.",
                    "Inspected trajectory file: command output supports the score.",
                    "Score calibration/cap audit: no hard cap applies; maximum score allowed is 1.0.",
                ],
                "llm_usage": {
                    "cost_usd": 1.23456,
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "cache_read_tokens": 900,
                },
            }
        )
    )
    eval_dir = tmp_path / "guidance_eval"
    eval_dir.mkdir()
    (eval_dir / "stdout.txt").write_text(
        "[retry 1/2] Retrying guidance score after invalid judge output\n"
        "[retry 2/2] Retrying guidance score after invalid judge output\n"
    )
    manifest = [
        {
            "slug": "env__tracked_guidance_run",
            "trial_dir": "/trial",
            "split": "eval",
            "judge_guidance_path": str(tmp_path / "judge_guidance.md"),
        }
    ]
    rubric_results = [{"slug": "env__tracked_guidance_run", "trial_dir": "/trial", "status": "ok", "reward": 0.7}]
    guidance_results = [
        {
            "slug": "env__tracked_guidance_run",
            "trial_dir": "/trial",
            "status": "ok",
            "reward": 0.75,
            "info_path": str(info_path),
            "eval_dir": str(eval_dir),
        }
    ]

    rows = module.joined_rows(manifest, rubric_results, guidance_results)
    summary = module.summarize(rows)

    assert rows[0]["guidance_retry_count"] == 2
    assert rows[0]["guidance_retried"] is True
    assert rows[0]["guidance_llm_cost_usd"] == 1.2346
    assert rows[0]["guidance_prompt_tokens"] == 1000
    assert rows[0]["guidance_completion_tokens"] == 20
    assert rows[0]["guidance_cache_read_tokens"] == 900
    assert summary["by_split"]["eval"]["mean_guidance_retry_count"] == 2.0
    assert summary["by_split"]["eval"]["pct_guidance_retried"] == 1.0
    assert summary["by_split"]["eval"]["total_guidance_llm_cost_usd"] == 1.2346
    assert summary["by_split"]["eval"]["mean_guidance_llm_cost_usd"] == 1.2346


def test_analyze_guidance_eval_summarizes_failed_guidance_runs(tmp_path: pathlib.Path) -> None:
    module = load_script_module("analyze_guidance_eval")
    ok_info_path = tmp_path / "ok_info.json"
    ok_info_path.write_text(json.dumps({"llm_usage": {"cost_usd": 0.25, "prompt_tokens": 100}}))
    failed_info_path = tmp_path / "failed_info.json"
    failed_info_path.write_text(json.dumps({"llm_usage": {"cost_usd": 0.75, "prompt_tokens": 300}}))
    ok_eval_dir = tmp_path / "ok_eval"
    ok_eval_dir.mkdir()
    (ok_eval_dir / "stdout.txt").write_text("[guidance] Evaluating holistic score\n")
    failed_eval_dir = tmp_path / "failed_eval"
    failed_eval_dir.mkdir()
    (failed_eval_dir / "stdout.txt").write_text(
        "[guidance] Evaluating holistic score\n[retry 1/2] Retrying guidance score...\n"
    )
    rows = [
        {
            "slug": "env__ok",
            "split": "eval",
            "guidance_reward": 0.8,
            "rubric_reward": 0.75,
            "abs_diff": 0.05,
            "signed_diff": 0.05,
            "evidence_quality": 1.0,
            "required_audit_coverage": 1.0,
            "guidance_vocab_coverage": 0.2,
            "guidance_retry_count": 0,
            "guidance_retried": False,
            "guidance_llm_cost_usd": 0.25,
            "guidance_prompt_tokens": 100,
            "guidance_completion_tokens": 0,
            "guidance_cache_read_tokens": 0,
            "declared_score_ceiling": 1.0,
            "guidance_exceeds_declared_score_ceiling": False,
            "near_declared_ceiling_with_foundational_failure": False,
            "has_file_or_path_evidence": True,
            "mentions_trajectory_or_tools": True,
            "mentions_score_calibration_cap_audit": True,
            "mentions_rubric_language": False,
            "mentions_source_guidance_conflict": False,
            "mentions_formula_cache_source_audit": False,
            "mentions_output_location_conflict_audit": False,
            "mentions_action_side_effect_audit": False,
            "mentions_source_availability_audit": False,
            "mentions_source_verification_audit": False,
            "mentions_action_or_side_effects": False,
            "requires_action_or_side_effect_check": False,
            "requires_output_location_conflict_check": False,
            "requires_source_availability_check": False,
            "requires_source_verification_check": False,
            "potential_rubric_task_mismatch": False,
        }
    ]
    guidance_results = [
        {
            "slug": "env__ok",
            "trial_dir": "/ok",
            "status": "ok",
            "reward": 0.8,
            "info_path": str(ok_info_path),
            "eval_dir": str(ok_eval_dir),
        },
        {
            "slug": "env__failed",
            "trial_dir": "/failed",
            "status": "failed",
            "info_path": str(failed_info_path),
            "eval_dir": str(failed_eval_dir),
        },
    ]

    summary = module.summarize(rows, guidance_results=guidance_results)

    assert summary["guidance_runs"]["n"] == 2
    assert summary["guidance_runs"]["status_counts"] == {"failed": 1, "ok": 1}
    assert summary["guidance_runs"]["failed_count"] == 1
    assert summary["guidance_runs"]["missing_reward_count"] == 1
    assert summary["guidance_runs"]["failure_rate"] == 0.5
    assert summary["guidance_runs"]["mean_guidance_retry_count"] == 0.5
    assert summary["guidance_runs"]["pct_guidance_retried"] == 0.5
    assert summary["guidance_runs"]["total_guidance_llm_cost_usd"] == 1.0
    assert summary["guidance_runs"]["mean_guidance_prompt_tokens"] == 200.0


def test_eval_guidance_scores_defaults_to_openai_when_env_file_has_only_openai_key(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module("eval_guidance_scores")
    for key in [
        "ANTHROPIC_API_KEY",
        "GANDALF_EVAL_MODEL",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "env"
    env_file.write_text("OPENAI_API_KEY=fake-openai-key\n")

    env = module.build_grader_env(env_file)
    model = module.choose_default_model(env)
    module.set_llm_api_key_for_model(env, model)

    assert model == "openai/gpt-5.5"
    assert env["LLM_API_KEY"] == "fake-openai-key"


def test_eval_guidance_scores_does_not_use_openai_key_for_explicit_gemini_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module("eval_guidance_scores")
    for key in [
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    env = {"OPENAI_API_KEY": "fake-openai-key"}

    module.set_llm_api_key_for_model(env, "gemini/gemini-2.5-flash")

    assert "LLM_API_KEY" not in env
