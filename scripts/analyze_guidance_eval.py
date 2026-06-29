#!/usr/bin/env python3
"""Analyze guidance-mode scoring quality with guidance-grounded audit signals."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gandalf.guidance_evidence import (
    extract_score_calibration_ceiling,
    has_action_side_effect_audit,
    has_output_location_conflict_audit,
    has_score_calibration_audit,
    has_source_availability_audit,
    has_source_guidance_conflict_language,
    has_source_verification_audit,
    has_trajectory_evidence,
    has_workspace_artifact_evidence,
    requires_above_midpoint_justification,
    requires_action_side_effect_audit,
    requires_output_location_conflict_audit,
    requires_source_availability_audit,
    requires_source_verification_audit,
)

MIN_CORRELATION_POINTS = 2
MIN_RUBRIC_ALIGNMENT_TOKENS = 20
RUBRIC_TASK_MISMATCH_COVERAGE_THRESHOLD = 0.5
RUBRIC_LANGUAGE_RE = re.compile(r"\brubrics?\b", re.IGNORECASE)
GUIDANCE_RETRY_RE = re.compile(
    r"^\[retry\s+\d+/\d+\]\s+Retrying guidance score",
    re.IGNORECASE | re.MULTILINE,
)
FORMULA_CACHE_SOURCE_AUDIT_RE = re.compile(
    r"\bformula cache/source-value audit\b"
    r"|\bformulas?\s+without\s+cached\s+values?\b"
    r"|\bno\s+cached\s+values?\b"
    r"|\buncached\s+formulas?\b"
    r"|\b(?:posted|displayed)\s+(?:book-of-record\s+)?(?:actuals?|values?)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "about",
    "actual",
    "agent",
    "also",
    "analysis",
    "and",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "with",
    "workbook",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines into a list of dictionaries."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty dict on missing/invalid input."""
    try:
        with Path(path).open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_json_any(path: str | Path) -> Any:
    """Load JSON data, returning None on missing/invalid input."""
    try:
        with Path(path).open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def result_key(record: dict[str, Any]) -> tuple[str, str]:
    """Stable key shared by manifest and eval result records."""
    return str(record.get("slug", "")), str(record.get("trial_dir", ""))


def tokens(text: str) -> set[str]:
    """Tokenize text into content-ish words."""
    return {tok for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()) if tok not in STOPWORDS}


def read_text_path(path: str | Path) -> str:
    """Read a text file, returning an empty string on missing/invalid input."""
    try:
        return Path(path).read_text()
    except (OSError, TypeError):
        return ""


def safe_float(value: Any) -> float:
    """Coerce a value to float, returning 0 for missing/invalid input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    """Coerce a value to int, returning 0 for missing/invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def guidance_retry_count(result: dict[str, Any]) -> int:
    """Count guidance judge retry attempts recorded in a trial eval stdout file."""
    eval_dir = result.get("eval_dir")
    if not eval_dir:
        return 0
    return len(GUIDANCE_RETRY_RE.findall(read_text_path(Path(str(eval_dir)) / "stdout.txt")))


def guidance_usage_metrics(info: dict[str, Any]) -> dict[str, Any]:
    """Extract additive LLM usage metrics from guidance info.json."""
    usage = info.get("llm_usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "guidance_llm_cost_usd": round(safe_float(usage.get("cost_usd")), 4),
        "guidance_prompt_tokens": safe_int(usage.get("prompt_tokens")),
        "guidance_completion_tokens": safe_int(usage.get("completion_tokens")),
        "guidance_cache_read_tokens": safe_int(usage.get("cache_read_tokens")),
    }


def guidance_tokens(record: dict[str, Any]) -> set[str]:
    """Load and tokenize free-form grading guidance for a manifest record."""
    text = read_text_path(record.get("judge_guidance_path", ""))
    return tokens(text)


def extract_rubric_text(data: Any) -> str:
    """Extract human-readable criterion text from supported rubric JSON shapes."""
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                criterion = item.get("criterion")
                if criterion is not None:
                    parts.append(str(criterion))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    if isinstance(data, dict):
        criteria = data.get("criteria")
        if criteria is not None:
            return extract_rubric_text(criteria)
        criterion = data.get("criterion")
        if criterion is not None:
            return str(criterion)
    return ""


def rubric_tokens(record: dict[str, Any]) -> set[str]:
    """Load and tokenize rubric criterion text for a manifest record."""
    return tokens(extract_rubric_text(load_json_any(record.get("rubric_path", ""))))


def task_context_tokens(record: dict[str, Any]) -> set[str]:
    """Tokenize the task-facing context that a matching rubric should describe."""
    text = "\n".join(
        [
            read_text_path(record.get("instruction_path", "")),
            read_text_path(record.get("judge_guidance_path", "")),
        ]
    )
    return tokens(text)


def rubric_alignment_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Estimate whether a rubric appears to describe the same task as the manifest record."""
    rubric = rubric_tokens(record)
    task_context = task_context_tokens(record)
    coverage = len(rubric & task_context) / len(rubric) if rubric else 0.0
    potential_mismatch = (
        len(rubric) >= MIN_RUBRIC_ALIGNMENT_TOKENS
        and bool(task_context)
        and coverage < RUBRIC_TASK_MISMATCH_COVERAGE_THRESHOLD
    )
    return {
        "rubric_task_vocab_coverage": round(coverage, 4),
        "potential_rubric_task_mismatch": potential_mismatch,
    }


def info_text(info: dict[str, Any]) -> str:
    """Concatenate guidance reasoning and evidence text."""
    evidence = info.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return " ".join([str(info.get("reasoning", "")), *[str(item) for item in evidence]])


def evidence_metrics(info: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Compute non-score signals for guidance reasoning/evidence."""
    evidence = info.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    combined = info_text(info)
    combined_lower = combined.lower()
    guidance = guidance_tokens(record)
    g_tokens = tokens(combined)
    coverage = len(guidance & g_tokens) / len(guidance) if guidance else 0.0
    has_file_or_path = has_workspace_artifact_evidence([str(item) for item in evidence])
    mentions_trajectory = has_trajectory_evidence([str(item) for item in evidence])
    mentions_score_calibration = has_score_calibration_audit([str(item) for item in evidence])
    declared_score_ceiling = extract_score_calibration_ceiling([str(item) for item in evidence])
    mentions_output_location_audit = has_output_location_conflict_audit([str(item) for item in evidence])
    mentions_action_side_effect_audit = has_action_side_effect_audit([str(item) for item in evidence])
    mentions_source_availability_audit = has_source_availability_audit([str(item) for item in evidence])
    mentions_source_verification_audit = has_source_verification_audit([str(item) for item in evidence])
    mentions_rubric_language = bool(RUBRIC_LANGUAGE_RE.search(combined))
    mentions_source_guidance_conflict = any(
        has_source_guidance_conflict_language(str(item)) for item in [info.get("reasoning", ""), *evidence]
    )
    mentions_formula_cache_source_audit = bool(FORMULA_CACHE_SOURCE_AUDIT_RE.search(combined))
    instructions_text = read_text_path(record.get("instruction_path", ""))
    guidance_text = read_text_path(record.get("judge_guidance_path", ""))
    requires_action_check = requires_action_side_effect_audit(
        instructions_text,
        guidance_text,
    )
    requires_output_location_check = requires_output_location_conflict_audit(
        instructions_text,
        guidance_text,
    )
    requires_source_availability_check = requires_source_availability_audit(
        instructions_text,
        guidance_text,
    )
    requires_source_verification_check = requires_source_verification_audit(
        instructions_text,
        guidance_text,
    )
    mentions_action_or_side_effect = any(
        term in combined_lower
        for term in [
            "draft",
            "sent",
            "send",
            "email",
            "outlook",
            "calendar",
            "live action",
            "side effect",
            "quickbooks",
            "shopify",
            "square",
        ]
    )
    evidence_count = len(evidence)
    baseline_evidence_quality = (
        min(1.0, evidence_count / 3.0) * 0.5
        + (0.25 if has_file_or_path else 0.0)
        + (0.25 if mentions_trajectory else 0.0)
    )
    required_audits = [mentions_score_calibration]
    if requires_output_location_check:
        required_audits.append(mentions_output_location_audit)
    if requires_action_check:
        required_audits.append(mentions_action_side_effect_audit)
    if requires_source_availability_check:
        required_audits.append(mentions_source_availability_audit)
    if requires_source_verification_check:
        required_audits.append(mentions_source_verification_audit)
    required_audit_coverage = mean([1.0 if audit_present else 0.0 for audit_present in required_audits])
    evidence_quality = baseline_evidence_quality * 0.75 + required_audit_coverage * 0.25
    return {
        "evidence_count": evidence_count,
        "has_file_or_path_evidence": has_file_or_path,
        "mentions_trajectory_or_tools": mentions_trajectory,
        "guidance_vocab_coverage": round(coverage, 4),
        "mentions_score_calibration_cap_audit": mentions_score_calibration,
        "declared_score_ceiling": round(declared_score_ceiling, 4) if declared_score_ceiling is not None else None,
        "mentions_output_location_conflict_audit": mentions_output_location_audit,
        "requires_output_location_conflict_check": requires_output_location_check,
        "mentions_action_side_effect_audit": mentions_action_side_effect_audit,
        "requires_action_or_side_effect_check": requires_action_check,
        "mentions_source_availability_audit": mentions_source_availability_audit,
        "requires_source_availability_check": requires_source_availability_check,
        "mentions_source_verification_audit": mentions_source_verification_audit,
        "requires_source_verification_check": requires_source_verification_check,
        "mentions_rubric_language": mentions_rubric_language,
        "mentions_source_guidance_conflict": mentions_source_guidance_conflict,
        "mentions_formula_cache_source_audit": mentions_formula_cache_source_audit,
        "mentions_action_or_side_effects": mentions_action_or_side_effect,
        "baseline_evidence_quality": round(baseline_evidence_quality, 4),
        "required_audit_coverage": round(required_audit_coverage, 4),
        "evidence_quality": round(evidence_quality, 4),
    }


def mean(values: list[float]) -> float:
    """Return arithmetic mean, or 0 for empty input."""
    return sum(values) / len(values) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation without external dependencies."""
    if len(xs) < MIN_CORRELATION_POINTS or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    """Return average ranks for values."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            out[ordered[k][0]] = rank
        i = j
    return out


def joined_rows(
    manifest: list[dict[str, Any]],
    rubric_results: list[dict[str, Any]],
    guidance_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join manifest, rubric result, and guidance result rows."""
    manifest_by_key = {result_key(record): record for record in manifest}
    rubric_by_key = {result_key(row): row for row in rubric_results}
    guidance_by_key = {result_key(row): row for row in guidance_results}
    rows: list[dict[str, Any]] = []
    for key, record in manifest_by_key.items():
        rubric = rubric_by_key.get(key)
        guidance = guidance_by_key.get(key)
        if not rubric or not guidance:
            continue
        rubric_reward = rubric.get("reward")
        guidance_reward = guidance.get("reward")
        if rubric_reward is None or guidance_reward is None:
            continue
        guidance_info = load_json(guidance.get("info_path", ""))
        evidence = evidence_metrics(guidance_info, record)
        retry_count = guidance_retry_count(guidance)
        usage_metrics = guidance_usage_metrics(guidance_info)
        rubric_alignment = rubric_alignment_metrics(record)
        declared_score_ceiling = evidence["declared_score_ceiling"]
        exceeds_declared_ceiling = declared_score_ceiling is not None and float(guidance_reward) > float(
            declared_score_ceiling
        )
        near_ceiling_with_foundational_failure = requires_above_midpoint_justification(
            float(guidance_reward),
            float(declared_score_ceiling) if declared_score_ceiling is not None else None,
            reasoning=str(guidance_info.get("reasoning", "")),
            evidence=[str(item) for item in guidance_info.get("evidence", [])],
        )
        diff = float(guidance_reward) - float(rubric_reward)
        rows.append(
            {
                "slug": record.get("slug", ""),
                "split": record.get("split", ""),
                "env": record.get("env", ""),
                "task": record.get("task", ""),
                "trial_dir": record.get("trial_dir", ""),
                "rubric_reward": float(rubric_reward),
                "guidance_reward": float(guidance_reward),
                "signed_diff": round(diff, 4),
                "abs_diff": round(abs(diff), 4),
                "guidance_reasoning": guidance_info.get("reasoning", ""),
                "guidance_evidence": guidance_info.get("evidence", []),
                "guidance_exceeds_declared_score_ceiling": exceeds_declared_ceiling,
                "near_declared_ceiling_with_foundational_failure": near_ceiling_with_foundational_failure,
                "guidance_retry_count": retry_count,
                "guidance_retried": retry_count > 0,
                **usage_metrics,
                **evidence,
                **rubric_alignment,
            }
        )
    return rows


def threshold_agreement(rows: list[dict[str, Any]], threshold: float) -> float:
    """Compute agreement of rubric/guidance pass labels at threshold."""
    if not rows:
        return 0.0
    matches = sum((row["rubric_reward"] >= threshold) == (row["guidance_reward"] >= threshold) for row in rows)
    return matches / len(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize joined rows for one split or all splits."""
    if not rows:
        return {"n": 0}
    abs_diffs = [float(row["abs_diff"]) for row in rows]
    signed_diffs = [float(row["signed_diff"]) for row in rows]
    rubric = [float(row["rubric_reward"]) for row in rows]
    guidance = [float(row["guidance_reward"]) for row in rows]
    evidence_quality = [float(row["evidence_quality"]) for row in rows]
    coverage = [float(row["guidance_vocab_coverage"]) for row in rows]
    required_audit_coverage = [float(row["required_audit_coverage"]) for row in rows]
    retry_counts = [float(row["guidance_retry_count"]) for row in rows]
    llm_costs = [float(row["guidance_llm_cost_usd"]) for row in rows]
    prompt_tokens = [float(row["guidance_prompt_tokens"]) for row in rows]
    completion_tokens = [float(row["guidance_completion_tokens"]) for row in rows]
    cache_read_tokens = [float(row["guidance_cache_read_tokens"]) for row in rows]
    declared_ceiling_rows = [row for row in rows if row["declared_score_ceiling"] is not None]
    action_required_rows = [row for row in rows if row["requires_action_or_side_effect_check"]]
    output_location_required_rows = [row for row in rows if row["requires_output_location_conflict_check"]]
    source_required_rows = [row for row in rows if row["requires_source_availability_check"]]
    source_verification_required_rows = [row for row in rows if row["requires_source_verification_check"]]
    return {
        "n": len(rows),
        "mae": round(mean(abs_diffs), 4),
        "rmse": round(math.sqrt(mean([value * value for value in abs_diffs])), 4),
        "bias": round(mean(signed_diffs), 4),
        "pearson": round(pearson(rubric, guidance), 4) if pearson(rubric, guidance) is not None else None,
        "spearman": round(pearson(ranks(rubric), ranks(guidance)), 4)
        if pearson(ranks(rubric), ranks(guidance)) is not None
        else None,
        "threshold_agreement_0_5": round(threshold_agreement(rows, 0.5), 4),
        "threshold_agreement_0_8": round(threshold_agreement(rows, 0.8), 4),
        "mean_evidence_quality": round(mean(evidence_quality), 4),
        "mean_required_audit_coverage": round(mean(required_audit_coverage), 4),
        "mean_guidance_vocab_coverage": round(mean(coverage), 4),
        "mean_guidance_retry_count": round(mean(retry_counts), 4),
        "pct_guidance_retried": round(mean([1.0 if row["guidance_retried"] else 0.0 for row in rows]), 4),
        "total_guidance_llm_cost_usd": round(sum(llm_costs), 4),
        "mean_guidance_llm_cost_usd": round(mean(llm_costs), 4),
        "total_guidance_prompt_tokens": int(sum(prompt_tokens)),
        "mean_guidance_prompt_tokens": round(mean(prompt_tokens), 4),
        "total_guidance_completion_tokens": int(sum(completion_tokens)),
        "mean_guidance_completion_tokens": round(mean(completion_tokens), 4),
        "total_guidance_cache_read_tokens": int(sum(cache_read_tokens)),
        "mean_guidance_cache_read_tokens": round(mean(cache_read_tokens), 4),
        "pct_has_file_or_path_evidence": round(
            mean([1.0 if row["has_file_or_path_evidence"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_trajectory_or_tools": round(
            mean([1.0 if row["mentions_trajectory_or_tools"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_score_calibration_cap_audit": round(
            mean([1.0 if row["mentions_score_calibration_cap_audit"] else 0.0 for row in rows]), 4
        ),
        "pct_has_declared_score_ceiling": round(
            mean([1.0 if row["declared_score_ceiling"] is not None else 0.0 for row in rows]), 4
        ),
        "pct_guidance_exceeds_declared_score_ceiling": round(
            mean([1.0 if row["guidance_exceeds_declared_score_ceiling"] else 0.0 for row in rows]), 4
        ),
        "pct_near_declared_ceiling_with_foundational_failure": round(
            mean([1.0 if row["near_declared_ceiling_with_foundational_failure"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_rubric_language": round(
            mean([1.0 if row["mentions_rubric_language"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_source_guidance_conflict": round(
            mean([1.0 if row["mentions_source_guidance_conflict"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_formula_cache_source_audit": round(
            mean([1.0 if row["mentions_formula_cache_source_audit"] else 0.0 for row in rows]), 4
        ),
        "pct_potential_rubric_task_mismatch": round(
            mean([1.0 if row["potential_rubric_task_mismatch"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_output_location_conflict_audit": round(
            mean([1.0 if row["mentions_output_location_conflict_audit"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_action_side_effect_audit": round(
            mean([1.0 if row["mentions_action_side_effect_audit"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_source_availability_audit": round(
            mean([1.0 if row["mentions_source_availability_audit"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_source_verification_audit": round(
            mean([1.0 if row["mentions_source_verification_audit"] else 0.0 for row in rows]), 4
        ),
        "pct_requires_action_or_side_effect_check": round(
            mean([1.0 if row["requires_action_or_side_effect_check"] else 0.0 for row in rows]), 4
        ),
        "pct_requires_output_location_conflict_check": round(
            mean([1.0 if row["requires_output_location_conflict_check"] else 0.0 for row in rows]), 4
        ),
        "pct_requires_source_availability_check": round(
            mean([1.0 if row["requires_source_availability_check"] else 0.0 for row in rows]), 4
        ),
        "pct_requires_source_verification_check": round(
            mean([1.0 if row["requires_source_verification_check"] else 0.0 for row in rows]), 4
        ),
        "pct_mentions_action_or_side_effects_when_required": round(
            mean([1.0 if row["mentions_action_or_side_effects"] else 0.0 for row in action_required_rows]), 4
        )
        if action_required_rows
        else 0.0,
        "pct_mentions_action_side_effect_audit_when_required": round(
            mean([1.0 if row["mentions_action_side_effect_audit"] else 0.0 for row in action_required_rows]), 4
        )
        if action_required_rows
        else 0.0,
        "pct_guidance_exceeds_declared_score_ceiling_when_declared": round(
            mean([1.0 if row["guidance_exceeds_declared_score_ceiling"] else 0.0 for row in declared_ceiling_rows]),
            4,
        )
        if declared_ceiling_rows
        else 0.0,
        "pct_mentions_output_location_conflict_audit_when_required": round(
            mean(
                [
                    1.0 if row["mentions_output_location_conflict_audit"] else 0.0
                    for row in output_location_required_rows
                ]
            ),
            4,
        )
        if output_location_required_rows
        else 0.0,
        "pct_mentions_source_availability_audit_when_required": round(
            mean([1.0 if row["mentions_source_availability_audit"] else 0.0 for row in source_required_rows]), 4
        )
        if source_required_rows
        else 0.0,
        "pct_mentions_source_verification_audit_when_required": round(
            mean(
                [1.0 if row["mentions_source_verification_audit"] else 0.0 for row in source_verification_required_rows]
            ),
            4,
        )
        if source_verification_required_rows
        else 0.0,
    }


def summarize_guidance_runs(guidance_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all guidance run attempts, including failed runs without rewards."""
    if not guidance_results:
        return {"n": 0}
    status_counts = Counter(str(result.get("status", "unknown") or "unknown") for result in guidance_results)
    retry_counts = [float(guidance_retry_count(result)) for result in guidance_results]
    usages = [guidance_usage_metrics(load_json(result.get("info_path", ""))) for result in guidance_results]
    llm_costs = [float(usage["guidance_llm_cost_usd"]) for usage in usages]
    prompt_tokens = [float(usage["guidance_prompt_tokens"]) for usage in usages]
    completion_tokens = [float(usage["guidance_completion_tokens"]) for usage in usages]
    cache_read_tokens = [float(usage["guidance_cache_read_tokens"]) for usage in usages]
    missing_reward_count = sum(1 for result in guidance_results if result.get("reward") is None)
    failed_count = status_counts.get("failed", 0)
    ok_count = status_counts.get("ok", 0)
    return {
        "n": len(guidance_results),
        "status_counts": dict(sorted(status_counts.items())),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "missing_reward_count": missing_reward_count,
        "success_rate": round(ok_count / len(guidance_results), 4),
        "failure_rate": round(failed_count / len(guidance_results), 4),
        "mean_guidance_retry_count": round(mean(retry_counts), 4),
        "pct_guidance_retried": round(mean([1.0 if count > 0 else 0.0 for count in retry_counts]), 4),
        "total_guidance_llm_cost_usd": round(sum(llm_costs), 4),
        "mean_guidance_llm_cost_usd": round(mean(llm_costs), 4),
        "total_guidance_prompt_tokens": int(sum(prompt_tokens)),
        "mean_guidance_prompt_tokens": round(mean(prompt_tokens), 4),
        "total_guidance_completion_tokens": int(sum(completion_tokens)),
        "mean_guidance_completion_tokens": round(mean(completion_tokens), 4),
        "total_guidance_cache_read_tokens": int(sum(cache_read_tokens)),
        "mean_guidance_cache_read_tokens": round(mean(cache_read_tokens), 4),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    guidance_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize all rows and each split."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", ""))].append(row)
    summary = {
        "all": summarize_rows(rows),
        "by_split": {split: summarize_rows(split_rows) for split, split_rows in sorted(by_split.items())},
    }
    if guidance_results is not None:
        summary["guidance_runs"] = summarize_guidance_runs(guidance_results)
    return summary


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze guidance scoring quality with guidance-grounded signals.")
    parser.add_argument("--manifest", type=Path, default=Path("_run/rollouts_no_verifier_manifest_split.jsonl"))
    parser.add_argument("--rubric-results", type=Path, default=Path("_run/gandalf_eval/results_rubric.jsonl"))
    parser.add_argument("--guidance-results", type=Path, default=Path("_run/gandalf_eval/results_guidance.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("_run/gandalf_eval/analysis"))
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    guidance_results = load_jsonl(args.guidance_results)
    rows = joined_rows(load_jsonl(args.manifest), load_jsonl(args.rubric_results), guidance_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, guidance_results=guidance_results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_jsonl(rows, args.output_dir / "joined_rows.jsonl")
    top_disagreements = sorted(rows, key=lambda row: row["abs_diff"], reverse=True)[: args.top_n]
    write_jsonl(top_disagreements, args.output_dir / "top_disagreements.jsonl")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
