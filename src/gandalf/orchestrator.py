"""Outer grader orchestrator.

Runs as the grader user and spawns the inner judge as the sandbox user
(via sudo) to evaluate rubric criteria using an OpenHands agent-as-judge.

Supports two evaluation modes (configured via ``mode`` in the TOML config):
  - **sequential** (default): one agent session per rubric criterion.
  - **batch**: all criteria evaluated in a single agent session.

Produces (in ``output_dir``):
  reward.json  - Reward file ([0,1] reward)
  info.json    - Detailed per-criteria results + LLM usage
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from pydantic import TypeAdapter

from gandalf.config import (
    BatchCriterion,
    BatchJudgeInput,
    CriteriaResult,
    EvaluationInfo,
    GraderConfig,
    JudgeInput,
    LLMUsage,
    RubricItem,
    Verdict,
    load_config,
    load_rubric,
)
from gandalf.trajectory import load_trajectory_final_output

# Environment variables forwarded to the inner judge subprocess (via sudo).
# Only these are passed — everything else is stripped to avoid leaking secrets
# or host-specific state into the sandbox.
_JUDGE_ENV_ALLOWLIST = (
    "PATH",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "PYTHONPATH",
    "UV_TOOL_DIR",
    "UV_TOOL_BIN_DIR",
    "UV_PYTHON_INSTALL_DIR",
    # OpenTelemetry — forwarded so the inner judge can export traces
    # to any OTEL-compatible backend (e.g. Langfuse, Jaeger, Honeycomb).
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
)


def _judge_env_vars() -> list[str]:
    """Build the ``KEY=VALUE`` list for the judge subprocess environment."""
    return [f"{k}={v}" for k, v in os.environ.items() if k in _JUDGE_ENV_ALLOWLIST and v]


def _resolve_optional_file(
    inline: str | None,
    path: str | None,
    label: str,
) -> str | None:
    """Return *inline* content, or read from *path*, or ``None``.

    The caller is expected to ensure *inline* and *path* are mutually
    exclusive (enforced by ``GraderConfig``'s model validator).  If a
    path is given but does not exist, exits with a clear error.
    """
    if inline is not None:
        return inline
    if not path:
        return None
    if not os.path.isfile(path):
        print(  # noqa: T201
            f"ERROR: File not found: {path}\n  Configured via: {label}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        return f.read()


def resolve_judge_prompt(config: GraderConfig) -> str | None:
    """Resolve the custom judge prompt template (inline, path, or env var).

    Resolution order:
      1. config.judge_prompt (inline in TOML)
      2. config.judge_prompt_path (from TOML)
      3. GRADER_JUDGE_PROMPT_PATH env var
      4. No custom template (returns None, uses built-in)
    """
    path = config.judge_prompt_path or os.environ.get("GRADER_JUDGE_PROMPT_PATH")
    source = "judge_prompt_path in grader config" if config.judge_prompt_path else "GRADER_JUDGE_PROMPT_PATH env var"
    return _resolve_optional_file(
        config.judge_prompt,
        path,
        source,
    )


def resolve_judge_guidance(config: GraderConfig) -> str:
    """Resolve judge guidance content (inline, path, or env var).

    Resolution order:
      1. config.judge_guidance (inline in TOML)
      2. config.judge_guidance_path (from TOML)
      3. GRADER_JUDGE_GUIDANCE_PATH env var
      4. No guidance (empty string)
    """
    path = config.judge_guidance_path or os.environ.get("GRADER_JUDGE_GUIDANCE_PATH")
    source = (
        "judge_guidance_path in grader config" if config.judge_guidance_path else "GRADER_JUDGE_GUIDANCE_PATH env var"
    )
    result = _resolve_optional_file(
        config.judge_guidance,
        path,
        source,
    )
    return result or ""


def _clone_workspace(src: str) -> str:
    """Clone workspace into a temp directory accessible to the sandbox user.

    Walks the source tree once, skipping unreadable directories and files with
    a warning.  Each directory and file is made world-accessible inline so no
    second pass is needed.

    ``shutil.copytree`` is not used because its ``copy_function`` hook only
    covers per-file errors — directory listing errors (e.g. a 0o700 dir owned
    by the agent) cannot be caught there.
    """
    clone_dir = tempfile.mkdtemp(prefix="judge_workspace_", dir="/tmp")
    # Root dir is created by mkdtemp at 0o700; open it up immediately so
    # sandbox_user can traverse and write to it.
    os.chmod(clone_dir, 0o777)  # noqa: S103
    skipped: list[str] = []

    def _on_walk_error(err: OSError) -> None:
        skipped.append(err.filename or str(err))

    for dirpath, _dirnames, filenames in os.walk(src, onerror=_on_walk_error):
        rel = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(clone_dir, rel)
        os.makedirs(dst_dir, exist_ok=True)
        os.chmod(dst_dir, 0o777)  # noqa: S103

        for fname in filenames:
            src_file = os.path.join(dirpath, fname)
            dst_file = os.path.join(dst_dir, fname)
            try:
                shutil.copyfile(src_file, dst_file)
                # Preserve execute bits from source so scripts/binaries
                # remain runnable, while granting world read/write.
                src_mode = os.stat(src_file).st_mode
                os.chmod(dst_file, 0o666 | (src_mode & 0o111))
            except OSError:
                # Covers PermissionError, FileNotFoundError (broken symlinks),
                # IsADirectoryError (symlinks to dirs in filenames), etc.
                skipped.append(src_file)

    max_skipped_log = 20
    if skipped:
        print(  # noqa: T201
            f"[gandalf] workspace clone: skipped {len(skipped)} unreadable path(s):",
            file=sys.stderr,
        )
        for p in skipped[:max_skipped_log]:
            print(f"  - {p}", file=sys.stderr)  # noqa: T201
        if len(skipped) > max_skipped_log:
            print(f"  ... and {len(skipped) - max_skipped_log} more", file=sys.stderr)  # noqa: T201

    return clone_dir


class _JudgeSubprocessError(Exception):
    """Raised when the judge subprocess fails for any reason."""


def _run_judge_subprocess(
    judge_input: JudgeInput | BatchJudgeInput,
    sandbox_user: str | None,
    trace_path: str,
    timeout: int,
) -> Any:
    """Clone workspace, run the judge subprocess, and return parsed output JSON.

    Handles workspace cloning, temp file creation, command construction,
    subprocess execution, trace saving, and cleanup.

    Raises ``_JudgeSubprocessError`` on any failure (clone, timeout,
    non-zero exit, bad output).
    """
    batch = isinstance(judge_input, BatchJudgeInput)

    try:
        clone_dir = _clone_workspace(judge_input.workdir)
    except Exception as e:
        msg = f"Failed to clone workspace: {e}"
        raise _JudgeSubprocessError(msg) from e

    cloned_input = judge_input.model_copy(update={"workdir": clone_dir})

    prefix = "judge_batch_" if batch else "judge_"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{prefix}input_",
        dir=clone_dir,
        delete=False,
    ) as input_f:
        input_f.write(cloned_input.model_dump_json())
        input_path = input_f.name

    # Pre-create the output file so sandbox_user can write to it without
    # needing general write access to /tmp (which may not be world-writable).
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{prefix}output_",
        dir=clone_dir,
        delete=False,
    ) as output_f:
        output_path = output_f.name
    os.chmod(output_path, 0o666)  # noqa: S103

    try:
        os.chmod(input_path, 0o644)
        env_vars = [f"HOME={clone_dir}", *_judge_env_vars()]

        cmd = []
        if sandbox_user is not None:
            cmd += ["sudo", "-u", sandbox_user]
        cmd += [
            "env",
            *env_vars,
            "gandalf-the-grader-judge",
            "--input",
            input_path,
            "--output",
            output_path,
        ]
        if batch:
            cmd.append("--batch")

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=clone_dir,
        )

        _save_trace(trace_path, result.stdout, result.stderr, result.returncode)

        if result.returncode != 0:
            msg = f"Judge process failed (exit {result.returncode}): {result.stderr[:500]}"
            raise _JudgeSubprocessError(msg)

        with open(output_path) as f:
            return json.load(f)

    except subprocess.TimeoutExpired as e:
        _save_trace(trace_path, "", "Judge execution timed out.", -1)
        msg = "Judge execution timed out."
        raise _JudgeSubprocessError(msg) from e
    except (json.JSONDecodeError, FileNotFoundError) as e:
        msg = f"Failed to read judge output: {e}"
        raise _JudgeSubprocessError(msg) from e
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)


def evaluate_criteria(
    judge_input: JudgeInput,
    sandbox_user: str | None,
    trace_path: str,
    timeout: int = 300,
) -> tuple[Verdict, LLMUsage]:
    """Run the inner judge for a single criteria.

    When *sandbox_user* is set the judge is executed via ``sudo -u``.
    When it is ``None`` the judge runs as the ambient (current) user.
    """
    try:
        data = _run_judge_subprocess(judge_input, sandbox_user, trace_path, timeout)
    except _JudgeSubprocessError as e:
        return Verdict(met=None, reasoning=str(e)), LLMUsage()

    verdict = Verdict.model_validate(data["verdict"])
    usage = LLMUsage.model_validate(data["llm_usage"])
    return verdict, usage


def _fail_all(n: int, reason: str) -> list[Verdict]:
    """Return *n* fail verdicts that all share the same reason."""
    return [Verdict(met=None, reasoning=reason) for _ in range(n)]


def evaluate_all_criteria(
    judge_input: BatchJudgeInput,
    sandbox_user: str | None,
    trace_path: str,
    timeout: int = 300,
) -> tuple[list[Verdict], LLMUsage]:
    """Run the inner judge in batch mode -- all criteria in one agent session.

    Args:
        judge_input: Batch input with all context needed by the judge.
        sandbox_user: Username to run the judge process as (via sudo),
            or ``None`` to run as the ambient user.
        trace_path: Path to write the judge's stdout/stderr trace.
        timeout: Max seconds to wait for the judge to complete.

    Returns:
        (verdicts, llm_usage) where verdicts is a list of Verdict models
        positionally aligned with the input criteria, and llm_usage is the
        aggregate token/cost metrics for the batch session.
    """
    n_criteria = len(judge_input.criteria)

    try:
        data = _run_judge_subprocess(judge_input, sandbox_user, trace_path, timeout)
    except _JudgeSubprocessError as e:
        return _fail_all(n_criteria, str(e)), LLMUsage()

    verdicts = TypeAdapter(list[Verdict]).validate_python(data["verdicts"])
    llm_usage = LLMUsage.model_validate(data["llm_usage"])
    return verdicts, llm_usage


def _save_trace(trace_path: str, stdout: str, stderr: str, returncode: int) -> None:
    """Write the judge's stdout/stderr to a trace file."""
    with contextlib.suppress(OSError), open(trace_path, "w") as f:
        f.write(f"exit_code: {returncode}\n")
        f.write("=== stdout ===\n")
        f.write(stdout)
        f.write("\n=== stderr ===\n")
        f.write(stderr)


def _merge_usage(a: LLMUsage, b: LLMUsage) -> LLMUsage:
    """Sum two LLMUsage instances field-by-field."""
    return LLMUsage(
        cost_usd=a.cost_usd + b.cost_usd,
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
    )


def _run_sequential(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
    judge_prompt_template: str | None,
) -> tuple[list[CriteriaResult], LLMUsage]:
    """Evaluate each criterion in its own agent session.

    Returns (results, llm_usage) where llm_usage is the aggregated
    token/cost totals across all individual judge sessions.
    """
    results: list[CriteriaResult] = []
    total_usage = LLMUsage()
    for i, item in enumerate(rubric):
        print(f"[{i + 1}/{len(rubric)}] Evaluating: {item.criteria[:80]}...")  # noqa: T201

        judge_input = JudgeInput(
            model=config.model,
            instructions=config.instructions,
            final_output=final_output,
            criteria=item.criteria,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt_template=judge_prompt_template,
        )

        trace_path = os.path.join(config.output_dir, f"judge_trace_{i}.txt")
        verdict, usage = evaluate_criteria(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=config.judge_timeout,
        )

        total_usage = _merge_usage(total_usage, usage)

        result = CriteriaResult(
            criteria=item.criteria,
            weight=item.weight,
            met=verdict.met,
            reasoning=verdict.reasoning,
            evidence=verdict.evidence,
        )
        results.append(result)

        status = "MET" if result.met is True else ("ERROR" if result.met is None else "UNMET")
        print(f"  -> {status}: {result.reasoning[:120]}")  # noqa: T201

    return results, total_usage


def _run_batch(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
    judge_prompt_template: str | None,
) -> tuple[list[CriteriaResult], LLMUsage]:
    """Evaluate all criteria in a single agent session.

    Returns (results, llm_usage) where llm_usage is the token/cost
    totals from the single batch agent session.
    """
    criteria_list = [BatchCriterion(index=i, criteria=item.criteria) for i, item in enumerate(rubric)]

    n_criteria = len(criteria_list)
    batch_timeout = config.judge_timeout * n_criteria
    if config.batch_timeout is not None:
        batch_timeout = min(batch_timeout, config.batch_timeout)

    print(f"[batch] Evaluating all {n_criteria} criteria in one session (timeout={batch_timeout}s)...")  # noqa: T201

    judge_input = BatchJudgeInput(
        model=config.model,
        instructions=config.instructions,
        final_output=final_output,
        criteria=criteria_list,
        workdir=config.workdir,
        mcp_servers=config.mcp_servers,
        judge_guidance=judge_guidance,
        judge_prompt_template=judge_prompt_template,
    )

    trace_path = os.path.join(config.output_dir, "judge_trace_batch.txt")
    verdicts, llm_usage = evaluate_all_criteria(
        judge_input,
        sandbox_user=config.sandbox_user,
        trace_path=trace_path,
        timeout=batch_timeout,
    )

    results: list[CriteriaResult] = []
    for i, item in enumerate(rubric):
        v = verdicts[i] if i < len(verdicts) else Verdict(met=None, reasoning="No reasoning provided.")
        result = CriteriaResult(
            criteria=item.criteria,
            weight=item.weight,
            met=v.met,
            reasoning=v.reasoning,
            evidence=v.evidence,
        )
        results.append(result)

        status = "MET" if result.met is True else ("ERROR" if result.met is None else "UNMET")
        print(f"  [{i + 1}/{len(rubric)}] {status}: {result.reasoning[:120]}")  # noqa: T201

    return results, llm_usage


def _get_errored_indices(results: list[CriteriaResult]) -> list[int]:
    """Return indices of criteria where met is None (infrastructure error)."""
    return [i for i, r in enumerate(results) if r.met is None]


def _retry_sequential(
    config: GraderConfig,
    rubric: list[RubricItem],
    results: list[CriteriaResult],
    llm_usage: LLMUsage,
    final_output: str,
    judge_guidance: str,
    judge_prompt_template: str | None,
    errored_indices: list[int],
) -> LLMUsage:
    """Re-run each errored criterion individually and merge results in-place.

    Returns the updated cumulative LLMUsage.
    """
    for idx in errored_indices:
        item = rubric[idx]
        print(f"  [retry {idx}] Evaluating: {item.criteria[:80]}...")  # noqa: T201

        judge_input = JudgeInput(
            model=config.model,
            instructions=config.instructions,
            final_output=final_output,
            criteria=item.criteria,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt_template=judge_prompt_template,
        )

        trace_path = os.path.join(config.output_dir, f"judge_trace_{idx}_retry.txt")
        verdict, usage = evaluate_criteria(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=config.judge_timeout,
        )

        llm_usage = _merge_usage(llm_usage, usage)

        results[idx] = CriteriaResult(
            criteria=item.criteria,
            weight=item.weight,
            met=verdict.met,
            reasoning=verdict.reasoning,
            evidence=verdict.evidence,
        )

        status = "MET" if results[idx].met is True else ("ERROR" if results[idx].met is None else "UNMET")
        print(f"    -> {status}: {results[idx].reasoning[:120]}")  # noqa: T201

    return llm_usage


def _retry_batch(
    config: GraderConfig,
    rubric: list[RubricItem],
    results: list[CriteriaResult],
    llm_usage: LLMUsage,
    final_output: str,
    judge_guidance: str,
    judge_prompt_template: str | None,
    errored_indices: list[int],
) -> LLMUsage:
    """Re-run errored criteria as a batch and merge results in-place.

    Returns the updated cumulative LLMUsage.
    """
    retry_criteria = [
        BatchCriterion(index=new_idx, criteria=rubric[orig_idx].criteria)
        for new_idx, orig_idx in enumerate(errored_indices)
    ]

    n_retry = len(retry_criteria)
    batch_timeout = config.judge_timeout * n_retry
    if config.batch_timeout is not None:
        batch_timeout = min(batch_timeout, config.batch_timeout)

    print(f"  [retry batch] Re-evaluating {n_retry} criteria (timeout={batch_timeout}s)...")  # noqa: T201

    judge_input = BatchJudgeInput(
        model=config.model,
        instructions=config.instructions,
        final_output=final_output,
        criteria=retry_criteria,
        workdir=config.workdir,
        mcp_servers=config.mcp_servers,
        judge_guidance=judge_guidance,
        judge_prompt_template=judge_prompt_template,
    )

    trace_path = os.path.join(config.output_dir, "judge_trace_batch_retry.txt")
    verdicts, retry_usage = evaluate_all_criteria(
        judge_input,
        sandbox_user=config.sandbox_user,
        trace_path=trace_path,
        timeout=batch_timeout,
    )

    llm_usage = _merge_usage(llm_usage, retry_usage)

    for new_idx, orig_idx in enumerate(errored_indices):
        v = verdicts[new_idx] if new_idx < len(verdicts) else Verdict(met=None, reasoning="No reasoning provided.")
        results[orig_idx] = CriteriaResult(
            criteria=rubric[orig_idx].criteria,
            weight=rubric[orig_idx].weight,
            met=v.met,
            reasoning=v.reasoning,
            evidence=v.evidence,
        )

        met = results[orig_idx].met
        status = "MET" if met is True else ("ERROR" if met is None else "UNMET")
        print(f"    [{orig_idx}] {status}: {results[orig_idx].reasoning[:120]}")  # noqa: T201

    return llm_usage


def _write_info(
    config: GraderConfig,
    results: list[CriteriaResult],
    llm_usage: LLMUsage,
    errored_criteria_count: int,
) -> tuple[float, float]:
    """Compute reward and raw score and write info.json. Returns (reward, raw_score).

    raw_score: sum of weights for criteria whose condition was met (negative weights
    contribute when their criterion is met).  Errored criteria (met=None) contribute 0.

    reward: clip(0, 1, raw_score / sum_of_positive_weights), always in [0, 1].
    """
    raw_score = round(
        sum(r.weight for r in results if r.met is True),
        4,
    )

    minimum_score = round(sum(r.weight for r in results if r.weight < 0), 4)
    maximum_score = round(sum(r.weight for r in results if r.weight > 0), 4)

    reward = round(
        max(0.0, min(1.0, raw_score / maximum_score)) if maximum_score > 0 else 0.0,
        4,
    )

    n_total = len(results)
    n_evaluated = n_total - errored_criteria_count
    evaluated_pct = round((n_evaluated / n_total * 100.0) if n_total > 0 else 100.0, 2)

    info = EvaluationInfo(
        reward=reward,
        raw_score=raw_score,
        minimum_score=minimum_score,
        maximum_score=maximum_score,
        criteria_results=results,
        llm_usage=llm_usage,
        errored_criteria_count=errored_criteria_count,
        evaluated_criteria_pct=evaluated_pct,
    )
    with open(os.path.join(config.output_dir, "info.json"), "w") as f:
        f.write(info.model_dump_json(indent=2))

    return reward, raw_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Grader: evaluate agent output via agent-as-judge")
    parser.add_argument("--config", required=True, help="Path to grader config TOML file")
    parser.add_argument(
        "--mode",
        choices=["sequential", "batch"],
        default=None,
        help=(
            "Override the evaluation mode from config. "
            "'sequential' runs each criterion separately; "
            "'batch' evaluates all criteria in one agent session."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode is not None:
        config.mode = args.mode

    # The model validator guarantees exactly one of rubric / rubric_path is set.
    rubric = config.rubric if config.rubric is not None else load_rubric(config.rubric_path)  # type: ignore[arg-type]
    final_output = load_trajectory_final_output(config.trajectory_path)
    judge_guidance = resolve_judge_guidance(config)
    judge_prompt_template = resolve_judge_prompt(config)

    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Initial evaluation
    if config.mode == "batch":
        results, llm_usage = _run_batch(config, rubric, final_output, judge_guidance, judge_prompt_template)
    else:
        results, llm_usage = _run_sequential(config, rubric, final_output, judge_guidance, judge_prompt_template)

    # 2. Record initial error count for observability
    initial_errored = len(_get_errored_indices(results))

    # 3. Retry loop
    for attempt in range(config.judge_retries):
        errored = _get_errored_indices(results)
        if not errored:
            break
        print(f"\n[retry {attempt + 1}/{config.judge_retries}] Retrying {len(errored)} errored criteria...")  # noqa: T201
        if config.mode == "batch":
            llm_usage = _retry_batch(
                config, rubric, results, llm_usage, final_output, judge_guidance, judge_prompt_template, errored
            )
        else:
            llm_usage = _retry_sequential(
                config, rubric, results, llm_usage, final_output, judge_guidance, judge_prompt_template, errored
            )

    # 4. ALWAYS write info.json (even on hard fail)
    final_errored = _get_errored_indices(results)
    errored_count = len(final_errored)
    reward, raw_score = _write_info(config, results, llm_usage, errored_count)

    # 5. If any criteria still errored: do NOT write reward.json, exit 1
    if final_errored:
        print(  # noqa: T201
            f"\nERROR: {errored_count} criteria could not be evaluated "
            f"(initial errors: {initial_errored}, after retries: {errored_count}).",
            file=sys.stderr,
        )
        print(f"info.json written to {config.output_dir}/ (reward.json NOT written)", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # 6. All resolved — write reward.json
    with open(os.path.join(config.output_dir, "reward.json"), "w") as f:
        json.dump({"reward": reward}, f, indent=2)

    print(f"\nReward: {reward} (raw: {raw_score})")  # noqa: T201
    if llm_usage.cost_usd > 0:
        print(  # noqa: T201
            f"Grader LLM cost: ${llm_usage.cost_usd:.4f} "
            f"({len(rubric)} criteria, "
            f"{llm_usage.prompt_tokens} prompt + {llm_usage.completion_tokens} completion tokens)"
        )
    print(f"Mode: {config.mode}")  # noqa: T201
    if initial_errored > 0:
        print(f"Retried: {initial_errored} criteria recovered after retry")  # noqa: T201
    print(f"Results written to {config.output_dir}/")  # noqa: T201
