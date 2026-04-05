"""Outer grader orchestrator.

Runs as the grader user and spawns the inner judge as the sandbox user
(via sudo) to evaluate rubric criteria using an OpenHands agent-as-judge.

Supports two evaluation modes (configured via ``mode`` in the TOML config):
  - **individual**: one agent session per rubric criterion.
  - **batch** (default): all criteria evaluated in a single agent session.

When ``max_concurrency`` > 1, multiple judge sessions run in parallel.
For batch mode this splits criteria into positional chunks; for individual
mode it runs multiple criterion evaluations concurrently.

Produces (in ``output_dir``):
  reward.json  - Reward file ([0,1] reward)
  info.json    - Detailed per-criterion results + LLM usage
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from gandalf.models import (
    BatchJudgeInput,
    CriterionResult,
    EvaluationInfo,
    GraderConfig,
    JudgeInput,
    RubricItem,
    load_config,
    load_rubric,
)

# Environment variables forwarded to the inner judge subprocess (via sudo).
# Only these are passed — everything else is stripped to avoid leaking secrets
# or host-specific state into the sandbox.
_JUDGE_ENV_ALLOWLIST = frozenset({
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
})


def load_trajectory_final_output(path: str) -> str:
    """Load an ATIF trajectory file and extract the final agent message."""
    with open(path) as f:
        data = json.load(f)

    steps = data.get("steps", [])

    # Extract final agent message (last with non-empty content, no tool calls)
    final_output = ""
    for step in reversed(steps):
        if step.get("source") == "agent" and not step.get("tool_calls"):
            msg = step.get("message", "")
            if msg.strip():
                final_output = msg
                break

    return final_output


def _judge_env_vars() -> list[str]:
    """Build the ``KEY=VALUE`` list for the judge subprocess environment."""
    return [f"{k}={v}" for k, v in os.environ.items() if k in _JUDGE_ENV_ALLOWLIST and v]


def _resolve_optional_file(
    inline: str | None,
    path: str | None,
    label: str,
) -> str | None:
    """Return *inline* content, or read from *path*, or ``None``.

    If a path is given but does not exist, exits with a clear error.
    """
    if inline is not None:
        return inline
    if not path:
        return None
    if not os.path.isfile(path):
        print(
            f"ERROR: File not found: {path}\n  Configured via: {label}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        return f.read()


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
        "judge_guidance_path in grader config"
        if config.judge_guidance_path
        else "GRADER_JUDGE_GUIDANCE_PATH env var"
    )
    return _resolve_optional_file(config.judge_guidance, path, source) or ""


def resolve_judge_prompt(config: GraderConfig) -> str | None:
    """Resolve the custom judge prompt template (inline, path, or env var).

    Resolution order:
      1. config.judge_prompt (inline in TOML)
      2. config.judge_prompt_path (from TOML)
      3. GRADER_JUDGE_PROMPT_PATH env var
      4. No custom template (returns None, uses built-in)
    """
    path = config.judge_prompt_path or os.environ.get("GRADER_JUDGE_PROMPT_PATH")
    source = (
        "judge_prompt_path in grader config"
        if config.judge_prompt_path
        else "GRADER_JUDGE_PROMPT_PATH env var"
    )
    return _resolve_optional_file(config.judge_prompt, path, source)


def _clone_workspace(src: str) -> str:
    """Clone workspace into a temp directory accessible to the sandbox user."""
    clone_dir = tempfile.mkdtemp(prefix="judge_workspace_")
    os.chmod(clone_dir, 0o777)
    skipped: list[str] = []

    def _on_walk_error(err: OSError) -> None:
        skipped.append(err.filename or str(err))

    for dirpath, _dirnames, filenames in os.walk(src, onerror=_on_walk_error):
        rel = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(clone_dir, rel)
        os.makedirs(dst_dir, exist_ok=True)
        os.chmod(dst_dir, 0o777)

        for fname in filenames:
            src_file = os.path.join(dirpath, fname)
            dst_file = os.path.join(dst_dir, fname)
            try:
                shutil.copyfile(src_file, dst_file)
                src_mode = os.stat(src_file).st_mode
                os.chmod(dst_file, 0o666 | (src_mode & 0o111))
            except OSError:
                skipped.append(src_file)

    if skipped:
        print(
            f"[gandalf] workspace clone: skipped {len(skipped)} unreadable path(s):",
            file=sys.stderr,
        )
        for p in skipped[:20]:
            print(f"  - {p}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more", file=sys.stderr)

    return clone_dir


def _fail_verdict(reason: str) -> dict[str, Any]:
    """Return a single fail verdict dict."""
    return {"met": None, "reasoning": reason, "evidence": []}


def _fail_all(n: int, reason: str) -> list[dict[str, Any]]:
    """Return *n* fail verdicts that all share the same reason."""
    return [{"index": i, **_fail_verdict(reason)} for i in range(n)]


def _run_judge(
    judge_input: JudgeInput | BatchJudgeInput,
    sandbox_user: str | None,
    trace_path: str,
    timeout: int = 300,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clone workspace, run the judge subprocess, and return parsed results."""
    is_batch = isinstance(judge_input, BatchJudgeInput)
    n = len(judge_input.criteria) if is_batch else 1

    try:
        clone_dir = _clone_workspace(judge_input.workdir)
    except Exception as e:
        if is_batch:
            return _fail_all(n, f"Failed to clone workspace: {e}"), {}
        return [_fail_verdict(f"Failed to clone workspace: {e}")], {}

    cloned_input = judge_input.model_copy(update={"workdir": clone_dir})

    prefix = "judge_batch_" if is_batch else "judge_"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{prefix}input_",
        dir=clone_dir,
        delete=False,
    ) as input_f:
        input_f.write(cloned_input.model_dump_json())
        input_path = input_f.name

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{prefix}output_",
        dir=clone_dir,
        delete=False,
    ) as output_f:
        output_path = output_f.name
    os.chmod(output_path, 0o666)

    try:
        os.chmod(input_path, 0o644)
        env_vars = [f"HOME={clone_dir}", *_judge_env_vars()]

        cmd: list[str] = []
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
        if is_batch:
            cmd.append("--batch")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=clone_dir,
        )

        _save_trace(trace_path, result.stdout, result.stderr, result.returncode)

        if result.returncode != 0:
            reason = f"Judge process failed (exit {result.returncode}): {result.stderr[:500]}"
            if is_batch:
                return _fail_all(n, reason), {}
            return [_fail_verdict(reason)], {}

        with open(output_path) as f:
            data = json.load(f)

        if is_batch:
            if isinstance(data, dict):
                verdicts = data.get("verdicts", [])
                llm_usage = data.get("llm_usage", {})
                return verdicts, llm_usage

            if isinstance(data, list):
                return data, {}

            reason = f"Unexpected JSON type from judge: {type(data).__name__}"
            return _fail_all(n, reason), {}
        else:
            llm_usage = data.get("llm_usage", {})
            return [data], llm_usage

    except subprocess.TimeoutExpired:
        _save_trace(trace_path, "", "Judge execution timed out.", -1)
        if is_batch:
            return _fail_all(n, "Judge execution timed out."), {}
        return [_fail_verdict("Judge execution timed out.")], {}
    except (json.JSONDecodeError, FileNotFoundError, TypeError, AttributeError) as e:
        if is_batch:
            return _fail_all(n, f"Failed to read judge output: {e}"), {}
        return [_fail_verdict(f"Failed to read judge output: {e}")], {}
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)
        for path in (input_path, output_path):
            with contextlib.suppress(OSError):
                os.unlink(path)


def _save_trace(trace_path: str, stdout: str, stderr: str, returncode: int) -> None:
    """Write the judge's stdout/stderr to a trace file."""
    with contextlib.suppress(OSError), open(trace_path, "w") as f:
        f.write(f"exit_code: {returncode}\n")
        f.write("=== stdout ===\n")
        f.write(stdout)
        f.write("\n=== stderr ===\n")
        f.write(stderr)


def _run_individual(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
    judge_prompt: str | None = None,
) -> tuple[list[CriterionResult], dict[str, Any]]:
    """Evaluate each criterion in its own agent session."""
    n = len(rubric)
    concurrency = config.max_concurrency or 1

    def _eval_one(i: int, item: RubricItem) -> tuple[int, CriterionResult, dict[str, Any]]:
        print(f"[{i + 1}/{n}] Evaluating: {item.criterion[:80]}...")

        judge_input = JudgeInput(
            model=config.model,
            instructions=config.instructions,
            final_output=final_output,
            criterion=item.criterion,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt=judge_prompt,
        )

        trace_path = os.path.join(config.output_dir, f"judge_trace_{i}.txt")
        verdicts, usage = _run_judge(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=config.judge_timeout,
        )

        v = verdicts[0]
        result = CriterionResult(
            criterion=item.criterion,
            weight=item.weight,
            met=v.get("met"),
            reasoning=v.get("reasoning", "No reasoning provided."),
            evidence=v.get("evidence", []),
        )

        status = "MET" if result.met is True else ("ERROR" if result.met is None else "UNMET")
        print(f"  [{i + 1}/{n}] {status}: {result.reasoning[:120]}")

        return i, result, usage

    total_usage: dict[str, float | int] = {}

    if concurrency == 1:
        results: list[CriterionResult] = []
        for i, item in enumerate(rubric):
            _, result, usage = _eval_one(i, item)
            results.append(result)
            for key in ("cost_usd", "prompt_tokens", "completion_tokens", "cache_read_tokens"):
                total_usage[key] = total_usage.get(key, 0) + usage.get(key, 0)
        return results, total_usage

    print(f"[individual] Evaluating {n} criteria with max_concurrency={concurrency}")
    indexed_results: list[tuple[int, CriterionResult]] = []

    with ThreadPoolExecutor(max_workers=min(concurrency, n)) as executor:
        futures = [
            executor.submit(_eval_one, i, item)
            for i, item in enumerate(rubric)
        ]
        for future in futures:
            i, result, usage = future.result()
            indexed_results.append((i, result))
            for key in ("cost_usd", "prompt_tokens", "completion_tokens", "cache_read_tokens"):
                total_usage[key] = total_usage.get(key, 0) + usage.get(key, 0)

    indexed_results.sort(key=lambda x: x[0])
    return [r for _, r in indexed_results], total_usage


def _run_batch(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
    judge_prompt: str | None = None,
) -> tuple[list[CriterionResult], dict[str, Any]]:
    """Evaluate all criteria in a single agent session."""
    criteria_list = [item.criterion for item in rubric]

    n_criteria = len(criteria_list)
    batch_timeout = config.judge_timeout * n_criteria
    if config.batch_timeout is not None:
        batch_timeout = min(batch_timeout, config.batch_timeout)

    print(
        f"[batch] Evaluating all {n_criteria} criteria in one session "
        f"(timeout={batch_timeout}s)..."
    )

    judge_input = BatchJudgeInput(
        model=config.model,
        instructions=config.instructions,
        final_output=final_output,
        criteria=criteria_list,
        workdir=config.workdir,
        mcp_servers=config.mcp_servers,
        judge_guidance=judge_guidance,
        judge_prompt=judge_prompt,
    )

    trace_path = os.path.join(config.output_dir, "judge_trace_batch.txt")
    verdicts, llm_usage = _run_judge(
        judge_input,
        sandbox_user=config.sandbox_user,
        trace_path=trace_path,
        timeout=batch_timeout,
    )

    results: list[CriterionResult] = []
    for i, item in enumerate(rubric):
        v = verdicts[i] if i < len(verdicts) else {}
        result = CriterionResult(
            criterion=item.criterion,
            weight=item.weight,
            met=v.get("met"),
            reasoning=v.get("reasoning", "No reasoning provided."),
            evidence=v.get("evidence", []),
        )
        results.append(result)

        status = "MET" if result.met is True else ("ERROR" if result.met is None else "UNMET")
        print(f"  [{i + 1}/{len(rubric)}] {status}: {result.reasoning[:120]}")

    return results, llm_usage


def _run_batch_concurrent(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
    judge_prompt: str | None = None,
) -> tuple[list[CriterionResult], dict[str, Any]]:
    """Split criteria into N positional chunks and evaluate each as a parallel batch."""
    concurrency = config.max_concurrency or 1
    n = len(rubric)
    if n == 0:
        return [], {}
    chunk_size = math.ceil(n / concurrency)
    chunks: list[list[tuple[int, RubricItem]]] = []
    for start in range(0, n, chunk_size):
        chunks.append([(i, rubric[i]) for i in range(start, min(start + chunk_size, n))])

    print(
        f"[batch-concurrent] Splitting {n} criteria into {len(chunks)} chunks "
        f"(sizes: {', '.join(str(len(c)) for c in chunks)})"
    )

    def _run_split(split_idx: int, chunk: list[tuple[int, RubricItem]]) -> tuple[list[tuple[int, CriterionResult]], dict[str, Any]]:
        criteria_list = [item.criterion for _orig_idx, item in chunk]

        n_criteria = len(criteria_list)
        batch_timeout = config.judge_timeout * n_criteria
        if config.batch_timeout is not None:
            batch_timeout = min(batch_timeout, config.batch_timeout)

        print(
            f"  [split {split_idx + 1}/{len(chunks)}] "
            f"{n_criteria} criteria (timeout={batch_timeout}s)..."
        )

        judge_input = BatchJudgeInput(
            model=config.model,
            instructions=config.instructions,
            final_output=final_output,
            criteria=criteria_list,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt=judge_prompt,
        )

        trace_path = os.path.join(
            config.output_dir, f"judge_trace_batch_split{split_idx}.txt"
        )
        verdicts, llm_usage = _run_judge(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=batch_timeout,
        )

        indexed_results: list[tuple[int, CriterionResult]] = []
        for j, (orig_idx, item) in enumerate(chunk):
            v = verdicts[j] if j < len(verdicts) else {}
            result = CriterionResult(
                criterion=item.criterion,
                weight=item.weight,
                met=v.get("met"),
                reasoning=v.get("reasoning", "No reasoning provided."),
                evidence=v.get("evidence", []),
            )
            indexed_results.append((orig_idx, result))

            status = "MET" if result.met is True else ("ERROR" if result.met is None else "UNMET")
            print(
                f"    [{orig_idx + 1}/{n}] {status}: {result.reasoning[:120]}"
            )

        return indexed_results, llm_usage

    all_indexed_results: list[tuple[int, CriterionResult]] = []
    total_usage: dict[str, float | int] = {}

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(_run_split, idx, chunk)
            for idx, chunk in enumerate(chunks)
        ]
        try:
            for future in futures:
                indexed_results, usage = future.result()
                all_indexed_results.extend(indexed_results)
                for key in ("cost_usd", "prompt_tokens", "completion_tokens", "cache_read_tokens"):
                    total_usage[key] = total_usage.get(key, 0) + usage.get(key, 0)
        except Exception as exc:
            executor.shutdown(wait=True, cancel_futures=True)
            print(f"[batch-concurrent] Split failed unexpectedly: {exc}", file=sys.stderr)
            return (
                [
                    CriterionResult(
                        criterion=item.criterion,
                        weight=item.weight,
                        met=None,
                        reasoning=f"Batch split failed: {exc}",
                    )
                    for item in rubric
                ],
                {},
            )

    all_indexed_results.sort(key=lambda x: x[0])
    results = [r for _, r in all_indexed_results]

    return results, total_usage


def _get_errored_indices(results: list[CriterionResult]) -> list[int]:
    """Return indices of criteria where met is None (infrastructure error)."""
    return [i for i, r in enumerate(results) if r.met is None]


def _retry_individual(
    config: GraderConfig,
    rubric: list[RubricItem],
    results: list[CriterionResult],
    llm_usage: dict[str, Any],
    final_output: str,
    judge_guidance: str,
    errored_indices: list[int],
    judge_prompt: str | None = None,
) -> None:
    """Re-run each errored criterion individually and merge results in-place."""
    for idx in errored_indices:
        item = rubric[idx]
        print(f"  [retry {idx}] Evaluating: {item.criterion[:80]}...")

        judge_input = JudgeInput(
            model=config.model,
            instructions=config.instructions,
            final_output=final_output,
            criterion=item.criterion,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt=judge_prompt,
        )

        trace_path = os.path.join(config.output_dir, f"judge_trace_{idx}_retry.txt")
        verdicts, usage = _run_judge(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=config.judge_timeout,
        )

        for key in ("cost_usd", "prompt_tokens", "completion_tokens", "cache_read_tokens"):
            llm_usage[key] = llm_usage.get(key, 0) + usage.get(key, 0)

        v = verdicts[0]
        results[idx] = CriterionResult(
            criterion=item.criterion,
            weight=item.weight,
            met=v.get("met"),
            reasoning=v.get("reasoning", "No reasoning provided."),
            evidence=v.get("evidence", []),
        )

        status = "MET" if results[idx].met is True else ("ERROR" if results[idx].met is None else "UNMET")
        print(f"    -> {status}: {results[idx].reasoning[:120]}")


def _retry_batch(
    config: GraderConfig,
    rubric: list[RubricItem],
    results: list[CriterionResult],
    llm_usage: dict[str, Any],
    final_output: str,
    judge_guidance: str,
    errored_indices: list[int],
    judge_prompt: str | None = None,
) -> None:
    """Re-run errored criteria as a batch and merge results in-place."""
    retry_criteria = [rubric[orig_idx].criterion for orig_idx in errored_indices]

    n_retry = len(retry_criteria)
    batch_timeout = config.judge_timeout * n_retry
    if config.batch_timeout is not None:
        batch_timeout = min(batch_timeout, config.batch_timeout)

    print(f"  [retry batch] Re-evaluating {n_retry} criteria (timeout={batch_timeout}s)...")

    judge_input = BatchJudgeInput(
        model=config.model,
        instructions=config.instructions,
        final_output=final_output,
        criteria=retry_criteria,
        workdir=config.workdir,
        mcp_servers=config.mcp_servers,
        judge_guidance=judge_guidance,
        judge_prompt=judge_prompt,
    )

    trace_path = os.path.join(config.output_dir, "judge_trace_batch_retry.txt")
    verdicts, retry_usage = _run_judge(
        judge_input,
        sandbox_user=config.sandbox_user,
        trace_path=trace_path,
        timeout=batch_timeout,
    )

    for key in ("cost_usd", "prompt_tokens", "completion_tokens", "cache_read_tokens"):
        llm_usage[key] = llm_usage.get(key, 0) + retry_usage.get(key, 0)

    for new_idx, orig_idx in enumerate(errored_indices):
        v = verdicts[new_idx] if new_idx < len(verdicts) else {}
        results[orig_idx] = CriterionResult(
            criterion=rubric[orig_idx].criterion,
            weight=rubric[orig_idx].weight,
            met=v.get("met"),
            reasoning=v.get("reasoning", "No reasoning provided."),
            evidence=v.get("evidence", []),
        )

        met = results[orig_idx].met
        status = "MET" if met is True else ("ERROR" if met is None else "UNMET")
        print(f"    [{orig_idx}] {status}: {results[orig_idx].reasoning[:120]}")


def _write_info(
    config: GraderConfig,
    results: list[CriterionResult],
    llm_usage: dict[str, Any],
    errored_criterion_count: int,
) -> tuple[float, float]:
    """Compute reward and raw score, ALWAYS write info.json. Returns (reward, raw_score)."""
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
    n_evaluated = n_total - errored_criterion_count
    evaluated_pct = round((n_evaluated / n_total * 100.0) if n_total > 0 else 100.0, 2)

    info = EvaluationInfo(
        reward=reward,
        raw_score=raw_score,
        minimum_score=minimum_score,
        maximum_score=maximum_score,
        criterion_results=results,
        llm_usage={
            "model": config.model,
            "total_cost_usd": llm_usage.get("cost_usd", 0),
            "total_prompt_tokens": llm_usage.get("prompt_tokens", 0),
            "total_completion_tokens": llm_usage.get("completion_tokens", 0),
            "total_cache_read_tokens": llm_usage.get("cache_read_tokens", 0),
        },
        errored_criterion_count=errored_criterion_count,
        evaluated_criteria_pct=evaluated_pct,
    )
    with open(os.path.join(config.output_dir, "info.json"), "w") as f:
        f.write(info.model_dump_json(indent=2))

    return reward, raw_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grader: evaluate agent output via agent-as-judge"
    )
    parser.add_argument(
        "--config", required=True, help="Path to grader config TOML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if config.rubric is not None:
        rubric = config.rubric
    else:
        assert config.rubric_path is not None
        rubric = load_rubric(config.rubric_path)
    final_output = load_trajectory_final_output(config.trajectory_path)
    judge_guidance = resolve_judge_guidance(config)
    judge_prompt = resolve_judge_prompt(config)

    os.makedirs(config.output_dir, exist_ok=True)

    concurrency = config.max_concurrency or 1

    # 1. Initial evaluation
    if config.mode == "batch":
        if concurrency > 1:
            results, llm_usage = _run_batch_concurrent(config, rubric, final_output, judge_guidance, judge_prompt)
        else:
            results, llm_usage = _run_batch(config, rubric, final_output, judge_guidance, judge_prompt)
    else:
        results, llm_usage = _run_individual(config, rubric, final_output, judge_guidance, judge_prompt)

    # 2. Record initial error count for observability
    initial_errored = len(_get_errored_indices(results))

    # 3. Retry loop
    for attempt in range(config.judge_retries):
        errored = _get_errored_indices(results)
        if not errored:
            break
        print(f"\n[retry {attempt + 1}/{config.judge_retries}] Retrying {len(errored)} errored criteria...")
        if config.mode == "batch":
            _retry_batch(config, rubric, results, llm_usage, final_output, judge_guidance, errored, judge_prompt)
        else:
            _retry_individual(config, rubric, results, llm_usage, final_output, judge_guidance, errored, judge_prompt)

    # 4. ALWAYS write info.json (even on hard fail)
    final_errored = _get_errored_indices(results)
    errored_count = len(final_errored)
    reward, raw_score = _write_info(config, results, llm_usage, errored_count)

    total_cost = llm_usage.get("cost_usd", 0)
    total_prompt = llm_usage.get("prompt_tokens", 0)
    total_completion = llm_usage.get("completion_tokens", 0)

    # 5. If any criteria still errored: do NOT write reward.json, exit 1
    if final_errored:
        print(
            f"\nERROR: {errored_count} criteria could not be evaluated "
            f"(initial errors: {initial_errored}, after retries: {errored_count}).",
            file=sys.stderr,
        )
        print(f"info.json written to {config.output_dir}/ (reward.json NOT written)", file=sys.stderr)
        sys.exit(1)

    # 6. All resolved — write reward.json
    with open(os.path.join(config.output_dir, "reward.json"), "w") as f:
        json.dump({"reward": reward}, f, indent=2)

    print(f"\nReward: {reward} (raw: {raw_score})")
    if total_cost > 0:
        print(
            f"Grader LLM cost: ${total_cost:.4f} "
            f"({len(rubric)} criteria, "
            f"{total_prompt} prompt + {total_completion} completion tokens)"
        )
    mode_str = config.mode
    if concurrency > 1:
        mode_str += f" (max_concurrency={concurrency})"
    print(f"Mode: {mode_str}")
    if initial_errored > 0:
        print(f"Retried: {initial_errored} criteria recovered after retry")
    print(f"Results written to {config.output_dir}/")


if __name__ == "__main__":
    main()
