"""Outer grader orchestrator.

Runs as the grader user and spawns the inner judge as the sandbox user
(via sudo) to evaluate rubric criteria using an OpenHands agent-as-judge.

Supports two evaluation modes (configured via ``mode`` in the TOML config):
  - **batch** (default): all criteria evaluated in a single agent session.
  - **individual**: one agent session per rubric criterion.

When ``batch_splits`` is set (batch mode only), criteria are split into
positional chunks evaluated as separate batch sessions.  ``max_concurrency``
controls the maximum number of parallel judge sessions (for both modes).

Produces (in ``output_dir``):
  reward.json  - Reward file ([0,1] reward)
  info.json    - Detailed per-criterion results + LLM usage
"""

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

from pydantic import TypeAdapter

from gandalf.models import (
    BatchJudgeInput,
    CriterionResult,
    EvaluationInfo,
    GraderConfig,
    JudgeInput,
    LLMUsage,
    RubricItem,
    Verdict,
    load_config,
    load_rubric,
)


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


# Environment variables forwarded to the inner judge subprocess (via sudo).
# Only these are passed — everything else is stripped to avoid leaking secrets
# or host-specific state into the sandbox.
JUDGE_ENV_ALLOWLIST = frozenset(
    {
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
    }
)


def judge_env_vars() -> list[str]:
    """Build the ``KEY=VALUE`` list for the judge subprocess environment."""
    return [f"{k}={v}" for k, v in os.environ.items() if k in JUDGE_ENV_ALLOWLIST and v]


def resolve_optional_file(
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


def resolve_config_value(
    inline: str | None,
    config_path: str | None,
    env_var: str,
    config_label: str,
) -> str | None:
    """Resolve a config value: inline content → config path → env var → ``None``."""
    path = config_path or os.environ.get(env_var)
    source = f"{config_label} in grader config" if config_path else f"{env_var} env var"
    return resolve_optional_file(inline, path, source)


def resolve_instructions(config: GraderConfig) -> str:
    """Resolve task instructions (inline, path, or env var).

    Resolution order:
      1. config.instructions (inline in TOML)
      2. config.instructions_path (from TOML)
      3. GRADER_INSTRUCTIONS_PATH env var
      4. Error — instructions are required
    """
    result = resolve_config_value(
        config.instructions,
        config.instructions_path,
        "GRADER_INSTRUCTIONS_PATH",
        "instructions_path",
    )
    if not result:
        print(  # noqa: T201
            "ERROR: No instructions provided. Set 'instructions' or 'instructions_path' "
            "in the config, or the GRADER_INSTRUCTIONS_PATH env var.",
            file=sys.stderr,
        )
        sys.exit(1)
    return result


def resolve_judge_prompt(config: GraderConfig) -> str | None:
    """Resolve the custom judge prompt template (inline, path, or env var).

    Resolution order:
      1. config.judge_prompt (inline in TOML)
      2. config.judge_prompt_path (from TOML)
      3. GRADER_JUDGE_PROMPT_PATH env var
      4. No custom template (returns None, uses built-in)
    """
    return resolve_config_value(
        config.judge_prompt,
        config.judge_prompt_path,
        "GRADER_JUDGE_PROMPT_PATH",
        "judge_prompt_path",
    )


def resolve_judge_guidance(config: GraderConfig) -> str:
    """Resolve judge guidance content (inline, path, or env var).

    Resolution order:
      1. config.judge_guidance (inline in TOML)
      2. config.judge_guidance_path (from TOML)
      3. GRADER_JUDGE_GUIDANCE_PATH env var
      4. No guidance (empty string)
    """
    return (
        resolve_config_value(
            config.judge_guidance,
            config.judge_guidance_path,
            "GRADER_JUDGE_GUIDANCE_PATH",
            "judge_guidance_path",
        )
        or ""
    )


def clone_workspace(src: str) -> str:
    """Clone workspace into a temp directory accessible to the sandbox user.

    Walks the source tree once, skipping unreadable directories and files with
    a warning.  Each directory and file is made world-accessible inline so no
    second pass is needed.

    ``shutil.copytree`` is not used because its ``copy_function`` hook only
    covers per-file errors — directory listing errors (e.g. a 0o700 dir owned
    by the agent) cannot be caught there.
    """
    clone_dir = tempfile.mkdtemp(prefix="judge_workspace_")
    # Root dir is created by mkdtemp at 0o700; open it up immediately so
    # sandbox_user can traverse and write to it.
    os.chmod(clone_dir, 0o777)  # noqa: S103
    skipped: list[str] = []

    def on_walk_error(err: OSError) -> None:
        skipped.append(err.filename or str(err))

    for dirpath, _dirnames, filenames in os.walk(src, onerror=on_walk_error):
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


def run_judge(
    judge_input: JudgeInput | BatchJudgeInput,
    sandbox_user: str | None,
    trace_path: str,
    timeout: int = 300,
    *,
    workspace_is_disposable: bool = False,
) -> tuple[list[Verdict], LLMUsage]:
    """Run the judge subprocess against the workspace and return parsed verdicts.

    The workspace is normally cloned first — see ``clone_workspace``.  When
    ``workspace_is_disposable`` the caller has guaranteed both of the things the
    clone provides, so the judge runs against ``judge_input.workdir`` itself and a
    full copy of the workspace is not made.  See ``GraderConfig``.

    Always returns a *list* of verdicts, even for a single-criterion
    ``JudgeInput`` (one-element list).  On any subprocess failure every
    verdict is set to ``met=None`` with the error message.
    """
    batch = isinstance(judge_input, BatchJudgeInput)
    n = len(judge_input.criteria) if isinstance(judge_input, BatchJudgeInput) else 1

    def fail(msg: str) -> tuple[list[Verdict], LLMUsage]:
        return Verdict.errors(n, msg), LLMUsage()

    if workspace_is_disposable:
        workdir = judge_input.workdir
        # The judge's scratch files still need somewhere world-writable to live, and it
        # is not the workspace: these are the grader's own plumbing, and writing them
        # into the tree under evaluation puts the criteria list in front of the judge as
        # if it were part of the agent's work, and leaves them behind in anything that
        # later inspects the graded tree.
        try:
            scratch_dir = tempfile.mkdtemp(prefix="judge_scratch_")
            os.chmod(scratch_dir, 0o777)  # noqa: S103
        except OSError as e:
            return fail(f"Failed to create judge scratch directory: {e}")
    else:
        try:
            workdir = clone_workspace(judge_input.workdir)
        except Exception as e:  # noqa: BLE001
            return fail(f"Failed to clone workspace: {e}")
        scratch_dir = workdir

    # Removing the clone removes its scratch files with it; a borrowed workspace must
    # outlive the run, so only the scratch directory is ours to delete.
    cleanup_dir = scratch_dir if workspace_is_disposable else workdir

    cloned_input = judge_input.model_copy(update={"workdir": workdir})

    prefix = "judge_batch_" if batch else "judge_"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{prefix}input_",
        dir=scratch_dir,
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
        dir=scratch_dir,
        delete=False,
    ) as output_f:
        output_path = output_f.name
    os.chmod(output_path, 0o666)  # noqa: S103

    try:
        os.chmod(input_path, 0o644)
        # HOME tracks the working directory in both modes, so a judge that resolves
        # tooling through HOME behaves the same whether or not the workspace was cloned.
        env_vars = [f"HOME={workdir}", *judge_env_vars()]

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
            cwd=workdir,
        )

        save_trace(trace_path, result.stdout, result.stderr, result.returncode)

        if result.returncode != 0:
            return fail(f"Judge process failed (exit {result.returncode}): {result.stderr[:500]}")

        with open(output_path) as f:
            data = json.load(f)

    except subprocess.TimeoutExpired:
        save_trace(trace_path, "", "Judge execution timed out.", -1)
        return fail("Judge execution timed out.")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return fail(f"Failed to read judge output: {e}")
    else:
        if batch:
            verdicts = TypeAdapter(list[Verdict]).validate_python(data["verdicts"])
        else:
            verdicts = [Verdict.model_validate(data["verdict"])]
        usage = LLMUsage.model_validate(data["llm_usage"])
        return verdicts, usage
    finally:
        shutil.rmtree(cleanup_dir, ignore_errors=True)


def save_trace(trace_path: str, stdout: str, stderr: str, returncode: int) -> None:
    """Write the judge's stdout/stderr to a trace file."""
    with contextlib.suppress(OSError), open(trace_path, "w") as f:
        f.write(f"exit_code: {returncode}\n")
        f.write("=== stdout ===\n")
        f.write(stdout)
        f.write("\n=== stderr ===\n")
        f.write(stderr)


def format_status(*, met: bool | None) -> str:
    """Format criterion evaluation status for display."""
    if met is True:
        return "MET"
    if met is None:
        return "ERROR"
    return "UNMET"


def verdict_to_result(item: RubricItem, verdict: Verdict) -> CriterionResult:
    """Convert a Verdict into a CriterionResult for the given rubric item."""
    return CriterionResult(
        criterion=item.criterion,
        weight=item.weight,
        met=verdict.met,
        reasoning=verdict.reasoning,
        evidence=verdict.evidence,
    )


def run_individual(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    instructions: str,
    judge_guidance: str,
    judge_prompt: str | None,
    trace_suffix: str = "",
) -> tuple[list[CriterionResult], LLMUsage]:
    """Evaluate each rubric item in its own agent session.

    When max_concurrency > 1, up to N criteria are evaluated in parallel
    via a thread pool.  Results are always returned in rubric order.
    """
    n = len(rubric)
    concurrency = config.max_concurrency or 1

    def _eval_one(i: int, item: RubricItem) -> tuple[int, CriterionResult, LLMUsage]:
        print(f"[{i + 1}/{n}] Evaluating: {item.criterion[:80]}...")  # noqa: T201
        judge_input = JudgeInput(
            model=config.model,
            instructions=instructions,
            final_output=final_output,
            criterion=item.criterion,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt=judge_prompt,
        )
        trace_path = os.path.join(config.output_dir, f"judge_trace_{i}{trace_suffix}.txt")
        verdicts, usage = run_judge(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=config.judge_timeout,
            workspace_is_disposable=config.workspace_is_disposable,
        )
        result = verdict_to_result(item, verdicts[0])
        print(f"  [{i + 1}/{n}] {format_status(met=verdicts[0].met)}: {verdicts[0].reasoning[:120]}")  # noqa: T201
        return i, result, usage

    total_usage = LLMUsage()

    if concurrency == 1:
        # Serial path — no thread pool overhead
        results: list[CriterionResult] = []
        for i, item in enumerate(rubric):
            _, result, usage = _eval_one(i, item)
            results.append(result)
            total_usage = total_usage + usage
        return results, total_usage

    # Concurrent path
    print(f"[individual] Evaluating {n} criteria with max_concurrency={concurrency}")  # noqa: T201
    indexed_results: list[tuple[int, CriterionResult]] = []

    with ThreadPoolExecutor(max_workers=min(concurrency, n)) as executor:
        futures = [executor.submit(_eval_one, i, item) for i, item in enumerate(rubric)]
        for future in futures:
            i, result, usage = future.result()
            indexed_results.append((i, result))
            total_usage = total_usage + usage

    indexed_results.sort(key=lambda x: x[0])
    return [r for _, r in indexed_results], total_usage


def run_batch(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    instructions: str,
    judge_guidance: str,
    judge_prompt: str | None,
    trace_suffix: str = "",
) -> tuple[list[CriterionResult], LLMUsage]:
    """Evaluate all rubric items in a single agent session."""
    criteria = [item.criterion for item in rubric]
    n = len(criteria)

    batch_timeout = config.judge_timeout * n
    if config.batch_timeout is not None:
        batch_timeout = min(batch_timeout, config.batch_timeout)

    print(f"[batch] Evaluating {n} criteria in one session (timeout={batch_timeout}s)...")  # noqa: T201
    judge_input = BatchJudgeInput(
        model=config.model,
        instructions=instructions,
        final_output=final_output,
        criteria=criteria,
        workdir=config.workdir,
        mcp_servers=config.mcp_servers,
        judge_guidance=judge_guidance,
        judge_prompt=judge_prompt,
    )
    trace_path = os.path.join(config.output_dir, f"judge_trace_batch{trace_suffix}.txt")
    verdicts, usage = run_judge(
        judge_input,
        sandbox_user=config.sandbox_user,
        trace_path=trace_path,
        timeout=batch_timeout,
        workspace_is_disposable=config.workspace_is_disposable,
    )

    results: list[CriterionResult] = []
    for i, item in enumerate(rubric):
        v = verdicts[i] if i < len(verdicts) else Verdict(met=None, reasoning="No reasoning provided.")
        results.append(verdict_to_result(item, v))
        print(f"  [{i + 1}/{n}] {format_status(met=v.met)}: {v.reasoning[:120]}")  # noqa: T201
    return results, usage


def run_batch_concurrent(
    config: GraderConfig,
    rubric: list[RubricItem],
    final_output: str,
    instructions: str,
    judge_guidance: str,
    judge_prompt: str | None,
    trace_suffix: str = "",
) -> tuple[list[CriterionResult], LLMUsage]:
    """Split criteria into N positional chunks and evaluate each as a parallel batch.

    Each chunk is sent to its own judge subprocess.  All chunks run in parallel
    via a thread pool (each thread blocks on subprocess.run).  Results are merged
    back in original rubric order.
    """
    splits = config.batch_splits or 1
    concurrency = config.max_concurrency if config.max_concurrency is not None else splits
    n = len(rubric)
    if n == 0:
        return [], LLMUsage()
    chunk_size = math.ceil(n / splits)
    chunks: list[list[tuple[int, RubricItem]]] = [
        [(i, rubric[i]) for i in range(start, min(start + chunk_size, n))] for start in range(0, n, chunk_size)
    ]

    print(  # noqa: T201
        f"[batch-concurrent] Splitting {n} criteria into {len(chunks)} chunks "
        f"(max_concurrency={concurrency}, sizes: {', '.join(str(len(c)) for c in chunks)})"
    )

    def _run_split(
        split_idx: int, chunk: list[tuple[int, RubricItem]]
    ) -> tuple[list[tuple[int, CriterionResult]], LLMUsage]:
        # Use local 0-based indices for the judge — the prompt says
        # "0 through N-1" and read_batch_verdict filters by 0 <= idx < N.
        # Global rubric indices are restored when building indexed_results.
        criteria_list = [item.criterion for _orig_idx, item in chunk]

        n_criteria = len(criteria_list)
        batch_timeout = config.judge_timeout * n_criteria
        if config.batch_timeout is not None:
            batch_timeout = min(batch_timeout, config.batch_timeout)

        print(  # noqa: T201
            f"  [split {split_idx + 1}/{len(chunks)}] {n_criteria} criteria (timeout={batch_timeout}s)..."
        )

        judge_input = BatchJudgeInput(
            model=config.model,
            instructions=instructions,
            final_output=final_output,
            criteria=criteria_list,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
            judge_prompt=judge_prompt,
        )

        trace_path = os.path.join(config.output_dir, f"judge_trace_batch_split{split_idx}{trace_suffix}.txt")
        verdicts, usage = run_judge(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=batch_timeout,
            workspace_is_disposable=config.workspace_is_disposable,
        )

        indexed_results: list[tuple[int, CriterionResult]] = []
        for j, (orig_idx, item) in enumerate(chunk):
            v = verdicts[j] if j < len(verdicts) else Verdict(met=None, reasoning="No reasoning provided.")
            indexed_results.append((orig_idx, verdict_to_result(item, v)))
            print(  # noqa: T201
                f"    [{orig_idx + 1}/{n}] {format_status(met=v.met)}: {v.reasoning[:120]}"
            )

        return indexed_results, usage

    # Run all splits in parallel
    all_indexed_results: list[tuple[int, CriterionResult]] = []
    total_usage = LLMUsage()

    with ThreadPoolExecutor(max_workers=min(concurrency, len(chunks))) as executor:
        futures = [executor.submit(_run_split, idx, chunk) for idx, chunk in enumerate(chunks)]
        try:
            for future in futures:
                indexed_results, usage = future.result()
                all_indexed_results.extend(indexed_results)
                total_usage = total_usage + usage
        except Exception as exc:  # noqa: BLE001
            # All-or-nothing: if any split raises, we fail *all* criteria so
            # the hard-fail path in main() writes info.json but not reward.json.
            executor.shutdown(wait=True, cancel_futures=True)
            print(f"[batch-concurrent] Split failed unexpectedly: {exc}", file=sys.stderr)  # noqa: T201
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
                LLMUsage(),
            )

    # Sort back to original rubric order
    all_indexed_results.sort(key=lambda x: x[0])
    results = [r for _, r in all_indexed_results]

    return results, total_usage


def get_errored_indices(results: list[CriterionResult]) -> list[int]:
    """Return indices of criteria where met is None (infrastructure error)."""
    return [i for i, r in enumerate(results) if r.met is None]


def apply_retries(
    results: list[CriterionResult],
    retry_results: list[CriterionResult],
    errored_indices: list[int],
) -> list[CriterionResult]:
    """Return a new results list with retry outcomes spliced in at *errored_indices*."""
    retry_map = dict(zip(errored_indices, retry_results, strict=False))
    return [retry_map.get(i, r) for i, r in enumerate(results)]


def write_info(
    config: GraderConfig,
    results: list[CriterionResult],
    llm_usage: LLMUsage,
    errored_criterion_count: int,
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
    n_evaluated = n_total - errored_criterion_count
    evaluated_pct = round((n_evaluated / n_total * 100.0) if n_total > 0 else 100.0, 2)

    info = EvaluationInfo(
        reward=reward,
        raw_score=raw_score,
        minimum_score=minimum_score,
        maximum_score=maximum_score,
        criterion_results=results,
        llm_usage=llm_usage,
        errored_criterion_count=errored_criterion_count,
        evaluated_criteria_pct=evaluated_pct,
    )
    with open(os.path.join(config.output_dir, "info.json"), "w") as f:
        f.write(info.model_dump_json(indent=2))

    return reward, raw_score


def check_tmux_available() -> None:
    """Verify tmux is installed and usable.

    OpenHands silently falls back to a less stable subprocess-based terminal
    when tmux is missing; we refuse to run in that mode.

    Raises:
        RuntimeError: If tmux is missing or cannot be run.
    """
    # Same probe as OpenHands terminal factory (_is_tmux_available): run tmux and
    # require exit 0. OpenHands silently falls back to a subprocess terminal
    # when this fails; we fail fast instead.
    try:
        result = subprocess.run(
            ["tmux", "-V"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        ok = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        ok = False
    if not ok:
        msg = "tmux is not installed or not on PATH."
        raise RuntimeError(msg)


def preflight_check() -> None:
    """Validate runtime prerequisites before doing any real work."""
    check_tmux_available()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grader: evaluate agent output via agent-as-judge")
    parser.add_argument("--config", required=True, help="Path to grader config TOML file")
    args = parser.parse_args()

    preflight_check()

    config = load_config(args.config)

    instructions = resolve_instructions(config)

    # The model validator guarantees exactly one of rubric / rubric_path is set.
    if config.rubric is not None:
        rubric = config.rubric
    else:
        assert config.rubric_path is not None  # noqa: S101  # guaranteed by model validator
        rubric = load_rubric(config.rubric_path)
    final_output = load_trajectory_final_output(config.trajectory_path)
    judge_guidance = resolve_judge_guidance(config)
    judge_prompt = resolve_judge_prompt(config)

    os.makedirs(config.output_dir, exist_ok=True)

    if config.mode == "batch":
        splits = config.batch_splits
        concurrency = config.max_concurrency if config.max_concurrency is not None else (splits or 1)
        run = run_batch_concurrent if splits is not None else run_batch
    else:
        concurrency = config.max_concurrency or 1
        run = run_individual

    # 1. Initial evaluation
    results, llm_usage = run(config, rubric, final_output, instructions, judge_guidance, judge_prompt)

    # 2. Record initial error count for observability
    initial_errored = len(get_errored_indices(results))

    # 3. Retry loop — retries always use the non-concurrent variant
    retry_run = run_batch if config.mode == "batch" else run_individual
    for attempt in range(config.judge_retries):
        errored = get_errored_indices(results)
        if not errored:
            break
        print(f"\n[retry {attempt + 1}/{config.judge_retries}] Retrying {len(errored)} errored criteria...")  # noqa: T201
        retry_rubric = [rubric[i] for i in errored]
        retry_results, retry_usage = retry_run(
            config,
            retry_rubric,
            final_output,
            instructions,
            judge_guidance,
            judge_prompt,
            trace_suffix=f"_retry{attempt + 1}",
        )
        results = apply_retries(results, retry_results, errored)
        llm_usage = llm_usage + retry_usage

    # 4. ALWAYS write info.json (even on hard fail)
    final_errored = get_errored_indices(results)
    errored_count = len(final_errored)
    reward, raw_score = write_info(config, results, llm_usage, errored_count)

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
    if config.mode == "batch" and config.batch_splits is not None:
        mode_display = f"batch (batch_splits={config.batch_splits}, max_concurrency={concurrency})"
    elif concurrency > 1:
        mode_display = f"{config.mode} (max_concurrency={concurrency})"
    else:
        mode_display = config.mode
    print(f"Mode: {mode_display}")  # noqa: T201
    if initial_errored > 0:
        print(f"Retried: {initial_errored} criteria recovered after retry")  # noqa: T201
    print(f"Results written to {config.output_dir}/")  # noqa: T201
