"""Outer verifier orchestrator.

Runs as the verifier user and spawns the inner judge as the sandbox user
(via sudo) to evaluate rubric criteria using an OpenHands agent-as-judge.

Supports two evaluation modes (configured via ``mode`` in the TOML config):
  - **sequential** (default): one agent session per rubric criterion.
  - **batch**: all criteria evaluated in a single agent session.

Produces:
  /logs/verifier/reward.json  - Reward file (weighted score)
  /logs/verifier/info.json    - Detailed per-criteria results + LLM usage
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

from gandalf_grader.config import (
    BatchCriterion,
    BatchJudgeInput,
    CriteriaResult,
    EvaluationInfo,
    JudgeInput,
    RubricItem,
    VerifierConfig,
    load_config,
    load_rubric,
)
from gandalf_grader.trajectory import load_trajectory_final_output

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


def resolve_judge_guidance(config: VerifierConfig) -> str:
    """Resolve and load judge guidance content.

    Resolution order:
      1. config.judge_guidance_path (from TOML)
      2. VERIFIER_JUDGE_GUIDANCE_PATH env var
      3. No guidance (empty string)

    If a path is resolved but the file does not exist, raises SystemExit
    with a clear error message.
    """
    path = config.judge_guidance_path or os.environ.get("VERIFIER_JUDGE_GUIDANCE_PATH")
    if not path:
        return ""
    if not os.path.isfile(path):
        source = (
            "judge_guidance_path in verifier config"
            if config.judge_guidance_path
            else "VERIFIER_JUDGE_GUIDANCE_PATH env var"
        )
        print(
            f"ERROR: Judge guidance file not found: {path}\n"
            f"  Configured via: {source}\n"
            f"  Fix: ensure the file exists at that path, or remove the setting to run without guidance.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        return f.read()


def _clone_workspace(src: str) -> str:
    """Clone workspace into a temp directory accessible to the sandbox user."""
    clone_dir = tempfile.mkdtemp(prefix="judge_workspace_", dir="/tmp")
    shutil.copytree(src, clone_dir, dirs_exist_ok=True)

    # Make the clone group-writable so the sandbox user (in the verifier group)
    # can use it as a normal workspace.
    for dirpath, _dirnames, filenames in os.walk(clone_dir):
        os.chmod(dirpath, os.stat(dirpath).st_mode | 0o070)
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            os.chmod(fpath, os.stat(fpath).st_mode | 0o060)
    return clone_dir


def evaluate_criteria(
    judge_input: JudgeInput,
    sandbox_user: str,
    trace_path: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run the inner judge as the sandbox user for a single criteria."""
    try:
        clone_dir = _clone_workspace(judge_input.workdir)
    except Exception as e:
        return {"passed": False, "reasoning": f"Failed to clone workspace: {e}"}

    cloned_input = judge_input.model_copy(update={"workdir": clone_dir})

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="judge_input_",
        dir="/tmp",
        delete=False,
    ) as input_f:
        input_f.write(cloned_input.model_dump_json())
        input_path = input_f.name

    output_path = tempfile.mktemp(suffix=".json", prefix="judge_output_", dir="/tmp")

    try:
        os.chmod(input_path, 0o644)
        cmd = [
            "sudo",
            "-u",
            sandbox_user,
            "env",
            *_judge_env_vars(),
            "gandalf-grader-judge",
            "--input",
            input_path,
            "--output",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=clone_dir,
        )

        _save_trace(trace_path, result.stdout, result.stderr, result.returncode)

        if result.returncode != 0:
            return {
                "passed": False,
                "reasoning": f"Judge process failed (exit {result.returncode}): {result.stderr[:500]}",
            }

        with open(output_path) as f:
            result_data: dict[str, Any] = json.load(f)
            return result_data

    except subprocess.TimeoutExpired:
        _save_trace(trace_path, "", "Judge execution timed out.", -1)
        return {"passed": False, "reasoning": "Judge execution timed out."}
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return {"passed": False, "reasoning": f"Failed to read judge output: {e}"}
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)
        for path in (input_path, output_path):
            with contextlib.suppress(OSError):
                os.unlink(path)


def _fail_all(n: int, reason: str) -> list[dict[str, Any]]:
    """Return *n* fail verdicts that all share the same reason."""
    return [{"index": i, "passed": False, "reasoning": reason, "evidence": []} for i in range(n)]


def _run_with_live_trace(
    cmd: list[str],
    cwd: str,
    trace_path: str,
    timeout: int,
) -> tuple[int, str, str, bool]:
    """Run a subprocess while streaming stdout/stderr into ``trace_path``."""
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    write_lock = threading.Lock()

    with open(trace_path, "w") as trace_f:
        trace_f.write("exit_code: running\n")
        trace_f.write("=== live output ===\n")
        trace_f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            bufsize=1,
        )

        def _pump(
            stream: Any,
            label: str,
            sink: list[str],
        ) -> None:
            if stream is None:
                return
            try:
                for line in iter(stream.readline, ""):
                    sink.append(line)
                    with write_lock:
                        trace_f.write(f"[{label}] {line}")
                        trace_f.flush()
            finally:
                with contextlib.suppress(Exception):
                    stream.close()

        out_thread = threading.Thread(target=_pump, args=(proc.stdout, "stdout", stdout_chunks), daemon=True)
        err_thread = threading.Thread(target=_pump, args=(proc.stderr, "stderr", stderr_chunks), daemon=True)
        out_thread.start()
        err_thread.start()

        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            returncode = -1
            with write_lock:
                trace_f.write("[verifier] Batch judge execution timed out.\n")
                trace_f.flush()

        out_thread.join()
        err_thread.join()

        with write_lock:
            trace_f.write(f"\nexit_code: {returncode}\n")
            if timed_out:
                trace_f.write("timeout: true\n")
            trace_f.flush()

    return returncode, "".join(stdout_chunks), "".join(stderr_chunks), timed_out


def evaluate_all_criteria(
    judge_input: BatchJudgeInput,
    sandbox_user: str,
    trace_path: str,
    timeout: int = 300,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the inner judge in batch mode -- all criteria in one agent session.

    Args:
        judge_input: Batch input with all context needed by the judge.
        sandbox_user: Username to run the judge process as (via sudo).
        trace_path: Path to write the judge's stdout/stderr trace.
        timeout: Max seconds to wait for the judge to complete.

    Returns:
        (verdicts, llm_usage) where verdicts is a list of dicts each with
        ``index``, ``passed``, ``reasoning``, ``evidence``, and llm_usage
        is the aggregate token/cost dict for the single batch session.
    """
    n_criteria = len(judge_input.criteria)

    try:
        clone_dir = _clone_workspace(judge_input.workdir)
    except Exception as e:
        return _fail_all(n_criteria, f"Failed to clone workspace: {e}"), {}

    cloned_input = judge_input.model_copy(update={"workdir": clone_dir})

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="judge_batch_input_",
        dir="/tmp",
        delete=False,
    ) as input_f:
        input_f.write(cloned_input.model_dump_json())
        input_path = input_f.name

    output_path = tempfile.mktemp(
        suffix=".json",
        prefix="judge_batch_output_",
        dir="/tmp",
    )

    try:
        os.chmod(input_path, 0o644)

        cmd = [
            "sudo",
            "-u",
            sandbox_user,
            "env",
            *_judge_env_vars(),
            "gandalf-grader-judge",
            "--input",
            input_path,
            "--output",
            output_path,
            "--batch",
        ]

        returncode, _stdout, stderr, timed_out = _run_with_live_trace(
            cmd=cmd,
            cwd=clone_dir,
            trace_path=trace_path,
            timeout=timeout,
        )

        if timed_out:
            return _fail_all(n_criteria, "Judge execution timed out."), {}

        if returncode != 0:
            reason = f"Judge process failed (exit {returncode}): {stderr[:500]}"
            return _fail_all(n_criteria, reason), {}

        with open(output_path) as f:
            data = json.load(f)

            if isinstance(data, dict):
                verdicts = data.get("verdicts", [])
                llm_usage = data.get("llm_usage", {})
                return verdicts, llm_usage

            if isinstance(data, list):
                # Legacy format: bare JSON array of verdicts, no usage info.
                return data, {}

            reason = f"Unexpected JSON type from judge: {type(data).__name__}"
            return _fail_all(n_criteria, reason), {}

    except (json.JSONDecodeError, FileNotFoundError, TypeError, AttributeError) as e:
        return _fail_all(n_criteria, f"Failed to read judge output: {e}"), {}
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


def _run_sequential(
    config: VerifierConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
) -> tuple[list[CriteriaResult], dict[str, Any]]:
    """Evaluate each criterion in its own agent session.

    Returns (results, llm_usage) where llm_usage is the aggregated
    token/cost totals across all individual judge sessions.
    """
    results: list[CriteriaResult] = []
    total_usage: dict[str, float | int] = {}
    for i, item in enumerate(rubric):
        print(f"[{i + 1}/{len(rubric)}] Evaluating: {item.criteria[:80]}...")

        judge_input = JudgeInput(
            model=config.model,
            instructions=config.instructions,
            final_output=final_output,
            criteria=item.criteria,
            workdir=config.workdir,
            mcp_servers=config.mcp_servers,
            judge_guidance=judge_guidance,
        )

        trace_path = os.path.join(config.output_dir, f"judge_trace_{i}.txt")
        verdict = evaluate_criteria(
            judge_input,
            sandbox_user=config.sandbox_user,
            trace_path=trace_path,
            timeout=config.judge_timeout,
        )

        usage = verdict.get("llm_usage", {})
        for key in ("cost_usd", "prompt_tokens", "completion_tokens", "cache_read_tokens"):
            total_usage[key] = total_usage.get(key, 0) + usage.get(key, 0)

        result = CriteriaResult(
            criteria=item.criteria,
            weight=item.weight,
            passed=verdict.get("passed", False),
            reasoning=verdict.get("reasoning", "No reasoning provided."),
            evidence=verdict.get("evidence", []),
        )
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status}: {result.reasoning[:120]}")

    return results, total_usage


def _run_batch(
    config: VerifierConfig,
    rubric: list[RubricItem],
    final_output: str,
    judge_guidance: str,
) -> tuple[list[CriteriaResult], dict[str, Any]]:
    """Evaluate all criteria in a single agent session.

    Returns (results, llm_usage) where llm_usage is the token/cost
    totals from the single batch agent session.
    """
    criteria_list = [
        BatchCriterion(index=i, criteria=item.criteria, weight=item.weight)
        for i, item in enumerate(rubric)
    ]

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
        v = verdicts[i] if i < len(verdicts) else {}
        result = CriteriaResult(
            criteria=item.criteria,
            weight=item.weight,
            passed=v.get("passed", False),
            reasoning=v.get("reasoning", "No reasoning provided."),
            evidence=v.get("evidence", []),
        )
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  [{i + 1}/{len(rubric)}] {status}: {result.reasoning[:120]}")

    return results, llm_usage


def _compute_and_write_results(
    config: VerifierConfig,
    rubric: list[RubricItem],
    results: list[CriteriaResult],
    llm_usage: dict[str, Any],
) -> None:
    """Compute the weighted score and write reward.json / info.json."""
    total_weight = sum(r.weight for r in results)
    score = round(
        sum(r.weight * (1.0 if r.passed else 0.0) for r in results) / total_weight if total_weight > 0 else 0.0,
        4,
    )

    total_cost = llm_usage.get("cost_usd", 0)
    total_prompt = llm_usage.get("prompt_tokens", 0)
    total_completion = llm_usage.get("completion_tokens", 0)
    total_cache_read = llm_usage.get("cache_read_tokens", 0)

    with open(os.path.join(config.output_dir, "reward.json"), "w") as f:
        json.dump({"score": score}, f, indent=2)

    info = EvaluationInfo(
        score=score,
        criteria_results=results,
        llm_usage={
            "model": config.model,
            "total_cost_usd": total_cost,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cache_read_tokens": total_cache_read,
        },
    )
    with open(os.path.join(config.output_dir, "info.json"), "w") as f:
        f.write(info.model_dump_json(indent=2))

    print(f"\nScore: {score}")
    if total_cost > 0:
        print(
            f"Verifier LLM cost: ${total_cost:.4f} "
            f"({len(rubric)} criteria, "
            f"{total_prompt} prompt + {total_completion} completion tokens)"
        )
    print(f"Mode: {config.mode}")
    print(f"Results written to {config.output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifier: evaluate agent output via agent-as-judge"
    )
    parser.add_argument(
        "--config", required=True, help="Path to verifier config TOML file"
    )
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

    rubric = load_rubric(config.rubric_path)
    final_output = load_trajectory_final_output(config.trajectory_path)
    judge_guidance = resolve_judge_guidance(config)

    os.makedirs(config.output_dir, exist_ok=True)

    if config.mode == "batch":
        results, llm_usage = _run_batch(config, rubric, final_output, judge_guidance)
    else:
        results, llm_usage = _run_sequential(config, rubric, final_output, judge_guidance)

    _compute_and_write_results(config, rubric, results, llm_usage)

    # If no criterion was actually evaluated by the judge (all failed due to
    # infrastructure issues like sudo, missing API keys, workspace clone errors),
    # exit non-zero so the caller can distinguish infrastructure errors from
    # a legitimate score of 0.0.
    evaluated = any(r.passed or r.evidence for r in results)
    if results and not evaluated:
        print(
            "\nERROR: No criteria were successfully evaluated. "
            "All failures appear to be infrastructure errors.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
