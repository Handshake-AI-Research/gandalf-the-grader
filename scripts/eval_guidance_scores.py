#!/usr/bin/env python3
"""Run Gandalf rubric or guidance scoring over indexed Harbor rollout trials."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

Mode = Literal["rubric", "guidance"]
MIN_QUOTED_ENV_VALUE_LENGTH = 2
DEFAULT_GEMINI_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "openai/gpt-5.5"
DEFAULT_ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5-20251001"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines into a list of dictionaries."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def toml_str(value: str) -> str:
    """Return a TOML string literal."""
    return json.dumps(value)


def load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE lines from an env file without printing values."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key.startswith("#"):
            continue
        if len(value) >= MIN_QUOTED_ENV_VALUE_LENGTH and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def build_grader_env(env_file: Path | None) -> dict[str, str]:
    """Build the environment used for Gandalf subprocesses."""
    env = os.environ.copy()
    if env_file is not None:
        env.update(load_env_file(env_file))

    return env


def choose_default_model(env: dict[str, str]) -> str:
    """Choose a default eval model that matches the available provider keys."""
    if env.get("GANDALF_EVAL_MODEL"):
        return env["GANDALF_EVAL_MODEL"]
    if env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY"):
        return DEFAULT_GEMINI_MODEL
    if env.get("OPENAI_API_KEY"):
        return DEFAULT_OPENAI_MODEL
    if env.get("ANTHROPIC_API_KEY"):
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_GEMINI_MODEL


def api_key_env_names_for_model(model: str) -> tuple[str, ...]:
    """Return provider API key env vars suitable for a LiteLLM model string."""
    normalized = model.lower()
    if normalized.startswith(("gemini/", "google/")):
        return ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if normalized.startswith(("openai/", "gpt-")):
        return ("OPENAI_API_KEY",)
    if normalized.startswith(("anthropic/", "claude-")):
        return ("ANTHROPIC_API_KEY",)
    if normalized.startswith("openrouter/"):
        return ("OPENROUTER_API_KEY",)
    return ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")


def set_llm_api_key_for_model(env: dict[str, str], model: str) -> None:
    """Populate LLM_API_KEY from the provider key matching *model*, if needed."""
    if env.get("LLM_API_KEY"):
        return
    for provider_key in api_key_env_names_for_model(model):
        if env.get(provider_key):
            env["LLM_API_KEY"] = env[provider_key]
            return


def validate_record(record: dict[str, Any], mode: Mode) -> list[str]:
    """Return missing manifest fields required for the requested scoring mode."""
    required = ["instruction_path", "trajectory_path", "workspace_path"]
    if mode == "rubric":
        required.append("rubric_path")
    else:
        required.append("judge_guidance_path")
    return [field for field in required if not record.get(field)]


def write_config(
    record: dict[str, Any],
    mode: Mode,
    config_path: Path,
    output_dir: Path,
    *,
    model: str,
    judge_timeout: int,
    judge_retries: int,
) -> None:
    """Write a Gandalf grader.toml for one collected trial."""
    lines = [
        f"model = {toml_str(model)}",
        f"grading_mode = {toml_str(mode)}",
        f"instructions_path = {toml_str(record['instruction_path'])}",
        f"workdir = {toml_str(record['workspace_path'])}",
        f"trajectory_path = {toml_str(record['trajectory_path'])}",
        f"output_dir = {toml_str(str(output_dir))}",
        f"judge_timeout = {judge_timeout}",
        f"judge_retries = {judge_retries}",
    ]
    if mode == "rubric":
        lines.append(f"rubric_path = {toml_str(record['rubric_path'])}")
        if record.get("judge_guidance_path"):
            lines.append(f"judge_guidance_path = {toml_str(record['judge_guidance_path'])}")
    else:
        lines.append(f"judge_guidance_path = {toml_str(record['judge_guidance_path'])}")

    config_path.write_text("\n".join(lines) + "\n")


def run_one(
    record: dict[str, Any],
    mode: Mode,
    trial_eval_dir: Path,
    *,
    gandalf_cmd: list[str],
    model: str,
    judge_timeout: int,
    judge_retries: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run Gandalf for one manifest record and return a result summary."""
    trial_eval_dir.mkdir(parents=True, exist_ok=True)
    output_dir = trial_eval_dir / "grader"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "slug": record.get("slug", ""),
        "env": record.get("env", ""),
        "task": record.get("task", ""),
        "split": record.get("split", ""),
        "trial_dir": record.get("trial_dir", ""),
        "mode": mode,
        "eval_dir": str(trial_eval_dir),
        "grader_output_dir": str(output_dir),
        "gandalf_cmd": gandalf_cmd,
    }

    missing = validate_record(record, mode)
    if missing:
        result.update({"status": "skipped", "error": f"Missing required manifest fields: {', '.join(missing)}"})
        return result

    config_path = trial_eval_dir / "grader.toml"
    write_config(
        record,
        mode,
        config_path,
        output_dir,
        model=model,
        judge_timeout=judge_timeout,
        judge_retries=judge_retries,
    )

    proc = subprocess.run(
        [*gandalf_cmd, "--config", str(config_path)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    (trial_eval_dir / "stdout.txt").write_text(proc.stdout)
    (trial_eval_dir / "stderr.txt").write_text(proc.stderr)
    result.update({"status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode})

    reward_path = output_dir / "reward.json"
    info_path = output_dir / "info.json"
    if reward_path.exists():
        result["reward"] = json.loads(reward_path.read_text()).get("reward")
    if info_path.exists():
        result["info_path"] = str(info_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gandalf scoring over a Harbor rollout manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("_run/rollouts_no_verifier_manifest.jsonl"))
    parser.add_argument("--mode", choices=["rubric", "guidance"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_run/gandalf_eval"))
    parser.add_argument(
        "--gandalf-cmd",
        default="uv run gandalf-the-grader",
        help="Command used to invoke Gandalf, e.g. 'uv run gandalf-the-grader'.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Judge model. Defaults to GANDALF_EVAL_MODEL, then a provider-compatible model from the loaded env file."
        ),
    )
    parser.add_argument("--judge-timeout", type=int, default=300)
    parser.add_argument("--judge-retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "eval", "test"], default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / "Downloads" / "env",
        help="Optional env file to load for Gandalf subprocesses. Values are not printed.",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Number of concurrent Gandalf subprocesses.")
    args = parser.parse_args()

    records = load_jsonl(args.manifest)
    if args.split is not None:
        records = [record for record in records if record.get("split") == args.split]
    if args.limit is not None:
        records = records[: args.limit]

    mode = args.mode
    gandalf_cmd = shlex.split(args.gandalf_cmd)
    grader_env = build_grader_env(args.env_file)
    model = args.model or choose_default_model(grader_env)
    set_llm_api_key_for_model(grader_env, model)
    results_path = args.output_dir / f"results_{mode}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    def run_indexed(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        slug = record.get("slug", f"trial_{index}")
        trial_eval_dir = args.output_dir / mode / f"{index:03d}_{slug}"
        result = run_one(
            record,
            cast("Mode", mode),
            trial_eval_dir,
            gandalf_cmd=gandalf_cmd,
            model=model,
            judge_timeout=args.judge_timeout,
            judge_retries=args.judge_retries,
            env=grader_env,
        )
        return index, result

    indexed_records = list(enumerate(records))
    results: list[dict[str, Any]] = []
    with results_path.open("w") as f:
        if args.max_workers <= 1:
            for index, record in indexed_records:
                _, result = run_indexed(index, record)
                results.append(result)
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                print(f"[{len(results)}/{len(records)}] {record.get('slug', f'trial_{index}')}: {result['status']}")
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_index = {
                    executor.submit(run_indexed, index, record): index for index, record in indexed_records
                }
                for future in as_completed(future_to_index):
                    index, result = future.result()
                    results.append(result)
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                    f.flush()
                    print(
                        f"[{len(results)}/{len(records)}] {records[index].get('slug', f'trial_{index}')}: {result['status']}"
                    )

    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
