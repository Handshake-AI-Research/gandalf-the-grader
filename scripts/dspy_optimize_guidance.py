#!/usr/bin/env python3
"""Optimize guidance-scoring prompt variants with DSPy against rubric rewards.

This is an eval/calibration helper only. Runtime Gandalf guidance mode still
uses the reactive OpenHands judge that inspects the cloned workspace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    import dspy as _dspy
except ImportError:
    _dspy = None

MIN_QUOTED_ENV_VALUE_LENGTH = 2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines into a list of dictionaries."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_env_file(path: Path | None) -> None:
    """Load simple KEY=VALUE lines into os.environ without printing values."""
    if path is None or not path.exists():
        return

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
        os.environ.setdefault(key, value)

    if not os.environ.get("LLM_API_KEY"):
        for provider_key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if os.environ.get(provider_key):
                os.environ["LLM_API_KEY"] = os.environ[provider_key]
                break


def load_final_output(trajectory_path: str) -> str:
    """Extract the final agent message from an ATIF trajectory."""
    try:
        data = json.loads(Path(trajectory_path).read_text())
    except (OSError, json.JSONDecodeError):
        return ""

    final_output = ""
    for step in reversed(data.get("steps", [])):
        if step.get("source") == "agent" and not step.get("tool_calls"):
            message = str(step.get("message", ""))
            if message.strip():
                final_output = message
                break
    return final_output


def result_key(record: dict[str, Any]) -> tuple[str, str]:
    """Stable key shared by manifest and eval result records."""
    return str(record.get("slug", "")), str(record.get("trial_dir", ""))


def build_examples_data(manifest: list[dict[str, Any]], rubric_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join manifest records to rubric rewards and load text inputs."""
    rewards_by_key = {
        result_key(result): float(result["reward"])
        for result in rubric_results
        if result.get("reward") is not None and result.get("status") == "ok"
    }
    examples: list[dict[str, Any]] = []
    for record in manifest:
        reward = rewards_by_key.get(result_key(record))
        guidance_path = record.get("judge_guidance_path")
        instructions_path = record.get("instruction_path")
        if reward is None or not guidance_path or not instructions_path:
            continue
        try:
            judge_guidance = Path(guidance_path).read_text()
            instructions = Path(instructions_path).read_text()
        except OSError:
            continue
        examples.append(
            {
                "instructions": instructions,
                "judge_guidance": judge_guidance,
                "final_output": load_final_output(str(record.get("trajectory_path", ""))),
                "target_score": reward,
                "split": record.get("split", ""),
                "slug": record.get("slug", ""),
                "trial_dir": record.get("trial_dir", ""),
            }
        )
    return examples


def load_dspy() -> Any:
    """Import DSPy with a helpful error when the eval dependency group is missing."""
    if _dspy is None:
        msg = "DSPy is not installed. Run this script with `uv run --group eval` or install dspy-ai."
        raise SystemExit(msg)
    return _dspy


def make_program(dspy: Any) -> Any:
    """Construct a DSPy program for text-only guidance score calibration."""

    class GuidanceScoreSignature(dspy.Signature):
        """Predict a holistic score in [0, 1] from task text, final output, and grading guidance."""

        task_instructions = dspy.InputField(desc="Original task instructions")
        judge_guidance = dspy.InputField(desc="Free-form grading guidance")
        final_output = dspy.InputField(desc="Final agent message extracted from the trajectory")
        score = dspy.OutputField(desc="A number from 0 to 1 inclusive")

    class GuidanceScoreProgram(dspy.Module):
        def __init__(self) -> None:
            self.predict = dspy.ChainOfThought(GuidanceScoreSignature)

        def forward(self, instructions: str, judge_guidance: str, final_output: str) -> Any:
            return self.predict(
                task_instructions=instructions,
                judge_guidance=judge_guidance,
                final_output=final_output,
            )

    return GuidanceScoreProgram()


def make_examples(dspy: Any, rows: list[dict[str, Any]]) -> list[Any]:
    """Convert joined calibration rows into DSPy examples."""
    examples = []
    for row in rows:
        example = dspy.Example(
            instructions=row["instructions"],
            judge_guidance=row["judge_guidance"],
            final_output=row["final_output"],
            target_score=row["target_score"],
            split=row.get("split", ""),
            slug=row.get("slug", ""),
        ).with_inputs("instructions", "judge_guidance", "final_output")
        examples.append(example)
    return examples


def score_metric(example: Any, pred: Any, _trace: Any = None) -> float:
    """DSPy metric: one minus absolute score error, clipped to [0, 1]."""
    try:
        predicted = float(pred.score)
        target = float(example.target_score)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    predicted = max(0.0, min(1.0, predicted))
    return max(0.0, 1.0 - abs(predicted - target))


def optimizer_class(dspy: Any, optimizer_name: str) -> Any:
    """Pick an available DSPy optimizer class across DSPy versions."""
    teleprompt = getattr(dspy, "teleprompt", None)
    if optimizer_name == "mipro":
        if hasattr(dspy, "MIPROv2"):
            return dspy.MIPROv2
        if teleprompt is not None and hasattr(teleprompt, "MIPROv2"):
            return teleprompt.MIPROv2
    if optimizer_name == "bootstrap":
        if hasattr(dspy, "BootstrapFewShot"):
            return dspy.BootstrapFewShot
        if teleprompt is not None and hasattr(teleprompt, "BootstrapFewShot"):
            return teleprompt.BootstrapFewShot
    msg = f"Could not find DSPy optimizer {optimizer_name!r}."
    raise SystemExit(msg)


def compile_program(
    dspy: Any,
    program: Any,
    trainset: list[Any],
    *,
    optimizer_name: str,
    max_bootstrapped_demos: int,
) -> Any:
    """Compile a DSPy program with a version-tolerant optimizer call."""
    if optimizer_name == "none":
        return program

    cls = optimizer_class(dspy, optimizer_name)
    if optimizer_name == "mipro":
        try:
            optimizer = cls(metric=score_metric, auto="light")
        except TypeError:
            optimizer = cls(metric=score_metric)
    else:
        try:
            optimizer = cls(metric=score_metric, max_bootstrapped_demos=max_bootstrapped_demos)
        except TypeError:
            optimizer = cls(metric=score_metric)

    try:
        return optimizer.compile(program, trainset=trainset)
    except TypeError:
        return optimizer.compile(program, trainset)


def mean(values: list[float]) -> float:
    """Return arithmetic mean, or 0 for empty input."""
    return sum(values) / len(values) if values else 0.0


def predict_score(program: Any, row: dict[str, Any]) -> float | None:
    """Run a DSPy program and parse its score output."""
    try:
        pred = program(
            instructions=row["instructions"],
            judge_guidance=row["judge_guidance"],
            final_output=row["final_output"],
        )
        score = float(pred.score)
    except (AttributeError, TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def summarize_predictions(program: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize DSPy score predictions against rubric rewards."""
    predictions: list[dict[str, Any]] = []
    for row in rows:
        pred = predict_score(program, row)
        if pred is None:
            predictions.append(
                {
                    "slug": row.get("slug", ""),
                    "target": row["target_score"],
                    "prediction": None,
                    "abs_error": None,
                }
            )
            continue
        target = float(row["target_score"])
        predictions.append(
            {
                "slug": row.get("slug", ""),
                "target": target,
                "prediction": round(pred, 4),
                "abs_error": round(abs(pred - target), 4),
            }
        )

    valid_errors = [float(item["abs_error"]) for item in predictions if item["abs_error"] is not None]
    return {
        "n": len(rows),
        "n_valid": len(valid_errors),
        "mae": round(mean(valid_errors), 4),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize DSPy guidance scorer against rubric rewards.")
    parser.add_argument("--manifest", type=Path, default=Path("_run/rollouts_no_verifier_manifest.jsonl"))
    parser.add_argument("--rubric-results", type=Path, default=Path("_run/gandalf_eval/results_rubric.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("_run/dspy_guidance_optimized.json"))
    parser.add_argument("--lm", default=os.environ.get("DSPY_MODEL", "openai/gpt-5.5"))
    parser.add_argument("--env-file", type=Path, default=Path.home() / "Downloads" / "env")
    parser.add_argument("--optimizer", choices=["none", "bootstrap", "mipro"], default="bootstrap")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_env_file(args.env_file)
    dspy = load_dspy()
    dspy.configure(lm=dspy.LM(args.lm))

    manifest = load_jsonl(args.manifest)
    rubric_results = load_jsonl(args.rubric_results)
    rows = build_examples_data(manifest, rubric_results)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        msg = "No calibration examples found."
        raise SystemExit(msg)

    train_rows = [row for row in rows if row.get("split") == args.train_split]
    eval_rows = [row for row in rows if row.get("split") == args.eval_split]
    test_rows = [row for row in rows if row.get("split") == args.test_split]
    if not train_rows:
        msg = f"No train examples found for split {args.train_split!r}."
        raise SystemExit(msg)

    examples = make_examples(dspy, train_rows)
    program = make_program(dspy)
    optimized = compile_program(
        dspy,
        program,
        examples,
        optimizer_name=args.optimizer,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
    )

    summary = {
        "optimizer": args.optimizer,
        "lm": args.lm,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_split": args.test_split,
        "train": summarize_predictions(optimized, train_rows),
        "eval": summarize_predictions(optimized, eval_rows),
        "test": summarize_predictions(optimized, test_rows),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(optimized, "save"):
        optimized.save(str(args.output))
    else:
        args.output.write_text(
            json.dumps(
                {
                    "note": "Optimized DSPy program object did not expose save().",
                    "examples": len(train_rows),
                },
                indent=2,
            )
        )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Optimized on {len(examples)} train example(s); wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
