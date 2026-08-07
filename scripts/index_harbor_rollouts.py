#!/usr/bin/env python3
"""Create a JSONL manifest for Harbor rollouts collected without verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def first_existing(candidates: list[Path]) -> Path | None:
    """Return the first path that exists, or None."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_slug(trajectory_path: Path, rollouts_root: Path) -> str:
    """Infer the Harbor env/task slug from a trajectory path under rollouts_root."""
    parts = trajectory_path.relative_to(rollouts_root).parts
    for part in parts:
        if "__" in part:
            return part
    return parts[0]


def split_slug(slug: str) -> tuple[str, str]:
    """Split a slug created by collect_harbor_rollouts.sh into env and task."""
    if "__" not in slug:
        return slug, ""
    env, task = slug.split("__", 1)
    return env, task


def trial_dir_for_trajectory(trajectory_path: Path) -> Path:
    """Return the Harbor trial directory for an agent/trajectory.json path."""
    if trajectory_path.parent.name == "agent":
        return trajectory_path.parent.parent
    return trajectory_path.parent


def is_canonical_trajectory(trajectory_path: Path, rollouts_root: Path) -> bool:
    """Return whether trajectory_path is Harbor's canonical trial trajectory.

    Harbor also captures requested artifacts under each trial's artifacts/
    directory. When /logs/agent is collected, that artifact contains a copied
    trajectory.json too. The manifest should index only the canonical
    <trial>/agent/trajectory.json so each rollout trial appears once.
    """
    try:
        parts = trajectory_path.relative_to(rollouts_root).parts
    except ValueError:
        return False
    return trajectory_path.parent.name == "agent" and "artifacts" not in parts


def workspace_path_for_trial(trial_dir: Path) -> Path | None:
    """Find the captured workspace artifact path for a Harbor trial."""
    return first_existing(
        [
            trial_dir / "artifacts" / "workspace",
            trial_dir / "artifacts" / "home" / "agent" / "workspace",
            trial_dir / "home" / "agent" / "workspace",
            trial_dir / "workspace",
        ]
    )


def task_metadata_paths(task_dir: Path) -> dict[str, str]:
    """Resolve common Harbor task metadata file locations."""
    instruction = first_existing([task_dir / "instruction.md", task_dir / "instructions.md", task_dir / "prompt.md"])
    judge_guidance = first_existing([task_dir / "judge_guidance.md", task_dir / "tests" / "judge_guidance.md"])
    rubric = first_existing([task_dir / "rubric.json", task_dir / "tests" / "rubric.json"])
    return {
        "instruction_path": str(instruction) if instruction else "",
        "judge_guidance_path": str(judge_guidance) if judge_guidance else "",
        "rubric_path": str(rubric) if rubric else "",
    }


def discover_rollout_records(rollouts_root: Path, tasks_root: Path) -> list[dict[str, Any]]:
    """Discover rollout trials and map each trial to its Harbor task metadata."""
    rollouts_root = rollouts_root.resolve()
    tasks_root = tasks_root.resolve()
    records: list[dict[str, Any]] = []

    for trajectory_path in sorted(rollouts_root.rglob("agent/trajectory.json")):
        if not is_canonical_trajectory(trajectory_path, rollouts_root):
            continue
        slug = infer_slug(trajectory_path, rollouts_root)
        env, task = split_slug(slug)
        task_dir = tasks_root / env / task if task else tasks_root / env
        trial_dir = trial_dir_for_trajectory(trajectory_path)
        workspace_path = workspace_path_for_trial(trial_dir)

        record: dict[str, Any] = {
            "slug": slug,
            "env": env,
            "task": task,
            "task_dir": str(task_dir),
            "trial_dir": str(trial_dir),
            "trajectory_path": str(trajectory_path),
            "workspace_path": str(workspace_path) if workspace_path else "",
        }
        record.update(task_metadata_paths(task_dir))
        records.append(record)

    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Harbor rollout outputs for Gandalf grading.")
    parser.add_argument(
        "--rollouts-root",
        type=Path,
        default=Path("_run/rollouts_no_verifier"),
        help="Root containing Harbor rollout outputs.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path.home() / "Downloads" / "harbor-research-v2-Batch1-2",
        help="Root containing Harbor-format task directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_run/rollouts_no_verifier_manifest.jsonl"),
        help="Path to write the JSONL manifest.",
    )
    args = parser.parse_args()

    records = discover_rollout_records(args.rollouts_root, args.tasks_root)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} record(s) to {args.output}")


if __name__ == "__main__":
    main()
