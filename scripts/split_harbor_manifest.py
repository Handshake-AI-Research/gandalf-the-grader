#!/usr/bin/env python3
"""Assign deterministic train/eval/test splits to an indexed Harbor rollout manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_RATIOS = {"train": 0.6, "eval": 0.2, "test": 0.2}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines into a list of dictionaries."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_score(value: str, seed: str) -> int:
    """Return a deterministic integer hash for split ordering."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest, 16)


def normalize_ratios(train: float, eval_ratio: float, test: float) -> dict[str, float]:
    """Normalize split ratios and validate they are positive."""
    ratios = {"train": train, "eval": eval_ratio, "test": test}
    if any(value < 0 for value in ratios.values()):
        msg = "Split ratios must be non-negative."
        raise ValueError(msg)
    total = sum(ratios.values())
    if total <= 0:
        msg = "At least one split ratio must be positive."
        raise ValueError(msg)
    return {key: value / total for key, value in ratios.items()}


def split_counts(n_items: int, ratios: dict[str, float]) -> dict[str, int]:
    """Return integer split counts that sum to n_items."""
    raw = {key: n_items * ratio for key, ratio in ratios.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remaining = n_items - sum(counts.values())
    for key, _value in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
        if remaining <= 0:
            break
        counts[key] += 1
        remaining -= 1
    return counts


def assign_splits(
    records: list[dict[str, Any]],
    *,
    seed: str = "gandalf-guidance-v1",
    ratios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Assign split labels, stratified by env and grouped by slug."""
    ratios = ratios or DEFAULT_RATIOS

    by_env: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        env = str(record.get("env", ""))
        slug = str(record.get("slug", ""))
        by_env[env][slug].append(record)

    split_by_slug: dict[str, str] = {}
    for env, slug_map in by_env.items():
        slugs = sorted(slug_map, key=lambda slug: stable_score(f"{env}:{slug}", seed))
        counts = split_counts(len(slugs), ratios)
        split_labels: list[str] = []
        for label in ("train", "eval", "test"):
            split_labels.extend([label] * counts[label])
        split_by_slug.update(dict(zip(slugs, split_labels, strict=True)))

    output: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched["split"] = split_by_slug[str(record.get("slug", ""))]
        enriched["split_seed"] = seed
        enriched["split_group"] = "slug"
        output.append(enriched)
    return output


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize split counts overall and by environment."""
    summary: dict[str, Any] = {"total": len(records), "splits": {}, "by_env": {}}
    for record in records:
        split = str(record.get("split", ""))
        env = str(record.get("env", ""))
        summary["splits"][split] = summary["splits"].get(split, 0) + 1
        env_summary = summary["by_env"].setdefault(env, {})
        env_summary[split] = env_summary.get(split, 0) + 1
    return summary


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign train/eval/test splits to a Harbor rollout manifest.")
    parser.add_argument("--input", type=Path, default=Path("_run/rollouts_no_verifier_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("_run/rollouts_no_verifier_manifest_split.jsonl"))
    parser.add_argument("--seed", default="gandalf-guidance-v1")
    parser.add_argument("--train", type=float, default=0.6)
    parser.add_argument("--eval", dest="eval_ratio", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.2)
    args = parser.parse_args()

    ratios = normalize_ratios(args.train, args.eval_ratio, args.test)
    records = assign_splits(load_jsonl(args.input), seed=args.seed, ratios=ratios)
    write_jsonl(records, args.output)
    print(json.dumps(split_summary(records), indent=2, sort_keys=True))
    print(f"Wrote split manifest to {args.output}")


if __name__ == "__main__":
    main()
