"""Trajectory parsing utilities for ATIF (Agent Trajectory Interchange Format)."""

import json


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
