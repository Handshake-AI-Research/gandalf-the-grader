"""Inner judge: evaluates rubric criteria using an OpenHands agent.

This script is invoked as the sandbox user (via sudo) from the outer verifier
orchestrator. It receives all context via an input JSON file and writes its
verdict to an output JSON file.

Supports two modes:
  - Single criterion (default): evaluates one criterion, writes a JSON object.
  - Batch (--batch): evaluates all criteria in one session, writes a JSON object
    with ``verdicts`` (array) and ``llm_usage`` (dict) keys.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import tempfile
from typing import Any

from gandalf_grader.config import BatchJudgeInput, JudgeInput, Verdict


def build_judge_prompt(
    instructions: str,
    final_output: str,
    criteria: str,
    verdict_path: str,
    judge_guidance: str = "",
) -> str:
    """Build the full prompt for the judge agent."""
    guidance_block = f"\n\n{judge_guidance}\n" if judge_guidance else ""
    return f"""\
You are an expert judge evaluating whether an AI agent successfully completed a task \
according to a specific evaluation criteria. You have access to the agent's working \
directory and can inspect files, run commands, and use tools to investigate.
{guidance_block}
## Task Instructions (given to the agent)
{instructions}

## Agent's Final Output
{final_output}

## Evaluation Criteria
{criteria}

## Your Task
Investigate the current state of the environment to determine whether the above \
criteria is satisfied. Use the available tools to read files, run commands, and \
inspect the environment as needed.

After your investigation, you MUST write your verdict as a JSON object to the file:
  {verdict_path}

IMPORTANT: You must use the file_editor tool (create command) or the terminal tool \
(e.g. cat <<'EOF' > {verdict_path}) to physically write the file to disk. \
Do NOT simply print or display the JSON in your response — the verdict will only be \
read from the file on disk.

The JSON object must have exactly these fields:
- "passed": true if the criteria is satisfied, false otherwise
- "reasoning": a brief explanation of your judgment
- "evidence": an array of strings, each describing a concrete check you performed \
(e.g. file path and observed content, command output, or tool-returned value)

Example:
```json
{{
  "passed": true,
  "reasoning": "The file foo.txt exists and contains the expected content.",
  "evidence": [
    "Read /workspace/foo.txt: contains 'hello world'",
    "Ran 'wc -l foo.txt': output '1 foo.txt'"
  ]
}}
```

Write ONLY valid JSON to that file, with no additional text."""


def build_batch_judge_prompt(
    instructions: str,
    final_output: str,
    criteria: list[dict[str, Any]],
    verdict_path: str,
    judge_guidance: str = "",
) -> str:
    """Build the prompt for batch mode -- all criteria evaluated in one session.

    Mirrors build_judge_prompt but evaluates multiple criteria at once,
    writing a JSON array of verdicts instead of a single object.
    """
    guidance_block = f"\n\n{judge_guidance}\n" if judge_guidance else ""

    criteria_lines = []
    for c in criteria:
        criteria_lines.append(f"  [{c['index']}] (weight={c['weight']}) {c['criteria']}")
    criteria_block = "\n".join(criteria_lines)
    n_max = len(criteria) - 1

    return f"""\
You are an expert judge evaluating whether an AI agent successfully completed a task \
according to multiple evaluation criteria. You have access to the agent's working \
directory and can inspect files, run commands, and use tools to investigate.
{guidance_block}
## Task Instructions (given to the agent)
{instructions}

## Agent's Final Output
{final_output}

## Evaluation Criteria

{criteria_block}

Each criterion has a weight indicating relative importance. Your verdict for each is \
binary: passed or failed.

## Your Task
Investigate the current state of the environment to determine whether each of the above \
criteria is satisfied. Use the available tools to read files, run commands, and \
inspect the environment as needed.

After your investigation, you MUST write your verdicts as a JSON array to the file:
  {verdict_path}

IMPORTANT: You must use the file_editor tool (create command) or the terminal tool \
(e.g. cat <<'EOF' > {verdict_path}) to physically write the file to disk. \
Do NOT simply print or display the JSON in your response — the verdict will only be \
read from the file on disk.

Each element in the array must have exactly these fields:
- "index": the criterion index (integer, 0-based)
- "passed": true if the criteria is satisfied, false otherwise
- "reasoning": a brief explanation of your judgment
- "evidence": an array of strings, each describing a concrete check you performed \
(e.g. file path and observed content, command output, or tool-returned value)

Example:
```json
[
  {{
    "index": 0,
    "passed": true,
    "reasoning": "The file foo.txt exists and contains the expected content.",
    "evidence": [
      "Read /workspace/foo.txt: contains 'hello world'",
      "Ran 'wc -l foo.txt': output '1 foo.txt'"
    ]
  }},
  {{
    "index": 1,
    "passed": false,
    "reasoning": "The output is missing the required header.",
    "evidence": ["Read /workspace/output.txt: no header line found"]
  }}
]
```

You MUST include a verdict for every criterion index (0 through {n_max}).
Write ONLY valid JSON to that file, with no additional text."""


# ---------------------------------------------------------------------------
# Verdict readers
# ---------------------------------------------------------------------------


def _read_verdict(verdict_path: str) -> Verdict:
    """Read and validate the verdict file written by the judge agent."""
    try:
        with open(verdict_path) as f:
            content = f.read().strip()
        if not content:
            return Verdict(passed=False, reasoning="Judge agent wrote an empty verdict file.")
        data = json.loads(content)
        if "passed" not in data:
            return Verdict(passed=False, reasoning=f"Verdict missing 'passed' field: {content[:200]}")
        return Verdict(
            passed=bool(data["passed"]),
            reasoning=str(data.get("reasoning", "No reasoning provided.")),
            evidence=list(data.get("evidence", [])),
        )
    except FileNotFoundError:
        return Verdict(passed=False, reasoning="Judge agent did not write a verdict file.")
    except json.JSONDecodeError as e:
        return Verdict(passed=False, reasoning=f"Judge agent wrote invalid JSON: {e}")


def _fail_all_verdicts(n: int, reason: str) -> list[dict[str, Any]]:
    """Return *n* fail verdict dicts sharing the same reason."""
    return [{"index": i, "passed": False, "reasoning": reason, "evidence": []} for i in range(n)]


def _read_batch_verdict(verdict_path: str, n_criteria: int) -> list[dict[str, Any]]:
    """Read and validate the batch verdict file written by the judge agent.

    Returns a list of verdict dicts, one per criterion index.  Missing
    indices get a default fail verdict.
    """
    try:
        with open(verdict_path) as f:
            content = f.read().strip()
        if not content:
            return _fail_all_verdicts(
                n_criteria,
                "Judge agent wrote an empty verdict file.",
            )

        verdicts_raw = json.loads(content)
        if not isinstance(verdicts_raw, list):
            return _fail_all_verdicts(
                n_criteria,
                f"Expected JSON array, got {type(verdicts_raw).__name__}",
            )

        by_index: dict[int, dict[str, Any]] = {}
        for v in verdicts_raw:
            idx = v.get("index")
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < n_criteria:
                by_index[idx] = {
                    "passed": bool(v.get("passed", False)),
                    "reasoning": str(v.get("reasoning", "No reasoning provided.")),
                    "evidence": list(v.get("evidence", [])),
                }

        results = []
        for i in range(n_criteria):
            if i in by_index:
                results.append({"index": i, **by_index[i]})
            else:
                results.append(
                    {
                        "index": i,
                        "passed": False,
                        "reasoning": f"Judge did not return a verdict for criterion {i}.",
                        "evidence": [],
                    }
                )
        return results

    except FileNotFoundError:
        return _fail_all_verdicts(
            n_criteria,
            "Judge agent did not write a verdict file.",
        )
    except json.JSONDecodeError as e:
        return _fail_all_verdicts(
            n_criteria,
            f"Judge agent wrote invalid JSON: {e}",
        )


# ---------------------------------------------------------------------------
# Agent session helpers
# ---------------------------------------------------------------------------


def _make_verdict_path(prefix: str = "verdict_") -> str:
    """Generate a unique path for the judge to write verdicts to.

    Unlike mkstemp, this does NOT pre-create the file — allowing the agent
    to use file_editor create rather than error-prone shell echo fallbacks.
    """
    return os.path.join(
        tempfile.gettempdir(),
        f"{prefix}{secrets.token_hex(8)}.json",
    )


def _patch_action_schemas_to_ignore_extra() -> None:
    """Relax Pydantic ``extra="forbid"`` on OpenHands tool action schemas.

    LLMs occasionally include unexpected keys in tool-call arguments (e.g.
    ``evidence`` from the verdict schema leaking into a ``terminal`` call).
    The upstream ``Action`` base sets ``extra="forbid"`` which causes a hard
    ``ValidationError`` for any extra key.  Changing it to ``"ignore"``
    silently drops the spurious keys and lets the tool call proceed.

    This is a targeted monkey-patch applied to the ``Action`` base class
    and the two concrete subclasses used by the judge agent.
    """
    from openhands.sdk.tool.schema import Action
    from openhands.tools.file_editor import FileEditorAction
    from openhands.tools.terminal import TerminalAction
    from pydantic import ConfigDict

    if Action.model_config.get("extra") == "ignore":
        return  # already patched

    # Pydantic bakes the config into a compiled validator at class-creation
    # time.  We must update model_config on every class in the hierarchy
    # and force a full rebuild so the validator is re-compiled.
    for cls in (Action, TerminalAction, FileEditorAction):
        cls.model_config = ConfigDict(
            **{**cls.model_config, "extra": "ignore"},
        )
        cls.model_rebuild(force=True)


def _run_agent_session(
    model: str,
    mcp_servers: list[dict[str, Any]],
    workdir: str,
    prompt: str,
) -> dict[str, Any]:
    """Create an OpenHands agent and run a single conversation.

    The agent writes its output to a file (path embedded in *prompt*).
    Returns a dict of LLM usage metrics (may be empty if extraction fails).
    """
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    _patch_action_schemas_to_ignore_extra()

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY environment variable is not set. "
            "The caller must map the provider-specific key "
            "(e.g. ANTHROPIC_API_KEY) to LLM_API_KEY."
        )

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL"),
    )

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
    ]

    mcp_config: dict[str, Any] | None = None
    if mcp_servers:
        mcp_config = {"mcpServers": {}}
        for mcp in mcp_servers:
            server_name = mcp.get("name", "mcp-server")
            server_cfg = {"command": mcp["command"]}
            if mcp.get("args"):
                server_cfg["args"] = mcp["args"]
            mcp_config["mcpServers"][server_name] = server_cfg

    agent_kwargs = {"llm": llm, "tools": tools}
    if mcp_config is not None:
        agent_kwargs["mcp_config"] = mcp_config
    agent = Agent(**agent_kwargs)  # type: ignore[arg-type]

    conversation = Conversation(agent=agent, workspace=workdir)
    conversation.send_message(prompt)  # type: ignore[attr-defined]
    conversation.run()  # type: ignore[attr-defined]

    llm_usage: dict[str, Any] = {}
    try:
        token_usage = llm.metrics.accumulated_token_usage
        llm_usage = {
            "cost_usd": llm.metrics.accumulated_cost,
            "prompt_tokens": token_usage.prompt_tokens if token_usage else 0,
            "completion_tokens": token_usage.completion_tokens if token_usage else 0,
            "cache_read_tokens": token_usage.cache_read_tokens if token_usage else 0,
        }
    except Exception:
        pass
    return llm_usage


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_judge(input_path: str, output_path: str) -> None:
    """Run the agent-as-judge for a single rubric criteria."""
    with open(input_path) as f:
        judge_input = JudgeInput.model_validate_json(f.read())

    verdict_path = _make_verdict_path(prefix="verdict_")

    prompt = build_judge_prompt(
        instructions=judge_input.instructions,
        final_output=judge_input.final_output,
        criteria=judge_input.criteria,
        verdict_path=verdict_path,
        judge_guidance=judge_input.judge_guidance,
    )

    mcp_servers = [
        {
            "name": srv.name,
            "command": srv.command,
            "args": srv.args,
        }
        for srv in judge_input.mcp_servers
    ]

    llm_usage: dict[str, Any] = {}
    try:
        llm_usage = _run_agent_session(
            judge_input.model, mcp_servers, judge_input.workdir, prompt
        )
        verdict = _read_verdict(verdict_path)
        output = {
            "passed": verdict.passed,
            "reasoning": verdict.reasoning,
            "evidence": verdict.evidence,
            "llm_usage": llm_usage,
        }
    except Exception as e:
        output = {
            "passed": False,
            "reasoning": f"Judge execution error: {e}",
            "evidence": [],
            "llm_usage": llm_usage,
        }
    finally:
        with contextlib.suppress(OSError):
            os.unlink(verdict_path)

    with open(output_path, "w") as f:
        json.dump(output, f)


def run_judge_batch(input_path: str, output_path: str) -> None:
    """Run the agent-as-judge for all rubric criteria in a single session.

    The input JSON must contain a ``criteria`` key whose value is a list of
    dicts, each with ``index``, ``criteria``, and ``weight`` fields.

    The output file will contain a JSON object with ``verdicts`` (array of
    verdict objects, one per criterion index) and ``llm_usage`` (aggregate
    token/cost dict for the session).
    """
    with open(input_path) as f:
        judge_input = BatchJudgeInput.model_validate_json(f.read())

    criteria_dicts = [c.model_dump() for c in judge_input.criteria]
    n_criteria = len(criteria_dicts)

    verdict_path = _make_verdict_path(prefix="verdict_batch_")

    prompt = build_batch_judge_prompt(
        instructions=judge_input.instructions,
        final_output=judge_input.final_output,
        criteria=criteria_dicts,
        verdict_path=verdict_path,
        judge_guidance=judge_input.judge_guidance,
    )

    mcp_servers = [
        {
            "name": srv.name,
            "command": srv.command,
            "args": srv.args,
        }
        for srv in judge_input.mcp_servers
    ]

    llm_usage: dict[str, Any] = {}
    try:
        llm_usage = _run_agent_session(
            judge_input.model, mcp_servers, judge_input.workdir, prompt
        )
        verdicts = _read_batch_verdict(verdict_path, n_criteria)
    except Exception as e:
        verdicts = _fail_all_verdicts(
            n_criteria,
            f"Judge execution error: {e}",
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(verdict_path)

    output = {"verdicts": verdicts, "llm_usage": llm_usage}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge rubric criteria")
    parser.add_argument("--input", required=True, help="Path to judge input JSON")
    parser.add_argument("--output", required=True, help="Path to write judge output JSON")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: evaluate all criteria in a single agent session",
    )
    args = parser.parse_args()

    if args.batch:
        run_judge_batch(args.input, args.output)
    else:
        run_judge(args.input, args.output)


if __name__ == "__main__":
    main()
