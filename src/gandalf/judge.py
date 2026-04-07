"""Inner judge: evaluates rubric criteria using an OpenHands agent.

This script is invoked as the sandbox user (via sudo) from the outer grader
orchestrator. It receives all context via an input JSON file and writes its
verdict to an output JSON file.

Supports two modes:
  - Single criterion (default): evaluates one criterion, writes a JSON object.
  - Batch (--batch): evaluates all criteria in one session, writes a JSON object
    with ``verdicts`` (array) and ``llm_usage`` (dict) keys.
"""

import argparse
import contextlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

import jinja2
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from pydantic import TypeAdapter

from gandalf.models import BatchJudgeInput, JudgeInput, LLMUsage, MCPServer, Verdict

TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_template(
    template_name: str,
    judge_prompt: str | None,
    **variables: Any,
) -> str:
    """Render a Jinja2 prompt template.

    The rendered result is sent to kick off a judge session.
    If *judge_prompt* is provided (a raw Jinja2 template string),
    it is used instead of the built-in template identified by *template_name*.
    """
    template_str = judge_prompt if judge_prompt is not None else (TEMPLATES_DIR / template_name).read_text()
    return jinja2.Template(template_str).render(**variables)


def build_judge_prompt(
    instructions: str,
    final_output: str,
    criterion: str,
    verdict_path: str,
    judge_guidance: str = "",
    judge_prompt: str | None = None,
) -> str:
    """Build the user message sent to kick off a single-criterion judge session."""
    return render_template(
        "judge_single.j2",
        judge_prompt,
        instructions=instructions,
        final_output=final_output,
        criterion=criterion,
        verdict_path=verdict_path,
        judge_guidance=judge_guidance,
    )


def build_batch_judge_prompt(
    instructions: str,
    final_output: str,
    criteria: list[str],
    verdict_path: str,
    judge_guidance: str = "",
    judge_prompt: str | None = None,
) -> str:
    """Build the user message sent to kick off a batch-mode judge session.

    Evaluates all criteria in one session, writing a JSON array of verdicts
    instead of a single object.
    """
    return render_template(
        "judge_batch.j2",
        judge_prompt,
        instructions=instructions,
        final_output=final_output,
        criteria=criteria,
        verdict_path=verdict_path,
        judge_guidance=judge_guidance,
    )


def read_verdict(verdict_path: str) -> Verdict:
    """Read and validate the verdict file written by the judge agent."""
    try:
        with open(verdict_path) as f:
            content = f.read().strip()
        if not content:
            return Verdict(met=None, reasoning="Judge agent wrote an empty verdict file.")
        data = json.loads(content)
        if "met" not in data:
            return Verdict(met=None, reasoning=f"Verdict missing 'met' field: {content[:200]}")
        return Verdict.from_raw(data)
    except FileNotFoundError:
        return Verdict(met=None, reasoning="Judge agent did not write a verdict file.")
    except json.JSONDecodeError as e:
        return Verdict(met=None, reasoning=f"Judge agent wrote invalid JSON: {e}")


def read_batch_verdict(verdict_path: str, n_criteria: int) -> list[Verdict]:
    """Read and validate the batch verdict file written by the judge agent.

    Returns a list of Verdicts, one per criterion index.  Missing
    indices get a default fail verdict.
    """
    try:
        with open(verdict_path) as f:
            content = f.read().strip()
        if not content:
            return Verdict.errors(
                n_criteria,
                "Judge agent wrote an empty verdict file.",
            )

        verdicts_raw = json.loads(content)
        if not isinstance(verdicts_raw, list):
            return Verdict.errors(
                n_criteria,
                f"Expected JSON array, got {type(verdicts_raw).__name__}",
            )

        by_index: dict[int, Verdict] = {}
        for v in verdicts_raw:
            idx = v.get("index")
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < n_criteria:
                by_index[idx] = Verdict.from_raw(v)

        results: list[Verdict] = []
        for i in range(n_criteria):
            if i in by_index:
                results.append(by_index[i])
            else:
                results.append(
                    Verdict(
                        met=None,
                        reasoning=f"Judge did not return a verdict for criterion {i}.",
                    )
                )

    except FileNotFoundError:
        return Verdict.errors(
            n_criteria,
            "Judge agent did not write a verdict file.",
        )
    except json.JSONDecodeError as e:
        return Verdict.errors(
            n_criteria,
            f"Judge agent wrote invalid JSON: {e}",
        )
    else:
        return results


def make_verdict_path(prefix: str = "verdict_", directory: str | None = None) -> str:
    """Generate a unique path for the judge to write verdicts to.

    Unlike mkstemp, this does NOT pre-create the file — allowing the agent
    to use file_editor create rather than error-prone shell echo fallbacks.

    Uses *directory* as the base directory when provided (e.g. the workdir,
    which the grader has already made world-writable), falling back to the
    system temp dir.  This avoids requiring sandbox_user to have general write
    access to /tmp.
    """
    base = directory if directory is not None else tempfile.gettempdir()
    return os.path.join(base, f"{prefix}{secrets.token_hex(8)}.json")


def run_agent_session(
    model: str,
    mcp_servers: list[MCPServer],
    workdir: str,
    prompt: str,
) -> LLMUsage:
    """Create an OpenHands agent and run a single conversation.

    The agent writes its output to a file (path embedded in *prompt*).
    Returns LLM usage metrics (empty defaults if extraction fails).
    """
    # Pin HOME to the judge workspace before instantiating the OpenHands SDK.
    # The SDK writes state to ~/.openhands/ (profiles, agents, etc.) on init.
    # Without this, HOME may point to a directory owned by a different user
    # (e.g. /home/agent when the judge runs as judge-sandbox via sudo),
    # causing PermissionError on mkdir.
    os.environ["HOME"] = workdir

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        msg = (
            "LLM_API_KEY environment variable is not set. "
            "The caller must map the provider-specific key "
            "(e.g. ANTHROPIC_API_KEY) to LLM_API_KEY."
        )
        raise RuntimeError(msg)

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL"),
    )

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
    ]

    if mcp_servers:
        mcp_config: dict[str, Any] = {
            "mcpServers": {
                srv.name: {"command": srv.command, **({"args": srv.args} if srv.args else {})} for srv in mcp_servers
            }
        }
        agent = Agent(llm=llm, tools=tools, mcp_config=mcp_config)
    else:
        agent = Agent(llm=llm, tools=tools)

    conversation = Conversation(agent=agent, workspace=workdir)
    conversation.send_message(prompt)  # type: ignore[attr-defined]
    conversation.run()  # type: ignore[attr-defined]

    try:
        token_usage = llm.metrics.accumulated_token_usage
        return LLMUsage(
            cost_usd=llm.metrics.accumulated_cost,
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            cache_read_tokens=token_usage.cache_read_tokens if token_usage else 0,
        )
    except Exception:  # noqa: BLE001
        return LLMUsage()


def run_judge(input_path: str, output_path: str) -> None:
    """Run the agent-as-judge for a single rubric criterion."""
    with open(input_path) as f:
        judge_input = JudgeInput.model_validate_json(f.read())

    verdict_path = make_verdict_path(prefix="verdict_", directory=judge_input.workdir)

    prompt = build_judge_prompt(
        instructions=judge_input.instructions,
        final_output=judge_input.final_output,
        criterion=judge_input.criterion,
        verdict_path=verdict_path,
        judge_guidance=judge_input.judge_guidance,
        judge_prompt=judge_input.judge_prompt,
    )

    llm_usage = LLMUsage()
    try:
        llm_usage = run_agent_session(judge_input.model, judge_input.mcp_servers, judge_input.workdir, prompt)
        verdict = read_verdict(verdict_path)
    except Exception as e:  # noqa: BLE001
        verdict = Verdict(met=None, reasoning=f"Judge execution error: {e}")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(verdict_path)

    output = {"verdict": verdict.model_dump(), "llm_usage": llm_usage.model_dump()}
    with open(output_path, "w") as f:
        json.dump(output, f)


def run_judge_batch(input_path: str, output_path: str) -> None:
    """Run the agent-as-judge for all rubric criteria in a single session.

    The input JSON must contain a ``criteria`` key whose value is a list of
    criterion strings.

    The output file will contain a JSON object with ``verdicts`` (array of
    verdict objects, one per criterion index) and ``llm_usage`` (aggregate
    token/cost dict for the session).
    """
    with open(input_path) as f:
        judge_input = BatchJudgeInput.model_validate_json(f.read())

    n_criteria = len(judge_input.criteria)

    verdict_path = make_verdict_path(prefix="verdict_batch_", directory=judge_input.workdir)

    prompt = build_batch_judge_prompt(
        instructions=judge_input.instructions,
        final_output=judge_input.final_output,
        criteria=judge_input.criteria,
        verdict_path=verdict_path,
        judge_guidance=judge_input.judge_guidance,
        judge_prompt=judge_input.judge_prompt,
    )

    llm_usage = LLMUsage()
    try:
        llm_usage = run_agent_session(judge_input.model, judge_input.mcp_servers, judge_input.workdir, prompt)
        verdicts = read_batch_verdict(verdict_path, n_criteria)
    except Exception as e:  # noqa: BLE001
        verdicts = Verdict.errors(
            n_criteria,
            f"Judge execution error: {e}",
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(verdict_path)

    output = {
        "verdicts": TypeAdapter(list[Verdict]).dump_python(verdicts),
        "llm_usage": llm_usage.model_dump(),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge rubric criterion")
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
