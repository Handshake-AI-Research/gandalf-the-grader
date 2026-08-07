# Gandalf the Grader [![Build Status](https://github.com/Handshake-AI-Research/gandalf-the-grader/actions/workflows/ci.yml/badge.svg)](https://github.com/Handshake-AI-Research/gandalf-the-grader/actions/workflows/ci.yml) [![Coverage](https://codecov.io/gh/Handshake-AI-Research/gandalf-the-grader/branch/main/graph/badge.svg)](https://app.codecov.io/gh/Handshake-AI-Research/gandalf-the-grader) [![PyPI](https://img.shields.io/pypi/v/gandalf-the-grader.svg)](https://pypi.org/pypi/gandalf-the-grader/) [![PyPI - Python version](https://img.shields.io/pypi/pyversions/gandalf-the-grader.svg)](https://pypi.org/pypi/gandalf-the-grader/)

### Your verifier is probably the bottleneck. We built one that isn't.

![Gandalf vs. baseline verifiers on BankerVerifierBench (cost vs. F1)](https://raw.githubusercontent.com/Handshake-AI-Research/assets/main/gandalf-the-grader/pareto_frontier.png)

Read the [launch blog post](https://joinhandshake.com/research/ai/gandalf-the-grader/) for the motivation, benchmark results, and design rationale behind Gandalf.

Gandalf is a reactive agent-as-judge for agent environments. In its default rubric mode, it grades binary criteria and computes a weighted reward. In guidance mode, it uses free-form grading guidance to assign one holistic score in `[0, 1]`. In both modes, it runs inside the rollout environment, uses the same tools as the rollout agent, and decides at inference time which files to open and which tool state to query.

That lets Gandalf grade criteria that depend on artifacts or state — formulas in a workbook, charts in a deck, files on disk, MCP tool state, or whether an email was actually sent — rather than just the final text response.

Gandalf is built around three design choices:

- **Environment alignment:** Gandalf runs in the same filesystem, Python interpreter, installed packages, and tool environment as the rollout agent, using the [OpenHands](https://github.com/All-Hands-AI/OpenHands) SDK as the agent harness.

- **Reactive verification:** Gandalf chooses what evidence to inspect while grading, instead of relying on a precomputed transcript or serialized snapshot.

- **Swappable domain guidance:** Domain knowledge enters as natural-language guidance at runtime, making the same verifier portable across domains.

In our evaluation, this design beat text-only, snapshot-based, and workflow-based agentic verifiers at a fraction of the cost — see the [blog post](https://joinhandshake.com/research/ai/gandalf-the-grader/) for the full meta-eval.

**Examples and integrations:** [BankerToolBench](https://github.com/Handshake-AI-Research/bankertoolbench) is a public agentic RL benchmark environment that uses Gandalf as the verifier. [rle-pkg](https://github.com/Handshake-AI-Research/rle-pkg) is a reference runtime that integrates Gandalf. Both run under the [Harbor](https://github.com/harbor-framework/harbor) framework, but Gandalf's design and implementation are framework-agnostic.

## Installation

Gandalf is published [on PyPI](https://pypi.org/project/gandalf-the-grader/).

```bash
uv tool install gandalf-the-grader
```

For production use, we recommend that you pin a specific version of Gandalf, and furthermore use the `[pinned]` version to [pin all transitive dependencies](https://github.com/edgarrmondragon/hatch-pinned-extra).

```bash
uv tool install 'gandalf-the-grader[pinned]==1.0.0'
```

## Runtime dependencies

**Important**: Gandalf is built on top of OpenHands, which [works best](https://github.com/OpenHands/software-agent-sdk/pull/120) when `tmux` is installed. The judge refuses to run when `tmux` is not on `PATH` rather than silently falling back to a less stable subprocess-based terminal.

## Quick start

The repo ships a runnable example under [`examples/quickstart/`](examples/quickstart) that grades a pre-staged workspace + ATIF trajectory against a 3-criterion rubric. Two criteria are designed to be met and one is designed to fail, so you can see Gandalf's partial-credit grading and per-criterion reasoning in one run. From a fresh clone:

```bash
# 1. Install
uv tool install gandalf-the-grader

# 2. Provide a Gemini API key (any litellm-compatible model works; see Configuration)
export LLM_API_KEY="<your-gemini-api-key>"

# 3. Run from the repo root
gandalf-the-grader --config examples/quickstart/grader.toml

# 4. Inspect the result
cat examples/quickstart/output/reward.json   # -> {"reward": 0.75}
cat examples/quickstart/output/info.json     # per-criterion verdicts + reasoning
```

Expected verdicts: the `welcome.txt` file exists (met), the message mentions Gandalf (met), and the message is *not* longer than 50 words (unmet, by design). Raw score 3.0 of a possible 4.0, for a reward of 0.75.

The example uses [`gemini/gemini-2.5-flash`](examples/quickstart/grader.toml) and runs the inner judge as the current user (no `sandbox_user`, no sudo). To adapt it to your own setup, edit [`examples/quickstart/grader.toml`](examples/quickstart/grader.toml). See the [Configuration](#configuration) section below for the full field reference.

The same fixture can also be graded holistically with free-form guidance:

```bash
gandalf-the-grader --config examples/quickstart/guidance_grader.toml
cat examples/quickstart/guidance_output/reward.json
cat examples/quickstart/guidance_output/info.json
```

## Configuration

### `grader.toml`

| Field | Required | Default | Description |
|---|---|---|---|
| `grading_mode` | No | `rubric` | Grading path: `rubric` for weighted criteria, or `guidance` for one holistic score |
| `instructions` | Yes\* | | Inline task instructions given to the original agent (mutually exclusive with `instructions_path`) |
| `instructions_path` | Yes\* | | Path to a file with task instructions (mutually exclusive with `instructions`) |
| `rubric` | Rubric mode\* | | Inline rubric as a TOML array of tables (mutually exclusive with `rubric_path`; invalid in guidance mode) |
| `rubric_path` | Rubric mode\* | | Path to rubric JSON file (mutually exclusive with `rubric`; invalid in guidance mode) |
| `judge_guidance` | Guidance mode\* | | Inline judge guidance text (mutually exclusive with `judge_guidance_path`; optional extra context in rubric mode) |
| `judge_guidance_path` | Guidance mode\* | | Path to a file with extra judge instructions (mutually exclusive with `judge_guidance`; optional extra context in rubric mode) |
| `workdir` | Yes | | Agent workspace directory |
| `trajectory_path` | Yes | | Path to ATIF trajectory JSON |
| `output_dir` | Yes | | Directory for grader output files |
| `model` | No | `gemini/gemini-2.5-flash` | LLM model for the judge agent |
| `mode` | No | `batch` | Rubric evaluation mode: `batch` or `individual` |
| `judge_timeout` | No | `300` | Max seconds per judge invocation |
| `batch_timeout` | No | | Rubric batch-mode max total seconds (caps `judge_timeout * N`; invalid in guidance mode) |
| `judge_retries` | No | `1` | Number of retry attempts for criteria or guidance scores that error due to infrastructure/parse failures |
| `batch_splits` | No | | Split rubric criteria into N chunks in batch mode (>= 2). Only valid with `mode = "batch"` and rubric mode. |
| `max_concurrency` | No | | Max parallel rubric judge sessions (>= 1). Invalid in guidance mode. |
| `sandbox_user` | No | | Username for running the inner judge (via sudo). When omitted the judge runs as the current user. |
| `judge_prompt` | No | | Inline Jinja2 template that completely overrides the built-in judge task prompt (mutually exclusive with `judge_prompt_path`) |
| `judge_prompt_path` | No | | Path to a Jinja2 template file that completely overrides the built-in judge task prompt (mutually exclusive with `judge_prompt`) |

MCP servers can be configured as TOML array of tables:

```toml
[[mcp_servers]]
name = "magic-server"
transport = "stdio"
command = "/usr/bin/mcp-server"
args = ["--verbose"]
```

### Custom Judge Prompt

By default, the grader uses a built-in prompt template to kick off each judge session. `judge_prompt` / `judge_prompt_path` let you replace it entirely with a custom [Jinja2](https://jinja.palletsprojects.com/) template.

> **Note:** This prompt is sent as the opening **user message** to the judge agent, not the LLM system prompt. The underlying agent framework (OpenHands) has its own immutable system message with coding and tool-use instructions that we never modify. Our prompt sits on top of that as the first user turn, setting up the grading task.

For most use cases, `judge_guidance` / `judge_guidance_path` is all you need: it injects extra instructions into the built-in prompt without replacing it. Fully overriding the judge prompt is an uncommon escape hatch for situations where the built-in prompt structure itself is unsuitable.

The template receives these variables:

| Variable | Type | Mode | Description |
|---|---|---|---|
| `instructions` | `str` | all | Task instructions given to the original agent |
| `final_output` | `str` | all | Agent's final message from the trajectory |
| `criterion` | `str` | individual | The single criterion string to evaluate |
| `criteria` | `list[str]` | batch | List of all criterion strings to evaluate |
| `verdict_path` | `str` | individual, batch | File path the judge must write its verdict to |
| `judge_guidance` | `str` | all | Additional guidance text; required in guidance mode |
| `trajectory_path` | `str` | guidance | Path to the copied trajectory JSON inside the cloned judge workspace |
| `score_path` | `str` | guidance | File path the judge must write its holistic score to |

Individual, batch, and guidance modes use separate built-in templates. In a custom template, use `{% if criterion is defined %}`, `{% if criteria is defined %}`, or `{% if score_path is defined %}` if you need to distinguish modes. In batch mode, use `loop.index0` for the criterion index (e.g., `{% for c in criteria %}[{{ loop.index0 }}] {{ c }}{% endfor %}`).

### Guidance Mode

Set `grading_mode = "guidance"` to grade with free-form guidance instead of a rubric:

```toml
grading_mode = "guidance"
instructions_path = "/path/to/instruction.md"
judge_guidance_path = "/path/to/judge_guidance.md"
workdir = "/path/to/final/workspace"
trajectory_path = "/path/to/agent/trajectory.json"
output_dir = "/path/to/grader-output"
```

Guidance mode rejects `rubric`, `rubric_path`, `batch_splits`, `batch_timeout`, and `max_concurrency` so those fields are not silently ignored. It runs one holistic judge session per attempt. The judge sees the final output as context, but the full trajectory JSON is copied into the cloned judge workspace and passed as `trajectory_path`; the full trajectory is not inlined into the prompt.

### Rubric JSON

A JSON array of objects with `criterion` (string) and `weight` (float). Weights can be negative to penalise undesired outcomes:

```json
[
  {"criterion": "The output file exists", "weight": 2.0},
  {"criterion": "The output contains correct totals", "weight": 3.0},
  {"criterion": "The agent used hardcoded values instead of computing", "weight": -1.0}
]
```

- **Positive weight**: adds to the raw score when the criterion's condition is met
- **Negative weight**: deducts from the raw score when the criterion's condition is met (the bad thing happened)
- The judge evaluates each criterion on its own merits; it never sees weights

## Trajectory Format (ATIF)

The grader reads agent trajectories in [Agent Trajectory Interchange Format (ATIF)](https://www.harborframework.com/docs/agents/trajectory-format). An ATIF file is a JSON object with a `steps` array:

```json
{
  "steps": [
    {"source": "user", "message": "Build a hello world web app"},
    {"source": "agent", "message": "I'll create the file now", "tool_calls": [...]},
    {"source": "agent", "message": "Done! I created index.html with a Hello World page."}
  ]
}
```

The grader extracts the final agent message (last `"source": "agent"` step with a non-empty message and no `tool_calls`) and passes it to the judge as context.

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_API_KEY` | API key for the LLM provider |
| `LLM_BASE_URL` | Base URL for the LLM API (optional) |
| `GRADER_INSTRUCTIONS_PATH` | Fallback path to task instructions file (if not set in TOML) |
| `GRADER_JUDGE_GUIDANCE_PATH` | Fallback path to judge guidance file (if not set in TOML) |
| `GRADER_JUDGE_PROMPT_PATH` | Fallback path to custom judge prompt template (if not set in TOML) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL for trace export (optional) |
| `OTEL_EXPORTER_OTLP_HEADERS` | OTLP auth headers, URL-encoded (optional) |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | OTLP transport protocol, e.g. `http/protobuf` (optional) |

### Tracing / Observability

Gandalf builds on top of OpenHands, which has built-in OpenTelemetry tracing that automatically instruments LLM calls, tool executions, and agent steps. Set the `OTEL_EXPORTER_OTLP_*` variables above to export traces to any OTEL-compatible backend with no code changes required.

**Example: Langfuse**

```bash
# Encode your Langfuse keys
echo -n "pk-lf-...:sk-lf-..." | base64

# Export the variables
export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20<base64-encoded-keys>"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
```

## Output

In rubric mode, the grader writes to `output_dir`:

- `reward.json`: Reward file (e.g., `{"reward": 0.75}`) (always in [0, 1]). **Only written when all criteria are successfully evaluated.** If any criteria still have errors after retries, the grader writes `info.json` but skips `reward.json` and exits with code 1.
- `info.json`: Always written. Per-criterion results with `met`/not-met, reasoning, evidence, LLM usage, plus `reward`, `raw_score`, `minimum_score`, `maximum_score`, `errored_criterion_count`, and `evaluated_criteria_pct`.
- `judge_trace_*.txt`: stdout/stderr capture for each judge invocation. Naming varies by mode: `judge_trace_{i}.txt` (individual), `judge_trace_batch.txt` (batch), `judge_trace_batch_split{i}.txt` (batch with splits). Retries append a `_retry{N}` suffix.

The `reward` in `reward.json` is `clip(0, 1, raw_score / sum_of_positive_weights)`, always in [0, 1]. `info.json` additionally includes `raw_score` (the raw sum of weights for met criteria, which can be negative) and `minimum_score`/`maximum_score` bounds for reference.

In guidance mode:

- `reward.json`: `{"reward": <score>}` where score is the holistic judge score rounded to 4 decimals. Only written when the judge returns a valid numeric score in `[0, 1]`.
- `info.json`: Always written after the guidance judge runs. Contains `grading_mode`, `reward`, `score`, `reasoning`, `evidence`, `llm_usage`, `errored`, and `error`.
- `judge_trace_guidance*.txt`: stdout/stderr capture for each holistic judge attempt. Retries are named `judge_trace_guidance_retry{N}.txt`.

The guidance judge output is considered valid only when it includes a numeric score, non-empty reasoning, and a non-empty evidence array. Missing or malformed audit fields are treated like invalid judge output and retried according to `judge_retries`. The default guidance prompt asks the judge to include artifact evidence, trajectory evidence, and an explicit score calibration/cap audit so score bands and hard penalties from the guidance are applied visibly. It also asks the judge to reconcile conflicts between the task instructions and grading guidance, especially artifact output-location conflicts, before applying path-related penalties. When tasks mention required or forbidden external actions, the prompt asks for an action/side-effect audit based on trajectory tool calls and final state.

Example guidance `info.json`:

```json
{
  "grading_mode": "guidance",
  "reward": 0.8,
  "score": 0.8,
  "reasoning": "The final artifact is mostly correct but omits one requested comparison.",
  "evidence": ["Read /workspace/report.md", "Inspected trajectory file for failed command"],
  "llm_usage": {"cost_usd": 0.02, "prompt_tokens": 1200, "completion_tokens": 400, "cache_read_tokens": 0},
  "errored": false,
  "error": null
}
```

## Harbor Rollouts and DSPy Calibration

The repo includes helper scripts for collecting Harbor rollouts without running any verifier, then evaluating rubric and guidance scoring later. The analysis helper reports rubric/guidance score agreement as a calibration signal, plus guidance-grounded audit signals such as guidance vocabulary coverage, trajectory evidence, score-cap audit evidence, output-location conflict audits, and required-action evidence:

```bash
scripts/collect_harbor_rollouts.sh
scripts/index_harbor_rollouts.py
scripts/eval_guidance_scores.py --mode rubric
scripts/eval_guidance_scores.py --mode guidance
uv run --group eval scripts/dspy_optimize_guidance.py
```

`collect_harbor_rollouts.sh` sources `$ENV_FILE` (default: `$HOME/Downloads/env`), discovers Harbor-format tasks under `$TASK_ROOT` (default: `$HOME/Downloads/harbor-research-v2-Batch1-2`), runs `harbor run` with `--disable-verification`, and caps outer concurrency at 5. DSPy is used only by the evaluation/calibration script; Gandalf's runtime guidance mode remains the reactive environment-inspecting judge.

## Next steps

- **Try the benchmark environment.** [BankerToolBench on Hugging Face](https://huggingface.co/datasets/handshake-ai-research/bankertoolbench) is the public RL environment that Gandalf was originally evaluated against. Clone it, run rollouts, and grade them with Gandalf.
- **Adapt Gandalf to a new rollout environment.** Edit [`examples/quickstart/grader.toml`](examples/quickstart/grader.toml) to point at your workspace, trajectory, and rubric. See the [Configuration](#configuration) and [Custom Judge Prompt](#custom-judge-prompt) sections for the full reference, including domain-specific judge guidance.

## License

Copyright (c) Handshake. Released under the Apache-2.0 license. See [LICENSE.txt](LICENSE.txt) for details.
