# Gandalf the Grader [![Build Status](https://github.com/Handshake-AI-Research/gandalf-the-grader/actions/workflows/ci.yml/badge.svg)](https://github.com/Handshake-AI-Research/gandalf-the-grader/actions/workflows/ci.yml) [![Coverage](https://codecov.io/gh/Handshake-AI-Research/gandalf-the-grader/branch/main/graph/badge.svg)](https://app.codecov.io/gh/Handshake-AI-Research/gandalf-the-grader) [![PyPI](https://img.shields.io/pypi/v/gandalf-the-grader.svg)](https://pypi.org/pypi/gandalf-the-grader/) [![PyPI - Python version](https://img.shields.io/pypi/pyversions/gandalf-the-grader.svg)](https://pypi.org/pypi/gandalf-the-grader/)

### Your verifier is probably the bottleneck. We built one that isn't.

![Gandalf vs. baseline verifiers on BankerVerifierBench (cost vs. F1)](https://raw.githubusercontent.com/Handshake-AI-Research/assets/main/gandalf-the-grader/pareto_frontier.png)

Read the [launch blog post](https://joinhandshake.com/research/ai/gandalf-the-grader/) for the motivation, benchmark results, and design rationale behind Gandalf.

Gandalf is a reactive agent-as-judge for rubric-graded agent environments. Given a rubric of binary criteria, it runs inside the rollout environment, uses the same tools as the rollout agent, and decides at inference time which files to open and which tool state to query.

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

## Configuration

### `grader.toml`

| Field | Required | Default | Description |
|---|---|---|---|
| `instructions` | Yes\* | | Inline task instructions given to the original agent (mutually exclusive with `instructions_path`) |
| `instructions_path` | Yes\* | | Path to a file with task instructions (mutually exclusive with `instructions`) |
| `rubric` | Yes\* | | Inline rubric as a TOML array of tables (mutually exclusive with `rubric_path`) |
| `rubric_path` | Yes\* | | Path to rubric JSON file (mutually exclusive with `rubric`) |
| `judge_guidance` | No | | Inline judge guidance text (mutually exclusive with `judge_guidance_path`) |
| `judge_guidance_path` | No | | Path to a file with extra judge instructions (mutually exclusive with `judge_guidance`) |
| `workdir` | Yes | | Agent workspace directory |
| `trajectory_path` | Yes | | Path to ATIF trajectory JSON |
| `output_dir` | Yes | | Directory for grader output files |
| `model` | No | `gemini/gemini-2.5-flash` | LLM model for the judge agent |
| `mode` | No | `batch` | Evaluation mode: `batch` or `individual` |
| `judge_timeout` | No | `300` | Max seconds per judge invocation |
| `batch_timeout` | No | | Max total seconds for batch mode (caps `judge_timeout * N`) |
| `judge_retries` | No | `1` | Number of retry attempts for criteria that error due to infrastructure failures |
| `batch_splits` | No | | Split criteria into N chunks in batch mode (>= 2). Each chunk is evaluated as a separate batch session. Only valid with `mode = "batch"`. |
| `max_concurrency` | No | | Max parallel judge sessions (>= 1). Defaults to 1 for individual mode, `batch_splits` for batch mode. |
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
| `instructions` | `str` | both | Task instructions given to the original agent |
| `final_output` | `str` | both | Agent's final message from the trajectory |
| `criterion` | `str` | individual | The single criterion string to evaluate |
| `criteria` | `list[str]` | batch | List of all criterion strings to evaluate |
| `verdict_path` | `str` | both | File path the judge must write its verdict to |
| `judge_guidance` | `str` | both | Additional guidance text (may be empty) |

Individual and batch modes use separate built-in templates. In a custom template, use `{% if criterion is defined %}` vs `{% if criteria is defined %}` if you need to distinguish modes. In batch mode, use `loop.index0` for the criterion index (e.g., `{% for c in criteria %}[{{ loop.index0 }}] {{ c }}{% endfor %}`).

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

The grader writes to `output_dir`:

- `reward.json`: Reward file (e.g., `{"reward": 0.75}`) (always in [0, 1]). **Only written when all criteria are successfully evaluated.** If any criteria still have errors after retries, the grader writes `info.json` but skips `reward.json` and exits with code 1.
- `info.json`: Always written. Per-criterion results with `met`/not-met, reasoning, evidence, LLM usage, plus `reward`, `raw_score`, `minimum_score`, `maximum_score`, `errored_criterion_count`, and `evaluated_criteria_pct`.
- `judge_trace_*.txt`: stdout/stderr capture for each judge invocation. Naming varies by mode: `judge_trace_{i}.txt` (individual), `judge_trace_batch.txt` (batch), `judge_trace_batch_split{i}.txt` (batch with splits). Retries append a `_retry{N}` suffix.

The `reward` in `reward.json` is `clip(0, 1, raw_score / sum_of_positive_weights)`, always in [0, 1]. `info.json` additionally includes `raw_score` (the raw sum of weights for met criteria, which can be negative) and `minimum_score`/`maximum_score` bounds for reference.

## Next steps

- **Try the benchmark environment.** [BankerToolBench on Hugging Face](https://huggingface.co/datasets/handshake-ai-research/bankertoolbench) is the public RL environment that Gandalf was originally evaluated against. Clone it, run rollouts, and grade them with Gandalf.
- **Adapt Gandalf to a new rollout environment.** Edit [`examples/quickstart/grader.toml`](examples/quickstart/grader.toml) to point at your workspace, trajectory, and rubric. See the [Configuration](#configuration) and [Custom Judge Prompt](#custom-judge-prompt) sections for the full reference, including domain-specific judge guidance.

## License

Copyright (c) Handshake. Released under the Apache-2.0 license. See [LICENSE.txt](LICENSE.txt) for details.
