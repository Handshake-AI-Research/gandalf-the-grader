# gandalf-the-grader

Agent-as-a-Judge grading framework for evaluating AI outputs against rubric criteria.
Unlike simple LLM-as-a-Judge graders, this grader can grade outputs that are complex files (such as Excel or Powerpoint deliverables).

![You shall not pass quote](https://raw.githubusercontent.com/Handshake-AI-Research/assets/refs/heads/main/gandalf-the-grader/shallnotpass.png)

## Overview

`gandalf-the-grader` uses LLM-powered judge agents to evaluate whether an AI agent successfully completed a task. It is the production grader component of the [rle-pkg](https://github.com/Handshake-AI-Research/rle-pkg) architecture.

Given a task description, a rubric of evaluation criteria, and the agent's trajectory, the grader spawns judge agents that inspect the agent's workspace — reading files, running commands, and using tools — to determine whether each criterion's condition is met. The final reward is always in [0, 1], with raw scoring details included in `info.json`.

## How It Works

The grader uses a two-process architecture:

- **Outer orchestrator** (`gandalf-the-grader`) — runs as the grader user, manages the evaluation loop, computes reward/raw scoring outputs, and writes output files.
- **Inner judge** (`gandalf-the-grader-judge`) — runs as the sandbox user (via `sudo`), executes an [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent-as-judge session that investigates the workspace and writes a verdict.

Two evaluation modes are supported (configured via `mode` in the TOML config):

- **Individual** (default): one agent session per rubric criterion.
- **Batch**: all criteria evaluated in a single agent session.

When `max_concurrency` > 1, multiple judge sessions run in parallel. For batch mode this splits criteria into positional chunks; for individual mode it runs multiple criterion evaluations concurrently.

```toml
mode = "batch"
max_concurrency = 2   # split criteria into 2 chunks, evaluate in parallel
```

## Installation

```bash
pip install git+https://github.com/Handshake-AI-Research/gandalf-the-grader.git@main
```

or with uv:

```bash
uv tool install git+https://github.com/Handshake-AI-Research/gandalf-the-grader.git@main
```

## Quick Start

Create a grader config (`grader.toml`):

```toml
model = "gemini/gemini-2.5-flash"
sandbox_user = "sandbox"
instructions = "Build a web app that displays hello world."
rubric_path = "/tests/rubric.json"
workdir = "/home/agent/workspace"
trajectory_path = "/logs/agent/trajectory.json"
output_dir = "/logs/grader"
```

Create a rubric (`rubric.json`):

```json
[
  {"criterion": "The file index.html exists in the workspace", "weight": 1.0},
  {"criterion": "The page displays 'Hello World'", "weight": 2.0}
]
```

Run the grader:

```bash
gandalf-the-grader --config /tests/grader.toml
```

## Configuration

### `grader.toml`

| Field | Required | Default | Description |
|---|---|---|---|
| `instructions` | Yes | | Task instructions given to the original agent |
| `rubric` | Yes\* | | Inline rubric as a TOML array of tables (mutually exclusive with `rubric_path`) |
| `rubric_path` | Yes\* | | Path to rubric JSON file (mutually exclusive with `rubric`) |
| `judge_guidance` | No | | Inline judge guidance text (mutually exclusive with `judge_guidance_path`) |
| `judge_guidance_path` | No | | Path to a file with extra judge instructions (mutually exclusive with `judge_guidance`) |
| `workdir` | Yes | | Agent workspace directory |
| `trajectory_path` | Yes | | Path to ATIF trajectory JSON |
| `output_dir` | Yes | | Directory for grader output files |
| `model` | No | `gemini/gemini-2.5-flash` | LLM model for the judge agent |
| `mode` | No | `batch` | Evaluation mode: `individual` or `batch` |
| `max_concurrency` | No | `None` | Max parallel judge sessions (None = no parallelism) |
| `judge_timeout` | No | `300` | Max seconds per judge invocation |
| `judge_retries` | No | `1` | Number of retry attempts for errored criteria |
| `batch_timeout` | No | | Max seconds per batch session (caps `judge_timeout * N_criteria_in_session`) |
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

> **Note:** This prompt is sent as the opening **user message** to the judge agent — it is not the LLM system prompt. The underlying agent framework (OpenHands) has its own immutable system message with coding and tool-use instructions that we never modify. Our prompt sits on top of that as the first user turn, setting up the grading task.

For most use cases, `judge_guidance` / `judge_guidance_path` is all you need — it injects extra instructions into the built-in prompt without replacing it. Fully overriding the judge prompt is an uncommon escape hatch for situations where the built-in prompt structure itself is unsuitable.

The template receives these variables:

| Variable | Type | Mode | Description |
|---|---|---|---|
| `instructions` | `str` | both | Task instructions given to the original agent |
| `final_output` | `str` | both | Agent's final message from the trajectory |
| `criterion` | `str` | sequential | The single criterion string to evaluate |
| `criteria` | `list[str]` | batch | List of all criterion strings to evaluate |
| `verdict_path` | `str` | both | File path the judge must write its verdict to |
| `judge_guidance` | `str` | both | Additional guidance text (may be empty) |

Sequential and batch modes use separate built-in templates. In a custom template, use `{% if criterion is defined %}` vs `{% if criteria is defined %}` if you need to distinguish modes. In batch mode, use `loop.index0` for the criterion index (e.g., `{% for c in criteria %}[{{ loop.index0 }}] {{ c }}{% endfor %}`).

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
- The judge evaluates each criterion on its own merits — it never sees weights

## Trajectory Format (ATIF)

The grader reads agent trajectories in ATIF (Agent Trajectory Interchange Format). An ATIF file is a JSON object with a `steps` array:

```json
{
  "steps": [
    {"source": "user", "message": "Build a hello world web app"},
    {"source": "agent", "message": "I'll create the file now", "tool_calls": [...]},
    {"source": "agent", "message": "Done! I created index.html with a Hello World page."}
  ]
}
```

The grader extracts the final agent message (last `"source": "agent"` step without `tool_calls`) and passes it to the judge as context.

## Docker Usage

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    uv tool install git+https://github.com/Handshake-AI-Research/gandalf-the-grader.git@main && \
    chmod -R a+rX /opt/uv-tools /opt/uv-python
```

For a complete container architecture with task runners and agent environments, see [rle-pkg](https://github.com/Handshake-AI-Research/rle-pkg).

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_API_KEY` | API key for the LLM provider |
| `LLM_BASE_URL` | Base URL for the LLM API (optional) |
| `GRADER_JUDGE_GUIDANCE_PATH` | Fallback path to judge guidance file (if not set in TOML) |
| `GRADER_JUDGE_PROMPT_PATH` | Fallback path to custom judge prompt template (if not set in TOML) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL for trace export (optional) |
| `OTEL_EXPORTER_OTLP_HEADERS` | OTLP auth headers, URL-encoded (optional) |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | OTLP transport protocol, e.g. `http/protobuf` (optional) |

### Tracing / Observability

The OpenHands SDK has built-in OpenTelemetry tracing that automatically instruments LLM calls, tool executions, and agent steps. Set the `OTEL_EXPORTER_OTLP_*` variables above to export traces to any OTEL-compatible backend — no code changes required.

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

- `reward.json` — Reward file: `{"reward": 0.75}` (always in [0, 1])
- `info.json` — Per-criterion results with `met`/not-met, reasoning, evidence, LLM usage, plus `reward`, `raw_score`, `minimum_score`, and `maximum_score`
- `judge_trace_<i>.txt` — stdout/stderr capture for each judge invocation

The `reward` in `reward.json` is `clip(0, 1, raw_score / sum_of_positive_weights)`, always in [0, 1]. `info.json` additionally includes `raw_score` (the raw sum of weights for met criteria, which can be negative) and `minimum_score`/`maximum_score` bounds for reference.

## Development

```bash
git clone https://github.com/Handshake-AI-Research/gandalf-the-grader.git
cd gandalf-the-grader
hatch test
hatch fmt --check
hatch run types:check
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for full setup details.

## License

Apache-2.0
