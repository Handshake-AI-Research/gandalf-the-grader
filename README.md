# gandalf-the-grader

Agent-as-judge grading framework for evaluating AI agent outputs against rubric criteria.

## Overview

`gandalf-grader` uses LLM-powered judge agents to evaluate whether an AI agent successfully completed a task. It is the production verifier component of the [rle-pkg](https://github.com/Handshake-AI-Research/rle-pkg) architecture.

Given a task description, a rubric of evaluation criteria, and the agent's trajectory, the grader spawns judge agents that inspect the agent's workspace — reading files, running commands, and using tools — to produce a binary pass/fail verdict for each criterion. The final score is the weighted average of all verdicts.

## How It Works

The grader uses a two-process architecture:

- **Outer orchestrator** (`gandalf-grader`) — runs as the verifier user, manages the evaluation loop, computes the final score, and writes output files.
- **Inner judge** (`gandalf-grader-judge`) — runs as the sandbox user (via `sudo`), executes an [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent-as-judge session that investigates the workspace and writes a verdict.

Two evaluation modes are supported:

- **Sequential** (default): one agent session per rubric criterion.
- **Batch**: all criteria evaluated in a single agent session.

## Installation

```bash
pip install git+https://github.com/Handshake-AI-Research/gandalf-the-grader.git@main
```

or with uv:

```bash
uv tool install git+https://github.com/Handshake-AI-Research/gandalf-the-grader.git@main
```

## Quick Start

Create a verifier config (`verifier.toml`):

```toml
model = "google/gemini-2.5-flash"
sandbox_user = "sandbox"
instructions = "Build a web app that displays hello world."
rubric_path = "/tests/rubric.json"
workdir = "/home/agent/workspace"
trajectory_path = "/logs/agent/trajectory.json"
```

Create a rubric (`rubric.json`):

```json
[
  {"criteria": "The file index.html exists in the workspace", "weight": 1.0},
  {"criteria": "The page displays 'Hello World'", "weight": 2.0}
]
```

Run the grader:

```bash
gandalf-grader --config /tests/verifier.toml
```

## Configuration

### `verifier.toml`

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | No | `google/gemini-2.5-flash` | LLM model for the judge agent |
| `sandbox_user` | Yes | | Username for running the inner judge (via sudo) |
| `instructions` | Yes | | Task instructions given to the original agent |
| `rubric_path` | Yes | | Path to rubric JSON file |
| `workdir` | Yes | | Agent workspace directory |
| `trajectory_path` | Yes | | Path to ATIF trajectory JSON |
| `output_dir` | No | `/logs/verifier` | Directory for output files |
| `judge_timeout` | No | `300` | Max seconds per judge invocation |
| `mode` | No | `sequential` | Evaluation mode: `sequential` or `batch` |
| `judge_guidance_path` | No | | Path to a markdown file with extra judge instructions |
| `batch_timeout` | No | | Max total seconds for batch mode (caps `judge_timeout * N`) |

MCP servers can be configured as TOML array of tables:

```toml
[[mcp_servers]]
name = "magic-server"
transport = "stdio"
command = "/usr/bin/mcp-server"
args = ["--verbose"]
```

### Rubric JSON

A JSON array of objects with `criteria` (string) and `weight` (float):

```json
[
  {"criteria": "Description of what to check", "weight": 1.0}
]
```

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
| `VERIFIER_JUDGE_GUIDANCE_PATH` | Fallback path to judge guidance file (if not set in TOML) |

## Output

The grader writes to `output_dir` (default `/logs/verifier`):

- `reward.json` — Reward file: `{"score": 0.75}`
- `info.json` — Per-criteria results with pass/fail, reasoning, evidence, and LLM usage
- `judge_trace_<i>.txt` — stdout/stderr capture for each judge invocation

Score is computed as a weighted average of binary criterion results, rounded to 4 decimal places.

## Development

```bash
git clone https://github.com/Handshake-AI-Research/gandalf-the-grader.git
cd gandalf-the-grader
uv sync --dev
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/
```

## License

Apache-2.0
